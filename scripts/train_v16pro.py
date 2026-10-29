"""Train v16 parents or matched decoder adaptations on the 13-cue protocol.

The detector pair is pretrained separately and always frozen here. Parent
selection and every recovery branch use the same validation-only criterion.
Limited-batch runs are explicitly marked smoke-only in every checkpoint.
"""

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import bootstrap  # noqa: F401

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (fix_console_encoding, load_config, log, save_checkpoint_atomic,
                    set_seed, spike_reg, unpack_paired_batch)
from data.dataset import build_loaders
from data.v16_protocol import make_cues, make_targets, sample_mode
from models.network import CrossModalSNN
from models.frozen_base import file_sha256
from models.noise_detector import load_pretrained_detectors
from paths import resolve_from_root
from scripts.train import (_img_recon_loss, _aud_recon_loss, _aud_active_loss,
                           _class_key_alignment_loss, _soft_cls_loss,
                           _build_train_optimizer)
from scripts.train_noise_detector import choose_device, seed_loader
from scripts.train_detector_v16pro import validate_v16_detector_checkpoint
from scripts.v16_evaluation import validate_model


SCHEMA = "v16pro.training.v1"
PARENTS = ("parent", "parent_no_recurrence")
ADAPTATIONS = ("main", "no_value", "no_recurrence")
RECOVERY_PREFIXES = ("image_decoder.", "image_refiner.", "audio_decoder.",
                     "audio_local_cue_multiscale.")
FORWARD_SECTIONS = ("dims", "snn", "index", "value", "audio", "ablation",
                    "detail_conditioning", "cross_key_conditioning",
                    "cross_detail_conditioning", "pair_alignment",
                    "image_refiner", "audio_refiner", "audio_local_cue")


def validate_training_config(cfg):
    stage = cfg.get("train", {}).get("stage")
    if not cfg.get("v16", {}).get("enabled", False) or stage not in (*PARENTS, *ADAPTATIONS):
        raise ValueError("v16 training requires an enabled protocol and a valid train.stage")
    if not cfg.get("noise_detector", {}).get("enabled", False):
        raise ValueError("v16 training requires noise detectors")
    nc = cfg["noise_detector"]
    if not nc.get("freeze", False) or nc.get("mask_source", "predicted") != "predicted":
        raise ValueError("v16 recovery trains with frozen detectors and predicted masks")
    for section in ("cross_key_conditioning", "cross_detail_conditioning", "pair_alignment"):
        if cfg.get(section, {}).get("enabled", False):
            raise ValueError(f"v16 excludes {section}")
    if bool(cfg.get("ablation", {}).get("use_recurrent", True)) != ("no_recurrence" not in stage):
        raise ValueError("train.stage and use_recurrent disagree")
    if bool(cfg["v16"].get("no_value", False)) != (stage == "no_value"):
        raise ValueError("train.stage and v16.no_value disagree")
    expected = {"64"} if stage in PARENTS else {"16", "32", "64"}
    if set(str(x) for x in cfg["v16"].get("local_scales", ())) != expected:
        raise ValueError("Parents require the masked 64 branch; adaptations use all three scales")
    if cfg.get("train", {}).get("freeze_base", False):
        raise ValueError("v16 uses an explicit recovery whitelist, not legacy freeze_base")
    if cfg.get("data", {}).get("pairing", {}).get("sample_targets_for_missing", False):
        raise ValueError("v16 missing-modality targets must be train-derived category representatives")
    if int(cfg["train"].get("epochs", 0)) <= 0:
        raise ValueError("train.epochs must be positive")
    if cfg.get("data", {}).get("dataset", "mnist_fsdd") == "mnist_fsdd":
        if not cfg.get("data", {}).get("paper_split", {}).get("enabled", False):
            raise ValueError("v16 selection requires separate train/val/test splits")
    if stage in ADAPTATIONS and not cfg["train"].get("init_ckpt_path"):
        raise ValueError("Recovery adaptations require a parent checkpoint")
    return stage


def configure_trainable(model, cfg):
    """Freeze the shared core in all matched 30-epoch recovery adaptations."""
    stage = validate_training_config(cfg)
    prefixes = tuple(cfg["train"].get("trainable_prefixes", RECOVERY_PREFIXES))
    if stage in ADAPTATIONS:
        if set(prefixes) != set(RECOVERY_PREFIXES):
            raise ValueError("v16 recovery must use the shared decoder/local-projector whitelist")
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith(prefixes))
    # Unused legacy modules and detached Value teachers are never optimized.
    unused = ("img_noise_detector.", "aud_noise_detector.", "audio_refiner.",
              "audio_local_cue_projector.", "memory.V_img.W_enc_to_V.",
              "memory.V_aud.W_enc_to_V.", "img_cross_adapter.",
              "aud_cross_adapter.", "aud_cross_adapter_mid.",
              "aud_to_img_cross_", "img_to_aud_cross_")
    for name, param in model.named_parameters():
        if name.startswith(unused):
            param.requires_grad_(False)
        if stage == "parent_no_recurrence" and name.startswith("memory.index.W_rec."):
            param.requires_grad_(False)
        if stage in PARENTS and name.startswith(("audio_local_cue_multiscale.16.",
                                                "audio_local_cue_multiscale.32.")):
            param.requires_grad_(False)
    return [name for name, param in model.named_parameters() if param.requires_grad]


def set_training_mode(model, cfg):
    if cfg["train"]["stage"] in PARENTS:
        model.train()
    else:
        model.eval()
        for name, module in model.named_children():
            if any(prefix.startswith(name + ".") for prefix in RECOVERY_PREFIXES):
                module.train()


def compute_losses(model, clean_img, clean_aud, labels, mode, cfg, proto_img, proto_aud):
    """True regions supervise losses only; they never select the forward repair mask."""
    cues = make_cues(clean_img, clean_aud, mode, cfg)
    target_i, target_a, kind_i, kind_a = make_targets(
        clean_img, clean_aud, labels, proto_img, proto_aud, mode)
    lc = cfg["loss"]
    parent = cfg["train"]["stage"] in PARENTS
    binding = parent and cfg.get("ablation", {}).get("use_binding_phase", True)
    out = model(x_img_cue=cues["img"], x_aud_cue=cues["aud"],
                x_img_target=target_i if binding else None,
                x_aud_target=target_a if binding else None,
                training_mode=True, phase="binding" if binding else "readout",
                img_cue_mask=cues["missing"]["img"],
                aud_cue_mask=cues["missing"]["aud"])
    terms = {}
    for name, target, helper in (("img", target_i, _img_recon_loss),
                                 ("aud", target_a, _aud_recon_loss)):
        missing, noise = cues["missing"][name], cues["noise"][name]
        region = None if missing is None else torch.maximum(missing, noise)
        # Per-region supervision uses the *true* affected pixels, never predictions.
        terms[name + "_final"] = float(lc["lambda_" + name]) * helper(
            out["recovered_" + name], target, lc, mask=region)
        terms[name + "_coarse"] = (float(lc["lambda_" + name])
            * float(cfg["noise_detector"].get("coarse_loss_weight", 0.5))
            * helper(out["recovered_" + name + "_coarse"], target, lc, mask=region))
    if kind_a == "instance" and lc.get("lambda_aud_active", 0) > 0:
        terms["aud_active"] = float(lc["lambda_aud_active"]) * _aud_active_loss(
            out["recovered_aud_coarse"], target_a)
    if parent:
        cls_weight = float(lc.get("lambda_cls", 1.0))
        if cues["img"] is None:
            cls_weight *= float(lc.get("lambda_cls_aud_only_mult", 1.0))
        terms["classification"] = cls_weight * F.cross_entropy(out["logits"], labels)
        if out.get("aux_aud_logits") is not None:
            terms["audio_aux"] = float(lc.get("lambda_audio_aux", 0)) * F.cross_entropy(
                out["aux_aud_logits"], labels)
        if binding:
            for name in ("img", "aud"):
                terms[name + "_binding"] = float(lc.get("lambda_bind_" + name, 0)) * F.mse_loss(
                    out["v_" + name + "_from_A"], out["v_" + name + "_target"].detach())
        aligned = _class_key_alignment_loss(out.get("key_img"), out.get("key_aud"),
                    labels, float(lc.get("key_align_temperature", 0.1)))
        if aligned is not None:
            terms["key_alignment"] = float(lc.get("lambda_key_align", 0)) * aligned
        # Clean teacher keeps exactly the same available modalities. Labels and
        # clean cues provide training supervision, never additional inference inputs.
        if any(state in ("P", "N") for state in cues["states"].values()) and (
                lc.get("lambda_index_cons", 0) > 0 or lc.get("lambda_soft_cls", 0) > 0):
            with torch.no_grad():
                teacher = model(x_img_cue=clean_img if cues["img"] is not None else None,
                                x_aud_cue=clean_aud if cues["aud"] is not None else None)
            terms["index_consistency"] = float(lc.get("lambda_index_cons", 0)) * F.mse_loss(
                out["index_state"], teacher["index_state"])
            terms["soft_classification"] = float(lc.get("lambda_soft_cls", 0)) * _soft_cls_loss(
                out["logits"], teacher["logits"], float(lc.get("soft_cls_temperature", 2)))
        terms["spike_activity"] = float(lc.get("lambda_reg", 0)) * spike_reg(out)
    total = sum(terms.values())
    return total, {key: float(value.detach()) for key, value in terms.items()}


def config_contract(cfg, parent=False):
    """Strict semantics; only locations and execution device are portable."""
    result = deepcopy(cfg)
    for key in ("device", "version", "paper_template", "extends"):
        result.pop(key, None)
    for key in ("root", "num_workers"):
        result.get("data", {}).pop(key, None)
    for key in ("fsdd_root", "norm_stats_path", "auto_download"):
        result.get("audio", {}).pop(key, None)
    for key in ("checkpoint", "last_checkpoint", "allow_smoke_checkpoint", "require_cuda"):
        result.get("noise_detector", {}).pop(key, None)
    if parent:
        result.pop("train", None)
        result.get("v16", {}).pop("local_scales", None)
        result.get("v16", {}).pop("no_value", None)
    else:
        for key in ("ckpt_path", "best_ckpt_path", "eval_ckpt_path", "init_ckpt_path",
                    "parent_sha256", "output_version", "require_cuda"):
            result.get("train", {}).pop(key, None)
    return result


def load_parent(model, cfg, detector_sha256, smoke_only):
    parent_path = resolve_from_root(cfg["train"]["init_ckpt_path"])
    if not parent_path.is_file():
        raise FileNotFoundError(f"Required v16 parent: {parent_path}")
    digest = file_sha256(parent_path)
    expected_hash = cfg["train"].get("parent_sha256")
    if expected_hash and expected_hash != digest:
        raise ValueError("Parent checkpoint SHA256 mismatch")
    state = torch.load(parent_path, map_location="cpu", weights_only=True)
    expected_stage = "parent_no_recurrence" if cfg["train"]["stage"] == "no_recurrence" else "parent"
    if state.get("schema") != SCHEMA or state.get("cfg", {}).get("train", {}).get("stage") != expected_stage:
        raise ValueError("Incorrect parent training lineage")
    if config_contract(cfg, parent=True) != config_contract(state["cfg"], parent=True):
        raise ValueError("Parent configuration/protocol mismatch")
    metadata = state.get("metadata", {})
    if metadata.get("runtime_sha256") != _runtime_sha256():
        raise ValueError("Parent runtime source fingerprint mismatch")
    if metadata.get("smoke_only", False) and not smoke_only:
        raise ValueError("Smoke-only parent cannot initialize a formal recovery experiment")
    if metadata.get("detector_sha256") != detector_sha256:
        raise ValueError("Parent used a different pretrained detector checkpoint")
    model.load_state_dict(state["model"], strict=True)
    return digest


def _runtime_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "common.py", root / "paths.py"]
    for directory in ("models", "data", "scripts"):
        paths.extend(sorted((root / directory).glob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def train_model(cfg, resume=False, max_batches=0, stop_after_epoch=None):
    cfg = deepcopy(cfg)
    stage = validate_training_config(cfg)
    if max_batches < 0:
        raise ValueError("max_batches must be nonnegative")
    tc = cfg["train"]
    total_epochs = int(tc["epochs"])
    last_path = resolve_from_root(tc["ckpt_path"])
    best_path = resolve_from_root(tc["best_ckpt_path"])
    if last_path == best_path:
        raise ValueError("Last and best checkpoint paths must differ")
    if not resume and (last_path.exists() or best_path.exists()):
        raise FileExistsError("Training output already exists; use --resume or new paths")
    if resume and not (last_path.is_file() and best_path.is_file()):
        raise FileNotFoundError("Resume requires both last and best checkpoints")
    device = choose_device(cfg)
    set_seed(int(cfg.get("seed", 0)))
    train_loader, val_loader = build_loaders(cfg, eval_split="val")
    for loader, split in ((train_loader, "train"), (val_loader, "val")):
        if getattr(loader.dataset, "split", None) != split:
            raise ValueError(f"Verified {split} split required")
        if getattr(loader, "persistent_workers", False):
            raise ValueError("Epoch-seeded resume requires persistent_workers=False")
    proto_i = train_loader.dataset.prototype_img.to(device)
    proto_a = train_loader.dataset.prototype_aud.to(device)
    model = CrossModalSNN(cfg).to(device)
    detector_state = load_pretrained_detectors(model, cfg)
    if detector_state is None:
        raise ValueError("Every v16 stage requires a validated pretrained detector pair")
    validate_v16_detector_checkpoint(detector_state, cfg)
    detector_hash = file_sha256(resolve_from_root(cfg["noise_detector"]["checkpoint"]))
    smoke = bool(max_batches or detector_state.get("metadata", {}).get("smoke_only", False))
    parent_hash = None
    if stage in ADAPTATIONS:
        parent_hash = load_parent(model, cfg, detector_hash, smoke)
    elif tc.get("init_ckpt_path"):
        raise ValueError("v16 parents must start from scratch")
    trainable = configure_trainable(model, cfg)
    optimizer = _build_train_optimizer(model, cfg)
    scheduler_name = tc.get("lr_scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=float(tc.get("lr_min", 1e-5)))
    elif scheduler_name in ("none", None):
        scheduler = None
    else:
        raise ValueError("v16 supports cosine or no learning-rate scheduler")
    metadata = {"smoke_only": smoke, "max_batches": max_batches, "selection_split": "val",
                "rng_policy": "epoch_seeded.v16", "parent_sha256": parent_hash,
                "detector_sha256": detector_hash, "runtime_sha256": _runtime_sha256(),
                "trainable_names": trainable}
    start_epoch, best_score = 1, float("inf")
    if resume:
        state = torch.load(last_path, map_location=device, weights_only=True)
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        for item in (state, best):
            if item.get("schema") != SCHEMA or config_contract(item.get("cfg", {})) != config_contract(cfg):
                raise ValueError("Resume configuration mismatch")
            if item.get("metadata") != metadata:
                raise ValueError("Resume source/lineage/smoke metadata mismatch")
        if state.get("best_validation_score") != best.get("best_validation_score"):
            raise ValueError("Last and best checkpoints disagree")
        if state["epoch"] < best["epoch"] or state["epoch"] > total_epochs:
            raise ValueError("Invalid resume epoch metadata")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["opt"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_score = state["epoch"] + 1, state["best_validation_score"]
    last_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    end_epoch = min(total_epochs, int(stop_after_epoch or total_epochs))
    for epoch in range(start_epoch, end_epoch + 1):
        seed_loader(train_loader, int(cfg.get("seed", 0)) + epoch * 1009)
        set_training_mode(model, cfg)
        sums, samples, steps, modes = {}, 0, 0, {}
        for step, batch in enumerate(train_loader):
            if max_batches and step >= max_batches:
                break
            img, aud, labels, _ = unpack_paired_batch(batch)
            img, aud, labels = img.to(device), aud.to(device), labels.to(device)
            mode = sample_mode(cfg)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = compute_losses(model, img, aud, labels, mode, cfg, proto_i, proto_a)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite v16 training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                           float(tc.get("grad_clip", 1.0)), error_if_nonfinite=True)
            optimizer.step()
            count = len(labels)
            samples += count
            steps += 1
            modes[mode] = modes.get(mode, 0) + count
            for key, value in {"total": float(loss.detach()), **terms}.items():
                sums[key] = sums.get(key, 0.0) + value * count
            if step == 0 or (step + 1) % int(tc.get("log_every", 50)) == 0:
                log(f"[{stage}] epoch={epoch}/{total_epochs} batch={step + 1} mode={mode} loss={float(loss.detach()):.6f}")
        if not samples:
            raise ValueError("Training loader has no usable batches")
        if scheduler is not None:
            scheduler.step()
        seed_loader(val_loader, int(cfg.get("seed", 0)) + 1_000_003)
        validation = validate_model(model, val_loader, cfg, device, max_batches=max_batches)
        score = float(validation["score"])
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite validation score")
        improved = score < best_score
        best_score = min(score, best_score)
        state = {"schema": SCHEMA, "stage": stage, "cfg": cfg, "model": model.state_dict(),
                 "opt": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
                 "epoch": epoch, "metadata": metadata, "best_validation_score": best_score,
                 "validation": validation, "training": {key: value / samples for key, value in sums.items()},
                 "mode_examples": modes}
        if improved:
            save_checkpoint_atomic(state, best_path)
        save_checkpoint_atomic(state, last_path)
        with last_path.with_suffix(".training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "stage": stage, "training": state["training"],
                                     "validation": validation, "mode_examples": modes,
                                     "smoke_only": smoke}, allow_nan=False) + "\n")
        log(f"[{stage}] epoch={epoch}/{total_epochs} validation={score:.6f} best={best_score:.6f} "
            f"examples={samples} batches={steps} smoke_only={smoke}")
    return {"checkpoint": str(best_path), "last_checkpoint": str(last_path),
            "best_validation_score": best_score, "smoke_only": smoke}


def main(argv=None):
    fix_console_encoding()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0, help="Limited smoke run; 0 uses all batches")
    args = parser.parse_args(argv)
    print(json.dumps(train_model(load_config(args.config), args.resume, args.max_batches),
                     ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
