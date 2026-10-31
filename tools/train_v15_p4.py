"""P4 controlled continuation from a full original v15 epoch-99 checkpoint.

This is a new experiment branch, not a relaxed resume of scripts/train.py.
The original runtime/config/losses/data sampling remain unchanged. Only the LR
schedule changes. Resume applies solely to this driver's own complete last.pt.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _folder in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from common import (load_config, log, sample_cue_mode, set_seed,
                    unpack_paired_batch)
from data.audio_features import _fsdd_train_wav_paths
from data.dataset import build_loaders
from data.splits import audio_split, image_indices, norm_fingerprint
from models.frozen_base import file_sha256, verify_audio_normalization
from models.network import CrossModalSNN
from scripts.evaluate_v15pro import verify_checkpoint_config
from scripts.noise_metrics import evaluate_local_noise, validation_score
from scripts.train import (_apply_trainable_prefixes, _build_train_optimizer,
                           _checkpoint_payload, compute_losses)
from tools.package_v15_eval_assets import runtime_fingerprint, state_dict_sha256


ORIGINAL_RUNTIME_SHA256 = "b9b072bdfc1aa26fe1b3c95976dde86364a1b131b3845cb39e5be1325f8244d7"
SOURCE_EPOCH = 99
CONTINUATION_EPOCHS = 30
MIN_LR = 1e-5
RESTART_BASE_LR = 5e-5


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def state_sha256(value):
    """Canonical nested tensor/state digest, independent of serialization/device."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(json_bytes(["tensor", str(tensor.dtype), list(tensor.shape)]))
            digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(item, dict):
            digest.update(b"dict[")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
            digest.update(b"]")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for member in item:
                visit(member)
            digest.update(b"]")
        else:
            digest.update(json_bytes([type(item).__name__, item]))
    visit(value)
    return digest.hexdigest()


def config_identity(cfg):
    result = deepcopy(cfg)
    result.pop("_config_path", None)  # Location of the YAML is not model semantics.
    return result


def verify_source_checkpoint(source, cfg=None):
    if (not isinstance(source, dict) or source.get("inference_only")
            or source.get("asset_packaging", {}).get("inference_only")):
        raise ValueError("P4 requires the full training checkpoint, not inference_only assets")
    if source.get("p4") is not None:
        raise ValueError("The source must be original v15, not a P4 branch")
    for key in ("model", "cfg", "opt", "sched"):
        if not isinstance(source.get(key), dict) or not source[key]:
            raise ValueError(f"Full source checkpoint must contain {key}")
    if source.get("epoch") != SOURCE_EPOCH:
        raise ValueError("P4 source epoch must be 99 (100 completed epochs)")
    if source.get("noise_training", {}).get("smoke_only", False):
        raise ValueError("A smoke checkpoint cannot initialize formal continuation")
    saved = deepcopy(source["cfg"])
    selected = deepcopy(saved if cfg is None else cfg)
    if "_audio_norm_stats" not in selected:
        selected["_audio_norm_stats"] = deepcopy(saved.get("_audio_norm_stats"))
    verify_checkpoint_config(selected, saved)
    if config_identity(selected) != config_identity(saved):
        changed = [key for key in set(selected) | set(saved)
                   if key != "_config_path" and selected.get(key) != saved.get(key)]
        raise ValueError(f"P4 source/config mismatch: {sorted(changed)}")
    tc, dc, nc, vc = (saved.get(k, {}) for k in ("train", "data", "noise_detector", "validation"))
    if tc.get("epochs") != 100 or not tc.get("epoch_seeded") or tc.get("freeze_base"):
        raise ValueError("P4 requires the original 100-epoch, epoch_seeded, unfrozen-base protocol")
    if tc.get("evaluation_only") or tc.get("trainable_prefixes") or saved.get("decoder_pretrain", {}).get("enabled"):
        raise ValueError("Unexpected original training/freezing protocol")
    if dc.get("batch_size") != 128 or dc.get("train_subset", 0) or dc.get("train_samples_per_epoch", 0):
        raise ValueError("P4 requires the full original train split and batch_size=128")
    if dc.get("dataset", "mnist_fsdd") != "mnist_fsdd" or not dc.get("use_mnist", True):
        raise ValueError("P4 requires real MNIST/FSDD")
    if not dc.get("paper_split", {}).get("enabled") or dc.get("pairing", {}).get("enabled"):
        raise ValueError("P4 requires original publication splits/category pairing")
    if not saved.get("audio", {}).get("use_real_audio", True) or not isinstance(saved.get("_audio_norm_stats"), dict):
        raise ValueError("P4 requires real audio and saved training normalization")
    if not nc.get("enabled") or not nc.get("freeze") or nc.get("mask_source") != "predicted" or nc.get("allow_smoke_checkpoint"):
        raise ValueError("P4 must preserve frozen detectors and predicted masks")
    if (not vc.get("enabled") or vc.get("split") != "val" or vc.get("every_epochs") != 1
            or vc.get("max_batches", 0) or vc.get("max_samples", 0)):
        raise ValueError("P4 requires full validation every epoch")
    if tc.get("lr_scheduler") != "cosine" or source["sched"].get("last_epoch") != 100:
        raise ValueError("Source scheduler must have completed the original 100-epoch cosine")
    if source["sched"].get("T_max") != 100 or not math.isclose(source["sched"].get("eta_min", -1), MIN_LR):
        raise ValueError("Source cosine schedule differs from the original protocol")
    if not source["opt"].get("state") or not source["opt"].get("param_groups"):
        raise ValueError("Source Adam moments/parameter groups are missing")
    if any(not math.isclose(group["lr"], MIN_LR, rel_tol=1e-8) for group in source["opt"]["param_groups"]):
        raise ValueError("All source groups must be at the original terminal lr=1e-5")
    return saved


def apply_original_freezing(model, cfg):
    _apply_trainable_prefixes(model, cfg)
    if model.use_noise_detector and cfg["noise_detector"].get("freeze", False):
        model.img_noise_detector.requires_grad_(False)
        model.aud_noise_detector.requires_grad_(False)
    # Audio-refiner bypass is frozen by the original model constructor.
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def initialize_branch(model, source, cfg, branch, epochs=CONTINUATION_EPOCHS):
    """Load identical source model+Adam state, then explicitly create a new LR policy."""
    if branch not in ("keep_lr", "restart_lr") or epochs != CONTINUATION_EPOCHS:
        raise ValueError("P4 is a fixed 30-epoch keep_lr/restart_lr comparison")
    model.load_state_dict(source["model"], strict=True)
    trainable_names = apply_original_freezing(model, cfg)
    optimizer = _build_train_optimizer(model, cfg)
    base_lr = float(cfg["train"]["lr"])
    multipliers = [float(group["lr"]) / base_lr for group in optimizer.param_groups]
    expected_groups = optimizer.state_dict()["param_groups"]
    if [g["params"] for g in expected_groups] != [g["params"] for g in source["opt"]["param_groups"]]:
        raise ValueError("Source optimizer groups do not match original trainable parameters")
    if not np.allclose(source["sched"].get("base_lrs", []),
                       [g["lr"] for g in expected_groups], rtol=1e-12, atol=0):
        raise ValueError("Source scheduler base learning rates disagree with original optimizer groups")
    optimizer.load_state_dict(deepcopy(source["opt"]))
    before_model = state_dict_sha256(model.state_dict())
    before_optimizer = state_sha256(optimizer.state_dict())
    if before_model != state_dict_sha256(source["model"]) or before_optimizer != state_sha256(source["opt"]):
        raise ValueError("Source model/optimizer state changed during initialization")
    moments_before = state_sha256(optimizer.state_dict()["state"])
    for group, multiplier in zip(optimizer.param_groups, multipliers):
        lr = MIN_LR if branch == "keep_lr" else RESTART_BASE_LR * multiplier
        if lr < MIN_LR:
            raise ValueError("A group multiplier would start below the common LR floor")
        group["lr"] = group["initial_lr"] = lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=MIN_LR)
    if state_sha256(optimizer.state_dict()["state"]) != moments_before:
        raise ValueError("Branch LR initialization modified Adam moments")
    names_by_id = {id(p): n for n, p in model.named_parameters()}
    metadata = dict(initialization="new_branch_from_original_full_checkpoint",
                    initial_model_sha256=before_model,
                    initial_optimizer_sha256=before_optimizer,
                    initial_optimizer_moments_sha256=moments_before,
                    source_scheduler_sha256=state_sha256(source["sched"]),
                    source_optimizer_steps=optimizer_steps(source["opt"]),
                    trainable_names=trainable_names,
                    frozen_names=[n for n, p in model.named_parameters() if not p.requires_grad],
                    parameter_groups=[dict(index=i, original_lr_multiplier=m,
                        initial_lr=g["lr"], parameters=[names_by_id[id(p)] for p in g["params"]])
                        for i, (g, m) in enumerate(zip(optimizer.param_groups, multipliers))])
    return optimizer, scheduler, metadata


def optimizer_steps(state):
    counts = Counter(int(value["step"]) for value in state["state"].values() if "step" in value)
    return {str(step): count for step, count in sorted(counts.items())}


def prepare_loader_config(cfg, data_root, fsdd_root, norm_stats, output):
    """Relocate data only after checking original model semantics; never mutate cfg."""
    prepared = deepcopy(cfg)
    prepared["data"]["root"] = str(Path(data_root).resolve())
    prepared["audio"]["fsdd_root"] = str(Path(fsdd_root).resolve())
    prepared["audio"]["norm_stats_path"] = str(Path(norm_stats).resolve())
    prepared["audio"]["auto_download"] = False
    prepared.setdefault("paper", {})
    if prepared["paper"] is None:
        prepared["paper"] = {}
    prepared["paper"]["root"] = str(Path(output).resolve() / "data_audit")
    stats = torch.load(norm_stats, map_location="cpu", weights_only=True)
    if stats != cfg["_audio_norm_stats"]:
        raise ValueError("Relocated normalization file differs from source checkpoint")
    train_paths = _fsdd_train_wav_paths(prepared)
    fingerprint = norm_fingerprint(train_paths, prepared)
    if not train_paths or fingerprint != stats.get("train_fingerprint"):
        raise ValueError("Relocated training WAV/config fingerprint differs from source normalization")
    if (stats.get("n_mels"), stats.get("n_frames")) != (cfg["audio"]["n_mels"], cfg["audio"]["n_frames"]):
        raise ValueError("Source normalization shape mismatch")
    # No download/fallback is permitted; the original loader's download=True is
    # then a no-op. Only training MNIST is needed for train+validation.
    from torchvision.datasets import MNIST
    MNIST(root=prepared["data"]["root"], train=True, download=False)
    return prepared, dict(train_normalization_fingerprint=fingerprint,
                          norm_stats_sha256=file_sha256(norm_stats),
                          norm_stats=stats, training_wav_count=len(train_paths))


def verify_loader_splits(train_loader, val_loader, cfg, source_audit=None):
    report = dict(protocol=cfg["data"]["paper_split"], evaluation_split="val",
                  normalization=cfg["_audio_norm_stats"], image_source="mnist", real_audio=True)
    for name, dataset in (("train", train_loader.dataset), ("evaluation", val_loader.dataset)):
        split = "train" if name == "train" else "val"
        if dataset.split != split or dataset._mode != "mnist" or not dataset.use_real_audio:
            raise ValueError("Unexpected dataset split or synthetic fallback")
        expected = image_indices(dataset._base.targets, split, cfg)
        if dataset._indices != expected:
            raise ValueError("Relocated MNIST indices differ from original config")
        recordings = {str(label): [Path(p).name for p in paths]
                      for label, paths in dataset._fsdd_paths.items()}
        for paths in dataset._fsdd_paths.values():
            if any(audio_split(path, cfg) != split for path in paths):
                raise ValueError("Relocated FSDD split differs from original config")
        report[name + "_image_indices"] = list(dataset._indices)
        report[name + "_recordings"] = recordings
    if set(report["train_image_indices"]) & set(report["evaluation_image_indices"]):
        raise ValueError("Train/validation image leakage")
    if source_audit:
        saved = json.loads(Path(source_audit).read_text(encoding="utf-8"))
        if saved != report:
            raise ValueError("Relocated split differs from original split audit")
    return report


def validate_full(model, loader, cfg, device, max_batches=0):
    """The exact original validate_local_noise evaluator and score, with full report."""
    report, _rows = evaluate_local_noise(model, loader, cfg, device,
        split="val", mask_source="predicted", max_batches=max_batches)
    if len(report["conditions"]) != 7:
        raise ValueError("P4 validation requires the original seven noise/cue cases")
    if not max_batches and report["n_unique_samples"] != len(loader.dataset):
        raise ValueError("P4 validation did not cover the complete validation split")
    summary = report["summary"]
    metrics = dict(score=validation_score(summary, cfg),
        img_mse=summary["img"]["global_recovered_mse"],
        aud_mse=summary["aud"]["global_recovered_mse"],
        img_masked_mse=summary["img"]["corrupt_recovered_mse"],
        aud_masked_mse=summary["aud"]["corrupt_recovered_mse"],
        acc=summary["accuracy"], n=summary["n"], pair_i2a_acc=None, pair_a2i_acc=None)
    if not math.isfinite(metrics["score"]):
        raise FloatingPointError("Non-finite validation score")
    return metrics, report


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def durable_checkpoint(payload, path):
    """Persist and fsync the complete last checkpoint before exporting records."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def verify_resume(state, expected):
    if state.get("p4", {}).get("identity") != expected:
        raise ValueError("P4 resume branch/source/config/runtime/data/smoke identity mismatch")
    if config_identity(state.get("cfg", {})) != expected["config"]:
        raise ValueError("P4 resume checkpoint cfg mismatch")
    epoch = state.get("epoch")
    if not isinstance(epoch, int) or not SOURCE_EPOCH < epoch <= SOURCE_EPOCH + CONTINUATION_EPOCHS:
        raise ValueError("Invalid P4 resume epoch")
    records = state["p4"].get("records", [])
    if [r["epoch"] for r in records] != list(range(SOURCE_EPOCH + 1, epoch + 1)):
        raise ValueError("P4 resume records are incomplete or repeat epochs")
    actual = dict(model=state_dict_sha256(state["model"]), optimizer=state_sha256(state["opt"]),
                  scheduler=state_sha256(state["sched"]))
    if actual != state["p4"].get("state_sha256"):
        raise ValueError("P4 resume checkpoint state hash mismatch")
    if state["sched"].get("last_epoch") != epoch - SOURCE_EPOCH:
        raise ValueError("P4 resume scheduler epoch mismatch")


def train_epoch(model, optimizer, loader, cfg, device, epoch, max_batches=0):
    epoch_seed = int(cfg["seed"]) + epoch * 1009
    set_seed(epoch_seed)
    generator = loader.generator or getattr(loader.sampler, "generator", None)
    if generator is not None:
        generator.manual_seed(epoch_seed)
    loader.dataset._rng = np.random.default_rng(epoch_seed)
    model.train()
    proto_img = loader.dataset.prototype_img.to(device)
    proto_aud = loader.dataset.prototype_aud.to(device)
    sums, counts, cues = defaultdict(float), Counter(), Counter()
    losses, batches, samples, updates = 0.0, 0, 0, 0
    stream = hashlib.sha256()
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        set_seed(((int(cfg["seed"]) + epoch * 1009) * 100003 + step) % (2**32))
        # Digest the actual clean input order, including stochastic audio pairing.
        stream.update(state_sha256(batch).encode())
        img, aud, labels, pair_ids = unpack_paired_batch(batch)
        img, aud, labels = img.to(device), aud.to(device), labels.to(device)
        if pair_ids is not None:
            pair_ids = pair_ids.to(device)
        mode = sample_cue_mode(cfg)
        loss, logs = compute_losses(model, img, aud, labels, mode, cfg,
            proto_img, proto_aud, pair_ids=pair_ids, epoch=epoch, step=step)
        optimizer.zero_grad()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}")
        if not loss.requires_grad:
            raise RuntimeError("P4 loss has no gradient path")
        loss.backward()
        parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not parameters or not torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all():
            raise FloatingPointError(f"Non-finite/missing gradients at epoch={epoch}, step={step}")
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
        optimizer.step()
        for key, value in logs.items():
            number = float(value)
            if not math.isfinite(number):
                raise FloatingPointError(f"Non-finite loss diagnostic {key} at epoch={epoch}, step={step}")
            sums[key] += number
            counts[key] += 1
        losses += float(loss.detach())
        batches += 1
        updates += 1
        samples += len(labels)
        cues[mode] += 1
        if step % int(cfg["train"].get("log_every", 50)) == 0:
            log(f"[P4 epoch {epoch} step {step}/{len(loader)}] loss={float(loss.detach()):.6f} cue={mode}")
    if not batches:
        raise RuntimeError("Empty training epoch")
    return dict(epoch_loss=losses / batches,
        loss_components={k: sums[k] / counts[k] for k in sums},
        loss_component_batches=dict(counts), batches=batches, samples=samples,
        optimizer_updates=updates, cue_mode_batches=dict(cues),
        epoch_seed=epoch_seed, clean_batch_stream_sha256=stream.hexdigest())


def run(args):
    if args.epochs != CONTINUATION_EPOCHS or args.threads < 1 or args.max_batches < 0:
        raise ValueError("P4 requires --epochs 30, positive threads, nonnegative max-batches")
    if not math.isfinite(args.cuda_memory_fraction) or not 0 < args.cuda_memory_fraction <= 1:
        raise ValueError("--cuda-memory-fraction must be finite and in (0, 1]")
    if args.stop_after_epoch is not None and not 100 <= args.stop_after_epoch <= 129:
        raise ValueError("--stop-after-epoch is an absolute epoch in 100..129")
    smoke = bool(args.max_batches or args.stop_after_epoch is not None)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no CPU fallback")
    if device.type != "cuda" and not smoke:
        raise ValueError("Formal P4 requires CUDA; CPU is allowed only for explicit smoke runs")
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=cuda_index)
    torch.set_num_threads(args.threads)
    source_path, output = Path(args.source_checkpoint).resolve(), Path(args.output).resolve()
    if output == source_path.parent or output in source_path.parents:
        raise ValueError("P4 output must not contain or equal the source checkpoint directory")
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory, or --resume for this P4 branch")
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    cfg = verify_source_checkpoint(source, load_config(args.config) if args.config else None)
    runtime = runtime_fingerprint()
    if runtime["source_sha256"] != ORIGINAL_RUNTIME_SHA256:
        raise ValueError("Original v15 runtime source hash changed")
    output.mkdir(parents=True, exist_ok=True)
    loader_cfg, normalization = prepare_loader_config(
        cfg, args.data_root, args.fsdd_root, args.norm_stats, output)
    set_seed(int(cfg["seed"]))
    train_loader, val_loader = build_loaders(loader_cfg, eval_split="val")
    if loader_cfg["_audio_norm_stats"] != cfg["_audio_norm_stats"]:
        raise ValueError("Loader changed original training normalization")
    split_report = verify_loader_splits(train_loader, val_loader, cfg, args.source_split_audit)
    model = CrossModalSNN(cfg).to(device)
    optimizer, scheduler, initial = initialize_branch(model, source, cfg, args.branch, args.epochs)
    verify_audio_normalization(model, cfg)
    identity = dict(schema_version=1, branch=args.branch, source_sha256=file_sha256(source_path),
        source_epoch=SOURCE_EPOCH, config=config_identity(cfg),
        config_sha256=state_sha256(config_identity(cfg)),
        runtime_sha256=runtime["source_sha256"], driver_sha256=file_sha256(__file__),
        state_helper_sha256=runtime["packager_sha256"],
        packages=runtime["packages"], torch_cuda_build=runtime["torch_cuda_build"],
        device=str(device), threads=args.threads,
        cuda_memory_fraction=args.cuda_memory_fraction if device.type == "cuda" else None,
        seed=int(cfg["seed"]), epochs=args.epochs,
        max_batches=args.max_batches, smoke_only=smoke,
        normalization=normalization, split_sha256=state_sha256(split_report),
        initialization=initial)
    last_path, best_path = output / "last.pt", output / "best.pt"
    records, start, best = [], SOURCE_EPOCH + 1, float("inf")
    best_epoch = None
    if args.resume:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        verify_resume(state, identity)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        records = state["p4"]["records"]
        start, best = state["epoch"] + 1, float(state["best_validation_score"])
        best_epoch = state["p4"]["best_epoch"]
        # Keep healthy checkpoint bytes stable, including when the last epoch
        # is best. A crash between last and best may leave a missing/older best.
        if best_path.is_file():
            best_state = torch.load(best_path, map_location="cpu", weights_only=True)
            verify_resume(best_state, identity)
            if best_state["p4"]["records"] != records[:best_state["epoch"] - SOURCE_EPOCH]:
                raise ValueError("P4 best checkpoint records disagree with last")
            if best_state["epoch"] == best_epoch and best_state["best_validation_score"] == best:
                if (best_epoch == state["epoch"]
                        and best_state["p4"]["state_sha256"] != state["p4"]["state_sha256"]):
                    raise ValueError("P4 best and last state differ at the same epoch")
            elif best_epoch == state["epoch"] and best_state["epoch"] < best_epoch:
                log("[P4 resume] repairing older best from the durable last checkpoint")
                durable_checkpoint(state, best_path)
            else:
                raise ValueError("P4 best checkpoint disagrees with the complete last checkpoint")
            del best_state
        elif best_epoch == state["epoch"]:
            log("[P4 resume] restoring missing best from the durable last checkpoint")
            durable_checkpoint(state, best_path)
        else:
            raise FileNotFoundError("P4 best checkpoint is missing and cannot be reconstructed from last")
        atomic_json(output / "epochs.json", records)
        atomic_json(output / "status.json", dict(last_epoch=state["epoch"], best_epoch=best_epoch,
            best_validation_score=best, smoke_only=smoke,
            complete=state["epoch"] == SOURCE_EPOCH + args.epochs,
            total_branch_updates=records[-1]["total_branch_updates"]))
    else:
        atomic_json(output / "manifest.json", dict(identity=identity,
            source_checkpoint=str(source_path), output=str(output),
            source_best_validation_score=source.get("best_validation_score"),
            source_validation={k: None if isinstance(v, float) and math.isnan(v) else v
                               for k, v in (source.get("validation") or {}).items()},
            best_selection="minimum original validation_score within this 30-epoch branch",
            runtime=runtime, loader_config=loader_cfg,
            created=datetime.now(timezone.utc).isoformat()))
        atomic_json(output / "split_audit.json", split_report)
    del source
    for epoch in range(start, SOURCE_EPOCH + args.epochs + 1):
        if args.stop_after_epoch is not None and epoch > args.stop_after_epoch:
            break
        began = time.monotonic()
        lr_used = [float(group["lr"]) for group in optimizer.param_groups]
        record = train_epoch(model, optimizer, train_loader, cfg, device, epoch, args.max_batches)
        scheduler.step()
        metrics, report = validate_full(model, val_loader, cfg, device, args.max_batches)
        is_best = metrics["score"] < best
        if is_best:
            best, best_epoch = metrics["score"], epoch
        record.update(epoch=epoch, completed_source_epochs=100, lr_groups=lr_used,
            next_lr_groups=[float(group["lr"]) for group in optimizer.param_groups],
            validation=metrics, validation_report=report, is_best=is_best,
            best_validation_score=best, best_epoch=best_epoch, smoke_only=smoke,
            elapsed_seconds=time.monotonic() - began,
            total_branch_updates=sum(r["optimizer_updates"] for r in records) + record["optimizer_updates"],
            total_branch_samples=sum(r["samples"] for r in records) + record["samples"],
            optimizer_steps=optimizer_steps(optimizer.state_dict()))
        hashes = dict(model=state_dict_sha256(model.state_dict()),
                      optimizer=state_sha256(optimizer.state_dict()),
                      scheduler=state_sha256(scheduler.state_dict()))
        record["state_sha256"] = hashes
        records.append(record)
        payload = _checkpoint_payload(model, optimizer, scheduler, cfg, epoch)
        payload.update(best_validation_score=best, validation=metrics,
                       validation_report=report)
        payload["p4"] = dict(identity=identity, smoke_only=smoke, records=records,
                             best_epoch=best_epoch, state_sha256=hashes)
        payload["noise_training"]["smoke_only"] = smoke
        durable_checkpoint(payload, last_path)
        if is_best:
            durable_checkpoint(payload, best_path)
        atomic_json(output / "epochs.json", records)
        atomic_json(output / "status.json", dict(last_epoch=epoch, best_epoch=best_epoch,
            best_validation_score=best, smoke_only=smoke,
            complete=epoch == SOURCE_EPOCH + args.epochs,
            total_branch_updates=record["total_branch_updates"]))
        log(f"[P4 {args.branch} epoch {epoch}] loss={record['epoch_loss']:.6f} "
            f"val={metrics['score']:.8f} best={best:.8f} lr={lr_used} smoke_only={smoke}")
    return dict(output=str(output), start_epoch=start,
                last_epoch=records[-1]["epoch"] if records else SOURCE_EPOCH,
                smoke_only=smoke, best_validation_score=best)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-checkpoint", required=True)
    ap.add_argument("--config", help="Original resolved config; omit to use source checkpoint cfg")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--fsdd-root", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--source-split-audit", help="Optional original split_val.json for direct cross-check")
    ap.add_argument("--output", required=True)
    ap.add_argument("--branch", choices=("keep_lr", "restart_lr"), required=True)
    ap.add_argument("--epochs", type=int, default=CONTINUATION_EPOCHS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cuda-memory-fraction", type=float, default=0.6,
                    help="CUDA allocator limit in (0, 1]; ignored on CPU; fixed across CUDA resume")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="Explicit smoke only: limit train and validation batches")
    ap.add_argument("--stop-after-epoch", type=int,
                    help="Explicit smoke only: stop after absolute epoch (100..129)")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
