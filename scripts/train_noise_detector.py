"""Pretrain local contamination detectors on train cues; select on val BCE.

This stage contains only two small CNN detectors. It never runs the SNN and
never uses a test split to select a checkpoint. Limited-batch runs are marked
as smoke checks, not experimental results.
"""

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import bootstrap  # noqa: F401

import argparse
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import (build_cue, fix_console_encoding, load_config, log,
                    resolve_train_corrupt_modes, sample_train_severity,
                    save_checkpoint_atomic, set_seed, unpack_paired_batch)
from data.dataset import build_loaders
from models.noise_detector import (DETECTOR_CHECKPOINT_SCHEMA, LocalNoiseDetector,
                                   supervised_mask_loss, validate_detector_checkpoint)
from paths import resolve_from_root


WARMUP_MODES = ("corrupt_both", "clean_both", "clean_img_corrupt_aud",
                "corrupt_img_clean_aud")


def choose_device(cfg):
    requested = str(cfg.get("device", "cpu"))
    required = bool(cfg.get("train", {}).get("require_cuda", False)
                   or cfg.get("noise_detector", {}).get("require_cuda", False))
    if required and (not requested.startswith("cuda") or not torch.cuda.is_available()):
        raise RuntimeError("Noise detector warmup requires CUDA; CPU fallback is disabled")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable; set device=cpu for a CPU smoke test")
    return torch.device(requested)


def seed_loader(loader, seed):
    """Reset per-epoch randomness, including sampler and dataset-local RNGs."""
    set_seed(seed)
    for owner in (loader, getattr(loader, "sampler", None)):
        generator = getattr(owner, "generator", None)
        if generator is not None:
            generator.manual_seed(seed)
    dataset = getattr(loader, "dataset", None)
    visited = set()
    while dataset is not None and id(dataset) not in visited:
        visited.add(id(dataset))
        if hasattr(dataset, "_rng"):
            dataset._rng = np.random.default_rng(seed)
        dataset = getattr(dataset, "dataset", getattr(dataset, "_base", None))


def run_epoch(detectors, loader, cfg, device, epoch, optimizer=None, max_batches=None):
    training = optimizer is not None
    detectors.train(training)
    seed = int(cfg.get("seed", 0)) + (epoch * 1009 if training else 1_000_003)
    seed_loader(loader, seed)
    total_loss, examples, batches = 0.0, 0, 0
    positive_weight = float(cfg["noise_detector"].get("positive_weight", 3.0))
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        clean_img, clean_aud, _, _ = unpack_paired_batch(batch)
        clean_img, clean_aud = clean_img.to(device), clean_aud.to(device)
        # Validation covers all four conditions with fixed noise every epoch.
        modes = (WARMUP_MODES[(step + epoch - 1) % 4],) if training else WARMUP_MODES
        for mode in modes:
            family_epoch = epoch if training else 0
            # Independent cycles prevent a two-family pool from aligning with
            # only the even/odd modes and starving one modality of a family.
            family_step = step // len(WARMUP_MODES) + epoch - 1 if training else step
            img_family, aud_family = resolve_train_corrupt_modes(cfg, family_epoch, step=family_step)
            severity = sample_train_severity(cfg, family_epoch)
            image, audio, labels = build_cue(
                clean_img, clean_aud, mode, cfg, severity=severity,
                img_mode=img_family, aud_mode=aud_family, return_masks=True)
            for modality, family in (("img", img_family), ("aud", aud_family)):
                corrupted = ((modality == "img" and mode in ("corrupt_both", "corrupt_img_clean_aud"))
                             or (modality == "aud" and mode in ("corrupt_both", "clean_img_corrupt_aud")))
                if corrupted and labels[modality] is None:
                    raise ValueError(f"Detector warmup requires explicit local support labels: {family}")
            with torch.set_grad_enabled(training):
                loss = (supervised_mask_loss(detectors["img"](image), labels["img"], positive_weight)
                        + supervised_mask_loss(detectors["aud"](audio), labels["aud"], positive_weight)) / 2
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite detector warmup loss")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            size = clean_img.shape[0]
            total_loss += float(loss.detach()) * size
            examples += size
        batches += 1
    if not examples:
        raise ValueError("Detector warmup loader has no usable batches")
    return total_loss / examples, batches


def _resume_contract(cfg):
    nc = cfg["noise_detector"]
    return {
        "seed": int(cfg.get("seed", 0)),
        "batch_size": cfg.get("data", {}).get("batch_size"),
        "num_workers": cfg.get("data", {}).get("num_workers"),
        "positive_weight": float(nc.get("positive_weight", 3.0)),
        "pretrain_lr": float(nc.get("pretrain_lr", 0.001)),
        "threshold": float(nc.get("threshold", 0.5)),
    }


def train_detectors(cfg, epochs=None, resume=False, max_batches=None):
    cfg = deepcopy(cfg)
    nc = cfg.get("noise_detector", {})
    if not nc.get("enabled", False):
        raise ValueError("noise_detector.enabled must be true for detector warmup")
    cfg["noise_detector"] = nc
    epochs = int(epochs if epochs is not None else nc.get("pretrain_epochs", 10))
    if epochs <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("epochs and max_batches must be positive")
    threshold = float(nc.get("threshold", 0.5))
    positive_weight = float(nc.get("positive_weight", 3.0))
    learning_rate = float(nc.get("pretrain_lr", 0.001))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("noise_detector.threshold must be finite and lie in [0, 1]")
    if not all(math.isfinite(v) and v > 0 for v in (positive_weight, learning_rate)):
        raise ValueError("Detector positive_weight and pretrain_lr must be finite and positive")
    if not nc.get("checkpoint"):
        raise ValueError("noise_detector.checkpoint must name the best detector checkpoint")
    best_path = resolve_from_root(nc["checkpoint"])
    last_path = best_path.with_name("last.pt")
    if best_path == last_path:
        raise ValueError("noise_detector.checkpoint must differ from sibling last.pt")
    if not resume and (best_path.exists() or last_path.exists()):
        raise FileExistsError("Detector checkpoint already exists; use --resume or a new checkpoint path")
    if resume and (not last_path.is_file() or not best_path.is_file()):
        raise FileNotFoundError("Detector resume requires both best checkpoint and sibling last.pt")
    device = choose_device(cfg)
    if cfg.get("data", {}).get("dataset", "mnist_fsdd") == "mnist_fsdd":
        if not cfg.get("data", {}).get("paper_split", {}).get("enabled", False):
            raise ValueError("Detector selection requires data.paper_split.enabled for a separate val split")
    set_seed(int(cfg.get("seed", 0)))
    train_loader, val_loader = build_loaders(cfg, eval_split="val")
    for loader, split in ((train_loader, "train"), (val_loader, "val")):
        if getattr(loader.dataset, "split", None) != split:
            raise ValueError(f"Detector warmup requires a verified {split} loader")
        if getattr(loader, "persistent_workers", False):
            raise ValueError("Epoch-seeded detector resume requires persistent_workers=False")
    hidden_channels = int(nc.get("hidden_channels", 32))
    detectors = nn.ModuleDict({name: LocalNoiseDetector(hidden_channels) for name in ("img", "aud")}).to(device)
    optimizer = torch.optim.Adam(detectors.parameters(), lr=learning_rate)
    start_epoch, best_loss = 1, float("inf")
    metadata = {
        "smoke_only": max_batches is not None,
        "max_batches": max_batches,
        "selection_split": "val",
        "rng_policy": "epoch_seeded.v1",
        "modes": list(WARMUP_MODES),
    }
    if resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        best_state = torch.load(best_path, map_location="cpu", weights_only=False)
        for item in (state, best_state):
            validate_detector_checkpoint(item, cfg)
            if _resume_contract(item["cfg"]) != _resume_contract(cfg):
                raise ValueError("Detector resume optimization/RNG configuration mismatch")
            if item.get("metadata") != metadata:
                raise ValueError("Detector resume smoke/RNG/validation metadata mismatch")
        if state["best_validation_loss"] != best_state["best_validation_loss"]:
            raise ValueError("Detector last and best checkpoints disagree on best validation loss")
        if not isinstance(state.get("epoch"), int) or state["epoch"] < best_state["epoch"]:
            raise ValueError("Detector resume epoch metadata is inconsistent")
        for modality in ("img", "aud"):
            detectors[modality].load_state_dict(state["detectors"][modality], strict=True)
        optimizer.load_state_dict(state["opt"])
        start_epoch, best_loss = state["epoch"] + 1, state["best_validation_loss"]
    best_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs + 1):
        training_loss, train_batches = run_epoch(
            detectors, train_loader, cfg, device, epoch, optimizer, max_batches)
        validation_loss, val_batches = run_epoch(
            detectors, val_loader, cfg, device, epoch, max_batches=max_batches)
        improved = validation_loss < best_loss
        best_loss = min(best_loss, validation_loss)
        state = {
            "schema": DETECTOR_CHECKPOINT_SCHEMA,
            "detectors": {name: detectors[name].state_dict() for name in ("img", "aud")},
            "hidden_channels": hidden_channels,
            "threshold": threshold,
            "cfg": cfg,
            "epoch": epoch,
            "opt": optimizer.state_dict(),
            "best_validation_loss": best_loss,
            "validation_loss": validation_loss,
            "training_loss": training_loss,
            "metadata": metadata,
        }
        if improved:
            save_checkpoint_atomic(state, best_path)
        save_checkpoint_atomic(state, last_path)
        log(f"[detector] epoch={epoch}/{epochs} train_bce={training_loss:.6f} "
            f"val_bce={validation_loss:.6f} best={best_loss:.6f} "
            f"batches={train_batches}/{val_batches} smoke_only={metadata['smoke_only']}")
    return {"checkpoint": str(best_path), "last_checkpoint": str(last_path),
            "best_validation_loss": best_loss, "smoke_only": metadata["smoke_only"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int, help="Limit train and val batches; marks checkpoints smoke-only")
    args = parser.parse_args(argv)
    fix_console_encoding()
    cfg = load_config(args.config)
    cfg["_config_path"] = args.config
    train_detectors(cfg, args.epochs, args.resume, args.max_batches)


if __name__ == "__main__":
    main()
