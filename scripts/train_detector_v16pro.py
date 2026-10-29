"""Train v16 blind local-noise detectors on all thirteen cue layouts.

Known missing support is excluded from noise BCE. Selection uses validation
only; checkpoint provenance records the cue protocol, source, optimizer and RNG.
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
import random

import numpy as np
import torch
import torch.nn as nn

from common import (fix_console_encoding, load_config, log, save_checkpoint_atomic,
                    set_seed, unpack_paired_batch)
from data.dataset import build_loaders
from data.v16_protocol import (MODES, detector_mask_loss, make_cues,
                               protocol_contract, sample_mode)
from models.noise_detector import (DETECTOR_CHECKPOINT_SCHEMA, LocalNoiseDetector,
                                   validate_detector_checkpoint)
from paths import resolve_from_root
from scripts.train_noise_detector import choose_device, seed_loader


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    files = ("scripts/train_detector_v16pro.py", "scripts/train_noise_detector.py",
             "data/v16_protocol.py", "data/corruption.py", "data/dataset.py",
             "models/noise_detector.py", "common.py")
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _resume_contract(cfg):
    noise = deepcopy(cfg.get("noise_detector", {}))
    for key in ("checkpoint", "last_checkpoint", "pretrain_epochs", "allow_smoke_checkpoint",
                "freeze", "mask_source", "require_pretrained"):
        noise.pop(key, None)
    return {
        "seed": int(cfg.get("seed", 0)), "protocol": protocol_contract(cfg),
        "data": deepcopy(cfg.get("data", {})), "audio": deepcopy(cfg.get("audio", {})),
        "corruption": deepcopy(cfg.get("corruption", {})), "noise_detector": noise,
        "num_classes": cfg.get("dims", {}).get("num_classes"),
    }


def validate_v16_detector_checkpoint(state, cfg):
    """Validate v16 support semantics before using a legacy-compatible pair."""
    validate_detector_checkpoint(state, cfg)
    metadata = state.get("metadata", {})
    if metadata.get("protocol") != protocol_contract(cfg):
        raise ValueError("v16 detector cue protocol mismatch")
    if metadata.get("selection_split") != "val" or metadata.get("missing_policy") != "excluded_from_bce":
        raise ValueError("v16 detector lacks validation/missing-region provenance")
    if metadata.get("smoke_only", False) and not cfg.get("noise_detector", {}).get("allow_smoke_checkpoint", False):
        raise ValueError("Smoke-only detector checkpoint cannot initialize a formal experiment")


def run_epoch(detectors, loader, cfg, device, epoch, optimizer=None, max_batches=None):
    training = optimizer is not None
    detectors.train(training)
    base_seed = int(cfg.get("seed", 0))
    seed_loader(loader, base_seed + (epoch * 1009 if training else 1_000_003))
    weighted_loss, examples, batches, offset = 0.0, 0, 0, 0
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        image, audio, _, ids = unpack_paired_batch(batch)
        image, audio = image.to(device), audio.to(device)
        if ids is None:
            ids = list(range(offset, offset + len(image)))
        modes = (sample_mode(cfg),) if training else MODES
        for mode in modes:
            cues = make_cues(image, audio, mode, cfg, ids=ids,
                             seed=None if training else base_seed + 1_000_003,
                             protocol="detector_validation" if not training else "train")
            with torch.set_grad_enabled(training):
                losses = [detector_mask_loss(detectors[name](cues[name]), cues["noise"][name],
                                             cues["missing"][name],
                                             cfg["noise_detector"].get("positive_weight", 3.0))
                          for name in ("img", "aud") if cues[name] is not None]
                loss = torch.stack(losses).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite v16 detector loss")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            weighted_loss += float(loss.detach()) * len(image)
            examples += len(image)
        batches += 1
        offset += len(image)
    if not examples:
        raise ValueError("Detector loader contains no usable observations")
    return weighted_loss / examples, batches


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def train_detectors(cfg, epochs=None, resume=False, max_batches=None):
    cfg = deepcopy(cfg)
    nc = cfg.get("noise_detector", {}) or {}
    if not nc.get("enabled", False):
        raise ValueError("v16 detector training requires noise_detector.enabled")
    epochs = int(nc.get("pretrain_epochs", 10) if epochs is None else epochs)
    if epochs <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("epochs and max_batches must be positive")
    for name, default in (("positive_weight", 3.0), ("pretrain_lr", 0.001), ("pretrain_lr_gamma", 1.0)):
        value = float(nc.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"noise_detector.{name} must be finite and positive")
    threshold = float(nc.get("threshold", 0.5))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("noise_detector.threshold must lie in [0,1]")
    if not nc.get("checkpoint"):
        raise ValueError("noise_detector.checkpoint is required")
    best = resolve_from_root(nc["checkpoint"])
    last = resolve_from_root(nc["last_checkpoint"]) if nc.get("last_checkpoint") else best.with_name("last.pt")
    if best == last:
        raise ValueError("Best and last detector paths must differ")
    if not resume and (best.exists() or last.exists()):
        raise FileExistsError("Detector output exists; use --resume or a new output")
    if resume and (not best.is_file() or not last.is_file()):
        raise FileNotFoundError("Detector resume requires both best and last checkpoints")
    device = choose_device(cfg)
    if cfg.get("data", {}).get("dataset", "mnist_fsdd") == "mnist_fsdd" and not cfg.get("data", {}).get("paper_split", {}).get("enabled", False):
        raise ValueError("v16 requires a separate validation split")
    set_seed(int(cfg.get("seed", 0)))
    train_loader, val_loader = build_loaders(cfg, eval_split="val")
    for loader, split in ((train_loader, "train"), (val_loader, "val")):
        if getattr(loader.dataset, "split", None) != split:
            raise ValueError(f"Detector requires a verified {split} split")
        if getattr(loader, "persistent_workers", False):
            raise ValueError("Reproducible epoch resume requires persistent_workers=False")
    synthetic = (any(getattr(loader.dataset, "_mode", None) == "synthetic" for loader in (train_loader, val_loader))
                 or not cfg.get("audio", {}).get("use_real_audio", True))
    metadata = {
        "smoke_only": max_batches is not None or synthetic, "max_batches": max_batches,
        "selection_split": "val", "rng_policy": "epoch_seeded.v16.1",
        "missing_policy": "excluded_from_bce", "protocol": protocol_contract(cfg),
        "source_sha256": source_fingerprint(),
    }
    contract = _resume_contract(cfg)
    hidden = int(nc.get("hidden_channels", 32))
    detectors = nn.ModuleDict({name: LocalNoiseDetector(hidden) for name in ("img", "aud")}).to(device)
    optimizer = torch.optim.Adam(detectors.parameters(), lr=float(nc.get("pretrain_lr", 0.001)))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(nc.get("pretrain_lr_gamma", 1.0)))
    start_epoch, best_loss = 1, float("inf")
    if resume:
        state = torch.load(last, map_location=device, weights_only=False)
        best_state = torch.load(best, map_location="cpu", weights_only=False)
        for item in (state, best_state):
            validate_detector_checkpoint(item, cfg)
            if item.get("resume_contract") != contract:
                raise ValueError("Detector resume configuration mismatch")
            if item.get("metadata") != metadata:
                raise ValueError("Detector resume provenance/smoke metadata mismatch")
        if state.get("best_validation_loss") != best_state.get("best_validation_loss") or state.get("epoch", 0) < best_state.get("epoch", 0):
            raise ValueError("Detector best/last checkpoint provenance is inconsistent")
        for name in ("img", "aud"):
            detectors[name].load_state_dict(state["detectors"][name], strict=True)
        optimizer.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"])
        start_epoch, best_loss = state["epoch"] + 1, state["best_validation_loss"]
    best.parent.mkdir(parents=True, exist_ok=True)
    last.parent.mkdir(parents=True, exist_ok=True)
    history_path = last.with_suffix(".training.jsonl")
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_batches = run_epoch(detectors, train_loader, cfg, device, epoch, optimizer, max_batches)
        val_loss, val_batches = run_epoch(detectors, val_loader, cfg, device, epoch, max_batches=max_batches)
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        scheduler.step()
        state = {
            "schema": DETECTOR_CHECKPOINT_SCHEMA,
            "detectors": {name: detectors[name].state_dict() for name in ("img", "aud")},
            "hidden_channels": hidden, "threshold": threshold, "cfg": cfg,
            "epoch": epoch, "opt": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": _rng_state(), "best_validation_loss": best_loss,
            "validation_loss": val_loss, "training_loss": train_loss,
            "metadata": metadata, "resume_contract": contract,
        }
        if improved:
            save_checkpoint_atomic(state, best)
        save_checkpoint_atomic(state, last)
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "train_bce": train_loss, "val_bce": val_loss,
                                     "best_val_bce": best_loss, "smoke_only": metadata["smoke_only"]}) + "\n")
        log(f"[v16 detector] epoch={epoch}/{epochs} train_bce={train_loss:.6f} "
            f"val_bce={val_loss:.6f} best={best_loss:.6f} batches={train_batches}/{val_batches} "
            f"smoke_only={metadata['smoke_only']}")
    return {"checkpoint": str(best), "last_checkpoint": str(last), "best_validation_loss": best_loss,
            "smoke_only": metadata["smoke_only"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int, help="Limit both splits; marks output smoke-only")
    args = parser.parse_args(argv)
    fix_console_encoding()
    cfg = load_config(args.config)
    train_detectors(cfg, args.epochs, args.resume, args.max_batches)


if __name__ == "__main__":
    main()
