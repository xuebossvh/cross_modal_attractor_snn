"""Opt-in publication splits shared by features, prototypes and loaders."""

import hashlib
import json
from pathlib import Path

import numpy as np


def paper_split(cfg):
    return cfg.get("data", {}).get("paper_split", {}) or {}


def audio_split(path, cfg):
    parts = Path(path).stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid FSDD recording name: {path}")
    int(parts[0])
    index = int(parts[-1])
    speaker = "_".join(parts[1:-1])
    pc = paper_split(cfg)
    if not pc.get("enabled", False):
        return "test" if index < 5 else "train"
    mode = pc.get("mode", "official")
    if mode == "speaker":
        val, test = set(pc.get("val_speakers", [])), set(pc.get("test_speakers", []))
        if not val or not test or val & test:
            raise ValueError("Speaker validation/test sets must be nonempty and disjoint")
        return "test" if speaker in test else "val" if speaker in val else "train"
    if mode != "official":
        raise ValueError(f"Unknown paper split mode: {mode}")
    return "test" if index < 5 else "val" if index < 10 else "train"


def image_indices(labels, split, cfg):
    labels = np.asarray(labels)
    if split == "test":
        return list(range(len(labels)))
    pc = paper_split(cfg)
    fraction = float(pc.get("val_fraction", 0.1))
    if not 0 < fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    rng = np.random.default_rng(int(pc.get("seed", 20260915)))
    selected = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError(f"Class {label} needs at least two images")
        indices = rng.permutation(indices)
        count = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        selected.extend(indices[:count] if split == "val" else indices[count:])
    return sorted(int(i) for i in selected)


def norm_fingerprint(paths, cfg):
    ac = cfg["audio"]
    description = {
        "split": paper_split(cfg),
        "features": {k: ac.get(k) for k in (
            "sample_rate", "n_mels", "n_frames", "duration_sec",
            "norm_percentile_lo", "norm_percentile_hi")},
        "files": [(Path(p).name, hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in sorted(paths)],
    }
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


def write_split_audit(train, evaluation, cfg):
    root = cfg.get("paper", {}).get("root")
    if not root:
        return
    def recordings(dataset):
        return {str(label): [Path(path).name for path in pool]
                for label, pool in (dataset._fsdd_paths or {}).items()}
    report = dict(protocol=paper_split(cfg), evaluation_split=evaluation.split,
        train_image_indices=train._indices, evaluation_image_indices=evaluation._indices,
        train_recordings=recordings(train), evaluation_recordings=recordings(evaluation),
        normalization=cfg.get("_audio_norm_stats"),
        image_source=train._mode, real_audio=train.use_real_audio)
    path = Path(root) / f"split_{evaluation.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != report:
        raise ValueError(f"Dataset split changed within the experiment: {path}")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
