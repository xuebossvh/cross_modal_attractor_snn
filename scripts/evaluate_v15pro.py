"""Evaluate v15pro local contamination without passing true support to detection.

Example:
  python scripts/evaluate_v15pro.py --config configs/v15pro.yaml \
      --checkpoint outputs/checkpoints/v15pro.pt --split test \
      --mask-source predicted --output outputs/v15pro/predicted.json
"""

if __package__ in (None, ""):
    import bootstrap  # noqa: F401

import argparse
import copy
import csv
import json
from pathlib import Path

import torch

from common import load_config, set_seed
from models.frozen_base import file_sha256
from scripts.noise_metrics import evaluate_local_noise


FORWARD_SECTIONS = ("snn", "dims", "index", "value", "ablation", "audio",
                    "detail_conditioning", "cross_detail_conditioning", "cross_key_conditioning",
                    "audio_local_cue", "image_refiner", "audio_refiner", "pair_alignment")


def verify_checkpoint_config(cfg, checkpoint_cfg):
    """Reject config drift even when tensors happen to retain compatible shapes.

Do not merge checkpoint configuration into a constructed model: e.g. a matching
state_dict cannot detect a changed LIF threshold or decoder paste-back policy.
Runtime batch size, worker count, output paths and device are intentionally free.
"""
    if not isinstance(checkpoint_cfg, dict) or not checkpoint_cfg:
        raise ValueError("Checkpoint must contain its resolved cfg")
    for section in (*FORWARD_SECTIONS, "corruption"):
        if cfg.get(section) != checkpoint_cfg.get(section):
            raise ValueError(f"Checkpoint/config mismatch: {section}")
    left, right = copy.deepcopy(cfg.get("noise_detector", {})), copy.deepcopy(checkpoint_cfg.get("noise_detector", {}))
    left.pop("mask_source", None)
    right.pop("mask_source", None)
    if left != right or not left.get("enabled", False):
        raise ValueError("Checkpoint/config mismatch: noise_detector")
    for section in ("data", "paper"):
        left, right = copy.deepcopy(cfg.get(section, {})), copy.deepcopy(checkpoint_cfg.get(section, {}))
        if section == "data":
            for key in ("batch_size", "num_workers", "eval_split", "train_samples_per_epoch"):
                left.pop(key, None)
                right.pop(key, None)
        if left != right:
            raise ValueError(f"Checkpoint/config mismatch: {section}")


def load_strict_checkpoint(model, path, cfg, device):
    # weights_only permits tensors plus ordinary Python configuration containers,
    # and avoids executing arbitrary checkpoint globals.
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError("Expected a training checkpoint containing model and cfg")
    verify_checkpoint_config(cfg, state.get("cfg"))
    if "_audio_norm_stats" in state["cfg"] and state["cfg"]["_audio_norm_stats"] != cfg.get("_audio_norm_stats"):
        raise ValueError("Checkpoint/config mismatch: training-only audio normalization statistics")
    model.load_state_dict(state["model"], strict=True)
    return state


def write_report(report, rows, output):
    path = Path(output)
    if path.suffix.lower() != ".json":
        raise ValueError("--output must be a .json file; per-sample CSV uses the same stem")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)  # None becomes empty CSV cells, not misleading zeros.
    return path, csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--mask-source", choices=("predicted", "oracle", "all"), default="predicted")
    parser.add_argument("--output", required=True, help="JSON file; per-sample CSV uses the same stem")
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates every sample")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    from data.dataset import build_loaders
    from data.splits import paper_split
    from models.network import CrossModalSNN
    from paths import resolve_from_root

    if cfg.get("data", {}).get("dataset", "mnist_fsdd") == "mnist_fsdd" and not paper_split(cfg).get("enabled", False):
        raise ValueError("v15pro evaluation requires independent train/val/test paper splits")
    set_seed(int(cfg.get("seed", 1234)))
    device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    # Existing loader constructs category medoids from train, never val/test.
    _, loader = build_loaders(cfg, eval_split=args.split)
    model = CrossModalSNN(cfg).to(device)
    checkpoint = resolve_from_root(args.checkpoint)
    state = load_strict_checkpoint(model, checkpoint, cfg, device)
    report, rows = evaluate_local_noise(
        model, loader, cfg, device, split=args.split,
        mask_source=args.mask_source, max_samples=args.max_samples)
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                  checkpoint_epoch=state.get("epoch"), config=str(Path(args.config).resolve()),
                  noise_training=state.get("noise_training"),
                  smoke_training=(state.get("noise_training") or {}).get("smoke_only"))
    output, csv_path = write_report(report, rows, args.output)
    print(json.dumps({"json": str(output), "csv": str(csv_path), "summary": report["summary"]},
                     ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
