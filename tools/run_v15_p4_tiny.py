"""Run the isolated P4 fixed 60-pair fitting diagnosis from full epoch-100 last.pt.

Example:
 python tools/run_v15_p4_tiny.py --checkpoint assets/source_last.pt \
   --train-npz assets/train.npz --val-npz assets/val.npz --output-dir outputs/p4/tiny

The source optimizer is deliberately not resumed: this diagnostic rebuilds the
original Adam parameter groups at base lr=5e-5. A/B continuation is separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch

from common import load_config, save_checkpoint_atomic, set_seed
from models.network import CrossModalSNN
from scripts.evaluate_v15pro import verify_checkpoint_config
from scripts.train import _apply_trainable_prefixes, _build_train_optimizer
from tools.evaluate_v15_foreground import save_json
from tools.export_v15pro import runtime_source_fingerprint
from tools.v15_foreground_protocol import preserve_random_state
from tools.v15_p4_diagnostics import (
    DEFAULT_OPTIONS, MODE_WEIGHTS, MODES, array_sha256, evaluate_cases, file_sha256,
    fixed_schedule, load_clean_npz, make_case_cache, train_step,
)


def validate_baseline_config(cfg):
    if (not cfg.get("noise_detector", {}).get("enabled")
            or cfg["noise_detector"].get("mask_source") != "predicted"
            or not cfg["noise_detector"].get("freeze")):
        raise ValueError("P4 requires original frozen detectors and predicted masks")
    if not cfg.get("audio_refiner", {}).get("bypass"):
        raise ValueError("P4 must preserve the baseline audio-refiner bypass")
    if cfg.get("cross_key_conditioning", {}).get("enabled"):
        raise ValueError("P4 stage capture is for the original v15 cross-key-disabled baseline")
    if cfg["corruption"].get("curriculum_mode", "fixed") != "fixed":
        raise ValueError("P4 fixed cues require the original fixed severity curriculum")
    if cfg.get("loss", {}).get("aud_sample_curriculum", {}).get("enabled"):
        raise ValueError("P4 both-present mode loss assumes original disabled target curriculum")
    if cfg.get("pair_alignment", {}).get("enabled"):
        raise ValueError("P4 baseline has pair alignment disabled")
    actual = tuple(float(cfg["cue_modes"].get("p_" + mode, 0)) for mode in MODES)
    if actual != MODE_WEIGHTS:
        raise ValueError(f"Unexpected original cue probabilities: {actual}")


def restore_history(path, completed):
    """Keep committed updates; archive a crash tail before deterministic replay."""
    if not path.is_file():
        raise ValueError("Resume history is missing")
    lines = path.read_text(encoding="utf-8").splitlines()
    committed, discarded = [], []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            discarded.append(line)
            continue
        if int(record["update"]) <= completed:
            committed.append(line)
        else:
            discarded.append(line)
    if [json.loads(line)["update"] for line in committed] != list(range(1, completed + 1)):
        raise ValueError("Resume history is missing, duplicated or reordered before the checkpoint")
    if discarded:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path.with_name("history_uncommitted_" + suffix + ".jsonl").write_text(
            "\n".join(discarded) + "\n", encoding="utf-8")
        path.write_text("\n".join(committed) + ("\n" if committed else ""), encoding="utf-8")


def run(args):
    if args.batch_size != 60:
        raise ValueError("P4 fixed diagnostic uses the same 60 pairs in every parameter update")
    if args.updates < 1 or args.eval_every < 1 or args.lr <= 0:
        raise ValueError("updates, eval-every and lr must be positive")
    memory_fraction = float(getattr(args, "cuda_memory_fraction", .6))
    if not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1:
        raise ValueError("cuda-memory-fraction must be in (0,1]")
    output = Path(args.output_dir).resolve()
    source = Path(args.checkpoint).resolve()
    if source == output or source.is_relative_to(output):
        raise ValueError("Source checkpoint must lie outside the new output directory")
    source_sha = file_sha256(source)
    state = torch.load(source, map_location="cpu", weights_only=False)
    if (not {"model", "cfg", "opt", "epoch"} <= state.keys()
            or int(state["epoch"]) != 99
            or state.get("asset_packaging", {}).get("inference_only")):
        raise ValueError("Expected complete original epoch-100 last.pt, not inference-only/best weights")
    original_cfg = deepcopy(state["cfg"])
    if args.config:
        verify_checkpoint_config(load_config(args.config), original_cfg)
    validate_baseline_config(original_cfg)
    cfg = deepcopy(original_cfg)
    cfg["device"] = args.device
    cfg["train"]["lr"] = args.lr
    cfg["train"]["lr_scheduler"] = "none"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no silent CPU fallback")
    if args.device.startswith("cuda"):
        cuda_device = torch.device(args.device)
        cuda_index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(memory_fraction, cuda_index)
    set_seed(args.seed)
    datasets = {split: load_clean_npz(path) for split, path in (
        ("train", args.train_npz), ("val", args.val_npz))}
    if "image_ids" in datasets["train"] and "image_ids" in datasets["val"]:
        if set(datasets["train"]["image_ids"]) & set(datasets["val"]["image_ids"]):
            raise ValueError("Train and validation clean image identities overlap")
    caches = {split: make_case_cache(arrays, cfg, args.seed) for split, arrays in datasets.items()}
    schedule = fixed_schedule(args.updates, args.seed)
    identity = {
        "schema": "v15-p4-tiny.v1", "source_checkpoint_sha256": source_sha,
        "source_epoch_zero_based": 99, "source_runtime_sha256": runtime_source_fingerprint(),
        "diagnostic_source_sha256": {name: file_sha256(ROOT / "tools" / name) for name in (
            "v15_p4_diagnostics.py", "run_v15_p4_tiny.py", "v15_foreground_protocol.py")},
        "source_config_sha256": __import__("hashlib").sha256(json.dumps(
            original_cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "datasets": {split: {"file_sha256": file_sha256(path),
            "arrays_sha256": array_sha256(datasets[split]),
            "sample_ids": datasets[split]["sample_ids"].tolist()} for split, path in (
                ("train", args.train_npz), ("val", args.val_npz))},
        "case_sha256": {split: {key: case["sha256"] for key, case in cache.items()}
                        for split, cache in caches.items()},
        "seed": args.seed, "updates": args.updates, "eval_every": args.eval_every,
        "batch_size": 60, "base_learning_rate": args.lr, "scheduler": None,
        "optimizer": "fresh Adam with original group multipliers/weight decay",
        "mode_weights": dict(zip(MODES, MODE_WEIGHTS)),
        "condition_update_counts": dict(Counter(schedule)), "schedule": schedule,
        "foreground_options": DEFAULT_OPTIONS, "device": args.device,
        "cuda_memory_fraction": memory_fraction if args.device.startswith("cuda") else None,
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
    }
    manifest_path = output / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError("--resume requires an existing P4 manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Resume provenance/settings differ from the original P4 run")
        for rel, digest in manifest["cue_cache_files"].items():
            if file_sha256(output / rel) != digest:
                raise ValueError(f"Cached cues changed: {rel}")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("P4 output directory is not empty; choose a new directory or --resume")
        output.mkdir(parents=True, exist_ok=True)
        files = {}
        for split, cache in caches.items():
            folder = output / "cue_cache" / split
            folder.mkdir(parents=True, exist_ok=True)
            for key, case in cache.items():
                path = folder / (key + ".npz")
                np.savez_compressed(path, **case["arrays"])
                metadata_path = folder / (key + ".json")
                save_json(metadata_path, case["metadata"])
                for artifact in (path, metadata_path):
                    files[artifact.relative_to(output).as_posix()] = file_sha256(artifact)
        manifest = {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_checkpoint": str(source), "cue_cache_files": files,
                    "status": "running", "completed_updates": 0,
                    "scope": "P4 fixed fitting diagnostic; train/val only; no P5/P6 changes",
                    "audio_stage_interpretation": "coarse -> final is paste-back; audio refiner stays bypassed",
                    "stage_interpretation": "image coarse probability -> clipped coarse+actual delta before paste-back -> final",
                    "loss": "unaltered scripts.train.compute_losses; cached cue factory; truth removed from forward kwargs"}
        save_json(manifest_path, manifest)
    model = CrossModalSNN(cfg).to(args.device)
    model.load_state_dict(state["model"], strict=True)
    del state
    _apply_trainable_prefixes(model, cfg)
    model.img_noise_detector.requires_grad_(False)
    model.aud_noise_detector.requires_grad_(False)
    optimizer = _build_train_optimizer(model, cfg)
    checkpoint = output / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = output / "history.jsonl"
    completed = 0
    if args.resume:
        if file_sha256(checkpoint) != manifest["last_checkpoint_sha256"]:
            raise ValueError("Resume checkpoint bytes differ from the committed manifest")
        if manifest.get("status") == "complete":
            for rel, digest in manifest.get("completed_artifacts_sha256", {}).items():
                if file_sha256(output / rel) != digest:
                    raise ValueError(f"Completed P4 artifact changed: {rel}")
        saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Resume checkpoint identity mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["opt"])
        completed = int(saved["completed_updates"])
        restore_history(history, completed)
        del saved
    else:
        history.touch(exist_ok=False)
    train = datasets["train"]
    image = torch.from_numpy(train["images"]).to(args.device)
    audio = torch.from_numpy(train["unique_audio"][train["audio_indices"]]).to(args.device)
    labels = torch.from_numpy(train["labels"]).to(args.device)

    def checkpoint_and_evaluate(update):
        print(f"[P4 tiny] evaluate update={update}/{args.updates}", flush=True)
        evaluation = {"update": update, "splits": {}}
        with preserve_random_state():
            for split in ("train", "val"):
                set_seed(args.seed + 900001)
                sample_dir = (output / "samples" / f"step_{update:06d}" / split
                              if update in (0, args.updates) else None)
                evaluation["splits"][split] = evaluate_cases(
                    model, datasets[split], caches[split], cfg, args.device,
                    sample_directory=sample_dir)
        save_json(output / "evaluations" / f"step_{update:06d}.json", evaluation)
        save_json(output / "metrics.json", evaluation)
        save_checkpoint_atomic({"model": model.state_dict(), "opt": optimizer.state_dict(),
            "cfg": cfg, "identity": identity, "completed_updates": update,
            "source_checkpoint_sha256": source_sha,
            "p4_diagnostic_only": True, "stage_metrics_path": f"evaluations/step_{update:06d}.json"},
            str(checkpoint))
        manifest["completed_updates"] = update
        manifest["status"] = "complete" if update == args.updates else "running"
        manifest["last_checkpoint_sha256"] = file_sha256(checkpoint)
        if update == args.updates:
            artifacts = [history, output / "metrics.json", *sorted((output / "evaluations").glob("*.json")),
                         *sorted((output / "samples").rglob("*.npz"))]
            manifest["completed_artifacts_sha256"] = {
                path.relative_to(output).as_posix(): file_sha256(path) for path in artifacts}
        save_json(manifest_path, manifest)

    if not args.resume:
        checkpoint_and_evaluate(0)
    started = time.monotonic()
    for update in range(completed + 1, args.updates + 1):
        set_seed((args.seed * 100003 + update) % (2 ** 32))
        key = schedule[update - 1]
        record = train_step(model, optimizer, image, audio, labels, cfg,
                            caches["train"][key], step=update - 1)
        record.update(update=update, condition=key, case_sha256=caches["train"][key]["sha256"],
                      elapsed_seconds=time.monotonic() - started)
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
        if update == 1 or update % 10 == 0:
            print(f"[P4 tiny] update={update}/{args.updates} loss={record['loss']:.6f} case={key}", flush=True)
        if update % args.eval_every == 0 or update == args.updates:
            checkpoint_and_evaluate(update)
    if file_sha256(source) != source_sha:
        raise RuntimeError("Source checkpoint changed during P4 execution")
    print(f"[P4 tiny] completed {args.updates} updates; {output}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional original config compatibility check")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=1504)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-memory-fraction", type=float, default=.6,
                        help="CUDA allocation cap, in (0,1]; ignored on CPU (default .6)")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
