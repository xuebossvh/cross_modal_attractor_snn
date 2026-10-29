"""Export v15pro inference as figures/, logs/, and tables/ without retraining.

Use --root outputs/v15pro for generated per-seed runs, or provide an explicit
--config, --checkpoint and --output-dir. Existing complete exports are verified
and skipped; different settings or incomplete output require a fresh directory.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import traceback

import torch

from common import load_config, set_seed, unpack_paired_batch
from data.dataset import build_loaders
from models.network import CrossModalSNN
from models.frozen_base import file_sha256
from scripts.evaluate_v15pro import load_strict_checkpoint
from scripts.noise_metrics import evaluate_local_noise
from tools.v15_examples import export_examples
from tools.v15_report import export_numeric_tables, plot_metric_summaries


def source_fingerprint():
    digest = hashlib.sha256()
    files = [Path(__file__), PROJECT_ROOT / "tools/v15_examples.py",
             PROJECT_ROOT / "tools/v15_report.py",
             PROJECT_ROOT / "scripts/noise_metrics.py",
             PROJECT_ROOT / "scripts/evaluate_v15pro.py"]
    for path in files:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_source_fingerprint():
    """Also invalidate exports when model, data, or inference semantics change."""
    digest = hashlib.sha256()
    files = [PROJECT_ROOT / "common.py", PROJECT_ROOT / "paths.py"]
    for directory in ("models", "data", "scripts"):
        files.extend((PROJECT_ROOT / directory).glob("*.py"))
    for path in sorted(files):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
        self.encoding = "utf-8"

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


class ProgressLoader:
    def __init__(self, loader, example_limit=0):
        self.loader = loader
        self.dataset = loader.dataset
        self.example_limit = example_limit
        self.examples = []
        self.seen = 0

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index % 20 == 0:
                print(f"[evaluation] batch {index + 1}/{len(self.loader)}", flush=True)
            img, aud, labels, ids = unpack_paired_batch(batch)
            take = min(len(labels), self.example_limit - len(self.examples))
            for position in range(max(0, take)):
                sample_id = (ids[position].detach().cpu().clone() if ids is not None
                             else torch.tensor(self.seen + position, dtype=torch.long))
                self.examples.append((img[position].detach().cpu().clone(),
                                      aud[position].detach().cpu().clone(),
                                      labels[position].detach().cpu().clone(), sample_id))
            self.seen += len(labels)
            yield batch


class CapturedExamples(torch.utils.data.Dataset):
    """The exact observations already evaluated, without re-running dataset transforms."""
    def __init__(self, examples, split):
        self.examples, self.split, self.train = examples, split, False

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def verified_export(output_dir, identity):
    manifest = output_dir / "tables/export_manifest.json"
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return None
    if not manifest.is_file():
        raise FileExistsError(f"Incomplete or unrelated export directory: {output_dir}; select a fresh output directory")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("identity") != identity or record.get("status") != "complete":
        raise ValueError(f"Export inputs changed: {output_dir}; use a new --output-name or --output-dir")
    artifacts = record.get("artifacts", {})
    if not artifacts:
        raise ValueError(f"Export artifact manifest is empty: {manifest}")
    for relative, expected in artifacts.items():
        path = (output_dir / relative).resolve()
        if not path.is_relative_to(output_dir) or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Export artifact missing or changed: {relative}")
    return record


def run_export(config_path, checkpoint_path, output_dir, *, split="test",
               mask_source="predicted", examples=8, max_samples=0, device=None):
    config_path, checkpoint_path = Path(config_path).resolve(), Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    if examples < 0 or max_samples < 0:
        raise ValueError("examples and max_samples must be non-negative")
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Both generated config and trained recovery checkpoint are required")
    config_hash = file_sha256(config_path)
    resolved_config = load_config(config_path)
    if file_sha256(config_path) != config_hash:
        raise RuntimeError("Configuration changed while loading")
    resolved_hash = hashlib.sha256(json.dumps(resolved_config, sort_keys=True).encode()).hexdigest()
    identity = dict(schema="v15pro-export.v1", config_sha256=config_hash,
                    resolved_config_sha256=resolved_hash, runtime_sha256=runtime_source_fingerprint(),
                    checkpoint_sha256=file_sha256(checkpoint_path), split=split,
                    mask_source=mask_source, examples=examples, max_samples=max_samples,
                    device_override=device, exporter_sha256=source_fingerprint())
    previous = verified_export(output_dir, identity)
    if previous is not None:
        print(f"[export] verified complete: {output_dir}", flush=True)
        return previous
    for name in ("figures", "logs", "tables"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs/evaluation.log"
    artifacts = []
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                print(f"[export] started {datetime.now(timezone.utc).isoformat()}")
                print(f"[export] config={config_path}\n[export] checkpoint={checkpoint_path}")
                cfg = deepcopy(resolved_config)
                if device is not None:
                    cfg["device"] = device
                selected_device = torch.device(cfg.get("device", "cpu"))
                if selected_device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
                if not cfg.get("noise_detector", {}).get("enabled", False):
                    raise ValueError("v15pro exports require a local-noise detector configuration")
                if (cfg.get("data", {}).get("dataset", "mnist_fsdd") == "mnist_fsdd"
                        and not cfg.get("data", {}).get("paper_split", {}).get("enabled", False)):
                    raise ValueError("v15pro exports require independent train/validation/test splits")
                set_seed(int(cfg.get("seed", 1234)))
                _, loader = build_loaders(cfg, eval_split=split)
                model = CrossModalSNN(cfg).to(selected_device)
                state = load_strict_checkpoint(model, checkpoint_path, cfg, selected_device)
                if file_sha256(checkpoint_path) != identity["checkpoint_sha256"]:
                    raise RuntimeError("Checkpoint changed while loading; retry with a stable completed checkpoint")
                example_limit = min(examples, max_samples) if max_samples else examples
                progress_loader = ProgressLoader(loader, example_limit=example_limit)
                report, rows = evaluate_local_noise(
                    model, progress_loader, cfg, selected_device, split=split,
                    mask_source=mask_source, max_samples=max_samples)
                report.update(checkpoint=str(checkpoint_path),
                              checkpoint_sha256=identity["checkpoint_sha256"],
                              checkpoint_epoch=state.get("epoch"), config=str(config_path),
                              noise_training=state.get("noise_training"),
                              smoke_training=(state.get("noise_training") or {}).get("smoke_only"),
                              partial_evaluation=bool(max_samples),
                              figures_note="Examples are the first dataset samples, not selected by recovery quality.")
                artifacts.extend(export_numeric_tables(report, rows, output_dir))
                artifacts.extend(plot_metric_summaries(report, output_dir))
                if progress_loader.examples:
                    example_loader = torch.utils.data.DataLoader(
                        CapturedExamples(progress_loader.examples, split),
                        batch_size=len(progress_loader.examples), shuffle=False)
                    artifacts.extend(export_examples(
                        model, example_loader, cfg, selected_device, output_dir, split=split,
                        mask_source=mask_source, num_examples=len(progress_loader.examples)))
                print(f"[export] complete: {report['n_unique_samples']} samples, "
                      f"{len(rows)} condition exposures, {output_dir}")
            except BaseException:
                traceback.print_exc()
                raise
    artifacts.append(log_path)
    hashes = {}
    for artifact in artifacts:
        artifact = Path(artifact).resolve()
        if not artifact.is_relative_to(output_dir):
            raise ValueError(f"Exporter wrote outside its destination: {artifact}")
        hashes[artifact.relative_to(output_dir).as_posix()] = file_sha256(artifact)
    record = dict(status="complete", identity=identity, artifacts=hashes,
                  n_unique_samples=report["n_unique_samples"], n_exposures=len(rows),
                  completed_utc=datetime.now(timezone.utc).isoformat())
    manifest = output_dir / "tables/export_manifest.json"
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Suite root, e.g. outputs/v15pro")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 2345, 3456])
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-name", help="Per-seed output folder; defaults to inference[_oracle|_all]")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--mask-source", choices=("predicted", "oracle", "all"), default="predicted")
    parser.add_argument("--examples", type=int, default=8, help="First N samples per condition; 0 disables sample panels")
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates all samples; positive values are partial checks")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.examples < 0 or args.max_samples < 0:
        parser.error("--examples and --max-samples must be non-negative")
    if args.root:
        if any((args.config, args.checkpoint, args.output_dir)):
            parser.error("--root cannot be combined with --config/--checkpoint/--output-dir")
        if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
            parser.error("--seeds must contain unique non-negative integers")
        name = args.output_name or ("inference" if args.mask_source == "predicted" else "inference_" + args.mask_source)
        if name in (".", "..") or Path(name).name != name or "/" in name or "\\" in name:
            parser.error("--output-name must be a single directory name")
        jobs = [(Path(args.root) / f"seed_{seed}" / "config.yaml",
                 Path(args.root) / f"seed_{seed}" / "recovery/best.pt",
                 Path(args.root) / f"seed_{seed}" / name) for seed in args.seeds]
    else:
        if not all((args.config, args.checkpoint, args.output_dir)):
            parser.error("Provide --root or all of --config, --checkpoint, --output-dir")
        if args.output_name:
            parser.error("--output-name applies only to --root mode")
        jobs = [(Path(args.config), Path(args.checkpoint), Path(args.output_dir))]
    missing = [str(path) for config, checkpoint, _ in jobs for path in (config, checkpoint) if not path.is_file()]
    if missing:
        parser.error("Training must finish before export; missing: " + ", ".join(missing))
    for config, checkpoint, output in jobs:
        run_export(config, checkpoint, output, split=args.split, mask_source=args.mask_source,
                   examples=args.examples, max_samples=args.max_samples, device=args.device)


if __name__ == "__main__":
    main()
