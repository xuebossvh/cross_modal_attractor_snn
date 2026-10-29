"""Export v16pro figures/logs/tables after all training jobs have completed.

Example: python tools/export_v16pro.py --root outputs/v16pro --device cuda
Per-sample rows are streamed to gzip CSV to bound memory and disk use. The three
model branches receive the same base evaluation; strength scans run on main.
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import re
import traceback

import torch

from common import load_config, set_seed, unpack_paired_batch
from data.dataset import build_loaders
from data.v16_protocol import protocol_contract
from models.frozen_base import file_sha256
from models.network import CrossModalSNN
from models.paper_baselines import CleanRecognizers
from scripts.evaluate_v15pro import verify_checkpoint_config
from scripts.v16_evaluation import evaluate
from scripts.noise_metrics import preserve_random_state
from tools.export_v15pro import Tee, verified_export, runtime_source_fingerprint
from tools.v16_report import (aggregate_reports, numeric_tables, plot_conditions,
                              save_examples, write_json)
from paths import resolve_from_root


MODELS = ("main", "no_value", "no_recurrence")


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_fingerprint():
    digest = hashlib.sha256()
    for name in ("tools/export_v16pro.py", "tools/v16_report.py", "tools/export_v15pro.py",
                 "scripts/v16_evaluation.py", "scripts/evaluate_v15pro.py", "data/v16_protocol.py"):
        digest.update(name.encode())
        digest.update((PROJECT_ROOT / name).read_bytes())
    return digest.hexdigest()


def strict_checkpoint(model, path, cfg, device, stage, allow_smoke=False):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("schema") != "v16pro.training.v1" or "model" not in state:
        raise ValueError("Expected a v16pro training checkpoint")
    saved = state.get("cfg", {})
    if saved.get("train", {}).get("stage") != stage or cfg.get("train", {}).get("stage") != stage:
        raise ValueError("Model stage/checkpoint/path mismatch")
    verify_checkpoint_config(cfg, saved)
    if cfg.get("seed") != saved.get("seed") or cfg.get("v16") != saved.get("v16"):
        raise ValueError("Checkpoint seed or v16 semantic configuration mismatch")
    if saved.get("_audio_norm_stats") != cfg.get("_audio_norm_stats"):
        raise ValueError("Checkpoint normalization statistics mismatch")
    from scripts.train_v16pro import _runtime_sha256
    if state.get("metadata", {}).get("runtime_sha256") != _runtime_sha256():
        raise ValueError("Training runtime source fingerprint changed; inference must use the trained implementation")
    if state.get("metadata", {}).get("smoke_only") and not allow_smoke:
        raise ValueError("Smoke-only checkpoint cannot produce formal results; use --allow-smoke for diagnostics")
    model.load_state_dict(state["model"], strict=True)
    return state


def load_recognizer(path, cfg, device):
    state = torch.load(path, map_location=device, weights_only=True)
    saved = state.get("cfg", {})
    if state.get("kind") != "recognizer" or saved.get("seed") != cfg.get("seed"):
        raise ValueError("Independent clean recognizer checkpoint kind/seed mismatch")
    for section in ("audio", "_audio_norm_stats"):
        if cfg.get(section) != saved.get(section):
            raise ValueError("Recognizer data normalization/configuration mismatch: " + section)
    left, right = deepcopy(cfg.get("data", {})), deepcopy(saved.get("data", {}))
    for key in ("batch_size", "num_workers", "eval_split", "train_samples_per_epoch"):
        left.pop(key, None)
        right.pop(key, None)
    if left != right:
        raise ValueError("Recognizer data split/configuration mismatch")
    recognizer = CleanRecognizers().to(device)
    recognizer.load_state_dict(state["model"], strict=True)
    recognizer.eval()
    return recognizer, state


@torch.no_grad()
def clean_recognizer_reference(recognizer, loader, device, *, max_samples=0, seed=20261030):
    """Clean held-out reference for interpreting independent content recognition."""
    count, correct_img, correct_aud = 0, 0, 0
    with preserve_random_state():
        set_seed(seed)
        for batch in loader:
            img, aud, labels, _ = unpack_paired_batch(batch)
            take = min(len(labels), max_samples - count) if max_samples else len(labels)
            if take <= 0:
                break
            labels = labels[:take].to(device)
            correct_img += int(recognizer.img(img[:take].to(device)).argmax(1).eq(labels).sum())
            correct_aud += int(recognizer.aud(aud[:take].to(device)).argmax(1).eq(labels).sum())
            count += take
            if max_samples and count >= max_samples:
                break
    if not count:
        raise ValueError("Empty clean recognizer reference split")
    return dict(n=count, img_acc=correct_img / count, aud_acc=correct_aud / count,
                seed=seed, input="original clean test observations; separate reference pass",
                purpose="Diagnostic reference for the content evaluator, not a recovery baseline")


def _complete(output, identity, artifacts, **metadata):
    output = Path(output).resolve()
    hashes = {}
    for path in artifacts:
        path = Path(path).resolve()
        if not path.is_relative_to(output):
            raise ValueError("Artifact outside export destination")
        hashes[path.relative_to(output).as_posix()] = file_sha256(path)
    record = dict(status="complete", identity=identity, artifacts=hashes,
                  completed_utc=datetime.now(timezone.utc).isoformat(), **metadata)
    write_json(output / "tables/export_manifest.json", record)
    return record


def run_export(config_path, checkpoint_path, recognizer_path, output_dir, *, stage="main",
               device="cuda", examples=8, max_samples=0, mask_source="predicted", allow_smoke=False,
               scans=None):
    config_path, checkpoint_path, recognizer_path, output = map(lambda p: Path(p).resolve(),
                                                               (config_path, checkpoint_path, recognizer_path, output_dir))
    if stage not in MODELS or min(examples, max_samples) < 0:
        raise ValueError("Invalid stage or negative export limit")
    include_scans = (stage == "main") if scans is None else bool(scans)
    if not all(path.is_file() for path in (config_path, checkpoint_path, recognizer_path)):
        raise FileNotFoundError("Completed recovery and recognizer checkpoints/config are required")
    cfg_hash = file_sha256(config_path)
    resolved = load_config(config_path)
    if cfg_hash != file_sha256(config_path):
        raise RuntimeError("Configuration changed during loading")
    identity = dict(schema="v16pro-export.v1", config_sha256=cfg_hash,
                    resolved_config_sha256=_json_sha(resolved), checkpoint_sha256=file_sha256(checkpoint_path),
                    recognizer_sha256=file_sha256(recognizer_path), stage=stage, device=device,
                    examples=examples, max_samples=max_samples, mask_source=mask_source, allow_smoke=allow_smoke,
                    scans=include_scans,
                    exporter_sha256=source_fingerprint(), runtime_sha256=runtime_source_fingerprint())
    stats_path = resolve_from_root(resolved.get("audio", {}).get("norm_stats_path", "_data/audio_norm_stats.pt"))
    identity["normalization_file_sha256"] = file_sha256(stats_path) if stats_path.is_file() else None
    previous = verified_export(output, identity)
    if previous is not None:
        print("[export] verified complete:", output, flush=True)
        return previous
    for name in ("figures", "logs", "tables"):
        (output / name).mkdir(parents=True, exist_ok=True)
    artifacts = []
    log_path = output / "logs/evaluation.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                cfg = deepcopy(resolved)
                cfg["device"] = device
                selected = torch.device(device)
                if selected.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable; use --device cpu for an explicit CPU run")
                if not cfg.get("v16", {}).get("enabled", False):
                    raise ValueError("v16 protocol must be enabled")
                if any(cfg.get(name, {}).get("enabled", False) for name in
                       ("cross_key_conditioning", "cross_detail_conditioning", "pair_alignment")):
                    raise ValueError("v16 export excludes Cross-Key, cross-detail and pair alignment")
                print(f"[export] {stage}; checkpoint={checkpoint_path}", flush=True)
                set_seed(int(cfg["seed"]))
                _, loader = build_loaders(cfg, eval_split="test")
                identity["normalization_file_sha256"] = file_sha256(stats_path) if stats_path.is_file() else None
                model = CrossModalSNN(cfg).to(selected)
                state = strict_checkpoint(model, checkpoint_path, cfg, selected, stage, allow_smoke)
                recognizer, recognizer_state = load_recognizer(recognizer_path, cfg, selected)
                if file_sha256(checkpoint_path) != identity["checkpoint_sha256"] or file_sha256(recognizer_path) != identity["recognizer_sha256"]:
                    raise RuntimeError("A checkpoint changed during loading")
                row_path = output / "tables/per_sample.csv.gz"
                with gzip.open(row_path, "wt", encoding="utf-8", newline="", compresslevel=5) as stream:
                    writer = None
                    def sink(row):
                        nonlocal writer
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                    report, captured = evaluate(model, loader, cfg, selected, split="test", recognizer=recognizer,
                                                mask_source=mask_source, max_samples=max_samples,
                                                scans=include_scans, examples=examples, row_sink=sink)
                artifacts.append(row_path)
                report.update(seed=cfg["seed"], stage=stage, checkpoint_epoch=state.get("epoch"),
                              checkpoint_sha256=identity["checkpoint_sha256"],
                              recognizer_sha256=identity["recognizer_sha256"],
                              recognizer_epoch=recognizer_state.get("epoch"),
                              smoke_only=bool(state.get("metadata", {}).get("smoke_only")),
                              cue_contract=protocol_contract(cfg),
                              model_parameters=sum(p.numel() for p in model.parameters()),
                              parent_sha256=state.get("metadata", {}).get("parent_sha256"),
                              detector_sha256=state.get("metadata", {}).get("detector_sha256"))
                report["clean_recognizer_reference"] = clean_recognizer_reference(
                    recognizer, loader, selected, max_samples=max_samples,
                    seed=int(cfg.get("v16", {}).get("evaluation_seed", 20261030)))
                artifacts.extend(numeric_tables(report, output))
                artifacts.extend(plot_conditions(report, output))
                artifacts.extend(save_examples(captured, output))
                print(f"[export] complete: {report['n_unique_samples']} unique samples, "
                      f"{len(report['conditions'])} conditions", flush=True)
            except BaseException:
                traceback.print_exc()
                raise
    artifacts.append(log_path)
    return _complete(output, identity, artifacts, n_unique_samples=report["n_unique_samples"],
                     n_conditions=len(report["conditions"]), smoke_only=report["smoke_only"])


def export_suite(root, seeds=(1234, 2345, 3456), *, device="cuda", examples=8, max_samples=0,
                 mask_source="predicted", allow_smoke=False, output_name=None,
                 models=MODELS, scans=None):
    root = Path(root).resolve()
    if len(set(seeds)) != len(seeds) or not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique non-negative integers")
    if not models or len(set(models)) != len(models) or any(model not in MODELS for model in models):
        raise ValueError("Models must be a nonempty unique subset of main/no_value/no_recurrence")
    if output_name is not None and (Path(output_name).name != output_name or output_name in (".", "..") or "/" in output_name or "\\" in output_name):
        raise ValueError("output-name must be one safe directory name")
    suffix = "" if mask_source == "predicted" else "_" + mask_source
    child_name, report_name = (output_name, "report_" + output_name) if output_name else ("inference" + suffix, "report" + suffix)
    jobs = [(seed, stage, root / f"seed_{seed}" / stage,
             root / f"seed_{seed}" / "recognizer/best.pt") for seed in seeds for stage in models]
    missing = [str(path) for _, _, base, recognizer in jobs
               for path in (base / "config.yaml", base / "best.pt", recognizer) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Training must finish before export: " + ", ".join(missing))
    reports, manifests = [], {}
    for seed, stage, base, recognizer in jobs:
        cfg = load_config(base / "config.yaml")
        if cfg.get("seed") != seed:
            raise ValueError("Suite directory seed disagrees with configuration")
        run_export(base / "config.yaml", base / "best.pt", recognizer, base / child_name,
                   stage=stage, device=device, examples=examples, max_samples=max_samples,
                   mask_source=mask_source, allow_smoke=allow_smoke, scans=scans)
        manifest = base / child_name / "tables/export_manifest.json"
        manifests[manifest.relative_to(root).as_posix()] = file_sha256(manifest)
        reports.append((seed, stage, json.loads((base / child_name / "tables/summary.json").read_text(encoding="utf-8"))))
    contracts = {_json_sha(report["cue_contract"]) for _, _, report in reports}
    counts = {report["n_unique_samples"] for _, _, report in reports}
    if len(contracts) != 1 or len(counts) != 1:
        raise ValueError("Cannot compare models with different cue contracts or evaluation sample counts")
    output = root / report_name
    identity = dict(schema="v16pro-aggregate.v1", child_manifests=manifests, exporter_sha256=source_fingerprint())
    previous = verified_export(output, identity)
    if previous is not None:
        print("[aggregate] verified complete:", output)
        return previous
    for name in ("figures", "logs", "tables"):
        (output / name).mkdir(parents=True, exist_ok=True)
    artifacts = aggregate_reports(reports, output)
    log_path = output / "logs/aggregation.log"
    log_path.write_text(f"Compared {len(seeds)} seeds and {len(models)} models.\n"
                        "Mean and sample SD across seeds; null SD for one seed.\n"
                        "Paired differences reuse seed identity; no significance claims.\n", encoding="utf-8")
    artifacts.append(log_path)
    return _complete(output, identity, artifacts,
                     partial_evaluation=any(report.get("partial_evaluation", False) for _, _, report in reports),
                     smoke_only=any(report.get("smoke_only", False) for _, _, report in reports))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v16pro")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 2345, 3456])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="0 is the full test set; positive limits are marked partial")
    parser.add_argument("--mask-source", choices=("predicted", "oracle", "all"), default="predicted")
    parser.add_argument("--allow-smoke", action="store_true", help="Explicitly allow diagnostic smoke checkpoints")
    parser.add_argument("--output-name", help="Fresh child directory for changed settings; never overwrite existing results")
    parser.add_argument("--skip-scans", action="store_true", help="Skip strength scans, e.g. for main-only oracle/all diagnostics")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", args.device):
        parser.error("--device must be cpu, cuda, or cuda:N")
    export_suite(args.root, args.seeds, device=args.device, examples=args.examples,
                 max_samples=args.max_samples, mask_source=args.mask_source,
                 allow_smoke=args.allow_smoke, output_name=args.output_name,
                 models=args.models, scans=False if args.skip_scans else None)


if __name__ == "__main__":
    main()
