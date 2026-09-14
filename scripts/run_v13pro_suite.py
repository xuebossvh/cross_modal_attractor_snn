"""v13pro: reproducible publication experiments, independent of old checkpoints."""

import bootstrap  # noqa: F401
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

from common import load_config
from models.frozen_base import file_sha256
from paths import PROJECT_ROOT
from scripts.job_runner import run_job


RECOVERY = ("control", "main", "no_causal", "no_cross")
BASELINES = ("recognizer", "classifier", "cue_cnn", "conditioned_cnn", "matched_cnn")


def matched_ann_width(cfg):
    """Choose ANN width once from architecture size, before seeing validation/test data."""
    from models.network import CrossModalSNN
    from models.paper_baselines import RecoveryCNN
    target = sum(p.numel() for p in CrossModalSNN(cfg).parameters())
    candidates = tuple(range(16, 257, 8))
    scored = []
    for width in candidates:
        count = sum(p.numel() for p in RecoveryCNN(width=width).parameters())
        scored.append((abs(count - target), width, count))
    _, width, count = min(scored)
    return width, target, count


def code_fingerprint():
    digest = hashlib.sha256()
    paths = [PROJECT_ROOT / "common.py", PROJECT_ROOT / "paths.py"]
    for folder in ("data", "models", "scripts"):
        paths.extend(sorted((PROJECT_ROOT / folder).glob("*.py")))
    for path in sorted(paths):
        if path.name.startswith(("run_v12", "smoke_test_v12")):
            continue
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_plan(base, root, seeds=(1234, 2345, 3456), parent_epochs=100,
               epochs=30, baseline_epochs=30, matched_ann_epochs=130, mechanism=False,
               speakers=None, holdout=None):
    root = Path(root).resolve()
    configs, training, testing = {}, [], []
    ann_width, snn_params, ann_params = matched_ann_width(base)
    for seed in seeds:
        names = ["parent", *RECOVERY, *BASELINES]
        if mechanism:
            names += ["parent_no_recurrence", "no_recurrence", "parent_no_kwta", "no_kwta"]
        for name in names:
            cfg = deepcopy(base)
            cfg.pop("paper_template", None)
            directory = root / f"seed_{seed}" / name
            cfg["seed"] = seed
            cfg["data"]["paper_split"] = dict(enabled=True, mode="official", seed=20260915, val_fraction=.1)
            if speakers:
                cfg["data"]["paper_split"].update(mode="speaker", test_speakers=speakers[0], val_speakers=speakers[1])
            cfg["data"]["pairing"].update(enabled=False, return_pair_id=False, sample_targets_for_missing=False)
            cfg["audio"]["norm_stats_path"] = str(root / "train_audio_norm.pt")
            cfg["paper"] = dict(experiment=name, seed=seed, root=str(root),
                                primary_endpoint="audio partial-cue masked MSE", init="fresh_lineage",
                                ann_width=ann_width, matched_ann_snn_parameters=snn_params,
                                matched_ann_parameters=ann_params)
            cfg["validation"].update(enabled=True, split="val", max_batches=0, every_epochs=1)
            cfg["validation"]["score"] = dict(lambda_img=1., lambda_aud=4., lambda_pair=0., lambda_cls=.5)
            cfg["train"].update(epochs=epochs, start_epoch=0, init_required=True, init_load_optimizer=False,
                                evaluation_only=False, freeze_base=False, parent_sha256="",
                                ckpt_path=str(directory / "last.pt"), best_ckpt_path=str(directory / "best.pt"),
                                eval_ckpt_path=str(directory / "best.pt"), output_version=f"paper_{seed}_{name}",
                                epoch_seeded=True)
            cfg["decoder_pretrain"]["enabled"] = False
            parent_name = "parent"
            if "no_recurrence" in name:
                cfg["ablation"]["use_recurrent"] = False
                parent_name = "parent_no_recurrence"
            if "no_kwta" in name:
                cfg["ablation"]["use_kwta"] = False
                parent_name = "parent_no_kwta"
            cfg["train"]["init_ckpt_path"] = str(root / f"seed_{seed}" / parent_name / "best.pt")
            if name.startswith("parent") or name == "control":
                cfg["audio_local_cue"]["multiscale"] = False
                cfg["cross_key_conditioning"]["intermediate"] = False
            if name.startswith("parent"):
                cfg["train"].update(epochs=parent_epochs, lr=.0002, init_ckpt_path="", init_required=False,
                                    trainable_prefixes=[], init_allowed_missing_prefixes=[])
            if name in ("no_causal", "no_cross"):
                cfg["cross_key_conditioning"]["causal_training"].update(enabled=False, loss_weight=0.)
            if name == "no_cross":
                cfg["cross_key_conditioning"]["enabled"] = False
            if name in BASELINES:
                cfg["train"].update(epochs=baseline_epochs, lr=.001, init_ckpt_path="", init_required=False)
            if name == "matched_cnn":
                cfg["train"]["epochs"] = matched_ann_epochs
            if holdout:
                for key in ("aud_train_modes",):
                    cfg["corruption"][key] = [x for x in cfg["corruption"][key] if x != holdout]
                cfg["paper"]["heldout_audio_family"] = holdout
            path = directory / "config.yaml"
            configs[str(path)] = cfg
            command = (["scripts/paper_baseline.py", "--kind", name] if name in BASELINES
                       else ["scripts/train.py"])
            training.append(dict(id=f"train_{seed}_{name}", command=command + ["--config", str(path)],
                                 config=str(path), output=str(directory / "best.pt"),
                                 last=str(directory / "last.pt"), epochs=cfg["train"]["epochs"]))
            if not name.startswith("parent") and name != "recognizer":
                command = ["scripts/paper_evaluate.py", "--config", str(path),
                           "--recognizer", str(root / f"seed_{seed}" / "recognizer" / "best.pt")]
                testing.append(dict(id=f"eval_{seed}_{name}", command=command,
                                    config=str(path), output=str(directory / "evaluation" / "complete.json")))
    return configs, training, testing


def task_fingerprint(job, cfg, source_sha):
    dependencies = []
    parent = cfg["train"].get("init_ckpt_path")
    if parent and Path(parent).is_file():
        dependencies.append((parent, file_sha256(parent)))
    if job["id"].startswith("eval"):
        for path in (cfg["train"]["eval_ckpt_path"], job["command"][job["command"].index("--recognizer") + 1]):
            if Path(path).is_file():
                dependencies.append((path, file_sha256(path)))
    payload = dict(job=job, cfg=cfg, code=source_sha, dependencies=dependencies)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def completed(marker, job, fingerprint):
    if not marker.is_file() or not Path(job["output"]).is_file():
        return False
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if saved["fingerprint"] != fingerprint:
        raise RuntimeError(f"Completed task changed; choose a new --output instead of overwriting: {marker}")
    if saved["output_sha256"] != file_sha256(job["output"]):
        raise RuntimeError(f"Task artifact changed: {job['output']}")
    if job.get("id", "").startswith("eval"):
        metadata = json.loads(Path(job["output"]).read_text(encoding="utf-8"))
        for name, expected in metadata["artifacts"].items():
            artifact = Path(job["output"]).parent / name
            if not artifact.is_file() or file_sha256(artifact) != expected:
                raise RuntimeError(f"Evaluation artifact missing/changed: {artifact}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v13pro.yaml")
    ap.add_argument("--output", default="outputs/v13pro")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 2345, 3456])
    ap.add_argument("--parent_epochs", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--baseline_epochs", type=int, default=30)
    ap.add_argument("--matched_ann_epochs", type=int, default=130)
    ap.add_argument("--mechanism_ablations", action="store_true")
    ap.add_argument("--speaker_test", nargs="+")
    ap.add_argument("--speaker_val", nargs="+")
    ap.add_argument("--holdout_audio_family", choices=("time_mask", "freq_mask", "feature_dropout", "partial_temporal", "time_freq_block"))
    ap.add_argument("--severities", nargs="+", type=float, default=[.4])
    ap.add_argument("--mask_seeds", nargs="+", type=int, default=[5678])
    ap.add_argument("--all_family_pairs", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if bool(args.speaker_test) != bool(args.speaker_val):
        ap.error("Specify both --speaker_test and --speaker_val")
    if set(args.speaker_test or []) & set(args.speaker_val or []):
        ap.error("Validation and test speakers must be disjoint")
    if any(not 0 <= seed < 2**31 for seed in args.seeds):
        ap.error("Training seeds must be within [0,2**31)")
    if len(set(args.seeds)) != len(args.seeds) or min(args.parent_epochs, args.epochs,
                                                       args.baseline_epochs, args.matched_ann_epochs) < 1:
        ap.error("Unique seeds and positive epoch budgets are required")
    root = Path(args.output).resolve()
    base = load_config(args.config)
    if not base["audio"].get("use_real_audio") or not base["data"].get("use_mnist"):
        ap.error("The formal suite requires real MNIST and FSDD")
    configs, training, testing = build_plan(base, root, args.seeds,
        args.parent_epochs, args.epochs, args.baseline_epochs, args.matched_ann_epochs,
        args.mechanism_ablations,
        (args.speaker_test, args.speaker_val) if args.speaker_test else None, args.holdout_audio_family)
    for job in testing:
        job["command"] += ["--severities", *map(str, args.severities), "--mask_seeds", *map(str, args.mask_seeds)]
        if args.all_family_pairs:
            job["command"].append("--all_family_pairs")
    jobs = ([] if args.eval_only else training) + testing
    print(f"Training jobs={len(training)}, summed model-epochs={sum(j['epochs'] for j in training)}, evaluation jobs={len(testing)}")
    print("No existing legacy parent is used. Epoch budget is not a convergence claim.")
    if not args.run or args.dry_run:
        for job in jobs:
            print(job["id"], subprocess.list2cmdline([sys.executable, "-u", *job["command"]]))
        return
    root.mkdir(parents=True, exist_ok=True)
    sha = code_fingerprint()
    manifest = root / "plan.json"
    description = dict(code_sha256=sha, configs=configs, training=training, testing=testing)
    if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")) != description:
        raise RuntimeError("Plan/code changed; use a different --output directory")
    manifest.write_text(json.dumps(description, indent=2), encoding="utf-8")
    for path, cfg in configs.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    for job in jobs:
        marker = root / f"{job['id']}.done.json"
        fingerprint = task_fingerprint(job, configs[job["config"]], sha)
        if completed(marker, job, fingerprint):
            print(f"[paper] verified complete: {job['id']}", flush=True)
            continue
        command = list(job["command"])
        if job.get("last") and Path(job["last"]).is_file():
            command.append("--resume")
        run_job(command, root / "logs" / f"{job['id']}.log")
        if not Path(job["output"]).is_file():
            raise RuntimeError(f"Task exited without expected artifact: {job}")
        marker.write_text(json.dumps(dict(fingerprint=fingerprint,
            output_sha256=file_sha256(job["output"])), indent=2), encoding="utf-8")
    run_job(["scripts/paper_statistics.py", "--root", str(root)], root / "logs" / "statistics.log")
    run_job(["scripts/paper_profile.py", "--root", str(root)], root / "logs" / "complexity_profile.log")
    print("[paper] ALL REQUESTED TASKS COMPLETED; paper claims still require reviewing the results.", flush=True)


if __name__ == "__main__":
    main()
