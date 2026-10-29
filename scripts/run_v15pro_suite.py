"""Plan or run isolated local-contamination detector/recovery experiments.

Without --run this prints commands and does not create output directories.
Re-running an identical plan resumes incomplete jobs and verifies completed
checkpoint hashes. Changed code/configuration requires a new output directory.
"""

import bootstrap  # noqa: F401

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

from common import load_config
from models.frozen_base import file_sha256
from paths import PROJECT_ROOT
from scripts.job_runner import run_job


MASK_SOURCES = ("predicted", "oracle", "all")


def source_fingerprint():
    digest = hashlib.sha256()
    files = [PROJECT_ROOT / "common.py", PROJECT_ROOT / "paths.py"]
    for directory in ("models", "data", "scripts"):
        files.extend((PROJECT_ROOT / directory).glob("*.py"))
    for filename in sorted(files):
        digest.update(filename.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(filename.read_bytes())
    return digest.hexdigest()


def payload_fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_plan(base, root, seeds=(1234, 2345, 3456), device=None,
               detector_epochs=None, epochs=None, data_root=None, fsdd_root=None,
               base_checkpoint=None):
    """Build serializable configs/jobs without touching output paths."""
    root = Path(root).resolve()
    configurations, jobs = {}, []
    for seed in seeds:
        cfg = deepcopy(base)
        cfg["paper_template"] = False
        cfg["version"] = "v15pro"
        cfg["seed"] = seed
        if device is not None:
            cfg["device"] = device
        cfg["train"]["require_cuda"] = str(cfg["device"]).startswith("cuda")
        if data_root is not None:
            cfg["data"]["root"] = str(data_root)
        if fsdd_root is not None:
            cfg["audio"]["fsdd_root"] = str(fsdd_root)
        seed_dir = root / f"seed_{seed}"
        detector_dir, recovery_dir = seed_dir / "detector", seed_dir / "recovery"
        cfg["audio"]["norm_stats_path"] = str(root / "train_audio_norm.pt")
        cfg["noise_detector"].update(
            checkpoint=str(detector_dir / "best.pt"),
            last_checkpoint=str(detector_dir / "last.pt"))
        if detector_epochs is not None:
            cfg["noise_detector"]["pretrain_epochs"] = detector_epochs
        if epochs is not None:
            cfg["train"]["epochs"] = epochs
        cfg["train"].update(
            ckpt_path=str(recovery_dir / "last.pt"),
            best_ckpt_path=str(recovery_dir / "best.pt"),
            eval_ckpt_path=str(recovery_dir / "best.pt"),
            output_version=f"v15pro_seed_{seed}")
        if base_checkpoint is not None:
            cfg["train"].update(
                init_ckpt_path=str(base_checkpoint), init_required=True,
                parent_sha256=file_sha256(base_checkpoint), init_strict=False,
                init_allowed_missing_prefixes=["img_noise_detector.", "aud_noise_detector."],
                init_allowed_unexpected_prefixes=[], init_load_optimizer=False)
        config_path = str(seed_dir / "config.yaml")
        configurations[config_path] = cfg
        for stage, script, directory in (
                ("detector", "train_noise_detector.py", detector_dir),
                ("recovery", "train.py", recovery_dir)):
            dependencies = []
            if stage == "recovery":
                dependencies.append(str(detector_dir / "best.pt"))
                if cfg["train"].get("init_ckpt_path"):
                    dependencies.append(cfg["train"]["init_ckpt_path"])
            jobs.append(dict(
                id=f"{seed}_{stage}", kind=stage, config=config_path,
                command=[f"scripts/{script}", "--config", config_path],
                outputs=[str(directory / "best.pt"), str(directory / "last.pt")],
                last=str(directory / "last.pt"), dependencies=dependencies))
        for source in MASK_SOURCES:
            output = str(seed_dir / "evaluation" / f"{source}.json")
            jobs.append(dict(
                id=f"{seed}_evaluation_{source}", kind="evaluation", config=config_path,
                command=["scripts/evaluate_v15pro.py", "--config", config_path,
                         "--checkpoint", str(recovery_dir / "best.pt"), "--split", "test",
                         "--mask-source", source, "--output", output],
                outputs=[output, str(Path(output).with_suffix(".csv"))],
                dependencies=[str(recovery_dir / "best.pt")]))
    return configurations, jobs


def write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def job_fingerprint(job, plan_hash):
    dependencies = {}
    for value in job.get("dependencies", []):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing dependency for {job['id']}: {path}")
        dependencies[str(path)] = file_sha256(path)
    return payload_fingerprint(dict(plan=plan_hash, job=job, dependencies=dependencies))


def verified_complete(marker, job, fingerprint):
    if not marker.is_file():
        return False
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if saved.get("fingerprint") != fingerprint:
        raise RuntimeError(f"Completed task inputs changed; use a new --output: {marker}")
    expected_outputs = saved.get("outputs", {})
    if set(expected_outputs) != set(job["outputs"]):
        raise RuntimeError(f"Completed task artifact list changed: {marker}")
    for name, expected_hash in expected_outputs.items():
        artifact = Path(name)
        if not artifact.is_file() or file_sha256(artifact) != expected_hash:
            raise RuntimeError(f"Completed task artifact missing or changed: {artifact}")
    return True


def execute_plan(root, configurations, jobs, skip_eval=False):
    description = dict(version="v15pro", source_sha256=source_fingerprint(),
                       configs=configurations, jobs=jobs)
    fingerprint = payload_fingerprint(description)
    manifest = root / "plan.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved != dict(fingerprint=fingerprint, plan=description):
            raise RuntimeError("Plan/code changed; select a new --output directory")
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError("Nonempty output directory has no plan.json; choose a new --output")
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest, dict(fingerprint=fingerprint, plan=description))
    for name, cfg in configurations.items():
        path = Path(name)
        if path.is_file() and yaml.safe_load(path.read_text(encoding="utf-8")) != cfg:
            raise RuntimeError(f"Generated configuration was edited: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    for job in jobs:
        if skip_eval and job["kind"] == "evaluation":
            continue
        marker = root / f"{job['id']}.done.json"
        task_hash = job_fingerprint(job, fingerprint)
        if verified_complete(marker, job, task_hash):
            print(f"[v15pro] verified complete: {job['id']}", flush=True)
            continue
        command = list(job["command"])
        if job.get("last") and Path(job["last"]).is_file():
            command.append("--resume")
        run_job(command, root / "logs" / f"{job['id']}.log")
        missing = [name for name in job["outputs"] if not Path(name).is_file()]
        if missing:
            raise RuntimeError(f"Task exited without expected artifacts: {missing}")
        write_json_atomic(marker, dict(
            fingerprint=task_hash,
            outputs={name: file_sha256(name) for name in job["outputs"]}))
    print("[v15pro] requested jobs completed; inspect metrics before claiming improvement.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v15pro.yaml")
    parser.add_argument("--output", default="outputs/v15pro")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456])
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--fsdd-root", default=None)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--detector-epochs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**31 for s in args.seeds):
        parser.error("Seeds must be unique integers in [0, 2**31)")
    if any(v is not None and v < 1 for v in (args.detector_epochs, args.epochs)):
        parser.error("Epoch budgets must be positive")
    if args.device is not None and not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", args.device):
        parser.error("--device must be cpu, cuda, or cuda:N")
    for name in ("data_root", "fsdd_root", "base_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            if not Path(value).is_absolute():
                parser.error(f"--{name.replace('_', '-')} requires an absolute path")
            setattr(args, name, str(Path(value).resolve()))
    if args.base_checkpoint is not None and not Path(args.base_checkpoint).is_file():
        parser.error("--base-checkpoint must identify an existing checkpoint")
    root = Path(args.output).resolve()
    base = load_config(args.config)
    if not base.get("noise_detector", {}).get("enabled"):
        parser.error("v15pro requires noise_detector.enabled=true")
    if not base.get("data", {}).get("paper_split", {}).get("enabled"):
        parser.error("v15pro requires independent train/validation/test splits")
    configurations, jobs = build_plan(
        base, root, args.seeds, args.device, args.detector_epochs, args.epochs,
        args.data_root, args.fsdd_root, args.base_checkpoint)
    print(f"v15pro: {len(args.seeds)} seed(s); detector warmup then recovery; "
          "predicted-mask primary evaluation plus oracle/all diagnostics.")
    if not args.run:
        for job in jobs:
            if args.skip_eval and job["kind"] == "evaluation":
                continue
            print(job["id"], subprocess.list2cmdline([sys.executable, "-u", *job["command"]]))
        print("Dry run only: no output directory created. Add --run to execute.")
        return
    execute_plan(root, configurations, jobs, skip_eval=args.skip_eval)


if __name__ == "__main__":
    main()
