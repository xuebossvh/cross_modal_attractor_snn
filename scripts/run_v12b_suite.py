"""Sequential train/evaluate/demo suite; no background process is created here."""

import bootstrap  # noqa: F401

import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import load_config
from models.frozen_base import file_sha256
from paths import PROJECT_ROOT, ensure_output_dirs, logs_dir, resolve_from_root


def select_configs(with_ablations=False, start_from="main"):
    names = ["main", "control"]
    if with_ablations:
        names.append("no_causal")
    if start_from not in names:
        raise ValueError("--start_from no_causal requires --with_ablations")
    selected = names[names.index(start_from):]
    return [PROJECT_ROOT / "configs" / (
        "v12b.yaml" if name == "main" else f"v12b_{name}.yaml")
        for name in selected]


def build_jobs(configs, eval_only=False, resume=False, severity=0.4,
               max_batches=None):
    jobs = []
    for path in configs:
        cfg = load_config(path)
        cfg["_config_path"] = str(path)
        base = ["--config", str(path)]
        directory = logs_dir(cfg)
        if not eval_only and not cfg["train"].get("evaluation_only", False):
            args = base + (["--resume"] if resume else [])
            jobs.append((["scripts/train.py"] + args, directory / "train.log"))
        extra = ["--severity", str(severity)]
        if max_batches is not None:
            extra += ["--max_batches", str(max_batches)]
        for protocol in ("fixed_mask", "legacy_random"):
            for intervention in ("normal", "sweep"):
                args = base + extra + ["--protocol", protocol, "--cross_key", intervention]
                if protocol == "fixed_mask" and intervention == "normal":
                    args += ["--family_breakdown"]
                jobs.append((["scripts/evaluate.py"] + args,
                             directory / f"eval_{protocol}_{intervention}.log"))
            jobs.append((["scripts/demo_inference.py"] + base + [
                "--protocol", protocol, "--severity", str(severity), "--num", "10"],
                directory / f"demo_{protocol}.log"))
    return jobs


def run_job(command, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print(f"[suite] start: {' '.join(command)}\n[suite] log: {logfile}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with logfile.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([sys.executable, "-u"] + command,
                                   cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)
    print(f"[suite] done: {command[0]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_ablations", action="store_true",
                    help="Also train/evaluate v12b_no_causal from the same parent")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--start_from", choices=("main", "control", "no_causal"),
                    default="main", help="Start at this experiment; skip earlier experiments")
    ap.add_argument("--resume", action="store_true", help="Require existing training checkpoints")
    ap.add_argument("--severity", type=float, default=0.4)
    ap.add_argument("--max_batches", type=int, default=None,
                    help="Evaluation smoke limit only; never shortens training")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    try:
        configs = select_configs(args.with_ablations, args.start_from)
    except ValueError as exc:
        ap.error(str(exc))
    jobs = build_jobs(configs, args.eval_only, args.resume, args.severity, args.max_batches)
    if args.dry_run:
        for command, logfile in jobs:
            print(f"{subprocess.list2cmdline([sys.executable, '-u'] + command)} -> {logfile}")
        return
    for path in configs:
        cfg = load_config(path)
        parent = resolve_from_root(cfg["train"]["init_ckpt_path"])
        if cfg["train"].get("init_required", False) and not parent.is_file():
            raise FileNotFoundError(f"Required parent checkpoint: {parent}")
        expected_sha = cfg["train"].get("parent_sha256", "")
        if expected_sha and (not parent.is_file() or file_sha256(parent) != expected_sha):
            raise RuntimeError(f"Missing or wrong parent checkpoint: {parent}")
        if (args.resume or args.eval_only) and not cfg["train"].get("evaluation_only"):
            checkpoint = resolve_from_root(cfg["train"]["ckpt_path"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        ensure_output_dirs(cfg)
    for command, logfile in jobs:
        run_job(command, logfile)
    print("[suite] ALL STAGES COMPLETED", flush=True)


if __name__ == "__main__":
    main()
