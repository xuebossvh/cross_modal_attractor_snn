"""Run v9c training, optionally followed by the first three ablations.

Use this as the single long-running server command. The suite creates output
directories, writes one log per experiment, and stops on the first failed run.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import bootstrap  # noqa: F401

from common import load_config
from paths import PROJECT_ROOT, logs_dir
from make_v9c_ablations import write_variants


def _config_rel(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.relative_to(PROJECT_ROOT).as_posix()


def _mkdir_outputs(config):
    cmd = [sys.executable, "scripts/mkdir_outputs.py", "--config", _config_rel(config)]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _log_path(config):
    cfg = load_config(str(config))
    cfg["_config_path"] = str(config)
    tag = cfg["train"]["output_version"]
    return logs_dir(cfg) / f"train_{tag}_50ep.log"


def _run_train(config, label):
    config = Path(config)
    _mkdir_outputs(config)
    log_path = _log_path(config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "scripts/train.py", "--config", _config_rel(config)]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[suite] start {label}", flush=True)
    print(f"[suite] config: {_config_rel(config)}", flush=True)
    print(f"[suite] log: {log_path.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise SystemExit(
            f"[suite] {label} failed with code {proc.returncode}; "
            f"see {log_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"[suite] done {label}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v9c.yaml")
    ap.add_argument("--with_ablations", action="store_true",
                    help="Run the three predefined v9c ablations after main training.")
    ap.add_argument("--ablations_only", action="store_true",
                    help="Skip main training and run only the three ablations.")
    args = ap.parse_args()

    if not args.ablations_only:
        _run_train(args.config, "main")

    if args.with_ablations or args.ablations_only:
        written = write_variants(args.config)
        for tag, path in written:
            _run_train(path, tag)

    print("[suite] all requested runs complete", flush=True)


if __name__ == "__main__":
    main()
