"""Plan or run the v17pro architecture comparison on MNIST and AudioMNIST."""

import bootstrap  # noqa: F401
import argparse
from pathlib import Path
import re
import subprocess
import sys

from common import load_config
from scripts.job_runner import run_job
from scripts.run_v16pro_suite import build_plan, execute_plan
from scripts.v17_protocol import ARMS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v17pro.yaml")
    parser.add_argument("--output", default="outputs/v17pro")
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root")
    parser.add_argument("--audiomnist-root")
    parser.add_argument("--detector-epochs", type=int, default=10)
    parser.add_argument("--parent-epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--recognizer-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Arms and seeds must be unique")
    if any(not 0 <= seed < 2**31 for seed in args.seeds) or not re.fullmatch(r"cpu|cuda(?::\d+)?", args.device):
        parser.error("Invalid seed or device")
    if min(args.detector_epochs, args.parent_epochs, args.epochs, args.recognizer_epochs) < 1:
        parser.error("Epoch budgets must be positive")
    if (min(args.max_batches, args.max_samples, args.examples) < 0 or
            (args.batch_size is not None and args.batch_size < 1) or
            (args.num_workers is not None and args.num_workers < 0)):
        parser.error("Invalid batch size, worker count, or diagnostic limit")
    if args.max_batches and not (args.max_samples or args.skip_eval):
        parser.error("Use --max-samples or --skip-eval with --max-batches")
    base = load_config(args.config)
    if not base.get("v17") or base.get("data", {}).get("dataset") != "mnist_audiomnist":
        parser.error("v17pro requires the AudioMNIST protocol")
    root = Path(args.output).resolve()
    configs, jobs = build_plan(base, root, args.seeds, args.device, args.detector_epochs,
                               args.parent_epochs, args.epochs, args.recognizer_epochs,
                               args.data_root, arms=args.arms, audiomnist_root=args.audiomnist_root,
                               max_batches=args.max_batches, batch_size=args.batch_size,
                               num_workers=args.num_workers)
    export = ["tools/export_v16pro.py", "--root", str(root), "--models", *args.arms,
              "--seeds", *map(str, args.seeds), "--device", args.device,
              "--max-samples", str(args.max_samples), "--examples", str(args.examples)]
    if args.max_batches:
        export += ["--allow-smoke", "--skip-scans"]
    print(f"v17pro: {len(jobs)} training jobs, {sum(job['epochs'] for job in jobs)} job-epochs.")
    if not args.run:
        for job in jobs:
            print(job["id"], subprocess.list2cmdline([sys.executable, "-u", *job["command"]]))
        if not args.skip_eval:
            print("evaluation", subprocess.list2cmdline([sys.executable, "-u", *export]))
        return
    execute_plan(root, configs, jobs, version="v17pro")
    if not args.skip_eval:
        run_job(export, root / "logs/export.log")


if __name__ == "__main__":
    main()
