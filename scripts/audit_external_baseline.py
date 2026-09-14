"""Audit metadata for an external literature baseline before comparison."""

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED = (
    "method_name", "source_paper", "source_url", "code_url", "code_commit",
    "dataset", "split_manifest_sha256", "training_budget", "seed", "batch_size",
    "parameters", "checkpoint_sha256", "metrics_path",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--metrics")
    args = ap.parse_args()
    metadata_path = Path(args.metadata).resolve()
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise SystemExit(f"missing required baseline audit fields: {missing}")
    for label, value in (("checkpoint", args.checkpoint), ("metrics", args.metrics)):
        if value:
            path = Path(value).resolve()
            if not path.is_file():
                raise SystemExit(f"{label} file does not exist: {path}")
            expected = data["checkpoint_sha256"] if label == "checkpoint" else None
            if expected and label == "checkpoint" and sha256(path) != expected:
                raise SystemExit("external checkpoint SHA256 does not match metadata")
            if label == "metrics" and Path(data["metrics_path"]).name != path.name:
                raise SystemExit("metrics file name does not match metadata metrics_path")
    print(json.dumps(dict(status="ok", metadata=str(metadata_path), method=data["method_name"]), indent=2))


if __name__ == "__main__":
    main()
