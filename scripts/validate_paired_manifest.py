"""Validate a real paired audiovisual manifest before a v14pro run."""

import bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from common import load_config
from data.dataset import TruePairedManifestDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="configs/v14pro.yaml")
    args = ap.parse_args()
    manifest = Path(args.manifest).resolve()
    if not manifest.is_file():
        ap.error(f"manifest does not exist: {manifest}")
    cfg = load_config(args.config)
    cfg["data"].update(dataset="paired_manifest", manifest_path=str(manifest),
                        validate_paths=True, require_unique_modalities=True,
                        require_speaker_disjoint=True, paper_split={"enabled": False})
    counts = {}
    speakers = {}
    for split in ("train", "val", "test"):
        dataset = TruePairedManifestDataset(cfg, split=split)
        counts[split] = len(dataset)
        speakers[split] = sorted({row["speaker_id"] for row in dataset.rows})
    result = dict(status="ok", manifest=str(manifest), counts=counts, speakers=speakers,
                  required_columns=sorted(TruePairedManifestDataset.REQUIRED_COLUMNS))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
