"""Seed dispersion and paired cluster intervals, with explicit estimands."""

import bootstrap  # noqa: F401
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


PARTIAL = {"corrupt_aud_only", "clean_img_corrupt_aud", "corrupt_both"}


def cluster_interval(differences, clusters, seed=20260915, draws=2000):
    if len(differences) != len(clusters) or not len(differences):
        raise ValueError("Nonempty matched differences and cluster identities required")
    buckets = {}
    for value, cluster in zip(differences, clusters):
        if not np.isfinite(value):
            raise ValueError("Non-finite paired primary endpoint")
        buckets.setdefault(cluster, []).append(float(value))
    values = np.array([np.mean(buckets[k]) for k in sorted(buckets)])
    mean = float(values.mean())
    if len(values) < 2:
        return dict(mean=mean, low=None, high=None, clusters=len(values), paired_exposures=len(differences))
    rng = np.random.default_rng(seed)
    sampled = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(draws)])
    low, high = np.quantile(sampled, [.025, .975])
    return dict(mean=mean, low=float(low), high=float(high), clusters=len(values), paired_exposures=len(differences))


def primary_rows(path, cluster="audio_id"):
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["protocol"] != "fixed" or float(row["severity"]) != .4 or row["cue"] not in PARTIAL:
                continue
            if row["aud_target"] != "sample":
                raise ValueError("Primary endpoint must use present-audio sample targets")
            key = tuple(row[k] for k in ("image_id", "audio_id", "cue", "mask_seed", "img_family", "aud_family", "severity"))
            if key in rows:
                raise ValueError(f"Duplicate paired observation: {key}")
            rows[key] = (float(row["aud_missing_mse"]), row[cluster], row["aud_family"])
    return rows


def paired_difference(reference, candidate, family=None, draws=2000):
    if reference.keys() != candidate.keys():
        raise ValueError("Paired coverage differs; do not silently intersect test observations")
    keys = [k for k in reference if family is None or reference[k][2] == family]
    if not keys:
        raise ValueError("No observations for requested primary endpoint")
    if any(reference[k][1:] != candidate[k][1:] for k in keys):
        raise ValueError("Cluster/family identity differs between models")
    return cluster_interval([candidate[k][0]-reference[k][0] for k in keys],
                            [reference[k][1] for k in keys], draws=draws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cluster", choices=("audio_id", "speaker"), default="audio_id")
    ap.add_argument("--bootstrap_draws", type=int, default=2000)
    args = ap.parse_args()
    if args.bootstrap_draws < 100:
        ap.error("Use at least 100 bootstrap draws")
    root = Path(args.root)
    aggregate, comparisons, missing = {}, [], []
    for seed_dir in sorted(root.glob("seed_*")):
        for experiment in sorted(seed_dir.iterdir()):
            complete = experiment / "evaluation" / "complete.json"
            if not complete.is_file():
                if not experiment.name.startswith("parent") and experiment.name != "recognizer":
                    missing.append(str(complete))
                continue
            meta = json.loads(complete.read_text(encoding="utf-8"))
            if not meta.get("full_test"):
                raise ValueError(f"Smoke output cannot enter publication summary: {complete}")
            summary = json.loads((complete.parent / "summary.json").read_text(encoding="utf-8"))
            for key, metrics in summary.items():
                for metric, value in metrics.items():
                    token = json.dumps([json.loads(key), metric])
                    aggregate.setdefault(token, []).append((seed_dir.name, value["mean"], value["n"]))
        reference_path = seed_dir / "control" / "evaluation" / "per_item.csv.gz"
        if not reference_path.is_file():
            continue
        reference = primary_rows(reference_path, args.cluster)
        if not reference:
            missing.append(f"No severity=.4 primary endpoint: {reference_path}")
            continue
        candidates = {}
        for name in ("main", "no_causal", "no_cross"):
            path = seed_dir / name / "evaluation" / "per_item.csv.gz"
            if not path.is_file():
                continue
            candidate = primary_rows(path, args.cluster)
            candidates[name] = candidate
            for family in (None, "partial_temporal"):
                comparisons.append(dict(seed=seed_dir.name, candidate=name, reference="control",
                    family=family or "all", **paired_difference(reference, candidate, family, args.bootstrap_draws)))
        if "main" in candidates:
            for baseline in ("no_causal", "no_cross"):
                if baseline in candidates:
                    for family in (None, "partial_temporal"):
                        comparisons.append(dict(seed=seed_dir.name, candidate="main", reference=baseline,
                            family=family or "all", **paired_difference(candidates[baseline], candidates["main"], family, args.bootstrap_draws)))
    seed_summary = {key: dict(mean=float(np.mean([x[1] for x in rows])),
        sample_std=float(np.std([x[1] for x in rows], ddof=1)) if len(rows) > 1 else None,
        training_seeds=len(rows), observations=rows) for key, rows in aggregate.items()}
    report = dict(seed_statistics=seed_summary, paired_intervals=comparisons, missing=missing,
        cluster=args.cluster, endpoint="fixed severity=.4, 3 partial-audio cues, candidate-control masked MSE",
        estimand="Equal-cluster mean of per-cluster paired exposure differences; NOT exposure-weighted mean",
        cautions=["Intervals conditional on each trained model; do not include training variance",
                  "Recording clusters do not account for dependence between recordings of one speaker",
                  "Few speakers make speaker intervals unstable; report cluster count",
                  "Repeated masks are not training seeds; many exploratory metrics do not imply significance",
                  "Negative paired MSE differences favor the candidate; no multiplicity-adjusted significance claim"])
    (root / "statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Metrics={len(seed_summary)}, paired comparisons={len(comparisons)}, missing evaluations={len(missing)}")


if __name__ == "__main__":
    main()
