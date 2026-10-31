"""Build an auditable Chinese review from cached supplementary v15 evaluation.

No model, checkpoint or inference runtime is loaded. Existing numerical results
are the only source of plots and aggregates. Synthetic inputs belong in tests.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
from urllib.parse import quote

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from tools.v15_joint_panels import FIELDS, MODALITIES, joint_artifact_paths, render_joint_sample

REGIONS = ("global", "corrupt", "clean", "foreground", "corrupt_foreground",
           "corrupt_background", "clean_foreground", "clean_background")
DISPLAY_REGIONS = ("corrupt", "corrupt_foreground", "corrupt_background", "clean")
REGION_LABELS = {"corrupt": "污染区", "corrupt_foreground": "污染前景",
                 "corrupt_background": "污染背景", "clean": "干净区"}
POLICY_LABELS = {"uniform": "均匀位置 uniform", "foreground": "前景约束 foreground"}
EXPECTED_CONDITIONS = {("uniform", "clean", "clean_both")} | {
    (policy, family, cue) for policy in POLICY_LABELS for family in ("local_gaussian", "local_impulse")
    for cue in ("corrupt_img_clean_aud", "clean_img_corrupt_aud", "corrupt_both")}
TABLE_PATHS = ("tables/by_condition.csv", "tables/paired_differences.csv")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant {value} in {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _component(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ValueError(f"Invalid condition path component: {value!r}")
    return value


def condition_folder(root, condition):
    seed = int(condition["training_seed"])
    policy = condition["policy"]
    if seed < 0 or policy not in POLICY_LABELS:
        raise ValueError("Invalid training seed or location policy")
    return root / f"seed_{seed}" / policy / (
        _component(condition["family"]) + "__" + _component(condition["cue_mode"]))


def _source_path(folder, relative):
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("Example source path escapes its condition directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _summary(values, expected=3):
    valid = {}
    for seed, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Invalid aggregate value for seed {seed}: {value!r}")
        valid[str(seed)] = float(value)
    numbers = list(valid.values())
    return {"mean": statistics.mean(numbers) if numbers else None,
            "sd": statistics.stdev(numbers) if len(numbers) > 1 else None,
            "n_seeds": len(numbers), "expected_seeds": expected,
            "complete_three_seed": len(numbers) == expected == 3, "by_seed": valid}


def verify_condition_sources(root, summary):
    """Verify the evaluator's completion records without importing its runtime."""
    records = []
    for condition in summary["conditions"]:
        folder = condition_folder(root, condition)
        source, marker = folder / "summary.json", folder / "complete.json"
        actual, completion = read_json(source), read_json(marker)
        if actual != condition:
            raise ValueError(f"Root and condition summaries differ: {folder}")
        if completion.get("status") != "complete":
            raise ValueError(f"Condition is not complete: {folder}")
        identity = completion["identity"]
        for key, value in identity.items():
            if key != "examples" and condition.get(key) != value:
                raise ValueError(f"Condition identity differs: {folder}, {key}")
        expected_count = min(int(identity["examples"]), int(condition["samples"]))
        if len(condition["examples"]) != expected_count:
            raise ValueError(f"Condition lacks its fixed first-N example cache: {folder}")
        required = {"summary.json"} | {example[key] for example in condition["examples"] for key in ("npz", "metadata")}
        if not required.issubset(completion["artifacts"]):
            raise ValueError(f"Completion record omits a required source artifact: {folder}")
        for relative, digest in completion["artifacts"].items():
            if sha(_source_path(folder, relative)) != digest:
                raise ValueError(f"Evaluation artifact hash differs: {folder / relative}")
        indices = []
        for example in condition["examples"]:
            metadata = read_json(_source_path(folder, example["metadata"]))
            indices.append(int(metadata.get("index", metadata.get("sample_index", -1))))
        if sorted(indices) != list(range(expected_count)):
            raise ValueError(f"Condition cache does not contain the fixed first N indices: {folder}")
        records.append({"training_seed": condition["training_seed"], "policy": condition["policy"],
                        "family": condition["family"], "cue_mode": condition["cue_mode"],
                        "samples": condition["samples"], "partial_evaluation": condition.get("partial_evaluation", False),
                        "total_split_samples": condition.get("total_split_samples"),
                        "original_checkpoint_sha256": condition.get("original_checkpoint_sha256"),
                        "data_sha256": condition.get("data_sha256"),
                        "source_summary": source.relative_to(root).as_posix(), "source_summary_sha256": sha(source),
                        "complete_marker": marker.relative_to(root).as_posix(), "complete_marker_sha256": sha(marker),
                        "verified_artifact_count": len(completion["artifacts"]), "expected_examples": expected_count})
    return records


def _flat_metrics(condition):
    stats = condition["statistics"]
    result = {"index_accuracy": stats["index_accuracy"]}
    for modality in MODALITIES:
        data = stats["modalities"][modality]
        for region in REGIONS:
            for key, value in data["regions"].get(region, {}).items():
                result[f"{modality}.regions.{region}.{key}"] = value
        for category in ("detector", "placement"):
            for key, value in data.get(category, {}).items():
                result[f"{modality}.{category}.{key}"] = value
    return result


def aggregate_conditions(summary):
    """Equal-weight seed means/SDs; never pool different location conditions."""
    conditions = summary["conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("The source summary must contain a non-empty conditions list")
    grouped = defaultdict(list)
    seen = set()
    for condition in conditions:
        key = (int(condition["training_seed"]), condition["policy"],
               condition["family"], condition["cue_mode"])
        if key in seen:
            raise ValueError(f"Duplicate seed/condition: {key}")
        seen.add(key)
        if int(condition["samples"]) != int(condition["statistics"]["n"]):
            raise ValueError(f"Sample count differs from statistics.n: {key}")
        grouped[key[1:]].append(condition)
    groups = []
    for (policy, family, cue_mode), records in sorted(grouped.items()):
        flattened = {str(item["training_seed"]): _flat_metrics(item) for item in records}
        keys = sorted(set().union(*(record.keys() for record in flattened.values())))
        groups.append({"policy": policy, "family": family, "cue_mode": cue_mode,
                       "seeds": sorted(int(seed) for seed in flattened),
                       "samples_by_seed": {str(r["training_seed"]): r["samples"] for r in records},
                       "total_split_samples_by_seed": {str(r["training_seed"]): r.get("total_split_samples") for r in records},
                       "partial_evaluation": any(r.get("partial_evaluation", False) for r in records),
                       "metrics": {key: _summary({seed: record.get(key) for seed, record in flattened.items()})
                                   for key in keys}})
    by_key = {(int(c["training_seed"]), c["policy"], c["family"], c["cue_mode"]): c for c in conditions}
    paired = []
    matching = sorted({(c["family"], c["cue_mode"]) for c in conditions if c["policy"] == "foreground"})
    for family, cue_mode in matching:
        seed_rows = []
        for seed in sorted({int(c["training_seed"]) for c in conditions}):
            first = by_key.get((seed, "uniform", family, cue_mode))
            second = by_key.get((seed, "foreground", family, cue_mode))
            reasons = []
            if first is None or second is None:
                reasons.append("missing policy condition")
            else:
                for field in ("samples", "split", "data_sha256", "original_checkpoint_sha256", "evaluation_seed",
                              "area_coverage", "options", "source_sha256", "runtime_sha256"):
                    if first.get(field) is None or second.get(field) is None:
                        reasons.append(f"missing {field}")
                    elif first[field] != second[field]:
                        reasons.append(f"different {field}")
            row = {"training_seed": seed, "comparable": not reasons, "reasons": reasons}
            if not reasons:
                a, b = _flat_metrics(first), _flat_metrics(second)
                row["partial_evaluation"] = bool(first.get("partial_evaluation") or second.get("partial_evaluation"))
                row["differences"] = {key: (b[key] - a[key] if b[key] is not None and a[key] is not None else None)
                                      for key in sorted(a.keys() & b.keys())}
            seed_rows.append(row)
        valid = [row for row in seed_rows if row["comparable"]]
        keys = sorted(set().union(*(row["differences"].keys() for row in valid))) if valid else []
        paired.append({"family": family, "cue_mode": cue_mode, "direction": "foreground minus uniform",
                       "per_seed": seed_rows, "metrics": {key: _summary({str(r["training_seed"]):
                            r["differences"].get(key) for r in valid}) for key in keys}})
    observed_seeds = sorted({int(c["training_seed"]) for c in conditions})
    missing = {str(seed): [list(key) for key in sorted(EXPECTED_CONDITIONS - {
        (c["policy"], c["family"], c["cue_mode"]) for c in conditions if int(c["training_seed"]) == seed})]
        for seed in observed_seeds}
    unexpected = {str(seed): [list(key) for key in sorted({
        (c["policy"], c["family"], c["cue_mode"]) for c in conditions if int(c["training_seed"]) == seed}
        - EXPECTED_CONDITIONS)] for seed in observed_seeds}
    coverage = {"expected_training_seed_count": 3, "expected_conditions_per_seed": 13,
                "expected_total_conditions": 39, "observed_training_seeds": observed_seeds,
                "missing_conditions_by_seed": missing,
                "unexpected_conditions_by_seed": unexpected,
                "complete_three_seed_condition_grid": len(observed_seeds) == 3 and not any(missing.values()) and not any(unexpected.values())}
    return {"schema": "v15-foreground-report.v1", "partial_evaluation": bool(summary.get("partial_evaluation") or
                any(c.get("partial_evaluation") for c in conditions)),
            "evaluation_status": summary.get("status", "unknown"), "split": summary.get("split"),
            "seeds": summary.get("seeds", sorted({int(c["training_seed"]) for c in conditions})),
            "condition_count": len(conditions), "condition_coverage": coverage,
            "groups": groups, "matched_policy_differences": paired,
            "definitions": {"seed_aggregation": "equal-weight mean and sample SD (ddof=1) of per-seed statistics; n<2 gives null SD",
                "mse": "Within each seed, SSE / the true-region pixel count. Empty regions stay null.",
                "index_accuracy": "Index classification accuracy from corrupted input cues; not restored-content recognition accuracy",
                "foreground_coverage": "mean per-sample fraction of clean-target foreground covered by true corruption, among active evaluable samples",
                "constraint_not_met_rate": "constraint_not_met / constraint_applicable; see separate infeasible/no-foreground counts",
                "comparison": "Compare policies only within identical family/cue and matched training seed. No 7-condition versus 6-condition grand mean."}}


def _same_npz(source, destination):
    with np.load(source, allow_pickle=False) as first, np.load(destination, allow_pickle=False) as second:
        for modality in MODALITIES:
            for field in FIELDS:
                key = f"{modality}_{field}"
                left, right = first[key], second[key]
                if left.ndim == 3 and left.shape[0] == 1:
                    left = left[0]
                if not np.array_equal(left, right):
                    return False
    return True


def export_aggregate_tables(root, aggregate):
    """Serialize the existing aggregate values exactly; do not re-aggregate."""
    root = Path(root)
    seeds = sorted({str(seed) for seed in aggregate["seeds"]} | {
        str(seed) for section in ("groups", "matched_policy_differences")
        for group in aggregate[section] for metric in group["metrics"].values()
        for seed in metric["by_seed"]}, key=int)
    fields = ["policy", "family", "cue", "modality", "region", "category", "metric",
              "metric_key", "direction", "mean", "sd", "n_seeds", "expected_seeds",
              "complete_three_seed", "partial_evaluation"] + [f"seed_{seed}" for seed in seeds]
    outputs = []
    for relative, section in zip(TABLE_PATHS, ("groups", "matched_policy_differences")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for group in aggregate[section]:
                paired = section == "matched_policy_differences"
                for key, metric in sorted(group["metrics"].items()):
                    pieces = key.split(".")
                    row = {"policy": "foreground_minus_uniform" if paired else group["policy"],
                           "family": group["family"], "cue": group["cue_mode"],
                           "modality": pieces[0] if len(pieces) > 1 else "index",
                           "region": pieces[2] if len(pieces) == 4 and pieces[1] == "regions" else "",
                           "category": pieces[1] if len(pieces) > 1 else "classification",
                           "metric": pieces[-1], "metric_key": key,
                           "direction": group["direction"] if paired else "",
                           "partial_evaluation": any(r.get("partial_evaluation", False) for r in group["per_seed"])
                               if paired else group["partial_evaluation"]}
                    row.update({field: metric[field] for field in
                                ("mean", "sd", "n_seeds", "expected_seeds", "complete_three_seed")})
                    row.update({f"seed_{seed}": metric["by_seed"].get(seed) for seed in seeds})
                    writer.writerow(row)
        temporary.replace(path)
        outputs.append(path)
    return outputs


def render_examples(root, conditions, audit_scripts):
    """Verified complete bundles are skipped; damaged/incomplete bundles rebuild."""
    result = []
    renderer_sha = sha(PROJECT / "tools/v15_joint_panels.py")
    audit_versions = {}
    for name in ("audit_panel_alignment", "audit_pdf_text", "audit_figure_collisions"):
        spec = importlib.util.find_spec(name) if audit_scripts is None else None
        path = Path(audit_scripts) / (name + ".py") if audit_scripts is not None else Path(spec.origin) if spec else None
        if path is None or not path.is_file():
            raise RuntimeError("Pass --audit-scripts or configure the three nature-figure audit modules")
        audit_versions[name] = sha(path)
    total = sum(len(c.get("examples", [])) for c in conditions)
    for condition in sorted(conditions, key=lambda c: (int(c["training_seed"]), c["family"], c["cue_mode"], c["policy"])):
        folder = condition_folder(root, condition)
        indices = set()
        for example in condition.get("examples", []):
            source_npz = _source_path(folder, example["npz"])
            source_json = _source_path(folder, example["metadata"])
            metadata = read_json(source_json)
            if "index" not in metadata:
                metadata["index"] = metadata["sample_index"]
            index = int(metadata["index"])
            if index < 0 or index in indices:
                raise ValueError("Invalid or duplicate example index")
            indices.add(index)
            for key in ("training_seed", "policy", "family", "cue_mode", "split"):
                if metadata[key] != condition[key]:
                    raise ValueError(f"Example metadata differs from condition: {key}")
            if metadata["sample_id"] != example["sample_id"]:
                raise ValueError("Example sample_id differs from metadata")
            stem = root / "figures" / folder.relative_to(root) / f"sample_{index:05d}"
            paths = joint_artifact_paths(stem)
            marker = stem.with_name(stem.name + ".render-source.json")
            source = {"schema": "v15-joint-source.v1", "npz": source_npz.relative_to(root).as_posix(),
                      "metadata": source_json.relative_to(root).as_posix(), "npz_sha256": sha(source_npz),
                      "metadata_sha256": sha(source_json), "renderer_sha256": renderer_sha,
                      "audit_modules_sha256": audit_versions}
            verified = False
            if marker.exists():
                try:
                    old = read_json(marker)
                    if old.get("source") == source and all(p.is_file() for p in paths):
                        verified = all(old.get("artifacts", {}).get(p.name) == sha(p) for p in paths)
                        if verified:
                            verified = _same_npz(source_npz, paths[2])
                except (OSError, ValueError, KeyError):
                    verified = False
            action = "verified" if verified else "rebuilt" if any(p.exists() for p in paths) else "rendered"
            if not verified:
                stem.parent.mkdir(parents=True, exist_ok=True)
                with np.load(source_npz, allow_pickle=False) as archive:
                    arrays = {modality: {field: archive[f"{modality}_{field}"] for field in FIELDS}
                              for modality in MODALITIES}
                # Publish the source marker last so interrupted replacement never looks complete.
                with tempfile.TemporaryDirectory(prefix=".joint-report-", dir=stem.parent) as temporary:
                    staged = render_joint_sample(Path(temporary) / stem.name, arrays, metadata, audit_scripts=audit_scripts)
                    if not _same_npz(source_npz, staged[2]):
                        raise ValueError("Rendered numeric archive differs from source arrays")
                    for first, second in zip(staged, paths):
                        os.replace(first, second)
                write_json(marker, {"source": source, "artifacts": {p.name: sha(p) for p in paths}})
            result.append({"training_seed": condition["training_seed"], "policy": condition["policy"],
                           "family": condition["family"], "cue_mode": condition["cue_mode"],
                           "sample_id": metadata["sample_id"], "index": index,
                           "partial_evaluation": condition.get("partial_evaluation", False),
                           "source": source, "action": action,
                           "artifacts": {p.suffix.lstrip(".") if p.suffix != ".json" else
                                "alignment" if p.name.endswith(".alignment.json") else
                                "qa" if p.name.endswith(".qa.json") else "metadata": p.relative_to(root).as_posix()
                                for p in paths}, "source_marker": marker.relative_to(root).as_posix()})
            if len(result) % 10 == 0 or len(result) == total:
                print(f"[figures] {len(result)}/{total}; last={action}", flush=True)
        if indices != set(range(len(condition.get("examples", [])))):
            raise ValueError(f"Example indices are not the fixed first N indices: {folder}")
    return result


def compare_original(conditions, original_results):
    """Record numeric deltas, without turning GPU/batch differences into failures."""
    if original_results is None:
        return {"requested": False, "records": [], "sources": []}
    original_results = Path(original_results).resolve()
    records, sources = [], []
    originals = {}
    for seed in sorted({int(c["training_seed"]) for c in conditions if c["policy"] == "uniform"}):
        path = original_results / f"seed_{seed}" / "evaluation/predicted.json"
        if path.is_file():
            originals[seed] = read_json(path)
            sources.append({"path": str(path), "sha256": sha(path), "training_seed": seed})
    for condition in conditions:
        if condition["policy"] != "uniform":
            continue
        seed = int(condition["training_seed"])
        item = {"training_seed": seed, "family": condition["family"], "cue_mode": condition["cue_mode"]}
        original = originals.get(seed)
        if original is None:
            records.append({**item, "status": "original_missing", "comparable": False})
            continue
        old_conditions = original.get("conditions", {})
        if isinstance(old_conditions, list):
            old = next((c for c in old_conditions if c.get("family") == condition["family"] and
                        c.get("cue_mode") == condition["cue_mode"]), None)
        else:
            old = old_conditions.get(condition["family"] + "__" + condition["cue_mode"])
        if old is None:
            records.append({**item, "status": "original_condition_missing", "comparable": False})
            continue
        original_sha, current_sha = original.get("checkpoint_sha256"), condition.get("original_checkpoint_sha256")
        reasons = []
        if condition.get("partial_evaluation"):
            reasons.append("new evaluation is partial")
        if int(condition["statistics"]["n"]) != int(old.get("n", original.get("n_unique_samples", -1))):
            reasons.append("sample counts differ")
        if original.get("split") != condition.get("split"):
            reasons.append("splits differ")
        if original.get("seed") != condition.get("evaluation_seed"):
            reasons.append("evaluation seeds differ")
        if original.get("mask_source") != "predicted":
            reasons.append("original mask source is not predicted")
        if "area_coverage" in condition and original.get("severity") != condition["area_coverage"]:
            reasons.append("corruption area severity differs")
        if original_sha is None or current_sha is None:
            reasons.append("checkpoint provenance is unavailable")
        elif original_sha != current_sha:
            reasons.append("checkpoint SHA256 differs")
        differences = {}
        new_stats = condition["statistics"]
        differences["index_accuracy"] = _difference(new_stats["index_accuracy"], old.get("accuracy"))
        for modality in MODALITIES:
            for region in ("global", "corrupt", "clean"):
                old_pixels = old.get(modality, {}).get(region + "_pixels")
                new_pixels = new_stats["modalities"][modality]["regions"][region].get("pixels")
                if old_pixels is None or new_pixels is None or old_pixels != new_pixels:
                    reasons.append(f"{modality}.{region} pixel counts differ or unavailable")
                for metric in ("input_sse", "recovered_sse", "input_mse", "recovered_mse"):
                    differences[f"{modality}.{region}.{metric}"] = _difference(
                        new_stats["modalities"][modality]["regions"][region].get(metric),
                        old.get(modality, {}).get(f"{region}_{metric}"))
        records.append({**item, "status": "compared", "comparable": not reasons, "reasons": reasons,
                        "old_checkpoint_sha256": original_sha, "new_original_checkpoint_sha256": current_sha,
                        "checkpoint_sha256_match": original_sha == current_sha if original_sha and current_sha else None,
                        "old_samples": old.get("n", original.get("n_unique_samples")), "new_samples": condition["samples"],
                        "differences": differences})
    maxima = {}
    for label, ending in (("input_sse", ".input_sse"), ("input_mse", ".input_mse"),
                          ("recovered_mse", ".recovered_mse"), ("index_accuracy", "index_accuracy")):
        for only_comparable, suffix in ((False, "all_recorded"), (True, "comparable_only")):
            values = [value["absolute_difference"] for row in records if row.get("status") == "compared" and
                      (not only_comparable or row["comparable"]) for key, value in row["differences"].items()
                      if key.endswith(ending) and value["absolute_difference"] is not None]
            maxima[label + "_max_absolute_difference_" + suffix] = max(values) if values else None
    return {"requested": True, "original_root": str(original_results), "records": records, "sources": sources,
            "maxima": maxima, "notes": "Numeric differences are recorded without a bitwise-equality or tolerance claim. Partial/different populations are not consistency validation. Noise SD and detector threshold are inherited from the evaluator's verified checkpoint configuration; this report has no independent new-identity fields for those settings."}


def _difference(current, original):
    return {"new": current, "old": original,
            "difference": current - original if current is not None and original is not None else None,
            "absolute_difference": abs(current - original) if current is not None and original is not None else None}


def _format(metric, percent=False):
    if not metric or metric.get("mean") is None:
        return "n/a"
    scale, suffix = (100, "%") if percent else (1, "")
    mean = metric["mean"] * scale
    text = f"{mean:.2f}" if percent else f"{mean:.5g}"
    sd = metric["sd"]
    text += f" ± {sd * scale:.2f}" if percent and sd is not None else f" ± {sd:.3g}" if sd is not None else "（SD n/a）"
    return text + suffix + (f" [n={metric['n_seeds']}]" if metric["n_seeds"] != 3 else "")


def _href(path):
    return quote(str(path).replace("\\", "/"), safe="/._-")


def _difference_display(metric, percent=False):
    # Absolute coverage differences are percentage points, not relative changes.
    return _format(metric, percent).replace("%", " pp")


def _scalar_display(value):
    return "n/a" if value is None else f"{value:.6g}"


def make_html(aggregate, figures):
    partial = aggregate["partial_evaluation"]
    grid_complete = aggregate["condition_coverage"]["complete_three_seed_condition_grid"]
    status = ("部分样本评估 · 非完整测试集结论" if partial else
              "seed / 条件覆盖不完整" if not grid_complete or aggregate["evaluation_status"] != "complete" else "完整三 seed 样本评估")
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>v15 前景覆盖补充评估</title><style>body{font:15px/1.6 system-ui,sans-serif;color:#17212b;background:#f7f8fa;margin:0}main{max-width:1500px;margin:auto;padding:28px}h1{margin:.2em 0}h2{margin-top:2em}p{max-width:1100px}.notice{background:#fff1cc;border-left:5px solid #b77c0a;padding:12px 18px}.card,details{background:white;border:1px solid #d5dbe1;border-radius:7px;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:650}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dde3e8;text-align:left;vertical-align:top}th{background:#edf1f5;white-space:nowrap}.numeric{white-space:nowrap}.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(600px,100%),1fr));gap:14px}figure{margin:12px 0}img{width:100%;height:auto;border:1px solid #dde3e8}figcaption{font-weight:bold}a{color:#185e9b}select{padding:6px;margin:5px}code{overflow-wrap:anywhere}small{color:#526170}details[hidden]{display:none}</style><main>",
        "<h1>v15 前景覆盖补充评估</h1>", f"<div class='notice'><strong>{status}</strong><br>" +
        f"输入状态：{escape(aggregate['evaluation_status'])}；split：{escape(str(aggregate['split']))}；{aggregate['condition_count']} / 39 个 seed×位置策略×噪声条件；训练 seeds：{escape(str(aggregate['seeds']))}。" +
        "表中均值 ± SD 为同一条件下训练 seed 的等权均值和样本标准差；有效 seed 不足 3 时明确显示 n。</div>",
        "<p>只在相同噪声 family、cue mode 与训练 seed 内比较 uniform 和 foreground。uniform 的 clean 条件单独展示，不与 foreground 的六个污染条件合成总体均值。前景由干净目标定义，仅用于生成位置约束和评价；真实污染掩码不作为 predicted 模式的模型输入。</p>",
        "<p><strong>Index ACC 是输入 cue 的 Index 分类准确率，不是恢复内容识别率。</strong>区域 MSE 先在每个 seed 内按真实区域像素累计 SSE/像素数；空区域保持 n/a。前景覆盖率是活跃、可评价样本的覆盖比例均值；未满足率分母为适用约束样本，具体计数与不可行/无前景情况保存在数值文件。</p>",
        "<p><a href='aggregate.json'>完整聚合数值</a> · <a href='tables/by_condition.csv' download>分条件 CSV</a> · <a href='tables/paired_differences.csv' download>配对差值 CSV</a> · <a href='summary.json'>输入评估 summary</a> · <a href='report_manifest.json'>报告来源与完整性清单</a></p><h2>同条件跨 seed 汇总</h2>"]
    grouped = defaultdict(list)
    for group in aggregate["groups"]:
        grouped[(group["family"], group["cue_mode"])].append(group)
    for (family, mode), groups in sorted(grouped.items()):
        parts.append(f"<details open><summary>{escape(family)} · {escape(mode)}</summary>")
        for group in groups:
            sample_counts = {seed: f"{count}/{group['total_split_samples_by_seed'].get(seed) or '?'}"
                             for seed, count in group['samples_by_seed'].items()}
            parts.append(f"<p><strong>{POLICY_LABELS[group['policy']]}</strong> · Index ACC: {_format(group['metrics']['index_accuracy'], True)} · "
                         f"各 seed 已评估/完整split样本数：{escape(str(sample_counts))}" + (" · 部分评估" if group["partial_evaluation"] else "") + "</p>")
        parts.append("<div class='scroll'><table><tr><th>策略 / 模态</th>" + "".join(f"<th>{REGION_LABELS[r]} MSE<br>输入 → 恢复</th>" for r in DISPLAY_REGIONS) + "<th>前景覆盖率</th><th>约束未满足率</th><th>检测 F1</th></tr>")
        for group in groups:
            for modality in MODALITIES:
                parts.append(f"<tr><td>{escape(group['policy'])}<br>{'图像' if modality == 'img' else '音频'}</td>")
                for region in DISPLAY_REGIONS:
                    pre = f"{modality}.regions.{region}."
                    parts.append(f"<td class='numeric'>{_format(group['metrics'].get(pre + 'input_mse'))}<br>→ {_format(group['metrics'].get(pre + 'recovered_mse'))}</td>")
                for key, percent in (("placement.mean_foreground_coverage", True), ("placement.constraint_not_met_rate", True), ("detector.f1", False)):
                    parts.append(f"<td class='numeric'>{_format(group['metrics'].get(modality + '.' + key), percent)}</td>")
                parts.append("</tr>")
        parts.append("</table></div></details>")
    parts.append("<h2>同 seed 配对差值</h2><p>方向为 foreground − uniform；MSE 的负差值表示该区域误差更低。这是位置策略之间的差异，不是跨模态机制的因果验证。均值与 SD 先按 seed 求差，再跨 seed 汇总；覆盖率差以百分点（pp）表示。</p><div class='scroll'><table><tr><th>条件</th><th>模态</th><th>污染区恢复 MSE 差</th><th>污染前景恢复 MSE 差</th><th>前景覆盖率差</th></tr>")
    for item in aggregate["matched_policy_differences"]:
        for modality in MODALITIES:
            parts.append(f"<tr><td>{escape(item['family'])}<br>{escape(item['cue_mode'])}</td><td>{modality}</td>" + "".join(
                f"<td>{_difference_display(item['metrics'].get(modality + '.' + key), percent)}</td>" for key, percent in (
                    ("regions.corrupt.recovered_mse", False), ("regions.corrupt_foreground.recovered_mse", False),
                    ("placement.mean_foreground_coverage", True))) + "</tr>")
    parts.append("</table></div><h2>原 uniform 结果复核</h2>")
    original = aggregate["original_comparison"]
    if not original["requested"]:
        parts.append("<p>本次未指定原始结果目录，未执行新旧一致性复核。</p>")
    else:
        parts.append("<p>按同 seed、family 和 cue 对照旧 predicted 结果。下列数值是实际差值，不假定跨 batch/GPU 逐位一致；partial、样本数或权重不同的记录不算一致性验证。</p><div class='scroll'><table><tr><th>差值项目</th><th>所有已读记录最大绝对差</th><th>仅可比记录最大绝对差</th></tr>")
        for metric in ("input_sse", "input_mse", "recovered_mse", "index_accuracy"):
            parts.append(f"<tr><td>{metric}</td>" + "".join(f"<td>{_scalar_display(original['maxima'].get(metric + '_max_absolute_difference_' + scope))}</td>" for scope in ("all_recorded", "comparable_only")) + "</tr>")
        parts.append("</table></div><details><summary>逐条件 SHA 与可比性</summary><table><tr><th>seed / 条件</th><th>checkpoint SHA 匹配</th><th>可比性与原因</th></tr>")
        for row in original["records"]:
            parts.append(f"<tr><td>{row['training_seed']} / {escape(row['family'])} / {escape(row['cue_mode'])}</td><td>{escape(str(row.get('checkpoint_sha256_match', 'n/a')))}</td><td>{escape(str(row.get('comparable')))}; {escape('; '.join(row.get('reasons', [row['status']])))}</td></tr>")
        parts.append("</table></details>")
    parts.append("<h2>全部固定样例图库</h2><p>直接展示缓存中的前 N 个固定索引，不按恢复质量筛选。每张图均为同一个配对样本：上排图像、下排音频、六列共享布局。展开同一条目即可对照两种位置策略；点击图片可打开原尺寸 PNG。PDF、数值数组和来源 metadata 均可访问。</p><div class='card'>")
    for key, label in (("training_seed", "seed"), ("family", "family"), ("cue_mode", "cue"), ("index", "样本索引")):
        values = sorted({str(f[key]) for f in figures})
        parts.append(f"<label>{label}<select data-filter='{key}'><option value=''>全部</option>" + "".join(f"<option value='{escape(v, quote=True)}'>{escape(v)}</option>" for v in values) + "</select></label>")
    parts.append("<span id='shown'></span></div>")
    gallery = defaultdict(dict)
    for figure in figures:
        key = (figure["training_seed"], figure["family"], figure["cue_mode"], figure["index"], str(figure["sample_id"]))
        if figure["policy"] in gallery[key]:
            raise ValueError("Duplicate gallery policy/sample")
        gallery[key][figure["policy"]] = figure
    for (seed, family, mode, index, sample_id), policies in sorted(gallery.items()):
        parts.append(f"<details class='sample' data-training_seed='{seed}' data-family='{escape(family, quote=True)}' data-cue_mode='{escape(mode, quote=True)}' data-index='{index}'><summary>seed {seed} · {escape(family)} · {escape(mode)} · 样本 {index} / Pair {escape(sample_id)}</summary><div class='pair'>")
        for policy in ("uniform", "foreground"):
            if policy not in policies:
                if policy == "foreground" and family == "clean":
                    parts.append("<p>clean 条件由两种位置策略共享，不重复生成 foreground clean 图。</p>")
                else:
                    parts.append(f"<p>{policy}：本次结果中没有对应样例。</p>")
                continue
            figure = policies[policy]
            paths = figure["artifacts"]
            png = _href(paths["png"])
            parts.append(f"<figure><figcaption>{POLICY_LABELS[policy]}</figcaption><a href='{png}' target='_blank'><img loading='lazy' src='{png}' alt='{escape(policy)}，seed {seed}，Pair {escape(sample_id, quote=True)} 的图像与音频联合恢复'></a><small>" + " · ".join(
                f"<a href='{_href(paths[key])}'>{label}</a>" for key, label in (("pdf", "PDF"), ("npz", "数值NPZ"), ("metadata", "metadata"), ("qa", "QA"))) +
                f" · <a href='{_href(figure['source']['metadata'])}'>原始样例来源</a></small></figure>")
        parts.append("</div></details>")
    parts.append("<script>const controls=[...document.querySelectorAll('[data-filter]')],items=[...document.querySelectorAll('.sample')];function filter(){let n=0;for(const item of items){const show=controls.every(c=>!c.value||item.getAttribute('data-'+c.dataset.filter)===c.value);item.hidden=!show;if(show)n++;}document.getElementById('shown').textContent='显示 '+n+' / '+items.length+' 个配对条目';}controls.forEach(c=>c.addEventListener('change',filter));filter();</script></main></html>")
    return "\n".join(parts)


def build_report(root, *, audit_scripts=None, original_results=None):
    root = Path(root).resolve()
    source = root / "summary.json"
    source_digest = sha(source)
    summary = read_json(source)
    verified_sources = verify_condition_sources(root, summary)
    aggregate = aggregate_conditions(summary)
    figures = render_examples(root, summary["conditions"], audit_scripts)
    if sha(source) != source_digest or verify_condition_sources(root, summary) != verified_sources:
        raise ValueError("Evaluation sources changed while generating this report")
    aggregate["original_comparison"] = compare_original(summary["conditions"], original_results)
    aggregate["figure_count"] = len(figures)
    aggregate["source_summary_sha256"] = sha(source)
    write_json(root / "aggregate.json", aggregate)
    export_aggregate_tables(root, aggregate)
    html = make_html(aggregate, figures)
    temporary = root / "review.html.tmp"
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(root / "review.html")
    manifest = {"schema": "v15-foreground-report-manifest.v1", "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": summary.get("status", "unknown"), "partial_evaluation": aggregate["partial_evaluation"],
        "report_scope": "All conditions and every cached example listed in the source summary; no sample selection by outcome",
        "summary_source": {"path": "summary.json", "sha256": sha(source)},
        "report_source_sha256": sha(Path(__file__)), "renderer_source_sha256": sha(PROJECT / "tools/v15_joint_panels.py"),
        "source_condition_count": len(summary["conditions"]), "figure_count": len(figures),
        "actions": {action: sum(f["action"] == action for f in figures) for action in ("rendered", "rebuilt", "verified")},
        "condition_coverage": aggregate["condition_coverage"], "condition_sources": verified_sources,
        "original_comparison_sources": aggregate["original_comparison"]["sources"], "figures": figures,
        "artifacts": {name: sha(root / name) for name in ("aggregate.json", "review.html", *TABLE_PATHS)},
        "limitations": ["Automated PDF checks do not replace visual assessment of every experimental panel.",
                        "Index accuracy does not measure recognition of recovered content.",
                        "A partial evaluation or missing training seed is not a complete three-seed test-set result."]}
    write_json(root / "report_manifest.json", manifest)
    print(f"[report complete] {len(figures)} figures; partial={aggregate['partial_evaluation']}; {root / 'review.html'}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--audit-scripts", type=Path)
    parser.add_argument("--original-results", type=Path)
    args = parser.parse_args(argv)
    build_report(args.root, audit_scripts=args.audit_scripts, original_results=args.original_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
