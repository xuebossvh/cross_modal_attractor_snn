"""Stream and independently audit exported v15 foreground evaluation numbers.

Uses only the Python standard library; never loads a model or changes original
result artifacts. This audits recorded true-region sufficient statistics and
their aggregation, not a fresh pixel-level evaluation of every model output.
CSV undefined values must be empty; the corresponding JSON values must be null.
Missing seeds are explicitly incomplete scope, never a full-39-condition pass.
"""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


EXPECTED_SEEDS = (1234, 2345, 3456)
REGIONS = ("global", "corrupt", "clean", "foreground", "corrupt_foreground",
           "corrupt_background", "clean_foreground", "clean_background")
PARTITIONS = (("global", ("corrupt", "clean")),
              ("corrupt", ("corrupt_foreground", "corrupt_background")),
              ("clean", ("clean_foreground", "clean_background")),
              ("foreground", ("corrupt_foreground", "clean_foreground")),
              ("global", ("foreground", "corrupt_background", "clean_background")))
MODES = ("corrupt_both", "clean_img_corrupt_aud", "corrupt_img_clean_aud")
EXPECTED_CONDITIONS = {("uniform", "clean", "clean_both")} | {
    (policy, family, mode) for policy in ("uniform", "foreground")
    for family in ("local_gaussian", "local_impulse") for mode in MODES}
OPTIONS = {"min_foreground_coverage": .35, "img_threshold": .1,
           "aud_absolute_threshold": .02, "aud_relative_threshold": .1}
REL_TOL, ABS_TOL = 1e-9, 1e-6


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def same(actual, expected, where):
    if expected is None:
        require(actual is None, f"{where}: expected null, got {actual!r}")
    elif isinstance(expected, bool):
        require(actual is expected, f"{where}: expected {expected}, got {actual!r}")
    elif isinstance(expected, int):
        require(type(actual) is int and actual == expected,
                f"{where}: expected integer {expected}, got {actual!r}")
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(actual, expected,
                    rel_tol=REL_TOL, abs_tol=ABS_TOL),
                f"{where}: expected {expected!r}, got {actual!r}")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{where}: keys differ")
        for key, value in expected.items():
            same(actual[key], value, f"{where}.{key}")
    else:
        require(actual == expected, f"{where}: expected {expected!r}, got {actual!r}")


def read_json(path):
    def bad_constant(value):
        raise AuditError(f"{path}: nonstandard JSON numeric literal {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=bad_constant)


def integer(row, key):
    text = row[key]
    try:
        value = int(text)
    except (ValueError, TypeError) as exc:
        raise AuditError(f"{key}: invalid integer {text!r}") from exc
    require(str(value) == text and value >= 0, f"{key}: noncanonical/nonnegative integer {text!r}")
    return value


def number(row, key, nullable=False):
    text = row[key]
    if nullable and text == "":
        return None
    try:
        value = float(text)
    except (ValueError, TypeError) as exc:
        raise AuditError(f"{key}: invalid numeric value {text!r}") from exc
    require(math.isfinite(value) and value >= 0, f"{key}: non-finite/negative value {text!r}")
    return value


def boolean(row, key, nullable=False):
    value = row[key]
    if nullable and value == "":
        return None
    require(value in ("True", "False"), f"{key}: invalid boolean/null {value!r}")
    return value == "True"


def fresh_totals():
    return {"regions": {region: {"pixels": 0, "input_sse": 0., "recovered_sse": 0.,
                                "valid_samples": 0} for region in REGIONS},
            "detector": {key: 0 for key in ("tp", "fp", "fn", "tn")},
            "placement": {"active_samples": 0, "foreground_evaluable": 0,
                "constraint_applicable": 0, "constraint_met": 0, "constraint_not_met": 0,
                "constraint_infeasible": 0, "no_foreground": 0, "coverage_sum": 0.,
                "coverage_count": 0, "zero_overlap_active": 0}}


def audit_placement(row, name, regions, condition, totals):
    prefix = name + "_placement_"
    counts = {key: integer(row, prefix + key) for key in (
        "foreground_pixels", "covered_foreground_pixels", "background_pixels",
        "covered_background_pixels", "total_pixels", "corrupt_pixels",
        "candidate_count", "eligible_candidate_count")}
    flags = {key: boolean(row, prefix + key) for key in (
        "active", "evaluable", "constraint_applicable", "constraint_applied")}
    met = boolean(row, prefix + "constraint_met", nullable=True)
    coverage = number(row, prefix + "foreground_coverage", nullable=True)
    feasible = number(row, prefix + "max_feasible", nullable=True)
    threshold = number(row, prefix + "actual_threshold")
    status = row[prefix + "status"]
    foreground = regions["foreground"]["pixels"]
    covered = regions["corrupt_foreground"]["pixels"]
    corrupt = regions["corrupt"]["pixels"]
    pixels = regions["global"]["pixels"]
    for key, value in {
        "foreground_pixels": foreground, "covered_foreground_pixels": covered,
        "background_pixels": pixels - foreground,
        "covered_background_pixels": regions["corrupt_background"]["pixels"],
        "total_pixels": pixels, "corrupt_pixels": corrupt}.items():
        same(counts[key], value, prefix + key)
    measured_coverage = ratio(covered, foreground)
    same(coverage, measured_coverage, prefix + "foreground_coverage")
    same(number(row, prefix + "mask_area_coverage"), corrupt / pixels, prefix + "mask_area_coverage")
    same(flags["evaluable"], foreground > 0, prefix + "evaluable")
    active = flags["active"]
    same(active, corrupt > 0, prefix + "active_vs_support")
    applicable = active and foreground > 0
    same(flags["constraint_applicable"], applicable, prefix + "constraint_applicable")
    same(flags["constraint_applied"], applicable and condition["policy"] == "foreground",
         prefix + "constraint_applied")
    minimum = condition["options"]["min_foreground_coverage"]
    same(met, measured_coverage >= minimum if applicable else None, prefix + "constraint_met")
    same(pixels, 784 if name == "img" else 4096, prefix + "expected_spatial_pixels")
    if name == "img":
        same(threshold, condition["options"]["img_threshold"], prefix + "actual_threshold")
    else:
        require(threshold >= condition["options"]["aud_absolute_threshold"]
                and threshold <= max(condition["options"]["aud_absolute_threshold"],
                                     condition["options"]["aud_relative_threshold"]),
                prefix + "audio threshold outside normalized-input limits")
    if not foreground:
        same(feasible, None, prefix + "max_feasible")
        same(status, "no_foreground", prefix + "status")
        same(counts["eligible_candidate_count"], 0, prefix + "eligible_candidate_count")
    elif not active:
        same(feasible, 0., prefix + "max_feasible")
        same(status, "inactive", prefix + "status")
    else:
        require(feasible is not None and coverage <= feasible + ABS_TOL and feasible <= 1.,
                prefix + "coverage exceeds maximum feasible coverage")
        expected_status = ("constraint_met" if met else "constraint_infeasible"
                           if feasible < minimum else "below_requested_coverage")
        same(status, expected_status, prefix + "status")
        require((counts["eligible_candidate_count"] > 0) == (feasible >= minimum),
                prefix + "eligible window count disagrees with feasibility")
        if condition["policy"] == "foreground":
            require(met or status == "constraint_infeasible", prefix + "feasible constraint was missed")
            if not met:
                same(coverage, feasible, prefix + "infeasible selected coverage must equal maximum")
    if active:
        side = 28 if name == "img" else 64
        block = side // 2
        rectangle = json.loads(row[prefix + "rectangle"])
        require(isinstance(rectangle, list) and len(rectangle) == 4
                and all(type(v) is int for v in rectangle), prefix + "invalid rectangle")
        top, left, height, width = rectangle
        require(0 <= top <= side - block and 0 <= left <= side - block
                and height == width == block, prefix + "rectangle area/bounds mismatch")
        same(corrupt, block * block, prefix + "rectangle_pixel_count")
        same(counts["candidate_count"], (side - block + 1) ** 2, prefix + "candidate_count")
        require(counts["eligible_candidate_count"] <= counts["candidate_count"],
                prefix + "eligible windows exceed all windows")
        p = totals["placement"]
        p["active_samples"] += 1
        p["foreground_evaluable"] += int(flags["evaluable"])
        p["constraint_applicable"] += int(applicable)
        p["constraint_met"] += int(met is True)
        p["constraint_not_met"] += int(met is False)
        p["constraint_infeasible"] += int(status == "constraint_infeasible")
        p["no_foreground"] += int(status == "no_foreground")
        p["zero_overlap_active"] += int(covered == 0)
        if coverage is not None:
            p["coverage_sum"] += coverage
            p["coverage_count"] += 1
    else:
        same(row[prefix + "rectangle"], "", prefix + "inactive rectangle")
        same(counts["candidate_count"], 0, prefix + "inactive candidate count")
        same(counts["eligible_candidate_count"], 0, prefix + "inactive eligible count")
    permitted = condition["family"] != "clean" and condition["cue_mode"] in (
        ("corrupt_both", "corrupt_img_clean_aud") if name == "img" else
        ("corrupt_both", "clean_img_corrupt_aud"))
    same(active, permitted, prefix + "mode_vs_active")


def audit_modality(row, name, condition, totals):
    regions = {}
    for region in REGIONS:
        prefix = f"{name}_{region}_"
        count = integer(row, prefix + "pixels")
        values = {"pixels": count}
        accumulated = totals["regions"][region]
        accumulated["pixels"] += count
        accumulated["valid_samples"] += int(count > 0)
        for source in ("input", "recovered"):
            sse = number(row, prefix + source + "_sse")
            mse = number(row, prefix + source + "_mse", nullable=True)
            same(mse, ratio(sse, count), prefix + source + "_mse")
            if not count:
                same(sse, 0., prefix + source + "_sse on empty region")
            values[source + "_sse"] = sse
            accumulated[source + "_sse"] += sse
        regions[region] = values
    for whole, parts in PARTITIONS:
        for key in ("pixels", "input_sse", "recovered_sse"):
            same(regions[whole][key], sum(regions[part][key] for part in parts),
                 f"{name}.{whole}.{key} partition into {parts}")
    same(regions["clean"]["input_sse"], 0., name + " clean input must be preserved")
    counts = {key: integer(row, f"{name}_detector_{key}") for key in ("tp", "fp", "fn", "tn")}
    same(sum(counts.values()), regions["global"]["pixels"], name + " confusion total")
    same(counts["tp"] + counts["fn"], regions["corrupt"]["pixels"], name + " confusion positives")
    same(counts["fp"] + counts["tn"], regions["clean"]["pixels"], name + " confusion negatives")
    for key, value in counts.items():
        totals["detector"][key] += value
    audit_placement(row, name, regions, condition, totals)


def finalized(totals):
    for r in totals["regions"].values():
        for source in ("input", "recovered"):
            r[source + "_mse"] = ratio(r[source + "_sse"], r["pixels"])
    d = totals["detector"]
    tp, fp, fn, tn = (d[key] for key in ("tp", "fp", "fn", "tn"))
    d.update(precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
             f1=ratio(2 * tp, 2 * tp + fp + fn), iou=ratio(tp, tp + fp + fn),
             false_positive_rate=ratio(fp, fp + tn))
    p = totals["placement"]
    for key, numerator, denominator in (
            ("mean_foreground_coverage", "coverage_sum", "coverage_count"),
            ("constraint_met_rate", "constraint_met", "constraint_applicable"),
            ("constraint_not_met_rate", "constraint_not_met", "constraint_applicable"),
            ("constraint_infeasible_rate", "constraint_infeasible", "constraint_applicable"),
            ("no_foreground_rate", "no_foreground", "active_samples"),
            ("zero_overlap_rate", "zero_overlap_active", "active_samples")):
        p[key] = ratio(p[numerator], p[denominator])
    return totals


def audit_condition(folder, seed, reference):
    summary = read_json(folder / "summary.json")
    marker = read_json(folder / "complete.json")
    require(marker["status"] == "complete", "Condition does not have a completed marker")
    same(summary["training_seed"], seed, "training_seed")
    same(summary["options"], OPTIONS, "fixed_options")
    same(summary["area_coverage"], .25, "fixed_area_coverage")
    same(summary["split"], "test", "formal split")
    same(summary["partial_evaluation"], False, "formal partial_evaluation")
    for key, value in marker["identity"].items():
        # The summary intentionally replaces the requested example count by its
        # concrete example artifact records.
        same(len(summary["examples"]) if key == "examples" else summary[key], value,
             "completion_identity." + key)
    require("summary.json" in marker["artifacts"] and "per_sample.csv.gz" in marker["artifacts"],
            "Completion marker lacks required artifact hashes")
    for relative, expected_hash in marker["artifacts"].items():
        path = (folder / relative).resolve()
        require(path.is_relative_to(folder.resolve()), "Artifact path escapes condition folder")
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        same(actual_hash, expected_hash, "artifact hash " + relative)
    seen, indices = set(), set()
    totals = {name: fresh_totals() for name in ("img", "aud")}
    correct_sum, count = 0, 0
    local_reference = {} if reference is None else reference
    with gzip.open(folder / "per_sample.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
                "CSV missing headers or containing duplicate column names")
        for row_number, row in enumerate(reader, 2):
            try:
                require(None not in row and None not in row.values(), "Malformed CSV row")
                sample_id, sample_index = row["sample_id"], integer(row, "sample_index")
                require(sample_id not in seen, f"Duplicate sample ID {sample_id}")
                require(sample_index not in indices, f"Duplicate sample index {sample_index}")
                seen.add(sample_id)
                indices.add(sample_index)
                label, prediction, correct = (integer(row, key) for key in
                                              ("label", "index_prediction", "correct"))
                same(correct, int(label == prediction), "correct flag")
                signature = (sample_index, label, row["image_ids"], row["audio_ids"], row["speakers"])
                if reference is None:
                    local_reference[sample_id] = signature
                else:
                    require(sample_id in reference and reference[sample_id] == signature,
                            f"Cross-policy/condition pairing differs for sample ID {sample_id}")
                for key in ("policy", "family", "cue_mode", "split"):
                    same(row[key], summary[key], "row identity " + key)
                same(integer(row, "training_seed"), seed, "row training_seed")
                for name in ("img", "aud"):
                    audit_modality(row, name, summary, totals[name])
                correct_sum += correct
                count += 1
            except (AuditError, KeyError, ValueError, TypeError) as exc:
                raise AuditError(f"CSV line {row_number}, sample {row.get('sample_id')}: {exc}") from exc
    same(count, summary["samples"], "CSV row count versus summary")
    same(count, summary["total_split_samples"], "CSV row count versus full split")
    same(count, 10000, "formal test split row count")
    require(indices == set(range(count)), "Sample indices do not span the complete split")
    require(set(local_reference) == seen, "Conditions have different sample ID sets")
    calculated = {"n": count, "index_accuracy": correct_sum / count,
                  "modalities": {name: finalized(value) for name, value in totals.items()}}
    same(summary["statistics"], calculated, "statistics independently reaggregated from CSV")
    record = {"policy": summary["policy"], "family": summary["family"],
              "cue_mode": summary["cue_mode"], "status": "passed", "rows": count,
              "unique_sample_ids": len(seen), "artifact_hashes_checked": len(marker["artifacts"]),
              "recomputed_statistics": calculated}
    return record, local_reference, summary


def audit(root):
    root = Path(root).resolve()
    require(root.is_dir(), f"Results root is not a directory: {root}")
    records, errors, missing, complete_seeds = [], [], {}, []
    audited_rows = 0
    for seed in EXPECTED_SEEDS:
        seed_folder = root / f"seed_{seed}"
        reference, passed_conditions = None, set()
        seed_index = None
        if (seed_folder / "summary.json").exists():
            try:
                index = read_json(seed_folder / "summary.json")
                same(index["seed"], seed, "seed summary identity")
                seed_index = {(c["policy"], c["family"], c["cue_mode"]): c for c in index["conditions"]}
                require(len(seed_index) == len(index["conditions"]), "Duplicate condition in seed summary")
                require(set(seed_index) == EXPECTED_CONDITIONS, "Seed summary does not list exactly 13 conditions")
            except (AuditError, KeyError, ValueError, TypeError) as exc:
                seed_index = None
                errors.append({"seed": seed, "scope": "seed_summary", "error": str(exc)})
        for policy, family, mode in sorted(EXPECTED_CONDITIONS, key=lambda c: (c[0] != "uniform", c)):
            folder = seed_folder / policy / f"{family}__{mode}"
            if not (folder / "complete.json").exists():
                continue
            identity = {"seed": seed, "policy": policy, "family": family, "cue_mode": mode}
            print(f"[audit] seed={seed} {policy} {family} {mode}", flush=True)
            try:
                record, candidate_reference, summary = audit_condition(folder, seed, reference)
                same((summary["policy"], summary["family"], summary["cue_mode"]),
                     (policy, family, mode), "directory condition identity")
                if seed_index is not None:
                    same(summary, seed_index[(policy, family, mode)], "seed summary matches condition summary")
                record["seed"] = seed
                records.append(record)
                audited_rows += record["rows"]
                passed_conditions.add((policy, family, mode))
                reference = candidate_reference
            except (AuditError, OSError, KeyError, ValueError, TypeError, EOFError) as exc:
                errors.append({**identity, "scope": "condition", "error": str(exc)})
        missing[str(seed)] = [dict(policy=p, family=f, cue_mode=m)
                              for p, f, m in sorted(EXPECTED_CONDITIONS - passed_conditions)]
        if passed_conditions == EXPECTED_CONDITIONS and seed_index is not None:
            complete_seeds.append(seed)
    full39 = not errors and len(records) == 39 and complete_seeds == list(EXPECTED_SEEDS)
    passed = bool(records) and not errors
    report = {
        "schema": "v15-foreground-numeric-integrity.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "status": "passed_full39" if full39 else "passed_available_scope" if passed else "failed",
        "numeric_checks_passed_for_audited_scope": passed,
        "full39": full39, "expected_seed_count": 3, "expected_condition_count": 39,
        "complete_seed_count": len(complete_seeds), "complete_seeds": complete_seeds,
        "audited_seeds": sorted({r["seed"] for r in records}),
        "audited_condition_count": len(records), "audited_row_count": audited_rows,
        "missing_seeds": [s for s in EXPECTED_SEEDS if not (root / f"seed_{s}").exists()],
        "missing_or_failed_conditions": missing, "errors": errors,
        "tolerance": {"relative": REL_TOL, "absolute": ABS_TOL},
        "checks": ["Complete-marker artifact SHA-256 hashes", "Unique IDs and complete sample indices",
                   "Every recorded pixel count, SSE partition, MSE denominator, and undefined value",
                   "True-region sufficient-statistic aggregation versus condition and seed summaries",
                   "Detector confusion counts and micro rates", "Foreground coverage and constraint honesty",
                   "Cross-policy/condition sample ID, label, image ID, audio ID, and speaker pairing"],
        "scope_limits": ["Reaggregates exported true-region statistics; does not rerun all model forwards or reconstruct every pixel error.",
                         "CSV nulls are empty fields; JSON undefined quantities are null.",
                         "A passed_available_scope status is not a completed three-seed evaluation."],
        "conditions": records,
    }
    target = root / "numeric_integrity.json"
    temporary = root / "numeric_integrity.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)
    print(f"[audit] {report['status']}: {len(complete_seeds)} complete seeds, "
          f"{len(records)}/39 conditions, {audited_rows} rows; {len(errors)} errors", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root containing seed_1234/seed_2345/seed_3456")
    args = parser.parse_args(argv)
    result = audit(args.root)
    return 0 if result["numeric_checks_passed_for_audited_scope"] else 1


if __name__ == "__main__":
    sys.exit(main())
