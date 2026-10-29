"""Numeric tables and diagnostic plots for one v15pro evaluation checkpoint.

This module deliberately does not import training code or modify the report.
``None`` means an undefined metric, not zero; non-finite numeric data is rejected.
Plots show pixel-pooled metrics already calculated by ``noise_metrics.py`` and
do not invent sample standard deviations, confidence intervals, or error bars.
"""

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path


MODALITIES = (("img", "Image"), ("aud", "Audio log-mel"))
REGIONS = (("global", "All positions"),
           ("corrupt", "True contaminated positions"),
           ("clean", "True clean positions"))
DETECTOR_RATES = (("precision", "Precision"), ("recall", "Recall"),
                  ("f1", "F1"), ("iou", "IoU"),
                  ("false_positive_rate", "False-positive rate"))


def _validate_json(value, location="report"):
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{location} requires nonempty string keys")
            _validate_json(item, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{location}[{index}]")
        return
    raise ValueError(f"{location} contains unsupported value {type(value).__name__}")


def _number(value, name, maximum=None, allow_none=True):
    if value is None and allow_none:
        return
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid metric {name}: {value!r}")


def _validate_summary(summary, location):
    if not isinstance(summary, Mapping):
        raise ValueError(f"{location} must be an object")
    _number(summary.get("n"), f"{location}.n", allow_none=False)
    if summary["n"] < 1 or int(summary["n"]) != summary["n"]:
        raise ValueError(f"{location}.n must be a positive integer")
    _number(summary.get("accuracy"), f"{location}.accuracy", maximum=1)
    for name, _ in MODALITIES:
        if not isinstance(summary.get(name), Mapping):
            raise ValueError(f"{location}.{name} must be an object")
        values = summary[name]
        for metric, value in values.items():
            if metric.endswith(("_pixels", "_valid_samples", "_sse")):
                _number(value, f"{location}.{name}.{metric}", allow_none=False)
        for region, _ in REGIONS:
            for source in ("input", "recovered"):
                metric = f"{region}_{source}_mse"
                if metric not in values:
                    raise ValueError(f"Missing metric {location}.{name}.{metric}")
                _number(values[metric], f"{location}.{name}.{metric}")
        detector = values.get("detector")
        if not isinstance(detector, Mapping):
            raise ValueError(f"{location}.{name}.detector must be an object")
        for rate, _ in DETECTOR_RATES:
            if rate not in detector:
                raise ValueError(f"Missing metric {location}.{name}.detector.{rate}")
            _number(detector[rate], f"{location}.{name}.detector.{rate}", maximum=1)
        for count in ("tp", "fp", "fn", "tn"):
            if count in detector:
                _number(detector[count], f"{location}.{name}.detector.{count}", allow_none=False)


def _validate_report(report):
    if not isinstance(report, Mapping):
        raise ValueError("report must be an object")
    _validate_json(report)
    if report.get("mask_source") not in ("predicted", "oracle", "all"):
        raise ValueError("report.mask_source must be predicted, oracle, or all")
    if report.get("split") not in ("val", "test"):
        raise ValueError("report.split must be val or test")
    conditions = report.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("report.conditions must be a nonempty list")
    _validate_summary(report.get("summary"), "report.summary")
    keys = set()
    for index, condition in enumerate(conditions):
        location = f"report.conditions[{index}]"
        _validate_summary(condition, location)
        key = (condition.get("family"), condition.get("cue_mode"))
        if any(not isinstance(item, str) or not item for item in key):
            raise ValueError(f"{location} needs family and cue_mode")
        if key in keys:
            raise ValueError(f"Duplicate condition {key}")
        keys.add(key)
    return conditions


def _flatten(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, Mapping):
            nested = _flatten(value, name)
            if result.keys() & nested.keys():
                raise ValueError(f"Colliding flattened column at {name}")
            result.update(nested)
        elif isinstance(value, (list, tuple)):
            raise ValueError(f"CSV column {name} cannot contain an array")
        else:
            if name in result:
                raise ValueError(f"Colliding flattened column at {name}")
            result[name] = value
    return result


def _fieldnames(rows, first):
    keys = {key for row in rows for key in row}
    return [key for key in first if key in keys] + sorted(keys - set(first))


def _write_csv(path, rows, first):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_fieldnames(rows, first))
        writer.writeheader()
        writer.writerows(rows)  # csv writes None as an empty cell, never 0.


def export_numeric_tables(report, rows, output_dir: Path) -> list[Path]:
    """Export exact numbers, preserving undefined metrics as null / blank cells.

    Conditions and samples retain their input order; metric column names are
    sorted to make CSV schemas independent of dictionary insertion order.
    Validation happens before any output is created.
    """
    conditions = _validate_report(report)
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("rows must be a nonempty sequence")
    if len(rows) != report["summary"]["n"]:
        raise ValueError("rows length differs from report.summary.n")
    _validate_json(rows, "rows")
    condition_keys = {(c["family"], c["cue_mode"]) for c in conditions}
    row_keys = None
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"rows[{index}] must be an object")
        if (row.get("family"), row.get("cue_mode")) not in condition_keys:
            raise ValueError(f"rows[{index}] does not match report conditions")
        if row.get("split") != report["split"] or row.get("mask_source") != report["mask_source"]:
            raise ValueError(f"rows[{index}] split/mask_source differs from report")
        if row_keys is None:
            row_keys = set(row)
        elif set(row) != row_keys:
            raise ValueError(f"rows[{index}] has inconsistent columns")
        required_metrics = {
            f"{modality}_{region}_{source}_mse"
            for modality, _ in MODALITIES for region, _ in REGIONS
            for source in ("input", "recovered")
        } | {f"{modality}_detector_{rate}"
             for modality, _ in MODALITIES for rate, _ in DETECTOR_RATES}
        if not required_metrics.issubset(row):
            raise ValueError(f"rows[{index}] is missing required metrics")
        for key, value in row.items():
            if isinstance(value, (Mapping, list, tuple)):
                raise ValueError(f"rows[{index}].{key} must be scalar")
            if key.endswith(("_mse", "_sse", "_pixels", "_tp", "_fp", "_fn", "_tn")):
                _number(value, f"rows[{index}].{key}")
            if any(key.endswith(f"_detector_{rate}") for rate, _ in DETECTOR_RATES):
                _number(value, f"rows[{index}].{key}", maximum=1)
    flat_conditions = [dict(split=report["split"], mask_source=report["mask_source"],
                            **_flatten(condition)) for condition in conditions]
    directory = Path(output_dir) / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / "summary.json", directory / "per_sample.csv",
             directory / "by_condition.csv"]
    paths[0].write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(paths[1], rows, ("sample_index", "sample_id", "split", "family", "cue_mode",
                              "mask_source", "severity", "label", "correct"))
    _write_csv(paths[2], flat_conditions, ("split", "mask_source", "family", "cue_mode",
                                         "n", "accuracy"))
    return paths


def _condition_key(conditions):
    return "\n".join(f"C{i + 1}: {c['family']} | {c['cue_mode']}"
                     for i, c in enumerate(conditions))


def _missing_label(axis, x, y, text="N/A", color="#555555"):
    axis.text(x, y, text, ha="center", va="bottom", fontsize=7, color=color,
              rotation=90, transform=axis.get_xaxis_transform())


def plot_metric_summaries(report, output_dir: Path) -> list[Path]:
    """Create single-checkpoint diagnostic plots (no invented uncertainty)."""
    conditions = _validate_report(report)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    count = len(conditions)
    x = np.arange(count)
    labels = [f"C{i + 1}" for i in range(count)]
    context = f"mask_source={report['mask_source']} | split={report['split']} | single checkpoint"
    key = _condition_key(conditions)
    footer_height = 0.20 + 0.17 * count
    directory = Path(output_dir) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / "mse_summary.png", directory / "detector_summary.png"]
    colors = ("#7F8C8D", "#2474B5")

    height = 9.0 + footer_height
    fig, axes = plt.subplots(3, 2, figsize=(max(12, count * 1.5), height), squeeze=False)
    try:
        for row, (region, region_label) in enumerate(REGIONS):
            for col, (modality, modality_label) in enumerate(MODALITIES):
                axis = axes[row, col]
                for source_index, (source, title) in enumerate((("input", "Input cue"),
                                                              ("recovered", "Recovered"))):
                    values = [c[modality][f"{region}_{source}_mse"] for c in conditions]
                    positions = x + (-0.19 if source_index == 0 else 0.19)
                    axis.bar(positions, [np.nan if v is None else v for v in values],
                             width=0.36, color=colors[source_index], label=title)
                    for position, value in zip(positions, values):
                        if value is None:
                            _missing_label(axis, position, 0.02, color=colors[source_index])
                axis.set_title(f"{modality_label} — {region_label}", fontsize=11)
                axis.set_ylabel("MSE")
                axis.set_xticks(x, labels)
                axis.set_xlim(-0.6, count - 0.4)
                axis.set_ylim(bottom=0)
                axis.grid(axis="y", alpha=0.2)
                axis.set_axisbelow(True)
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axes[0, 0].legend(loc="upper right", fontsize=9)
        fig.suptitle(f"Input versus recovery MSE\n{context}", fontsize=14)
        fig.text(0.04, 0.02, key, fontsize=8, va="bottom", family="monospace")
        fig.text(0.98, 0.02, "Regions use true contamination support.\nN/A = empty region; not zero.\n"
                 "Pixel-pooled means; no error bars.", fontsize=8, va="bottom", ha="right")
        fig.tight_layout(rect=(0, footer_height / height, 1, 0.93))
        fig.savefig(paths[0], dpi=160, facecolor="white")
    finally:
        plt.close(fig)

    height = 6.2 + footer_height
    fig, axes = plt.subplots(2, 5, figsize=(17, height), squeeze=False)
    try:
        for row, (modality, modality_label) in enumerate(MODALITIES):
            for col, (rate, rate_label) in enumerate(DETECTOR_RATES):
                axis = axes[row, col]
                values = [c[modality]["detector"][rate] for c in conditions]
                axis.plot(x, [np.nan if v is None else v for v in values],
                          marker="o", markersize=4, linestyle="none", color="#2474B5")
                for position, value in zip(x, values):
                    if value is None:
                        _missing_label(axis, position, 0.03)
                axis.set_ylim(0, 1)
                axis.set_xlim(-0.5, count - 0.5)
                axis.set_xticks(x, labels, fontsize=8)
                axis.set_title(f"{modality_label}\n{rate_label}", fontsize=10)
                axis.set_ylabel("Rate")
                axis.grid(alpha=0.2)
        fig.suptitle(f"Detector quality\n{context}", fontsize=14)
        fig.text(0.04, 0.02, key, fontsize=8, va="bottom", family="monospace")
        fig.text(0.98, 0.02, "Rates use summed confusion counts.\nN/A = undefined denominator; not zero.\n"
                 "No error bars. FPR: lower is better.", fontsize=8, va="bottom", ha="right")
        fig.tight_layout(rect=(0, footer_height / height, 1, 0.92))
        fig.savefig(paths[1], dpi=160, facecolor="white")
    finally:
        plt.close(fig)
    return paths
