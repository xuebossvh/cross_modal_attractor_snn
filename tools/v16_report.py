"""Tables and diagnostic plots for v16pro; never select examples by success."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def task_summary(report):
    """Macro over modes after averaging their families; target types stay apart."""
    buckets = defaultdict(lambda: defaultdict(list))
    for row in report["conditions"]:
        if row["sweep"] != "main" or row["protocol"] != "fixed":
            continue
        for modality in ("img", "aud"):
            state = row[modality + "_state"]
            task, region, source = {"C": ("clean_decoder", "global", "coarse"),
                                    "P": ("instance_completion", "missing", "recovered"),
                                    "N": ("instance_denoising", "noise", "recovered"),
                                    "A": ("category_readout", "global", "recovered")}[state]
            metrics = {"coarse_mse" if state == "C" else "mse": row[f"{modality}_{region}_{source}_mse"],
                       "index_acc": row["index_acc"],
                       "final_content_acc" if state == "C" else "content_acc": row.get(modality + "_recovered_content_acc")}
            if state != "C":
                for baseline in ("hard_template", "soft_template", "oracle_template"):
                    metrics[baseline + "_mse"] = row.get(f"{modality}_{region}_{baseline}_mse")
            for metric, value in metrics.items():
                if value is not None:
                    buckets[(modality, task, metric)][row["mode"]].append(value)
    return [dict(modality=modality, task=task, metric=metric,
                 value=statistics.mean(statistics.mean(v) for v in modes.values()),
                 n_modes=len(modes), aggregation="equal modes, equal families within mode")
            for (modality, task, metric), modes in sorted(buckets.items())]


def save_examples(captured, output):
    output = Path(output)
    records, artifacts = [], []
    columns = ("Clean target", "Observed cue", "Known missing", "True noise\n(scoring only)",
               "Predicted noise", "Applied repair", "Recovered", "Absolute error")
    for group, examples in captured.items():
        first = examples[0]
        name = first["modality"]
        digest = hashlib.sha256(group.encode()).hexdigest()[:12]
        for start in range(0, len(examples), 4):
            page = examples[start:start + 4]
            fig, axes = plt.subplots(len(page), len(columns), figsize=(19, 2.6 * len(page)), squeeze=False)
            path = output / "figures" / f"examples_{name}_{digest}_{start // 4 + 1}.png"
            try:
                for r, item in enumerate(page):
                    values = (item["target"], item["cue"], item["missing"], item["noise"],
                              item["predicted"], item["repair"], item["recovered"],
                              (item["recovered"] - item["target"]).abs())
                    for c, value in enumerate(values):
                        axis = axes[r, c]
                        if value is None:
                            axis.text(.5, .5, "ABSENT / N.A.", ha="center", va="center", transform=axis.transAxes, fontsize=8)
                        else:
                            plane = value.squeeze().numpy()
                            cmap = "inferno" if c == 7 else ("gray" if name == "img" or c in (2, 3, 4, 5) else "magma")
                            axis.imshow(plane, origin="lower" if name == "aud" else "upper",
                                        cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                        axis.set_xticks([])
                        axis.set_yticks([])
                        if r == 0:
                            axis.set_title(columns[c], fontsize=9)
                        if c == 0:
                            axis.set_ylabel(f"id={item['sample_id']}  y={item['label']}\n"
                                            f"Index={item['index_prediction']}\n"
                                            f"Content correct={item['content_prediction_correct']}\n"
                                            f"target={item['target_kind']}", fontsize=8)
                    records.append(dict(condition_id=item["condition"]["id"], modality=name,
                                        sample_id=item["sample_id"], sample_index=item["sample_index"],
                                        label=item["label"], index_prediction=item["index_prediction"],
                                        content_correct=item["content_prediction_correct"],
                                        target_kind=item["target_kind"], state=item["state"],
                                        figure=path.relative_to(output).as_posix(),
                                        selection="first indices; no success filter"))
                condition = first["condition"]
                fig.suptitle(f"{name} | {condition['mode']} | {condition['img_family']}/{condition['aud_family']} | "
                             "first unfiltered samples", fontsize=11)
                fig.tight_layout(rect=(0, 0, 1, .96))
                fig.savefig(path, dpi=130)
            finally:
                plt.close(fig)
            artifacts.append(path)
    artifacts.append(write_json(output / "tables/examples_manifest.json", records))
    return artifacts


def plot_conditions(report, output):
    output = Path(output)
    rows = [r for r in report["conditions"] if r["sweep"] == "main" and r["protocol"] == "fixed"]
    labels = [r["mode"] + "\n" + (r["img_family"] or "clean") + "/" + (r["aud_family"] or "clean") for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(18, max(9, len(rows) * .38)), sharey=True)
    y = np.arange(len(rows))
    try:
        axes[0].barh(y, [r["index_acc"] for r in rows], color="#426f9b")
        axes[0].set_xlim(0, 1)
        axes[0].set_title("Index classification accuracy")
        for axis, modality in zip(axes[1:], ("img", "aud")):
            values, colors = [], []
            for row in rows:
                state = row[modality + "_state"]
                region, source, color = {"C": ("global", "coarse", "#999999"),
                                        "P": ("missing", "recovered", "#d58d38"),
                                        "N": ("noise", "recovered", "#ab5473"),
                                        "A": ("global", "recovered", "#3c927d")}[state]
                value = row[f"{modality}_{region}_{source}_mse"]
                values.append(np.nan if value is None else value)
                colors.append(color)
            axis.barh(y, values, color=colors)
            axis.set_title(modality + " | task-specific MSE")
            axis.set_xlabel("C: coarse global; P: missing; N: noise; A: category global")
        axes[0].set_yticks(y, labels, fontsize=7)
        axes[0].invert_yaxis()
        fig.tight_layout()
        path = output / "figures/classification_and_recovery.png"
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)
    artifacts = [path]
    if any(row["sweep"] != "main" for row in report["conditions"]):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        for column, (sweep, factor) in enumerate((("missing_severity", "severity"), ("noise_coverage", "coverage"), ("gaussian_std", "noise_std"))):
            for row_index, modality in enumerate(("img", "aud")):
                axis = axes[row_index, column]
                groups = defaultdict(list)
                for row in report["conditions"]:
                    if row["sweep"] == sweep and row[modality + "_state"] in ("P", "N"):
                        groups[(row["mode"], row[modality + "_family"] if modality + "_family" in row else row[modality + "_state"], row["img_family"], row["aud_family"])].append(row)
                for key, values in groups.items():
                    values.sort(key=lambda row: row[factor])
                    region = "missing" if sweep == "missing_severity" else "noise"
                    axis.plot([row[factor] for row in values],
                              [row[f"{modality}_{region}_recovered_mse"] for row in values], marker="o",
                              label=key[0] + " / " + str(key[2] if modality == "img" else key[3]))
                axis.set_title(modality + " | " + sweep)
                axis.set_xlabel(factor)
                axis.set_ylabel("True affected-region MSE")
                axis.legend(fontsize=6)
        fig.tight_layout()
        path = output / "figures/strength_curves.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        artifacts.append(path)
    return artifacts


def numeric_tables(report, output):
    output = Path(output)
    return [write_json(output / "tables/summary.json", report),
            write_csv(output / "tables/by_condition.csv", report["conditions"]),
            write_csv(output / "tables/task_summary.csv", task_summary(report))]


def aggregate_reports(reports, output):
    """Seed is the statistical unit; never treat thousands of samples as seeds."""
    output = Path(output)
    buckets, task_buckets = defaultdict(list), defaultdict(list)
    per_seed = []
    metadata = {}
    for seed, model_name, report in reports:
        for row in report["conditions"]:
            metadata[(model_name, row["id"])] = {key: row[key] for key in
                ("mode", "protocol", "sweep", "img_family", "aud_family", "severity", "coverage", "noise_std",
                 "img_state", "aud_state", "img_target_kind", "aud_target_kind")}
            for metric, value in row.items():
                if value is not None and (metric.endswith(("_mse", "_acc", "_f1", "_iou", "_precision", "_recall", "_false_positive_rate", "_fraction"))):
                    buckets[(model_name, row["id"], metric)].append((seed, float(value)))
        for row in task_summary(report):
            per_seed.append(dict(seed=seed, model=model_name, **row))
            task_buckets[(model_name, row["modality"], row["task"], row["metric"])].append((seed, row["value"]))
    def stats(values):
        seeds = [seed for seed, _ in values]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seed in aggregate")
        numbers = [value for _, value in values]
        return dict(n_seeds=len(numbers), seeds=";".join(map(str, seeds)), mean=statistics.mean(numbers),
                    sample_sd=statistics.stdev(numbers) if len(numbers) > 1 else None)
    rows = [dict(model=model, condition_id=condition_id, metric=metric,
                 **metadata[(model, condition_id)], **stats(values))
            for (model, condition_id, metric), values in sorted(buckets.items())]
    tasks = [dict(model=model, modality=modality, task=task, metric=metric, **stats(values))
             for (model, modality, task, metric), values in sorted(task_buckets.items())]
    paired = []
    for baseline in ("no_value", "no_recurrence"):
        for key, main in task_buckets.items():
            model, modality, task, metric = key
            if model != "main":
                continue
            control = dict(task_buckets.get((baseline, modality, task, metric), []))
            differences = [(seed, value - control[seed]) for seed, value in main if seed in control]
            if differences:
                paired.append(dict(comparison="main minus " + baseline, modality=modality,
                                   task=task, metric=metric, **stats(differences)))
    artifacts = [write_csv(output / "tables/across_seeds.csv", rows),
                 write_csv(output / "tables/task_summary_by_seed.csv", per_seed),
                 write_csv(output / "tables/task_summary_mean_sd.csv", tasks),
                 write_csv(output / "tables/paired_ablation_differences.csv", paired)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
    names = ("main", "no_value", "no_recurrence")
    for r, modality in enumerate(("img", "aud")):
        for c, task in enumerate(("instance_completion", "instance_denoising", "category_readout")):
            axis = axes[r, c]
            selected = {row["model"]: row for row in tasks if row["modality"] == modality and row["task"] == task and row["metric"] == "mse"}
            available = [name for name in names if name in selected]
            axis.bar(available, [selected[name]["mean"] for name in available],
                     yerr=[selected[name]["sample_sd"] or 0 for name in available],
                     capsize=4, color=["#426f9b", "#d58d38", "#3c927d"][:len(available)])
            axis.set_title(modality + " | " + task)
            axis.set_ylabel("MSE; seed mean and sample SD")
            axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = output / "figures/mechanism_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts.append(path)
    return artifacts
