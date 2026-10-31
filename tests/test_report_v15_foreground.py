"""Synthetic report fixtures are functional tests, never experimental results."""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.report_v15_foreground import (REGIONS, aggregate_conditions, build_report,
    compare_original, condition_folder, export_aggregate_tables, sha, write_json)


def fixture_condition(seed=7, policy="uniform", family="local_gaussian", value=.1):
    modalities = {}
    for modality in ("img", "aud"):
        modalities[modality] = {
            "regions": {region: {"pixels": 16, "input_sse": 1., "recovered_sse": value * 16,
                "input_mse": .0625, "recovered_mse": value, "valid_samples": 1} for region in REGIONS},
            "detector": {"tp": 4, "fp": 0, "fn": 0, "tn": 12, "precision": 1.,
                "recall": 1., "f1": 1., "iou": 1., "false_positive_rate": 0.},
            "placement": {"active_samples": 1, "foreground_evaluable": 1,
                "constraint_applicable": 1, "constraint_met": 1, "constraint_not_met": 0,
                "constraint_infeasible": 0, "no_foreground": 0, "coverage_sum": .25,
                "coverage_count": 1, "mean_foreground_coverage": .25,
                "constraint_met_rate": 1., "constraint_not_met_rate": 0.}}
    return {"training_seed": seed, "policy": policy, "family": family, "cue_mode": "corrupt_both",
        "samples": 1, "partial_evaluation": True, "split": "test", "data_sha256": "synthetic-data",
        "original_checkpoint_sha256": "synthetic-original-checkpoint", "evaluation_seed": 99,
        "area_coverage": .25, "options": {"fixture": True}, "source_sha256": "synthetic-source",
        "runtime_sha256": "synthetic-runtime", "statistics": {"n": 1, "index_accuracy": .5, "modalities": modalities},
        "examples": [], "source_kind": "synthetic functional fixture, not experiment results"}


def add_fixture_example(root, condition):
    folder = condition_folder(root, condition)
    directory = folder / "examples"
    directory.mkdir(parents=True)
    arrays = {}
    for modality in ("img", "aud"):
        target = np.full((4, 4), .5, dtype=np.float32)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:2, :2] = 1
        cue = target.copy()
        cue[mask == 1] = .75
        for field, value in {"target": target, "cue": cue, "true_mask": mask,
            "predicted_mask": mask, "repair_mask": mask, "recovered": target}.items():
            arrays[f"{modality}_{field}"] = value
    np.savez_compressed(directory / "sample_00000.npz", **arrays)
    metadata = {"sample_index": 0, "sample_id": "synthetic-pair-not-results", "label": 2,
        "split": condition["split"],
        "index_prediction": 2, "training_seed": condition["training_seed"], "policy": condition["policy"],
        "family": condition["family"], "cue_mode": condition["cue_mode"], "severity": .25,
        "noise_std": .2, "coverage": {m: {"foreground_coverage": .25} for m in ("img", "aud")},
        "source_kind": "synthetic functional fixture, not experiment results"}
    write_json(directory / "sample_00000.json", metadata)
    condition["examples"].append({"npz": "examples/sample_00000.npz", "metadata": "examples/sample_00000.json",
                                  "sample_id": metadata["sample_id"]})
    write_json(folder / "summary.json", condition)
    identity = {key: value for key, value in condition.items() if key not in ("statistics", "examples")}
    identity["examples"] = len(condition["examples"])
    files = [folder / "summary.json", directory / "sample_00000.npz", directory / "sample_00000.json"]
    write_json(folder / "complete.json", {"status": "complete", "identity": identity,
               "artifacts": {path.relative_to(folder).as_posix(): sha(path) for path in files}})


class AggregateTests(unittest.TestCase):
    def test_csv_values_match_the_same_aggregate_and_preserve_empty_cells(self):
        conditions = []
        for seed, value in ((7, .1), (8, .2), (9, .3)):
            conditions.extend([fixture_condition(seed, "uniform", value=value),
                               fixture_condition(seed, "foreground", value=value + .05)])
        for condition in conditions:
            condition["statistics"]["modalities"]["aud"]["regions"]["corrupt_background"]["recovered_mse"] = None
        aggregate = aggregate_conditions({"conditions": conditions})
        with tempfile.TemporaryDirectory() as temporary:
            paths = export_aggregate_tables(Path(temporary), aggregate)
            for path, section in zip(paths, ("groups", "matched_policy_differences")):
                with path.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), sum(len(group["metrics"]) for group in aggregate[section]))
                for row in rows:
                    group = next(g for g in aggregate[section] if g["family"] == row["family"] and
                        g["cue_mode"] == row["cue"] and (section != "groups" or g["policy"] == row["policy"]))
                    metric = group["metrics"][row["metric_key"]]
                    for field in ("mean", "sd"):
                        self.assertEqual(None if row[field] == "" else float(row[field]), metric[field])
                    self.assertEqual(int(row["n_seeds"]), metric["n_seeds"])
                    for seed in ("7", "8", "9"):
                        self.assertEqual(None if row[f"seed_{seed}"] == "" else float(row[f"seed_{seed}"]),
                                         metric["by_seed"].get(seed))
                    if section != "groups":
                        self.assertEqual(row["policy"], "foreground_minus_uniform")
                        self.assertEqual(row["direction"], "foreground minus uniform")

    def test_uses_three_seed_sample_sd_and_matches_conditions(self):
        conditions = []
        for seed, value in ((7, .1), (8, .2), (9, .3)):
            conditions.extend([fixture_condition(seed, "uniform", value=value),
                               fixture_condition(seed, "foreground", value=value + .05),
                               fixture_condition(seed, "uniform", "clean", value=0)])
        aggregate = aggregate_conditions({"conditions": conditions, "partial_evaluation": True})
        self.assertEqual(len(aggregate["groups"]), 3)
        uniform = next(g for g in aggregate["groups"] if g["policy"] == "uniform" and g["family"] != "clean")
        metric = uniform["metrics"]["img.regions.corrupt.recovered_mse"]
        self.assertAlmostEqual(metric["mean"], .2)
        self.assertAlmostEqual(metric["sd"], .1)
        paired = aggregate["matched_policy_differences"]
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired[0]["metrics"]["img.regions.corrupt.recovered_mse"]["mean"], .05)
        self.assertTrue(aggregate["partial_evaluation"])

    def test_empty_regions_and_single_seed_have_no_invented_sd(self):
        condition = fixture_condition()
        condition["statistics"]["modalities"]["img"]["regions"]["corrupt_background"]["recovered_mse"] = None
        aggregate = aggregate_conditions({"conditions": [condition]})
        metrics = aggregate["groups"][0]["metrics"]
        self.assertIsNone(metrics["index_accuracy"]["sd"])
        self.assertEqual(metrics["index_accuracy"]["n_seeds"], 1)
        self.assertIsNone(metrics["img.regions.corrupt_background.recovered_mse"]["mean"])
        self.assertFalse(aggregate["condition_coverage"]["complete_three_seed_condition_grid"])

    def test_source_hash_changes_block_reporting_before_any_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition = fixture_condition()
            add_fixture_example(root, condition)
            write_json(root / "summary.json", {"status": "complete", "conditions": [condition]})
            metadata = condition_folder(root, condition) / condition["examples"][0]["metadata"]
            metadata.write_text(metadata.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with patch("tools.report_v15_foreground.render_examples", side_effect=AssertionError("must not render")):
                with self.assertRaisesRegex(ValueError, "artifact hash differs"):
                    build_report(root)

    def test_missing_pair_identity_does_not_become_comparable(self):
        first, second = fixture_condition(), fixture_condition(policy="foreground")
        del first["source_sha256"]
        del second["source_sha256"]
        pair = aggregate_conditions({"conditions": [first, second]})["matched_policy_differences"][0]
        self.assertFalse(pair["per_seed"][0]["comparable"])
        self.assertIn("missing source_sha256", pair["per_seed"][0]["reasons"])

    def test_missing_or_different_populations_do_not_get_paired_deltas(self):
        first, second = fixture_condition(), fixture_condition(policy="foreground")
        second["data_sha256"] = "different-synthetic-data"
        aggregate = aggregate_conditions({"conditions": [first, second]})
        pair = aggregate["matched_policy_differences"][0]
        self.assertEqual(pair["metrics"], {})
        self.assertFalse(pair["per_seed"][0]["comparable"])

    def test_original_list_schema_checks_original_checkpoint_and_partial(self):
        current = fixture_condition()
        original_condition = {"family": current["family"], "cue_mode": current["cue_mode"], "n": 1,
                              "accuracy": .4}
        for modality in ("img", "aud"):
            original_condition[modality] = {region + "_" + metric: values[metric]
                for region, values in current["statistics"]["modalities"][modality]["regions"].items()
                for metric in ("input_sse", "recovered_sse", "input_mse", "recovered_mse", "pixels")}
        original = {"conditions": [original_condition], "split": "test", "seed": 99,
                    "mask_source": "predicted", "severity": .25, "n_unique_samples": 1,
                    "checkpoint_sha256": current["original_checkpoint_sha256"]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "seed_7/evaluation"
            folder.mkdir(parents=True)
            write_json(folder / "predicted.json", original)
            report = compare_original([current], root)
            self.assertTrue(report["records"][0]["checkpoint_sha256_match"])
            self.assertFalse(report["records"][0]["comparable"])
            self.assertAlmostEqual(report["maxima"]["index_accuracy_max_absolute_difference_all_recorded"], .1)
            self.assertIsNone(report["maxima"]["index_accuracy_max_absolute_difference_comparable_only"])
            current["partial_evaluation"] = False
            second = compare_original([current], root)
            self.assertTrue(second["records"][0]["comparable"])


class ReportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit_scripts = os.environ.get("NATURE_FIGURE_AUDIT_SCRIPTS")
        if not cls.audit_scripts or not (importlib.util.find_spec("pymupdf") or importlib.util.find_spec("fitz")):
            raise unittest.SkipTest("Requires the plotting environment and nature-figure audits")

    def test_report_resumes_verified_outputs_and_rebuilds_missing_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conditions = [fixture_condition(policy=policy) for policy in ("uniform", "foreground")]
            for condition in conditions:
                add_fixture_example(root, condition)
            write_json(root / "summary.json", {"status": "complete", "conditions": conditions,
                "seeds": [7], "split": "test", "partial_evaluation": True,
                "source_kind": "synthetic functional fixture, not experiment results"})
            first = build_report(root, audit_scripts=self.audit_scripts)
            self.assertEqual(first["actions"]["rendered"], 2)
            self.assertEqual(first["figure_count"], 2)
            html = (root / "review.html").read_text(encoding="utf-8")
            self.assertIn("不是恢复内容识别率", html)
            self.assertIn("部分样本评估", html)
            self.assertEqual(html.count("class='sample'"), 1)
            self.assertEqual(html.count("<img loading='lazy'"), 2)
            self.assertNotIn("data:image", html)
            with patch("tools.report_v15_foreground.render_joint_sample", side_effect=AssertionError("Should verify, not rerender")):
                second = build_report(root, audit_scripts=self.audit_scripts)
            self.assertEqual(second["actions"]["verified"], 2)
            incomplete = root / first["figures"][0]["artifacts"]["svg"]
            incomplete.unlink()
            third = build_report(root, audit_scripts=self.audit_scripts)
            self.assertEqual(third["actions"]["rebuilt"], 1)
            self.assertEqual(third["actions"]["verified"], 1)
            self.assertTrue(incomplete.is_file())
            manifest = json.loads((root / "report_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["partial_evaluation"])
            self.assertEqual(manifest["status"], "complete")


if __name__ == "__main__":
    unittest.main()
