"""Synthetic smoke data only: these fixtures are not experiment results."""

import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v15_report import export_numeric_tables, plot_metric_summaries


def synthetic_smoke_fixture():
    """Two one-sample conditions; literal values keep expected exports explicit."""
    clean = {"global_input_mse": 0.0, "global_recovered_mse": 0.002,
             "corrupt_input_mse": None, "corrupt_recovered_mse": None,
             "clean_input_mse": 0.0, "clean_recovered_mse": 0.002,
             "detector": {"tp": 0, "fp": 1, "fn": 0, "tn": 9,
                          "precision": 0.0, "recall": None, "f1": 0.0,
                          "iou": 0.0, "false_positive_rate": 0.1}}
    noisy = {"global_input_mse": 0.04, "global_recovered_mse": 0.01,
             "corrupt_input_mse": 0.08, "corrupt_recovered_mse": 0.02,
             "clean_input_mse": 0.0, "clean_recovered_mse": 0.0,
             "detector": {"tp": 4, "fp": 0, "fn": 1, "tn": 5,
                          "precision": 1.0, "recall": 0.8, "f1": 8 / 9,
                          "iou": 0.8, "false_positive_rate": 0.0}}
    conditions = [dict(family="clean", cue_mode="clean_both", n=1, accuracy=1.0,
                       img=deepcopy(clean), aud=deepcopy(clean)),
                  dict(family="local_gaussian", cue_mode="corrupt_both", n=1, accuracy=1.0,
                       img=deepcopy(noisy), aud=deepcopy(noisy))]
    # This export fixture does not claim an aggregate from real model inference.
    overall = {"global_input_mse": 0.02, "global_recovered_mse": 0.006,
               "corrupt_input_mse": 0.08, "corrupt_recovered_mse": 0.02,
               "clean_input_mse": 0.0, "clean_recovered_mse": 0.02 / 15,
               "detector": {"tp": 4, "fp": 1, "fn": 1, "tn": 14,
                            "precision": 0.8, "recall": 0.8, "f1": 0.8,
                            "iou": 4 / 6, "false_positive_rate": 1 / 15}}
    report = dict(schema_version=1, split="test", mask_source="predicted", seed=1234,
                  summary=dict(n=2, accuracy=1.0, img=deepcopy(overall), aud=deepcopy(overall)),
                  conditions=conditions, fixture="synthetic smoke data")
    rows = []
    for condition in conditions:
        row = dict(sample_index=0, sample_id=17, split="test", family=condition["family"],
                   cue_mode=condition["cue_mode"], mask_source="predicted", label=7, correct=1,
                   severity=0.0 if condition["family"] == "clean" else 0.4)
        for modality in ("img", "aud"):
            for key, value in condition[modality].items():
                if key == "detector":
                    row.update({f"{modality}_detector_{name}": number for name, number in value.items()})
                else:
                    row[f"{modality}_{key}"] = value
        rows.append(row)
    return report, rows


class V15ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="v15-report-synthetic-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "result"
        self.report, self.rows = synthetic_smoke_fixture()

    def test_numeric_exports_roundtrip_and_keep_condition_order(self):
        before = deepcopy((self.report, self.rows))
        paths = export_numeric_tables(self.report, self.rows, self.output)
        self.assertEqual([path.name for path in paths],
                         ["summary.json", "per_sample.csv", "by_condition.csv"])
        self.assertEqual(json.loads(paths[0].read_text(encoding="utf-8")), self.report)
        with paths[1].open(encoding="utf-8", newline="") as stream:
            samples = list(csv.DictReader(stream))
        with paths[2].open(encoding="utf-8", newline="") as stream:
            conditions = list(csv.DictReader(stream))
        self.assertEqual([row["family"] for row in conditions], ["clean", "local_gaussian"])
        self.assertEqual(samples[0]["img_corrupt_recovered_mse"], "")
        self.assertEqual(conditions[0]["img_detector_recall"], "")
        self.assertEqual(float(conditions[1]["aud_corrupt_recovered_mse"]), 0.02)
        self.assertAlmostEqual(float(conditions[1]["img_detector_f1"]), 8 / 9)
        self.assertEqual(float(samples[0]["img_global_input_mse"]), 0.0)
        self.assertEqual((self.report, self.rows), before)

    def test_rejects_empty_rows_before_creating_output(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            export_numeric_tables(self.report, [], self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_invalid_numbers_before_creating_output(self):
        for value in (float("nan"), float("inf"), float("-inf"), -1.0):
            with self.subTest(value=value):
                rows = deepcopy(self.rows)
                rows[0]["img_global_recovered_mse"] = value
                with self.assertRaises(ValueError):
                    export_numeric_tables(self.report, rows, self.output)
                self.assertFalse(self.output.exists())
        self.report["conditions"][0]["img"]["detector"]["precision"] = 1.1
        with self.assertRaisesRegex(ValueError, "Invalid metric"):
            plot_metric_summaries(self.report, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_report_nan_and_mismatched_condition(self):
        report = deepcopy(self.report)
        report["summary"]["aud"]["global_recovered_mse"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            export_numeric_tables(report, self.rows, self.output)
        rows = deepcopy(self.rows)
        rows[0]["mask_source"] = "oracle"
        with self.assertRaisesRegex(ValueError, "differs"):
            export_numeric_tables(self.report, rows, self.output)
        self.assertFalse(self.output.exists())

    def test_diagnostic_pngs_render_with_undefined_metrics(self):
        before = deepcopy(self.report)
        paths = plot_metric_summaries(self.report, self.output)
        self.assertEqual([p.name for p in paths], ["mse_summary.png", "detector_summary.png"])
        for path in paths:
            with self.subTest(path=path.name):
                self.assertGreater(path.stat().st_size, 10_000)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(self.report, before)


if __name__ == "__main__":
    unittest.main()
