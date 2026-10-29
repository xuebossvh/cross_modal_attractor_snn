"""Meaningful contracts for local-noise reporting, leakage and reproducibility."""

import copy
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_v15pro import verify_checkpoint_config, load_strict_checkpoint, write_report
from scripts.noise_metrics import (detection_rates, deterministic_cues, evaluate_local_noise,
                                   sample_metrics, summarize_rows, validate_local_noise,
                                   validation_score)


def config():
    return {"noise_detector": {"enabled": True, "threshold": 0.5, "hidden_channels": 8},
            "corruption": {"img_mode": "local_gaussian", "aud_mode": "local_gaussian",
                           "train_severity": 0.25, "noise_std": 0.2},
            "noise_evaluation": {"seed": 123, "families": ["local_gaussian", "local_impulse"]},
            "validation": {"split": "val", "score": {"lambda_img": 1, "lambda_aud": 4,
                                                        "region_weight": 1}}}


class FakeDetectorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))
        self.noise_mask_source = "predicted"
        self.calls = []

    def forward(self, x_img_cue, x_aud_cue, training_mode,
                img_cue_mask=None, aud_cue_mask=None):
        self.calls.append((self.noise_mask_source, img_cue_mask, aud_cue_mask))
        img_pred = (x_img_cue > 0.6).float()
        aud_pred = (x_aud_cue > 0.6).float()
        return {"recovered_img": torch.zeros_like(x_img_cue),
                "recovered_aud": torch.full_like(x_aud_cue, 0.5),
                "logits": torch.zeros(len(x_img_cue), 10, device=x_img_cue.device),
                "img_predicted_mask": img_pred, "aud_predicted_mask": aud_pred,
                "img_noise_logits": img_pred * 2 - 1, "aud_noise_logits": aud_pred * 2 - 1}


def loader(batch_size=2, n=3):
    image = torch.linspace(0.1, 0.9, n * 64).reshape(n, 1, 8, 8)
    audio = torch.linspace(0.9, 0.1, n * 64).reshape(n, 8, 8)
    return DataLoader(TensorDataset(image, audio, torch.zeros(n, dtype=torch.long),
                                   torch.arange(n) + 101), batch_size=batch_size)


class NoiseMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_region_mse_uses_truth_not_predicted_region(self):
        clean = torch.tensor([0., 0., 0., 0.])
        cue = torch.tensor([1., 2., 0., 0.])
        rec = torch.tensor([0., 1., 2., 0.])
        truth = torch.tensor([1., 1., 0., 0.])
        predicted = torch.tensor([1., 0., 1., 0.])
        result = sample_metrics(cue, rec, clean, truth, predicted)
        self.assertEqual(result["corrupt_input_mse"], 2.5)
        self.assertEqual(result["corrupt_recovered_mse"], 0.5)
        self.assertEqual(result["clean_recovered_mse"], 2.0)
        self.assertEqual(result["global_recovered_mse"], 1.25)
        for key in ("tp", "fp", "fn", "tn"):
            self.assertEqual(result[f"detector_{key}"], 1)
        self.assertEqual(result["detector_precision"], 0.5)
        self.assertEqual(result["detector_recall"], 0.5)
        self.assertAlmostEqual(result["detector_iou"], 1 / 3)

    def test_empty_region_and_undefined_detection_are_none(self):
        value = torch.zeros(4)
        metrics = sample_metrics(value, value, value, None, value)
        self.assertIsNone(metrics["corrupt_recovered_mse"])
        self.assertIsNone(metrics["corrupt_input_mse"])
        self.assertIsNone(metrics["detector_recall"])
        self.assertIsNone(metrics["detector_precision"])
        self.assertIsNone(metrics["detector_f1"])
        self.assertEqual(metrics["detector_false_positive_rate"], 0.0)
        self.assertIsNone(detection_rates(4, 0, 0, 0)["false_positive_rate"])

    def test_summary_pools_error_and_true_region_pixel_counts(self):
        rows = []
        target = torch.zeros(4)
        for mask, recovered in ((torch.tensor([1., 0., 0., 0.]), torch.ones(4)),
                                (torch.ones(4), torch.full((4,), 2.))):
            values = sample_metrics(target, recovered, target, mask, torch.zeros(4))
            row = {"correct": 1}
            for modality in ("img", "aud"):
                row.update({f"{modality}_{key}": value for key, value in values.items()})
            rows.append(row)
        result = summarize_rows(rows)
        self.assertEqual(result["img"]["corrupt_pixels"], 5)
        self.assertEqual(result["img"]["corrupt_recovered_mse"], 17 / 5)
        self.assertEqual(result["img"]["detector"]["fn"], 5)
        self.assertEqual(result["img"]["detector"]["recall"], 0)

    def test_rejects_broadcast_masks_and_nonfinite_predictions(self):
        value = torch.zeros(2, 2)
        with self.assertRaises(ValueError):
            sample_metrics(value, value, value, torch.zeros(2), value)
        with self.assertRaises(ValueError):
            sample_metrics(value, value + float("nan"), value, value, value)
        with self.assertRaises(ValueError):
            sample_metrics(value, value, value, value, None)

    def test_batch_size_independent_corruption_and_report(self):
        model1, model2 = FakeDetectorModel(), FakeDetectorModel()
        report1, rows1 = evaluate_local_noise(model1, loader(1), config(), torch.device("cpu"))
        report2, rows2 = evaluate_local_noise(model2, loader(3), config(), torch.device("cpu"))
        order = lambda row: (row["sample_id"], row["family"], row["cue_mode"])
        self.assertEqual(sorted(rows1, key=order), sorted(rows2, key=order))
        self.assertEqual(report1["n_unique_samples"], 3)
        self.assertEqual(report1["summary"]["n"], 3 * 7)
        for name in ("img", "aud"):
            for key, value in report1["summary"][name].items():
                other = report2["summary"][name][key]
                if isinstance(value, float):
                    self.assertAlmostEqual(value, other, places=14)
                else:
                    self.assertEqual(value, other)
        self.assertEqual(report1["severity"], 0.25)

    def test_corrupted_modality_is_identical_across_cue_modes(self):
        image, audio, _, ids = next(iter(loader(3)))
        for family in ("local_gaussian", "local_impulse"):
            both = deterministic_cues(image, audio, ids.tolist(), "corrupt_both", family,
                                      config(), 0.25, 42)
            image_only = deterministic_cues(image, audio, ids.tolist(), "corrupt_img_clean_aud",
                                            family, config(), 0.25, 42)
            audio_only = deterministic_cues(image, audio, ids.tolist(), "clean_img_corrupt_aud",
                                            family, config(), 0.25, 42)
            self.assertTrue(torch.equal(both[0], image_only[0]))
            self.assertTrue(torch.equal(both[1], audio_only[1]))
            self.assertTrue(torch.equal(both[2]["aud"], audio_only[2]["aud"]))

    def test_truth_never_enters_predicted_or_all_forward(self):
        model = FakeDetectorModel()
        for mode in ("predicted", "all"):
            model.calls.clear()
            evaluate_local_noise(model, loader(), config(), torch.device("cpu"),
                                 mask_source=mode, max_samples=1)
            self.assertTrue(all(img is None and aud is None for _, img, aud in model.calls))
        model.calls.clear()
        evaluate_local_noise(model, loader(), config(), torch.device("cpu"),
                             mask_source="oracle", max_samples=1)
        self.assertTrue(all(img is not None and aud is not None for _, img, aud in model.calls))
        self.assertEqual(model.noise_mask_source, "predicted")

    def test_preserves_training_mode_and_all_random_streams(self):
        model = FakeDetectorModel()
        model.train()
        random.seed(10)
        np.random.seed(10)
        torch.manual_seed(10)
        expected = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        evaluate_local_noise(model, loader(), config(), torch.device("cpu"), max_samples=1)
        self.assertTrue(model.training)
        self.assertEqual(random.getstate(), expected[0])
        np.testing.assert_equal(np.random.get_state(), expected[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), expected[2]))

    def test_limits_and_validation_score(self):
        model = FakeDetectorModel()
        cfg = config()
        cfg["validation"]["max_batches"] = 1
        result = validate_local_noise(model, loader(2), cfg, torch.device("cpu"))
        self.assertEqual(result["n"], 14)
        expected = result["img_mse"] + result["img_masked_mse"]
        expected += 4 * (result["aud_mse"] + result["aud_masked_mse"])
        self.assertAlmostEqual(result["score"], expected)
        cfg["validation"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation split"):
            validate_local_noise(model, loader(), cfg, torch.device("cpu"))
        clean_summary = {name: {"global_recovered_mse": 0.1, "corrupt_recovered_mse": None}
                         for name in ("img", "aud")}
        self.assertAlmostEqual(validation_score(clean_summary, config()), 0.5)

    def test_empty_loader_and_unsupported_modes_fail(self):
        with self.assertRaisesRegex(ValueError, "Empty evaluation"):
            evaluate_local_noise(FakeDetectorModel(), [], config(), torch.device("cpu"))
        cfg = config()
        cfg["noise_evaluation"]["cue_modes"] = ["corrupt_aud_only"]
        with self.assertRaises(ValueError):
            evaluate_local_noise(FakeDetectorModel(), loader(), cfg, torch.device("cpu"))

    def test_checkpoint_config_and_state_are_strict(self):
        cfg = config()
        cfg["snn"] = {"T": 20, "beta": 0.9}
        saved = copy.deepcopy(cfg)
        verify_checkpoint_config(cfg, saved)
        for section, key, changed in (("snn", "T", 30), ("noise_detector", "threshold", 0.7),
                                      ("corruption", "noise_std", 0.3)):
            other = copy.deepcopy(saved)
            other[section][key] = changed
            with self.subTest(section=section), self.assertRaises(ValueError):
                verify_checkpoint_config(cfg, other)
        saved["noise_detector"]["mask_source"] = "oracle"
        verify_checkpoint_config(cfg, saved)
        model = FakeDetectorModel()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save({"model": model.state_dict(), "cfg": saved}, path)
            load_strict_checkpoint(model, path, cfg, torch.device("cpu"))
            torch.save({"model": {}, "cfg": saved}, path)
            with self.assertRaises(RuntimeError):
                load_strict_checkpoint(model, path, cfg, torch.device("cpu"))

    def test_json_and_csv_keep_empty_regions_unavailable(self):
        cfg = config()
        cfg["noise_evaluation"]["cue_modes"] = ["clean_both"]
        report, rows = evaluate_local_noise(FakeDetectorModel(), loader(), cfg,
                                            torch.device("cpu"), max_samples=1)
        with tempfile.TemporaryDirectory() as folder:
            path, csv_path = write_report(report, rows, Path(folder) / "result.json")
            actual = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(actual["summary"]["img"]["corrupt_recovered_mse"])
            with csv_path.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["img_corrupt_recovered_mse"], "")


if __name__ == "__main__":
    unittest.main()
