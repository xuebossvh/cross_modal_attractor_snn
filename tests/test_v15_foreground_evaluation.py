"""Independent numeric and leakage contracts for the supplementary CLI runner."""

from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.noise_metrics import sample_metrics
from tools.evaluate_v15_foreground import Accumulator, REGIONS, batch_metrics, run_seed, sha
from tools.v15_foreground_protocol import (
    DEFAULT_OPTIONS, foreground_mask, foreground_metrics, make_cues,
)


def config():
    return {
        "corruption": {"img_mode": "local_gaussian", "aud_mode": "local_gaussian",
                       "train_severity": .25, "noise_std": .2},
        "noise_evaluation": {"seed": 17, "severity": .25,
                             "families": ["local_gaussian"],
                             "cue_modes": ["clean_both", "corrupt_both"]},
    }


def synthetic_metrics(n=7, seed=928):
    """Include empty/all foreground and empty/full truth in both modalities."""
    rng = np.random.default_rng(seed)
    values = {}
    for modality, shape in (("img", (n, 1, 9, 8)), ("aud", (n, 10, 11))):
        target = rng.uniform(0, 1, shape).astype(np.float32)
        target[0] = 0
        target[1] = .8
        cue = rng.uniform(0, 1, shape).astype(np.float32)
        recovered = rng.uniform(0, 1, shape).astype(np.float32)
        truth = (rng.uniform(0, 1, shape) > .6).astype(np.float32)
        truth[0] = 0
        truth[1] = 1
        predicted = (rng.uniform(0, 1, shape) > .7).astype(np.float32)
        predicted[0] = 0
        values[modality] = (cue, recovered, target, truth, predicted)
    return values


def calculate(values):
    return {name: batch_metrics(*arrays, name, DEFAULT_OPTIONS)
            for name, arrays in values.items()}


def metadata_for_accumulator():
    # Positive, feasible-but-missed, infeasible, empty foreground, and inactive
    # conditions distinguish the denominator of each placement rate.
    rows = [
        dict(active=True, evaluable=True, constraint_applicable=True,
             constraint_met=True, status="constraint_met", covered_foreground_pixels=4,
             foreground_coverage=.5),
        dict(active=True, evaluable=True, constraint_applicable=True,
             constraint_met=False, status="below_requested_coverage", covered_foreground_pixels=0,
             foreground_coverage=0.),
        dict(active=True, evaluable=True, constraint_applicable=True,
             constraint_met=False, status="constraint_infeasible", covered_foreground_pixels=2,
             foreground_coverage=.25),
        dict(active=True, evaluable=False, constraint_applicable=False,
             constraint_met=None, status="no_foreground", covered_foreground_pixels=0,
             foreground_coverage=None),
        dict(active=False, evaluable=True, constraint_applicable=False,
             constraint_met=None, status="inactive", covered_foreground_pixels=0,
             foreground_coverage=0.),
        dict(active=True, evaluable=True, constraint_applicable=True,
             constraint_met=True, status="constraint_met", covered_foreground_pixels=8,
             foreground_coverage=1.),
        dict(active=False, evaluable=False, constraint_applicable=False,
             constraint_met=None, status="no_foreground", covered_foreground_pixels=0,
             foreground_coverage=None),
    ]
    return {"samples": [{"sample_id": i, "modalities": {
        "img": deepcopy(item), "aud": deepcopy(item)}} for i, item in enumerate(rows)]}


class BatchMetricsTests(unittest.TestCase):
    def assertNumericEqual(self, actual, expected):
        if expected is None:
            self.assertIsNone(actual)
        elif isinstance(expected, int):
            self.assertEqual(actual, expected)
        else:
            self.assertAlmostEqual(float(actual), float(expected), delta=1e-11)

    def test_matches_both_scalar_references_for_every_statistic(self):
        values = synthetic_metrics()
        calculated = calculate(values)
        for name, arrays in values.items():
            result = calculated[name]
            for index in range(len(arrays[0])):
                cue, recovered, target, truth, predicted = [torch.from_numpy(a[index]) for a in arrays]
                original = sample_metrics(cue, recovered, target, truth, predicted)
                foreground = foreground_metrics(cue, recovered, target, truth,
                                                foreground_mask(target, name))
                for region in REGIONS:
                    for suffix in ("pixels", "input_sse", "recovered_sse",
                                   "input_mse", "recovered_mse"):
                        key = f"{region}_{suffix}"
                        with self.subTest(modality=name, sample=index, key=key):
                            self.assertNumericEqual(result[key][index], foreground[key])
                            if key in original:
                                self.assertNumericEqual(result[key][index], original[key])
                for count in ("tp", "fp", "fn", "tn"):
                    key = f"detector_{count}"
                    self.assertEqual(result[key][index], original[key])

    def test_strict_threshold_matches_torch_at_float32_boundaries(self):
        for modality in ("img", "aud"):
            for maximum in (.1, .2, .7, 1.):
                threshold = .1 if modality == "img" else max(.02, .1 * float(np.float32(maximum)))
                boundary = np.float32(threshold)
                data = np.array([[0., np.nextafter(boundary, np.float32(-np.inf)), boundary,
                                  np.nextafter(boundary, np.float32(np.inf)), maximum]], dtype=np.float32)
                target = data[None, None] if modality == "img" else data[None]
                zeros = np.zeros_like(target)
                actual = batch_metrics(target, target, target, zeros, zeros, modality, DEFAULT_OPTIONS)
                expected = foreground_mask(torch.from_numpy(target[0]), modality)
                with self.subTest(modality=modality, maximum=maximum):
                    self.assertEqual(actual["foreground_pixels"][0], int(expected.sum()))

    def test_metric_validation_rejects_nonfinite_mismatched_and_nonbinary(self):
        arrays = synthetic_metrics()["img"]
        for index, replacement in ((0, np.full_like(arrays[0], np.nan)),
                                   (1, arrays[1][:1]),
                                   (3, np.full_like(arrays[3], .5)),
                                   (4, np.full_like(arrays[4], np.inf))):
            bad = list(arrays)
            bad[index] = replacement
            with self.subTest(index=index), self.assertRaises(ValueError):
                batch_metrics(*bad, "img", DEFAULT_OPTIONS)


class AccumulatorTests(unittest.TestCase):
    def assertNestedClose(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assertNestedClose(actual[key], expected[key])
        elif isinstance(expected, float):
            self.assertAlmostEqual(actual, expected, delta=1e-10)
        else:
            self.assertEqual(actual, expected)

    def test_batch_partition_does_not_change_aggregation(self):
        values = synthetic_metrics()
        whole_metrics = calculate(values)
        metadata = metadata_for_accumulator()
        correct = np.array([True, False, True, True, False, True, False])
        whole = Accumulator()
        whole.add(whole_metrics, metadata, correct)
        partitioned = Accumulator()
        for lo, hi in ((0, 1), (1, 4), (4, 7)):
            partial = {modality: tuple(a[lo:hi] for a in arrays)
                       for modality, arrays in values.items()}
            partitioned.add(calculate(partial), {"samples": metadata["samples"][lo:hi]}, correct[lo:hi])
        self.assertNestedClose(partitioned.finish(), whole.finish())
        summary = whole.finish()
        self.assertEqual(summary["n"], 7)
        self.assertEqual(summary["index_accuracy"], 4 / 7)
        for name, stats in summary["modalities"].items():
            for region in REGIONS:
                expected_pixels = int(np.sum(whole_metrics[name][region + "_pixels"]))
                actual = stats["regions"][region]
                self.assertEqual(actual["pixels"], expected_pixels)
                self.assertEqual(actual["valid_samples"],
                                 int(np.count_nonzero(whole_metrics[name][region + "_pixels"])))
                for kind in ("input", "recovered"):
                    expected_sse = float(np.sum(whole_metrics[name][f"{region}_{kind}_sse"]))
                    self.assertAlmostEqual(actual[kind + "_sse"], expected_sse)
                    self.assertAlmostEqual(actual[kind + "_mse"], expected_sse / expected_pixels)
        json.dumps(summary, allow_nan=False)

    def test_metadata_counts_missing_foreground_and_all_constraint_failures(self):
        accumulator = Accumulator()
        accumulator.add(calculate(synthetic_metrics()), metadata_for_accumulator(), np.ones(7, dtype=bool))
        for modality in accumulator.finish()["modalities"].values():
            placement = modality["placement"]
            self.assertEqual(placement["active_samples"], 5)
            self.assertEqual(placement["foreground_evaluable"], 4)
            self.assertEqual(placement["constraint_applicable"], 4)
            self.assertEqual(placement["constraint_met"], 2)
            self.assertEqual(placement["constraint_not_met"], 2)
            self.assertEqual(placement["constraint_met_rate"], .5)
            self.assertEqual(placement["constraint_not_met_rate"], .5)
            self.assertEqual(placement["constraint_infeasible"], 1)
            self.assertEqual(placement["no_foreground"], 1)
            self.assertEqual(placement["zero_overlap_active"], 2)
            self.assertEqual(placement["zero_overlap_rate"], .4)
            self.assertEqual(placement["coverage_count"], 4)
            self.assertEqual(placement["mean_foreground_coverage"], 1.75 / 4)

    def test_empty_regions_and_undefined_rates_stay_none_in_json(self):
        zero = np.zeros((1, 4, 4), dtype=np.float32)
        one = np.ones_like(zero)
        metrics = {name: batch_metrics(zero, one, zero, zero, zero, name, DEFAULT_OPTIONS)
                   for name in ("img", "aud")}
        metadata = {"samples": [{"modalities": {name: {
            "active": True, "evaluable": False, "constraint_applicable": False,
            "constraint_met": None, "status": "no_foreground", "covered_foreground_pixels": 0,
            "foreground_coverage": None} for name in ("img", "aud")}}]}
        accumulator = Accumulator()
        accumulator.add(metrics, metadata, np.array([False]))
        summary = accumulator.finish()
        for value in summary["modalities"].values():
            for region in ("foreground", "corrupt", "corrupt_foreground", "corrupt_background"):
                self.assertEqual(value["regions"][region]["pixels"], 0)
                self.assertEqual(value["regions"][region]["input_sse"], 0)
                self.assertIsNone(value["regions"][region]["input_mse"])
                self.assertIsNone(value["regions"][region]["recovered_mse"])
            self.assertIsNone(value["detector"]["precision"])
            self.assertIsNone(value["detector"]["recall"])
            self.assertIsNone(value["placement"]["constraint_met_rate"])
            self.assertIsNone(value["placement"]["mean_foreground_coverage"])
        json.dumps(summary, allow_nan=False)


class PredictedForwardTests(unittest.TestCase):
    def test_real_runner_never_passes_truth_and_outputs_serializable_artifacts(self):
        calls = []

        class SpyModel(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()

            def forward(self, **kwargs):
                calls.append({"source": self.noise_mask_source, "keys": set(kwargs)})
                image, audio = kwargs["x_img_cue"], kwargs["x_aud_cue"]
                return {
                    "recovered_img": image.clamp(.001, .999).logit(),
                    "recovered_aud": audio,
                    "img_predicted_mask": (image > .6).to(image.dtype),
                    "aud_predicted_mask": (audio > .6).to(audio.dtype),
                    "img_repair_mask": (image > .6).to(image.dtype),
                    "aud_repair_mask": (audio > .6).to(audio.dtype),
                    "logits": torch.ones(len(image), 1),
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((3, 1, 8, 8), dtype=np.float32)
            audio = np.zeros((3, 8, 8), dtype=np.float32)
            image[..., 2:6, 2:6] = .75
            audio[..., 2:6, 2:6] = .6
            dataset_path = root / "dataset.npz"
            np.savez_compressed(dataset_path, images=image, unique_audio=audio,
                                audio_indices=np.arange(3), sample_ids=np.arange(101, 104),
                                labels=np.zeros(3, dtype=np.int64))
            checkpoint_path = root / "checkpoint.pt"
            torch.save({"cfg": config(), "model": {}, "asset_packaging": {
                "inference_only": True, "original_checkpoint_sha256": "original"}}, checkpoint_path)
            manifest = {"seeds": {"13": {
                "checkpoint": checkpoint_path.name, "checkpoint_sha256": sha(checkpoint_path),
                "original_checkpoint_sha256": "original", "datasets": {"test": {
                    "file": dataset_path.name, "file_sha256": sha(dataset_path),
                    "total_split_samples": 3}}}}}
            args = Namespace(assets=str(root), output=str(root / "results"), device="cpu", split="test",
                             max_samples=0, batch_size=2, examples=1, policies=["uniform", "foreground"])
            with patch("tools.evaluate_v15_foreground.CrossModalSNN", SpyModel), \
                    patch("tools.evaluate_v15_foreground.verify_checkpoint_config"), \
                    redirect_stdout(io.StringIO()):
                results = run_seed(args, manifest, 13)
            self.assertEqual(len(results), 3)
            self.assertEqual(len(calls), 6)
            for call in calls:
                self.assertEqual(call["source"], "predicted")
                self.assertEqual(call["keys"], {"x_img_cue", "x_aud_cue", "training_mode"})
            self.assertTrue(list((root / "results").rglob("complete.json")))
            for artifact in (root / "results").rglob("*.json"):
                data = json.loads(artifact.read_text(encoding="utf-8"))
                json.dumps(data, allow_nan=False)
            json.dumps(results, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
