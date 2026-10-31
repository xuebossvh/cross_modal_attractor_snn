"""Contracts for the independent foreground evaluation sampler and metrics."""

import random
import unittest

import numpy as np
import torch

from scripts.noise_metrics import (
    _sample_seed, _seed_corruption, deterministic_cues, preserve_random_state,
)
from tools.v15_foreground_protocol import foreground_mask, foreground_metrics, make_cues


def config(**changes):
    corruption = {"img_mode": "local_gaussian", "aud_mode": "local_gaussian",
                  "train_severity": 0.25, "noise_std": 0.2}
    corruption.update(changes)
    return {"corruption": corruption}


def targets(n=3):
    img, aud = torch.zeros(n, 1, 28, 28), torch.zeros(n, 64, 64)
    img[..., 9:18, 9:18] = 0.75
    aud[..., 20:39, 20:39] = 0.6
    return img, aud


def cues(img=None, aud=None, **changes):
    if img is None:
        img, aud = targets()
    arguments = dict(img=img, aud=aud, ids=list(range(101, 101 + len(img))),
                     mode="corrupt_both", family="local_gaussian", cfg=config(),
                     severity=0.25, seed=13, policy="foreground")
    arguments.update(changes)
    return make_cues(**arguments)


class ForegroundProtocolTests(unittest.TestCase):
    def test_uniform_exactly_reuses_old_cues(self):
        img, aud = targets()
        for family in ("local_gaussian", "local_impulse"):
            for mode in ("clean_both", "corrupt_both", "clean_img_corrupt_aud",
                         "corrupt_img_clean_aud"):
                with self.subTest(family=family, mode=mode):
                    with preserve_random_state():
                        expected = deterministic_cues(img, aud, [101, 102, 103], mode,
                                                      family, config(), .25, 13)
                    actual = cues(img, aud, family=family, mode=mode, policy="uniform")
                    self.assertTrue(torch.equal(actual[0], expected[0]))
                    self.assertTrue(torch.equal(actual[1], expected[1]))
                    for name in ("img", "aud"):
                        self.assertTrue(torch.equal(actual[2][name], expected[2][name]))

    def test_deterministic_order_batch_and_mode_independence(self):
        img, aud = targets()
        first = cues(img, aud)
        repeated = cues(img, aud)
        reverse = cues(img.flip(0), aud.flip(0), ids=[103, 102, 101])
        for i, name in enumerate(("img", "aud")):
            self.assertTrue(torch.equal(first[i], repeated[i]))
            self.assertTrue(torch.equal(first[i], reverse[i].flip(0)))
            self.assertTrue(torch.equal(first[2][name], repeated[2][name]))
            self.assertTrue(torch.equal(first[2][name], reverse[2][name].flip(0)))
        self.assertEqual(first[3], repeated[3])
        for index in range(3):
            one = cues(img[index:index + 1], aud[index:index + 1], ids=[101 + index])
            for modality in (0, 1):
                self.assertTrue(torch.equal(first[modality][index], one[modality][0]))
            self.assertEqual(first[3]["samples"][index], one[3]["samples"][0])
        image_only = cues(img, aud, mode="corrupt_img_clean_aud")
        self.assertTrue(torch.equal(first[0], image_only[0]))

    def test_does_not_change_global_rng(self):
        for policy in ("uniform", "foreground"):
            random.seed(51)
            np.random.seed(51)
            torch.manual_seed(51)
            before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            cues(policy=policy)
            self.assertEqual(random.getstate(), before[0])
            after_np = np.random.get_state()
            self.assertEqual(after_np[0], before[1][0])
            np.testing.assert_array_equal(after_np[1], before[1][1])
            self.assertEqual(after_np[2:], before[1][2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_rectangles_foreground_constraint_and_outside_preservation(self):
        img, aud = targets()
        ci, ca, truth, metadata = cues(img, aud)
        for i, sample in enumerate(metadata["samples"]):
            for name, cue, target, side, candidates in (
                    ("img", ci, img, 14, 225), ("aud", ca, aud, 32, 1089)):
                item = sample["modalities"][name]
                self.assertEqual(item["rectangle"][2:], [side, side])
                self.assertEqual(item["corrupt_pixels"], side * side)
                self.assertEqual(item["candidate_count"], candidates)
                self.assertEqual(item["mask_area_coverage"], .25)
                self.assertGreaterEqual(item["foreground_coverage"], .35)
                self.assertTrue(item["constraint_met"])
                outside = ~truth[name][i].bool()
                self.assertTrue(torch.equal(cue[i][outside], target[i][outside]))

    def test_infeasible_uses_maximum_and_reports_failure(self):
        img = torch.ones(1, 1, 28, 28)
        aud = torch.ones(1, 64, 64)
        result = cues(img, aud)
        for item in result[3]["samples"][0]["modalities"].values():
            self.assertEqual(item["foreground_coverage"], .25)
            self.assertEqual(item["max_feasible"], .25)
            self.assertFalse(item["constraint_met"])
            self.assertEqual(item["status"], "constraint_infeasible")
            self.assertEqual(item["eligible_candidate_count"], 0)

    def test_infeasible_selects_global_optimum_not_first_attempt(self):
        img = torch.zeros(1, 1, 8, 8)
        aud = torch.zeros(1, 8, 8)
        img[..., 0:2, 0:2] = 1
        img[..., 6:8, 6:8] = 1
        aud[..., 0:2, 0:2] = 1
        aud[..., 6:8, 6:8] = 1
        result = cues(img, aud, options={"min_foreground_coverage": .75})
        for item in result[3]["samples"][0]["modalities"].values():
            self.assertEqual(item["max_feasible"], .5)
            self.assertEqual(item["foreground_coverage"], .5)
            self.assertFalse(item["constraint_met"])

    def test_empty_foreground_not_false_success(self):
        result = cues(torch.zeros(1, 1, 28, 28), torch.zeros(1, 64, 64))
        for item in result[3]["samples"][0]["modalities"].values():
            self.assertFalse(item["evaluable"])
            self.assertEqual(item["foreground_pixels"], 0)
            self.assertIsNone(item["foreground_coverage"])
            self.assertIsNone(item["constraint_met"])
            self.assertIsNone(item["max_feasible"])
            self.assertEqual(item["status"], "no_foreground")
            self.assertGreater(item["corrupt_pixels"], 0)

    def test_zero_severity_zero_strength_and_clean_modes(self):
        img, aud = targets()
        for changes in ({"severity": 0}, {"cfg": config(noise_std=0)},
                        {"mode": "clean_both", "family": "clean"}):
            for policy in ("uniform", "foreground"):
                result = cues(img, aud, policy=policy, **changes)
                self.assertTrue(torch.equal(result[0], img))
                self.assertTrue(torch.equal(result[1], aud))
                for name in ("img", "aud"):
                    self.assertEqual(int(result[2][name].sum()), 0)
                    item = result[3]["samples"][0]["modalities"][name]
                    self.assertIsNone(item["constraint_met"])
                    self.assertIsNone(item["rectangle"])

    def test_thresholds_are_strict_and_audio_max_is_per_sample(self):
        img = torch.tensor([[.09, .1, .10001]])
        self.assertEqual(foreground_mask(img, "img").tolist(), [[False, False, True]])
        aud = torch.tensor([[.01, .02, .021, .1]])
        self.assertEqual(foreground_mask(aud, "aud").tolist(), [[False, False, True, True]])
        aud = torch.tensor([[.02, .09, .1, 1.]])
        self.assertEqual(foreground_mask(aud, "aud").tolist(), [[False, False, False, True]])

    def test_original_noise_field_is_exactly_shared_across_policies(self):
        img, aud = targets()
        for family in ("local_gaussian", "local_impulse"):
            for policy in ("uniform", "foreground"):
                result = cues(img, aud, family=family, policy=policy)
                with preserve_random_state():
                    for i, sample_id in enumerate([101, 102, 103]):
                        for j, (name, target) in enumerate((("img", img), ("aud", aud))):
                            singleton = target[i:i + 1]
                            _seed_corruption(_sample_seed(13, sample_id, family, name))
                            if family == "local_gaussian":
                                field = (singleton + .2 * torch.randn_like(singleton)).clamp(0, 1)
                            else:
                                field = (torch.rand_like(singleton) < .5).to(singleton.dtype)
                            support = result[2][name][i:i + 1].bool()
                            self.assertTrue(torch.equal(result[j][i:i + 1][support], field[support]))

    def test_invalid_inputs_are_rejected(self):
        for changes in ({"severity": float("nan")}, {"severity": float("inf")},
                        {"severity": -.1}, {"severity": 1.1}, {"severity": True},
                        {"seed": 1.5}, {"seed": float("nan")},
                        {"policy": "unknown"}, {"mode": "corrupt_img_only"},
                        {"family": "gaussian"}, {"ids": [1]},
                        {"cfg": config(noise_std=-1)},
                        {"cfg": config(img_noise_std=float("inf"))},
                        {"options": {"min_foreground_coverage": 1.1}},
                        {"options": {"img_threshold": float("nan")}},
                        {"options": {"unknown": 1}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                cues(**changes)
        img, aud = targets()
        img[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            cues(img, aud)


class ForegroundMetricTests(unittest.TestCase):
    def test_exact_region_denominators_and_additivity(self):
        target = torch.zeros(2, 3)
        cue = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        recovered = cue / 2
        truth = torch.tensor([[1, 1, 0], [0, 1, 0]])
        foreground = torch.tensor([[1, 0, 1], [0, 1, 0]])
        result = foreground_metrics(cue, recovered, target, truth, foreground)
        for region, pixels, sse in (("foreground", 3, 35),
                                    ("corrupt_foreground", 2, 26),
                                    ("corrupt_background", 1, 4),
                                    ("clean_foreground", 1, 9), ("clean", 3, 61)):
            self.assertEqual(result[f"{region}_pixels"], pixels)
            self.assertEqual(result[f"{region}_input_sse"], sse)
            self.assertAlmostEqual(result[f"{region}_input_mse"], sse / pixels)
            self.assertAlmostEqual(result[f"{region}_recovered_mse"], sse / pixels / 4)
        self.assertEqual(result["foreground_coverage"], 2 / 3)
        self.assertEqual(result["global_input_sse"],
                         result["foreground_input_sse"] + result["background_input_sse"])

    def test_empty_denominators_are_none_and_exact_preservation(self):
        target = torch.zeros(2, 2)
        cue = torch.ones(2, 2)
        truth = torch.tensor([[1, 0], [0, 0]])
        recovered = cue.clone()
        recovered[0, 0] = 0
        result = foreground_metrics(cue, recovered, target, truth, torch.zeros_like(truth))
        self.assertEqual(result["foreground_pixels"], 0)
        self.assertEqual(result["foreground_input_sse"], 0)
        self.assertIsNone(result["foreground_input_mse"])
        self.assertIsNone(result["foreground_coverage"])
        self.assertFalse(result["foreground_evaluable"])
        self.assertTrue(result["clean_exactly_preserved"])
        recovered[0, 1] = .9
        self.assertFalse(foreground_metrics(cue, recovered, target, truth,
                                           truth)["clean_exactly_preserved"])

    def test_metric_validation(self):
        value = torch.zeros(2, 2)
        with self.assertRaises(ValueError):
            foreground_metrics(value, value, value, torch.ones(2, 2) * .5, value)
        with self.assertRaises(ValueError):
            foreground_metrics(value, value[:1], value, value, value)
        with self.assertRaises(ValueError):
            foreground_metrics(value + float("inf"), value, value, value, value)


if __name__ == "__main__":
    unittest.main()
