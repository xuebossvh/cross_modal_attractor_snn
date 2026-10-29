"""Local contamination support, reproducibility, and legacy API contracts."""

import random
import unittest
from unittest.mock import patch

import torch

from common import build_cue
from data.corruption import LOCAL_NOISE_MODES, corrupt_audio, corrupt_image


def seed(value=1234):
    random.seed(value)
    torch.manual_seed(value)


class LocalCorruptionTests(unittest.TestCase):
    def fixtures(self):
        return (
            (corrupt_image, torch.full((4, 1, 28, 28), 0.5)),
            (corrupt_audio, torch.full((4, 64, 64), 0.5)),
        )

    def test_support_is_rectangle_with_requested_area_and_clean_region_exact(self):
        for corrupt, clean in self.fixtures():
            for family in LOCAL_NOISE_MODES:
                with self.subTest(modality=corrupt.__name__, family=family):
                    seed()
                    original = clean.clone()
                    noisy, mask = corrupt(clean, family, severity=0.25,
                                          return_mask=True, noise_std=0.2)
                    self.assertEqual(noisy.shape, clean.shape)
                    self.assertEqual(mask.shape, clean.shape)
                    self.assertEqual(noisy.dtype, clean.dtype)
                    self.assertEqual(noisy.device, clean.device)
                    self.assertTrue(torch.equal(clean, original))
                    self.assertTrue(torch.equal(noisy[mask == 0], clean[mask == 0]))
                    self.assertTrue(torch.all((mask == 0) | (mask == 1)))
                    self.assertTrue(torch.all((noisy >= 0) & (noisy <= 1)))
                    self.assertTrue(torch.all(mask.flatten(1).mean(1) == 0.25))
                    self.assertFalse(torch.equal(noisy, clean))
                    for sample_mask in mask.reshape(-1, *mask.shape[-2:]):
                        rows, columns = torch.where(sample_mask.bool())
                        rectangle = sample_mask[rows.min():rows.max() + 1,
                                                columns.min():columns.max() + 1]
                        self.assertEqual(int(rectangle.sum()), rectangle.numel())

    def test_zero_severity_or_noise_std_is_clean(self):
        for corrupt, clean in self.fixtures():
            for family in LOCAL_NOISE_MODES:
                for severity, strength in ((0.0, 0.2), (0.4, 0.0)):
                    with self.subTest(modality=corrupt.__name__, family=family,
                                      severity=severity, noise_std=strength):
                        noisy, mask = corrupt(clean, family, severity,
                                              return_mask=True, noise_std=strength)
                        self.assertTrue(torch.equal(noisy, clean))
                        self.assertEqual(int(mask.sum()), 0)
                        self.assertNotEqual(noisy.data_ptr(), clean.data_ptr())

    def test_full_coverage_and_seeded_reproducibility(self):
        for corrupt, clean in self.fixtures():
            for family in LOCAL_NOISE_MODES:
                seed()
                first = corrupt(clean, family, 0.4, True)
                seed()
                second = corrupt(clean, family, 0.4, True)
                self.assertTrue(torch.equal(first[0], second[0]))
                self.assertTrue(torch.equal(first[1], second[1]))
                _, full_mask = corrupt(clean, family, 1.0, True)
                self.assertTrue(torch.all(full_mask == 1))

    def test_impulse_support_includes_unchanged_replacements(self):
        for corrupt, clean in self.fixtures():
            clean = torch.zeros_like(clean)
            seed()
            noisy, mask = corrupt(clean, "local_impulse", 0.25, True)
            self.assertTrue(torch.all((noisy == 0) | (noisy == 1)))
            self.assertTrue(torch.any((mask == 1) & (noisy == clean)))
            self.assertTrue(torch.any((mask == 1) & (noisy != clean)))

    def test_noise_strength_is_independent_of_support(self):
        for corrupt, clean in self.fixtures():
            seed()
            mild, mild_mask = corrupt(clean, "local_gaussian", 0.25, True, 0.05)
            seed()
            stronger, stronger_mask = corrupt(clean, "local_gaussian", 0.25, True, 0.1)
            self.assertTrue(torch.equal(mild_mask, stronger_mask))
            self.assertGreater(float((stronger - clean).square().mean()),
                               float((mild - clean).square().mean()))

    def test_build_cue_respects_per_modality_strength_and_default(self):
        image = torch.full((2, 1, 28, 28), 0.5)
        audio = torch.full((2, 64, 64), 0.5)
        config = {"corruption": {"train_severity": 0.25,
                                 "img_mode": "local_gaussian",
                                 "aud_mode": "local_gaussian",
                                 "noise_std": 0.2, "img_noise_std": 0.0}}
        seed()
        ci, ca, masks = build_cue(image, audio, "corrupt_both", config,
                                  return_masks=True)
        self.assertTrue(torch.equal(ci, image))
        self.assertEqual(int(masks["img"].sum()), 0)
        self.assertFalse(torch.equal(ca, audio))
        self.assertGreater(int(masks["aud"].sum()), 0)
        seed()
        without_masks = build_cue(image, audio, "corrupt_both", config)
        self.assertTrue(torch.equal(ci, without_masks[0]))
        self.assertTrue(torch.equal(ca, without_masks[1]))
        config["corruption"].update(img_noise_std=0.1, aud_noise_std=0.0)
        ci, ca, masks = build_cue(image, audio, "corrupt_both", config,
                                  return_masks=True)
        self.assertFalse(torch.equal(ci, image))
        self.assertTrue(torch.equal(ca, audio))
        self.assertEqual(int(masks["aud"].sum()), 0)

    def test_legacy_random_pool_and_gaussian_amplitude_unchanged(self):
        for corrupt, clean in self.fixtures():
            pools = []

            def choose(pool):
                pools.append(tuple(pool))
                return "gaussian"

            with patch("data.corruption.random.choice", side_effect=choose):
                corrupt(clean, "random", 0.3)
            self.assertEqual(len(pools), 1)
            self.assertTrue(set(LOCAL_NOISE_MODES).isdisjoint(pools[0]))
            seed()
            expected = (clean + 0.3 * torch.randn_like(clean)).clamp(0, 1)
            seed()
            actual, mask = corrupt(clean, "gaussian", 0.3, True, noise_std=0)
            self.assertTrue(torch.equal(actual, expected))
            self.assertIsNone(mask)

    def test_invalid_local_strength_and_flat_audio_fail_explicitly(self):
        for corrupt, clean in self.fixtures():
            for strength in (-0.1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    corrupt(clean, "local_gaussian", noise_std=strength)
        for family in LOCAL_NOISE_MODES:
            with self.assertRaisesRegex(ValueError, "Local audio contamination"):
                corrupt_audio(torch.zeros(2, 128), family)


if __name__ == "__main__":
    unittest.main()
