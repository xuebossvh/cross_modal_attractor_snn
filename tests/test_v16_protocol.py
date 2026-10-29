"""Small deterministic protocol checks; no external datasets or model claims."""

from copy import deepcopy
import random
import unittest

import numpy as np
import torch

from data.v16_protocol import (MODE_STATES, MODES, cue_probabilities,
                               detector_mask_loss, make_cues, make_targets,
                               protocol_contract, sample_mode)


class V16ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.img = torch.linspace(0, 1, 3 * 64).reshape(3, 1, 8, 8)
        self.aud = torch.linspace(1, 0, 3 * 64).reshape(3, 8, 8)
        self.cfg = {"v16": {"missing_severity": 0.4, "noise_coverage": 0.25, "noise_std": 0.2}}

    def test_thirteen_states_no_mutation_and_disjoint_support(self):
        originals = self.img.clone(), self.aud.clone()
        self.assertEqual(len(MODES), 13)
        for mode in MODES:
            with self.subTest(mode=mode):
                result = make_cues(self.img, self.aud, mode, self.cfg, seed=123, ids=[10, 20, 30])
                for name, clean, state in zip(("img", "aud"), originals, MODE_STATES[mode]):
                    self.assertEqual(result["states"][name], state)
                    if state == "A":
                        self.assertIsNone(result[name])
                        self.assertIsNone(result["noise"][name])
                        self.assertIsNone(result["missing"][name])
                        continue
                    missing, noise = result["missing"][name], result["noise"][name]
                    self.assertEqual(tuple(result[name].shape), tuple(clean.shape))
                    self.assertEqual(int((missing * noise).sum()), 0)
                    self.assertEqual(len(result[f"{name}_family"]), 3)
                    if state != "P":
                        self.assertEqual(int(missing.sum()), 0)
                    if state != "N":
                        self.assertEqual(int(noise.sum()), 0)
                    reliable = (missing + noise) == 0
                    torch.testing.assert_close(result[name][reliable], clean[reliable], rtol=0, atol=0)
        torch.testing.assert_close(self.img, originals[0], rtol=0, atol=0)
        torch.testing.assert_close(self.aud, originals[1], rtol=0, atol=0)

    def test_seeded_batch_order_and_mode_independence(self):
        for mode in ("corrupt_both", "noisy_both"):
            whole = make_cues(self.img, self.aud, mode, self.cfg, seed=73, ids=[10, 20, 30], protocol="test")
            reordered = make_cues(self.img[[2, 0, 1]], self.aud[[2, 0, 1]], mode, self.cfg,
                                  seed=73, ids=[30, 10, 20], protocol="test")
            single = [make_cues(self.img[i:i + 1], self.aud[i:i + 1], mode, self.cfg,
                                seed=73, ids=[ident], protocol="test") for i, ident in enumerate((10, 20, 30))]
            for name in ("img", "aud"):
                torch.testing.assert_close(whole[name], reordered[name][[1, 2, 0]], rtol=0, atol=0)
                torch.testing.assert_close(whole[name], torch.cat([r[name] for r in single]), rtol=0, atol=0)
                only = "noisy_img_only" if mode == "noisy_both" else "corrupt_img_only"
                image_only = make_cues(self.img, self.aud, only, self.cfg, seed=73, ids=[10, 20, 30], protocol="test")
                torch.testing.assert_close(whole["img"], image_only["img"], rtol=0, atol=0)

    def test_seeded_rng_restored_and_stochastic_training(self):
        random.seed(6)
        np.random.seed(6)
        torch.manual_seed(6)
        python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        make_cues(self.img, self.aud, "noisy_both", self.cfg, seed=4, ids=[1, 2, 3])
        self.assertEqual(python_before, random.getstate())
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        torch.testing.assert_close(torch_before, torch.get_rng_state(), rtol=0, atol=0)
        for before, after in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else []):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        one = make_cues(self.img, self.aud, "noisy_both", self.cfg)
        two = make_cues(self.img, self.aud, "noisy_both", self.cfg)
        self.assertFalse(torch.equal(one["img"], two["img"]))

    def test_probabilities_families_counts_and_zero_severity(self):
        weights = cue_probabilities(self.cfg)
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertAlmostEqual(sum(weights[m] for m in MODES[:8]), 0.5)
        forced = deepcopy(self.cfg)
        forced["v16"]["cue_probabilities"] = {"noisy_img_only": 1}
        self.assertEqual(sample_mode(forced), "noisy_img_only")
        noisy = make_cues(self.img, self.aud, "noisy_both", self.cfg, seed=3, ids=[1, 2, 3],
                          img_family="local_gaussian", aud_family="local_impulse")
        self.assertEqual(int(noisy["noise"]["img"].sum()), 3 * 16)
        self.assertEqual(int(noisy["noise"]["aud"].sum()), 3 * 16)
        clean = make_cues(self.img, self.aud, "corrupt_both", self.cfg, severity=0)
        torch.testing.assert_close(clean["img"], self.img, rtol=0, atol=0)
        torch.testing.assert_close(clean["aud"], self.aud, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "stable"):
            make_cues(self.img, self.aud, "clean_both", self.cfg, seed=1)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            make_cues(self.img, self.aud, "noisy_both", self.cfg, img_family="occlusion")
        with self.assertRaises(ValueError):
            protocol_contract({"v16": {"noise_std": float("nan")}})

    def test_targets_instances_and_category_representatives(self):
        image_prototypes, audio_prototypes = self.img.flip(0), self.aud.flip(0)
        labels = torch.tensor([1, 2, 0])
        for mode in MODES:
            ti, ta, ki, ka = make_targets(self.img, self.aud, labels, image_prototypes, audio_prototypes, mode)
            for target, kind, state, clean, prototypes in zip((ti, ta), (ki, ka), MODE_STATES[mode],
                                                              (self.img, self.aud), (image_prototypes, audio_prototypes)):
                self.assertEqual(kind, "category" if state == "A" else "instance")
                torch.testing.assert_close(target, prototypes[labels] if state == "A" else clean)

    def test_detector_loss_excludes_known_holes_and_all_holes(self):
        logits = torch.tensor([[[100., 0., 0.]]], requires_grad=True)
        noise = torch.tensor([[[0., 1., 0.]]])
        missing = torch.tensor([[[1., 0., 0.]]])
        loss = detector_mask_loss(logits, noise, missing, positive_weight=3)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[..., 1:], noise[..., 1:], pos_weight=torch.tensor(3.))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 0, 0]), 0)
        self.assertNotEqual(float(logits.grad[0, 0, 1]), 0)
        all_holes = detector_mask_loss(logits, noise, torch.ones_like(missing))
        self.assertEqual(float(all_holes.detach()), 0)
        with self.assertRaisesRegex(ValueError, "exactly"):
            detector_mask_loss(logits, noise.unsqueeze(1))


if __name__ == "__main__":
    unittest.main()
