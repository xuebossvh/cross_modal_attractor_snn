"""CPU checks for the detector's spatial contract and supervised gradients."""

import unittest

import torch
import torch.nn.functional as F

from models.noise_detector import LocalNoiseDetector, supervised_mask_loss


class NoiseDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_image_and_audio_shapes_and_binary_prediction(self):
        detector = LocalNoiseDetector()
        for shape in ((2, 1, 28, 28), (2, 64, 64)):
            with self.subTest(shape=shape):
                cue = torch.rand(shape)
                logits = detector(cue)
                probabilities = detector.probabilities(cue)
                mask = detector.predict_mask(cue)
                self.assertEqual(logits.shape, cue.shape)
                self.assertEqual(probabilities.shape, cue.shape)
                self.assertEqual(mask.shape, cue.shape)
                self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)))
                self.assertTrue(torch.all((mask == 0) | (mask == 1)))
                self.assertFalse(mask.requires_grad)

    def test_supervised_loss_trains_all_convolutions(self):
        torch.manual_seed(42)
        detector = LocalNoiseDetector(hidden_channels=8)
        cue = torch.rand(2, 1, 28, 28)
        target = torch.zeros_like(cue)
        target[:, :, 8:20, 10:22] = 1
        loss = supervised_mask_loss(detector(cue), target)
        loss.backward()
        for name, parameter in detector.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_absent_label_means_clean_present_cue(self):
        logits = torch.tensor([[[0.0, 1.0], [-1.0, 2.0]]], requires_grad=True)
        actual = supervised_mask_loss(logits, None)
        expected = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_loss_handles_singleton_channel_target_without_broadcast(self):
        logits = torch.randn(2, 1, 4, 5)
        target = torch.randint(2, (2, 4, 5))
        actual = supervised_mask_loss(logits, target)
        expected = F.binary_cross_entropy_with_logits(
            logits, target[:, None].float(), pos_weight=torch.tensor(3.0))
        torch.testing.assert_close(actual, expected)

    def test_extreme_logits_and_weighted_positive_examples_are_finite(self):
        logits = torch.tensor([[[1000.0, -1000.0]]], requires_grad=True)
        labels = torch.tensor([[[0.0, 1.0]]])
        loss = supervised_mask_loss(logits, labels, positive_weight=3.0)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, torch.tensor(2000.0))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_rejects_non_single_channel_and_invalid_shapes(self):
        detector = LocalNoiseDetector()
        for shape in ((2, 3, 28, 28), (28, 28), (2, 1, 1, 28, 28), (0, 28, 28)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                detector(torch.rand(shape))
        with self.assertRaises(TypeError):
            detector(torch.ones(2, 1, 4, 4, dtype=torch.uint8))

    def test_rejects_invalid_supervision_and_positive_weights(self):
        logits = torch.zeros(1, 1, 4, 4)
        for weight in (0, -1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                supervised_mask_loss(logits, None, positive_weight=weight)
        with self.assertRaises(ValueError):
            supervised_mask_loss(logits, torch.zeros(1, 1, 3, 4))
        for value in (0.5, float("nan"), float("inf")):
            with self.subTest(target=value), self.assertRaises(ValueError):
                supervised_mask_loss(logits, torch.full_like(logits, value))

    def test_rejects_invalid_threshold_and_width(self):
        detector = LocalNoiseDetector()
        cue = torch.rand(1, 1, 4, 4)
        for threshold in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                detector.predict_mask(cue, threshold=threshold)
        for width in (0, -1, 3.5, True):
            with self.subTest(width=width), self.assertRaises(ValueError):
                LocalNoiseDetector(hidden_channels=width)


if __name__ == "__main__":
    unittest.main()
