import unittest

import torch

from models.dendritic_index import DendriticIndexLayer
from models.memory import ValueLayer


class DendriticIndexTests(unittest.TestCase):
    def make_layer(self, **kwargs):
        layer = DendriticIndexLayer(2, 2, 2, dendritic_k=1, soma_k=1, **kwargs)
        with torch.no_grad():
            layer.image_projection.weight.copy_(torch.eye(2))
            layer.audio_projection.weight.copy_(torch.eye(2))
            layer.recurrent.weight.zero_()
        return layer

    def test_absent_compartment_and_coincidence_are_zero(self):
        keys = torch.ones(3, 1, 2)
        layer = self.make_layer()
        for args, absent in (((keys, None), "audio"), ((None, keys), "image")):
            with self.subTest(absent=absent):
                result = layer(*args)
                self.assertEqual(result[f"{absent}_voltage"].count_nonzero().item(), 0)
                self.assertEqual(result[f"{absent}_drive"].count_nonzero().item(), 0)
                self.assertEqual(result["coincidence"].count_nonzero().item(), 0)

    def test_coincidence_requires_positive_drive_at_the_same_unit(self):
        layer = self.make_layer()
        image = torch.tensor([[[1., 0.]]])
        for audio, expected in (([1., 0.], True), ([0., 1.], False), ([-1., 0.], False)):
            with self.subTest(audio=audio):
                result = layer(image, torch.tensor([[audio]]))
                self.assertEqual(bool(result["coincidence"].gt(0).any()), expected)

    def test_coincidence_ablation_uses_identical_weights_and_changes_firing(self):
        baseline = self.make_layer(coincidence_scale=0)
        treatment = self.make_layer(coincidence_scale=0.1)
        treatment.load_state_dict(baseline.state_dict(), strict=True)
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, treatment.state_dict()[key])
        keys = torch.tensor([[[0.54, 0.]]])
        self.assertEqual(baseline(keys, keys)["soma_spikes"].count_nonzero().item(), 0)
        self.assertEqual(treatment(keys, keys)["soma_spikes"].count_nonzero().item(), 1)

    def test_partial_reset_and_leak(self):
        layer = self.make_layer()
        keys = torch.tensor([[[1., 0.]], [[0., 0.]]])
        result = layer(keys, keys)
        self.assertEqual(result["soma_spikes"][0, 0, 0].item(), 1)
        torch.testing.assert_close(result["image_voltage"][1, 0, 0], torch.tensor(0.8 * 0.7))
        torch.testing.assert_close(result["audio_voltage"][1, 0, 0], torch.tensor(0.8 * 0.7))

    def test_attention_current_reaches_soma_directly(self):
        layer = self.make_layer()
        keys = torch.zeros(1, 1, 2)
        current = torch.tensor([[1.1, 0.]])
        result = layer(keys, keys, cross_modal_current=current)
        self.assertEqual(result["soma_spikes"][0, 0, 0].item(), 1)
        self.assertEqual(result["coincidence"].count_nonzero().item(), 0)
        self.assertEqual(result["image_voltage"].count_nonzero().item(), 0)
        self.assertEqual(result["audio_voltage"].count_nonzero().item(), 0)

    def test_separate_drives_feed_value_and_classifier(self):
        layer = self.make_layer()
        keys = torch.tensor([[[0.6, 0.]], [[0.5, 0.]]])
        result = layer(keys, keys)
        value = ValueLayer(n_index=2, d_enc=2, n_value=3, private_size=2)
        value_spikes, _, _ = value(result["soma_spikes"], private_spikes=result["image_drive"])
        self.assertEqual(value_spikes.shape, (2, 1, 3))
        self.assertEqual(result["classifier_state"].shape, (1, 6))
        result["classifier_state"].sum().backward()
        self.assertGreater(layer.image_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(layer.audio_projection.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
