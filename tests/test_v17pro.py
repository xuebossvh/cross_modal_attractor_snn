from pathlib import Path
import sys
import unittest

import torch

from common import load_config
from models.memory import CrossModalAttractorMemory, ValueLayer
from models.mult import AudioToImageAttention
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.run_v16pro_suite import build_plan
from scripts.train_v16pro import validate_training_config
from scripts.v17_protocol import ARMS


class V17Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1] / "configs/v17pro.yaml")

    def test_all_arms_use_the_matching_parent_and_audiomnist(self):
        configs, jobs = build_plan(self.cfg, "outputs/v17pro_test", (1234,), "cpu",
                                   arms=list(ARMS))
        self.assertEqual(len(jobs), 10)
        by_path = {Path(path).parent.name: cfg for path, cfg in configs.items()}
        for arm, switches in ARMS.items():
            parent = by_path[f"parent_{arm}"]
            adaptation = by_path[arm]
            self.assertEqual(validate_training_config(parent), "parent")
            self.assertEqual(validate_training_config(adaptation), arm)
            self.assertEqual(parent["v17"], adaptation["v17"])
            self.assertEqual((parent["v17"]["private_index"]["enabled"],
                              parent["v17"]["mult"]["enabled"]), switches)
            self.assertEqual(Path(adaptation["train"]["init_ckpt_path"]).parent.name,
                             f"parent_{arm}")
            self.assertEqual(adaptation["data"]["dataset"], "mnist_audiomnist")

    def test_private_readout_has_its_own_synapses(self):
        value = ValueLayer(1, 1, 1, private_size=1)
        with torch.no_grad():
            value.W_A_to_V.weight.zero_()
            value.W_A_to_V.bias.zero_()
            value.W_private_to_V.weight.fill_(2.0)
        shared = torch.zeros(3, 1, 1)
        private = torch.ones_like(shared)
        self.assertEqual(value(shared)[1].item(), 0)
        self.assertEqual(value(shared, private_spikes=private)[1].item(), 1)

    def test_missing_modality_and_three_state_shape(self):
        memory = CrossModalAttractorMemory(self.cfg)
        cue = torch.ones(3, 2, self.cfg["dims"]["D_img"])
        outputs = memory(spike_img_cue=cue)
        self.assertEqual(outputs["private_img_spikes"].shape, (3, 2, 128))
        self.assertTrue((outputs["private_img_spikes"].sum(-1) <= 24).all())
        self.assertIsNone(outputs["private_aud_spikes"])
        self.assertEqual(outputs["v_aud_from_A"].shape, (2, 768))

    def test_attention_excludes_missing_image_and_ignores_masked_audio_values(self):
        attention = AudioToImageAttention(64, 64, 16, 16, 4)
        image = torch.rand(2, 1, 28, 28)
        audio = torch.rand(2, 64, 64)
        image_mask = torch.zeros_like(image)
        image_mask[0].fill_(1)
        audio_mask = torch.zeros_like(audio)
        audio_mask[:, :, :8] = 1
        result = attention(image, audio, image_mask, audio_mask)
        modified = audio.masked_fill(audio_mask.bool(), 1)
        torch.testing.assert_close(result, attention(image, modified, image_mask, audio_mask))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result[0].abs().max().item(), 0)
        self.assertGreater(result[1].abs().max().item(), 0)
        result.square().sum().backward()
        self.assertGreater(attention.attention.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertIsNone(attention(image, None))


if __name__ == "__main__":
    unittest.main()
