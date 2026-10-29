"""Behavioral checks for blind mask routing and recovery gradients."""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import load_config, set_seed
from models.network import CrossModalSNN
from scripts.train import compute_losses


def small_config(enabled=True):
    cfg = load_config(ROOT / "configs/v13pro.yaml")
    cfg["snn"].update(T=2, aud_conv_ch1=2, aud_conv_ch2=4,
                      aud_decoder_base_ch=8, aud_decoder_refine_blocks=0)
    cfg["dims"].update(img_hidden=16, D_img=8, N_key_img=8,
                       aud_hidden=16, D_aud=8, N_key_aud=8,
                       N_index=16, N_value_img=8, N_value_aud=16)
    cfg["index"]["k_wta"] = 4
    cfg["detail_conditioning"].update(img_detail_dim=8, aud_detail_dim=8)
    cfg["audio_local_cue"].update(enabled=False, multiscale=False)
    cfg["image_refiner"]["enabled"] = False
    cfg["image_refiner"]["pasteback_only"] = True
    cfg["cross_key_conditioning"].update(enabled=False, build_modules=False,
                                         build_feature_modules=False)
    cfg["cross_key_conditioning"]["causal_training"]["enabled"] = False
    cfg["corruption"].update(img_mode="local_gaussian", aud_mode="local_gaussian",
                              img_train_modes=["local_gaussian"],
                              aud_train_modes=["local_gaussian"],
                              train_severity=.25, noise_std=.2)
    cfg["noise_detector"] = dict(enabled=enabled, hidden_channels=4,
        mask_source="predicted", threshold=.5, freeze=False,
        require_pretrained=False, loss_weight=1., coarse_loss_weight=.5)
    return cfg


def force_mask(model, value):
    with torch.no_grad():
        for detector in (model.img_noise_detector, model.aud_noise_detector):
            for parameter in detector.parameters():
                parameter.zero_()
            detector.net[-1].bias.fill_(20 if value else -20)


class BlindMaskIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        set_seed(7)
        self.cfg = small_config()
        self.img = torch.rand(2, 1, 28, 28).clamp(.01, .99)
        self.aud = torch.rand(2, 64, 64)

    def test_external_oracle_mask_cannot_change_blind_output(self):
        model = CrossModalSNN(self.cfg).eval()
        force_mask(model, False)
        with torch.no_grad():
            first = model(self.img, self.aud)
            second = model(self.img, self.aud, img_cue_mask=torch.ones_like(self.img),
                           aud_cue_mask=torch.ones_like(self.aud))
        torch.testing.assert_close(first["recovered_img"], second["recovered_img"])
        torch.testing.assert_close(first["recovered_aud"], second["recovered_aud"])
        torch.testing.assert_close(first["recovered_img"].sigmoid(), self.img)
        torch.testing.assert_close(first["recovered_aud"], self.aud)
        self.assertFalse(first["img_repair_mask"].any())

    def test_predicted_support_selects_decoder_instead_of_input(self):
        model = CrossModalSNN(self.cfg).eval()
        force_mask(model, True)
        out = model(self.img, self.aud)
        self.assertTrue(out["img_repair_mask"].all())
        torch.testing.assert_close(out["recovered_aud"], out["recovered_aud_coarse"])
        self.assertFalse(torch.allclose(out["recovered_aud"], self.aud))

    def test_missing_modality_is_not_treated_as_clean_detector_input(self):
        model = CrossModalSNN(self.cfg).eval()
        out = model(None, self.aud)
        self.assertIsNone(out["img_noise_logits"])
        self.assertIsNone(out["img_repair_mask"])
        self.assertEqual(out["recovered_img"].shape, self.img.shape)

    def test_coarse_auxiliary_trains_decoder_when_detector_misses_everything(self):
        model = CrossModalSNN(self.cfg)
        force_mask(model, False)
        labels = torch.tensor([1, 2])
        proto_img, proto_aud = torch.rand(10, 1, 28, 28), torch.rand(10, 64, 64)
        loss, logs = compute_losses(model, self.img, self.aud, labels,
            "corrupt_both", self.cfg, proto_img, proto_aud)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.image_decoder.cnn[-1].bias.grad.abs().sum().item(), 0)
        self.assertGreater(model.audio_decoder.cnn[-1].bias.grad.abs().sum().item(), 0)
        self.assertGreater(model.img_noise_detector.net[-1].bias.grad.abs().sum().item(), 0)
        self.assertIn("img_noise_coarse_mse", logs)

    def test_frozen_detectors_stay_frozen_during_restoration(self):
        self.cfg["noise_detector"]["freeze"] = True
        model = CrossModalSNN(self.cfg).train()
        self.assertFalse(model.img_noise_detector.training)
        self.assertFalse(any(p.requires_grad for p in model.img_noise_detector.parameters()))
        self.assertTrue(model.image_decoder.fc.weight.requires_grad)

    def test_legacy_model_has_no_detector_state_and_keeps_explicit_mask(self):
        model = CrossModalSNN(small_config(False)).eval()
        self.assertFalse(any("noise_detector" in key for key in model.state_dict()))
        out = model(self.img, self.aud, img_cue_mask=torch.zeros_like(self.img),
                    aud_cue_mask=torch.zeros_like(self.aud))
        torch.testing.assert_close(out["recovered_aud"], self.aud)


if __name__ == "__main__":
    unittest.main()
