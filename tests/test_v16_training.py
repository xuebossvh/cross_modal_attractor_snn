"""CPU checks of actual v16 routing, recovery gradients and strict lineage."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from common import load_config, set_seed
from data.v16_protocol import MODES, make_cues, protocol_contract
from models.network import CrossModalSNN
from models.noise_detector import LocalNoiseDetector
from scripts.train_v16pro import (SCHEMA, RECOVERY_PREFIXES, compute_losses,
                                  configure_trainable, load_parent, train_model,
                                  validate_training_config, _runtime_sha256)


ROOT = Path(__file__).resolve().parents[1]


def tiny_config(stage="parent"):
    cfg = load_config(ROOT / "configs/v16pro.yaml")
    cfg["device"] = "cpu"
    cfg["snn"].update(T=2, aud_conv_ch1=2, aud_conv_ch2=4,
                      aud_decoder_base_ch=8, aud_decoder_refine_blocks=0)
    cfg["dims"].update(img_hidden=16, D_img=8, N_key_img=8, aud_hidden=16,
                       D_aud=8, N_key_aud=8, N_index=16, N_value_img=8, N_value_aud=16)
    cfg["index"]["k_wta"] = 4
    cfg["detail_conditioning"].update(img_detail_dim=8, aud_detail_dim=8)
    cfg["audio_local_cue"]["hidden_channels"] = 4
    cfg["image_refiner"].update(enabled=False, pasteback_only=True)
    cfg["audio"].update(use_real_audio=False, encoder_norm_mode="global")
    cfg["noise_detector"].update(hidden_channels=4, freeze=True,
        require_pretrained=True, allow_smoke_checkpoint=True)
    cfg["data"].update(batch_size=2, num_workers=0)
    cfg["train"].update(stage=stage, epochs=2, lr=0.0002, require_cuda=False,
                        init_ckpt_path="", init_required=False)
    cfg["v16"].update(local_scales=["64"] if stage.startswith("parent") else ["16", "32", "64"],
                       no_value=stage == "no_value")
    cfg["ablation"]["use_recurrent"] = "no_recurrence" not in stage
    cfg["train"]["trainable_prefixes"] = [] if stage.startswith("parent") else list(RECOVERY_PREFIXES)
    return cfg


def fake_loaders(cfg, eval_split):
    if eval_split != "val":
        raise AssertionError("Training must not select on test data")
    gen = torch.Generator().manual_seed(419)
    images = torch.rand(2, 1, 28, 28, generator=gen).clamp(0.01, 0.99)
    audio = torch.rand(2, 64, 64, generator=gen)
    labels = torch.tensor([1, 2])
    prototype_img = torch.rand(10, 1, 28, 28, generator=gen)
    prototype_aud = torch.rand(10, 64, 64, generator=gen)
    result = []
    for split in ("train", "val"):
        ds = TensorDataset(images, audio, labels)
        ds.split, ds.prototype_img, ds.prototype_aud = split, prototype_img, prototype_aud
        result.append(DataLoader(ds, batch_size=2, shuffle=split == "train",
                                 generator=torch.Generator().manual_seed(1)))
    return tuple(result)


def force_detector(model, present):
    with torch.no_grad():
        for detector in (model.img_noise_detector, model.aud_noise_detector):
            for param in detector.parameters():
                param.zero_()
            detector.net[-1].bias.fill_(20 if present else -20)


def write_detector(cfg, path):
    set_seed(5)
    state = {"schema": "local-noise-detector.v1", "cfg": deepcopy(cfg),
             "hidden_channels": 4, "best_validation_loss": 0.1,
             "detectors": {name: LocalNoiseDetector(4).state_dict() for name in ("img", "aud")},
             "metadata": {"protocol": protocol_contract(cfg), "selection_split": "val",
                          "missing_policy": "excluded_from_bce", "smoke_only": True}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def tiny_validation(model, loader, cfg, device, max_batches=0, max_samples=0):
    model.eval()
    images, audio, _ = next(iter(loader))
    with torch.no_grad():
        result = model(images.to(device), audio.to(device))
    score = float((result["recovered_img_coarse"].sigmoid() - images.to(device)).square().mean())
    return {"score": score, "acc": 0.0, "img_mse": score, "aud_mse": 0.0}


class V16TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        set_seed(13)
        self.img = torch.rand(2, 1, 28, 28).clamp(.01, .99)
        self.aud = torch.rand(2, 64, 64)

    def test_known_missing_union_and_true_noise_never_leaks(self):
        model = CrossModalSNN(tiny_config()).eval()
        force_detector(model, False)
        missing = torch.zeros_like(self.img)
        missing[..., 5:10, 5:10] = 1
        with torch.no_grad():
            first = model(self.img, self.aud, img_cue_mask=missing,
                          img_noise_mask=torch.zeros_like(self.img), aud_noise_mask=torch.zeros_like(self.aud))
            second = model(self.img, self.aud, img_cue_mask=missing,
                           img_noise_mask=torch.ones_like(self.img), aud_noise_mask=torch.ones_like(self.aud))
        torch.testing.assert_close(first["img_repair_mask"], missing)
        self.assertFalse(first["img_predicted_mask"].any())
        torch.testing.assert_close(first["recovered_img"], second["recovered_img"])
        torch.testing.assert_close(first["recovered_img"].sigmoid() * (1 - missing), self.img * (1 - missing))
        model.noise_mask_source = "oracle"
        with self.assertRaisesRegex(ValueError, "explicit true noise"):
            model(self.img, self.aud, img_cue_mask=missing)
        noise = torch.zeros_like(self.img)
        noise[..., 15:19, 15:19] = 1
        result = model(self.img, self.aud, img_cue_mask=missing, img_noise_mask=noise,
                       aud_noise_mask=torch.zeros_like(self.aud))
        torch.testing.assert_close(result["img_repair_mask"], torch.maximum(missing, noise))
        model.noise_mask_source = "all"
        self.assertTrue(model(self.img, None)["img_repair_mask"].all())

    def test_all_thirteen_cues_forward_finite_and_absence_skips_detector(self):
        cfg = tiny_config()
        model = CrossModalSNN(cfg).eval()
        for mode in MODES:
            with self.subTest(mode=mode), torch.no_grad():
                cues = make_cues(self.img, self.aud, mode, cfg)
                out = model(cues["img"], cues["aud"], img_cue_mask=cues["missing"]["img"],
                            aud_cue_mask=cues["missing"]["aud"])
                for name in ("img", "aud"):
                    self.assertTrue(torch.isfinite(out["recovered_" + name]).all())
                    if cues[name] is None:
                        self.assertIsNone(out[name + "_noise_logits"])
                        self.assertIsNone(out[name + "_repair_mask"])

    def test_no_value_removes_gate_and_decoder_dependence(self):
        cfg = tiny_config("no_value")
        model = CrossModalSNN(cfg).eval()
        for name in ("img", "aud"):
            width = cfg["dims"]["N_value_" + name]
            detail = torch.rand(2, cfg["dims"]["D_" + name])
            v1 = torch.rand(2, width, requires_grad=True)
            v2 = torch.rand(2, width) * 100
            first = model._fuse_decoder_state(v1, detail, name)
            second = model._fuse_decoder_state(v2, detail, name)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertFalse(first[:, :width].any())
            first.sum().backward()
            self.assertIsNone(v1.grad)
        cfg["cross_key_conditioning"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "excludes"):
            CrossModalSNN(cfg)

    def test_masked_parent_and_multiscale_branch_have_compatible_weights(self):
        parent_cfg = tiny_config()
        parent = CrossModalSNN(parent_cfg)
        configure_trainable(parent, parent_cfg)
        branch_cfg = tiny_config("main")
        branch_cfg["train"]["init_ckpt_path"] = "required.pt"
        branch = CrossModalSNN(branch_cfg)
        branch.load_state_dict(parent.state_dict(), strict=True)
        names = configure_trainable(branch, branch_cfg)
        self.assertTrue(all(name.startswith(RECOVERY_PREFIXES) for name in names))
        with torch.no_grad():
            first, second = parent(self.img, self.aud), branch(self.img, self.aud)
        torch.testing.assert_close(first["logits"], second["logits"])
        for name, param in parent.named_parameters():
            if name.startswith(("audio_local_cue_multiscale.16.", "audio_local_cue_multiscale.32.")):
                self.assertFalse(param.requires_grad)

    def test_coarse_gradient_survives_total_noise_miss_and_core_frozen(self):
        cfg = tiny_config("main")
        cfg["train"]["init_ckpt_path"] = "required.pt"
        model = CrossModalSNN(cfg)
        configure_trainable(model, cfg)
        force_detector(model, False)
        loss, terms = compute_losses(model, self.img, self.aud, torch.tensor([1, 2]),
                "noisy_both", cfg, torch.rand(10, 1, 28, 28), torch.rand(10, 64, 64))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(terms["img_coarse"], 0)
        for prefix in ("image_decoder.", "audio_decoder."):
            self.assertGreater(sum(float(p.grad.abs().sum()) for name, p in model.named_parameters()
                                   if name.startswith(prefix) and p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters()
                            if name.startswith(("memory.", "classifier.", "img_noise_detector."))))

    def test_no_recurrence_has_no_recurrent_weight_gradient(self):
        cfg = tiny_config("parent_no_recurrence")
        model = CrossModalSNN(cfg)
        configure_trainable(model, cfg)
        model(self.img, self.aud)["logits"].sum().backward()
        self.assertFalse(model.memory.index.use_recurrent)
        for name, param in model.named_parameters():
            if name.startswith("memory.index.W_rec."):
                self.assertIsNone(param.grad)

    def test_resume_is_exact_and_rejects_semantic_or_smoke_drift(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "scripts.train_v16pro.build_loaders", side_effect=fake_loaders), patch(
                "scripts.train_v16pro.validate_model", side_effect=tiny_validation), patch(
                "scripts.train_v16pro._runtime_sha256", return_value="fixture-runtime"):
            cfg = tiny_config()
            cfg["noise_detector"]["checkpoint"] = str(Path(directory) / "detector.pt")
            write_detector(cfg, cfg["noise_detector"]["checkpoint"])
            full, resumed = deepcopy(cfg), deepcopy(cfg)
            for name, current in (("full", full), ("resumed", resumed)):
                current["train"].update(ckpt_path=str(Path(directory) / name / "last.pt"),
                                        best_ckpt_path=str(Path(directory) / name / "best.pt"))
            train_model(full, max_batches=1)
            train_model(resumed, max_batches=1, stop_after_epoch=1)
            train_model(resumed, resume=True, max_batches=1)
            a = torch.load(full["train"]["ckpt_path"], weights_only=True)
            b = torch.load(resumed["train"]["ckpt_path"], weights_only=True)
            self.assertEqual(a["epoch"], 2)
            self.assertEqual(a["best_validation_score"], b["best_validation_score"])
            self.assertTrue(a["metadata"]["smoke_only"])
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            changed = deepcopy(resumed)
            changed["snn"]["beta"] = .8
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                train_model(changed, resume=True, max_batches=1)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                train_model(resumed, resume=True, max_batches=2)
            with self.assertRaises(FileExistsError):
                train_model(resumed, max_batches=1)

    def test_parent_lineage_is_strict_but_accepts_intended_adaptation(self):
        with tempfile.TemporaryDirectory() as directory:
            parent_cfg = tiny_config()
            parent = CrossModalSNN(parent_cfg)
            path = Path(directory) / "parent.pt"
            torch.save({"schema": SCHEMA, "cfg": parent_cfg, "model": parent.state_dict(),
                        "metadata": {"detector_sha256": "detector", "smoke_only": True,
                                     "runtime_sha256": _runtime_sha256()}}, path)
            cfg = tiny_config("no_value")
            cfg["train"]["init_ckpt_path"] = str(path)
            validate_training_config(cfg)
            model = CrossModalSNN(cfg)
            self.assertTrue(load_parent(model, cfg, "detector", True))
            with self.assertRaisesRegex(ValueError, "Smoke-only"):
                load_parent(model, cfg, "detector", False)
            with patch("scripts.train_v16pro._runtime_sha256", return_value="changed-runtime"):
                with self.assertRaisesRegex(ValueError, "runtime source"):
                    load_parent(model, cfg, "detector", True)
            cfg["ablation"]["use_recurrent"] = False
            with self.assertRaisesRegex(ValueError, "configuration/protocol mismatch"):
                load_parent(model, cfg, "detector", True)


if __name__ == "__main__":
    unittest.main()
