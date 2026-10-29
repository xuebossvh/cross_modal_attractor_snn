"""Synthetic CPU integration checks; no FSDD/MNIST downloads or paper claims."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.noise_detector import LocalNoiseDetector, load_pretrained_detectors
from scripts.train_noise_detector import choose_device, main, train_detectors


def fake_loaders(cfg, eval_split):
    if eval_split != "val":
        raise AssertionError("test data must never be used for detector selection")
    generator = torch.Generator().manual_seed(17)
    images = torch.rand(4, 1, 8, 8, generator=generator)
    audio = torch.rand(4, 8, 8, generator=generator)
    labels = torch.arange(4)
    loaders = []
    for split in ("train", "val"):
        dataset = TensorDataset(images, audio, labels)
        dataset.split = split
        loaders.append(DataLoader(dataset, batch_size=2, shuffle=split == "train",
                                  generator=torch.Generator().manual_seed(5)))
    return tuple(loaders)


class DetectorPretrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, directory):
        return {
            "seed": 1234, "device": "cpu", "train": {"require_cuda": False},
            "data": {"dataset": "mnist_fsdd", "batch_size": 2, "num_workers": 0,
                     "paper_split": {"enabled": True, "seed": 99}},
            "audio": {"use_real_audio": False, "n_mels": 8, "n_frames": 8},
            "dims": {"num_classes": 4},
            "corruption": {"img_mode": "local_gaussian", "aud_mode": "local_impulse",
                           "img_train_modes": ["local_gaussian", "local_impulse"],
                           "aud_train_modes": ["local_impulse", "local_gaussian"],
                           "train_severity": 0.3, "noise_std": 0.2,
                           "curriculum_mode": "fixed", "family_sampling": "balanced"},
            "noise_detector": {"enabled": True, "hidden_channels": 4, "threshold": 0.5,
                               "positive_weight": 3.0, "pretrain_lr": 0.001,
                               "pretrain_epochs": 2, "checkpoint": str(Path(directory) / "best.pt")},
        }

    def fake_model(self):
        model = nn.Module()
        model.img_noise_detector = LocalNoiseDetector(4)
        model.aud_noise_detector = LocalNoiseDetector(4)
        return model

    def test_checkpoint_loading_smoke_guard_and_threshold_preservation(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "scripts.train_noise_detector.build_loaders", side_effect=fake_loaders):
            cfg = self.config(directory)
            result = train_detectors(cfg, epochs=1, max_batches=1)
            self.assertTrue(result["smoke_only"])
            self.assertTrue(Path(result["checkpoint"]).is_file())
            self.assertTrue(Path(result["last_checkpoint"]).is_file())
            model = self.fake_model()
            with self.assertRaisesRegex(ValueError, "Smoke-only"):
                load_pretrained_detectors(model, cfg)
            cfg["noise_detector"].update(allow_smoke_checkpoint=True, threshold=0.7)
            state = load_pretrained_detectors(model, cfg)
            self.assertEqual(cfg["noise_detector"]["threshold"], 0.7)
            self.assertEqual(state["schema"], "local-noise-detector.v1")
            self.assertEqual(state["metadata"]["selection_split"], "val")
            for key, actual in model.img_noise_detector.state_dict().items():
                torch.testing.assert_close(actual, state["detectors"]["img"][key])
            cfg["corruption"]["noise_std"] = 0.4
            with self.assertRaisesRegex(ValueError, "corruption"):
                load_pretrained_detectors(model, cfg)

    def test_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "scripts.train_noise_detector.build_loaders", side_effect=fake_loaders):
            uninterrupted = self.config(Path(directory) / "full")
            resumed = self.config(Path(directory) / "resumed")
            train_detectors(uninterrupted, epochs=2, max_batches=1)
            train_detectors(resumed, epochs=1, max_batches=1)
            train_detectors(resumed, epochs=2, resume=True, max_batches=1)
            first = torch.load(Path(directory) / "full/last.pt", weights_only=False)
            second = torch.load(Path(directory) / "resumed/last.pt", weights_only=False)
            self.assertEqual(first["epoch"], second["epoch"])
            self.assertEqual(first["best_validation_loss"], second["best_validation_loss"])
            for modality in ("img", "aud"):
                for key in first["detectors"][modality]:
                    torch.testing.assert_close(first["detectors"][modality][key],
                                               second["detectors"][modality][key], rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                train_detectors(resumed, epochs=2, max_batches=1)
            changed = deepcopy(resumed)
            changed["noise_detector"]["positive_weight"] = 5
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                train_detectors(changed, epochs=3, resume=True, max_batches=1)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                train_detectors(resumed, epochs=3, resume=True, max_batches=2)

    def test_fail_closed_device_and_data_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            cfg["train"]["require_cuda"] = True
            with patch("torch.cuda.is_available", return_value=False), self.assertRaises(RuntimeError):
                choose_device(cfg)
            cfg["train"]["require_cuda"] = False
            cfg["data"]["paper_split"]["enabled"] = False
            with self.assertRaisesRegex(ValueError, "separate val"):
                train_detectors(cfg)
            cfg["data"]["paper_split"]["enabled"] = True

            def wrong_split(config, eval_split):
                train, val = fake_loaders(config, eval_split)
                val.dataset.split = "test"
                return train, val

            with patch("scripts.train_noise_detector.build_loaders", side_effect=wrong_split):
                with self.assertRaisesRegex(ValueError, "verified val"):
                    train_detectors(cfg)

    def test_missing_detector_is_fatal_and_cli_forwards_options(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            with self.assertRaises(FileNotFoundError):
                load_pretrained_detectors(self.fake_model(), cfg)
            with patch("scripts.train_noise_detector.load_config", return_value=cfg), patch(
                    "scripts.train_noise_detector.fix_console_encoding"), patch(
                    "scripts.train_noise_detector.train_detectors") as train:
                main(["--config", "example.yaml", "--epochs", "3", "--resume", "--max-batches", "2"])
            train.assert_called_once_with(cfg, 3, True, 2)


if __name__ == "__main__":
    unittest.main()
