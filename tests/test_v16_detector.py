"""CPU-only detector smoke tests and exact epoch-resume reproducibility."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.noise_detector import LocalNoiseDetector, load_pretrained_detectors
from scripts.train_detector_v16pro import (main, train_detectors,
                                           validate_v16_detector_checkpoint)


def fake_loaders(cfg, eval_split):
    assert eval_split == "val", "No test data may select detector weights"
    generator = torch.Generator().manual_seed(42)
    image = torch.rand(4, 1, 8, 8, generator=generator)
    audio = torch.rand(4, 8, 8, generator=generator)
    labels, ids = torch.arange(4), torch.arange(4) + 100
    result = []
    for split in ("train", "val"):
        dataset = TensorDataset(image, audio, labels, ids)
        dataset.split = split
        result.append(DataLoader(dataset, batch_size=2, shuffle=split == "train",
                                 generator=torch.Generator().manual_seed(7)))
    return tuple(result)


class V16DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, directory):
        return {"seed": 1234, "device": "cpu", "train": {"require_cuda": False},
                "data": {"dataset": "mnist_fsdd", "batch_size": 2, "num_workers": 0,
                         "paper_split": {"enabled": True}},
                "audio": {"use_real_audio": False, "n_mels": 8, "n_frames": 8},
                "dims": {"num_classes": 4}, "v16": {},
                "noise_detector": {"enabled": True, "hidden_channels": 2,
                                   "checkpoint": str(Path(directory) / "best.pt"),
                                   "pretrain_epochs": 2, "pretrain_lr": 0.001,
                                   "pretrain_lr_gamma": 0.8, "positive_weight": 3.0}}

    def test_checkpoint_compatibility_and_smoke_provenance(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "scripts.train_detector_v16pro.build_loaders", side_effect=fake_loaders):
            cfg = self.config(directory)
            result = train_detectors(cfg, epochs=1, max_batches=1)
            self.assertTrue(result["smoke_only"])
            state = torch.load(result["checkpoint"], weights_only=False)
            self.assertEqual(state["metadata"]["missing_policy"], "excluded_from_bce")
            self.assertEqual(state["metadata"]["selection_split"], "val")
            self.assertIn("rng", state)
            self.assertIn("scheduler", state)
            with self.assertRaisesRegex(ValueError, "Smoke-only"):
                validate_v16_detector_checkpoint(state, cfg)
            cfg["noise_detector"]["allow_smoke_checkpoint"] = True
            validate_v16_detector_checkpoint(state, cfg)
            model = nn.Module()
            model.img_noise_detector = LocalNoiseDetector(2)
            model.aud_noise_detector = LocalNoiseDetector(2)
            load_pretrained_detectors(model, cfg)
            for key, value in model.img_noise_detector.state_dict().items():
                torch.testing.assert_close(value, state["detectors"]["img"][key], rtol=0, atol=0)
            cfg["v16"]["noise_coverage"] = 0.5
            with self.assertRaisesRegex(ValueError, "protocol"):
                validate_v16_detector_checkpoint(state, cfg)

    def test_resume_exact_and_reject_semantic_change(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "scripts.train_detector_v16pro.build_loaders", side_effect=fake_loaders):
            full, split = self.config(Path(directory) / "full"), self.config(Path(directory) / "split")
            train_detectors(full, epochs=2, max_batches=1)
            train_detectors(split, epochs=1, max_batches=1)
            train_detectors(split, epochs=2, resume=True, max_batches=1)
            one = torch.load(Path(directory) / "full/last.pt", weights_only=False)
            two = torch.load(Path(directory) / "split/last.pt", weights_only=False)
            self.assertEqual(one["best_validation_loss"], two["best_validation_loss"])
            self.assertEqual(one["scheduler"], two["scheduler"])
            for modality in ("img", "aud"):
                for key in one["detectors"][modality]:
                    torch.testing.assert_close(one["detectors"][modality][key], two["detectors"][modality][key], rtol=0, atol=0)
            changed = deepcopy(split)
            changed["v16"]["noise_std"] = 0.5
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                train_detectors(changed, epochs=3, resume=True, max_batches=1)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                train_detectors(split, epochs=3, resume=True, max_batches=2)
            with self.assertRaises(FileExistsError):
                train_detectors(split, epochs=2, max_batches=1)

    def test_fail_closed_split_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            cfg["data"]["paper_split"]["enabled"] = False
            with self.assertRaisesRegex(ValueError, "validation split"):
                train_detectors(cfg)
            cfg["data"]["paper_split"]["enabled"] = True

            def wrong_split(config, eval_split):
                train, val = fake_loaders(config, eval_split)
                val.dataset.split = "test"
                return train, val

            with patch("scripts.train_detector_v16pro.build_loaders", side_effect=wrong_split):
                with self.assertRaisesRegex(ValueError, "verified val"):
                    train_detectors(cfg)
            with patch("scripts.train_detector_v16pro.load_config", return_value=cfg), patch(
                    "scripts.train_detector_v16pro.fix_console_encoding"), patch(
                    "scripts.train_detector_v16pro.train_detectors") as run:
                main(["--config", "config.yaml", "--epochs", "2", "--resume", "--max-batches", "1"])
            run.assert_called_once_with(cfg, 2, True, 1)


if __name__ == "__main__":
    unittest.main()
