"""Asset integrity tests with tiny fixtures; no audio packages or model forward."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from models.frozen_base import file_sha256
from scripts.evaluate_v15pro import load_strict_checkpoint
from tools.package_v15_eval_assets import collect_dataset, package_assets, prepare_config


class FixtureDataset:
    def __init__(self, folder, split, changed=False, pair_ids=False):
        self.split, self.changed, self.pair_ids = split, changed, pair_ids
        self._indices = [7, 13, 18]
        self._fsdd_paths = {0: [str(folder / "0_fixture_0.wav")]}

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        image = torch.full((1, 28, 28), (index + int(self.changed)) / 10)
        sample = image, torch.full((4, 4), .25), 0
        return (*sample, 1000 + self._indices[index]) if self.pair_ids else sample

    def evaluation_identity(self, index):
        return dict(image_id=f"mnist_{self.split}_{self._indices[index]}",
                    audio_id="0_fixture_0.wav", label=0, speaker="fixture")


def fixture(root, seeds):
    (root / "0_fixture_0.wav").write_bytes(b"fixture source, not an actual WAV")
    cfg = dict(seed=0, data=dict(dataset="mnist_fsdd", use_mnist=True,
        paper_split=dict(enabled=True), pairing=dict(enabled=False)),
        audio=dict(use_real_audio=True, norm_mode="per_sample", sample_rate=8000,
                   duration_sec=1.0, n_mels=4, n_frames=4),
        noise_detector=dict(enabled=True), _audio_norm_stats=None)
    for seed in seeds:
        folder = root / f"seed_{seed}"
        (folder / "recovery").mkdir(parents=True)
        cfg["seed"] = seed
        (folder / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        torch.save(dict(model=torch.nn.Linear(2, 2).state_dict(), cfg=deepcopy(cfg),
                        epoch=5, opt=dict(unneeded=torch.ones(512)), sched={"step": 5},
                        noise_training=dict(smoke_only=True), frozen_base={}),
                   folder / "recovery/best.pt")
    return cfg


class AssetPackagingTests(unittest.TestCase):
    def test_lossless_audio_dedup_and_original_sample_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "0_fixture_0.wav").write_bytes(b"source")
            dataset = FixtureDataset(folder, "val")
            arrays, record = collect_dataset(dataset, limit=2)
            self.assertEqual(record["total_split_samples"], 3)
            self.assertEqual(record["samples"], 2)
            self.assertEqual(record["unique_audio"], 1)
            np.testing.assert_array_equal(arrays["sample_ids"], [0, 1])
            np.testing.assert_array_equal(arrays["image_indices"], [7, 13])
            for i in range(2):
                np.testing.assert_array_equal(arrays["images"][i], dataset[i][0])
                np.testing.assert_array_equal(
                    arrays["unique_audio"][arrays["audio_indices"][i]], dataset[i][1])
            arrays, _ = collect_dataset(FixtureDataset(folder, "test", pair_ids=True))
            np.testing.assert_array_equal(arrays["sample_ids"], [1007, 1013, 1018])

    def test_inference_subset_strict_load_and_no_reuse_when_tensors_differ(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root, output = parent / "suite", parent / "assets"
            root.mkdir()
            cfg = fixture(root, [1, 2, 3])
            hashes_before = {p: file_sha256(p) for p in root.rglob("*") if p.is_file()}
            with patch("tools.package_v15_eval_assets.build_eval_dataset",
                       side_effect=lambda prepared, split: FixtureDataset(
                           root, split, changed=prepared["seed"] == 3)):
                result = package_assets(root, output, [1, 2, 3],
                                        val_samples=2, expected_test=3, expected_val=3)
            self.assertEqual(result["status"], "complete")
            one, two, three = (result["seeds"][str(i)] for i in [1, 2, 3])
            for split in ("test", "val"):
                self.assertEqual(one["datasets"][split]["file"], two["datasets"][split]["file"])
                self.assertNotEqual(one["datasets"][split]["file"], three["datasets"][split]["file"])
                self.assertFalse(result["cross_seed_identical"][split])
                self.assertEqual(one["datasets"][split]["metadata_sha256"],
                                 three["datasets"][split]["metadata_sha256"])
            cfg["seed"] = 1
            model = torch.nn.Linear(2, 2)
            packed = load_strict_checkpoint(model, output / one["checkpoint"], cfg, "cpu")
            self.assertNotIn("opt", packed)
            self.assertNotIn("sched", packed)
            self.assertTrue(packed["asset_packaging"]["inference_only"])
            self.assertFalse(packed["asset_packaging"]["resume_supported"])
            original = torch.load(root / "seed_1/recovery/best.pt", weights_only=True)
            for name, value in original["model"].items():
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
            for path, digest in hashes_before.items():
                self.assertEqual(digest, file_sha256(path))
            for relative, digest in result["artifacts"].items():
                self.assertEqual(file_sha256(output / relative), digest)
            with np.load(output / one["datasets"]["test"]["file"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["images"].shape, (3, 1, 28, 28))
            with self.assertRaises(FileExistsError):
                package_assets(root, output, [1])

    def test_refuses_augmentation_and_missing_global_training_stats(self):
        cfg = dict(data=dict(dataset="mnist_fsdd", paper_split=dict(enabled=True),
                            pairing=dict(enabled=True)),
                   audio=dict(use_real_audio=True), noise_detector=dict(enabled=True))
        with self.assertRaisesRegex(ValueError, "unaugmented"):
            prepare_config(cfg, deepcopy(cfg))
        cfg["data"]["pairing"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "training-only"):
            prepare_config(cfg, deepcopy(cfg))


if __name__ == "__main__":
    unittest.main()
