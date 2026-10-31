"""P4 packaging contracts with tiny local fixtures; no downloads or training."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from data.splits import norm_fingerprint
from models.frozen_base import file_sha256
from tools.package_v15_p4_assets import (
    FixedCleanSubset, build_clean_dataset, package_assets, pairing_metadata,
    select_balanced, training_data_fingerprint, validate_training_checkpoint,
    verify_mapped_training_data,
)


class TinyDataset:
    def __init__(self, folder, split):
        self.split, self.train, self.return_pair_id = split, split == "train", False
        self._indices = list(range(80)) if self.train else list(range(100, 180))
        self._fsdd_paths = {label: [str(folder / f"{label}_fixture_{index}.wav")
            for index in ((10, 11, 12) if self.train else (5, 6, 7))] for label in range(10)}
        self._fsdd = {label: [torch.full((4, 4), (label * 3 + i) / 32)
                             for i in range(3)] for label in range(10)}
        self._base = self

    def __len__(self):
        return len(self._indices)

    def _label_for_item(self, index):
        return int(index) % 10

    def __getitem__(self, index):
        return torch.full((1, 28, 28), (index % 100) / 100), self._label_for_item(index)


def fixture(root):
    root.mkdir()
    for label in range(10):
        for index in (5, 6, 7, 10, 11, 12):
            (root / f"{label}_fixture_{index}.wav").write_bytes(f"{label}:{index}".encode())
    cfg = dict(seed=1234, dims=dict(num_classes=10),
        data=dict(root=str(root), dataset="mnist_fsdd", use_mnist=True,
                  paper_split=dict(enabled=True, mode="official", seed=5, val_fraction=.1),
                  pairing=dict(enabled=False, return_pair_id=False)),
        audio=dict(fsdd_root=str(root), use_real_audio=True, norm_mode="global", auto_download=False,
                   sample_rate=8000, duration_sec=1.0, n_mels=4, n_frames=4, noise_std=.3),
        noise_detector=dict(enabled=True), train=dict(epochs=100, epoch_seeded=True))
    paths = sorted(str(p) for p in root.glob("*.wav") if int(p.stem.split("_")[-1]) >= 10)
    stats = dict(lo=.1, hi=2., n_mels=4, n_frames=4, n_wavs=len(paths),
                 train_fingerprint=norm_fingerprint(paths, cfg))
    cfg["_audio_norm_stats"] = stats
    cfg["audio"]["norm_stats_path"] = str(root / "norm.pt")
    torch.save(stats, root / "norm.pt")
    source = root / "seed_1234"
    (source / "recovery").mkdir(parents=True)
    (source / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.00001)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    state = dict(model=model.state_dict(), cfg=deepcopy(cfg), epoch=99,
                 opt=optimizer.state_dict(), sched=dict(last_epoch=100, T_max=100),
                 noise_training=dict(version="v15pro", smoke_only=False), frozen_base={})
    torch.save(state, source / "recovery/last.pt")
    state["epoch"] = 90
    torch.save(state, source / "recovery/best.pt")
    return cfg, paths


class P4AssetTests(unittest.TestCase):
    def test_balanced_selection_is_stable_and_reports_audio_shortfall(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = TinyDataset(Path(temp), "train")
            records = pairing_metadata(dataset, 1234)
            chosen, classes = select_balanced(records, per_class=6)
            repeat, _ = select_balanced(list(reversed(records)), per_class=6)
            self.assertEqual(chosen, repeat)
            self.assertEqual(len(chosen), 60)
            self.assertEqual(len({r["image_index"] for r in chosen}), 60)
            for label, info in classes.items():
                self.assertEqual(info["samples"], 6)
                self.assertGreaterEqual(info["audio_diversity_shortfall"], 3)
                self.assertTrue(info["shortfall_reason"])
                original = {r["position"]: r for r in records}
                for row in chosen:
                    self.assertEqual(row, original[row["position"]])
            with self.assertRaisesRegex(ValueError, "fewer"):
                select_balanced(records, per_class=9)

    def test_train_pairing_replays_original_rng_and_val_original_formula(self):
        with tempfile.TemporaryDirectory() as temp:
            for split in ("train", "val"):
                dataset = TinyDataset(Path(temp), split)
                rng = np.random.default_rng(1234)
                records = pairing_metadata(dataset, 1234)
                subset = FixedCleanSubset(dataset, records)
                for row, position in zip(records, range(len(dataset))):
                    expected = (int(rng.integers(0, 3)) if split == "train" else
                                (row["image_index"] * 104729 + row["label"] * 1009) % 3)
                    self.assertEqual(row["audio_pool_index"], expected)
                    sample = subset[position]
                    torch.testing.assert_close(sample[1], dataset._fsdd[row["label"]][expected], rtol=0, atol=0)
                    self.assertEqual(sample[3], position)
                    self.assertEqual(int(row["audio_id"].split("_")[0]), row["label"])

    def test_test_split_is_rejected_before_any_dataset_loading(self):
        with self.assertRaisesRegex(ValueError, "test access"):
            build_clean_dataset({}, "test")
        with self.assertRaisesRegex(ValueError, "prohibit test"):
            pairing_metadata(SimpleNamespace(split="test"), 1234)

    def test_complete_checkpoint_copy_preserves_epoch_optimizer_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / "source", Path(temp) / "assets"
            cfg, paths = fixture(root)
            before = {p: file_sha256(p) for p in root.rglob("*") if p.is_file()}
            requested_splits = []
            def build(prepared, split):
                requested_splits.append(split)
                return TinyDataset(root, split)
            with patch("tools.package_v15_eval_assets._fsdd_train_wav_paths", return_value=paths), \
                 patch("tools.package_v15_p4_assets.build_clean_dataset", side_effect=build), \
                 patch("tools.package_v15_p4_assets.training_data_fingerprint", return_value={"content_sha256": "fixture"}):
                manifest = package_assets(root, output)
            self.assertEqual(requested_splits, ["train", "val"])
            self.assertEqual(set(manifest["datasets"]), {"train", "val"})
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["split_validation"]["train_val_disjoint"])
            self.assertFalse(manifest["split_validation"]["test_tensors_loaded"])
            original = torch.load(root / "seed_1234/recovery/last.pt", weights_only=True)
            copied = torch.load(output / manifest["checkpoint"], weights_only=True)
            self.assertEqual(copied["epoch"], 99)
            self.assertEqual(set(copied), set(original))
            self.assertEqual(file_sha256(output / manifest["checkpoint"]), before[root / "seed_1234/recovery/last.pt"])
            self.assertEqual(copied["opt"]["param_groups"], original["opt"]["param_groups"])
            for parameter, values in original["opt"]["state"].items():
                for name, tensor in values.items():
                    torch.testing.assert_close(copied["opt"]["state"][parameter][name], tensor, rtol=0, atol=0)
            best = torch.load(output / manifest["source_best_checkpoint"]["file"], weights_only=True)
            self.assertTrue(best["asset_packaging"]["inference_only"])
            with self.assertRaisesRegex(ValueError, "inference-only"):
                validate_training_checkpoint(best)
            for split in ("train", "val"):
                record = manifest["datasets"][split]
                self.assertEqual(record["samples"], 60)
                with np.load(output / record["file"], allow_pickle=False) as arrays:
                    self.assertEqual(set(arrays.files), {"images", "unique_audio", "audio_indices", "labels",
                        "sample_ids", "image_indices", "image_ids", "audio_ids", "speakers"})
                    np.testing.assert_array_equal(np.bincount(arrays["labels"]), np.full(10, 6))
                    self.assertEqual(arrays["images"].shape, (60, 1, 28, 28))
            for path, digest in before.items():
                self.assertEqual(file_sha256(path), digest)
            for path, digest in manifest["artifacts"].items():
                self.assertEqual(file_sha256(output / path), digest)
            with self.assertRaises(FileExistsError):
                package_assets(root, output)

    def test_training_checkpoint_rejects_wrong_epoch_or_missing_optimizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            fixture(root)
            state = torch.load(root / "seed_1234/recovery/last.pt", weights_only=True)
            self.assertEqual(validate_training_checkpoint(state)["completed_epochs"], 100)
            state["epoch"] = 98
            with self.assertRaisesRegex(ValueError, "epoch=99"):
                validate_training_checkpoint(state)
            state["epoch"] = 99
            state["opt"] = {}
            with self.assertRaisesRegex(ValueError, "optimizer"):
                validate_training_checkpoint(state)

    def test_path_migration_preserves_fingerprint_and_rejects_changed_norm_or_wav(self):
        with tempfile.TemporaryDirectory() as temp:
            source, migrated = Path(temp) / "source", Path(temp) / "gpu_data"
            cfg, paths = fixture(source)
            shutil.copytree(source, migrated)
            raw = torch.arange(8, dtype=torch.uint8).reshape(2, 2, 2)
            def mnist(root, train, download):
                self.assertTrue(train)
                self.assertFalse(download)
                return SimpleNamespace(data=raw, targets=torch.tensor([0, 1]), raw_folder=str(Path(root) / "raw"))
            # Patch only MNIST I/O; WAV discovery, file hashes and norm hash run for real.
            with patch("torchvision.datasets.MNIST", side_effect=mnist):
                mapped = deepcopy(cfg)
                mapped["data"]["root"] = str(migrated)
                mapped["audio"]["fsdd_root"] = str(migrated)
                mapped["audio"]["norm_stats_path"] = str(migrated / "norm.pt")
                # fsdd_recordings_dir accepts roots ending in recordings only.
                with patch("data.fsdd.fsdd_recordings_dir", side_effect=lambda c: c["audio"]["fsdd_root"]):
                    original = training_data_fingerprint(cfg)
                    manifest = dict(normalization=dict(stats=cfg["_audio_norm_stats"]), training_data=original,
                                    audio_preprocessing=cfg["audio"], data_protocol=cfg["data"])
                    verified = verify_mapped_training_data(mapped, manifest, migrated / "norm.pt")
                    self.assertTrue(verified["verified"])
                    self.assertFalse(verified["normalization_recomputed"])
                    wrong_path = deepcopy(mapped)
                    wrong_path["audio"]["norm_stats_path"] = str(source / "norm.pt")
                    with self.assertRaisesRegex(ValueError, "must point"):
                        verify_mapped_training_data(wrong_path, manifest, migrated / "norm.pt")
                    drifted = deepcopy(mapped)
                    drifted["audio"]["norm_mode"] = "per_sample"
                    with self.assertRaisesRegex(ValueError, "configuration changed"):
                        verify_mapped_training_data(drifted, manifest, migrated / "norm.pt")
                    changed = deepcopy(cfg["_audio_norm_stats"])
                    changed["hi"] = 3.
                    torch.save(changed, migrated / "norm.pt")
                    with self.assertRaisesRegex(ValueError, "normalization differs"):
                        verify_mapped_training_data(mapped, manifest, migrated / "norm.pt")
                    torch.save(cfg["_audio_norm_stats"], migrated / "norm.pt")
                    (migrated / Path(paths[0]).name).write_bytes(b"changed training waveform")
                    with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                        verify_mapped_training_data(mapped, manifest, migrated / "norm.pt")


if __name__ == "__main__":
    unittest.main()
