"""Contracts for real-forward example export and support provenance."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.noise_metrics import deterministic_cues, preserve_random_state
from tools.v15_examples import export_examples


def config(modes=None):
    return {"noise_detector": {"enabled": True, "threshold": 0.5},
            "corruption": {"img_mode": "local_gaussian", "aud_mode": "local_gaussian",
                           "train_severity": 0.25, "noise_std": 0.2},
            "noise_evaluation": {"seed": 123, "families": ["local_gaussian"],
                                 "cue_modes": modes or ["clean_both", "corrupt_both"]}}


def loader(batch_size=2, shuffle=False, n=6):
    image = torch.linspace(0.1, 0.9, n * 64).reshape(n, 1, 8, 8)
    audio = torch.linspace(0.9, 0.1, n * 64).reshape(n, 8, 8)
    return DataLoader(TensorDataset(image, audio, torch.arange(n), torch.arange(n) + 101),
                      batch_size=batch_size, shuffle=shuffle)


class FakeDetectorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = torch.nn.Identity()
        self.noise_mask_source = "predicted"
        self.calls = []

    def forward(self, x_img_cue, x_aud_cue, training_mode,
                img_cue_mask=None, aud_cue_mask=None):
        self.calls.append(dict(source=self.noise_mask_source,
                               img_mask=img_cue_mask, aud_mask=aud_cue_mask,
                               img_cue=x_img_cue.clone(), aud_cue=x_aud_cue.clone()))
        out = {"recovered_img": torch.zeros_like(x_img_cue),
               "recovered_aud": torch.full_like(x_aud_cue, 0.25)}
        for modality, cue, oracle in (("img", x_img_cue, img_cue_mask),
                                     ("aud", x_aud_cue, aud_cue_mask)):
            predicted = (cue > 0.6).to(cue.dtype)
            out[f"{modality}_predicted_mask"] = predicted
            out[f"{modality}_repair_mask"] = (
                oracle if self.noise_mask_source == "oracle" else
                torch.ones_like(cue) if self.noise_mask_source == "all" else predicted)
        return out


class ExampleExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_predicted_and_all_never_receive_truth_and_display_model_masks(self):
        for source in ("predicted", "all", "oracle"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as root:
                model = FakeDetectorModel()
                captured = []
                def capture(path, columns, modality, *args):
                    captured.append((modality, columns))
                with patch("tools.v15_examples._save_page", side_effect=capture):
                    export_examples(model, loader(), config(), "cpu", root,
                                    mask_source=source, num_examples=2)
                self.assertEqual(len(model.calls), 2)
                self.assertEqual(model.noise_mask_source, "predicted")
                for call in model.calls:
                    for modality in ("img", "aud"):
                        if source == "oracle":
                            self.assertIsNotNone(call[f"{modality}_mask"])
                        else:
                            self.assertIsNone(call[f"{modality}_mask"])
                for modality, columns in captured:
                    target, cue, truth, predicted, applied, recovered = columns
                    self.assertTrue(torch.equal(predicted, (cue > 0.6).float()))
                    expected = truth if source == "oracle" else (
                        torch.ones_like(cue) if source == "all" else predicted)
                    self.assertTrue(torch.equal(applied, expected))
                    self.assertTrue(torch.equal(recovered, torch.full_like(target,
                                                0.5 if modality == "img" else 0.25)))
                # Clean ground-truth support is zero while detector output need not be.
                self.assertEqual(int(captured[0][1][2].sum()), 0)
                self.assertGreater(int(captured[1][1][3].sum()), 0)

    def test_selected_ids_corruption_and_pagination_ignore_batch_size_and_shuffle(self):
        manifests, all_calls, page_indices = [], [], []
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for batch_size, shuffled, root in ((1, False, first), (6, True, second)):
                model = FakeDetectorModel()
                indices = []
                def capture(path, columns, modality, family, mode, source, selected, ids, labels):
                    indices.append(selected)
                with patch("tools.v15_examples._save_page", side_effect=capture):
                    paths = export_examples(model, loader(batch_size, shuffled),
                                            config(["corrupt_both"]), "cpu", root,
                                            num_examples=5)
                manifests.append(json.loads(paths[-1].read_text(encoding="utf-8")))
                all_calls.append(model.calls)
                page_indices.append(indices)
                self.assertEqual(len(paths), 5)  # 2 modalities x 2 pages + manifest
            self.assertEqual(manifests[0], manifests[1])
            self.assertEqual([row["sample_id"] for row in manifests[0]["examples"]],
                             [101, 102, 103, 104, 105])
            self.assertEqual(page_indices[0], [[0, 1, 2, 3], [4], [0, 1, 2, 3], [4]])
            self.assertEqual(page_indices[0], page_indices[1])
            for modality in ("img", "aud"):
                self.assertTrue(torch.equal(all_calls[0][0][f"{modality}_cue"],
                                            all_calls[1][0][f"{modality}_cue"]))
            # The exact same seed/id protocol as numeric evaluation is used.
            image, audio, _, ids = next(iter(loader(6)))
            with preserve_random_state():
                expected_i, expected_a, _ = deterministic_cues(
                    image[:5], audio[:5], ids[:5].tolist(), "corrupt_both", "local_gaussian",
                    config(), 0.25, 123)
            self.assertTrue(torch.equal(all_calls[0][0]["img_cue"], expected_i))
            self.assertTrue(torch.equal(all_calls[0][0]["aud_cue"], expected_a))

    def test_rng_mode_and_source_restore_even_on_shape_error(self):
        model = FakeDetectorModel()
        model.train()
        model.detector.eval()
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        before = random.getstate(), np.random.get_state(), torch.get_rng_state()
        original = model.forward
        def broken(*args, **kwargs):
            out = original(*args, **kwargs)
            out["img_predicted_mask"] = torch.zeros(1)
            return out
        with tempfile.TemporaryDirectory() as root:
            with patch("tools.v15_examples._save_page"):
                export_examples(model, loader(), config(), "cpu", root,
                                mask_source="all", num_examples=2)
            with patch.object(model, "forward", side_effect=broken):
                with self.assertRaisesRegex(ValueError, "shape"):
                    export_examples(model, loader(), config(), "cpu", root,
                                    mask_source="oracle", num_examples=1)
        self.assertTrue(model.training)
        self.assertFalse(model.detector.training)
        self.assertEqual(model.noise_mask_source, "predicted")
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_equal(np.random.get_state(), before[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_real_png_and_absent_clean_support(self):
        actual_cues = deterministic_cues
        def clean_none(*args, **kwargs):
            image, audio, _ = actual_cues(*args, **kwargs)
            return image, audio, {"img": None, "aud": None}
        with tempfile.TemporaryDirectory() as root:
            with patch("tools.v15_examples.deterministic_cues", side_effect=clean_none):
                paths = export_examples(FakeDetectorModel(), loader(), config(["clean_both"]),
                                        "cpu", root, mask_source="oracle", num_examples=1)
            self.assertEqual(len(paths), 3)
            for path in paths[:-1]:
                self.assertTrue(path.exists())
                with Image.open(path) as picture:
                    self.assertEqual(picture.format, "PNG")
                    self.assertGreater(picture.width, 1000)
                    self.assertGreater(picture.height, 300)
                    picture.verify()
            manifest = json.loads(paths[-1].read_text(encoding="utf-8"))
            self.assertEqual(manifest["display_range"], [0, 1])
            self.assertIn("evaluation only", manifest["truth_support_usage"])
            for name in manifest["examples"][0]["filenames"].values():
                self.assertTrue((Path(root) / name).exists())

    def test_zero_skip_and_invalid_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "not-created"
            self.assertEqual(export_examples(None, None, {}, "cpu", output, num_examples=0), [])
            self.assertFalse(output.exists())
            for options in ({"num_examples": -1}, {"split": "train"},
                            {"mask_source": "unknown"}):
                with self.assertRaises(ValueError):
                    export_examples(FakeDetectorModel(), loader(), config(), "cpu", output, **options)
            bad_modes = config(["corrupt_img_only"])
            with self.assertRaises(ValueError):
                export_examples(FakeDetectorModel(), loader(), bad_modes, "cpu", output)


if __name__ == "__main__":
    unittest.main()
