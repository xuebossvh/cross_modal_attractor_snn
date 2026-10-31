"""Synthetic fixtures test rendering contracts; these are not experiment results."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.v15_joint_panels import (_metrics, _planes, joint_artifact_paths,
                                   render_joint_sample)


def synthetic_fixture():
    """Small labelled synthetic fixture, never part of a formal result gallery."""
    planes = {}
    for modality, shape in (("img", (8, 8)), ("aud", (6, 10))):
        target = np.full(shape, 0.5, dtype=np.float32)
        support = np.zeros(shape, dtype=np.uint8)
        support[:2, :3] = 1
        cue = target.copy()
        cue[support == 1] = 0.75
        recovered = target.copy()
        recovered[support == 1] = 0.625
        planes[modality] = dict(target=target, cue=cue, true_mask=support,
                                predicted_mask=np.zeros_like(support),
                                repair_mask=np.ones_like(support), recovered=recovered)
    metadata = dict(sample_id="synthetic-fixture-not-results", index=0, label=2,
                    index_prediction=3, training_seed=7, policy="fixed_position",
                    family="local_gaussian", cue_mode="corrupt_both", severity=0.2,
                    noise_std=0.1, source_kind="synthetic unit-test fixture",
                    coverage={"img": {"foreground_coverage": 0.25},
                              "aud": {"foreground_coverage": None}})
    return planes, metadata


class JointPanelDataTests(unittest.TestCase):
    def test_metrics_are_per_modality_and_empty_regions_are_none(self):
        arrays, metadata = synthetic_fixture()
        arrays["aud"]["true_mask"][:] = 0
        metrics = _metrics(_planes(arrays), metadata)
        self.assertEqual(metrics["img"]["corrupted_pixels"], 6)
        self.assertEqual(metrics["img"]["cue_corrupted_mse"], 0.0625)
        self.assertEqual(metrics["img"]["recovered_corrupted_mse"], 0.015625)
        self.assertIsNone(metrics["aud"]["cue_corrupted_mse"])
        self.assertIsNone(metrics["aud"]["foreground_coverage"])

    def test_rejects_nonbinary_nonfinite_mismatched_and_batched_arrays(self):
        bad_values = (np.full((8, 8), 0.5), np.full((8, 8), np.nan),
                      np.zeros((9, 8)), np.zeros((2, 8, 8)))
        for bad in bad_values:
            with self.subTest(shape=bad.shape):
                arrays, _ = synthetic_fixture()
                arrays["img"]["repair_mask"] = bad
                with self.assertRaises(ValueError):
                    _planes(arrays)

    def test_preserves_planes_and_accepts_singleton_channel(self):
        arrays, _ = synthetic_fixture()
        expected = arrays["img"]["target"].copy()
        arrays["img"]["target"] = arrays["img"]["target"][None]
        actual = _planes(arrays)
        np.testing.assert_array_equal(actual["img"]["target"], expected)
        actual["img"]["target"][:] = 0
        np.testing.assert_array_equal(arrays["img"]["target"][0], expected)

    def test_invalid_input_creates_no_outputs(self):
        arrays, metadata = synthetic_fixture()
        arrays["img"]["cue"][0, 0] = 2
        with tempfile.TemporaryDirectory() as root:
            stem = Path(root) / "synthetic"
            with self.assertRaises(ValueError):
                render_joint_sample(stem, arrays, metadata)
            self.assertFalse(any(path.exists() for path in joint_artifact_paths(stem)))


class JointPanelRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit_scripts = os.environ.get("NATURE_FIGURE_AUDIT_SCRIPTS")
        available = cls.audit_scripts or importlib.util.find_spec("audit_panel_alignment")
        pdf_available = importlib.util.find_spec("pymupdf") or importlib.util.find_spec("fitz")
        if not available or not pdf_available:
            raise unittest.SkipTest("Integration rendering requires nature-figure audits and PyMuPDF")

    def test_bundle_is_audited_replayable_and_refuses_overwrite(self):
        arrays, metadata = synthetic_fixture()
        with tempfile.TemporaryDirectory() as root:
            stem = Path(root) / "synthetic_not_results"
            paths = render_joint_sample(stem, arrays, metadata, audit_scripts=self.audit_scripts)
            self.assertEqual(len(paths), 7)
            self.assertTrue(all(path.is_file() for path in paths))
            saved = json.loads(paths[3].read_text(encoding="utf-8"))
            self.assertEqual(saved["render_contract"]["audio_origin"], "lower")
            self.assertEqual(saved["render_metrics"]["img"]["cue_corrupted_mse"], 0.0625)
            with np.load(paths[2], allow_pickle=False) as archive:
                self.assertEqual(len(archive.files), 12)
                replay = {m: {f: archive[f"{m}_{f}"] for f in arrays[m]} for m in arrays}
                for modality, values in arrays.items():
                    for field, array in values.items():
                        np.testing.assert_array_equal(archive[f"{modality}_{field}"], array)
            qa = json.loads(paths[5].read_text(encoding="utf-8"))
            self.assertEqual(qa["alignment_verdict"], "PASS")
            self.assertGreaterEqual(qa["pdf_text"]["minimum_found_pt"], 5)
            self.assertEqual(qa["pdf_collisions"]["verdict"], "PASS")
            before = [path.read_bytes() for path in paths]
            with self.assertRaises(FileExistsError):
                render_joint_sample(stem, arrays, metadata, audit_scripts=self.audit_scripts)
            self.assertEqual([path.read_bytes() for path in paths], before)
            second = render_joint_sample(Path(root) / "redrawn_synthetic", replay, saved,
                                         audit_scripts=self.audit_scripts)
            self.assertEqual(json.loads(second[3].read_text(encoding="utf-8"))["render_metrics"],
                             saved["render_metrics"])
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
            with fitz.open(paths[1]) as document:
                text = document[0].get_text()
                self.assertEqual(text.count("Index prediction"), 1)
                self.assertIn("Input cue", text)
                self.assertNotIn("Noisy cue", text)
                self.assertIn("evaluation only", text)
                self.assertIn("n/a", text)
                self.assertIn("split=unspecified", text)

    def test_failed_render_releases_lock_and_publishes_nothing(self):
        arrays, metadata = synthetic_fixture()
        with tempfile.TemporaryDirectory() as root:
            stem = Path(root) / "deliberate_failure"
            with patch("tools.v15_joint_panels._draw", side_effect=RuntimeError("fixture failure")):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    render_joint_sample(stem, arrays, metadata, audit_scripts=self.audit_scripts)
            self.assertEqual(list(Path(root).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
