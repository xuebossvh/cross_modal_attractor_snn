"""Synthetic metadata/cache tests only; no real inference or scientific results."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools import report_v15_p4 as report


def fake_branches():
    """Fabricated records for integrity checks; never represent an actual run."""
    initial = {key: "FAKE" for key in ("initial_model_sha256", "initial_optimizer_sha256",
        "initial_optimizer_moments_sha256", "source_scheduler_sha256", "source_optimizer_steps",
        "trainable_names", "frozen_names")}
    common = {key: "FAKE" for key in ("config", "config_sha256", "driver_sha256", "state_helper_sha256",
        "packages", "torch_cuda_build", "device", "normalization", "split_sha256")}
    common.update(source_sha256="FAKE_SOURCE", source_epoch=99, runtime_sha256=report.ORIGINAL_RUNTIME_SHA256,
        threads=2, cuda_memory_fraction=.6, seed=1234, epochs=30, smoke_only=False, max_batches=0,
        initialization=initial)
    common["config"] = {"data": {"batch_size": 128}}
    bundles = {}
    for branch in report.BRANCHES:
        identity = dict(deepcopy(common), branch=branch)
        starts = [1e-5, 1e-5] if branch == "keep_lr" else [5e-5, 2.5e-5]
        identity["initialization"]["parameter_groups"] = [dict(index=i, original_lr_multiplier=multiplier,
            initial_lr=start, parameters=[f"FAKE_{i}"]) for i, (multiplier, start) in enumerate(zip((1., .5), starts))]
        records = []
        for i, epoch in enumerate(range(100, 130)):
            score = 1 / (i + 1)
            records.append(dict(epoch=epoch, smoke_only=False,
                validation_report=dict(split="val", mask_source="predicted", n_unique_samples=5999,
                                       conditions=[{"FAKE": True}] * 7),
                validation=dict(score=score, img_mse=score, aud_mse=score),
                epoch_loss=score,
                lr_groups=[1e-5 + (start - 1e-5) * (1 + math.cos(math.pi * i / 30)) / 2 for start in starts],
                next_lr_groups=[1e-5 + (start - 1e-5) * (1 + math.cos(math.pi * (i + 1) / 30)) / 2 for start in starts],
                is_best=True, best_epoch=epoch,
                best_validation_score=score, optimizer_updates=421, samples=53888, batches=421,
                total_branch_updates=421 * (i + 1), total_branch_samples=53888 * (i + 1),
                clean_batch_stream_sha256=f"FAKE_STREAM_{epoch}", epoch_seed=1234 + epoch * 1009,
                cue_mode_batches={"FAKE": 421}))
        bundles[branch] = dict(manifest=dict(identity=identity), epochs=records,
            status=dict(complete=True, last_epoch=129, smoke_only=False, best_epoch=129,
                        best_validation_score=1 / 30, total_branch_updates=421 * 30))
    return bundles


def fake_result(value=.2):
    def stats(region):
        pixels = 0 if region == "corrupt_background" else 8
        return dict(pixels=pixels, sse=value * pixels, mse=value if pixels else None,
                    saturation_ratio=.1 if pixels else None,
                    horizontal=dict(edges=1, gradient_mse=.3))
    return {"FAKE_CASE": dict(n=60, case_sha256="FAKE_CASE_HASH", total_loss=.5,
        index_accuracy=.75, audio_refiner_bypassed=True, policy="foreground", family="fake_noise",
        cue_mode="corrupt_both", modalities={
            modality: {stage: {region: stats(region) for region in report.REGIONS} for stage in stages}
            for modality, stages in (("img", report.STAGES), ("aud", ("input", "coarse", "final")))})}


class P4ReportTests(unittest.TestCase):
    def test_fairness_rejects_incomplete_smoke_source_and_stream_drift(self):
        bundles = fake_branches()
        result = report.verify_branches(bundles, "FAKE_SOURCE", 5999, 54001)
        self.assertEqual(result["matched_epochs"], 30)
        self.assertEqual(result["full_validation_samples"], 5999)
        for corruption in ("short", "smoke", "source", "optimizer", "stream", "val60", "budget", "best", "lr"):
            changed = deepcopy(bundles)
            branch = changed["restart_lr"]
            if corruption == "short":
                branch["epochs"].pop()
            elif corruption == "smoke":
                branch["manifest"]["identity"]["smoke_only"] = True
            elif corruption == "source":
                branch["manifest"]["identity"]["source_sha256"] = "OTHER"
            elif corruption == "optimizer":
                branch["manifest"]["identity"]["initialization"]["initial_optimizer_sha256"] = "OTHER"
            elif corruption == "stream":
                branch["epochs"][3]["clean_batch_stream_sha256"] = "OTHER"
            elif corruption == "val60":
                branch["epochs"][0]["validation_report"]["n_unique_samples"] = 60
            elif corruption == "budget":
                branch["epochs"][2]["optimizer_updates"] -= 1
            elif corruption == "best":
                branch["epochs"][2]["is_best"] = False
            elif corruption == "lr":
                branch["epochs"][2]["lr_groups"][0] = 0.001
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                report.verify_branches(changed, "FAKE_SOURCE", 5999, 54001)

    def test_tiny_requires_predeclared_full_protocol_not_one_update_smoke(self):
        identity = dict(schema="v15-p4-tiny.v1", updates=1000, eval_every=100, seed=1234, base_learning_rate=5e-5,
                        batch_size=60, source_checkpoint_sha256="FAKE_SOURCE", source_epoch_zero_based=99,
                        source_runtime_sha256=report.ORIGINAL_RUNTIME_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for field, value in (("updates", 1), ("eval_every", 10), ("seed", 1504), ("base_learning_rate", .001)):
                changed = dict(identity, **{field: value})
                report.atomic_json(folder / "manifest.json", dict(identity=changed, status="complete", completed_updates=changed["updates"]))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "different source/protocol"):
                    report.verify_tiny(folder, "FAKE_SOURCE", {}, {})

    def test_fake_complete_tiny_requires_all_endpoint_sample_artifacts(self):
        """1000 fabricated metadata rows; no model updates or real samples are run."""
        arrays = {"labels": np.arange(60) % 10, "sample_ids": np.arange(60)}
        datasets = {split: deepcopy(arrays) for split in report.SPLITS}
        cases = {split: {"FAKE_CASE": {"sha256": "FAKE_CASE_HASH"}} for split in report.SPLITS}
        identity = dict(schema="v15-p4-tiny.v1", updates=1000, eval_every=100, seed=1234,
            base_learning_rate=5e-5, batch_size=60, source_checkpoint_sha256="FAKE_SOURCE",
            source_epoch_zero_based=99, source_runtime_sha256=report.ORIGINAL_RUNTIME_SHA256,
            diagnostic_source_sha256={"v15_p4_diagnostics.py": report.file_sha256(report.ROOT / "tools/v15_p4_diagnostics.py")},
            datasets={split: {"arrays_sha256": report.array_sha256(arrays), "sample_ids": arrays["sample_ids"].tolist()}
                      for split in report.SPLITS},
            case_sha256={split: {"FAKE_CASE": "FAKE_CASE_HASH"} for split in report.SPLITS},
            schedule=["FAKE_CASE"] * 1000)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "checkpoints").mkdir()
            checkpoint = folder / "checkpoints/last.pt"
            torch.save(dict(identity=identity, completed_updates=1000,
                            source_checkpoint_sha256="FAKE_SOURCE", p4_diagnostic_only=True), checkpoint)
            (folder / "history.jsonl").write_text("".join(json.dumps(dict(update=i, condition="FAKE_CASE",
                case_sha256="FAKE_CASE_HASH", loss=.5)) + "\n" for i in range(1, 1001)), encoding="utf-8")
            cue_files = {}
            for split in report.SPLITS:
                for extension in ("npz", "json"):
                    path = folder / f"cue_cache/{split}/FAKE_CASE.{extension}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"FAKE_CUE_FILE_NOT_REAL_DATA")
                    cue_files[path.relative_to(folder).as_posix()] = report.file_sha256(path)
            for update in range(0, 1001, 100):
                path = folder / f"evaluations/step_{update:06d}.json"
                path.parent.mkdir(exist_ok=True)
                report.atomic_json(path, dict(update=update, splits={split: fake_result() for split in report.SPLITS}))
                if update in (0, 1000):
                    for split in report.SPLITS:
                        sample = folder / f"samples/step_{update:06d}/{split}/FAKE_CASE.npz"
                        sample.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(sample, fake_only=True)
            report.atomic_json(folder / "metrics.json", report.read_json(folder / "evaluations/step_001000.json"))
            artifacts = {p.relative_to(folder).as_posix(): report.file_sha256(p) for p in folder.rglob("*")
                         if p.is_file() and not p.is_relative_to(folder / "cue_cache") and p != checkpoint}
            manifest = dict(identity=identity, status="complete", completed_updates=1000,
                cue_cache_files=cue_files, completed_artifacts_sha256=artifacts,
                last_checkpoint_sha256=report.file_sha256(checkpoint))
            report.atomic_json(folder / "manifest.json", manifest)
            result = report.verify_tiny(folder, "FAKE_SOURCE", datasets, cases)
            self.assertEqual(len(result["evaluations"]), 11)
            self.assertEqual(len(result["history"]), 1000)
            del manifest["completed_artifacts_sha256"]["samples/step_001000/val/FAKE_CASE.npz"]
            report.atomic_json(folder / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "samples are absent"):
                report.verify_tiny(folder, "FAKE_SOURCE", datasets, cases)

    def test_complete_cache_skips_only_matching_and_untampered_stage(self):
        arrays = {"labels": np.arange(60) % 10}
        cases = {"FAKE_CASE": {"sha256": "FAKE_CASE_HASH"}}
        identity = dict(source="FAKE_SOURCE", model="FAKE_MODEL", split="val", fixture_only=True)
        def fake_evaluate(*args, sample_directory, **kwargs):
            Path(sample_directory).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(sample_directory) / "FAKE_CASE.npz", fixture_only=np.array([True]))
            return fake_result()
        with tempfile.TemporaryDirectory() as temporary, patch.object(report, "evaluate_cases", side_effect=fake_evaluate) as evaluate:
            folder = Path(temporary) / "cache"
            a = report.evaluate_cached(folder, None, arrays, cases, {}, "cpu", identity, 1504)
            b = report.evaluate_cached(folder, None, arrays, cases, {}, "cpu", identity, 1504)
            self.assertEqual(a, b)
            self.assertEqual(evaluate.call_count, 1)
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                report.evaluate_cached(folder, None, arrays, cases, {}, "cpu", dict(identity, model="OTHER"), 1504)
            (folder / "metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                report.evaluate_cached(folder, None, arrays, cases, {}, "cpu", identity, 1504)

    def test_partial_cache_is_recomputed_with_same_identity(self):
        arrays = {"labels": np.arange(60) % 10}
        cases = {"FAKE_CASE": {"sha256": "FAKE_CASE_HASH"}}
        identity = {"source": "FAKE", "fixture_only": True}
        def fake_evaluate(*args, sample_directory, **kwargs):
            Path(sample_directory).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(sample_directory) / "FAKE_CASE.npz", fixture_only=np.array([True]))
            return fake_result()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            report.atomic_json(folder / "identity.json", identity)
            (folder / "samples").mkdir()
            (folder / "samples/FAKE_CASE.npz").write_bytes(b"interrupted placeholder, not a real NPZ")
            with patch.object(report, "evaluate_cases", side_effect=fake_evaluate) as evaluate:
                report.evaluate_cached(folder, None, arrays, cases, {}, "cpu", identity, 1504)
                self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(report.read_json(folder / "completion.json")["status"], "complete")

    def test_stages_preserve_empty_regions_audio_bypass_and_descriptive_differences(self):
        evaluations = {"source_last": {"train": fake_result(.2), "val": fake_result(.3)},
                       "tiny_final": {"train": fake_result(.1), "val": fake_result(.4)}}
        rows = report.stage_rows(evaluations)
        train = next(row for row in rows if row["checkpoint"] == "tiny_final"
                     and row["split"] == "train" and row["modality"] == "img" and row["region"] == "global")
        self.assertAlmostEqual(train["final_minus_source_last_mse"], -.1)
        empty = next(row for row in rows if row["region"] == "corrupt_background")
        self.assertIsNone(empty["final_minus_source_last_mse"])
        audio = next(row for row in rows if row["modality"] == "aud")
        self.assertNotIn("refined", audio["stages"])
        bundles = fake_branches()
        summary = dict(scope={"full_validation_pairs": 5999}, diagnostics=evaluations,
                       stage_rows=rows, full_validation={b: {"epochs": bundles[b]["epochs"]} for b in report.BRANCHES},
                       tiny={"updates": 1000, "evaluation_files": [{"update": 0, "path": "FAKE_ONLY"}]})
        page = report.render_html(summary)
        self.assertIn("完整验证集为 5999 对", page)
        self.assertIn("val60 是固定诊断子集", page)
        self.assertIn("不能作为恢复内容的独立语义识别率", page)
        self.assertIn("音频 refiner 保持旁路", page)

    def test_paths_and_nonfinite_json_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "escapes"):
                report.child(temporary, "../outside.json")
            path = Path(temporary) / "bad.json"
            path.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                report.read_json(path)


if __name__ == "__main__":
    unittest.main()
