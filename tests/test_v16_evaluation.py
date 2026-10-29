"""Meaningful v16 evaluation guards: truth isolation, counts, pairing and export."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from scripts.v16_evaluation import (conditions, evaluate, region_metrics,
                                    batch_region_metrics, selection_metrics)
from tools.v16_report import aggregate_reports, task_summary, save_examples


class TinyData(Dataset):
    split, train = "test", False

    def __init__(self, split="test"):
        self.split = split
        generator = torch.Generator().manual_seed(66)
        self.img = torch.rand(2, 1, 8, 8, generator=generator)
        self.aud = torch.rand(2, 8, 8, generator=generator)
        self.prototype_img = self.img.repeat(5, 1, 1, 1)
        self.prototype_aud = self.aud.repeat(5, 1, 1)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.img[index], self.aud[index], torch.tensor(index), torch.tensor(100 + index)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.noise_mask_source = "predicted"
        self.calls = []

    def forward(self, x_img_cue=None, x_aud_cue=None, training_mode=False,
                img_cue_mask=None, aud_cue_mask=None, **kwargs):
        self.calls.append(kwargs)
        reference = x_img_cue if x_img_cue is not None else x_aud_cue
        size = len(reference)
        out = {"logits": torch.zeros(size, 10)}
        for name, cue, known, shape in (("img", x_img_cue, img_cue_mask, (size, 1, 8, 8)),
                                        ("aud", x_aud_cue, aud_cue_mask, (size, 8, 8))):
            coarse = torch.full(shape, .5)
            pred = torch.zeros_like(cue) if cue is not None else None
            repair = known
            if self.noise_mask_source == "oracle" and cue is not None:
                repair = torch.maximum(known, kwargs[name + "_noise_mask"])
            recovered = coarse if cue is None else repair * coarse + (1 - repair) * cue
            out["recovered_" + name] = torch.logit(recovered.clamp(.000001, .999999)) if name == "img" else recovered
            out["recovered_" + name + "_coarse"] = torch.logit(coarse) if name == "img" else coarse
            out[name + "_predicted_mask"] = pred
            out[name + "_repair_mask"] = repair
        return out


class EvaluationTests(unittest.TestCase):
    def test_export_device_contract_matches_training_suite(self):
        from tools.export_v16pro import main
        with patch("tools.export_v16pro.export_suite") as run:
            main(["--root", "outputs/example", "--device", "cuda:0", "--models", "main", "--skip-scans"])
            self.assertEqual(run.call_args.kwargs["device"], "cuda:0")
            self.assertEqual(run.call_args.kwargs["models"], ["main"])
            self.assertIs(run.call_args.kwargs["scans"], False)

    def test_checkpoint_provenance_and_smoke_guards(self):
        from tools.export_v16pro import strict_checkpoint
        from scripts.train_v16pro import _runtime_sha256
        model = TinyModel()
        cfg = dict(seed=1234, v16={"enabled": True}, train={"stage": "main"},
                   noise_detector={"enabled": True})
        state = dict(schema="v16pro.training.v1", cfg=deepcopy(cfg), model=model.state_dict(),
                     metadata={"runtime_sha256": _runtime_sha256(), "smoke_only": False})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save(state, path)
            strict_checkpoint(model, path, cfg, "cpu", "main")
            changed = deepcopy(cfg)
            changed["v16"]["no_value"] = True
            with self.assertRaisesRegex(ValueError, "semantic"):
                strict_checkpoint(model, path, changed, "cpu", "main")
            state["metadata"]["smoke_only"] = True
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "Smoke-only"):
                strict_checkpoint(model, path, cfg, "cpu", "main")
            strict_checkpoint(model, path, cfg, "cpu", "main", allow_smoke=True)
            state["metadata"]["runtime_sha256"] = "changed"
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                strict_checkpoint(model, path, cfg, "cpu", "main", allow_smoke=True)

    def test_true_regions_and_missing_excluded_from_detection(self):
        target = torch.zeros(1, 2, 2)
        missing = torch.tensor([[[1., 0.], [0., 0.]]])
        noise = torch.tensor([[[0., 1.], [0., 0.]]])
        predicted = torch.ones_like(target)
        values = region_metrics(target, {"input": torch.ones_like(target), "recovered": target},
                                missing, noise, predicted, "P")
        self.assertEqual(values["missing_pixels"], 1)
        self.assertEqual(values["noise_pixels"], 1)
        self.assertEqual(values["clean_pixels"], 2)
        self.assertEqual(values["detector_tp"], 1)
        self.assertEqual(values["detector_fp"], 2)  # missing pixel not a false positive
        self.assertEqual(values["noise_input_mse"], 1)
        batched = batch_region_metrics(target[None], {"input": torch.ones_like(target)[None], "recovered": target[None]},
                                       missing[None], noise[None], predicted[None], "P")[0]
        self.assertEqual(values, batched)

    def test_absence_has_no_observed_or_clean_error(self):
        target = torch.zeros(1, 2, 2)
        values = region_metrics(target, {"input": None, "recovered": target}, None, None, None, "A")
        self.assertIsNone(values["global_input_mse"])
        self.assertIsNone(values["clean_recovered_mse"])
        self.assertEqual(values["observed_pixels"], 0)
        self.assertEqual(sum(values["detector_" + k] for k in ("tp", "fp", "fn", "tn")), 0)

    def test_deterministic_batching_and_truth_does_not_enter_forward(self):
        cfg = {"v16": {"evaluation_seed": 77}}
        one, two = TinyModel(), TinyModel()
        a, examples = evaluate(one, DataLoader(TinyData(), batch_size=1), cfg, "cpu", examples=1)
        b, _ = evaluate(two, DataLoader(TinyData(), batch_size=2), cfg, "cpu")
        self.assertEqual(a, b)
        self.assertTrue(all(not kwargs for kwargs in one.calls))
        self.assertEqual(a["n_unique_samples"], 2)
        self.assertTrue(all(len(items) == 1 for items in examples.values()))
        self.assertTrue(all(items[0]["sample_id"] == 100 for items in examples.values()))
        # RNG/module-mode restoration, including when a child has its own mode.
        self.assertTrue(one.training)
        self.assertEqual(one.noise_mask_source, "predicted")
        for row in a["conditions"]:
            for modality in ("img", "aud"):
                if row[modality + "_state"] == "A":
                    self.assertEqual(row[modality + "_target_kind"], "category")
                    self.assertIsNone(row[modality + "_detector_false_positive_rate"])

    def test_oracle_explicit_and_validation_no_test_selection(self):
        model = TinyModel()
        cfg = {"v16": {"evaluation_seed": 77}}
        report, _ = evaluate(model, DataLoader(TinyData("val"), batch_size=2), cfg, "cpu",
                             split="val", validation=True, mask_source="oracle")
        self.assertTrue(all(set(kwargs) == {"img_noise_mask", "aud_noise_mask"} for kwargs in model.calls))
        self.assertEqual(set(selection_metrics(report, cfg)), {"score", "acc", "img_mse", "aud_mse"})
        with self.assertRaises(ValueError):
            evaluate(model, DataLoader(TinyData(), batch_size=2), cfg, "cpu", split="val")

    def test_conditions_scans_and_no_fake_sd(self):
        cases = conditions({}, scans=True)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertEqual(len([c for c in cases if c["mode"] == "clean_both"]), 1)
        self.assertEqual({c["noise_std"] for c in cases if c["sweep"] == "gaussian_std"}, {.1, .2, .3})
        with self.assertRaises(ValueError):
            conditions({"v16": {"missing_sweep": [.4, .2]}}, scans=True)
        report, examples = evaluate(TinyModel(), DataLoader(TinyData(), batch_size=2), {}, "cpu", examples=1)
        clean_metrics = {row["metric"] for row in task_summary(report) if row["task"] == "clean_decoder"}
        self.assertIn("coarse_mse", clean_metrics)
        self.assertNotIn("hard_template_mse", clean_metrics)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in ("figures", "tables", "logs"):
                (output / name).mkdir()
            paths = aggregate_reports([(1234, "main", report)], output)
            self.assertTrue(all(path.is_file() for path in paths))
            text = (output / "tables/task_summary_mean_sd.csv").read_text()
            self.assertIn("sample_sd", text)
            with (output / "tables/task_summary_mean_sd.csv").open(newline="") as stream:
                self.assertTrue(all(row["sample_sd"] == "" and row["n_seeds"] == "1" for row in csv.DictReader(stream)))
            # Rendering uses captured tensors; no model or dataset rerun.
            subset = dict(list(examples.items())[:1])
            paths = save_examples(subset, output)
            self.assertTrue(all(path.is_file() for path in paths))
            manifest = json.loads((output / "tables/examples_manifest.json").read_text())
            self.assertEqual(manifest[0]["sample_id"], 100)


if __name__ == "__main__":
    unittest.main()
