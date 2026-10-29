"""End-to-end export contracts on explicitly synthetic, tiny CPU fixtures."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from tools.export_v15pro import main, run_export


class SmallModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))
        self.noise_mask_source = "predicted"

    def forward(self, x_img_cue, x_aud_cue, training_mode=False,
                img_cue_mask=None, aud_cue_mask=None):
        masks = {"img": torch.zeros_like(x_img_cue), "aud": torch.zeros_like(x_aud_cue)}
        applied = masks if self.noise_mask_source == "predicted" else (
            {"img": img_cue_mask, "aud": aud_cue_mask} if self.noise_mask_source == "oracle"
            else {"img": torch.ones_like(x_img_cue), "aud": torch.ones_like(x_aud_cue)})
        return {"logits": torch.zeros(len(x_img_cue), 10),
                "recovered_img": torch.zeros_like(x_img_cue) + self.offset,
                "recovered_aud": torch.full_like(x_aud_cue, .5) + self.offset,
                "img_predicted_mask": masks["img"], "aud_predicted_mask": masks["aud"],
                "img_repair_mask": applied["img"], "aud_repair_mask": applied["aud"]}


def fixture(folder):
    cfg = {"device": "cpu", "seed": 1234,
           "data": {"dataset": "mnist_fsdd", "paper_split": {"enabled": True}},
           "noise_detector": {"enabled": True, "threshold": .5, "hidden_channels": 4},
           "corruption": {"img_mode": "local_gaussian", "aud_mode": "local_gaussian",
                          "train_severity": .25, "noise_std": .2},
           "noise_evaluation": {"seed": 11, "families": ["local_gaussian"],
                                "cue_modes": ["clean_both", "corrupt_both"]}}
    config = folder / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    checkpoint = folder / "best.pt"
    torch.save({"model": SmallModel(cfg).state_dict(), "cfg": copy.deepcopy(cfg),
                "epoch": 0, "noise_training": {"smoke_only": True}}, checkpoint)
    dataset = TensorDataset(torch.full((2, 1, 8, 8), .3), torch.full((2, 8, 8), .4),
                            torch.tensor([0, 1]))
    dataset.split = "test"
    return config, checkpoint, DataLoader(dataset, batch_size=2)


class ExportCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_export_has_three_parts_and_verifiable_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config, checkpoint, loader = fixture(folder)
            output = folder / "inference"
            with patch("tools.export_v15pro.build_loaders", return_value=(None, loader)) as build, \
                    patch("tools.export_v15pro.CrossModalSNN", SmallModel), redirect_stdout(io.StringIO()):
                record = run_export(config, checkpoint, output, examples=1, max_samples=1)
                first_calls = build.call_count
                repeated = run_export(config, checkpoint, output, examples=1, max_samples=1)
                self.assertEqual(build.call_count, first_calls)
            self.assertEqual(record, repeated)
            self.assertEqual({p.name for p in output.iterdir()}, {"figures", "logs", "tables"})
            self.assertGreater(len(list((output / "figures").glob("*.png"))), 2)
            self.assertIn("[export] complete", (output / "logs/evaluation.log").read_text())
            report = json.loads((output / "tables/summary.json").read_text())
            self.assertTrue(report["smoke_training"])
            self.assertTrue(report["partial_evaluation"])
            self.assertEqual(report["n_unique_samples"], 1)
            self.assertEqual(record["status"], "complete")
            with patch("tools.export_v15pro.runtime_source_fingerprint", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    run_export(config, checkpoint, output, examples=1, max_samples=1)
            # A changed per-sample table cannot be silently reused as a valid export.
            table = output / "tables/per_sample.csv"
            table.write_text(table.read_text() + "tampered\n")
            with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
                run_export(config, checkpoint, output, examples=1, max_samples=1)

    def test_missing_checkpoint_does_not_create_export(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            output = folder / "inference"
            with self.assertRaises(FileNotFoundError):
                run_export(folder / "absent.yaml", folder / "absent.pt", output)
            self.assertFalse(output.exists())

    def test_root_mode_routes_each_seed_without_starting_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed = root / "seed_1234"
            (seed / "recovery").mkdir(parents=True)
            (seed / "config.yaml").write_text("{}")
            (seed / "recovery/best.pt").write_bytes(b"fixture")
            with patch("tools.export_v15pro.run_export") as export:
                main(["--root", str(root), "--seeds", "1234", "--examples", "4"])
            self.assertEqual(export.call_count, 1)
            self.assertEqual(export.call_args.args,
                             (seed / "config.yaml", seed / "recovery/best.pt", seed / "inference"))
            self.assertEqual(export.call_args.kwargs["mask_source"], "predicted")
            self.assertEqual(export.call_args.kwargs["examples"], 4)

    def test_unrelated_output_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config, checkpoint, _ = fixture(folder)
            output = folder / "inference"
            output.mkdir()
            note = output / "notes.txt"
            note.write_text("keep")
            with self.assertRaises(FileExistsError):
                run_export(config, checkpoint, output)
            self.assertEqual(note.read_text(), "keep")

    def test_parent_config_change_invalidates_completed_export(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            parent, checkpoint, loader = fixture(folder)
            child = folder / "child.yaml"
            child.write_text("extends: config.yaml\n")
            output = folder / "inference"
            with patch("tools.export_v15pro.build_loaders", return_value=(None, loader)), \
                    patch("tools.export_v15pro.CrossModalSNN", SmallModel), redirect_stdout(io.StringIO()):
                run_export(child, checkpoint, output, examples=0, max_samples=1)
            cfg = yaml.safe_load(parent.read_text())
            cfg["noise_detector"]["threshold"] = .7
            parent.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                run_export(child, checkpoint, output, examples=0, max_samples=1)


if __name__ == "__main__":
    unittest.main()
