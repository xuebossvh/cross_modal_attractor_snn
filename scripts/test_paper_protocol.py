"""Fast CPU tests; synthetic fixtures are never publication results."""

import bootstrap  # noqa: F401
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common import CUE_MODES, build_cue, load_config, select_targets
from data.audio_features import ensure_audio_norm_stats
from data.dataset import PairedAudioVisualDataset, build_loaders
from data.splits import audio_split, image_indices, norm_fingerprint
from models.memory import RecurrentIndexLayer
from models.paper_baselines import CleanRecognizers, RecoveryCNN
from models.frozen_base import file_sha256
from scripts.paper_evaluate import global_ssim, reconstruction_metrics, sample_cues
from scripts.paper_statistics import cluster_interval, paired_difference
from scripts.run_v13pro_suite import build_plan, completed
from scripts.paper_validation import validate_recovery
from paths import PROJECT_ROOT


class FakeMNIST:
    def __init__(self, root=None, train=True, download=False, transform=None):
        self.targets = torch.arange(100) % 10
        self.data = self.targets[:, None, None].expand(-1, 28, 28).byte() * 20

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i].float().unsqueeze(0)/255, int(self.targets[i])


class PaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.base = load_config(PROJECT_ROOT / "configs/v13pro.yaml")

    def config(self):
        cfg = deepcopy(self.base)
        cfg["device"] = "cpu"
        cfg["data"].update(num_workers=0, batch_size=10)
        cfg["data"]["paper_split"] = dict(enabled=True, seed=7, val_fraction=.2, mode="official")
        return cfg

    def test_audio_disjoint(self):
        cfg = self.config()
        groups = {s: {i for i in range(50) if audio_split(f"3_jackson_{i}.wav", cfg) == s}
                  for s in ("train", "val", "test")}
        self.assertEqual([len(groups[s]) for s in groups], [40, 5, 5])
        self.assertFalse(groups["train"] & groups["val"])
        cfg["data"]["paper_split"].update(mode="speaker", val_speakers=["nicolas"], test_speakers=["jackson"])
        self.assertEqual(audio_split("3_jackson_49.wav", cfg), "test")
        self.assertEqual(audio_split("3_nicolas_0.wav", cfg), "val")
        self.assertEqual(audio_split("3_other_0.wav", cfg), "train")
        cfg["data"]["paper_split"]["val_speakers"] = ["jackson"]
        with self.assertRaises(ValueError):
            audio_split("3_jackson_0.wav", cfg)

    def test_stratified_images(self):
        cfg = self.config()
        labels = np.arange(100) % 10
        train, val = (image_indices(labels, s, cfg) for s in ("train", "val"))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(100)))
        self.assertEqual(np.bincount(labels[val]).tolist(), [2]*10)
        self.assertEqual(val, image_indices(labels, "val", cfg))

    def test_loaders_use_only_train_medoids_and_val_images(self):
        cfg = self.config()
        cfg["audio"]["use_real_audio"] = False
        with patch("torchvision.datasets.MNIST", FakeMNIST):
            train, val = build_loaders(cfg, eval_split="val")
            self.assertEqual(len(train.dataset), 80)
            self.assertEqual(len(val.dataset), 20)
            self.assertFalse(set(train.dataset._indices) & set(val.dataset._indices))
            self.assertIs(train.dataset.prototype_img, val.dataset.prototype_img)
            self.assertEqual(val.dataset.split, "val")
            self.assertEqual(val.dataset[0][1].tolist(), val.dataset[0][1].tolist())
        with patch("torchvision.datasets.MNIST", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "no synthetic fallback"):
                PairedAudioVisualDataset(cfg)

    def test_norm_cache_fingerprint(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp)/"0_a_10.wav"
            wav.write_bytes(b"fixture")
            fp = norm_fingerprint([wav], cfg)
            cfg["audio"]["norm_stats_path"] = str(Path(tmp)/"stats.pt")
            stale = dict(n_mels=64, n_frames=64, lo=0., hi=1., train_fingerprint="wrong")
            torch.save(stale, cfg["audio"]["norm_stats_path"])
            fresh = dict(stale, train_fingerprint=fp)
            with patch("data.audio_features._fsdd_train_wav_paths", return_value=[wav]), patch(
                    "data.audio_features.compute_audio_norm_stats", return_value=dict(fresh, n_wavs=1)) as compute:
                self.assertEqual(ensure_audio_norm_stats(cfg)["train_fingerprint"], fp)
                self.assertEqual(ensure_audio_norm_stats(cfg)["train_fingerprint"], fp)
                self.assertEqual(compute.call_count, 1)

    def test_index_default_and_full_current_withdrawal(self):
        layer = RecurrentIndexLayer(3, 3, 5, use_kwta=False)
        key = torch.rand(6, 2, 3)
        default, mean = layer(key, key)
        trace = layer(key, key, return_trace=True)
        self.assertTrue(torch.equal(default, trace["spikes"]))
        self.assertTrue(torch.equal(mean, trace["rate"]))
        with torch.no_grad():
            layer.W_img_to_A.bias.fill_(100.)
            layer.W_rec.weight.zero_()
        withdrawn = layer(key, external_steps=0, extra_steps=2, return_trace=True)
        self.assertEqual(withdrawn["spikes"].shape, (8, 2, 5))
        self.assertEqual(float(withdrawn["spikes"].sum()), 0.)
        self.assertEqual(float(withdrawn["voltage"].abs().sum()), 0.)

    def test_plan_equal_budget_and_full_lineage(self):
        configs, training, testing = build_plan(self.base, PROJECT_ROOT/"tmp"/"paper_plan", seeds=[7], mechanism=True)
        by_name = {cfg["paper"]["experiment"]: cfg for cfg in configs.values()}
        for name in ("control", "main", "no_causal", "no_cross"):
            self.assertEqual(by_name[name]["train"]["epochs"], 30)
            self.assertEqual(by_name[name]["train"]["init_ckpt_path"], by_name["main"]["train"]["init_ckpt_path"])
        self.assertEqual(by_name["parent"]["train"]["init_ckpt_path"], "")
        self.assertFalse(by_name["parent_no_recurrence"]["ablation"]["use_recurrent"])
        self.assertFalse(by_name["parent_no_kwta"]["ablation"]["use_kwta"])
        self.assertFalse(by_name["control"]["audio_local_cue"]["multiscale"])
        self.assertTrue(by_name["main"]["audio_local_cue"]["multiscale"])
        self.assertEqual(len(training), 13)
        self.assertEqual(len(testing), 9)
        self.assertNotEqual(by_name["recognizer"]["train"]["ckpt_path"], by_name["classifier"]["train"]["ckpt_path"])

    def test_holdout_removed_from_all_training(self):
        configs, _, _ = build_plan(self.base, PROJECT_ROOT/"tmp"/"paper_plan", seeds=[7], holdout="partial_temporal")
        for cfg in configs.values():
            self.assertNotIn("partial_temporal", cfg["corruption"]["aud_train_modes"])
            self.assertIn("partial_temporal", cfg["corruption"]["eval_fixed"]["aud_modes"])

    def test_mask_batch_size_independence(self):
        cfg = self.config()
        img, aud = torch.rand(4,1,28,28), torch.rand(4,64,64)
        ids = [dict(image_id=f"test_{i}") for i in range(4)]
        for protocol in ("fixed", "random"):
            whole = sample_cues(img, aud, ids, "corrupt_both", cfg, .4, ("occlusion", "time_mask"), 5, protocol)
            pieces = [sample_cues(img[i:i+1], aud[i:i+1], ids[i:i+1], "corrupt_both", cfg, .4,
                                 ("occlusion", "time_mask"), 5, protocol) for i in range(4)]
            for j in (0,1):
                self.assertTrue(torch.equal(whole[j], torch.cat([p[j] for p in pieces])))

    def test_baseline_all_cues_and_pasteback_gradients(self):
        cfg = self.config()
        img, aud, labels = torch.rand(2,1,28,28), torch.rand(2,64,64), torch.tensor([1,2])
        pi, pa = torch.rand(10,1,28,28), torch.rand(10,64,64)
        model = RecoveryCNN()
        for mode in CUE_MODES:
            ci, ca, masks = build_cue(img, aud, mode, cfg, .4, return_masks=True)
            logits, ri, ra = model(ci, ca, masks, pi, pa)
            ti, ta, ik, ak = select_targets(mode, img, aud, pi, pa, labels)
            self.assertEqual(ri.shape, img.shape)
            self.assertEqual(ra.shape, aud.shape)
            for rec, cue, mask in ((ri,ci,masks["img"]),(ra,ca,masks["aud"])):
                if cue is not None and mask is not None:
                    self.assertTrue(torch.equal(rec[mask==0], cue[mask==0]))
            loss = (ri-ti).square().mean()+(ra-ta).square().mean()+logits.square().mean()
            loss.backward()
        self.assertIsNotNone(model.aud_decoder.up[-1].weight.grad)

    def test_metrics_na_and_global_ssim(self):
        x = torch.rand(2,1,28,28)
        a = torch.rand(2,64,64)
        rec = CleanRecognizers()
        labels = torch.tensor([0,1])
        result = reconstruction_metrics(x,a,x,a,{"img":None,"aud":torch.ones_like(a)},rec,labels)
        self.assertTrue(torch.isnan(result["img_missing_mse"]).all())
        self.assertTrue(torch.isnan(result["aud_visible_mse"]).all())
        self.assertTrue(torch.equal(result["aud_missing_mse"], torch.zeros(2)))
        self.assertTrue(torch.allclose(global_ssim(x,x), torch.ones(2)))

    def test_paired_cluster_and_coverage(self):
        result = cluster_interval([-1.,-1.,-3.], ["a","a","b"], draws=100)
        self.assertEqual(result["mean"], -2.)
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result, cluster_interval([-1.,-1.,-3.], ["a","a","b"], draws=100))
        with self.assertRaises(ValueError):
            paired_difference({"x":(1.,"a","time")}, {})

    def test_done_marker_checks_artifact_and_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, marker = Path(tmp)/"result", Path(tmp)/"done.json"
            path.write_text("result", encoding="utf-8")
            job = dict(output=str(path))
            marker.write_text(json.dumps(dict(fingerprint="plan", output_sha256=file_sha256(path))), encoding="utf-8")
            self.assertTrue(completed(marker, job, "plan"))
            with self.assertRaises(RuntimeError):
                completed(marker, job, "new_plan")
            path.write_text("changed", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                completed(marker, job, "plan")

    def test_category_validation_without_pair_id(self):
        cfg = self.config()
        cfg["audio"]["use_real_audio"] = False
        with patch("torchvision.datasets.MNIST", FakeMNIST):
            _, val = build_loaders(cfg, eval_split="val")
        result = validate_recovery(RecoveryCNN(), val, cfg, torch.device("cpu"), cnn=True)
        self.assertTrue(np.isfinite(result["score"]))
        self.assertEqual(result["n"], 60)

    def test_parent_and_child_training_checkpoint_roundtrip(self):
        import sys
        import yaml
        from scripts.train import main
        cfg = self.config()
        cfg["audio"]["use_real_audio"] = False
        cfg["data"]["batch_size"] = 40
        cfg["snn"].update(T=2, aud_decoder_base_ch=32, aud_decoder_refine_blocks=0,
                          aud_conv_ch1=4, aud_conv_ch2=8)
        cfg["dims"].update(img_hidden=32, aud_hidden=32, D_img=16, D_aud=16,
                           N_key_img=8, N_key_aud=8, N_index=16, N_value_img=16, N_value_aud=32)
        cfg["detail_conditioning"].update(img_detail_dim=8, aud_detail_dim=8)
        cfg["train"]["require_cuda"] = False
        with tempfile.TemporaryDirectory() as tmp:
            configs, _, _ = build_plan(cfg, tmp, seeds=[3], parent_epochs=1, epochs=1)
            selected = {c["paper"]["experiment"]:(p,c) for p,c in configs.items()}
            for name in ("parent", "control", "main", "no_cross"):
                path, candidate = selected[name]
                candidate["data"]["paper_split"]["val_fraction"] = .2
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(yaml.safe_dump(candidate), encoding="utf-8")
                with patch("torchvision.datasets.MNIST", FakeMNIST), patch("scripts.train.fix_console_encoding"), patch(
                        "scripts.train.ensure_output_dirs"), patch("scripts.train.log"), patch.object(sys, "argv", ["train", "--config", path]):
                    main()
                best = torch.load(candidate["train"]["best_ckpt_path"], map_location="cpu")
                self.assertEqual(best["epoch"], 0)
                self.assertTrue(np.isfinite(best["best_validation_score"]))
                if name == "main":
                    before = file_sha256(candidate["train"]["best_ckpt_path"])
                    with patch("torchvision.datasets.MNIST", FakeMNIST), patch("scripts.train.fix_console_encoding"), patch(
                            "scripts.train.ensure_output_dirs"), patch("scripts.train.log"), patch.object(sys, "argv", ["train", "--config", path, "--resume"]):
                        main()
                    self.assertEqual(before, file_sha256(candidate["train"]["best_ckpt_path"]))
                if name != "parent":
                    parent = torch.load(selected["parent"][1]["train"]["best_ckpt_path"], map_location="cpu")
                    for key, value in parent["model"].items():
                        if key.startswith(("memory.", "classifier.", "img_encoder.", "aud_encoder.")):
                            self.assertTrue(torch.equal(value, best["model"][key]), key)

    def test_end_to_end_smoke_evaluation_all_model_kinds(self):
        import sys
        import yaml
        from models.network import CrossModalSNN
        from scripts.paper_evaluate import main
        from scripts.paper_baseline import make_model

        cfg = self.config()
        cfg["data"]["batch_size"] = 2
        cfg["audio"]["use_real_audio"] = False
        cfg["snn"].update(T=2, aud_decoder_base_ch=32, aud_decoder_refine_blocks=0)
        cfg["corruption"]["eval_fixed"].update(img_modes=["occlusion"], aud_modes=["time_mask"])
        with tempfile.TemporaryDirectory() as tmp:
            external = Path(tmp)/"external.pt"
            torch.save(dict(model=CleanRecognizers().state_dict(), kind="recognizer", cfg=cfg), external)
            for kind in ("main", "classifier", "cue_cnn"):
                directory = Path(tmp)/kind
                directory.mkdir()
                checkpoint = directory/"best.pt"
                cfg["paper"] = dict(experiment=kind)
                cfg["train"]["eval_ckpt_path"] = str(checkpoint)
                model = CrossModalSNN(cfg) if kind == "main" else make_model(kind)
                torch.save(dict(model=model.state_dict()), checkpoint)
                config = directory/"config.yaml"
                config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
                with patch("torchvision.datasets.MNIST", FakeMNIST), patch.object(sys, "argv", [
                    "paper_evaluate.py", "--config", str(config), "--recognizer", str(external), "--smoke_batches", "1"]):
                    main()
                meta = json.loads((directory/"smoke_evaluation"/"complete.json").read_text(encoding="utf-8"))
                self.assertFalse(meta["full_test"])
                self.assertEqual(meta["external_clean_n"], 2)
                summary = json.loads((directory/"smoke_evaluation"/"summary.json").read_text(encoding="utf-8"))
                self.assertEqual(len(summary), 48 if kind == "classifier" else 16)


if __name__ == "__main__":
    unittest.main()
