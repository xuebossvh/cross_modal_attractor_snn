"""P4 control-flow/optimizer tests with tiny dummy models, not real SNN training."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import load_config
from tools import train_v15_p4 as p4


class TinyModel(torch.nn.Module):
    """Dummy graph used only to exercise optimizer and restart mechanics."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.freeze_base = False
        self.use_noise_detector = True
        self.noise_mask_source = "predicted"
        self.base = torch.nn.Linear(2, 2)
        self.image_refiner = torch.nn.Linear(2, 2)
        self.audio_refiner = torch.nn.Linear(2, 2)
        self.audio_refiner.requires_grad_(False)
        self.img_noise_detector = torch.nn.Linear(2, 1)
        self.aud_noise_detector = torch.nn.Linear(2, 1)

    def forward(self, value):
        return self.image_refiner(self.base(value))


class TinyData(Dataset):
    def __init__(self):
        self.prototype_img = torch.zeros(10, 2)
        self.prototype_aud = torch.zeros(10, 2)
        self._rng = np.random.default_rng(0)

    def __len__(self):
        return 4

    def __getitem__(self, index):
        value = torch.tensor([float(index), float(self._rng.random())])
        return value, value + .2, index % 2


def loaders(_cfg=None, eval_split="val"):
    return (DataLoader(TinyData(), batch_size=2, shuffle=True,
                       generator=torch.Generator().manual_seed(7)),
            DataLoader(TinyData(), batch_size=2))


def dummy_loss(model, image, audio, labels, mode, cfg, pi, pa, **kwargs):
    loss = (model(image) - audio).square().mean()
    return loss, {"dummy_mse": float(loss.detach())}


def fixture():
    cfg = load_config(p4.PROJECT_ROOT / "configs/v15pro.yaml")
    cfg["_audio_norm_stats"] = dict(lo=0., hi=1., train_fingerprint="dummy")
    torch.manual_seed(4)
    model = TinyModel(cfg)
    p4.apply_original_freezing(model, cfg)
    opt = p4._build_train_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=p4.MIN_LR)
    for _ in range(100):
        opt.zero_grad()
        model(torch.ones(2, 2)).square().sum().backward()
        opt.step()
        scheduler.step()
    state = p4._checkpoint_payload(model, opt, scheduler, cfg, 99)
    state["best_validation_score"] = .5
    state["validation"] = {"score": .6, "pair_i2a_acc": float("nan"), "pair_a2i_acc": float("nan")}
    return cfg, deepcopy(state)


class P4TrainingTests(unittest.TestCase):
    def test_cuda_memory_fraction_default_and_invalid_values(self):
        argv = ["--source-checkpoint", "unused.pt", "--data-root", "unused",
                "--fsdd-root", "unused", "--norm-stats", "unused.pt",
                "--output", "unused", "--branch", "keep_lr", "--device", "cpu",
                "--max-batches", "1"]
        self.assertEqual(p4.parser().parse_args(argv).cuda_memory_fraction, .6)
        for value in ("0", "-0.1", "1.01", "nan", "inf"):
            args = p4.parser().parse_args(argv + ["--cuda-memory-fraction", value])
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cuda-memory-fraction"):
                p4.run(args)
        for requested, expected in (("cuda", 1), ("cuda:0", 0)):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(device=requested):
                args = p4.parser().parse_args(argv + ["--output", temporary, "--device", requested])
                with patch.object(torch.cuda, "is_available", return_value=True), \
                        patch.object(torch.cuda, "current_device", return_value=1), \
                        patch.object(torch.cuda, "set_per_process_memory_fraction") as limit, \
                        patch.object(torch, "load", side_effect=RuntimeError("stop before source load")):
                    with self.assertRaisesRegex(RuntimeError, "stop before source load"):
                        p4.run(args)
                    limit.assert_called_once_with(.6, device=expected)

    def test_source_requirements_and_full_semantic_config_validation(self):
        cfg, source = fixture()
        self.assertEqual(p4.verify_source_checkpoint(source), cfg)
        for change in (dict(epoch=98), dict(opt={}), dict(sched=None),
                       dict(asset_packaging=dict(inference_only=True)),
                       dict(noise_training=dict(smoke_only=True))):
            invalid = deepcopy(source)
            invalid.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                p4.verify_source_checkpoint(invalid)
        for section, key, value in (("loss", "lambda_cls", 999),
                                    ("data", "batch_size", 2),
                                    ("audio", "fsdd_root", "migrated"),
                                    ("snn", "steps", 999)):
            drift = deepcopy(cfg)
            drift[section][key] = value
            with self.subTest(section=section), self.assertRaises(ValueError):
                p4.verify_source_checkpoint(source, drift)

    def test_shared_initial_state_and_exact_original_freezing(self):
        cfg, source = fixture()
        initial = []
        for branch in ("keep_lr", "restart_lr"):
            model = TinyModel(cfg)
            opt, sched, meta = p4.initialize_branch(model, source, cfg, branch)
            initial.append(meta)
            self.assertEqual(meta["initial_optimizer_sha256"], p4.state_sha256(source["opt"]))
            self.assertEqual(p4.state_sha256(opt.state_dict()["state"]),
                             p4.state_sha256(source["opt"]["state"]))
            self.assertTrue(all(not p.requires_grad for p in model.audio_refiner.parameters()))
            self.assertTrue(all(not p.requires_grad for p in model.img_noise_detector.parameters()))
            self.assertTrue(all(not p.requires_grad for p in model.aud_noise_detector.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.base.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.image_refiner.parameters()))
            self.assertEqual(sched.last_epoch, 0)
        for key in ("initial_model_sha256", "initial_optimizer_sha256",
                    "initial_optimizer_moments_sha256", "trainable_names", "frozen_names"):
            self.assertEqual(initial[0][key], initial[1][key])

    def test_exact_cosine_trajectory_and_common_group_floor(self):
        cfg, source = fixture()
        for branch, starts in (("keep_lr", [1e-5, 1e-5]),
                               ("restart_lr", [5e-5, 2.5e-5])):
            model = TinyModel(cfg)
            optimizer, scheduler, _ = p4.initialize_branch(model, source, cfg, branch)
            for epoch in range(31):
                expected = [p4.MIN_LR + (lr - p4.MIN_LR) *
                            (1 + np.cos(np.pi * epoch / 30)) / 2 for lr in starts]
                np.testing.assert_allclose([g["lr"] for g in optimizer.param_groups], expected,
                                           rtol=1e-12, atol=1e-15)
                self.assertTrue(all(g["lr"] >= p4.MIN_LR for g in optimizer.param_groups))
                if epoch < 30:
                    optimizer.step()
                    scheduler.step()
            self.assertEqual([g["lr"] for g in optimizer.param_groups], [1e-5, 1e-5])

    def test_original_network_freezing_is_preserved(self):
        cfg = load_config(p4.PROJECT_ROOT / "configs/v15pro.yaml")
        model = p4.CrossModalSNN(cfg)
        before = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        after = p4.apply_original_freezing(model, cfg)
        self.assertEqual(before, after)
        self.assertTrue(any(name.startswith("image_refiner.") for name in after))
        self.assertFalse(any(name.startswith(("audio_refiner.", "img_noise_detector.", "aud_noise_detector."))
                             for name in after))
        model.train()
        self.assertFalse(model.img_noise_detector.training)
        self.assertFalse(model.aud_noise_detector.training)

    def test_nonfinite_loss_and_gradient_fail_without_update(self):
        cfg, source = fixture()
        model = TinyModel(cfg)
        optimizer, _, _ = p4.initialize_branch(model, source, cfg, "keep_lr")
        before = p4.state_sha256(optimizer.state_dict())
        for kind in ("loss", "gradient"):
            def bad_loss(*args, **kwargs):
                value = model.base.weight.sum()
                if kind == "loss":
                    value = value * float("nan")
                else:
                    value.register_hook(lambda grad: grad * float("nan"))
                return value, {}
            with patch.object(p4, "compute_losses", side_effect=bad_loss), self.assertRaises(FloatingPointError):
                p4.train_epoch(model, optimizer, loaders()[0], cfg, torch.device("cpu"), 100, 1)
            self.assertEqual(before, p4.state_sha256(optimizer.state_dict()))

    def test_dummy_smoke_resume_matches_uninterrupted_and_refuses_identity_drift(self):
        """This test is not evidence of real inference or a real-data training smoke."""
        cfg, source = fixture()
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            source_path = root / "source" / "last.pt"
            source_path.parent.mkdir()
            torch.save(source, source_path)
            stack.enter_context(patch.object(p4, "CrossModalSNN", TinyModel))
            stack.enter_context(patch.object(p4, "prepare_loader_config", return_value=(deepcopy(cfg), {})))
            stack.enter_context(patch.object(p4, "build_loaders", side_effect=loaders))
            stack.enter_context(patch.object(p4, "verify_loader_splits", return_value={"fixture": "dummy"}))
            stack.enter_context(patch.object(p4, "compute_losses", side_effect=dummy_loss))
            stack.enter_context(patch.object(p4, "validate_full", return_value=({"score": .4}, {"dummy": True})))
            memory_limit = stack.enter_context(patch.object(torch.cuda, "set_per_process_memory_fraction"))

            def arguments(output, stop, resume=False, branch="restart_lr"):
                argv = ["--source-checkpoint", str(source_path), "--data-root", str(root),
                        "--fsdd-root", str(root), "--norm-stats", str(root / "unused.pt"),
                        "--output", str(root / output), "--branch", branch,
                        "--device", "cpu", "--max-batches", "1", "--stop-after-epoch", str(stop)]
                return p4.parser().parse_args(argv + (["--resume"] if resume else []))

            source_hash = p4.file_sha256(source_path)
            p4.run(arguments("uninterrupted", 101))
            # Simulate a kill after durable last but before best/JSON exports.
            persist = p4.durable_checkpoint
            def crash_before_best(payload, path):
                if Path(path).name == "best.pt":
                    raise OSError("simulated best save interruption")
                persist(payload, path)
            with patch.object(p4, "durable_checkpoint", side_effect=crash_before_best), self.assertRaises(OSError):
                p4.run(arguments("crash", 100))
            self.assertTrue((root / "crash/last.pt").is_file())
            self.assertFalse((root / "crash/epochs.json").exists())
            p4.run(arguments("crash", 101, resume=True))
            p4.run(arguments("resumed", 100))
            healthy_best = p4.file_sha256(root / "resumed/best.pt")
            p4.run(arguments("resumed", 100, resume=True))
            self.assertEqual(healthy_best, p4.file_sha256(root / "resumed/best.pt"))
            p4.run(arguments("resumed", 101, resume=True))
            one = torch.load(root / "uninterrupted/last.pt", weights_only=True)
            two = torch.load(root / "resumed/last.pt", weights_only=True)
            self.assertEqual(p4.state_sha256(one["model"]), p4.state_sha256(two["model"]))
            self.assertEqual(p4.state_sha256(one["opt"]), p4.state_sha256(two["opt"]))
            self.assertEqual(p4.state_sha256(one["sched"]), p4.state_sha256(two["sched"]))
            crashed = torch.load(root / "crash/last.pt", weights_only=True)
            self.assertEqual(p4.state_sha256(one["model"]), p4.state_sha256(crashed["model"]))
            self.assertTrue((root / "crash/best.pt").is_file())
            self.assertEqual([r["epoch"] for r in two["p4"]["records"]], [100, 101])
            self.assertTrue(two["p4"]["smoke_only"])
            self.assertTrue(two["noise_training"]["smoke_only"])
            self.assertEqual(source_hash, p4.file_sha256(source_path))
            for a, b in zip(one["p4"]["records"], two["p4"]["records"]):
                self.assertEqual(a["clean_batch_stream_sha256"], b["clean_batch_stream_sha256"])
                self.assertEqual(a["epoch_loss"], b["epoch_loss"])
            identity = deepcopy(two["p4"]["identity"])
            for field in ("branch", "source_sha256", "runtime_sha256", "driver_sha256", "config_sha256", "smoke_only"):
                changed = deepcopy(identity)
                changed[field] = "changed"
                with self.subTest(field=field), self.assertRaises(ValueError):
                    p4.verify_resume(two, changed)
            with self.assertRaises(ValueError):
                p4.run(arguments("resumed", 102, resume=True, branch="keep_lr"))
            modified = deepcopy(two)
            modified["model"]["base.weight"][0, 0] += 1
            with self.assertRaises(ValueError):
                p4.verify_resume(modified, identity)
            self.assertEqual(json.loads((root / "resumed/epochs.json").read_text())[-1]["epoch"], 101)
            memory_limit.assert_not_called()
            self.assertIsNone(two["p4"]["identity"]["cuda_memory_fraction"])


if __name__ == "__main__":
    unittest.main()
