"""P4 numerical, leakage, original-loss and isolated-update contracts."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import load_config, set_seed
from models.network import CrossModalSNN
from scripts import train as original_train
from tools.v15_p4_diagnostics import (
    MODES, CachedCueProvider, StageCapture, array_sha256, cached_compute_losses,
    cached_loss_cues, case_loss_config, condition_key, evaluate_cases, file_sha256,
    fixed_schedule, load_clean_npz, make_case_cache, stage_metrics, train_step,
)
from tools.run_v15_p4_tiny import restore_history, validate_baseline_config


def small_config():
    cfg = load_config(ROOT / "configs/v15pro.yaml")
    cfg["snn"].update(T=2, aud_conv_ch1=2, aud_conv_ch2=4,
                      aud_decoder_base_ch=8, aud_decoder_refine_blocks=0)
    cfg["dims"].update(img_hidden=16, D_img=8, N_key_img=8,
                       aud_hidden=16, D_aud=8, N_key_aud=8,
                       N_index=16, N_value_img=8, N_value_aud=16)
    cfg["index"]["k_wta"] = 4
    cfg["detail_conditioning"].update(img_detail_dim=8, aud_detail_dim=8)
    cfg["audio_local_cue"].update(enabled=False, multiscale=False)
    cfg["image_refiner"].update(enabled=True, hidden_ch=4, blocks=1)
    cfg["audio_refiner"].update(hidden_ch=4, blocks=1)
    cfg["cross_key_conditioning"].update(enabled=False, build_modules=False,
                                         build_feature_modules=False)
    cfg["noise_detector"].update(hidden_channels=4, freeze=True, require_pretrained=False)
    cfg["train"].update(lr=5e-5, device="cpu", require_cuda=False)
    return cfg


def clean_arrays(n=2):
    rng = np.random.default_rng(800)
    return dict(images=rng.uniform(.01, .99, (n, 1, 28, 28)).astype(np.float32),
                unique_audio=rng.uniform(.01, .99, (n, 64, 64)).astype(np.float32),
                audio_indices=np.arange(n, dtype=np.int64),
                labels=np.arange(n, dtype=np.int64) % 10,
                sample_ids=np.arange(n, dtype=np.int64) + 100)


def force_mask(model, value):
    with torch.no_grad():
        for detector in (model.img_noise_detector, model.aud_noise_detector):
            for parameter in detector.parameters():
                parameter.zero_()
            detector.net[-1].bias.fill_(20 if value else -20)


class FixedCueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schedule_exact_weights_and_balanced_factors(self):
        schedule = fixed_schedule(1000, 1504)
        self.assertEqual(schedule, fixed_schedule(1000, 1504))
        self.assertNotEqual(schedule, fixed_schedule(1000, 1505))
        for start in range(0, 1000, 100):
            counts = Counter(key.split("__")[-1] for key in schedule[start:start + 100])
            self.assertEqual(counts, dict(zip(MODES, [10, 40, 25, 25])))
        factors = Counter("__".join(key.split("__")[:2]) for key in schedule if "__clean__" not in key)
        self.assertEqual(set(factors.values()), {225})

    def test_cache_repeatability_readonly_and_context_exception_restoration(self):
        cfg, arrays = small_config(), clean_arrays()
        first = make_case_cache(arrays, cfg, 15)
        second = make_case_cache(arrays, cfg, 15)
        self.assertEqual(len(first), 13)
        self.assertEqual({key: value["sha256"] for key, value in first.items()},
                         {key: value["sha256"] for key, value in second.items()})
        case = first[condition_key("foreground", "local_impulse", "corrupt_both")]
        ci, ca = torch.from_numpy(arrays["images"]), torch.from_numpy(arrays["unique_audio"])
        provider = CachedCueProvider(case, ci, ca)
        before = array_sha256(case["arrays"])
        result = provider(ci, ca, case["mode"], cfg, return_masks=True)
        result[0].zero_()
        result[2]["img"].zero_()
        self.assertEqual(before, array_sha256(case["arrays"]))
        original = original_train.build_cue
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with cached_loss_cues(provider):
                self.assertIs(original_train.build_cue, provider)
                with self.assertRaisesRegex(RuntimeError, "nested"):
                    with cached_loss_cues(provider):
                        pass
                raise RuntimeError("deliberate")
        self.assertIs(original_train.build_cue, original)

    def test_loss_and_gradients_equal_original_for_same_actual_cues(self):
        cfg, arrays = small_config(), clean_arrays()
        image, audio = torch.from_numpy(arrays["images"]), torch.from_numpy(arrays["unique_audio"])
        labels = torch.from_numpy(arrays["labels"])
        for mode in MODES:
            for family in ("local_gaussian", "local_impulse"):
                with self.subTest(mode=mode, family=family):
                    set_seed(321)
                    model = CrossModalSNN(cfg).train()
                    other = deepcopy(model)
                    cfg_case = case_loss_config(cfg, {"family": family})
                    ci, ca, masks = original_train.build_cue(image, audio, mode, cfg_case,
                        severity=.25, img_mode=family, aud_mode=family, return_masks=True)
                    case = {"mode": mode, "family": family, "arrays": {
                        "img_cue": ci.numpy(), "aud_cue": ca.numpy(),
                        "img_truth": (masks["img"] if masks["img"] is not None else torch.zeros_like(ci)).numpy(),
                        "aud_truth": (masks["aud"] if masks["aud"] is not None else torch.zeros_like(ca)).numpy()}}
                    def original_cues(*args, **kwargs):
                        return ci.clone(), ca.clone(), {key: value.clone() if value is not None else None
                                                       for key, value in masks.items()}
                    set_seed(700)
                    with cached_loss_cues(original_cues):
                        expected, expected_logs = original_train.compute_losses(
                            model, image, audio, labels, mode, cfg_case, None, None, epoch=100)
                    expected.backward()
                    observed_kwargs = []
                    def observer(module, args, kwargs):
                        observed_kwargs.append(dict(kwargs))
                    handle = other.register_forward_pre_hook(observer, with_kwargs=True)
                    try:
                        set_seed(700)
                        actual, actual_logs = cached_compute_losses(other, image, audio, labels, cfg, case)
                        actual.backward()
                    finally:
                        handle.remove()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual_logs, expected_logs)
                    for (name, param), (other_name, other_param) in zip(model.named_parameters(), other.named_parameters()):
                        self.assertEqual(name, other_name)
                        if param.grad is None:
                            self.assertIsNone(other_param.grad)
                        else:
                            torch.testing.assert_close(other_param.grad, param.grad, rtol=0, atol=0)
                    self.assertTrue(any(row.get("phase") == "binding" for row in observed_kwargs))
                    for kwargs in observed_kwargs:
                        self.assertNotIn("img_cue_mask", kwargs)
                        self.assertNotIn("aud_cue_mask", kwargs)


class StageAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_refiner_stages_and_audio_bypass_pasteback(self):
        cfg, arrays = small_config(), clean_arrays()
        model = CrossModalSNN(cfg).eval()
        image, audio = torch.from_numpy(arrays["images"]), torch.from_numpy(arrays["unique_audio"])
        for selected in (False, True):
            force_mask(model, selected)
            with torch.no_grad(), StageCapture(model) as capture:
                out = model(x_img_cue=image, x_aud_cue=audio, training_mode=False)
            stages, extra = capture.stages(out)
            self.assertEqual(set(stages["img"]), {"coarse", "refined", "final"})
            self.assertEqual(set(stages["aud"]), {"coarse", "final"})
            torch.testing.assert_close(stages["img"]["refined"],
                (stages["img"]["coarse"] + extra["img_refiner_delta"]).clamp(0, 1))
            torch.testing.assert_close(stages["aud"]["final"], stages["aud"]["coarse"] if selected else audio)
            torch.testing.assert_close(stages["img"]["final"],
                stages["img"]["refined"].clamp(1e-4, 1 - 1e-4) if selected else image)
            self.assertFalse(model.image_refiner._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, "hook cleanup"):
            with StageCapture(model):
                raise RuntimeError("hook cleanup")
        self.assertFalse(model.image_refiner._forward_hooks)

    def test_scalar_regions_gradient_denominators_and_empty_region(self):
        target = np.zeros((1, 2, 3), dtype=np.float32)
        prediction = np.array([[[0, 1, 0], [0, 0, 0]]], dtype=np.float32)
        truth = prediction.copy()
        recovered = np.array([[[0, .5, 1], [0, .5, 1]]], dtype=np.float32)
        result = stage_metrics({"coarse": recovered}, target, truth, prediction, prediction, "aud")["coarse"]
        self.assertEqual(result["global"]["pixels"], 6)
        self.assertAlmostEqual(result["global"]["mse"], 2.5 / 6)
        self.assertEqual(result["global"]["time"]["edges"], 4)
        self.assertEqual(result["global"]["time"]["gradient_mae"], .5)
        self.assertEqual(result["global"]["frequency"]["gradient_mse"], 0)
        self.assertEqual(result["true_positive_repair"]["pixels"], 1)
        self.assertIsNone(result["false_negative"]["mse"])
        self.assertIsNone(result["foreground"]["time"]["gradient_mae"])

    def test_real_original_loss_update_keeps_detectors_bypass_and_sources_unchanged(self):
        cfg, arrays = small_config(), clean_arrays()
        validate_baseline_config(cfg)
        model = CrossModalSNN(cfg).train()
        force_mask(model, True)
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        runtime_paths = [ROOT / "scripts/train.py", ROOT / "models/network.py", ROOT / "common.py"]
        source_hashes = [file_sha256(path) for path in runtime_paths]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "old_source.pt"
            torch.save({"model": initial}, source)
            source_digest = file_sha256(source)
            cases = make_case_cache(arrays, cfg, 123)
            case = cases[condition_key("foreground", "local_gaussian", "corrupt_both")]
            optimizer = original_train._build_train_optimizer(model, cfg)
            record = train_step(model, optimizer, torch.from_numpy(arrays["images"]),
                torch.from_numpy(arrays["unique_audio"]), torch.from_numpy(arrays["labels"]),
                cfg, case, step=0)
            self.assertTrue(np.isfinite(record["loss"]))
            self.assertEqual(record["learning_rates"], [5e-5, 2.5e-5])
            self.assertGreater(record["gradient_norms_before_clip"]["image_decoder"]["l2_norm"], 0)
            self.assertGreater(record["gradient_norms_before_clip"]["audio_decoder"]["l2_norm"], 0)
            self.assertGreater(record["gradient_norms_before_clip"]["image_refiner"]["l2_norm"], 0)
            self.assertEqual(record["gradient_norms_before_clip"]["audio_refiner"]["trainable"], 0)
            self.assertEqual(source_digest, file_sha256(source))
        self.assertTrue(any(not torch.equal(value, model.state_dict()[key]) for key, value in initial.items()
                            if key.startswith("image_decoder.")))
        for key, value in initial.items():
            if "noise_detector" in key or key.startswith("audio_refiner."):
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(source_hashes, [file_sha256(path) for path in runtime_paths])

    def test_evaluation_archive_has_real_stages_and_no_truth_forward(self):
        cfg, arrays = small_config(), clean_arrays()
        model = CrossModalSNN(cfg).eval()
        cases = make_case_cache(arrays, cfg, 12)
        key = condition_key("foreground", "local_impulse", "corrupt_both")
        seen = []
        def observer(module, args, kwargs):
            seen.append(kwargs)
        handle = model.register_forward_pre_hook(observer, with_kwargs=True)
        try:
            with tempfile.TemporaryDirectory() as directory:
                metrics = evaluate_cases(model, arrays, {key: cases[key]}, cfg, "cpu", sample_directory=directory)
                with np.load(Path(directory) / (key + ".npz")) as archive:
                    self.assertIn("img_refined", archive.files)
                    self.assertIn("img_refiner_delta", archive.files)
                    self.assertNotIn("aud_refined", archive.files)
                    self.assertTrue(np.array_equal(archive["sample_ids"], arrays["sample_ids"]))
                self.assertTrue(metrics[key]["audio_refiner_bypassed"])
                self.assertEqual(set(metrics[key]["modalities"]["aud"]), {"input", "coarse", "final"})
        finally:
            handle.remove()
        for kwargs in seen:
            self.assertNotIn("img_cue_mask", kwargs)
            self.assertNotIn("aud_cue_mask", kwargs)

    def test_real_adam_restart_replays_identical_parameter_updates(self):
        cfg, arrays = small_config(), clean_arrays()
        set_seed(93)
        continuous = CrossModalSNN(cfg)
        interrupted = deepcopy(continuous)
        first_opt = original_train._build_train_optimizer(continuous, cfg)
        second_opt = original_train._build_train_optimizer(interrupted, cfg)
        cases = make_case_cache(arrays, cfg, 1234)
        schedule = fixed_schedule(3, 1234)
        image, audio = torch.from_numpy(arrays["images"]), torch.from_numpy(arrays["unique_audio"])
        labels = torch.from_numpy(arrays["labels"])
        def update(model, opt, index):
            set_seed((1234 * 100003 + index + 1) % (2 ** 32))
            return train_step(model, opt, image, audio, labels, cfg, cases[schedule[index]], step=index)
        continuous_losses = [update(continuous, first_opt, index)["loss"] for index in range(3)]
        resumed_losses = [update(interrupted, second_opt, index)["loss"] for index in range(2)]
        saved_model, saved_opt = deepcopy(interrupted.state_dict()), deepcopy(second_opt.state_dict())
        resumed = CrossModalSNN(cfg)
        resumed.load_state_dict(saved_model)
        resumed_opt = original_train._build_train_optimizer(resumed, cfg)
        resumed_opt.load_state_dict(saved_opt)
        resumed_losses.append(update(resumed, resumed_opt, 2)["loss"])
        self.assertEqual(continuous_losses, resumed_losses)
        for key, value in continuous.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


class AssetResumeTests(unittest.TestCase):
    def test_assets_reject_nonfloat32_and_wrong_class_balance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.npz"
            arrays = clean_arrays(60)
            np.savez(path, **arrays)
            self.assertEqual(len(load_clean_npz(path)["labels"]), 60)
            arrays["images"] = arrays["images"].astype(np.float64)
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "float32"):
                load_clean_npz(path)
            arrays["images"] = arrays["images"].astype(np.float32)
            arrays["labels"].fill(0)
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "each of ten classes"):
                load_clean_npz(path)

    def test_resume_archives_uncommitted_tail_and_rejects_missing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text("\n".join(json.dumps({"update": i}) for i in range(1, 5)) + "\n{bad", encoding="utf-8")
            restore_history(path, 2)
            self.assertEqual([json.loads(line)["update"] for line in path.read_text().splitlines()], [1, 2])
            self.assertEqual(len(list(Path(directory).glob("history_uncommitted_*.jsonl"))), 1)
            path.write_text('{"update": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing, duplicated or reordered"):
                restore_history(path, 2)


if __name__ == "__main__":
    unittest.main()
