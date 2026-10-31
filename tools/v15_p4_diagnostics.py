"""P4 fixed-cue fitting and read-only stage diagnostics.

The training loss is scripts.train.compute_losses, including its binding and
teacher passes. Only its cue factory is temporarily replaced. Ground-truth
supports remain loss labels; the predicted-mode forward adapter strips them.
No runtime source file is changed. Image stages are captured from the actual
refiner invocation, while bypassed audio has no fabricated refiner stage.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import threading

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import train as original_train
from tools.v15_foreground_protocol import DEFAULT_OPTIONS, make_cues

MODES = ("clean_both", "corrupt_both", "clean_img_corrupt_aud", "corrupt_img_clean_aud")
MODE_WEIGHTS = (.1, .4, .25, .25)
FAMILIES = ("local_gaussian", "local_impulse")
POLICIES = ("uniform", "foreground")
_CUE_LOCK = threading.Lock()


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha256(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def condition_key(policy, family, mode):
    return f"{policy}__{family}__{mode}"


def conditions():
    yield "uniform", "clean", "clean_both"
    for policy in POLICIES:
        for family in FAMILIES:
            for mode in MODES[1:]:
                yield policy, family, mode


def fixed_schedule(updates, seed):
    """Exact original mode proportions in every 100 updates; balanced factors.

    Family/location rotate within each mode. Opposite starting phases for the
    two 25%-weight modes give exactly equal four-factor counts over 1000 steps.
    Clean is shared because it has neither a family nor a corruption location.
    """
    if updates < 1:
        raise ValueError("updates must be positive")
    rng = np.random.default_rng(seed)
    counts = dict.fromkeys(MODES, 0)
    offsets = {MODES[1]: 0, MODES[2]: 0, MODES[3]: 2}
    result = []
    while len(result) < updates:
        block = np.repeat(np.arange(4), [10, 40, 25, 25])
        rng.shuffle(block)
        for index in block:
            mode = MODES[int(index)]
            if index == 0:
                policy, family = "uniform", "clean"
            else:
                factor = (counts[mode] + offsets[mode]) % 4
                policy, family = POLICIES[factor // 2], FAMILIES[factor % 2]
            counts[mode] += 1
            result.append(condition_key(policy, family, mode))
            if len(result) == updates:
                break
    return result


def load_clean_npz(path, *, per_class=6):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {"images", "unique_audio", "audio_indices", "labels", "sample_ids"}
    if not required <= arrays.keys():
        raise ValueError(f"Missing clean asset keys: {required - arrays.keys()}")
    images, unique_audio = arrays["images"], arrays["unique_audio"]
    n = len(images)
    if images.shape != (n, 1, 28, 28) or unique_audio.shape[1:] != (64, 64):
        raise ValueError("Expected original image 1x28x28 and audio 64x64 tensors")
    for array in (images, unique_audio):
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("Clean tensors must be finite, exact float32")
        if array.min() < 0 or array.max() > 1:
            raise ValueError("Clean tensors must lie in [0,1]")
    for key in ("audio_indices", "labels", "sample_ids"):
        if arrays[key].shape != (n,) or arrays[key].dtype.kind not in "iu":
            raise ValueError(f"Invalid identity/index vector: {key}")
    if len(np.unique(arrays["sample_ids"])) != n:
        raise ValueError("Fixed sample IDs must be unique")
    if np.any(arrays["audio_indices"] < 0) or np.any(arrays["audio_indices"] >= len(unique_audio)):
        raise ValueError("Audio index outside the stored pool")
    if per_class and (n != 10 * per_class or not np.array_equal(
            np.bincount(arrays["labels"], minlength=10), np.full(10, per_class))):
        raise ValueError(f"Expected exactly {per_class} fixed pairs for each of ten classes")
    return arrays


def make_case_cache(arrays, cfg, seed):
    image = torch.from_numpy(arrays["images"])
    audio = torch.from_numpy(arrays["unique_audio"][arrays["audio_indices"]])
    cases = {}
    severity = float(cfg["corruption"]["train_severity"])
    for policy, family, mode in conditions():
        ci, ca, truth, metadata = make_cues(
            image, audio, arrays["sample_ids"].tolist(), mode, family,
            cfg, severity, seed, policy, DEFAULT_OPTIONS)
        numeric = {"img_cue": ci.numpy(), "aud_cue": ca.numpy(),
                   "img_truth": truth["img"].numpy(), "aud_truth": truth["aud"].numpy()}
        cases[condition_key(policy, family, mode)] = dict(
            policy=policy, family=family, mode=mode, arrays=numeric,
            metadata=metadata, sha256=array_sha256(numeric))
    return cases


def case_loss_config(cfg, case):
    result = deepcopy(cfg)
    family = case["family"] if case["family"] != "clean" else FAMILIES[0]
    result["corruption"].update(img_train_modes=[family], aud_train_modes=[family],
                                img_mode=family, aud_mode=family)
    return result


class CachedCueProvider:
    """Immutable CPU cache -> fresh device tensors, validated against clean batch."""
    def __init__(self, case, clean_img, clean_aud, indices=None):
        self.case = case
        self.clean_img = clean_img.detach().clone()
        self.clean_aud = clean_aud.detach().clone()
        self.indices = slice(None) if indices is None else indices
        self.calls = 0

    def __call__(self, clean_img, clean_aud, mode, cfg, severity=None,
                 img_mode=None, aud_mode=None, return_masks=False):
        if mode != self.case["mode"] or not return_masks:
            raise ValueError("Cached loss provider called with a different mode/interface")
        if not torch.equal(clean_img, self.clean_img) or not torch.equal(clean_aud, self.clean_aud):
            raise ValueError("Cached loss provider received different clean samples")
        self.calls += 1
        arrays = self.case["arrays"]
        def tensor(key, target):
            return torch.from_numpy(arrays[key][self.indices].copy()).to(target)
        return tensor("img_cue", clean_img), tensor("aud_cue", clean_aud), {
            "img": tensor("img_truth", clean_img), "aud": tensor("aud_truth", clean_aud)}


@contextmanager
def cached_loss_cues(provider):
    """Single-thread process-local factory substitution, restored on every exit."""
    if not _CUE_LOCK.acquire(blocking=False):
        raise RuntimeError("Cached loss context cannot be nested or used concurrently")
    saved = original_train.build_cue
    original_train.build_cue = provider
    try:
        yield
    finally:
        original_train.build_cue = saved
        _CUE_LOCK.release()


class PredictedLossAdapter:
    """Keep original losses/binding while withholding truth from model forward."""
    def __init__(self, model):
        if not model.use_noise_detector or model.noise_mask_source != "predicted":
            raise ValueError("P4 loss adapter requires the blind predicted-mask model")
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        kwargs.pop("img_cue_mask", None)
        kwargs.pop("aud_cue_mask", None)
        return self.model(*args, **kwargs)


def cached_compute_losses(model, clean_img, clean_aud, labels, cfg, case,
                          *, epoch=100, step=0, indices=None):
    # All supported modes contain both modalities, and original v15 disables
    # target curricula: prototypes are not read by select_targets in these modes.
    provider = CachedCueProvider(case, clean_img, clean_aud, indices)
    with cached_loss_cues(provider):
        loss, logs = original_train.compute_losses(
            PredictedLossAdapter(model), clean_img, clean_aud, labels, case["mode"],
            case_loss_config(cfg, case), None, None, epoch=epoch, step=step)
    if provider.calls != 1:
        raise RuntimeError(f"Expected one original loss cue factory call, got {provider.calls}")
    return loss, logs


class StageCapture:
    """Capture actual image delta without modifying forward or running it twice."""
    def __init__(self, model):
        self.model = model
        self.records = []
        self.handle = None

    def __enter__(self):
        if self.model.image_refiner is not None:
            def hook(module, inputs, output):
                self.records.append((inputs[0].detach(), output.detach()))
            self.handle = self.model.image_refiner.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()

    def stages(self, out):
        image = {"coarse": out["recovered_img_coarse"].detach().sigmoid()}
        extras = {}
        if self.model.image_refiner is not None:
            if len(self.records) != 1:
                raise RuntimeError("Stage diagnostic requires exactly one actual image refiner invocation")
            coarse, delta = self.records[0]
            if not torch.equal(coarse, image["coarse"]):
                raise RuntimeError("Refiner input differs from reported coarse stage")
            # v15 visible-paste-back uses the unmasked delta, then clips, then PB.
            if not self.model.image_refiner_visible_paste_back:
                raise ValueError("P4 diagnostic expects original visible image paste-back")
            extras["img_refiner_delta"] = delta
            extras["img_refined_preclip"] = coarse + delta
            image["refined"] = (coarse + delta).clamp(0, 1)
        image["final"] = out["recovered_img"].detach().sigmoid()
        if not self.model.audio_refiner_bypass:
            raise ValueError("P4 baseline requires unchanged bypassed audio refiner")
        audio = {"coarse": out["recovered_aud_coarse"].detach(),
                 "final": out["recovered_aud"].detach()}
        return {"img": image, "aud": audio}, extras


def _region_masks(target, truth, predicted, repair, modality, options):
    if modality == "img":
        foreground = target > options["img_threshold"]
    else:
        maximum = target.reshape(len(target), -1).max(axis=1)
        threshold = np.maximum(options["aud_absolute_threshold"],
                               options["aud_relative_threshold"] * maximum.astype(np.float64))
        foreground = target > threshold.astype(target.dtype).reshape(-1, 1, 1)
    truth, predicted, repair = truth.astype(bool), predicted.astype(bool), repair.astype(bool)
    masks = {"global": np.ones_like(truth), "corrupt": truth, "clean": ~truth,
             "foreground": foreground, "background": ~foreground,
             "corrupt_foreground": truth & foreground,
             "corrupt_background": truth & ~foreground,
             "clean_foreground": ~truth & foreground,
             "clean_background": ~truth & ~foreground,
             "true_positive_repair": truth & predicted & repair,
             "false_negative": truth & ~predicted, "clean_false_positive": ~truth & predicted,
             "repair": repair}
    return masks


def stage_metrics(stages, target, truth, predicted, repair, modality,
                  options=None, saturation_epsilon=1e-4):
    """Pixel/edge pooled statistics; empty regions stay null, never become zero.

    Directional edge errors include edges whose two endpoints are in the region.
    Image axes are vertical/horizontal; audio axes are frequency/time. The same
    masks and denominator rules apply to train, validation and every stage.
    """
    options = DEFAULT_OPTIONS if options is None else options
    target, truth, predicted, repair = map(np.asarray, (target, truth, predicted, repair))
    if any(x.shape != target.shape for x in (truth, predicted, repair)):
        raise ValueError("Stage diagnostic shapes differ")
    if any(not np.isin(x, [0, 1]).all() for x in (truth, predicted, repair)):
        raise ValueError("Stage diagnostic masks must be binary")
    masks = _region_masks(target, truth, predicted, repair, modality, options)
    result = {}
    for name, values in stages.items():
        values = np.asarray(values)
        if values.shape != target.shape or not np.isfinite(values).all() or not np.isfinite(target).all():
            raise ValueError("Invalid stage diagnostic values")
        error = values.astype(np.float64) - target.astype(np.float64)
        stage = {}
        for region, mask in masks.items():
            n = int(mask.sum())
            sse = float(np.square(error)[mask].sum())
            sat = int(((values <= saturation_epsilon) | (values >= 1 - saturation_epsilon))[mask].sum())
            stats = {"pixels": n, "sse": sse, "mse": sse / n if n else None,
                     "saturated_pixels": sat, "saturation_ratio": sat / n if n else None}
            for axis, label in ((-2, "vertical" if modality == "img" else "frequency"),
                                (-1, "horizontal" if modality == "img" else "time")):
                left, right = [slice(None)] * mask.ndim, [slice(None)] * mask.ndim
                left[axis], right[axis] = slice(None, -1), slice(1, None)
                edge_mask = mask[tuple(left)] & mask[tuple(right)]
                gradient_error = np.diff(error, axis=axis)
                count = int(edge_mask.sum())
                absolute = float(np.abs(gradient_error)[edge_mask].sum())
                squared = float(np.square(gradient_error)[edge_mask].sum())
                stats[label] = {"edges": count, "absolute_error_sum": absolute,
                                "squared_error_sum": squared,
                                "gradient_mae": absolute / count if count else None,
                                "gradient_mse": squared / count if count else None}
            stage[region] = stats
        result[name] = stage
    return result


def gradient_norms(model):
    """Per-major-module gradient paths, including explicit frozen/no-grad counts."""
    groups = {}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        stats = groups.setdefault(group, dict(parameters=0, trainable=0,
                                             with_gradient=0, nonzero_gradient=0, squared_norm=0.))
        stats["parameters"] += parameter.numel()
        stats["trainable"] += parameter.numel() if parameter.requires_grad else 0
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all():
                raise FloatingPointError(f"Non-finite gradient in {name}")
            stats["with_gradient"] += parameter.numel()
            stats["nonzero_gradient"] += int(torch.count_nonzero(grad))
            stats["squared_norm"] += float(grad.double().square().sum())
    for stats in groups.values():
        stats["l2_norm"] = stats.pop("squared_norm") ** .5
    return groups


def train_step(model, optimizer, clean_img, clean_aud, labels, cfg, case, *, step):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, logs = cached_compute_losses(model, clean_img, clean_aud, labels, cfg, case, step=step)
    if not loss.requires_grad or not torch.isfinite(loss):
        raise FloatingPointError("P4 fixed-cue loss has no finite gradient path")
    loss.backward()
    gradients = gradient_norms(model)
    clip = float(cfg["train"].get("grad_clip", 0.))
    if clip > 0:
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                       clip, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "loss_components": logs,
            "gradient_norms_before_clip": gradients,
            "learning_rates": [group["lr"] for group in optimizer.param_groups]}


@torch.no_grad()
def evaluate_cases(model, arrays, cases, cfg, device, *, sample_directory=None):
    """Evaluate every fixed case; optional raw archive stores every selected pair."""
    model.eval()
    image = torch.from_numpy(arrays["images"]).to(device)
    audio_np = arrays["unique_audio"][arrays["audio_indices"]]
    audio = torch.from_numpy(audio_np).to(device)
    labels = torch.from_numpy(arrays["labels"]).to(device)
    results = {}
    for key, case in cases.items():
        numeric = case["arrays"]
        ci, ca = [torch.from_numpy(numeric[m + "_cue"]).to(device) for m in ("img", "aud")]
        with StageCapture(model) as capture:
            out = model(x_img_cue=ci, x_aud_cue=ca, training_mode=False)
        stages, extras = capture.stages(out)
        loss, logs = cached_compute_losses(model, image, audio, labels, cfg, case)
        prediction = out["logits"].argmax(1).cpu().numpy()
        row = {"policy": case["policy"], "family": case["family"], "cue_mode": case["mode"],
               "n": len(labels), "case_sha256": case["sha256"], "total_loss": float(loss),
               "loss_components": logs, "index_accuracy": float(np.mean(prediction == arrays["labels"])),
               "audio_refiner_bypassed": True, "modalities": {}}
        archive = {"sample_ids": arrays["sample_ids"], "labels": arrays["labels"],
                   "index_prediction": prediction, **numeric}
        for modality, target in (("img", arrays["images"]), ("aud", audio_np)):
            values = {name: value.cpu().numpy() for name, value in stages[modality].items()}
            values = {"input": numeric[modality + "_cue"], **values}
            predicted = out[modality + "_predicted_mask"].cpu().numpy()
            repair = out[modality + "_repair_mask"].cpu().numpy()
            if not np.array_equal(predicted, repair):
                raise RuntimeError("P4 predicted and actual repair masks differ")
            row["modalities"][modality] = stage_metrics(
                values, target, numeric[modality + "_truth"], predicted, repair, modality)
            archive.update({modality + "_target": target, modality + "_predicted": predicted,
                            modality + "_repair": repair})
            archive.update({modality + "_" + name: value for name, value in values.items()})
        archive.update({key: value.cpu().numpy() for key, value in extras.items()})
        if "img_refined_preclip" in archive:
            preclip = archive["img_refined_preclip"]
            row["image_refiner_clip"] = {"pixels": preclip.size,
                "below_zero": int((preclip < 0).sum()), "above_one": int((preclip > 1).sum())}
        if sample_directory is not None:
            folder = Path(sample_directory)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / (key + ".npz"), **archive)
        results[key] = row
    return results
