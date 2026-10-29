"""The v16 thirteen-condition cue protocol, with separate missing/noise labels.

Only ``missing`` is an observed input. ``noise`` is synthetic support supervision
or evaluation ground truth and must not enter the predicted-mask forward path.
Seeded corruption runs on CPU one observation at a time, keyed by stable IDs;
this makes comparisons independent of batch size, accelerator and cue pairing.
"""

from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from common import CUE_MODES
from data.corruption import (AUD_TRAIN_MODES, IMG_TRAIN_MODES, LOCAL_NOISE_MODES,
                             corrupt_audio, corrupt_image)


MODE_STATES = {
    "corrupt_img_only": ("P", "A"), "corrupt_aud_only": ("A", "P"),
    "corrupt_both": ("P", "P"), "clean_img_corrupt_aud": ("C", "P"),
    "corrupt_img_clean_aud": ("P", "C"), "clean_img_only": ("C", "A"),
    "clean_aud_only": ("A", "C"), "clean_both": ("C", "C"),
    "noisy_img_only": ("N", "A"), "noisy_aud_only": ("A", "N"),
    "noisy_both": ("N", "N"), "clean_img_noisy_aud": ("C", "N"),
    "noisy_img_clean_aud": ("N", "C"),
}
MODES = tuple(MODE_STATES)
PROTOCOL_SCHEMA = "v16-cues.v1"


def protocol_contract(cfg):
    """Canonical cue semantics, without model-stage or output-directory fields."""
    vc = cfg.get("v16", {}) or {}
    return {
        "schema": PROTOCOL_SCHEMA,
        "modes": dict(MODE_STATES),
        "cue_probabilities": cue_probabilities(cfg),
        "missing_severity": _fraction(vc.get("missing_severity", 0.4), "missing_severity"),
        "noise_coverage": _fraction(vc.get("noise_coverage", 0.25), "noise_coverage"),
        "noise_std": _nonnegative(vc.get("noise_std", 0.2), "noise_std"),
        "image_missing_families": _pool(vc, "image_missing_families", IMG_TRAIN_MODES),
        "audio_missing_families": _pool(vc, "audio_missing_families", AUD_TRAIN_MODES),
        "noise_families": _pool(vc, "noise_families", LOCAL_NOISE_MODES),
    }


def _fraction(value, name):
    value = _nonnegative(value, name)
    if value > 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _pool(settings, key, allowed):
    values = list(settings.get(key, allowed))
    if not values or len(set(values)) != len(values) or any(v not in allowed for v in values):
        raise ValueError(f"v16.{key} must be a nonempty, unique subset of {list(allowed)}")
    return values


def cue_probabilities(cfg):
    override = (cfg.get("v16", {}) or {}).get("cue_probabilities")
    if override is None:
        return {mode: 0.0625 if mode in CUE_MODES else 0.1 for mode in MODES}
    if not isinstance(override, dict) or set(override) - set(MODES):
        raise ValueError("v16.cue_probabilities contains unknown modes")
    weights = {mode: _nonnegative(override.get(mode, 0), mode) for mode in MODES}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("v16.cue_probabilities must contain positive mass")
    return {mode: value / total for mode, value in weights.items()}


def sample_mode(cfg):
    probabilities = cue_probabilities(cfg)
    return random.choices(MODES, [probabilities[m] for m in MODES], k=1)[0]


@contextmanager
def preserve_rng():
    """Do not let deterministic validation advance training's random streams."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _stable_seed(*parts):
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def _ids(ids, size, seeded):
    if ids is None:
        if seeded:
            raise ValueError("Seeded cues require stable per-sample ids")
        return list(range(size))
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    ids = list(ids)
    if len(ids) != size:
        raise ValueError("ids length must match batch size")
    return ids


def make_cues(img, aud, mode, cfg, *, seed=None, ids=None, protocol="train",
              severity=None, coverage=None, noise_std=None, img_family=None,
              aud_family=None):
    """Produce observations and separate masks without mutating clean inputs.

    C=clean, P=partially missing, N=locally noisy, A=absent. Explicit family
    overrides apply only to damaged states; family ``random`` selects from the
    corresponding configured pool independently for each sample/modality.
    """
    if mode not in MODE_STATES:
        raise ValueError(f"Unknown v16 cue mode: {mode}")
    if img.ndim != 4 or img.shape[1] != 1 or aud.ndim != 3 or len(img) != len(aud) or not len(img):
        raise ValueError("Expected matching nonempty image [B,1,H,W] and audio [B,H,W]")
    if not img.is_floating_point() or not aud.is_floating_point():
        raise TypeError("Cues must be floating-point tensors")
    contract = protocol_contract(cfg)
    severity = _fraction(contract["missing_severity"] if severity is None else severity, "severity")
    coverage = _fraction(contract["noise_coverage"] if coverage is None else coverage, "coverage")
    noise_std = _nonnegative(contract["noise_std"] if noise_std is None else noise_std, "noise_std")
    sample_ids = _ids(ids, len(img), seed is not None)
    result = {"missing": {}, "noise": {}, "states": dict(zip(("img", "aud"), MODE_STATES[mode]))}
    with preserve_rng() if seed is not None else nullcontext():
        for modality, clean, override in (("img", img, img_family), ("aud", aud, aud_family)):
            state = result["states"][modality]
            if state == "A":
                result[modality] = result["missing"][modality] = result["noise"][modality] = None
                result[f"{modality}_family"] = ["absent"] * len(clean)
                continue
            if state == "C":
                result[modality] = clean.clone()
                result["missing"][modality] = torch.zeros_like(clean)
                result["noise"][modality] = torch.zeros_like(clean)
                result[f"{modality}_family"] = ["clean"] * len(clean)
                continue
            pool_key = "noise_families" if state == "N" else f"{'image' if modality == 'img' else 'audio'}_missing_families"
            pool = contract[pool_key]
            if override not in (None, "random") and override not in pool:
                raise ValueError(f"{modality} family {override!r} is incompatible with state {state}")
            cues, supports, families = [], [], []
            for index, sample_id in enumerate(sample_ids):
                value = clean[index:index + 1]
                if seed is not None:
                    value = value.detach().cpu()
                    _seed_cpu(_stable_seed(seed, protocol, sample_id, modality, state, "family"))
                family = random.choice(pool) if override in (None, "random") else override
                amount = coverage if state == "N" else severity
                if seed is not None:
                    _seed_cpu(_stable_seed(seed, protocol, sample_id, modality, family, amount, noise_std if state == "N" else 0))
                if amount == 0:
                    cue, support = value.clone(), torch.zeros_like(value)
                else:
                    corruption = corrupt_image if modality == "img" else corrupt_audio
                    cue, support = corruption(value, family, amount, return_mask=True, noise_std=noise_std)
                if support is None:
                    raise ValueError(f"v16 family {family} did not produce an explicit support mask")
                cues.append(cue)
                supports.append(support)
                families.append(family)
            result[modality] = torch.cat(cues).to(device=clean.device, dtype=clean.dtype)
            support = torch.cat(supports).to(device=clean.device, dtype=clean.dtype)
            result["missing"][modality] = support if state == "P" else torch.zeros_like(clean)
            result["noise"][modality] = support if state == "N" else torch.zeros_like(clean)
            result[f"{modality}_family"] = families
    return result


def make_targets(img, aud, labels, proto_img, proto_aud, mode):
    """Instances for present cues, train-only category representatives if absent."""
    if mode not in MODE_STATES:
        raise ValueError(f"Unknown v16 cue mode: {mode}")
    targets, kinds = [], []
    for clean, prototypes, state in zip((img, aud), (proto_img, proto_aud), MODE_STATES[mode]):
        if state == "A":
            if prototypes is None:
                raise ValueError("Absent modalities require training-only category prototypes")
            targets.append(prototypes.to(clean.device)[labels.to(clean.device, dtype=torch.long)])
            kinds.append("category")
        else:
            targets.append(clean)
            kinds.append("instance")
    return targets[0], targets[1], kinds[0], kinds[1]


def detector_mask_loss(logits, true_noise, missing=None, positive_weight=3.0):
    """Noise BCE over observed positions only; absent modalities are skipped outside.

    Known holes are not positive noise examples or clean negatives. An all-hole
    cue returns differentiable zero, so it cannot train the detector indirectly.
    """
    if not logits.is_floating_point():
        raise TypeError("Detector logits must be floating point")
    positive_weight = _nonnegative(positive_weight, "positive_weight")
    if positive_weight <= 0:
        raise ValueError("positive_weight must be positive")
    target = torch.zeros_like(logits) if true_noise is None else true_noise.detach().to(logits)
    known = torch.zeros_like(logits) if missing is None else missing.detach().to(logits)
    if target.shape != logits.shape or known.shape != logits.shape:
        raise ValueError("Detector masks must exactly match logit shape")
    if not bool(torch.all((target == 0) | (target == 1))) or not bool(torch.all((known == 0) | (known == 1))):
        raise ValueError("Detector masks must be binary")
    valid = 1.0 - known
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=logits.new_tensor(positive_weight))
    return (losses * valid).sum() / valid.sum().clamp_min(1)
