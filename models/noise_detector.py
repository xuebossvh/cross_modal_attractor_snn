"""Local contamination detection for image and log-mel cues.

Adapted from the MIT-licensed FCN in IDKiro/CBDNet-pytorch at commit
09a2e55b2098039ee99ada8c634a06fc28c6d8a1, model/cbdnet.py (class FCN).
Copyright (c) 2018 IDKiro. See third_party/cbdnet/LICENSE.

The original network estimates RGB noise levels. This adaptation predicts
single-channel contamination logits; it must be trained on this task and does
not include pretrained weights. A detector observes only the supplied cue.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


DETECTOR_CHECKPOINT_SCHEMA = "local-noise-detector.v1"


def _as_single_channel(value, name):
    """Normalize [B,H,W] or [B,1,H,W], without allowing channel broadcasting."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim == 3:
        value = value.unsqueeze(1)
    elif value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,H,W] or [B,1,H,W]")
    if any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must have nonempty batch and spatial dimensions")
    return value


class LocalNoiseDetector(nn.Module):
    """Five 3x3 convolutions producing a same-size contamination logit map.

    Inputs and outputs have matching shapes: [B,1,H,W] for images, or
    [B,H,W] for log-mel cues. Each modality should use its own detector.
    The final layer has no activation, so training can use stable logit BCE.
    """

    def __init__(self, hidden_channels=32):
        super().__init__()
        if isinstance(hidden_channels, bool) or not isinstance(hidden_channels, int):
            raise ValueError("hidden_channels must be a positive integer")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be a positive integer")
        layers = [nn.Conv2d(1, hidden_channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(3):
            layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(hidden_channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, cue):
        image = _as_single_channel(cue, "cue")
        if not image.is_floating_point():
            raise TypeError("cue must use a floating-point dtype")
        logits = self.net(image)
        return logits.squeeze(1) if cue.ndim == 3 else logits

    def probabilities(self, cue):
        """Return per-position contamination probabilities, preserving shape."""
        return torch.sigmoid(self(cue))

    def predict_mask(self, cue, threshold=0.5):
        """Return a hard 0/1 repair mask; 1 means predict instead of paste back.

        The threshold is a validation-set choice, not a trainable parameter.
        This comparison is intentionally nondifferentiable: supervise the
        detector logits separately using ``supervised_mask_loss``.
        """
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and lie in [0, 1]")
        probabilities = self.probabilities(cue)
        return (probabilities >= threshold).to(probabilities.dtype)


def supervised_mask_loss(logits, true_mask, positive_weight=3.0):
    """Weighted binary cross entropy against known contamination support.

    ``true_mask=None`` denotes an all-clean *present* cue. An absent modality
    must be skipped by the caller, not passed as a clean training example.
    Targets are binary (1 contaminated, 0 clean); they are supervision only
    and must not be supplied to the detector during blind inference.
    """
    positive_weight = float(positive_weight)
    if not math.isfinite(positive_weight) or positive_weight <= 0.0:
        raise ValueError("positive_weight must be finite and positive")
    normalized_logits = _as_single_channel(logits, "logits")
    if not normalized_logits.is_floating_point():
        raise TypeError("logits must use a floating-point dtype")
    if true_mask is None:
        target = torch.zeros_like(normalized_logits)
    else:
        target = _as_single_channel(true_mask, "true_mask")
        if target.shape != normalized_logits.shape:
            raise ValueError("true_mask must match logits batch and spatial dimensions")
        target = target.detach().to(device=logits.device, dtype=logits.dtype)
        if not bool(torch.all((target == 0) | (target == 1))):
            raise ValueError("true_mask must contain only binary 0/1 values")
    return F.binary_cross_entropy_with_logits(
        normalized_logits,
        target,
        pos_weight=logits.new_tensor(positive_weight),
    )


def detector_training_contract(cfg):
    """Data/noise semantics that must match when transferring a detector pair."""
    data = deepcopy(cfg.get("data", {}))
    for key in ("batch_size", "num_workers", "eval_split"):
        data.pop(key, None)
    audio = deepcopy(cfg.get("audio", {}))
    for key in ("auto_download", "encoder_norm_mode", "encoder_local_mix"):
        audio.pop(key, None)
    return {
        "data": data,
        "audio": audio,
        "corruption": deepcopy(cfg.get("corruption", {})),
        "num_classes": cfg.get("dims", {}).get("num_classes"),
    }


def validate_detector_checkpoint(state, cfg):
    """Validate provenance before any detector state is installed."""
    if state.get("schema") != DETECTOR_CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported or missing local noise detector checkpoint schema")
    nc = cfg.get("noise_detector", {}) or {}
    if state.get("hidden_channels") != int(nc.get("hidden_channels", 32)):
        raise ValueError("Noise detector checkpoint hidden_channels mismatch")
    if not isinstance(state.get("cfg"), dict):
        raise ValueError("Noise detector checkpoint is missing its training configuration")
    expected = detector_training_contract(cfg)
    actual = detector_training_contract(state["cfg"])
    for section in expected:
        if actual[section] != expected[section]:
            raise ValueError(f"Noise detector checkpoint configuration mismatch: {section}")
    saved_norm = state["cfg"].get("_audio_norm_stats")
    if "_audio_norm_stats" in cfg and cfg["_audio_norm_stats"] != saved_norm:
        raise ValueError("Noise detector audio normalization statistics mismatch")
    if set(state.get("detectors", {})) != {"img", "aud"}:
        raise ValueError("Noise detector checkpoint must contain both img and aud states")
    score = state.get("best_validation_loss")
    if not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
        raise ValueError("Noise detector checkpoint has no finite best validation loss")


def load_pretrained_detectors(model, cfg):
    """Strictly load the pretrained pair without changing configured thresholds.

    Missing checkpoints are fatal by default. A smoke checkpoint can only be
    used with an explicit ``allow_smoke_checkpoint`` testing configuration.
    Returns the inspected checkpoint, or None when loading is disabled/optional.
    """
    from paths import resolve_from_root

    nc = cfg.get("noise_detector", {}) or {}
    if not nc.get("enabled", False):
        return None
    configured_path = nc.get("checkpoint")
    path = resolve_from_root(configured_path) if configured_path else None
    if path is None or not path.is_file():
        if nc.get("require_pretrained", True):
            raise FileNotFoundError(f"Required pretrained noise detector checkpoint: {configured_path}")
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    validate_detector_checkpoint(state, cfg)
    if state.get("metadata", {}).get("smoke_only", False) and not nc.get("allow_smoke_checkpoint", False):
        raise ValueError("Smoke-only detector checkpoint cannot initialize a formal experiment")
    for modality in ("img", "aud"):
        detector = getattr(model, f"{modality}_noise_detector", None)
        if not isinstance(detector, LocalNoiseDetector):
            raise ValueError(f"Model is missing the {modality} LocalNoiseDetector")
        detector.load_state_dict(state["detectors"][modality], strict=True)
    return state
