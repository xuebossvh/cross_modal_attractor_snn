"""Independent, evaluation-only foreground sampling for the frozen v15 runtime.

Clean targets determine corruption placement and metric regions only. Neither
foreground masks nor true corruption masks are model inputs in predicted runs.
The uniform policy calls the original deterministic_cues unchanged. Foreground
placement uses a separate Python RNG; full-image noise uses the original seed
and Torch operations, so changing placement preserves the paired noise field.
"""

import math
import numbers
import random
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from scripts.noise_metrics import (
    DEFAULT_MODES, LOCAL_FAMILIES, _sample_seed, _seed_corruption,
    deterministic_cues, preserve_random_state, safe_ratio,
)


DEFAULT_OPTIONS = {
    "min_foreground_coverage": 0.35,
    "img_threshold": 0.1,
    "aud_absolute_threshold": 0.02,
    "aud_relative_threshold": 0.1,
}


def _finite_number(value, name, minimum=0.0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _options(options):
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("options must be a mapping")
    supplied = {} if options is None else dict(options)
    unknown = set(supplied) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown foreground options: {sorted(unknown)}")
    result = dict(DEFAULT_OPTIONS, **supplied)
    for name, value in result.items():
        result[name] = _finite_number(value, name, maximum=1.0)
    return result


def _check_tensor(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be nonempty and finite")


def _threshold(target, modality, options):
    if modality == "img":
        return options["img_threshold"]
    if modality == "aud":
        return max(options["aud_absolute_threshold"],
                   options["aud_relative_threshold"] * float(target.max()))
    raise ValueError("modality must be img or aud")


def foreground_mask(target, modality, options=None):
    """Return the strict-threshold foreground for ONE clean sample.

    Shapes are preserved. Audio's relative threshold uses this sample's maximum,
    so call separately for each sample rather than passing an entire batch.
    """
    _check_tensor(target, "target")
    if target.ndim < 2:
        raise ValueError("Foreground targets require two spatial dimensions")
    settings = _options(options)
    return target > _threshold(target, modality, settings)


def _block_shape(height, width, severity):
    # Exactly the dimensions of data.corruption._local_contamination.
    block_h = min(height, max(1, int(round(height * math.sqrt(severity)))))
    block_w = min(width, max(1, int(round(severity * height * width / block_h))))
    return block_h, block_w


def _window_counts(foreground, block_h, block_w):
    """Enumerate EVERY spatial window using integer summed-area tables."""
    height, width = foreground.shape[-2:]
    spatial = foreground.long().reshape(-1, height, width).sum(0)
    integral = F.pad(spatial.cumsum(0).cumsum(1), (1, 0, 1, 0))
    return (integral[block_h:, block_w:] - integral[:-block_h, block_w:]
            - integral[block_h:, :-block_w] + integral[:-block_h, :-block_w])


def _rectangle(mask):
    spatial = mask.bool().reshape(-1, *mask.shape[-2:]).any(0)
    positions = spatial.nonzero()
    if not len(positions):
        return None
    top, left = positions.min(0).values.tolist()
    bottom, right = positions.max(0).values.tolist()
    return [top, left, bottom - top + 1, right - left + 1]


def _sample_metadata(target, mask, modality, settings, active, counts, policy):
    foreground = foreground_mask(target, modality, settings)
    total = int(foreground.sum())
    covered = int((foreground & mask.bool()).sum())
    coverage = safe_ratio(covered, total)
    eligible_count = (int((counts >= settings["min_foreground_coverage"] * total).sum())
                      if active and total else 0)
    max_feasible = (safe_ratio(int(counts.max()), total) if active else
                    (0.0 if total else None))
    applicable = active and total > 0
    constraint_met = (coverage >= settings["min_foreground_coverage"]
                      if applicable else None)
    if not total:
        status = "no_foreground"
    elif not active:
        status = "inactive"
    elif constraint_met:
        status = "constraint_met"
    elif max_feasible < settings["min_foreground_coverage"]:
        status = "constraint_infeasible"
    else:
        status = "below_requested_coverage"
    return {
        "actual_threshold": _threshold(target, modality, settings),
        "foreground_pixels": total,
        "covered_foreground_pixels": covered,
        "foreground_coverage": coverage,
        "background_pixels": target.numel() - total,
        "covered_background_pixels": int((~foreground & mask.bool()).sum()),
        "total_pixels": target.numel(),
        "corrupt_pixels": int(mask.bool().sum()),
        "mask_area_coverage": float(mask.bool().sum()) / target.numel(),
        "rectangle": _rectangle(mask),
        "active": active,
        "evaluable": total > 0,
        "constraint_applicable": applicable,
        "constraint_applied": policy == "foreground" and applicable,
        "constraint_met": constraint_met,
        "max_feasible": max_feasible,
        "candidate_count": counts.numel() if active else 0,
        "eligible_candidate_count": eligible_count,
        "status": status,
    }


def make_cues(img, aud, ids, mode, family, cfg, severity, seed, policy, options=None):
    """Return CPU cues, true masks, and JSON-compatible per-sample metadata.

    ``truth`` has the existing ``img``/``aud`` keys. Metadata can be indexed as
    ``metadata['samples'][i]['modalities']['img' or 'aud']``. A missing foreground
    has coverage/max_feasible/constraint_met=None and evaluable=False; it never
    counts as satisfying the constraint. Infeasible placements select uniformly
    among all maximum-coverage windows and explicitly report constraint_met=False.
    Severity sets rectangle area (0.25 -> 14x14 for 28x28, 32x32 for 64x64).
    """
    settings = _options(options)
    severity = _finite_number(severity, "severity", maximum=1.0)
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if policy not in ("uniform", "foreground"):
        raise ValueError("policy must be uniform or foreground")
    if mode not in DEFAULT_MODES:
        raise ValueError("Both modalities must be present in the evaluation mode")
    if family not in (*LOCAL_FAMILIES, "clean"):
        raise ValueError("Unsupported local contamination family")
    if family == "clean" and mode != "clean_both":
        raise ValueError("The clean family requires clean_both mode")
    for name, target, ndim in (("img", img, 4), ("aud", aud, 3)):
        _check_tensor(target, name)
        if target.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions")
    if len(img) != len(aud):
        raise ValueError("Image and audio batch sizes must match")
    if isinstance(ids, (str, bytes)):
        raise ValueError("ids must contain one sample ID per batch member")
    ids = [value.item() if isinstance(value, torch.Tensor) and value.numel() == 1
           else value for value in ids]
    if len(ids) != len(img):
        raise ValueError("ids must contain one sample ID per batch member")
    normalized_ids = []
    for value in ids:
        if isinstance(value, bool) or not isinstance(value, (str, numbers.Integral)):
            raise ValueError("Each sample ID must be a string or integer")
        normalized_ids.append(int(value) if isinstance(value, numbers.Integral) else value)
    ids = normalized_ids
    if not isinstance(cfg, Mapping) or not isinstance(cfg.get("corruption"), Mapping):
        raise ValueError("cfg.corruption must be a mapping")
    corruption = cfg["corruption"]
    strengths = {name: _finite_number(corruption.get(f"{name}_noise_std",
                         corruption.get("noise_std", 0.2)), f"{name}_noise_std")
                 for name in ("img", "aud")}
    targets = {"img": img.detach().cpu(), "aud": aud.detach().cpu()}
    active = {
        "img": (mode in ("corrupt_both", "corrupt_img_clean_aud")
                and family != "clean" and severity > 0 and strengths["img"] > 0),
        "aud": (mode in ("corrupt_both", "clean_img_corrupt_aud")
                and family != "clean" and severity > 0 and strengths["aud"] > 0),
    }
    metadata = {
        "policy": policy, "options": settings, "requested_area_coverage": severity,
        "noise_field": "original per-sample/family/modality seed and full-image Torch draw",
        "samples": [],
    }
    with preserve_random_state():
        if policy == "uniform":
            ci, ca, truth = deterministic_cues(img, aud, ids, mode, family, cfg, severity, seed)
            cues = {"img": ci, "aud": ca}
        else:
            cues = {name: target.clone() for name, target in targets.items()}
            truth = {name: torch.zeros_like(target) for name, target in targets.items()}
        for index, sample_id in enumerate(ids):
            sample_metadata = {"sample_id": sample_id, "modalities": {}}
            for name, batch in targets.items():
                target = batch[index]
                foreground = foreground_mask(target, name, settings)
                if active[name]:
                    block_h, block_w = _block_shape(*target.shape[-2:], severity)
                    counts = _window_counts(foreground, block_h, block_w)
                    if policy == "foreground":
                        total = int(foreground.sum())
                        eligible = counts >= settings["min_foreground_coverage"] * total
                        if not bool(eligible.any()):
                            eligible = counts == counts.max()
                        positions = eligible.nonzero().tolist()
                        position_rng = random.Random(_sample_seed(
                            seed, sample_id, family, f"{name}|foreground_position"))
                        top, left = positions[position_rng.randrange(len(positions))]
                        mask = truth[name][index:index + 1]
                        mask[..., top:top + block_h, left:left + block_w] = 1
                        # Matching the old singleton shape is essential: a whole
                        # batch draw changes Gaussian fields across batch sizes.
                        singleton = batch[index:index + 1].clone()
                        _seed_corruption(_sample_seed(seed, sample_id, family, name))
                        if family == "local_gaussian":
                            field = (singleton + strengths[name] * torch.randn_like(singleton)).clamp(0, 1)
                        else:
                            field = (torch.rand_like(singleton) < 0.5).to(singleton.dtype)
                        cues[name][index:index + 1] = torch.where(mask.bool(), field, singleton)
                else:
                    counts = torch.zeros((0, 0), dtype=torch.long)
                sample_metadata["modalities"][name] = _sample_metadata(
                    target, truth[name][index], name, settings, active[name], counts, policy)
            metadata["samples"].append(sample_metadata)
    return cues["img"], cues["aud"], truth, metadata


def foreground_metrics(cue, recovered, target, truth, foreground_mask):
    """Pixel-pooled sufficient statistics; empty-region MSE/RMSE is None.

    Call for one sample and modality. Sum each region's SSE and pixel count
    before division when aggregating samples; do not average sample MSEs.
    """
    for name, value in (("cue", cue), ("recovered", recovered), ("target", target)):
        _check_tensor(value, name)
        if value.shape != target.shape:
            raise ValueError("cue, recovered, and target shapes must match")
    masks = []
    for name, value in (("truth", truth), ("foreground_mask", foreground_mask)):
        if not isinstance(value, torch.Tensor) or value.shape != target.shape:
            raise ValueError(f"{name} shape must equal target shape")
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError(f"{name} must be binary")
        masks.append(value.to(device=target.device, dtype=torch.bool))
    truth, foreground = masks
    errors = {
        "input": (cue.to(target.device).double() - target.double()).square(),
        "recovered": (recovered.to(target.device).double() - target.double()).square(),
    }
    regions = {
        "global": torch.ones_like(truth), "corrupt": truth, "clean": ~truth,
        "foreground": foreground, "background": ~foreground,
        "corrupt_foreground": truth & foreground,
        "corrupt_background": truth & ~foreground,
        "clean_foreground": ~truth & foreground,
        "clean_background": ~truth & ~foreground,
    }
    result = {}
    for name, region in regions.items():
        count = int(region.sum())
        result[f"{name}_pixels"] = count
        for source, error in errors.items():
            sse = float(error[region].sum())
            mse = safe_ratio(sse, count)
            result[f"{name}_{source}_sse"] = sse
            result[f"{name}_{source}_mse"] = mse
            result[f"{name}_{source}_rmse"] = math.sqrt(mse) if mse is not None else None
    result["foreground_coverage"] = safe_ratio(
        result["corrupt_foreground_pixels"], result["foreground_pixels"])
    result["mask_area_coverage"] = safe_ratio(result["corrupt_pixels"], target.numel())
    result["foreground_evaluable"] = result["foreground_pixels"] > 0
    result["clean_exactly_preserved"] = bool(torch.equal(
        recovered.to(target.device)[~truth], cue.to(target.device)[~truth]))
    return result
