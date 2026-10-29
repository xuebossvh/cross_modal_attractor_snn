"""Leakage-free local-contamination evaluation, shared by validation and CLI.

Region errors use *true* contamination support, never a detector-selected region.
Each sample/modality/family receives a stable corruption seed independent of batch
size and cue mode. Undefined rates and empty-region errors are represented by None.
"""

import hashlib
import math
import random
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

from common import build_cue, unpack_paired_batch


LOCAL_FAMILIES = ("local_gaussian", "local_impulse")
DEFAULT_MODES = ("clean_both", "corrupt_both", "clean_img_corrupt_aud",
                 "corrupt_img_clean_aud")


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def detection_rates(tp, fp, fn, tn):
    """Micro rates; an undefined denominator is not perfect performance or zero."""
    return dict(precision=safe_ratio(tp, tp + fp), recall=safe_ratio(tp, tp + fn),
                f1=safe_ratio(2 * tp, 2 * tp + fp + fn),
                iou=safe_ratio(tp, tp + fp + fn),
                false_positive_rate=safe_ratio(fp, fp + tn))


def _mask_like(mask, target, name):
    if mask is None:
        return torch.zeros_like(target, dtype=torch.bool)
    if mask.shape != target.shape:
        raise ValueError(f"{name} shape {tuple(mask.shape)} != {tuple(target.shape)}")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must contain only binary values")
    return mask.to(device=target.device, dtype=torch.bool)


def sample_metrics(cue, recovered, target, true_mask, predicted_mask):
    """Compute pixel-pooled sufficient statistics for one sample of one modality."""
    if cue.shape != target.shape or recovered.shape != target.shape:
        raise ValueError("cue, recovery, and target shapes must match exactly")
    if predicted_mask is None:
        raise ValueError("Detector prediction is required even for oracle/all diagnostics")
    for value in (cue, recovered, target):
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite evaluation value")
    truth = _mask_like(true_mask, target, "true mask")
    predicted = _mask_like(predicted_mask, target, "predicted mask")
    input_error = (cue.double() - target.double()).square()
    recovered_error = (recovered.double() - target.double()).square()
    result = {}
    regions = {"global": torch.ones_like(truth), "corrupt": truth, "clean": ~truth}
    for name, region in regions.items():
        pixels = int(region.sum())
        result[f"{name}_pixels"] = pixels
        for source, error in (("input", input_error), ("recovered", recovered_error)):
            sse = float(error[region].sum())
            result[f"{name}_{source}_sse"] = sse
            result[f"{name}_{source}_mse"] = safe_ratio(sse, pixels)
    counts = dict(tp=int((predicted & truth).sum()), fp=int((predicted & ~truth).sum()),
                  fn=int((~predicted & truth).sum()), tn=int((~predicted & ~truth).sum()))
    result.update({f"detector_{key}": value for key, value in counts.items()})
    result.update({f"detector_{key}": value for key, value in detection_rates(**counts).items()})
    return result


def summarize_rows(rows):
    if not rows:
        raise ValueError("Empty evaluation split")
    summary = {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows)}
    for modality in ("img", "aud"):
        values = {}
        for region in ("global", "corrupt", "clean"):
            pixels = sum(r[f"{modality}_{region}_pixels"] for r in rows)
            values[f"{region}_pixels"] = pixels
            values[f"{region}_valid_samples"] = sum(
                r[f"{modality}_{region}_pixels"] > 0 for r in rows)
            for source in ("input", "recovered"):
                sse = sum(r[f"{modality}_{region}_{source}_sse"] for r in rows)
                values[f"{region}_{source}_sse"] = sse
                values[f"{region}_{source}_mse"] = safe_ratio(sse, pixels)
        counts = {key: sum(r[f"{modality}_detector_{key}"] for r in rows)
                  for key in ("tp", "fp", "fn", "tn")}
        values["detector"] = dict(counts, **detection_rates(**counts))
        summary[modality] = values
    return summary


def _sample_seed(base_seed, sample_id, family, modality):
    text = f"{base_seed}|{sample_id}|{family}|{modality}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


@contextmanager
def preserve_random_state():
    """Validation must not change training's Python, NumPy, or Torch streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _seed_corruption(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def deterministic_cues(img, aud, ids, mode, family, cfg, severity, base_seed):
    """CPU singleton corruption avoids dependence on batching or accelerator RNG."""
    if mode not in DEFAULT_MODES:
        raise ValueError("Local-noise evaluation requires both modalities to be present")
    if family not in LOCAL_FAMILIES and family != "clean":
        raise ValueError(f"Unsupported local corruption family: {family}")
    images, audio, image_masks, audio_masks = [], [], [], []
    for index, sample_id in enumerate(ids):
        image = img[index:index + 1].detach().cpu()
        sound = aud[index:index + 1].detach().cpu()
        image_cue, audio_cue = image.clone(), sound.clone()
        image_mask, audio_mask = torch.zeros_like(image), torch.zeros_like(sound)
        if mode in ("corrupt_both", "corrupt_img_clean_aud"):
            _seed_corruption(_sample_seed(base_seed, sample_id, family, "img"))
            image_cue, _, masks = build_cue(
                image, sound, "corrupt_img_only", cfg, severity=severity,
                img_mode=family, aud_mode=family, return_masks=True)
            image_mask = masks["img"]
        if mode in ("corrupt_both", "clean_img_corrupt_aud"):
            _seed_corruption(_sample_seed(base_seed, sample_id, family, "aud"))
            _, audio_cue, masks = build_cue(
                image, sound, "corrupt_aud_only", cfg, severity=severity,
                img_mode=family, aud_mode=family, return_masks=True)
            audio_mask = masks["aud"]
        images.append(image_cue)
        audio.append(audio_cue)
        image_masks.append(image_mask)
        audio_masks.append(audio_mask)
    return torch.cat(images), torch.cat(audio), {"img": torch.cat(image_masks),
                                                "aud": torch.cat(audio_masks)}


def _conditions(cfg):
    settings = cfg.get("noise_evaluation", {}) or {}
    families = settings.get("families", list(LOCAL_FAMILIES))
    modes = settings.get("cue_modes", list(DEFAULT_MODES))
    if not families or any(f not in LOCAL_FAMILIES for f in families):
        raise ValueError(f"noise_evaluation.families must use {LOCAL_FAMILIES}")
    if not modes or any(m not in DEFAULT_MODES for m in modes):
        raise ValueError(f"noise_evaluation.cue_modes must use {DEFAULT_MODES}")
    if len(set(families)) != len(families) or len(set(modes)) != len(modes):
        raise ValueError("Duplicate evaluation families or modes")
    result = [(family, mode) for family in families for mode in modes if mode != "clean_both"]
    if "clean_both" in modes:
        result.insert(0, ("clean", "clean_both"))
    return result


@torch.no_grad()
def evaluate_local_noise(model, loader, cfg, device, split="val", mask_source="predicted",
                         max_samples=0, max_batches=0):
    """Run fixed local-noise cases; truth masks enter forward only for oracle."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if mask_source not in ("predicted", "oracle", "all"):
        raise ValueError("mask_source must be predicted, oracle, or all")
    if not cfg.get("noise_detector", {}).get("enabled", False):
        raise ValueError("Local-noise evaluation requires noise_detector.enabled")
    dataset = getattr(loader, "dataset", None)
    actual_split = getattr(dataset, "split", None)
    if actual_split is not None and actual_split != split:
        raise ValueError(f"Requested {split} evaluation but loader contains {actual_split}")
    if getattr(dataset, "train", False):
        raise ValueError("Evaluation cannot use the training dataset")
    if max_samples < 0 or max_batches < 0:
        raise ValueError("Evaluation limits must be non-negative")
    settings = cfg.get("noise_evaluation", {}) or {}
    seed = int(settings.get("seed", 20261029))
    severity = float(settings.get("severity", cfg["corruption"]["train_severity"]))
    if not math.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("Evaluation severity must be finite and in [0,1]")
    conditions = _conditions(cfg)
    rows, seen = [], 0
    was_training = model.training
    original_source = model.noise_mask_source
    model.noise_mask_source = mask_source
    try:
        model.eval()
        with preserve_random_state():
            for batch_index, batch in enumerate(loader):
                if max_batches and batch_index >= max_batches:
                    break
                img, aud, labels, pair_ids = unpack_paired_batch(batch)
                take = min(len(labels), max_samples - seen) if max_samples else len(labels)
                if take <= 0:
                    break
                img, aud, labels = img[:take], aud[:take], labels[:take]
                ids = (pair_ids[:take].tolist() if pair_ids is not None
                       else list(range(seen, seen + take)))
                img_device, aud_device = img.to(device), aud.to(device)
                labels_device = labels.to(device)
                for family, mode in conditions:
                    ci, ca, truth = deterministic_cues(
                        img, aud, ids, mode, family, cfg, severity, seed)
                    cue_cpu = {"img": ci, "aud": ca}
                    ci, ca = ci.to(device), ca.to(device)
                    # Never expose support labels to predicted/all forward calls.
                    forward_masks = ({k: value.to(device) for k, value in truth.items()}
                                     if mask_source == "oracle" else {"img": None, "aud": None})
                    out = model(x_img_cue=ci, x_aud_cue=ca, training_mode=False,
                                img_cue_mask=forward_masks["img"], aud_cue_mask=forward_masks["aud"])
                    # Transfer once per batch rather than synchronize CUDA for every
                    # scalar region/count of every sample.
                    recovered = {"img": out["recovered_img"].sigmoid().cpu(),
                                 "aud": out["recovered_aud"].cpu()}
                    predictions = {}
                    for name in ("img", "aud"):
                        prediction = out.get(f"{name}_predicted_mask")
                        if prediction is None:
                            raise ValueError(f"Model did not return {name}_predicted_mask")
                        predictions[name] = prediction.cpu()
                    correct = out["logits"].argmax(1).eq(labels_device).cpu()
                    for index, sample_id in enumerate(ids):
                        row = dict(sample_index=seen + index, sample_id=sample_id, split=split,
                                   family=family, cue_mode=mode, mask_source=mask_source,
                                   label=int(labels[index]), correct=int(correct[index]),
                                   severity=0.0 if family == "clean" else severity)
                        for name, target in (("img", img), ("aud", aud)):
                            metrics = sample_metrics(cue_cpu[name][index], recovered[name][index],
                                                     target[index].cpu(), truth[name][index],
                                                     predictions[name][index])
                            row.update({f"{name}_{key}": value for key, value in metrics.items()})
                        rows.append(row)
                seen += take
                if max_samples and seen >= max_samples:
                    break
    finally:
        model.noise_mask_source = original_source
        model.train(was_training)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["cue_mode"])].append(row)
    report = dict(schema_version=1, split=split, mask_source=mask_source, seed=seed,
                  threshold=float(cfg["noise_detector"].get("threshold", 0.5)),
                  threshold_source="checkpoint_config; not tuned on evaluation split",
                  severity=severity, n_unique_samples=seen,
                  noise_std={name: cfg["corruption"].get(f"{name}_noise_std",
                            cfg["corruption"].get("noise_std", 0.2)) for name in ("img", "aud")},
                  aggregation="SSE / true-region pixel count; detector rates from summed confusion counts",
                  summary=summarize_rows(rows),
                  conditions=[dict(family=family, cue_mode=mode, **summarize_rows(group))
                              for (family, mode), group in groups.items()])
    return report, rows


def validation_score(summary, cfg):
    settings = cfg.get("validation", {}).get("score", {}) or {}
    weights = {"img": float(settings.get("lambda_img", 1.0)),
               "aud": float(settings.get("lambda_aud", 4.0))}
    region_weight = float(settings.get("region_weight", 1.0))
    if any(not math.isfinite(w) or w < 0 for w in (*weights.values(), region_weight)):
        raise ValueError("Validation score weights must be finite and non-negative")
    if sum(weights.values()) == 0:
        raise ValueError("At least one modality must have nonzero validation score weight")
    score = 0.0
    for name, weight in weights.items():
        values = summary[name]
        score += weight * values["global_recovered_mse"]
        if values["corrupt_recovered_mse"] is not None:
            score += weight * region_weight * values["corrupt_recovered_mse"]
    return score


def validate_local_noise(model, loader, cfg, device):
    settings = cfg.get("validation", {}) or {}
    if settings.get("split", "val") != "val":
        raise ValueError("Local-noise checkpoint selection must use the validation split")
    report, _ = evaluate_local_noise(
        model, loader, cfg, device, split="val", mask_source="predicted",
        max_samples=int(settings.get("max_samples", 0) or 0),
        max_batches=int(settings.get("max_batches", 0) or 0))
    metrics = report["summary"]
    return dict(score=validation_score(metrics, cfg), img_mse=metrics["img"]["global_recovered_mse"],
                aud_mse=metrics["aud"]["global_recovered_mse"],
                img_masked_mse=metrics["img"]["corrupt_recovered_mse"],
                aud_masked_mse=metrics["aud"]["corrupt_recovered_mse"],
                acc=metrics["accuracy"], n=metrics["n"],
                pair_i2a_acc=float("nan"), pair_a2i_acc=float("nan"))
