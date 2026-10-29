"""Paired, leakage-free v16pro validation and publication-result evaluation.

All conditions use sample-ID seeded CPU corruption. True noise support is used
only for scoring (or explicitly labelled oracle diagnostics). Region statistics
are pooled by pixel counts, while model selection balances the 13 cue modes.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import math

import torch

from common import unpack_paired_batch
from data.v16_protocol import MODES, make_cues, make_targets
from scripts.noise_metrics import detection_rates, preserve_random_state, safe_ratio


LEGACY_PAIRS = tuple(zip(
    ("occlusion", "pixel_delete", "mask_vertical", "mask_horizontal", "salt_mask"),
    ("time_mask", "freq_mask", "feature_dropout", "partial_temporal", "time_freq_block")))
NOISE_FAMILIES = ("local_gaussian", "local_impulse")
REGIONS = ("global", "missing", "noise", "clean")
SOURCES = ("input", "recovered", "coarse", "hard_template", "soft_template", "oracle_template")
PARTIAL_MODES = ("corrupt_both", "clean_img_corrupt_aud", "corrupt_img_clean_aud")
NOISE_SCAN_MODES = ("noisy_both", "clean_img_noisy_aud", "noisy_img_clean_aud")


def conditions(cfg, *, validation=False, scans=False):
    """Deduplicated full modes; scans vary one factor and only three dual modes."""
    result = []
    settings = cfg.get("v16", {}) or {}
    default_severity = float(settings.get("missing_severity", .4))
    default_coverage = float(settings.get("noise_coverage", .25))
    default_std = float(settings.get("noise_std", .2))
    def sweep_values(key, default, maximum=1):
        values = [float(value) for value in settings.get(key, default)]
        if not values or values != sorted(set(values)) or any(not math.isfinite(v) or v <= 0 or (maximum is not None and v > maximum) for v in values):
            raise ValueError(f"v16.{key} must contain sorted, unique positive values in range")
        return values

    def add(mode, protocol="fixed", family_pair=None, sweep="main", severity=None,
            coverage=None, noise_std=None, examples=False):
        im, am = family_pair or (None, None)
        record = dict(mode=mode, protocol=protocol, sweep=sweep,
                      img_family=im, aud_family=am,
                      severity=default_severity if severity is None else severity,
                      coverage=default_coverage if coverage is None else coverage,
                      noise_std=default_std if noise_std is None else noise_std,
                      examples=examples)
        record["id"] = ":".join(str(record[k]) for k in
            ("sweep", "protocol", "mode", "img_family", "aud_family", "severity", "coverage", "noise_std"))
        result.append(record)

    for mode in MODES:
        if "noisy" in mode:
            for family in NOISE_FAMILIES:
                add(mode, family_pair=(family, family), examples=True)
        elif "corrupt" in mode:
            for index, pair in enumerate(LEGACY_PAIRS[:1] if validation else LEGACY_PAIRS):
                add(mode, family_pair=pair, examples=index == 0)
        else:
            add(mode, examples=True)
    if not validation:
        # C/A-only modes do not have a random corruption protocol to repeat.
        for mode in MODES:
            if "corrupt" in mode or "noisy" in mode:
                add(mode, protocol="random")
    if scans:
        for severity in sweep_values("missing_sweep", (.2, .4, .6, .8)):
            for mode in PARTIAL_MODES:
                add(mode, family_pair=LEGACY_PAIRS[0], sweep="missing_severity", severity=severity)
        for coverage in sweep_values("coverage_sweep", (.1, .25, .4)):
            for family in NOISE_FAMILIES:
                for mode in NOISE_SCAN_MODES:
                    add(mode, family_pair=(family, family), sweep="noise_coverage", coverage=coverage)
        for std in sweep_values("noise_std_sweep", (.1, .2, .3), maximum=None):
            for mode in NOISE_SCAN_MODES:
                add(mode, family_pair=("local_gaussian", "local_gaussian"),
                    sweep="gaussian_std", noise_std=std)
    # Main and scan baseline points refer to identical input conditions. They
    # remain separate reporting labels; evaluate() caches one forward per batch.
    if len({c["id"] for c in result}) != len(result):
        raise ValueError("Duplicate evaluation condition IDs")
    return result


def _as_mask(mask, target):
    if mask is None:
        return torch.zeros_like(target, dtype=torch.bool)
    if mask.shape != target.shape or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Region masks must be binary and exactly match the target")
    return mask.bool()


def region_metrics(target, predictions, missing, noise, predicted, state):
    """One sample; absent input/undefined regions use None, never invented zero."""
    if not torch.isfinite(target).all():
        raise ValueError("Non-finite target")
    missing, noise = _as_mask(missing, target), _as_mask(noise, target)
    if (missing & noise).any():
        raise ValueError("Known missing and noise truth must not overlap")
    present = state != "A"
    clean = (~missing & ~noise) if present else torch.zeros_like(missing)
    regions = dict(global_=torch.ones_like(missing), missing=missing, noise=noise, clean=clean)
    result = {"observed_pixels": int((~missing).sum()) if present else 0}
    for name, region in regions.items():
        name = "global" if name == "global_" else name
        pixels = int(region.sum())
        result[name + "_pixels"] = pixels
        for source, prediction in predictions.items():
            if prediction is None:
                result[f"{name}_{source}_sse"] = None
                result[f"{name}_{source}_mse"] = None
                continue
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise ValueError("Finite prediction must exactly match target shape")
            sse = float((prediction.double() - target.double()).square()[region].sum())
            result[f"{name}_{source}_sse"] = sse
            result[f"{name}_{source}_mse"] = safe_ratio(sse, pixels)
    if present:
        if predicted is None:
            raise ValueError("Present modality requires a detector prediction")
        predicted = _as_mask(predicted, target)
        valid = ~missing
        counts = dict(tp=int((predicted & noise & valid).sum()),
                      fp=int((predicted & ~noise & valid).sum()),
                      fn=int((~predicted & noise & valid).sum()),
                      tn=int((~predicted & ~noise & valid).sum()))
    else:
        counts = dict(tp=0, fp=0, fn=0, tn=0)
    result.update({"detector_" + key: value for key, value in counts.items()})
    return result


def batch_region_metrics(target, predictions, missing, noise, predicted, state):
    """Vectorized sufficient statistics; GPU synchronization is once per column."""
    missing, noise = _as_mask(missing, target), _as_mask(noise, target)
    if (missing & noise).any() or not torch.isfinite(target).all():
        raise ValueError("Invalid target or overlapping true supports")
    present = state != "A"
    valid = ~missing if present else torch.zeros_like(missing)
    regions = dict(global_=torch.ones_like(missing), missing=missing, noise=noise,
                   clean=(valid & ~noise))
    columns = {"observed_pixels": valid.flatten(1).sum(1)}
    errors = {}
    for source, prediction in predictions.items():
        if prediction is not None:
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise ValueError("Invalid prediction")
            errors[source] = (prediction.double() - target.double()).square().flatten(1)
    for region, mask in regions.items():
        region = "global" if region == "global_" else region
        columns[region + "_pixels"] = mask.flatten(1).sum(1)
        for source in predictions:
            if source in errors:
                columns[f"{region}_{source}_sse"] = (errors[source] * mask.flatten(1)).sum(1)
    if present:
        if predicted is None:
            raise ValueError("Present modality requires predicted mask")
        pred = _as_mask(predicted, target)
        counts = dict(tp=pred & noise & valid, fp=pred & ~noise & valid,
                      fn=~pred & noise & valid, tn=~pred & ~noise & valid)
    else:
        counts = {key: valid for key in ("tp", "fp", "fn", "tn")}
    for key, mask in counts.items():
        columns["detector_" + key] = mask.flatten(1).sum(1)
    # One small transfer rather than a synchronization for each scalar sample.
    keys = list(columns)
    matrix = torch.stack([columns[key].double() for key in keys], dim=1).cpu().tolist()
    result = [{key: value for key, value in zip(keys, row)} for row in matrix]
    for row in result:
        for region in REGIONS:
            for source in predictions:
                sse = row.get(f"{region}_{source}_sse")
                row[f"{region}_{source}_sse"] = sse
                row[f"{region}_{source}_mse"] = safe_ratio(sse, row[region + "_pixels"]) if sse is not None else None
    return result


class Accumulator:
    """Bounded-memory sufficient statistics, retaining distinct target types."""
    def __init__(self, condition):
        self.condition = {key: value for key, value in condition.items() if key != "examples"}
        self.n = self.correct = 0
        self.values = {name: defaultdict(float) for name in ("img", "aud")}
        self.states, self.kinds = {}, {}

    def add(self, row):
        self.n += 1
        self.correct += row["correct"]
        for name in ("img", "aud"):
            self.states[name], self.kinds[name] = row[name + "_state"], row[name + "_target_kind"]
            values = self.values[name]
            for region in REGIONS:
                pixels = row[f"{name}_{region}_pixels"]
                values[region + "_pixels"] += pixels
                for source in SOURCES:
                    key = f"{region}_{source}_sse"
                    value = row.get(name + "_" + key)
                    if value is not None:
                        values[key] += value
                        values[f"{region}_{source}_valid_pixels"] += pixels
            for key in ("observed_pixels", "detector_tp", "detector_fp", "detector_fn", "detector_tn"):
                values[key] += row[name + "_" + key]
            for source in ("recovered", "hard_template", "soft_template", "oracle_template"):
                key = source + "_correct"
                if row.get(name + "_" + key) is not None:
                    values[key] += row[name + "_" + key]
                    values[source + "_recognizer_n"] += 1

    def finish(self):
        result = dict(self.condition, n=self.n, index_acc=safe_ratio(self.correct, self.n))
        for name, values in self.values.items():
            result[name + "_state"] = self.states.get(name)
            result[name + "_target_kind"] = self.kinds.get(name)
            for key, value in values.items():
                result[name + "_" + key] = value
            for region in REGIONS:
                for source in SOURCES:
                    result[f"{name}_{region}_{source}_mse"] = safe_ratio(
                        values[f"{region}_{source}_sse"], values[f"{region}_{source}_valid_pixels"])
            counts = {key: values["detector_" + key] for key in ("tp", "fp", "fn", "tn")}
            result.update({name + "_detector_" + key: value for key, value in detection_rates(**counts).items()})
            for source in ("recovered", "hard_template", "soft_template", "oracle_template"):
                result[name + "_" + source + "_content_acc"] = safe_ratio(
                    values[source + "_correct"], values[source + "_recognizer_n"])
            total = values["global_pixels"]
            result[name + "_actual_missing_fraction"] = safe_ratio(values["missing_pixels"], total)
            result[name + "_actual_noise_fraction"] = safe_ratio(values["noise_pixels"], total)
        return result


def _templates(prototype, probabilities, labels):
    hard = prototype[probabilities.argmax(1)]
    soft = (probabilities @ prototype.flatten(1)).reshape_as(hard)
    return dict(hard_template=hard, soft_template=soft, oracle_template=prototype[labels])


def _copy_visible(prediction, cue, repair):
    if cue is None:
        return prediction
    if repair is None:
        raise ValueError("Present cue requires the actual applied repair mask")
    return repair * prediction + (1 - repair) * cue


def _canonical_key(condition):
    return tuple(condition[key] for key in
                 ("mode", "protocol", "img_family", "aud_family", "severity", "coverage", "noise_std"))


@torch.no_grad()
def evaluate(model, loader, cfg, device, *, split="test", recognizer=None,
             mask_source="predicted", max_batches=0, max_samples=0,
             validation=False, scans=False, examples=0, row_sink=None):
    """Return (report, captured_examples), with optional streamed per-sample rows."""
    if split not in ("val", "test") or mask_source not in ("predicted", "oracle", "all"):
        raise ValueError("Invalid evaluation split or mask source")
    if min(max_batches, max_samples, examples) < 0:
        raise ValueError("Evaluation limits must be non-negative")
    dataset = loader.dataset
    if getattr(dataset, "train", False) or getattr(dataset, "split", split) != split:
        raise ValueError("Evaluation loader must use the requested non-training split")
    proto_img = dataset.prototype_img.to(device)
    proto_aud = dataset.prototype_aud.to(device)
    cases = conditions(cfg, validation=validation, scans=scans)
    repeated = Counter(_canonical_key(condition) for condition in cases)
    accumulators = [Accumulator(case) for case in cases]
    captured, seen = defaultdict(list), 0
    original_source = model.noise_mask_source
    module_modes = [(module, module.training) for module in model.modules()]
    recognizer_modes = ([(m, m.training) for m in recognizer.modules()] if recognizer is not None else [])
    model.noise_mask_source = mask_source
    model.eval()
    if recognizer is not None:
        recognizer.eval()
    seed = int(cfg.get("v16", {}).get("evaluation_seed", 20261030))
    try:
        with preserve_random_state():
            for batch_index, batch in enumerate(loader):
                if max_batches and batch_index >= max_batches:
                    break
                img, aud, labels, pair_ids = unpack_paired_batch(batch)
                take = min(len(labels), max_samples - seen) if max_samples else len(labels)
                if take <= 0:
                    break
                img, aud, labels = img[:take], aud[:take], labels[:take]
                labels_cpu = labels.cpu()
                ids = pair_ids[:take].tolist() if pair_ids is not None else list(range(seen, seen + take))
                img, aud, labels = img.to(device), aud.to(device), labels.to(device)
                cache = {}
                for condition, accumulator in zip(cases, accumulators):
                    key = _canonical_key(condition)
                    if key not in cache:
                        cue = make_cues(img, aud, condition["mode"], cfg, seed=seed, ids=ids,
                                        protocol=condition["protocol"], severity=condition["severity"],
                                        coverage=condition["coverage"], noise_std=condition["noise_std"],
                                        img_family=condition["img_family"], aud_family=condition["aud_family"])
                        kwargs = dict(x_img_cue=cue["img"], x_aud_cue=cue["aud"], training_mode=False,
                                      img_cue_mask=cue["missing"]["img"], aud_cue_mask=cue["missing"]["aud"])
                        if mask_source == "oracle":
                            kwargs.update(img_noise_mask=cue["noise"]["img"], aud_noise_mask=cue["noise"]["aud"])
                        out = model(**kwargs)
                        ti, ta, ki, ka = make_targets(img, aud, labels, proto_img, proto_aud, condition["mode"])
                        probabilities = out["logits"].softmax(1)
                        prediction = out["logits"].argmax(1)
                        data = {}
                        for name, target, prototype, kind in (("img", ti, proto_img, ki), ("aud", ta, proto_aud, ka)):
                            rec = out["recovered_" + name]
                            coarse = out["recovered_" + name + "_coarse"]
                            if name == "img":
                                rec, coarse = rec.sigmoid(), coarse.sigmoid()
                            sources = dict(input=cue[name], recovered=rec, coarse=coarse)
                            if not validation:
                                sources.update({source: _copy_visible(value, cue[name], out.get(name + "_repair_mask"))
                                                for source, value in _templates(prototype, probabilities, labels).items()})
                            recognitions = {}
                            if recognizer is not None:
                                head = getattr(recognizer, name)
                                for source in ("recovered", "hard_template", "soft_template", "oracle_template"):
                                    recognitions[source] = head(sources[source]).argmax(1).eq(labels).cpu()
                            metric_rows = batch_region_metrics(target, sources, cue["missing"][name],
                                                               cue["noise"][name], out.get(name + "_predicted_mask"),
                                                               cue["states"][name])
                            data[name] = dict(target=target.cpu(), kind=kind, metrics=metric_rows,
                                              sources={s: v.cpu() if v is not None else None for s, v in sources.items()},
                                              predicted=out.get(name + "_predicted_mask"),
                                              repair=out.get(name + "_repair_mask"), recognitions=recognitions)
                            for field in ("predicted", "repair"):
                                if data[name][field] is not None:
                                    data[name][field] = data[name][field].cpu()
                        for name in ("img", "aud"):
                            if cue[name] is not None:
                                cue[name] = cue[name].cpu()
                            for field in ("missing", "noise"):
                                if cue[field][name] is not None:
                                    cue[field][name] = cue[field][name].cpu()
                        prediction, probabilities = prediction.cpu(), probabilities.cpu()
                        if repeated[key] > 1:
                            cache[key] = (cue, data, prediction, probabilities)
                    else:
                        cue, data, prediction, probabilities = cache[key]
                    for index, sample_id in enumerate(ids):
                        row = dict(condition_id=condition["id"], sweep=condition["sweep"],
                                   protocol=condition["protocol"], cue_mode=condition["mode"],
                                   sample_index=seen + index, sample_id=sample_id, label=int(labels_cpu[index]),
                                   index_prediction=int(prediction[index]), correct=int(prediction[index] == labels_cpu[index]),
                                   mask_source=mask_source)
                        for name in ("img", "aud"):
                            item = data[name]
                            state = cue["states"][name]
                            sources = {s: v[index] if v is not None else None for s, v in item["sources"].items()}
                            missing = cue["missing"][name]
                            noise = cue["noise"][name]
                            missing = missing[index].cpu() if missing is not None else None
                            noise = noise[index].cpu() if noise is not None else None
                            pred = item["predicted"][index] if item["predicted"] is not None else None
                            metrics = item["metrics"][index]
                            row.update({name + "_" + k: v for k, v in metrics.items()})
                            row.update({name + "_state": state, name + "_target_kind": item["kind"],
                                        name + "_family": cue[name + "_family"][index]})
                            for source, correct in item["recognitions"].items():
                                row[name + "_" + source + "_correct"] = int(correct[index])
                            if condition["examples"] and seen + index < examples:
                                captured[condition["id"] + "|" + name].append(dict(
                                    modality=name, sample_id=sample_id, sample_index=seen + index,
                                    label=int(labels_cpu[index]), index_prediction=int(prediction[index]),
                                    content_prediction_correct=row.get(name + "_recovered_correct"),
                                    target_kind=item["kind"], state=state, condition=condition,
                                    target=item["target"][index].clone(), cue=sources["input"],
                                    missing=missing, noise=noise, predicted=pred,
                                    repair=item["repair"][index] if item["repair"] is not None else None,
                                    recovered=sources["recovered"].clone()))
                        accumulator.add(row)
                        if row_sink is not None:
                            row_sink(row)
                seen += take
                if batch_index % 20 == 0 and not validation:
                    print(f"[evaluation] {seen} samples; {len(cases)} conditions", flush=True)
                if max_samples and seen >= max_samples:
                    break
    finally:
        model.noise_mask_source = original_source
        for module, training in module_modes + recognizer_modes:
            module.training = training
    if seen == 0:
        raise ValueError("Empty evaluation split")
    report = dict(schema="v16pro.evaluation.v1", split=split, mask_source=mask_source,
                  n_unique_samples=seen, partial_evaluation=bool(max_batches or max_samples),
                  evaluation_seed=seed, conditions=[a.finish() for a in accumulators],
                  notes={"regions": "SSE / true-region pixels; undefined denominators are null",
                         "targets": "present = current clean instance; absent = train-only category medoid",
                         "templates": "hard/soft use this model's same Index probabilities and same applied paste-back mask; oracle template uses true label for diagnosis only",
                         "coarse": "decoder output before image refiner and visible paste-back",
                         "examples": "first sample indices, unfiltered; captured in the evaluated forward",
                         "scans": "main model only, fixed canonical missing family; three dual-modality modes; factors varied separately",
                         "random": "families sampled independently per sample/modality with stable sample-ID seeds"})
    report["notes"]["validation_selection"] = "Equal mean over 13 cue modes, equal family mean within each mode; per modality C uses coarse global MSE, P true missing MSE, N true noise MSE, A category global MSE; score = lambda_img*image mean + lambda_aud*audio mean (defaults 1 and 4). This scalar is selection-only, not a pooled reported recovery metric."
    return report, dict(captured)


def selection_metrics(report, cfg):
    """Equal mode weight; equal family weight within mode, no test tuning."""
    groups = defaultdict(list)
    for row in report["conditions"]:
        if row["sweep"] == "main" and row["protocol"] == "fixed":
            groups[row["mode"]].append(row)
    if set(groups) != set(MODES):
        raise ValueError("Selection requires all 13 cue modes")
    means = {"acc": [], "img_mse": [], "aud_mse": []}
    for mode in MODES:
        means["acc"].append(sum(row["index_acc"] for row in groups[mode]) / len(groups[mode]))
        for name in ("img", "aud"):
            values = []
            for row in groups[mode]:
                state = row[name + "_state"]
                region = {"C": "global", "P": "missing", "N": "noise", "A": "global"}[state]
                source = "coarse" if state == "C" else "recovered"
                value = row[f"{name}_{region}_{source}_mse"]
                if value is not None:
                    values.append(value)
            if not values:
                raise ValueError(f"No defined selection region for {mode}/{name}")
            means[name + "_mse"].append(sum(values) / len(values))
    metrics = {key: sum(values) / len(values) for key, values in means.items()}
    weights = cfg.get("validation", {}).get("score", {})
    metrics["score"] = float(weights.get("lambda_img", 1.0)) * metrics["img_mse"] + float(weights.get("lambda_aud", 4.0)) * metrics["aud_mse"]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite validation score")
    return metrics


def validate_model(model, loader, cfg, device, max_batches=0, max_samples=0):
    report, _ = evaluate(model, loader, cfg, device, split="val", validation=True,
                         max_batches=max_batches, max_samples=max_samples)
    return selection_metrics(report, cfg)
