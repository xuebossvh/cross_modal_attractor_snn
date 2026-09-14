"""Full-test per-item evidence, independent content scoring and Index probes."""

import bootstrap  # noqa: F401
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from common import CUE_MODES, build_cue, load_config, select_targets, set_seed, unpack_paired_batch
from data.dataset import build_loaders
from models.frozen_base import file_sha256
from models.network import CrossModalSNN
from models.paper_baselines import CleanRecognizers, pasteback
from scripts.paper_baseline import make_model
from scripts.evaluate import _paired_cross_metrics, _same_class_indices, _wrong_class_indices
from paths import resolve_from_root


def region_values(rec, target, mask, power=2):
    if mask is None:
        return torch.full((len(rec),), float("nan"), device=rec.device)
    denominator = mask.flatten(1).sum(1)
    values = ((rec - target).abs().pow(power) * mask).flatten(1).sum(1) / denominator.clamp_min(1)
    return values.masked_fill(denominator == 0, float("nan"))


def global_ssim(rec, target):
    x, y = rec.flatten(1), target.flatten(1)
    mx, my = x.mean(1), y.mean(1)
    vx, vy = (x - mx[:, None]).square().mean(1), (y - my[:, None]).square().mean(1)
    covariance = ((x - mx[:, None]) * (y - my[:, None])).mean(1)
    return ((2*mx*my + .01**2) * (2*covariance + .03**2) /
            ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2)))


def reconstruction_metrics(ri, ra, ti, ta, masks, recognizer, labels):
    result = {}
    for name, rec, target in (("img", ri, ti), ("aud", ra, ta)):
        mask = masks[name]
        result[name + "_mse"] = (rec - target).square().flatten(1).mean(1)
        result[name + "_global_ssim"] = global_ssim(rec, target)
        result[name + "_psnr"] = -10 * torch.log10(result[name + "_mse"].clamp_min(1e-8))
        for region, region_mask in (("missing", mask), ("visible", 1-mask if mask is not None else torch.ones_like(target))):
            for metric, power in (("mse", 2), ("l1", 1)):
                result[f"{name}_{region}_{metric}"] = region_values(rec, target, region_mask, power)
        result[name + "_missing_fraction"] = (mask.flatten(1).mean(1) if mask is not None else torch.zeros_like(labels, dtype=torch.float))
        logits = getattr(recognizer, name)(rec)
        result[name + "_external_acc"] = (logits.argmax(1) == labels).float()
    tr, rr = ta.flatten(1), ra.flatten(1)
    k = max(1, round(.15 * tr.shape[1]))
    tm = torch.zeros_like(tr).scatter(1, tr.topk(k, dim=1).indices, 1.)
    rm = torch.zeros_like(rr).scatter(1, rr.topk(k, dim=1).indices, 1.)
    result["aud_top15_recall"] = (tm * rm).sum(1) / k
    result["aud_foreground_missing_mse"] = region_values(ra, ta,
        tm.reshape_as(ta) * masks["aud"] if masks["aud"] is not None else None)
    for name, values in (("rec", rr), ("target", tr)):
        result[f"aud_{name}_mean"] = values.mean(1)
        result[f"aud_{name}_std"] = values.std(1, unbiased=False)
        result[f"aud_{name}_max"] = values.max(1).values
    return result


def sample_cues(img, aud, identities, mode, cfg, severity, families, seed, protocol):
    # Each mask is keyed by image identity, not batch size or model RNG usage.
    images, audio, imasks, amasks, selected = [], [], [], [], []
    for i, identity in enumerate(identities):
        token = f"{identity['image_id']}|{mode}|{severity}|{families}|{seed}|{protocol}"
        value = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little")
        set_seed(value)
        pair = families
        if protocol == "random":
            pair = (random.choice(cfg["corruption"]["eval_fixed"]["img_modes"]),
                    random.choice(cfg["corruption"]["eval_fixed"]["aud_modes"]))
        ci, ca, masks = build_cue(img[i:i+1], aud[i:i+1], mode, cfg, severity=severity,
                                 img_mode=pair[0], aud_mode=pair[1], return_masks=True)
        images.append(ci)
        audio.append(ca)
        imasks.append(masks.get("img"))
        amasks.append(masks.get("aud"))
        selected.append(pair)
    def join(items):
        return None if items[0] is None else torch.cat(items)
    return join(images), join(audio), {"img": join(imasks), "aud": join(amasks)}, selected


def index_probe(model, img, aud, labels, seed=5678):
    """Independent membrane perturbations; no gradients, checkpoint changes or bias drive."""
    ki = model.memory.K_img(model.img_encoder(img))
    ka = model.memory.K_aud(model.aud_encoder(model._normalize_audio_for_encoder(aud)))
    layer, duration = model.memory.index, ki.shape[0]
    set_seed(seed)
    noise = torch.randn(len(labels), layer.n_index, device=img.device) * .2
    rows = []
    reference = layer(ki, ka, extra_steps=10, return_trace=True)
    for withdraw in sorted(set((max(1, duration//4), max(1, duration//2), duration))):
        clean = layer(ki, ka, external_steps=withdraw, extra_steps=10, return_trace=True)
        perturbed = layer(ki, ka, external_steps=withdraw, extra_steps=10,
                          perturb_step=withdraw, perturbation=noise, return_trace=True)
        for t in range(duration + 10):
            start = max(0, t-4)
            cr = clean["spikes"][start:t+1].mean(0)
            pr = perturbed["spikes"][start:t+1].mean(0)
            rr = reference["spikes"][start:t+1].mean(0)
            pred = model.classifier(pr).argmax(1)
            rows.append(dict(withdraw_step=withdraw, step=t, n=len(labels),
                clean_accuracy=float((model.classifier(cr).argmax(1) == labels).float().mean()),
                perturbed_accuracy=float((pred == labels).float().mean()),
                agreement_with_unperturbed=float((pred == model.classifier(cr).argmax(1)).float().mean()),
                rate_mse_to_unperturbed=float((pr-cr).square().mean()),
                rate_mse_to_full_cue=float((pr-rr).square().mean()),
                voltage_mse=float((perturbed["voltage"][t]-clean["voltage"][t]).square().mean()),
                firing_rate=float(pr.mean())))
    return rows


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--recognizer", required=True)
    ap.add_argument("--severities", type=float, nargs="+", default=[.4])
    ap.add_argument("--mask_seeds", type=int, nargs="+", default=[5678])
    ap.add_argument("--all_family_pairs", action="store_true")
    ap.add_argument("--smoke_batches", type=int, default=0)
    args = ap.parse_args()
    if any(not 0 <= x <= 1 for x in args.severities) or args.smoke_batches < 0:
        ap.error("Severity must be within [0,1]; smoke_batches must be nonnegative")
    cfg = load_config(args.config)
    directory = Path(args.config).parent / ("smoke_evaluation" if args.smoke_batches else "evaluation")
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    _, loader = build_loaders(cfg, eval_split="test")
    dataset = loader.dataset
    if not args.smoke_batches and (not dataset.use_real_audio or dataset._mode != "mnist"):
        raise ValueError("Formal v13pro evaluation requires real MNIST and FSDD")
    protos = (dataset.prototype_img.to(device), dataset.prototype_aud.to(device))
    kind = cfg["paper"]["experiment"]
    is_snn = kind not in ("classifier", "cue_cnn", "conditioned_cnn", "matched_cnn")
    state = torch.load(resolve_from_root(cfg["train"]["eval_ckpt_path"]), map_location="cpu")
    if state.get("cfg", {}).get("_audio_norm_stats") != cfg.get("_audio_norm_stats"):
        raise ValueError("Evaluation normalization differs from training checkpoint")
    if not args.smoke_batches:
        for section in ("paper", "dims", "snn", "index", "ablation", "audio_local_cue", "cross_key_conditioning"):
            if state.get("cfg", {}).get(section) != cfg.get(section):
                raise ValueError(f"Checkpoint configuration mismatch: {section}")
    model = (CrossModalSNN(cfg) if is_snn else make_model(kind, cfg)).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    external_state = torch.load(args.recognizer, map_location="cpu")
    if external_state.get("kind") != "recognizer":
        raise ValueError("External recognizer must be trained clean-only in its own task")
    if external_state.get("cfg", {}).get("_audio_norm_stats") != cfg.get("_audio_norm_stats"):
        raise ValueError("Independent recognizer normalization differs from evaluation")
    for key in ("paper_split",):
        if external_state["cfg"]["data"][key] != cfg["data"][key]:
            raise ValueError("Recognizer and evaluated model use different data splits")
    external = CleanRecognizers().to(device)
    external.load_state_dict(external_state["model"])
    external.eval()
    identities = [dataset.evaluation_identity(i) for i in range(len(dataset))]
    manifest_hash = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    with (directory / "test_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(identities, stream)
    fixed = cfg["corruption"]["eval_fixed"]
    families = list(itertools.product(fixed["img_modes"], fixed["aud_modes"])) if args.all_family_pairs else list(zip(fixed["img_modes"], fixed["aud_modes"]))
    summaries, probe_sums, probe_counts = {}, {}, {}
    parameters = sum(p.numel() for p in model.parameters())
    prefix = tuple(cfg["train"].get("trainable_prefixes", [])) if is_snn else ()
    trainable = sum(p.numel() for name, p in model.named_parameters() if not prefix or name.startswith(prefix))
    csv_path = directory / "per_item.csv.gz"
    writer, fields = None, None
    total_forward_seconds, forward_exposures, external_clean = 0., 0, []
    with gzip.open(csv_path, "wt", encoding="utf-8", newline="") as stream:
        for protocol in ("fixed", "random"):
            for severity, mask_seed, family, mode in itertools.product(args.severities, args.mask_seeds,
                    families if protocol == "fixed" else [("random", "random")], CUE_MODES):
                offset = 0
                for bi, batch in enumerate(loader):
                    if args.smoke_batches and bi >= args.smoke_batches:
                        break
                    img, aud, labels, _ = unpack_paired_batch(batch)
                    ids = identities[offset:offset+len(labels)]
                    offset += len(labels)
                    img, aud, labels = img.to(device), aud.to(device), labels.to(device)
                    ci, ca, masks, selected = sample_cues(img, aud, ids, mode, cfg, severity, family, mask_seed, protocol)
                    ti, ta, ik, ak = select_targets(mode, img, aud, *protos, labels)
                    em = {"img": torch.ones_like(ti) if ci is None else masks["img"],
                          "aud": torch.ones_like(ta) if ca is None else masks["aud"]}
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    paired = {}
                    variants = {}
                    if is_snn:
                        kwargs = dict(x_img_cue=ci, x_aud_cue=ca, img_cue_mask=masks["img"], aud_cue_mask=masks["aud"], training_mode=False)
                        normal = model(**kwargs)
                        variants[kind] = (normal["logits"], normal["recovered_img"].sigmoid(), normal["recovered_aud"])
                    elif kind == "classifier":
                        logs = ([model.img(ci)] if ci is not None else []) + ([model.aud(ca)] if ca is not None else [])
                        logits = torch.stack(logs).mean(0)
                        for variant, prob in (("medoid_hard", F.one_hot(logits.argmax(1), 10).float()),
                                              ("medoid_soft", logits.softmax(1)),
                                              ("oracle_medoid", F.one_hot(labels, 10).float())):
                            ri = torch.einsum("bc,cnhw->bnhw", prob, protos[0])
                            ra = torch.einsum("bc,chw->bhw", prob, protos[1])
                            variants[variant] = (logits, pasteback(ri, ci, masks["img"]), pasteback(ra, ca, masks["aud"]))
                    else:
                        variants[kind] = model(ci, ca, masks, *protos)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    total_forward_seconds += time.perf_counter() - started
                    forward_exposures += len(labels)
                    if is_snn:
                        zero = model(**kwargs, disable_img_to_aud_cross=True, disable_aud_to_img_cross=True)
                        wp, wv = _wrong_class_indices(labels)
                        sp, sv = _same_class_indices(labels)
                        def intervention(perm):
                            over = {}
                            for name in ("img", "aud"):
                                key = normal.get("key_" + name)
                                if key is not None:
                                    over[f"cross_key_{name}_rate_override"] = key.mean(0)[perm]
                            return model(**kwargs, **over)
                        wrong, same = intervention(wp), intervention(sp)
                        for candidate in (zero, wrong, same):
                            if not torch.equal(candidate["index_state"], normal["index_state"]):
                                raise RuntimeError("Cross-Key intervention changed Index")
                        paired = _paired_cross_metrics(normal, zero, wrong, same, ti, ta, em["img"], em["aud"], wv, sv)
                    first_scenario = (protocol == "fixed" and severity == args.severities[0] and
                                      mask_seed == args.mask_seeds[0] and family == families[0] and mode == "clean_both")
                    if first_scenario:
                        external_clean.append(((external.img(img).argmax(1) == labels).sum().item(),
                                               (external.aud(aud).argmax(1) == labels).sum().item(), len(labels)))
                        if is_snn:
                            for row in index_probe(model, img, aud, labels, mask_seed+bi):
                                key = (row["withdraw_step"], row["step"])
                                probe_counts[key] = probe_counts.get(key, 0) + row["n"]
                                for metric, value in row.items():
                                    if metric not in ("n", "step", "withdraw_step"):
                                        token = (*key, metric)
                                        probe_sums[token] = probe_sums.get(token, 0.) + value * row["n"]
                    for variant, (logits, ri, ra) in variants.items():
                        metrics = reconstruction_metrics(ri, ra, ti, ta, em, external, labels)
                        nan = torch.full_like(labels, float("nan"), dtype=torch.float)
                        accuracy = (logits.argmax(1) == labels).float()
                        metrics["index_acc"] = accuracy if is_snn else nan
                        metrics["classifier_acc"] = accuracy if not is_snn else nan
                        for name, (values, valid) in paired.items():
                            metrics[name] = values.masked_fill(~valid, float("nan"))
                        # A fixed union prevents late cue modes introducing new CSV columns.
                        for direction in ("img2aud", "aud2img"):
                            for metric in ("correct_gain", "wrong_damage", "same_damage", "normal_mse", "zero_mse", "wrong_mse", "same_class_mse", "win_zero", "win_wrong", "win_both", "gate", "ratio"):
                                metrics.setdefault(direction + "_" + metric, nan)
                        values = {k: v.detach().cpu().tolist() for k, v in metrics.items()}
                        for i, identity in enumerate(ids):
                            row = dict(experiment=variant, seed=cfg["seed"], protocol=protocol, severity=severity,
                                       mask_seed=mask_seed, img_family=selected[i][0], aud_family=selected[i][1],
                                       family_group="/".join(family), cue=mode, img_target=ik, aud_target=ak, **identity)
                            row.update({k: v[i] for k, v in values.items()})
                            if writer is None:
                                fields = list(row)
                                writer = csv.DictWriter(stream, fieldnames=fields)
                                writer.writeheader()
                            writer.writerow(row)
                            key = json.dumps([variant, protocol, severity, mask_seed, row["family_group"], mode])
                            totals = summaries.setdefault(key, {})
                            for metric in metrics:
                                value = row[metric]
                                if math.isfinite(value):
                                    total, count = totals.get(metric, (0., 0))
                                    totals[metric] = (total+value, count+1)
                if not args.smoke_batches and offset != len(dataset):
                    raise RuntimeError(f"Incomplete test coverage: {offset}/{len(dataset)}")
                print(f"[paper-eval] {kind} {protocol} sev={severity} seed={mask_seed} {family} {mode} n={offset}", flush=True)
    summary = {key: {metric: dict(mean=total/n, n=n) for metric, (total, n) in val.items()} for key, val in summaries.items()}
    (directory / "summary.json").write_text(json.dumps(clean_json(summary), indent=2), encoding="utf-8")
    probes = [dict(withdraw_step=a, step=b, n=n, **{m: value/n for (x,y,m),value in probe_sums.items() if (x,y)==(a,b)}) for (a,b),n in probe_counts.items()]
    (directory / "index_probes.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")
    clean_n = sum(x[2] for x in external_clean)
    metadata = dict(full_test=not bool(args.smoke_batches), test_images=len(dataset),
        unique_audio=len({x["audio_id"] for x in identities}), speakers=sorted({x["speaker"] for x in identities}),
        manifest_sha256=manifest_hash, model_sha256=file_sha256(cfg["train"]["eval_ckpt_path"]),
        recognizer_sha256=file_sha256(args.recognizer), parameters=parameters, configured_trainable_parameters=trainable,
        device=str(device), torch_version=torch.__version__, forward_exposures=forward_exposures,
        forward_seconds=total_forward_seconds, forward_ms_per_exposure=1000*total_forward_seconds/forward_exposures,
        external_clean_img_acc=sum(x[0] for x in external_clean)/clean_n,
        external_clean_aud_acc=sum(x[1] for x in external_clean)/clean_n,
        external_clean_n=clean_n, config=cfg, schema=1,
        artifacts={name: file_sha256(directory / name) for name in
                   ("per_item.csv.gz", "summary.json", "index_probes.json", "test_manifest.json")},
        notes=["global_ssim is legacy global, not windowed SSIM", "latency includes batch forward only, not energy",
               "Index probes use a 5-step rate readout; stability is not proof of an attractor",
               "paired key donors are within batches; invalid matches are NaN", "oracle_medoid uses labels and is diagnostic only"])
    (directory / "complete.json").write_text(json.dumps(clean_json(metadata), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
