"""评估跨模态 SNN 联想记忆网络。

对 8 种 cue 模式分别评估（推理时禁用 target；v11b Decoder 输入由
v_*_from_A、对侧 Key residual 与当前 cue 的同模态 detail state 构成）：
    corrupt_img_only / corrupt_aud_only / corrupt_both
    clean_img_corrupt_aud / corrupt_img_clean_aud
    clean_img_only / clean_aud_only / clean_both

指标：
    分类   accuracy
    图像   MSE / PSNR / SSIM（recovered_img vs clean_img）
    音频   MSE（recovered log-mel vs clean log-mel，[B,n_mels,n_frames]）
    多样性 像素方差 / 样本间 L2（检测是否塌缩成同一张图）
    音频塌缩诊断 rec/target 的 mean/std/max + top-k 能量召回（检测近黑图）

评估协议（--protocol）：
    fixed_mask     论文主对照：固定 seed + 固定 corruption family + 同一套 mask，
                    保证不同版本在完全相同的残缺输入上可比。
    legacy_random  旧随机协议：family 随机、不固定 seed，用于鲁棒性抽查。

可选：--severity_curve 对 corrupt_* 模式扫描 severity，输出退化曲线。
可选：--family_breakdown 按音频腐蚀 family 拆解 audio-only、clean-image assist 与 corrupt-both。
可选：--cross_key sweep 在同一 cue/mask 下比较 correct/zero/wrong-class Key。

用法：
    python -u scripts/evaluate.py --config configs/v11b.yaml --protocol fixed_mask
    python -u scripts/evaluate.py --config configs/v11b.yaml --protocol legacy_random
    python -u scripts/evaluate.py --config configs/v11b.yaml --protocol fixed_mask --family_breakdown
    python -u scripts/evaluate.py --config configs/v11b.yaml --protocol fixed_mask --cross_key sweep
    python -u scripts/evaluate.py --max_batches 20 --severity_curve
"""

import bootstrap  # noqa: F401

import argparse
import csv
import math
import random
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import (fix_console_encoding, log, load_config, set_seed,
                    batch_ssim, batch_psnr, build_cue, select_targets,
                    batch_reconstruction_variance, format_table_row,
                    aud_collapse_stats)
from paths import resolve_from_root, tables_dir
from data.corruption import (AUD_MODES, AUD_FAMILY_GROUPS,
                             AUD_TRAIN_MODES, IMG_TRAIN_MODES)
from data.dataset import build_loaders
from models.network import CrossModalSNN
from models.lif import rate

EVAL_MODES = ["corrupt_img_only", "corrupt_aud_only", "corrupt_both",
              "clean_img_corrupt_aud", "corrupt_img_clean_aud",
              "clean_img_only", "clean_aud_only", "clean_both"]

_MASK_SEED_ALIAS = {
    "clean_img_corrupt_aud": "corrupt_aud_only",
    "corrupt_img_clean_aud": "corrupt_img_only",
}


def _reseed(seed):
    """同时重置 python random 与 torch RNG，使 corruption mask 确定可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)


def _fixed_eval_families(cfg):
    """fixed_mask 协议使用的固定残缺 family（论文主对照）。"""
    ef = cfg["corruption"].get("eval_fixed", {}) or {}
    return ef.get("img_mode", "occlusion"), ef.get("aud_mode", "time_freq_block")


def _as_list(value, fallback):
    if value is None:
        return list(fallback)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _fixed_eval_family_pairs(cfg):
    """fixed_mask 主评估 family 列表；默认 zip image/audio 五 family。"""
    ef = cfg["corruption"].get("eval_fixed", {}) or {}
    img_modes = _as_list(ef.get("img_modes"), [_fixed_eval_families(cfg)[0]])
    aud_modes = _as_list(ef.get("aud_modes"), [_fixed_eval_families(cfg)[1]])
    if len(img_modes) == 1 and len(aud_modes) > 1:
        img_modes = img_modes * len(aud_modes)
    if len(aud_modes) == 1 and len(img_modes) > 1:
        aud_modes = aud_modes * len(img_modes)
    if len(img_modes) != len(aud_modes):
        n = min(len(img_modes), len(aud_modes))
        img_modes, aud_modes = img_modes[:n], aud_modes[:n]
    return list(zip(img_modes, aud_modes))


def _audio_family_group(family):
    for group, families in AUD_FAMILY_GROUPS.items():
        if family in families:
            return group
    return "other"


def _region_error(rec, target, region, power=2):
    region = region.to(device=rec.device, dtype=rec.dtype)
    denom = region.flatten(1).sum(dim=1)
    valid = denom > 0
    if not valid.any():
        return float("nan")
    err = (rec - target).abs() if power == 1 else (rec - target).pow(2)
    per_sample = (err * region).flatten(1).sum(dim=1) / denom.clamp_min(1.0)
    return per_sample[valid].mean().item()


def _region_error_per_sample(rec, target, region, power=2):
    """返回逐样本 region error 与有效 mask，供 paired cross-key 归因。"""
    if region is None:
        return None, None
    region = region.to(device=rec.device, dtype=rec.dtype)
    denom = region.flatten(1).sum(dim=1)
    valid = denom > 0
    err = (rec - target).abs() if power == 1 else (rec - target).pow(2)
    values = (err * region).flatten(1).sum(dim=1) / denom.clamp_min(1.0)
    return values, valid


def _wrong_class_indices(labels):
    """构造一对一的 batch 内异类索引；无法匹配的样本 valid=False。"""
    n = labels.numel()
    labels_cpu = labels.detach().cpu().tolist()
    perm_cpu = list(range(n))
    valid_cpu = [False] * n
    groups = {}
    for idx, label in enumerate(labels_cpu):
        groups.setdefault(label, []).append(idx)
    if len(groups) > 1:
        ordered_groups = sorted(groups.values(), key=len, reverse=True)
        majority = ordered_groups[0]
        others = [idx for group in ordered_groups[1:] for idx in group]
        if len(majority) <= n - len(majority):
            ordered = [idx for group in ordered_groups for idx in group]
            shift = len(majority)
            targets = ordered[shift:] + ordered[:shift]
            for source, target in zip(ordered, targets):
                perm_cpu[source] = target
                valid_cpu[source] = True
        else:
            # 完全异类置换不存在时，最大可用子集为 2 * 非多数类样本数。
            for major_idx, other_idx in zip(majority, others):
                perm_cpu[major_idx] = other_idx
                perm_cpu[other_idx] = major_idx
                valid_cpu[major_idx] = True
                valid_cpu[other_idx] = True
    perm = torch.tensor(perm_cpu, dtype=torch.long, device=labels.device)
    valid = torch.tensor(valid_cpu, dtype=torch.bool, device=labels.device)
    if valid.any() and torch.any(labels[perm[valid]] == labels[valid]):
        raise RuntimeError("wrong-class permutation contains a same-class pair")
    selected = perm[valid]
    if selected.unique().numel() != selected.numel():
        raise RuntimeError("wrong-class permutation reuses a Key index")
    return perm, valid


def _same_class_indices(labels):
    """构造同类不同样本置换；batch 内单例类别 valid=False。"""
    labels_cpu = labels.detach().cpu().tolist()
    groups = {}
    for idx, label in enumerate(labels_cpu):
        groups.setdefault(int(label), []).append(idx)
    perm_cpu = list(range(len(labels_cpu)))
    valid_cpu = [False] * len(labels_cpu)
    for group in groups.values():
        if len(group) < 2:
            continue
        shifted = group[1:] + group[:1]
        for source, target in zip(group, shifted):
            perm_cpu[source] = target
            valid_cpu[source] = True
    perm = torch.tensor(perm_cpu, dtype=torch.long, device=labels.device)
    valid = torch.tensor(valid_cpu, dtype=torch.bool, device=labels.device)
    if valid.any():
        if torch.any(perm[valid] == torch.arange(
                labels.numel(), device=labels.device)[valid]):
            raise RuntimeError("same-class permutation contains an identity pair")
        if torch.any(labels[perm[valid]] != labels[valid]):
            raise RuntimeError("same-class permutation contains a wrong-class pair")
    return perm, valid


def _sum_paired_metric(sums, counts, key, values, valid):
    if values is None or valid is None:
        return
    valid = valid & torch.isfinite(values)
    if not valid.any():
        return
    sums[key] = sums.get(key, 0.0) + values[valid].sum().item()
    counts[key] = counts.get(key, 0) + int(valid.sum().item())


def _paired_cross_metrics(normal_out, zero_out, wrong_out, same_out,
                          tgt_img, tgt_aud, img_mask, aud_mask,
                          wrong_valid, same_valid):
    """同 cue/mask 下计算 normal/zero/wrong/same-class 配对指标。"""
    result = {}

    def add_direction(prefix, normal_final, zero_final, wrong_final, same_final,
                      normal_coarse, zero_coarse, wrong_coarse, same_coarse,
                      target, mask, gate_key, ratio_key, source_key):
        if mask is None or normal_out.get(source_key) is None:
            return
        quartets = {
            "final": (normal_final, zero_final, wrong_final, same_final),
            "coarse": (normal_coarse, zero_coarse, wrong_coarse, same_coarse),
        }
        for stage, (normal_rec, zero_rec, wrong_rec, same_rec) in quartets.items():
            n_err, region_valid = _region_error_per_sample(
                normal_rec, target, mask, power=2)
            z_err, _ = _region_error_per_sample(zero_rec, target, mask, power=2)
            w_err, _ = _region_error_per_sample(wrong_rec, target, mask, power=2)
            s_err, _ = _region_error_per_sample(same_rec, target, mask, power=2)
            result[f"{prefix}_{stage}_correct_gain"] = (
                z_err - n_err, region_valid)
            result[f"{prefix}_{stage}_wrong_damage"] = (
                w_err - n_err, region_valid & wrong_valid)
            result[f"{prefix}_{stage}_same_damage"] = (
                s_err - n_err, region_valid & same_valid)

        gate = normal_out.get(gate_key)
        if gate is not None:
            result[f"{prefix}_gate"] = (
                gate.flatten(), torch.ones_like(wrong_valid))
        ratio = normal_out.get(ratio_key)
        if ratio is not None:
            result[f"{prefix}_ratio"] = (
                ratio.flatten(), torch.ones_like(wrong_valid))

    add_direction(
        "img2aud",
        normal_out["recovered_aud"], zero_out["recovered_aud"],
        wrong_out["recovered_aud"], same_out["recovered_aud"],
        normal_out["recovered_aud_coarse"],
        zero_out["recovered_aud_coarse"],
        wrong_out["recovered_aud_coarse"], same_out["recovered_aud_coarse"],
        tgt_aud, aud_mask, "img_to_aud_cross_gate",
        "img_to_aud_cross_ratio", "key_img")
    add_direction(
        "aud2img",
        torch.sigmoid(normal_out["recovered_img"]),
        torch.sigmoid(zero_out["recovered_img"]),
        torch.sigmoid(wrong_out["recovered_img"]),
        torch.sigmoid(same_out["recovered_img"]),
        torch.sigmoid(normal_out["recovered_img_coarse"]),
        torch.sigmoid(zero_out["recovered_img_coarse"]),
        torch.sigmoid(wrong_out["recovered_img_coarse"]),
        torch.sigmoid(same_out["recovered_img_coarse"]),
        tgt_img, img_mask, "aud_to_img_cross_gate",
        "aud_to_img_cross_ratio", "key_aud")
    return result


def _audio_masked_metrics(rec, target, mask):
    if mask is None:
        return {
            "aud_masked_mse": float("nan"),
            "aud_masked_l1": float("nan"),
            "aud_visible_mse": float("nan"),
            "aud_visible_l1": float("nan"),
        }
    mask = mask.to(device=rec.device, dtype=rec.dtype)
    visible = 1.0 - mask
    return {
        "aud_masked_mse": _region_error(rec, target, mask, power=2),
        "aud_masked_l1": _region_error(rec, target, mask, power=1),
        "aud_visible_mse": _region_error(rec, target, visible, power=2),
        "aud_visible_l1": _region_error(rec, target, visible, power=1),
    }


def _image_masked_metrics(rec_img_prob, target, mask):
    if mask is None:
        return {
            "img_masked_mse": float("nan"),
            "img_masked_l1": float("nan"),
            "img_visible_mse": float("nan"),
            "img_visible_l1": float("nan"),
        }
    mask = mask.to(device=rec_img_prob.device, dtype=rec_img_prob.dtype)
    visible = 1.0 - mask
    return {
        "img_masked_mse": _region_error(rec_img_prob, target, mask, power=2),
        "img_masked_l1": _region_error(rec_img_prob, target, mask, power=1),
        "img_visible_mse": _region_error(rec_img_prob, target, visible, power=2),
        "img_visible_l1": _region_error(rec_img_prob, target, visible, power=1),
    }


def _add_metric(sums, counts, key, value):
    if value is None or not math.isfinite(float(value)):
        return
    sums[key] = sums.get(key, 0.0) + float(value)
    counts[key] = counts.get(key, 0) + 1


def _mean_metric(sums, counts, key):
    count = counts.get(key, 0)
    if count <= 0:
        return float("nan")
    return sums.get(key, 0.0) / count


def _fmt_float(value, digits=4):
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _fmt_na(value, digits=4):
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _log_audio_diag(diag_rows):
    """音频塌缩诊断块：rec/target 的 mean/std/max + top-k 能量召回。

    近黑图（能量塌缩）一眼可辨：rec_std / rec_max 远小于 target，topk 召回偏低。
    单独打印（不混入主表），不影响 plot_eval_summary 解析主表。
    """
    dw = [24, 9, 9, 9, 9, 9, 9, 10]
    da = ["l", "r", "r", "r", "r", "r", "r", "r"]
    hdr = ["cue模式", "rec均值", "rec标准差", "rec最大",
           "tgt均值", "tgt标准差", "tgt最大", "top15%召回"]
    log("-" * sum(dw))
    log("[音频塌缩诊断] recovered_aud vs target_aud（log-mel 能量统计）")
    log("-" * sum(dw))
    log(format_table_row(hdr, dw, da))
    for mode, d in diag_rows:
        if not d:
            continue
        log(format_table_row([
            mode,
            f"{d['rec_mean']:.4f}", f"{d['rec_std']:.4f}", f"{d['rec_max']:.4f}",
            f"{d['tgt_mean']:.4f}", f"{d['tgt_std']:.4f}", f"{d['tgt_max']:.4f}",
            f"{d['topk_recall']*100:.1f}%",
        ], dw, da))


@torch.no_grad()
def eval_mode(model, loader, cfg, mode, device, severity, proto_img, proto_aud,
              max_batches=None, protocol="fixed_mask", mode_idx=0,
              fixed_img_mode_override=None, fixed_aud_mode_override=None,
              cross_key_mode="normal"):
    """按 cue 模式对应的恢复粒度 target 计算指标。

    图像/音频指标均对照 select_targets 选出的 target（区分样本级/类别级）：
        audio-only : 图像 vs 类别代表原型      音频 vs 本样本 clean
        image-only : 图像 vs 本样本 clean       音频 vs 类别代表原型
        双模态（含非对称 clean/corrupt）: 图像/音频均 vs 本样本 clean

    protocol=fixed_mask：每个 batch 用确定性 seed 重置 RNG，并使用固定 family，
        使任意模型在同一套 mask 上评估（masks 与模型无关，可跨版本对比）。
    protocol=legacy_random：沿用配置里的 family（通常 random），不固定 seed。
    """
    model.eval()
    n = 0
    correct = 0
    sum_img_mse = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_aud_mse = 0.0
    image_metric_sums = {}
    image_metric_counts = {}
    audio_metric_sums = {}
    audio_metric_counts = {}
    nb = 0
    all_rec = []
    img_kind = aud_kind = "?"
    diag_sum = {}
    cross_metric_sums = {}
    cross_metric_counts = {}

    base_seed = int(cfg.get("seed", 0))
    fixed_img_mode, fixed_aud_mode = _fixed_eval_families(cfg)
    if fixed_img_mode_override is not None:
        fixed_img_mode = fixed_img_mode_override
    if fixed_aud_mode_override is not None:
        fixed_aud_mode = fixed_aud_mode_override

    iterator = enumerate(loader)
    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    pbar = tqdm(iterator, total=total, desc=f"{protocol}:{mode}", unit="batch",
                file=sys.stdout, ascii=True)
    for bi, (x_img, x_aud, labels) in pbar:
        if max_batches is not None and bi >= max_batches:
            break
        x_img = x_img.to(device)
        x_aud = x_aud.to(device)
        labels = labels.to(device)

        if protocol == "fixed_mask":
            # 与模型无关的确定性 mask：仅依赖 (seed, mode, batch)
            _reseed(base_seed * 100000 + mode_idx * 10000 + bi)
            img_cue, aud_cue, cue_masks = build_cue(
                x_img, x_aud, mode, cfg, severity=severity,
                img_mode=fixed_img_mode, aud_mode=fixed_aud_mode,
                return_masks=True)
        else:
            img_cue, aud_cue, cue_masks = build_cue(
                x_img, x_aud, mode, cfg, severity=severity,
                return_masks=True)
        img_mask = cue_masks.get("img")
        aud_mask = cue_masks.get("aud")

        tgt_img, tgt_aud, img_kind, aud_kind = select_targets(
            mode, x_img, x_aud, proto_img, proto_aud, labels)
        def run_model(**cross_kwargs):
            return model(
                x_img_cue=img_cue, x_aud_cue=aud_cue,
                training_mode=False, phase="readout",
                img_cue_mask=img_mask, aud_cue_mask=aud_mask,
                **cross_kwargs)

        normal_out = run_model()
        out = normal_out
        if cross_key_mode == "zero":
            out = run_model(
                disable_img_to_aud_cross=True,
                disable_aud_to_img_cross=True)
            if not torch.equal(normal_out["index_state"], out["index_state"]):
                raise RuntimeError("cross-key zero intervention changed index_state")
        elif cross_key_mode in ("shuffle_wrong", "same_class", "sweep"):
            wrong_perm, wrong_valid = _wrong_class_indices(labels)
            same_perm, same_valid = _same_class_indices(labels)
            img_rate = (rate(normal_out["key_img"]).detach()
                        if normal_out.get("key_img") is not None else None)
            aud_rate = (rate(normal_out["key_aud"]).detach()
                        if normal_out.get("key_aud") is not None else None)
            wrong_kwargs = {}
            if img_rate is not None:
                wrong_kwargs["cross_key_img_rate_override"] = img_rate[wrong_perm]
            if aud_rate is not None:
                wrong_kwargs["cross_key_aud_rate_override"] = aud_rate[wrong_perm]
            same_kwargs = {}
            if img_rate is not None:
                same_kwargs["cross_key_img_rate_override"] = img_rate[same_perm]
            if aud_rate is not None:
                same_kwargs["cross_key_aud_rate_override"] = aud_rate[same_perm]

            if cross_key_mode == "sweep":
                zero_out = run_model(
                    disable_img_to_aud_cross=True,
                    disable_aud_to_img_cross=True)
                wrong_out = run_model(**wrong_kwargs)
                same_out = run_model(**same_kwargs)
                for label, candidate in (("zero", zero_out),
                                         ("wrong", wrong_out),
                                         ("same", same_out)):
                    if not torch.equal(normal_out["index_state"],
                                       candidate["index_state"]):
                        raise RuntimeError(
                            f"cross-key {label} intervention changed index_state")
                    if not torch.equal(
                            normal_out["logits"].argmax(dim=1),
                            candidate["logits"].argmax(dim=1)):
                        raise RuntimeError(
                            f"cross-key {label} intervention changed ACC path")
                paired = _paired_cross_metrics(
                    normal_out, zero_out, wrong_out, same_out,
                    tgt_img, tgt_aud, img_mask, aud_mask,
                    wrong_valid, same_valid)
                for key, (values, valid) in paired.items():
                    _sum_paired_metric(
                        cross_metric_sums, cross_metric_counts,
                        key, values, valid)
            elif cross_key_mode == "shuffle_wrong":
                out = run_model(**wrong_kwargs)
                if not torch.equal(normal_out["index_state"],
                                   out["index_state"]):
                    raise RuntimeError(
                        "cross-key wrong intervention changed index_state")
            else:
                out = run_model(**same_kwargs)
                if not torch.equal(normal_out["index_state"],
                                   out["index_state"]):
                    raise RuntimeError(
                        "cross-key same-class intervention changed index_state")

        pred = out["logits"].argmax(dim=1)
        correct += (pred == labels).sum().item()
        n += labels.size(0)

        rec_img = torch.sigmoid(out["recovered_img"])
        rec_img_coarse = torch.sigmoid(out["recovered_img_coarse"])
        sum_img_mse += F.mse_loss(rec_img, tgt_img).item()
        sum_psnr += batch_psnr(rec_img, tgt_img).item()
        sum_ssim += batch_ssim(rec_img, tgt_img).item()
        for mk, mv in _image_masked_metrics(rec_img, tgt_img, img_mask).items():
            _add_metric(image_metric_sums, image_metric_counts, mk, mv)
        for mk, mv in _image_masked_metrics(
                rec_img_coarse, tgt_img, img_mask).items():
            key = mk.replace("img_", "img_coarse_")
            _add_metric(image_metric_sums, image_metric_counts, key, mv)

        rec_aud = out["recovered_aud"]
        rec_aud_coarse = out["recovered_aud_coarse"]
        sum_aud_mse += F.mse_loss(rec_aud, tgt_aud).item()   # log-mel [B,M,T]
        _add_metric(
            audio_metric_sums, audio_metric_counts, "aud_ssim",
            batch_ssim(rec_aud.unsqueeze(1), tgt_aud.unsqueeze(1)).item())
        for mk, mv in _audio_masked_metrics(rec_aud, tgt_aud, aud_mask).items():
            _add_metric(audio_metric_sums, audio_metric_counts, mk, mv)
        for mk, mv in _audio_masked_metrics(
                rec_aud_coarse, tgt_aud, aud_mask).items():
            key = mk.replace("aud_", "aud_coarse_")
            _add_metric(audio_metric_sums, audio_metric_counts, key, mv)

        d = aud_collapse_stats(rec_aud, tgt_aud)
        for kk, vv in d.items():
            diag_sum[kk] = diag_sum.get(kk, 0.0) + vv

        all_rec.append(rec_img.cpu())
        nb += 1

    acc = correct / max(n, 1)
    rec_all = torch.cat(all_rec, dim=0) if all_rec else torch.zeros(1, 1, 28, 28)
    pix_var, pair_l2 = batch_reconstruction_variance(rec_all)
    diag = {kk: vv / max(nb, 1) for kk, vv in diag_sum.items()}
    cross_attr = {
        key: _mean_metric(cross_metric_sums, cross_metric_counts, key)
        for key in sorted(cross_metric_sums)
    }
    return {
        "acc": acc,
        "img_mse": sum_img_mse / max(nb, 1),
        "psnr": sum_psnr / max(nb, 1),
        "ssim": sum_ssim / max(nb, 1),
        "img_masked_mse": _mean_metric(image_metric_sums, image_metric_counts,
                                       "img_masked_mse"),
        "img_masked_l1": _mean_metric(image_metric_sums, image_metric_counts,
                                      "img_masked_l1"),
        "img_visible_mse": _mean_metric(image_metric_sums, image_metric_counts,
                                       "img_visible_mse"),
        "img_visible_l1": _mean_metric(image_metric_sums, image_metric_counts,
                                      "img_visible_l1"),
        "img_coarse_masked_mse": _mean_metric(
            image_metric_sums, image_metric_counts, "img_coarse_masked_mse"),
        "img_coarse_visible_mse": _mean_metric(
            image_metric_sums, image_metric_counts, "img_coarse_visible_mse"),
        "aud_mse": sum_aud_mse / max(nb, 1),
        "aud_ssim": _mean_metric(audio_metric_sums, audio_metric_counts,
                                 "aud_ssim"),
        "aud_masked_mse": _mean_metric(audio_metric_sums, audio_metric_counts,
                                       "aud_masked_mse"),
        "aud_masked_l1": _mean_metric(audio_metric_sums, audio_metric_counts,
                                      "aud_masked_l1"),
        "aud_visible_mse": _mean_metric(audio_metric_sums, audio_metric_counts,
                                       "aud_visible_mse"),
        "aud_visible_l1": _mean_metric(audio_metric_sums, audio_metric_counts,
                                      "aud_visible_l1"),
        "aud_coarse_masked_mse": _mean_metric(
            audio_metric_sums, audio_metric_counts, "aud_coarse_masked_mse"),
        "aud_coarse_visible_mse": _mean_metric(
            audio_metric_sums, audio_metric_counts, "aud_coarse_visible_mse"),
        "pix_var": pix_var,
        "pair_l2": pair_l2,
        "img_kind": img_kind,
        "aud_kind": aud_kind,
        "diag": diag,
        "cross_attr": cross_attr,
    }


@torch.no_grad()
def eval_audio_family_breakdown(model, loader, cfg, device, severity,
                                proto_img, proto_aud, max_batches=None):
    """固定 seed/mask，逐个音频 corruption family 评估随机协议的薄弱环节。"""
    rows = []
    fixed_img_mode, _ = _fixed_eval_families(cfg)
    modes = ["corrupt_aud_only", "clean_img_corrupt_aud", "corrupt_both"]
    ef = cfg["corruption"].get("eval_fixed", {}) or {}
    aud_families = _as_list(ef.get("aud_modes"), AUD_TRAIN_MODES)
    for mode_idx, mode in enumerate(modes):
        for fam_idx, aud_family in enumerate(aud_families):
            seed_mode = EVAL_MODES.index(_MASK_SEED_ALIAS.get(mode, mode))
            r = eval_mode(
                model, loader, cfg, mode, device, severity,
                proto_img, proto_aud, max_batches=max_batches,
                protocol="fixed_mask", mode_idx=100 + seed_mode * 10 + fam_idx,
                fixed_img_mode_override=fixed_img_mode,
                fixed_aud_mode_override=aud_family)
            d = r["diag"]
            rows.append({
                "cue_mode": mode,
                "aud_family": aud_family,
                "family_group": _audio_family_group(aud_family),
                "acc": r["acc"],
                "img_mse": r["img_mse"],
                "psnr": r["psnr"],
                "img_ssim": r["ssim"],
                "aud_mse": r["aud_mse"],
                "aud_ssim": r["aud_ssim"],
                "aud_masked_mse": r["aud_masked_mse"],
                "aud_masked_l1": r["aud_masked_l1"],
                "aud_visible_mse": r["aud_visible_mse"],
                "aud_visible_l1": r["aud_visible_l1"],
                "rec_std": d.get("rec_std", 0.0),
                "tgt_std": d.get("tgt_std", 0.0),
                "top15_recall": d.get("topk_recall", 0.0),
            })

    out_dir = tables_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audio_family_breakdown_fixed.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bw = [24, 18, 24, 8, 9, 9, 11, 10]
    ba = ["l", "l", "l", "r", "r", "r", "r", "r"]
    log("=" * sum(bw))
    log(f"[音频 family breakdown] fixed seed/mask -> {out_path}")
    log(format_table_row(["cue模式", "audio family", "group", "acc",
                          "audMSE", "audSSIM", "maskedMSE", "top15%"],
                         bw, ba))
    for r in rows:
        log(format_table_row([
            r["cue_mode"], r["aud_family"], r["family_group"],
            f"{r['acc']*100:.1f}%", _fmt_float(r["aud_mse"]),
            _fmt_float(r["aud_ssim"], digits=3),
            _fmt_float(r["aud_masked_mse"]),
            f"{r['top15_recall']*100:.1f}%",
        ], bw, ba))


def main():
    fix_console_encoding()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v11b.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--severity", type=float, default=0.5)
    ap.add_argument("--severity_curve", action="store_true")
    ap.add_argument("--protocol", default="fixed_mask",
                    choices=["fixed_mask", "legacy_random"],
                    help="fixed_mask=论文主对照(固定mask) | legacy_random=旧随机协议")
    ap.add_argument("--family_breakdown", action="store_true",
                    help="按音频 family 评估 audio-only/clean-image assist/corrupt-both")
    ap.add_argument(
        "--cross_key", default="normal",
        choices=["normal", "zero", "shuffle_wrong", "same_class", "sweep"],
        help=("Decoder cross-Key 条件干预；sweep 同 cue/mask 对比 "
              "normal/zero/wrong/same-class"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    # 固定全局 RNG（fixed_mask 协议下逐 batch 还会再确定性重置）
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda" if (cfg["device"] == "cuda"
                          and torch.cuda.is_available()) else "cpu")
    ckpt_path = str(resolve_from_root(args.ckpt or cfg["train"]["ckpt_path"]))

    log(f"[评估] 设备: {device}  加载 checkpoint: {ckpt_path}")
    model = CrossModalSNN(cfg).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    except FileNotFoundError as e:
        raise SystemExit(f"[错误] 未找到 checkpoint: {ckpt_path}") from e
    except RuntimeError as e:
        raise SystemExit(
            f"[错误] checkpoint 结构不匹配，禁止使用随机权重继续评估。\n{e}") from e

    _, test_loader = build_loaders(cfg)
    proto_img = test_loader.dataset.prototype_img.to(device)
    proto_aud = test_loader.dataset.prototype_aud.to(device)

    eval_w = [24, 7, 9, 8, 7, 10, 9, 8, 10, 10, 9, 16]
    eval_a = ["l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"]
    eval_hdr = ["cue模式", "acc", "imgMSE", "PSNR", "SSIM", "imgMaskMSE",
                "audMSE", "audSSIM", "audMaskMSE", "像素方差", "样本L2",
                "tgt(img/aud)"]
    family_pairs = (_fixed_eval_family_pairs(cfg)
                    if args.protocol == "fixed_mask"
                    else [_fixed_eval_families(cfg)])
    log("=" * sum(eval_w))
    log(f"[评估] 8 种 cue 模式  (corrupt severity={args.severity})  "
        f"协议={args.protocol}  cross_key={args.cross_key}")
    if args.protocol == "fixed_mask":
        fam_text = ", ".join(
            f"{i + 1}:{im}/{am}" for i, (im, am) in enumerate(family_pairs))
        log(f"  固定残缺 family pairs: {fam_text}")
        log(
            f"seed={int(cfg.get('seed', 0))}（masks 与模型无关，可跨版本对比）")
    else:
        log("  随机残缺：family 随机、不固定 seed（鲁棒性抽查，不可跨版本严格对比）")
    log("  指标按恢复粒度对照 target：img/aud 列后缀 (smp)=样本级  (cat)=类别代表原型")
    for fam_idx, (fixed_img_mode, fixed_aud_mode) in enumerate(family_pairs):
        log("=" * sum(eval_w))
        log(f"[评估 family {fam_idx + 1}/{len(family_pairs)}] "
            f"img={fixed_img_mode}  aud={fixed_aud_mode}")
        log(format_table_row(eval_hdr, eval_w, eval_a))
        diag_rows = []
        attr_rows = []
        cross_rows = []
        for mi, mode in enumerate(EVAL_MODES):
            seed_mode = EVAL_MODES.index(_MASK_SEED_ALIAS.get(mode, mode))
            r = eval_mode(
                model, test_loader, cfg, mode, device,
                args.severity, proto_img, proto_aud, args.max_batches,
                protocol=args.protocol, mode_idx=fam_idx * 100 + seed_mode,
                fixed_img_mode_override=fixed_img_mode,
                fixed_aud_mode_override=fixed_aud_mode,
                cross_key_mode=args.cross_key)
            tgt = f"{r['img_kind']}/{r['aud_kind']}"
            log(format_table_row([
                mode, f"{r['acc']*100:.1f}%",
                f"{r['img_mse']:.4f}", f"{r['psnr']:.2f}", f"{r['ssim']:.3f}",
                _fmt_float(r["img_masked_mse"]),
                f"{r['aud_mse']:.4f}",
                _fmt_float(r["aud_ssim"], digits=3),
                _fmt_float(r["aud_masked_mse"]),
                f"{r['pix_var']:.4f}", f"{r['pair_l2']:.4f}",
                tgt,
            ], eval_w, eval_a))
            diag_rows.append((mode, r["diag"]))
            attr_rows.append((mode, r))
            cross_rows.append((mode, r.get("cross_attr", {})))

        _log_audio_diag(diag_rows)

        attr_w = [24, 10, 10, 10, 10, 10, 10, 10, 10]
        attr_a = ["l"] + ["r"] * 8
        log("=" * sum(attr_w))
        log("[归因] coarse/final masked/visible MSE（主看 mask coarse->final；"
            "final visible≈0 是 paste-back 机制，不代表可见区学习）")
        log(format_table_row(
            ["cue模式", "imgCmask", "imgFmask", "imgCvis", "imgFvis",
             "audCmask", "audFmask", "audCvis", "audFvis"],
            attr_w, attr_a))
        for mode, r in attr_rows:
            log(format_table_row([
                mode,
                _fmt_float(r["img_coarse_masked_mse"]),
                _fmt_float(r["img_masked_mse"]),
                _fmt_float(r["img_coarse_visible_mse"]),
                _fmt_float(r["img_visible_mse"]),
                _fmt_float(r["aud_coarse_masked_mse"]),
                _fmt_float(r["aud_masked_mse"]),
                _fmt_float(r["aud_coarse_visible_mse"]),
                _fmt_float(r["aud_visible_mse"]),
            ], attr_w, attr_a))

        if args.cross_key == "sweep":
            cross_w = [24, 12, 9, 9, 11, 11, 11, 11, 11, 11]
            cross_a = ["l", "l"] + ["r"] * 8
            log("=" * sum(cross_w))
            log("[Cross-Key归因] 同 cue/mask 的 normal/zero/wrong/same-class；"
                "gain=zero-normal，damage=替换条件-normal（masked MSE）")
            log(format_table_row(
                ["cue模式", "方向", "gate", "res/V", "Cgain", "Fgain",
                 "Cwrong", "Fwrong", "Csame", "Fsame"], cross_w, cross_a))
            for mode, values in cross_rows:
                for prefix, direction in (("img2aud", "img->aud"),
                                          ("aud2img", "aud->img")):
                    log(format_table_row([
                        mode, direction,
                        _fmt_na(values.get(f"{prefix}_gate")),
                        _fmt_na(values.get(f"{prefix}_ratio")),
                        _fmt_na(values.get(
                            f"{prefix}_coarse_correct_gain")),
                        _fmt_na(values.get(
                            f"{prefix}_final_correct_gain")),
                        _fmt_na(values.get(
                            f"{prefix}_coarse_wrong_damage")),
                        _fmt_na(values.get(
                            f"{prefix}_final_wrong_damage")),
                        _fmt_na(values.get(
                            f"{prefix}_coarse_same_damage")),
                        _fmt_na(values.get(
                            f"{prefix}_final_same_damage")),
                    ], cross_w, cross_a))

    if args.family_breakdown:
        eval_audio_family_breakdown(model, test_loader, cfg, device,
                                    args.severity, proto_img, proto_aud,
                                    args.max_batches)

    if args.severity_curve:
        log("=" * 78)
        log("[评估] 严重度曲线（corrupt_aud_only -> 类别代表图像恢复 & 分类）")
        log(f"{'severity':>9}{'acc':>8}{'imgMSE':>9}{'PSNR':>8}{'SSIM':>7}")
        for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            r = eval_mode(model, test_loader, cfg, "corrupt_aud_only",
                          device, s, proto_img, proto_aud, args.max_batches)
            log(f"{s:>9.1f}{r['acc']*100:>7.1f}%{r['img_mse']:>9.4f}"
                f"{r['psnr']:>8.2f}{r['ssim']:>7.3f}")

    log("[评估] 完成。")


if __name__ == "__main__":
    main()
