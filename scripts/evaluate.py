"""评估跨模态 SNN 联想记忆网络。

对 8 种 cue 模式分别评估（推理时禁用 target；v11g 保留 Value + own detail，
对侧 Key 只在缺失区域调制 decoder 内部特征）：
    corrupt_img_only / corrupt_aud_only / corrupt_both
    clean_img_corrupt_aud / corrupt_img_clean_aud
    clean_img_only / clean_aud_only / clean_both

指标：
    分类   accuracy
    图像   MSE / PSNR / SSIM（recovered_img vs clean_img）
    音频   MSE（recovered log-mel vs clean log-mel，[B,n_mels,n_frames]）
    多样性 像素方差 / 样本间 L2（检测是否塌缩成同一张图）
    音频塌缩诊断 rec/target 的 mean/std/max + top-k 能量召回（检测近黑图）
    v11g 恢复内容类别一致性（冻结原模型代理，不是独立识别器）

评估协议（--protocol）：
    fixed_mask     论文主对照：固定 seed + 固定 corruption family + 同一套 mask，
                    保证不同版本在完全相同的残缺输入上可比。
    legacy_random  v11g 按独立 random_seed 抽样 family/mask，用于鲁棒性抽查。

可选：--severity_curve 对 corrupt_* 模式扫描 severity，输出退化曲线。
可选：--family_breakdown 按音频腐蚀 family 拆解 audio-only、clean-image assist 与 corrupt-both。
可选：--cross_key sweep 在同一 cue/mask 下比较 correct/zero/wrong-class Key。
可选：--cross_detail sweep 比较 correct/zero/same-class wrong-pair Detail。

用法：
    python -u scripts/evaluate.py --config configs/v11g.yaml --protocol fixed_mask
    python -u scripts/evaluate.py --config configs/v11g.yaml --protocol legacy_random
    python -u scripts/evaluate.py --config configs/v11g.yaml --protocol fixed_mask --family_breakdown
    python -u scripts/evaluate.py --config configs/v11g.yaml --protocol fixed_mask --cross_key sweep
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
                    aud_collapse_stats, unpack_paired_batch)
from paths import resolve_from_root, tables_dir
from data.corruption import (AUD_MODES, AUD_FAMILY_GROUPS,
                             AUD_TRAIN_MODES, IMG_TRAIN_MODES)
from data.dataset import build_loaders
from models.network import CrossModalSNN
from models.lif import rate
from models.frozen_base import load_evaluation_checkpoint, verify_audio_normalization

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


def _same_class_indices(labels, pair_ids=None):
    """构造同类不同 pair_id 置换；batch 内单例类别 valid=False。"""
    labels_cpu = labels.detach().cpu().tolist()
    pair_cpu = (pair_ids.detach().cpu().tolist()
                if pair_ids is not None else list(range(len(labels_cpu))))
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
            if pair_cpu[source] == pair_cpu[target]:
                continue
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
        if pair_ids is not None and torch.any(
                pair_ids[perm[valid]] == pair_ids[valid]):
            raise RuntimeError("same-class permutation reused the same pair_id")
    return perm, valid


def _global_pair_retrieval(img_embed, aud_embed, labels, pair_ids,
                           chunk_size=512):
    """Exact-pair Recall@1 among same-class, different-source candidates."""
    if img_embed is None or aud_embed is None or len(labels) < 2:
        return float("nan"), float("nan")
    img_embed = F.normalize(img_embed.float(), dim=1)
    aud_embed = F.normalize(aud_embed.float(), dim=1)
    labels = labels.view(-1).to(img_embed.device)
    pair_ids = pair_ids.view(-1).to(img_embed.device)
    n = labels.numel()

    def direction(query, gallery):
        correct = total = 0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            scores = query[start:end] @ gallery.t()
            candidate = labels[start:end, None].eq(labels[None, :])
            same_source = pair_ids[start:end, None].eq(pair_ids[None, :])
            target = torch.arange(start, end, device=scores.device)
            diagonal = torch.zeros_like(candidate)
            diagonal[torch.arange(end - start), target] = True
            candidate = candidate & (~same_source | diagonal)
            valid = candidate.sum(dim=1) > 1
            if valid.any():
                predicted = scores.masked_fill(~candidate, -1e9).argmax(dim=1)
                correct += predicted[valid].eq(target[valid]).sum().item()
                total += valid.sum().item()
        return correct / total if total else float("nan")

    return direction(img_embed, aud_embed), direction(aud_embed, img_embed)


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

    def add_direction(prefix, normal_rec, zero_rec, wrong_rec, same_rec,
                      target, mask, gate_key, ratio_key, source_key):
        if mask is None or normal_out.get(source_key) is None:
            return
        n_err, region_valid = _region_error_per_sample(
            normal_rec, target, mask, power=2)
        z_err, _ = _region_error_per_sample(zero_rec, target, mask, power=2)
        w_err, _ = _region_error_per_sample(wrong_rec, target, mask, power=2)
        s_err, _ = _region_error_per_sample(same_rec, target, mask, power=2)
        result[f"{prefix}_correct_gain"] = (
            z_err - n_err, region_valid)
        result[f"{prefix}_wrong_damage"] = (
            w_err - n_err, region_valid & wrong_valid)
        result[f"{prefix}_same_damage"] = (
            s_err - n_err, region_valid & same_valid)
        for name, error, valid in (
                ("normal", n_err, region_valid), ("zero", z_err, region_valid),
                ("wrong", w_err, region_valid & wrong_valid),
                ("same_class", s_err, region_valid & same_valid)):
            result[f"{prefix}_{name}_mse"] = (error, valid)
        result[f"{prefix}_win_zero"] = ((n_err < z_err - 1e-8).float(), region_valid)
        result[f"{prefix}_win_wrong"] = (
            (n_err < w_err - 1e-8).float(), region_valid & wrong_valid)
        result[f"{prefix}_win_both"] = (
            ((n_err < z_err - 1e-8) & (n_err < w_err - 1e-8)).float(),
            region_valid & wrong_valid)

        gate = normal_out.get(gate_key)
        if gate is not None:
            result[f"{prefix}_gate"] = (
                gate.flatten(), region_valid)
        ratio = normal_out.get(ratio_key)
        if ratio is not None:
            result[f"{prefix}_ratio"] = (
                ratio.flatten(), region_valid)

    add_direction(
        "img2aud",
        normal_out["recovered_aud"], zero_out["recovered_aud"],
        wrong_out["recovered_aud"], same_out["recovered_aud"],
        tgt_aud, aud_mask, "img_to_aud_cross_gate",
        "img_to_aud_cross_ratio", "key_img")
    add_direction(
        "aud2img",
        torch.sigmoid(normal_out["recovered_img"]),
        torch.sigmoid(zero_out["recovered_img"]),
        torch.sigmoid(wrong_out["recovered_img"]),
        torch.sigmoid(same_out["recovered_img"]),
        tgt_img, img_mask, "aud_to_img_cross_gate",
        "aud_to_img_cross_ratio", "key_aud")
    return result


def _paired_detail_metrics(normal_out, zero_out, same_out,
                           tgt_img, tgt_aud, img_mask, aud_mask, same_valid):
    """Cross-Detail 正确/关闭/同类错配的逐样本归因指标。"""
    result = {}

    def add_direction(prefix, normal_rec, zero_rec, same_rec, target, mask,
                      gate_key, ratio_key, source_key):
        if mask is None or normal_out.get(source_key) is None:
            return
        n_err, region_valid = _region_error_per_sample(
            normal_rec, target, mask, power=2)
        z_err, _ = _region_error_per_sample(zero_rec, target, mask, power=2)
        s_err, _ = _region_error_per_sample(same_rec, target, mask, power=2)
        result[f"detail_{prefix}_correct_gain"] = (
            z_err - n_err, region_valid)
        result[f"detail_{prefix}_same_damage"] = (
            s_err - n_err, region_valid & same_valid)
        gate = normal_out.get(gate_key)
        if gate is not None:
            result[f"detail_{prefix}_gate"] = (
                gate.flatten(1).mean(dim=1), torch.ones_like(same_valid))
        ratio = normal_out.get(ratio_key)
        if ratio is not None:
            result[f"detail_{prefix}_ratio"] = (
                ratio.flatten(), torch.ones_like(same_valid))

    add_direction(
        "img2aud", normal_out["recovered_aud"], zero_out["recovered_aud"],
        same_out["recovered_aud"], tgt_aud, aud_mask,
        "img_to_aud_detail_gate", "img_to_aud_detail_ratio",
        "img_cross_detail_state")
    add_direction(
        "aud2img", torch.sigmoid(normal_out["recovered_img"]),
        torch.sigmoid(zero_out["recovered_img"]),
        torch.sigmoid(same_out["recovered_img"]), tgt_img, img_mask,
        "aud_to_img_detail_gate", "aud_to_img_detail_ratio",
        "aud_cross_detail_state")
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


def _add_metric(sums, counts, key, value, weight=1):
    if value is None or not math.isfinite(float(value)):
        return
    sums[key] = sums.get(key, 0.0) + float(value) * int(weight)
    counts[key] = counts.get(key, 0) + int(weight)


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
              cross_key_mode="normal", cross_detail_mode="normal"):
    """按 cue 模式对应的恢复粒度 target 计算指标。

    图像/音频指标均对照 select_targets：v11g 缺失模态使用 train medoid，
    存在的模态使用 clean sample。历史真实配对数据接口仅保留兼容。

    protocol=fixed_mask：每个 batch 用确定性 seed 重置 RNG，并使用固定 family，
        使任意模型在同一套 mask 上评估（masks 与模型无关，可跨版本对比）。
    protocol=legacy_random：v11g 用独立 seed 随机抽样 family/mask。
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
    all_rec = []
    img_kind = aud_kind = "?"
    diag_sum = {}
    cross_metric_sums = {}
    cross_metric_counts = {}

    def record_recovery_content(candidate, labels, variant, valid=None):
        if not cfg.get("eval", {}).get("recovery_classification", False):
            return
        if valid is None:
            valid = torch.ones_like(labels, dtype=torch.bool)
        for modality in ("img", "aud"):
            kwargs = ({"image": torch.sigmoid(candidate["recovered_img"])}
                      if modality == "img" else {"audio": candidate["recovered_aud"]})
            logits = model.classify_recovered(**kwargs)
            values = (logits.argmax(1) == labels).float()
            _sum_paired_metric(cross_metric_sums, cross_metric_counts,
                               f"content_{modality}_{variant}_acc", values, valid)
    all_img_pair = []
    all_aud_pair = []
    all_pair_labels = []
    all_pair_ids = []

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
    for bi, batch in pbar:
        if max_batches is not None and bi >= max_batches:
            break
        x_img, x_aud, labels, pair_ids = unpack_paired_batch(batch)
        x_img = x_img.to(device)
        x_aud = x_aud.to(device)
        labels = labels.to(device)
        if pair_ids is not None:
            pair_ids = pair_ids.to(device)

        if protocol == "fixed_mask":
            # 与模型无关的确定性 mask：仅依赖 (seed, mode, batch)
            _reseed(base_seed * 100000 + mode_idx * 10000 + bi)
            img_cue, aud_cue, cue_masks = build_cue(
                x_img, x_aud, mode, cfg, severity=severity,
                img_mode=fixed_img_mode, aud_mode=fixed_aud_mode,
                return_masks=True)
        else:
            # v11g random protocol samples families, independently of model work/RNG.
            random_seed = cfg.get("eval", {}).get("random_seed")
            if random_seed is not None:
                _reseed(int(random_seed) * 100000 + mode_idx * 10000 + bi)
                random_img = random.choice(cfg["corruption"]["eval_fixed"]["img_modes"])
                random_aud = random.choice(cfg["corruption"]["eval_fixed"]["aud_modes"])
            else:
                random_img = random_aud = None
            img_cue, aud_cue, cue_masks = build_cue(
                x_img, x_aud, mode, cfg, severity=severity,
                img_mode=random_img, aud_mode=random_aud,
                return_masks=True)
        img_mask = cue_masks.get("img")
        aud_mask = cue_masks.get("aud")

        tgt_img, tgt_aud, img_kind, aud_kind = select_targets(
            mode, x_img, x_aud, proto_img, proto_aud, labels,
            paired_missing_targets=bool(
                cfg.get("data", {}).get("pairing", {}).get(
                    "sample_targets_for_missing", False)),
            paired_target_kind=(
                "paired-sample" if cfg.get("data", {}).get("dataset")
                == "paired_manifest" else "sample"))
        # A completely absent modality is a 100% missing region, not "no mask".
        # Keep the model input mask unchanged; this mask is for metrics only.
        eval_img_mask = (
            torch.ones_like(tgt_img) if img_cue is None else img_mask)
        eval_aud_mask = (
            torch.ones_like(tgt_aud) if aud_cue is None else aud_mask)

        def run_model(**cross_kwargs):
            return model(
                x_img_cue=img_cue, x_aud_cue=aud_cue,
                training_mode=False, phase="readout",
                img_cue_mask=img_mask, aud_cue_mask=aud_mask,
                **cross_kwargs)

        normal_out = run_model()
        if (normal_out.get("img_pair_embedding") is not None
                and normal_out.get("aud_pair_embedding") is not None
                and pair_ids is not None):
            all_img_pair.append(normal_out["img_pair_embedding"].cpu())
            all_aud_pair.append(normal_out["aud_pair_embedding"].cpu())
            all_pair_labels.append(labels.cpu())
            all_pair_ids.append(pair_ids.cpu())
        out = normal_out
        if cross_key_mode == "zero":
            out = run_model(
                disable_img_to_aud_cross=True,
                disable_aud_to_img_cross=True)
            if not torch.equal(normal_out["index_state"], out["index_state"]):
                raise RuntimeError("cross-key zero intervention changed index_state")
        elif cross_key_mode in ("shuffle_wrong", "same_class", "sweep"):
            wrong_perm, wrong_valid = _wrong_class_indices(labels)
            same_perm, same_valid = _same_class_indices(labels, pair_ids)
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
                    tgt_img, tgt_aud, eval_img_mask, eval_aud_mask,
                    wrong_valid, same_valid)
                record_recovery_content(zero_out, labels, "zero")
                record_recovery_content(wrong_out, labels, "wrong", wrong_valid)
                record_recovery_content(same_out, labels, "same_class", same_valid)
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

        record_recovery_content(out, labels,
                                "normal" if cross_key_mode == "sweep" else cross_key_mode)
        if cross_detail_mode != "normal":
            if cross_key_mode != "normal":
                raise ValueError(
                    "cross_key and cross_detail interventions cannot be swept "
                    "in the same evaluation run")
            same_perm, same_valid = _same_class_indices(labels, pair_ids)
            zero_detail_out = run_model(
                disable_img_to_aud_detail=True,
                disable_aud_to_img_detail=True)
            same_detail_kwargs = {}
            img_detail = normal_out.get("img_cross_detail_state")
            aud_detail = normal_out.get("aud_cross_detail_state")
            if img_detail is not None:
                same_detail_kwargs["cross_detail_img_rate_override"] = (
                    img_detail[same_perm])
            if aud_detail is not None:
                same_detail_kwargs["cross_detail_aud_rate_override"] = (
                    aud_detail[same_perm])
            same_detail_out = run_model(**same_detail_kwargs)
            for candidate in (zero_detail_out, same_detail_out):
                if not torch.equal(normal_out["index_state"],
                                   candidate["index_state"]):
                    raise RuntimeError(
                        "cross-detail intervention changed index_state")
            if cross_detail_mode == "zero":
                out = zero_detail_out
            elif cross_detail_mode == "same_class":
                out = same_detail_out
            elif cross_detail_mode == "sweep":
                paired = _paired_detail_metrics(
                    normal_out, zero_detail_out, same_detail_out,
                    tgt_img, tgt_aud, eval_img_mask, eval_aud_mask,
                    same_valid)
                for key, (values, valid) in paired.items():
                    _sum_paired_metric(
                        cross_metric_sums, cross_metric_counts,
                        key, values, valid)

        pred = out["logits"].argmax(dim=1)
        correct += (pred == labels).sum().item()
        batch_size = labels.size(0)
        n += batch_size

        rec_img = torch.sigmoid(out["recovered_img"])
        rec_img_coarse = torch.sigmoid(out["recovered_img_coarse"])
        sum_img_mse += F.mse_loss(rec_img, tgt_img).item() * batch_size
        sum_psnr += batch_psnr(rec_img, tgt_img).item() * batch_size
        sum_ssim += batch_ssim(rec_img, tgt_img).item() * batch_size
        for mk, mv in _image_masked_metrics(
                rec_img, tgt_img, eval_img_mask).items():
            _add_metric(
                image_metric_sums, image_metric_counts, mk, mv, batch_size)
        for mk, mv in _image_masked_metrics(
                rec_img_coarse, tgt_img, eval_img_mask).items():
            key = mk.replace("img_", "img_coarse_")
            _add_metric(
                image_metric_sums, image_metric_counts, key, mv, batch_size)

        rec_aud = out["recovered_aud"]
        sum_aud_mse += (
            F.mse_loss(rec_aud, tgt_aud).item() * batch_size
        )  # log-mel [B,M,T]
        _add_metric(
            audio_metric_sums, audio_metric_counts, "aud_ssim",
            batch_ssim(rec_aud.unsqueeze(1), tgt_aud.unsqueeze(1)).item(),
            batch_size)
        for mk, mv in _audio_masked_metrics(
                rec_aud, tgt_aud, eval_aud_mask).items():
            _add_metric(
                audio_metric_sums, audio_metric_counts, mk, mv, batch_size)

        d = aud_collapse_stats(rec_aud, tgt_aud)
        for kk, vv in d.items():
            diag_sum[kk] = diag_sum.get(kk, 0.0) + vv * batch_size

        all_rec.append(rec_img.cpu())

    acc = correct / max(n, 1)
    rec_all = torch.cat(all_rec, dim=0) if all_rec else torch.zeros(1, 1, 28, 28)
    pix_var, pair_l2 = batch_reconstruction_variance(rec_all)
    diag = {kk: vv / max(n, 1) for kk, vv in diag_sum.items()}
    cross_attr = {
        key: _mean_metric(cross_metric_sums, cross_metric_counts, key)
        for key in sorted(cross_metric_sums)
    }
    pair_i2a_r1 = pair_a2i_r1 = float("nan")
    if all_img_pair:
        pair_i2a_r1, pair_a2i_r1 = _global_pair_retrieval(
            torch.cat(all_img_pair), torch.cat(all_aud_pair),
            torch.cat(all_pair_labels), torch.cat(all_pair_ids))
    return {
        "acc": acc,
        "img_mse": sum_img_mse / max(n, 1),
        "psnr": sum_psnr / max(n, 1),
        "ssim": sum_ssim / max(n, 1),
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
        "aud_mse": sum_aud_mse / max(n, 1),
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
        "pix_var": pix_var,
        "pair_l2": pair_l2,
        "img_kind": img_kind,
        "aud_kind": aud_kind,
        "diag": diag,
        "cross_attr": cross_attr,
        "cross_counts": cross_metric_counts,
        "metric_counts": {**image_metric_counts, **audio_metric_counts, **cross_metric_counts},
        "n": n,
        "pair_i2a_r1": pair_i2a_r1,
        "pair_a2i_r1": pair_a2i_r1,
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
    ap.add_argument("--config", default="configs/v11g.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--random_seed", type=int, default=None)
    ap.add_argument("--severity", type=float, default=0.4)
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
    ap.add_argument(
        "--cross_detail", default="normal",
        choices=["normal", "zero", "same_class", "sweep"],
        help=("Decoder Cross-Detail 实例条件干预；sweep 比较 "
              "correct/zero/same-class-wrong-pair"))
    args = ap.parse_args()
    if args.cross_key != "normal" and args.cross_detail != "normal":
        ap.error("--cross_key 与 --cross_detail 不能在同一次运行中同时干预")

    cfg = load_config(args.config)
    if args.random_seed is not None:
        cfg.setdefault("eval", {})["random_seed"] = args.random_seed
    # 固定全局 RNG（fixed_mask 协议下逐 batch 还会再确定性重置）
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda" if (cfg["device"] == "cuda"
                          and torch.cuda.is_available()) else "cpu")
    ckpt_path = str(resolve_from_root(
        args.ckpt or cfg["train"].get(
            "eval_ckpt_path", cfg["train"]["ckpt_path"])))

    log(f"[评估] 设备: {device}  加载 checkpoint: {ckpt_path}")
    model = CrossModalSNN(cfg).to(device)
    try:
        load_evaluation_checkpoint(model, ckpt_path, device)
    except FileNotFoundError as e:
        raise SystemExit(f"[错误] 未找到 checkpoint: {ckpt_path}") from e
    except RuntimeError as e:
        raise SystemExit(
            f"[错误] checkpoint 结构不匹配，禁止使用随机权重继续评估。\n{e}") from e

    _, test_loader = build_loaders(cfg, train_required=False)
    verify_audio_normalization(model, cfg)
    proto_img = test_loader.dataset.prototype_img.to(device)
    proto_aud = test_loader.dataset.prototype_aud.to(device)

    eval_w = [24, 7, 9, 8, 7, 10, 9, 8, 10, 10, 9, 17, 16]
    eval_a = ["l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"]
    eval_hdr = ["cue模式", "acc", "imgMSE", "PSNR", "SSIM", "imgMaskMSE",
                "audMSE", "audSSIM", "audMaskMSE", "像素方差", "样本L2",
                "pairR1(i2a/a2i)", "tgt(img/aud)"]
    family_pairs = (_fixed_eval_family_pairs(cfg)
                    if args.protocol == "fixed_mask"
                    else [_fixed_eval_families(cfg)])
    log("=" * sum(eval_w))
    log(f"[评估] 8 种 cue 模式  (corrupt severity={args.severity})  "
        f"协议={args.protocol}  cross_key={args.cross_key} "
        f"cross_detail={args.cross_detail}")
    if args.protocol == "fixed_mask":
        fam_text = ", ".join(
            f"{i + 1}:{im}/{am}" for i, (im, am) in enumerate(family_pairs))
        log(f"  固定残缺 family pairs: {fam_text}")
        log(
            f"seed={int(cfg.get('seed', 0))}（masks 与模型无关，可跨版本对比）")
    else:
        log(f"  随机 family/mask 抽样：seed={cfg.get('eval', {}).get('random_seed')}；"
            "相同抽样 seed 可复现，不是固定单一 family 协议。")
    if cfg.get("data", {}).get("dataset") == "paired_manifest":
        log("  target：所有 cue 模式均为同一真实 source_id 的 paired-sample；无 category target")
    else:
        log("  target：img/aud 后缀 sample=样本级，category=类别代表原型")
    csv_path = tables_dir(cfg) / (
        f"eval_{args.protocol}_sev{args.severity:g}_key_{args.cross_key}"
        f"_detail_{args.cross_detail}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = ["protocol", "severity", "img_family", "aud_family", "cue_mode",
                  "img_target", "aud_target", "metric", "value", "n"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(csv_fields)
    if cfg.get("eval", {}).get("recovery_classification", False):
        log("[内容分类] 恢复内容经冻结原模型单模态再分类：内部一致性代理，非独立识别器。")
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
                cross_key_mode=args.cross_key,
                cross_detail_mode=args.cross_detail)
            tgt = f"{r['img_kind']}/{r['aud_kind']}"
            log(format_table_row([
                mode, f"{r['acc']*100:.1f}%",
                f"{r['img_mse']:.4f}", f"{r['psnr']:.2f}", f"{r['ssim']:.3f}",
                _fmt_float(r["img_masked_mse"]),
                f"{r['aud_mse']:.4f}",
                _fmt_float(r["aud_ssim"], digits=3),
                _fmt_float(r["aud_masked_mse"]),
                f"{r['pix_var']:.4f}", f"{r['pair_l2']:.4f}",
                (f"{r['pair_i2a_r1']*100:.1f}%/{r['pair_a2i_r1']*100:.1f}%"
                 if math.isfinite(r["pair_i2a_r1"]) else "n/a"),
                tgt,
            ], eval_w, eval_a))
            diag_rows.append((mode, r["diag"]))
            attr_rows.append((mode, r))
            cross_rows.append((mode, r.get("cross_attr", {})))
            metrics = {k: v for k, v in r.items()
                       if isinstance(v, (float, int)) and k != "n"}
            metrics.update(r.get("cross_attr", {}))
            with csv_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                for key, value in metrics.items():
                    writer.writerow([
                        args.protocol, args.severity,
                        fixed_img_mode if args.protocol == "fixed_mask" else "random_mix",
                        fixed_aud_mode if args.protocol == "fixed_mask" else "random_mix",
                        mode, r["img_kind"], r["aud_kind"], key, value,
                        r.get("metric_counts", {}).get(key, r["n"] if math.isfinite(value) else 0)])

        _log_audio_diag(diag_rows)
        if cfg.get("eval", {}).get("recovery_classification", False):
            log("[恢复内容类别一致性] cue | variant | image ACC | audio ACC")
            for mode, values in cross_rows:
                for variant in ("normal", "zero", "wrong", "same_class"):
                    img_acc = values.get(f"content_img_{variant}_acc")
                    aud_acc = values.get(f"content_aud_{variant}_acc")
                    if img_acc is not None or aud_acc is not None:
                        log(f"  {mode} | {variant} | {_fmt_na(img_acc)} | {_fmt_na(aud_acc)}")

        attr_w = [24, 10, 10, 10, 10, 10, 10]
        attr_a = ["l"] + ["r"] * 6
        log("=" * sum(attr_w))
        log("[归因] 图像 decoder/final 与单一音频输出的 masked/visible MSE")
        log(format_table_row(
            ["cue模式", "imgCmask", "imgFmask", "imgCvis", "imgFvis",
             "audMask", "audVis"],
            attr_w, attr_a))
        for mode, r in attr_rows:
            log(format_table_row([
                mode,
                _fmt_float(r["img_coarse_masked_mse"]),
                _fmt_float(r["img_masked_mse"]),
                _fmt_float(r["img_coarse_visible_mse"]),
                _fmt_float(r["img_visible_mse"]),
                _fmt_float(r["aud_masked_mse"]),
                _fmt_float(r["aud_visible_mse"]),
            ], attr_w, attr_a))

        if args.cross_key == "sweep":
            cross_w = [24, 12, 9, 9, 11, 11, 11]
            cross_a = ["l", "l"] + ["r"] * 5
            log("=" * sum(cross_w))
            log("[Cross-Key归因] 同 cue/mask 的 normal/zero/wrong/same-class；"
                "gain=zero-normal，damage=替换条件-normal（masked MSE）")
            log(format_table_row(
                ["cue模式", "方向", "gate", "res/base", "gain",
                 "wrong", "same"], cross_w, cross_a))
            for mode, values in cross_rows:
                for prefix, direction in (("img2aud", "img->aud"),
                                          ("aud2img", "aud->img")):
                    log(format_table_row([
                        mode, direction,
                        _fmt_na(values.get(f"{prefix}_gate")),
                        _fmt_na(values.get(f"{prefix}_ratio")),
                        _fmt_na(values.get(f"{prefix}_correct_gain")),
                        _fmt_na(values.get(f"{prefix}_wrong_damage")),
                        _fmt_na(values.get(f"{prefix}_same_damage")),
                    ], cross_w, cross_a))

        if args.cross_detail == "sweep":
            detail_w = [24, 12, 9, 9, 11, 11]
            detail_a = ["l", "l"] + ["r"] * 4
            log("=" * sum(detail_w))
            log("[Cross-Detail归因] correct/zero/same-class pair；"
                "gain=zero-correct，damage=same-correct（masked MSE）")
            log(format_table_row(
                ["cue模式", "方向", "gate", "res/V", "gain", "same"],
                detail_w, detail_a))
            for mode, values in cross_rows:
                for prefix, direction in (("img2aud", "img->aud"),
                                          ("aud2img", "aud->img")):
                    log(format_table_row([
                        mode, direction,
                        _fmt_na(values.get(f"detail_{prefix}_gate")),
                        _fmt_na(values.get(f"detail_{prefix}_ratio")),
                        _fmt_na(values.get(
                            f"detail_{prefix}_correct_gain")),
                        _fmt_na(values.get(
                            f"detail_{prefix}_same_damage")),
                    ], detail_w, detail_a))

    if args.family_breakdown:
        eval_audio_family_breakdown(model, test_loader, cfg, device,
                                    args.severity, proto_img, proto_aud,
                                    args.max_batches)

    if args.severity_curve:
        log("=" * 78)
        target_text = ("真实 paired-sample 图像恢复"
                       if cfg.get("data", {}).get("dataset") == "paired_manifest"
                       else "类别代表图像恢复")
        log(f"[评估] 严重度曲线（corrupt_aud_only -> {target_text} & 分类）")
        log(f"{'severity':>9}{'acc':>8}{'imgMSE':>9}{'PSNR':>8}{'SSIM':>7}")
        for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            r = eval_mode(model, test_loader, cfg, "corrupt_aud_only",
                          device, s, proto_img, proto_aud, args.max_batches)
            log(f"{s:>9.1f}{r['acc']*100:>7.1f}%{r['img_mse']:>9.4f}"
                f"{r['psnr']:>8.2f}{r['ssim']:>7.3f}")

    log(f"[评估] CSV -> {csv_path}")
    log("[评估] 完成。")


if __name__ == "__main__":
    main()
