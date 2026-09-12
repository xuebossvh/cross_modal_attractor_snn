"""训练跨模态 SNN 联想记忆网络（binding + readout 两阶段）。

用法（在项目根目录）：
    python -u scripts/train.py --config configs/v12a.yaml
    python -u scripts/train.py --epochs 30
"""

import bootstrap  # noqa: F401

import argparse
import os
import random

import torch
import torch.nn.functional as F

from common import (fix_console_encoding, log, load_config, set_seed,
                    sample_cue_mode, sample_train_severity, build_cue,
                    select_targets, is_aud_only_mode, spike_reg,
                    resolve_train_corrupt_modes, batch_ssim,
                    unpack_paired_batch)
from paths import ensure_output_dirs, resolve_from_root
from data.dataset import build_loaders
from models.network import CrossModalSNN
from models.lif import rate
from models.frozen_base import (ADAPTER_PREFIXES, load_frozen_parent,
                                frozen_metadata, verify_frozen_resume,
                                verify_audio_normalization)


def _img_edge_loss(prob, target):
    """图像水平/垂直一阶差分 L1，锐化重建边缘。"""
    dx_p = prob[..., :, 1:] - prob[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = prob[..., 1:, :] - prob[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def _masked_image_error(rec_logits, target, mask, power=2):
    mask = mask.to(device=rec_logits.device, dtype=rec_logits.dtype)
    prob = torch.sigmoid(rec_logits)
    diff = prob - target
    err = diff.abs() if power == 1 else diff.pow(2)
    flat_mask = mask.flatten(1)
    denom = flat_mask.sum(dim=1).clamp_min(1.0)
    per_sample = (err * mask).flatten(1).sum(dim=1) / denom
    return per_sample.mean()


def _img_recon_loss(rec, x_img, lc, mask=None):
    if lc["img_recon"] == "bce":
        base = F.binary_cross_entropy_with_logits(rec, x_img)
    else:
        base = F.mse_loss(torch.sigmoid(rec), x_img)
    lam_l1 = lc.get("lambda_img_l1", 0.0)
    lam_edge = lc.get("lambda_img_edge", 0.0)
    if lam_l1 > 0 or lam_edge > 0:
        prob = torch.sigmoid(rec)
        if lam_l1 > 0:
            base = base + lam_l1 * F.l1_loss(prob, x_img)
        if lam_edge > 0:
            base = base + lam_edge * _img_edge_loss(prob, x_img)
    lam_masked = lc.get("lambda_img_masked", 0.0)
    if lam_masked > 0 and mask is not None:
        base = base + lam_masked * (
            _masked_image_error(rec, x_img, mask, power=1)
            + _masked_image_error(rec, x_img, mask, power=2)
        )
    return base


def _aud_tf_grad_loss(rec, target):
    """时频方向一阶差分 L1，约束谱图边缘结构。"""
    dt_r = rec[:, :, 1:] - rec[:, :, :-1]
    dt_t = target[:, :, 1:] - target[:, :, :-1]
    df_r = rec[:, 1:, :] - rec[:, :-1, :]
    df_t = target[:, 1:, :] - target[:, :-1, :]
    return F.l1_loss(dt_r, dt_t) + F.l1_loss(df_r, df_t)


def _aud_active_loss(rec, target):
    """惩罚 decoder 输出能量塌缩（全零谱图）。"""
    rec_std = rec.flatten(1).std(dim=1)
    tgt_std = target.flatten(1).std(dim=1)
    return F.relu(tgt_std - rec_std).mean()


def _aud_foreground_loss(rec, target, top_fraction=0.15):
    frac = max(0.0, min(1.0, float(top_fraction)))
    if frac <= 0:
        return rec.new_tensor(0.0)
    flat = target.flatten(1)
    k = max(1, int(round(frac * flat.size(1))))
    thresh = torch.topk(flat, k, dim=1).values[:, -1].view(-1, 1, 1)
    mask = (target >= thresh).to(target.dtype)
    denom = mask.sum().clamp_min(1.0)
    diff = rec - target
    return ((diff.abs() + diff.pow(2)) * mask).sum() / denom


def _aud_marginal_loss(rec, target):
    loss_t = F.l1_loss(rec.mean(dim=1), target.mean(dim=1))
    loss_f = F.l1_loss(rec.mean(dim=2), target.mean(dim=2))
    return loss_t + loss_f


def _masked_audio_error(rec, target, mask, power=2):
    mask = mask.to(device=rec.device, dtype=rec.dtype)
    diff = rec - target
    err = diff.abs() if power == 1 else diff.pow(2)
    flat_mask = mask.flatten(1)
    denom = flat_mask.sum(dim=1).clamp_min(1.0)
    per_sample = (err * mask).flatten(1).sum(dim=1) / denom
    return per_sample.mean()


def _masked_audio_weighted_mse(rec, target, mask, gamma=5.0):
    """缺失区能量加权 MSE；高能语音位置比静音位置权重更高。"""
    mask = mask.to(device=rec.device, dtype=rec.dtype)
    weight = 1.0 + float(gamma) * target.clamp_min(0.0)
    weighted_mask = weight * mask
    denom = weighted_mask.flatten(1).sum(dim=1).clamp_min(1e-8)
    per_sample = (
        (rec - target).pow(2) * weighted_mask
    ).flatten(1).sum(dim=1) / denom
    return per_sample.mean()


def _masked_mse_per_sample(rec, target, mask):
    """逐样本 masked MSE；同时返回 mask 非空的样本标记。"""
    mask = mask.to(device=rec.device, dtype=rec.dtype)
    denom = mask.flatten(1).sum(dim=1)
    valid = denom > 0
    values = ((rec - target).pow(2) * mask).flatten(1).sum(dim=1)
    values = values / denom.clamp_min(1.0)
    return values, valid


def _wrong_class_indices(labels):
    """构造不复用索引的异类置换；不可完全置换时返回最大有效子集。"""
    labels_cpu = labels.detach().cpu().tolist()
    n = len(labels_cpu)
    groups = {}
    for idx, label in enumerate(labels_cpu):
        groups.setdefault(int(label), []).append(idx)

    perm_cpu = list(range(n))
    valid_cpu = [False] * n
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
    """构造同类但不同 pair_id 的一一置换；同类单例不参与。"""
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
        if torch.any(labels[perm[valid]] != labels[valid]):
            raise RuntimeError("same-class permutation contains a wrong class")
        if pair_ids is not None and torch.any(
                pair_ids[perm[valid]] == pair_ids[valid]):
            raise RuntimeError("same-class permutation reused the same pair_id")
    return perm, valid


def _pair_alignment_loss(out, labels, pair_ids, cfg):
    """Symmetric instance retrieval loss with same-class hard negatives.

    Only the diagonal image/audio observations are positives.  Other sources
    with the same digit label are the hard negatives, so class identity alone
    cannot solve this objective.  Repeated optimizer exposures of one source
    are excluded as negatives by pair_id.
    """
    pair_cfg = cfg.get("pair_alignment", {}) or {}
    if not pair_cfg.get("enabled", False):
        return out["logits"].new_tensor(0.0), {}
    if pair_ids is None:
        raise RuntimeError(
            "pair_alignment requires stable pair_id values from a real-pair manifest")
    img = out.get("img_pair_embedding")
    aud = out.get("aud_pair_embedding")
    if img is None or aud is None:
        return out["logits"].new_tensor(0.0), {"pair_align_n": 0.0}
    if img.shape != aud.shape:
        raise ValueError(
            f"pair embeddings must have the same shape, got {img.shape}/{aud.shape}")

    temperature = float(pair_cfg.get("temperature", 0.07))
    if temperature <= 0:
        raise ValueError("pair_alignment.temperature must be > 0")
    logits = img @ aud.t() / temperature
    n = logits.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    same_source = pair_ids[:, None].eq(pair_ids[None, :])
    if pair_cfg.get("same_class_only", True):
        candidates = labels[:, None].eq(labels[None, :])
    else:
        candidates = torch.ones_like(eye)
    # Keep the designated diagonal positive; exclude repeated views of the
    # same source from the negative set.
    candidates = candidates & (~same_source | eye)
    candidates = candidates | eye
    valid = candidates.sum(dim=1) > 1
    if not valid.any():
        return logits.new_tensor(0.0), {"pair_align_n": 0.0}

    target = torch.arange(n, device=logits.device)
    masked_i2a = logits.masked_fill(~candidates, -1e9)
    masked_a2i = logits.t().masked_fill(~candidates.t(), -1e9)
    loss_i2a = F.cross_entropy(masked_i2a[valid], target[valid])
    loss_a2i = F.cross_entropy(masked_a2i[valid], target[valid])
    loss = 0.5 * (loss_i2a + loss_a2i)
    with torch.no_grad():
        i2a_acc = masked_i2a[valid].argmax(dim=1).eq(target[valid]).float().mean()
        a2i_acc = masked_a2i[valid].argmax(dim=1).eq(target[valid]).float().mean()
    return loss, {
        "pair_align": loss.item(),
        "pair_i2a_acc": i2a_acc.item(),
        "pair_a2i_acc": a2i_acc.item(),
        "pair_align_n": float(valid.sum().item()),
    }


def _masked_tf_grad_loss(rec, target, mask):
    """缺失区时频一阶差分 L1（F4，逐样本归一）。"""
    m = mask.to(device=rec.device, dtype=rec.dtype)
    dt = (rec[:, :, 1:] - rec[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])
    mt = m[:, :, 1:]
    df = (rec[:, 1:, :] - rec[:, :-1, :]) - (target[:, 1:, :] - target[:, :-1, :])
    mf = m[:, 1:, :]
    lt = (dt.abs() * mt).flatten(1).sum(1) / mt.flatten(1).sum(1).clamp_min(1.0)
    lf = (df.abs() * mf).flatten(1).sum(1) / mf.flatten(1).sum(1).clamp_min(1.0)
    return (lt + lf).mean()


def _aud_recon_loss(rec, target, lc, mask=None):
    """L1 + MSE + weighted_MSE + 时频梯度 loss。"""
    gamma = lc.get("aud_weight_gamma", 3.0)
    l1 = F.l1_loss(rec, target)
    mse = F.mse_loss(rec, target)
    w = 1.0 + gamma * target
    wmse = (w * (rec - target).pow(2)).mean()
    loss = l1 + mse + wmse
    lam_g = lc.get("lambda_aud_grad", 0.0)
    if lam_g > 0:
        loss = loss + lam_g * _aud_tf_grad_loss(rec, target)
    lam_fg = lc.get("lambda_aud_foreground", 0.0)
    if lam_fg > 0:
        loss = loss + lam_fg * _aud_foreground_loss(
            rec, target, lc.get("aud_foreground_top_fraction", 0.15))
    lam_m = lc.get("lambda_aud_marginal", 0.0)
    if lam_m > 0:
        loss = loss + lam_m * _aud_marginal_loss(rec, target)
    lam_masked = lc.get("lambda_aud_masked", 0.0)
    if lam_masked > 0 and mask is not None:
        loss_masked = (
            _masked_audio_error(rec, target, mask, power=1)
            + _masked_audio_error(rec, target, mask, power=2)
        )
        loss = loss + lam_masked * loss_masked
    lam_masked_weighted = lc.get("lambda_aud_masked_weighted", 0.0)
    if lam_masked_weighted > 0 and mask is not None:
        loss = loss + lam_masked_weighted * _masked_audio_weighted_mse(
            rec, target, mask,
            gamma=lc.get("aud_masked_weight_gamma", gamma))
    lam_ssim = lc.get("lambda_aud_ssim", 0.0)
    if lam_ssim > 0:
        ssim = batch_ssim(rec.unsqueeze(1), target.unsqueeze(1))
        loss = loss + lam_ssim * (1.0 - ssim)
    lam_mgrad = lc.get("lambda_aud_masked_grad", 0.0)
    if lam_mgrad > 0 and mask is not None:
        loss = loss + lam_mgrad * _masked_tf_grad_loss(rec, target, mask)
    return loss


def _aud_feature_loss(model, rec, target):
    """冻结 aud_encoder 特征空间 L1（F4）。

    target 侧 detach；提取 rec 特征时临时关闭 encoder 参数的 requires_grad，
    使梯度只回传到 rec（decoder/refiner），不改写 aud_encoder 权重。
    """
    enc = model.aud_encoder
    with torch.no_grad():
        target_feat = rate(enc(model._normalize_audio_for_encoder(target)))
    saved = [p.requires_grad for p in enc.parameters()]
    for p in enc.parameters():
        p.requires_grad_(False)
    try:
        rec_feat = rate(enc(model._normalize_audio_for_encoder(rec)))
    finally:
        for p, s in zip(enc.parameters(), saved):
            p.requires_grad_(s)
    return F.l1_loss(rec_feat, target_feat)


def _drop_detail_state(detail, drop_prob):
    if detail is None:
        return None
    p = max(0.0, min(1.0, float(drop_prob)))
    if p <= 0.0:
        return detail
    if p >= 1.0:
        return torch.zeros_like(detail)
    keep = (torch.rand(detail.size(0), 1, device=detail.device) >= p)
    return detail * keep.to(detail.dtype)


def _target_value_state(value_layer, enc_spikes, delay):
    _, target_state = value_layer._run_target(enc_spikes, delay)
    return target_state.detach()


def _pretrain_decoder_states(model, x_img, x_aud, detail_dropout,
                             x_img_detail=None, x_aud_detail=None):
    """Build decoder pretrain inputs from clean target Value, without Key/Index."""
    with torch.no_grad():
        spike_img = model.img_encoder(x_img)
        spike_aud = model.aud_encoder(model._normalize_audio_for_encoder(x_aud))

        delay = model.memory.binding_delay if model.memory.use_delayed_target else 0
        img_state = _target_value_state(model.memory.V_img, spike_img, delay)
        aud_state = _target_value_state(model.memory.V_aud, spike_aud, delay)

        if model.use_detail_conditioning:
            if x_img_detail is None:
                spike_img_detail = spike_img
            else:
                spike_img_detail = model.img_encoder(x_img_detail)
            if x_aud_detail is None:
                spike_aud_detail = spike_aud
            else:
                spike_aud_detail = model.aud_encoder(
                    model._normalize_audio_for_encoder(x_aud_detail))
            img_detail = _drop_detail_state(
                rate(spike_img_detail).detach(), detail_dropout)
            aud_detail = _drop_detail_state(
                rate(spike_aud_detail).detach(), detail_dropout)
        else:
            img_detail = aud_detail = None

    img_state = model._fuse_decoder_state(
        img_state, img_detail, "img", cross_key_rate=None)
    aud_state = model._fuse_decoder_state(
        aud_state, aud_detail, "aud", cross_key_rate=None)
    return img_state, aud_state


def _set_decoder_pretrain_requires_grad(model, cfg, freeze_non_decoders=True):
    previous = []
    pc = cfg.get("decoder_pretrain", {})
    img_cfg = cfg.get("image_refiner", {})
    aud_cfg = cfg.get("audio_refiner", {})
    train_img_refiner = (
        pc.get("train_image_refiner", False)
        and img_cfg.get("enabled", False)
        and not img_cfg.get("pasteback_only", False)
    )
    train_aud_refiner = (
        pc.get("train_audio_refiner", False)
        and aud_cfg.get("enabled", False)
        and not aud_cfg.get("bypass", False)
        and not aud_cfg.get("pasteback_only", False)
    )
    decoder_prefixes = (
        "image_decoder.", "audio_decoder.",
        "audio_local_cue_projector.",
        "img_detail_projector.", "aud_detail_projector.",
        "img_detail_gate.", "aud_detail_gate.",
    )
    if train_img_refiner:
        decoder_prefixes = decoder_prefixes + ("image_refiner.",)
    if train_aud_refiner:
        decoder_prefixes = decoder_prefixes + ("audio_refiner.",)
    for name, param in model.named_parameters():
        previous.append((param, param.requires_grad))
        if freeze_non_decoders:
            param.requires_grad_(name.startswith(decoder_prefixes))
    return previous


def _restore_requires_grad(previous):
    for param, requires_grad in previous:
        param.requires_grad_(requires_grad)


def _save_decoder_pretrain_ckpt(model, cfg, pre_ckpt, epoch, epochs):
    os.makedirs(os.path.dirname(pre_ckpt), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg,
        "decoder_pretrain_epoch": epoch,
        "decoder_pretrain_epochs": epochs,
    }, pre_ckpt)


def pretrain_decoders(model, train_loader, cfg, device):
    pc = cfg.get("decoder_pretrain", {})
    epochs = int(pc.get("epochs", 0))
    if not pc.get("enabled", False) or epochs <= 0:
        return
    if pc.get("train_cross_key_conditioning", False):
        raise ValueError(
            "first-stage decoder pretrain requires "
            "decoder_pretrain.train_cross_key_conditioning=false")

    lc = cfg["loss"]
    lam_img = pc.get("lambda_img", lc.get("lambda_img", 1.0))
    lam_aud = pc.get("lambda_aud", lc.get("lambda_aud", 1.0))
    lam_act = pc.get("lambda_aud_active", lc.get("lambda_aud_active", 0.0))
    detail_dropout = pc.get("detail_dropout", 0.0)
    corrupt_detail = bool(pc.get("corrupt_detail", False))
    corrupt_severity = float(pc.get(
        "corrupt_severity", cfg["corruption"].get("train_severity", 0.5)))
    pre_img_mode = pc.get("img_mode", cfg["corruption"].get("img_mode", "random"))
    pre_aud_mode = pc.get("aud_mode", cfg["corruption"].get("aud_mode", "random"))
    pre_img_modes = pc.get("img_modes")
    pre_aud_modes = pc.get("aud_modes")
    pre_family_sampling = pc.get("family_sampling", "balanced")
    use_masked_pretrain = bool(pc.get("use_masked_audio_loss", False))
    grad_clip = float(pc.get("grad_clip", 0.0))
    log_every = int(pc.get("log_every", cfg["train"].get("log_every", 50)))
    pre_ckpt = str(resolve_from_root(pc.get(
        "ckpt_path", "outputs/checkpoints/decoder_pretrain.pt")))
    start_epoch = 0
    if pc.get("resume", True) and os.path.isfile(pre_ckpt):
        state = torch.load(pre_ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        start_epoch = int(state.get("decoder_pretrain_epoch", -1)) + 1
        log(f"[decoder-pretrain] resume from {pre_ckpt} "
            f"start_epoch={start_epoch}/{epochs - 1}")
        if start_epoch >= epochs:
            log("[decoder-pretrain] already complete; skipping.")
            return

    previous_requires_grad = _set_decoder_pretrain_requires_grad(
        model, cfg, pc.get("freeze_non_decoders", True))
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        _restore_requires_grad(previous_requires_grad)
        raise RuntimeError("[decoder-pretrain] no trainable decoder parameters")
    opt = torch.optim.Adam(
        params,
        lr=pc.get("lr", cfg["train"]["lr"]),
        weight_decay=pc.get("weight_decay", 0.0),
    )

    steps_per_epoch = len(train_loader)
    lam_coarse_aux = float(pc.get("lambda_coarse_aux", 0.0))
    train_img_refiner = (
        pc.get("train_image_refiner", False)
        and cfg.get("image_refiner", {}).get("enabled", False)
        and not cfg.get("image_refiner", {}).get("pasteback_only", False)
    )
    train_aud_refiner = (
        pc.get("train_audio_refiner", False)
        and cfg.get("audio_refiner", {}).get("enabled", False)
        and not cfg.get("audio_refiner", {}).get("bypass", False)
        and not cfg.get("audio_refiner", {}).get("pasteback_only", False)
    )
    use_img_final_helper = (
        train_img_refiner
        or cfg.get("image_refiner", {}).get("pasteback_only", False)
    )
    use_aud_final_helper = (
        train_aud_refiner
        or cfg.get("audio_refiner", {}).get("pasteback_only", False)
    )
    log("[decoder-pretrain] start "
        f"epochs={epochs} lr={pc.get('lr', cfg['train']['lr'])} "
        f"detail_dropout={float(detail_dropout):.2f} "
        f"corrupt_detail={corrupt_detail} "
        f"grad_clip={grad_clip:.2f} "
        f"freeze_non_decoders={pc.get('freeze_non_decoders', True)} "
        f"train_image_refiner={train_img_refiner} "
        f"train_audio_refiner={train_aud_refiner} "
        f"train_cross_key_conditioning=False "
        f"img_final_path={'helper' if use_img_final_helper else 'coarse'} "
        f"aud_final_path={'helper' if use_aud_final_helper else 'coarse'} "
        f"lambda_coarse_aux={lam_coarse_aux:.3f}")
    if corrupt_detail:
        log("[decoder-pretrain] corrupt detail "
            f"img_mode={pre_img_mode} aud_mode={pre_aud_mode} "
            f"img_modes={pre_img_modes or '-'} aud_modes={pre_aud_modes or '-'} "
            f"severity={corrupt_severity:.2f} "
            f"masked_audio_loss={use_masked_pretrain}")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0
            for step, batch in enumerate(train_loader):
                x_img, x_aud, labels, _ = unpack_paired_batch(batch)
                del labels
                x_img = x_img.to(device)
                x_aud = x_aud.to(device)

                x_img_detail = None
                x_aud_detail = None
                img_mask = None
                aud_mask = None
                if corrupt_detail:
                    cur_pre_img_mode = _select_family_mode(
                        pre_img_modes, pre_img_mode, step, pre_family_sampling)
                    cur_pre_aud_mode = _select_family_mode(
                        pre_aud_modes, pre_aud_mode, step, pre_family_sampling)
                    x_img_detail, x_aud_detail, cue_masks = build_cue(
                        x_img, x_aud, "corrupt_both", cfg,
                        severity=corrupt_severity,
                        img_mode=cur_pre_img_mode, aud_mode=cur_pre_aud_mode,
                        return_masks=True)
                    img_mask = cue_masks.get("img")
                    aud_mask = cue_masks.get("aud")
                else:
                    cur_pre_img_mode = pre_img_mode
                    cur_pre_aud_mode = pre_aud_mode

                img_state, aud_state = _pretrain_decoder_states(
                    model, x_img, x_aud, detail_dropout,
                    x_img_detail=x_img_detail, x_aud_detail=x_aud_detail)

                coarse_img = model.image_decoder(img_state)
                decoder_aud = model.audio_decoder(aud_state)
                rec_img = (
                    model._apply_image_refiner(coarse_img, x_img_detail, img_mask)
                    if use_img_final_helper else coarse_img
                )
                rec_aud = (
                    model._finalize_audio(decoder_aud, x_aud_detail, aud_mask)
                    if use_aud_final_helper else decoder_aud
                )

                mask_for_img_loss = None
                img_masked_families = set(lc.get("img_masked_families", []))
                if (pc.get("use_masked_image_loss", False)
                        and cur_pre_img_mode in img_masked_families
                        and img_mask is not None):
                    mask_for_img_loss = img_mask
                loss_img = _img_recon_loss(rec_img, x_img, lc,
                                           mask=mask_for_img_loss)
                if lam_coarse_aux > 0:
                    loss_img = loss_img + lam_coarse_aux * _img_recon_loss(
                        coarse_img, x_img, lc, mask=mask_for_img_loss)
                mask_for_loss = None
                masked_families = set(lc.get("aud_masked_families", []))
                if (use_masked_pretrain
                        and cur_pre_aud_mode in masked_families
                        and aud_mask is not None):
                    mask_for_loss = aud_mask
                loss_aud = _aud_recon_loss(rec_aud, x_aud, lc,
                                           mask=mask_for_loss)
                if lam_coarse_aux > 0:
                    loss_aud = loss_aud + lam_coarse_aux * _aud_recon_loss(
                        decoder_aud, x_aud, lc, mask=mask_for_loss)
                loss = lam_img * loss_img + lam_aud * loss_aud
                logs = {
                    "img": loss_img.item(),
                    "aud": loss_aud.item(),
                }
                if mask_for_loss is not None:
                    logs["aud_mask_l1"] = _masked_audio_error(
                        rec_aud, x_aud, mask_for_loss, power=1).item()
                    logs["aud_mask_mse"] = _masked_audio_error(
                        rec_aud, x_aud, mask_for_loss, power=2).item()
                    logs["aud_mask_wmse"] = _masked_audio_weighted_mse(
                        rec_aud, x_aud, mask_for_loss,
                        gamma=lc.get("aud_masked_weight_gamma", 5.0)).item()
                    visible_mask = 1.0 - mask_for_loss.to(
                        device=rec_aud.device, dtype=rec_aud.dtype)
                    logs["aud_visible_l1"] = _masked_audio_error(
                        rec_aud, x_aud, visible_mask, power=1).item()
                    logs["aud_visible_mse"] = _masked_audio_error(
                        rec_aud, x_aud, visible_mask, power=2).item()
                if mask_for_img_loss is not None:
                    logs["img_mask_l1"] = _masked_image_error(
                        rec_img, x_img, mask_for_img_loss, power=1).item()
                    logs["img_mask_mse"] = _masked_image_error(
                        rec_img, x_img, mask_for_img_loss, power=2).item()
                    visible_img_mask = 1.0 - mask_for_img_loss.to(
                        device=rec_img.device, dtype=rec_img.dtype)
                    logs["img_visible_l1"] = _masked_image_error(
                        rec_img, x_img, visible_img_mask, power=1).item()
                    logs["img_visible_mse"] = _masked_image_error(
                        rec_img, x_img, visible_img_mask, power=2).item()
                if lam_act > 0:
                    loss_act = _aud_active_loss(rec_aud, x_aud)
                    loss = loss + lam_act * loss_act
                    logs["aud_act"] = loss_act.item()

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "[decoder-pretrain] non-finite loss at "
                        f"epoch={epoch} step={step}: {logs}")

                opt.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        params, grad_clip, error_if_nonfinite=True)
                opt.step()
                epoch_loss += loss.item()

                if step % log_every == 0 or step == steps_per_epoch - 1:
                    parts = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                    log(f"[decoder-pretrain] epoch {epoch}/{epochs - 1} "
                        f"step {step}/{steps_per_epoch - 1} "
                        f"loss={loss.item():.4f} | {parts}")

            avg_loss = epoch_loss / max(steps_per_epoch, 1)
            log(f"[decoder-pretrain] epoch {epoch} avg_loss={avg_loss:.4f}")
            if pc.get("save_every_epoch", True):
                _save_decoder_pretrain_ckpt(model, cfg, pre_ckpt, epoch, epochs)
                log(f"[decoder-pretrain] checkpoint saved -> {pre_ckpt}")
    finally:
        _restore_requires_grad(previous_requires_grad)

    if pc.get("save_ckpt", True):
        _save_decoder_pretrain_ckpt(model, cfg, pre_ckpt, epochs - 1, epochs)
        log(f"[decoder-pretrain] checkpoint saved -> {pre_ckpt}")


def _build_train_optimizer(model, cfg):
    """Adam with optional lr_mult for refiner/cross conditioning groups."""
    base_lr = cfg["train"]["lr"]
    wd = cfg["train"]["weight_decay"]
    img_mult = float(cfg.get("image_refiner", {}).get("lr_mult", 1.0))
    aud_mult = float(cfg.get("audio_refiner", {}).get("lr_mult", 1.0))
    cross_mult = float(
        cfg.get("cross_key_conditioning", {}).get("lr_mult", 1.0))
    detail_mult = float(
        cfg.get("cross_detail_conditioning", {}).get("lr_mult", 1.0))
    pair_mult = float(cfg.get("pair_alignment", {}).get("lr_mult", 1.0))
    cross_prefixes = (
        "img_cross_adapter.", "aud_cross_adapter.",
        "aud_to_img_cross_proj.", "img_to_aud_cross_proj.",
        "aud_to_img_cross_gate.", "img_to_aud_cross_gate.",
    )
    detail_prefixes = (
        "aud_to_img_detail_proj.", "img_to_aud_detail_proj.",
        "aud_to_img_detail_gate.", "img_to_aud_detail_gate.",
    )
    pair_prefixes = ("img_pair_projector.", "aud_pair_projector.")
    base_params = []
    img_refiner_params = []
    aud_refiner_params = []
    cross_params = []
    detail_params = []
    pair_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(pair_prefixes):
            pair_params.append(param)
        elif name.startswith(detail_prefixes):
            detail_params.append(param)
        elif name.startswith(cross_prefixes):
            cross_params.append(param)
        elif name.startswith("image_refiner."):
            img_refiner_params.append(param)
        elif name.startswith("audio_refiner."):
            aud_refiner_params.append(param)
        else:
            base_params.append(param)
    groups = [{"params": base_params, "lr": base_lr}]
    if img_refiner_params:
        groups.append({"params": img_refiner_params, "lr": base_lr * img_mult})
    if aud_refiner_params:
        groups.append({"params": aud_refiner_params, "lr": base_lr * aud_mult})
    if cross_params:
        groups.append({"params": cross_params, "lr": base_lr * cross_mult})
    if detail_params:
        groups.append({"params": detail_params, "lr": base_lr * detail_mult})
    if pair_params:
        groups.append({"params": pair_params, "lr": base_lr * pair_mult})
    return torch.optim.Adam(groups, lr=base_lr, weight_decay=wd)


def _apply_trainable_prefixes(model, cfg):
    """可选地冻结除指定前缀外的参数，用于同父 checkpoint 的低风险微调。"""
    prefixes = cfg.get("train", {}).get("trainable_prefixes", [])
    if not prefixes:
        return None
    prefixes = tuple(str(prefix) for prefix in prefixes)
    if model.freeze_base and prefixes != ADAPTER_PREFIXES:
        raise ValueError("Frozen-base training only permits the two feature adapters")
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes)
        if param.requires_grad:
            trainable.append(name)
    if not trainable:
        raise RuntimeError(
            f"train.trainable_prefixes matched no parameters: {prefixes}")
    return trainable


def _checkpoint_payload(model, opt, scheduler, cfg, epoch):
    return {
        "frozen_base": frozen_metadata(model),
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": scheduler.state_dict() if scheduler is not None else None,
        "cfg": cfg,
        "epoch": epoch,
    }


def _milestone_checkpoint_path(ckpt_path, completed_epochs):
    stem, ext = os.path.splitext(ckpt_path)
    return f"{stem}_ep{completed_epochs}{ext or '.pt'}"


def _select_family_mode(pool, fallback, step, strategy="balanced"):
    if pool:
        if str(strategy).lower() == "balanced":
            return pool[int(step) % len(pool)]
        return random.choice(pool)
    return fallback


def _mode_enabled(cue_mode, modes):
    return cue_mode in set(modes or [])


def _soft_cls_loss(student_logits, teacher_logits, temperature=2.0):
    t = max(float(temperature), 1e-6)
    log_p = F.log_softmax(student_logits / t, dim=1)
    q = F.softmax(teacher_logits.detach() / t, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (t * t)


def _teacher_cues_for_mode(cue_mode, clean_img, clean_aud, match_modality):
    if not match_modality:
        return clean_img, clean_aud
    if cue_mode in ("corrupt_aud_only", "clean_aud_only"):
        return None, clean_aud
    if cue_mode in ("corrupt_img_only", "clean_img_only"):
        return clean_img, None
    return clean_img, clean_aud


def _class_key_alignment_loss(key_img, key_aud, labels, temperature=0.1):
    if key_img is None or key_aud is None:
        return None
    img = key_img.mean(dim=0)
    aud = key_aud.mean(dim=0)
    if img.size(1) != aud.size(1):
        return None

    img = F.normalize(img, dim=1)
    aud = F.normalize(aud, dim=1)
    temp = max(float(temperature), 1e-6)
    sim_i2a = img @ aud.t() / temp
    sim_a2i = aud @ img.t() / temp
    pos = (labels.view(-1, 1) == labels.view(1, -1)).to(img.dtype)

    def sup_ce(sim):
        log_prob = F.log_softmax(sim, dim=1)
        denom = pos.sum(dim=1).clamp_min(1.0)
        return -((pos * log_prob).sum(dim=1) / denom).mean()

    return 0.5 * (sup_ce(sim_i2a) + sup_ce(sim_a2i))


def _audio_detail_consistency_loss(model, out_r, clean_aud, cue_mode, cfg):
    lc = cfg["loss"]
    lam = float(lc.get("lambda_aud_detail_cons", 0.0))
    if lam <= 0 or not _mode_enabled(cue_mode, lc.get("aud_detail_cons_modes", [])):
        return out_r["index_state"].new_tensor(0.0), {}

    detail_noisy = out_r.get("aud_detail_state")
    if detail_noisy is None:
        return out_r["index_state"].new_tensor(0.0), {}

    with torch.no_grad():
        clean_spikes = model.aud_encoder(
            model._normalize_audio_for_encoder(clean_aud))
        detail_clean = rate(clean_spikes)

    loss = F.mse_loss(detail_noisy, detail_clean.detach())
    cos = F.cosine_similarity(
        detail_noisy.detach(), detail_clean.detach(), dim=1).mean()
    return lam * loss, {
        "aud_det": loss.item(),
        "aud_det_cos": cos.item(),
    }


def _alignment_losses(model, out_r, clean_img, clean_aud, labels, cue_mode, cfg):
    lc = cfg["loss"]
    total = out_r["index_state"].new_tensor(0.0)
    logs = {}

    teacher_modes = lc.get("align_teacher_modes", [])
    need_teacher = (
        (lc.get("lambda_index_cons", 0.0) > 0
         or lc.get("lambda_soft_cls", 0.0) > 0)
        and _mode_enabled(cue_mode, teacher_modes)
    )
    if need_teacher:
        with torch.no_grad():
            teacher_img, teacher_aud = _teacher_cues_for_mode(
                cue_mode, clean_img, clean_aud,
                lc.get("align_teacher_match_modality", False))
            out_clean = model(x_img_cue=teacher_img, x_aud_cue=teacher_aud,
                              training_mode=False, phase="readout")
        lam_idx = lc.get("lambda_index_cons", 0.0)
        if lam_idx > 0:
            loss_idx = F.mse_loss(out_r["index_state"],
                                  out_clean["index_state"].detach())
            total = total + lam_idx * loss_idx
            logs["idx_cons"] = loss_idx.item()

        lam_soft = lc.get("lambda_soft_cls", 0.0)
        if lam_soft > 0:
            loss_soft = _soft_cls_loss(
                out_r["logits"], out_clean["logits"],
                lc.get("soft_cls_temperature", 2.0))
            total = total + lam_soft * loss_soft
            logs["soft_cls"] = loss_soft.item()

    lam_key = lc.get("lambda_key_align", 0.0)
    if lam_key > 0 and _mode_enabled(cue_mode, lc.get("key_align_modes", [])):
        loss_key = _class_key_alignment_loss(
            out_r.get("key_img"), out_r.get("key_aud"), labels,
            lc.get("key_align_temperature", 0.1))
        if loss_key is not None:
            total = total + lam_key * loss_key
            logs["key_align"] = loss_key.item()

    return total, logs


def _apply_audio_target_curriculum(tgt_aud, labels, cue_mode, aud_kind,
                                   cfg, proto_aud, epoch):
    cur = cfg["loss"].get("aud_sample_curriculum", {})
    if not isinstance(cur, dict) or not cur.get("enabled", False):
        return tgt_aud, None
    if aud_kind != "sample" or cue_mode not in cur.get("modes", ["corrupt_both"]):
        return tgt_aud, None

    start = int(cur.get("start_epoch", 0))
    end = int(cur.get("end_epoch", 35))
    if epoch <= start:
        sample_w = 0.0
    elif epoch >= end:
        sample_w = 1.0
    else:
        sample_w = float(epoch - start) / max(float(end - start), 1.0)
    sample_w = min(sample_w, float(cur.get("max_sample_mix", 1.0)))

    cat_aud = proto_aud[labels]
    mixed = sample_w * tgt_aud + (1.0 - sample_w) * cat_aud
    return mixed.clamp(0.0, 1.0), sample_w


def _cross_key_causal_loss(model, out_correct, img_cue, aud_cue,
                           tgt_img, tgt_aud, img_mask, aud_mask,
                           labels, cue_mode, cfg, epoch=0, step=0):
    """用 detached zero/wrong reference 强制正确对侧 Key 改善最终输出误差。"""
    causal = cfg.get("cross_key_conditioning", {}).get(
        "causal_training", {})
    if not causal.get("enabled", False):
        return out_correct["logits"].new_tensor(0.0), {}
    probability = float(causal.get("batch_probability", 0.25))
    # Do not perturb the global cue/corruption RNG stream in the main run.
    decision_seed = (
        (int(cfg.get("seed", 0)) + 1) * 1_000_003
        + int(epoch) * 10_007
        + int(step)
    )
    if (probability <= 0
            or random.Random(decision_seed).random() >= probability):
        return out_correct["logits"].new_tensor(0.0), {}

    use_img2aud = (
        cue_mode in ("corrupt_img_only", "corrupt_both",
                     "clean_img_corrupt_aud", "clean_img_only")
        and out_correct.get("key_img") is not None
    )
    use_aud2img = (
        cue_mode in ("corrupt_aud_only", "corrupt_both",
                     "corrupt_img_clean_aud", "clean_aud_only")
        and out_correct.get("key_aud") is not None
    )
    if not (use_img2aud or use_aud2img):
        return out_correct["logits"].new_tensor(0.0), {}

    wrong_perm, wrong_valid = _wrong_class_indices(labels)

    def run_reference(**cross_kwargs):
        return model(
            x_img_cue=img_cue, x_aud_cue=aud_cue,
            training_mode=True, phase="readout",
            img_cue_mask=img_mask, aud_cue_mask=aud_mask,
            **cross_kwargs)

    rng_devices = []
    if labels.device.type == "cuda":
        rng_devices = [
            labels.device.index
            if labels.device.index is not None
            else torch.cuda.current_device()
        ]
    # Reference interventions must not advance the main/control Torch RNG.
    with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
        zero_out = run_reference(
            disable_img_to_aud_cross=True,
            disable_aud_to_img_cross=True)
        wrong_kwargs = {}
        if out_correct.get("key_img") is not None:
            img_rate = rate(out_correct["key_img"]).detach()
            wrong_kwargs["cross_key_img_rate_override"] = img_rate[wrong_perm]
        if out_correct.get("key_aud") is not None:
            aud_rate = rate(out_correct["key_aud"]).detach()
            wrong_kwargs["cross_key_aud_rate_override"] = aud_rate[wrong_perm]
        wrong_out = run_reference(**wrong_kwargs)

    margin_ratio = float(causal.get("margin_ratio", 0.05))
    direction_losses = []
    logs = {}
    effective_aud_mask = (
        torch.ones_like(tgt_aud) if aud_cue is None else aud_mask)
    effective_img_mask = (
        torch.ones_like(tgt_img) if img_cue is None else img_mask)

    def add_direction(name, correct, zero, wrong, target, mask):
        if mask is None:
            return
        correct_err, region_valid = _masked_mse_per_sample(
            correct, target, mask)
        zero_err, _ = _masked_mse_per_sample(zero, target, mask)
        wrong_err, _ = _masked_mse_per_sample(wrong, target, mask)
        valid = region_valid
        if not valid.any():
            return
        correct_sel = correct_err[valid]
        zero_sel = zero_err[valid].detach()
        wrong_sel = wrong_err[valid].detach()
        if causal.get("normalization") == "batch_floor":
            floor_key = "aud_scale_floor" if name == "img2aud" else "img_scale_floor"
            floor = float(causal.get(floor_key, 0.005))
            if floor <= 0:
                raise ValueError("Causal normalization floor must be positive")
            scale = zero_sel.mean().clamp_min(floor)
        else:
            scale = zero_sel.clamp_min(1e-6)
        correct_rel = correct_sel / scale
        wrong_rel = wrong_sel / scale
        pair_loss = F.relu((correct_sel - zero_sel) / scale + margin_ratio).mean()
        valid_wrong = wrong_valid[valid]
        if valid_wrong.any():
            pair_loss = pair_loss + F.relu(
                correct_rel[valid_wrong] - wrong_rel[valid_wrong] + margin_ratio).mean()
        direction_losses.append(pair_loss)
        logs[f"cross_{name}"] = pair_loss.item()
        logs[f"{name}_correct"] = correct_sel.detach().mean().item()
        logs[f"{name}_zero"] = zero_sel.mean().item()
        logs[f"{name}_wrong"] = wrong_sel.mean().item()
        logs[f"{name}_correct_rel"] = correct_rel.detach().mean().item()
        logs[f"{name}_wrong_rel"] = wrong_rel.mean().item()
        logs[f"{name}_n"] = float(valid.sum().item())
        logs[f"{name}_wrong_n"] = float(valid_wrong.sum().item())
        logs[f"{name}_gain"] = (zero_sel - correct_sel.detach()).mean().item()
        logs[f"{name}_win_zero"] = (correct_sel.detach() < zero_sel - 1e-8).float().mean().item()

    if use_img2aud:
        add_direction(
            "img2aud",
            out_correct["recovered_aud"],
            zero_out["recovered_aud"],
            wrong_out["recovered_aud"],
            tgt_aud, effective_aud_mask)
    if use_aud2img:
        add_direction(
            "aud2img",
            torch.sigmoid(out_correct["recovered_img"]),
            torch.sigmoid(zero_out["recovered_img"]),
            torch.sigmoid(wrong_out["recovered_img"]),
            tgt_img, effective_img_mask)

    if not direction_losses:
        return out_correct["logits"].new_tensor(0.0), {}
    loss = torch.stack(direction_losses).mean()
    logs["cross_pair"] = loss.item()
    return loss, logs


def _cross_detail_causal_loss(model, out_correct, img_cue, aud_cue,
                              tgt_img, tgt_aud, img_mask, aud_mask,
                              labels, pair_ids, cue_mode, cfg,
                              epoch=0, step=0):
    """要求正确 pair 的实例 Detail 优于关闭及同类错误 pair。"""
    causal = cfg.get("cross_detail_conditioning", {}).get(
        "causal_training", {})
    if not causal.get("enabled", False):
        return out_correct["logits"].new_tensor(0.0), {}

    probability = float(causal.get("batch_probability", 1.0))
    decision_seed = (
        (int(cfg.get("seed", 0)) + 17) * 1_000_003
        + int(epoch) * 10_007
        + int(step)
    )
    if (probability <= 0
            or random.Random(decision_seed).random() >= probability):
        return out_correct["logits"].new_tensor(0.0), {}

    use_img2aud = (
        img_cue is not None
        and out_correct.get("img_cross_detail_state") is not None)
    use_aud2img = (
        aud_cue is not None
        and out_correct.get("aud_cross_detail_state") is not None)
    if not (use_img2aud or use_aud2img):
        return out_correct["logits"].new_tensor(0.0), {}

    same_perm, same_valid = _same_class_indices(labels, pair_ids)
    if not same_valid.any():
        return out_correct["logits"].new_tensor(0.0), {
            "detail_pair_n": 0.0,
        }

    def run_reference(**kwargs):
        return model(
            x_img_cue=img_cue, x_aud_cue=aud_cue,
            training_mode=True, phase="readout",
            img_cue_mask=img_mask, aud_cue_mask=aud_mask,
            **kwargs)

    rng_devices = []
    if labels.device.type == "cuda":
        rng_devices = [labels.device.index
                       if labels.device.index is not None
                       else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
        zero_out = run_reference(
            disable_img_to_aud_detail=True,
            disable_aud_to_img_detail=True)
        same_kwargs = {}
        img_detail = out_correct.get("img_cross_detail_state")
        aud_detail = out_correct.get("aud_cross_detail_state")
        if img_detail is not None:
            same_kwargs["cross_detail_img_rate_override"] = (
                img_detail.detach()[same_perm])
        if aud_detail is not None:
            same_kwargs["cross_detail_aud_rate_override"] = (
                aud_detail.detach()[same_perm])
        same_out = run_reference(**same_kwargs)

    margin_ratio = float(causal.get("margin_ratio", 0.05))
    direction_losses = []
    logs = {}
    effective_aud_mask = (
        torch.ones_like(tgt_aud) if aud_cue is None else aud_mask)
    effective_img_mask = (
        torch.ones_like(tgt_img) if img_cue is None else img_mask)

    def add_direction(name, correct, zero, same, target, mask):
        if mask is None:
            return
        correct_err, region_valid = _masked_mse_per_sample(
            correct, target, mask)
        zero_err, _ = _masked_mse_per_sample(zero, target, mask)
        same_err, _ = _masked_mse_per_sample(same, target, mask)
        valid = same_valid & region_valid
        if not valid.any():
            return
        correct_sel = correct_err[valid]
        zero_sel = zero_err[valid].detach()
        same_sel = same_err[valid].detach()
        scale = zero_sel.clamp_min(1e-6)
        correct_rel = correct_sel / scale
        same_rel = same_sel / scale
        pair_loss = (
            F.relu(correct_rel - 1.0 + margin_ratio)
            + F.relu(correct_rel - same_rel + margin_ratio)
        ).mean()
        direction_losses.append(pair_loss)
        logs[f"detail_{name}"] = pair_loss.item()
        logs[f"{name}_detail_correct"] = correct_sel.detach().mean().item()
        logs[f"{name}_detail_zero"] = zero_sel.mean().item()
        logs[f"{name}_detail_same"] = same_sel.mean().item()
        logs[f"{name}_detail_n"] = float(valid.sum().item())

    if use_img2aud:
        add_direction(
            "img2aud", out_correct["recovered_aud"],
            zero_out["recovered_aud"], same_out["recovered_aud"],
            tgt_aud, effective_aud_mask)
    if use_aud2img:
        add_direction(
            "aud2img", torch.sigmoid(out_correct["recovered_img"]),
            torch.sigmoid(zero_out["recovered_img"]),
            torch.sigmoid(same_out["recovered_img"]),
            tgt_img, effective_img_mask)

    if not direction_losses:
        return out_correct["logits"].new_tensor(0.0), logs
    loss = torch.stack(direction_losses).mean()
    logs["detail_pair"] = loss.item()
    return loss, logs


def compute_losses(model, clean_img, clean_aud, labels, cue_mode, cfg,
                   proto_img, proto_aud, pair_ids=None, epoch=0, step=0):
    """返回 (总损失, 日志字典)。"""
    lc = cfg["loss"]
    ab = cfg.get("ablation", {})
    use_binding = ab.get("use_binding_phase", True) and not model.freeze_base

    severity = sample_train_severity(cfg, epoch)
    img_mode, aud_mode = resolve_train_corrupt_modes(cfg, epoch, step=step)
    img_cue, aud_cue, cue_masks = build_cue(
        clean_img, clean_aud, cue_mode, cfg, severity=severity,
        img_mode=img_mode, aud_mode=aud_mode, return_masks=True)
    img_mask = cue_masks.get("img")
    aud_mask = cue_masks.get("aud")
    tgt_img, tgt_aud, img_kind, aud_kind = select_targets(
        cue_mode, clean_img, clean_aud, proto_img, proto_aud, labels,
        paired_missing_targets=bool(
            cfg.get("data", {}).get("pairing", {}).get(
                "sample_targets_for_missing", False)),
        paired_target_kind=(
            "paired-sample" if cfg.get("data", {}).get("dataset")
            == "paired_manifest" else "sample"))
    tgt_aud, aud_mix = _apply_audio_target_curriculum(
        tgt_aud, labels, cue_mode, aud_kind, cfg, proto_aud, epoch)

    total = 0.0
    logs = {"sev": severity}
    if aud_mix is not None:
        logs["aud_mix"] = aud_mix

    if use_binding:
        out_b = model(x_img_cue=img_cue, x_aud_cue=aud_cue,
                      x_img_target=tgt_img, x_aud_target=tgt_aud,
                      training_mode=True, phase="binding")
        if out_b["v_img_target"] is not None:
            bind_img = F.mse_loss(out_b["v_img_from_A"],
                                  out_b["v_img_target"].detach())
            total = total + lc["lambda_bind_img"] * bind_img
            logs["bind_img"] = bind_img.item()
        if out_b["v_aud_target"] is not None:
            bind_aud = F.mse_loss(out_b["v_aud_from_A"],
                                  out_b["v_aud_target"].detach())
            total = total + lc["lambda_bind_aud"] * bind_aud
            logs["bind_aud"] = bind_aud.item()

    out_r = model(x_img_cue=img_cue, x_aud_cue=aud_cue,
                  training_mode=True, phase="readout",
                  img_cue_mask=img_mask, aud_cue_mask=aud_mask)

    if not model.freeze_base:
        loss_align, align_logs = _alignment_losses(
            model, out_r, clean_img, clean_aud, labels, cue_mode, cfg)
        total = total + loss_align
        logs.update(align_logs)

    loss_detail, detail_logs = _audio_detail_consistency_loss(
        model, out_r, clean_aud, cue_mode, cfg)
    total = total + loss_detail
    logs.update(detail_logs)

    loss_pair_align, pair_align_logs = _pair_alignment_loss(
        out_r, labels, pair_ids, cfg)
    pair_align_weight = float(
        cfg.get("pair_alignment", {}).get("loss_weight", 0.0))
    total = total + pair_align_weight * loss_pair_align
    logs.update(pair_align_logs)

    loss_cls = F.cross_entropy(out_r["logits"], labels)
    cls_w = lc["lambda_cls"]
    if is_aud_only_mode(cue_mode):
        cls_w *= lc.get("lambda_cls_aud_only_mult", 2.0)
    total = total + cls_w * loss_cls
    logs["cls"] = loss_cls.item()

    if out_r.get("aux_aud_logits") is not None and aud_cue is not None:
        loss_aux = F.cross_entropy(out_r["aux_aud_logits"], labels)
        lam_aux = lc.get("lambda_audio_aux", 0.0)
        if lam_aux > 0:
            total = total + lam_aux * loss_aux
            logs["aux_aud"] = loss_aux.item()

    img_masked_families = set(lc.get("img_masked_families", []))
    use_img_mask = (
        cue_mode in ("corrupt_img_only", "corrupt_both",
                     "corrupt_img_clean_aud")
        and img_kind == "sample"
        and img_mode in img_masked_families
        and img_mask is not None
        and lc.get("lambda_img_masked", 0.0) > 0
    )
    mask_for_img_loss = img_mask if use_img_mask else None
    loss_img = _img_recon_loss(out_r["recovered_img"], tgt_img, lc,
                               mask=mask_for_img_loss)
    total = total + lc["lambda_img"] * loss_img
    logs[f"img({img_kind[:3]})"] = loss_img.item()
    if use_img_mask:
        logs["img_mask_l1"] = _masked_image_error(
            out_r["recovered_img"], tgt_img, img_mask, power=1).item()
        logs["img_mask_mse"] = _masked_image_error(
            out_r["recovered_img"], tgt_img, img_mask, power=2).item()
        visible_img_mask = 1.0 - img_mask.to(
            device=out_r["recovered_img"].device,
            dtype=out_r["recovered_img"].dtype)
        logs["img_visible_l1"] = _masked_image_error(
            out_r["recovered_img"], tgt_img, visible_img_mask, power=1).item()
        logs["img_visible_mse"] = _masked_image_error(
            out_r["recovered_img"], tgt_img, visible_img_mask, power=2).item()

    masked_families = set(lc.get("aud_masked_families", []))
    use_aud_mask = (
        cue_mode in ("corrupt_aud_only", "corrupt_both",
                     "clean_img_corrupt_aud")
        and aud_kind == "sample"
        and aud_mode in masked_families
        and aud_mask is not None
    )
    mask_for_loss = aud_mask if use_aud_mask else None
    loss_aud = _aud_recon_loss(out_r["recovered_aud"], tgt_aud, lc,
                               mask=mask_for_loss)
    total = total + lc["lambda_aud"] * loss_aud
    logs[f"aud({aud_kind[:3]})"] = loss_aud.item()

    if use_aud_mask:
        logs["aud_mask_l1"] = _masked_audio_error(
            out_r["recovered_aud"], tgt_aud, aud_mask, power=1).item()
        logs["aud_mask_mse"] = _masked_audio_error(
            out_r["recovered_aud"], tgt_aud, aud_mask, power=2).item()
        logs["aud_mask_wmse"] = _masked_audio_weighted_mse(
            out_r["recovered_aud"], tgt_aud, aud_mask,
            gamma=lc.get("aud_masked_weight_gamma", 5.0)).item()
        visible_mask = 1.0 - aud_mask.to(
            device=out_r["recovered_aud"].device,
            dtype=out_r["recovered_aud"].dtype)
        logs["aud_visible_l1"] = _masked_audio_error(
            out_r["recovered_aud"], tgt_aud, visible_mask, power=1).item()
        logs["aud_visible_mse"] = _masked_audio_error(
            out_r["recovered_aud"], tgt_aud, visible_mask, power=2).item()

    lam_feat = lc.get("lambda_aud_feat", 0.0)
    if lam_feat > 0 and aud_kind == "sample":
        loss_feat = _aud_feature_loss(model, out_r["recovered_aud"], tgt_aud)
        total = total + lam_feat * loss_feat
        logs["aud_feat"] = loss_feat.item()

    if aud_kind == "sample":
        lam_act = lc.get("lambda_aud_active", 0.0)
        if lam_act > 0:
            loss_act = _aud_active_loss(out_r["recovered_aud"], tgt_aud)
            total = total + lam_act * loss_act
            logs["aud_act"] = loss_act.item()

    causal_cfg = cfg.get("cross_key_conditioning", {}).get(
        "causal_training", {})
    causal_weight = float(causal_cfg.get("loss_weight", 0.0))
    if causal_weight > 0:
        loss_cross, cross_logs = _cross_key_causal_loss(
            model, out_r, img_cue, aud_cue,
            tgt_img, tgt_aud, img_mask, aud_mask,
            labels, cue_mode, cfg, epoch=epoch, step=step)
        if cross_logs:
            total = total + causal_weight * loss_cross
            logs.update(cross_logs)

    detail_causal_cfg = cfg.get("cross_detail_conditioning", {}).get(
        "causal_training", {})
    detail_causal_weight = float(detail_causal_cfg.get("loss_weight", 0.0))
    if detail_causal_weight > 0:
        loss_pair, pair_logs = _cross_detail_causal_loss(
            model, out_r, img_cue, aud_cue,
            tgt_img, tgt_aud, img_mask, aud_mask,
            labels, pair_ids, cue_mode, cfg, epoch=epoch, step=step)
        if pair_logs:
            total = total + detail_causal_weight * loss_pair
            logs.update(pair_logs)

    total = total + lc["lambda_reg"] * spike_reg(out_r)

    return total, logs


@torch.no_grad()
def _validate_model(model, loader, cfg, device):
    """Deterministic paired validation used for best-checkpoint selection."""
    validation = cfg.get("validation", {}) or {}
    if not validation.get("enabled", False):
        return None
    max_batches = int(validation.get("max_batches", 20))
    severity = float(validation.get("severity", 0.4))
    fixed = cfg.get("corruption", {}).get("eval_fixed", {}) or {}

    def first_mode(value, fallback):
        if isinstance(value, (list, tuple)):
            return value[0] if value else fallback
        return value or fallback

    img_mode = first_mode(fixed.get("img_modes", fixed.get("img_mode")),
                          "occlusion")
    aud_mode = first_mode(fixed.get("aud_modes", fixed.get("aud_mode")),
                          "time_mask")
    was_training = model.training
    model.eval()
    n = 0
    img_sum = aud_sum = correct = 0.0
    pair_i2a_sum = pair_a2i_sum = pair_n = 0.0
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        x_img, x_aud, labels, pair_ids = unpack_paired_batch(batch)
        x_img = x_img.to(device)
        x_aud = x_aud.to(device)
        labels = labels.to(device)
        if pair_ids is None:
            raise RuntimeError("v11e validation requires pair_id")
        pair_ids = pair_ids.to(device)

        devices = []
        if device.type == "cuda":
            devices = [device.index if device.index is not None
                       else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            seed = int(cfg.get("seed", 0)) * 100003 + bi
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            img_cue, aud_cue, masks = build_cue(
                x_img, x_aud, "corrupt_both", cfg, severity=severity,
                img_mode=img_mode, aud_mode=aud_mode, return_masks=True)
            out = model(
                x_img_cue=img_cue, x_aud_cue=aud_cue,
                img_cue_mask=masks.get("img"),
                aud_cue_mask=masks.get("aud"), training_mode=False)

        batch_size = labels.size(0)
        n += batch_size
        img_sum += F.mse_loss(
            torch.sigmoid(out["recovered_img"]), x_img).item() * batch_size
        aud_sum += F.mse_loss(
            out["recovered_aud"], x_aud).item() * batch_size
        correct += out["logits"].argmax(dim=1).eq(labels).sum().item()
        _, pair_logs = _pair_alignment_loss(out, labels, pair_ids, cfg)
        valid_n = float(pair_logs.get("pair_align_n", 0.0))
        if valid_n > 0:
            pair_i2a_sum += pair_logs["pair_i2a_acc"] * valid_n
            pair_a2i_sum += pair_logs["pair_a2i_acc"] * valid_n
            pair_n += valid_n

    if was_training:
        model.train()
    if n == 0:
        raise RuntimeError("validation loader produced no samples")
    metrics = {
        "img_mse": img_sum / n,
        "aud_mse": aud_sum / n,
        "acc": correct / n,
        "pair_i2a_acc": pair_i2a_sum / max(pair_n, 1.0),
        "pair_a2i_acc": pair_a2i_sum / max(pair_n, 1.0),
        "n": float(n),
    }
    score_cfg = validation.get("score", {}) or {}
    pair_acc = 0.5 * (
        metrics["pair_i2a_acc"] + metrics["pair_a2i_acc"])
    metrics["score"] = (
        float(score_cfg.get("lambda_img", 1.0)) * metrics["img_mse"]
        + float(score_cfg.get("lambda_aud", 4.0)) * metrics["aud_mse"]
        + float(score_cfg.get("lambda_pair", 0.5)) * (1.0 - pair_acc)
        + float(score_cfg.get("lambda_cls", 0.5)) * (1.0 - metrics["acc"])
    )
    return metrics


def main():
    fix_console_encoding()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v12a.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start_epoch", type=int, default=None)
    ap.add_argument("--skip_decoder_pretrain", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    frozen = bool(cfg["train"].get("freeze_base", False))
    if cfg["train"].get("evaluation_only", False):
        ap.error("This is a frozen reference; use evaluate.py, not train.py")
    if cfg["train"].get("require_cuda", False) and (
            not torch.cuda.is_available() or not str(cfg["device"]).startswith("cuda")):
        raise RuntimeError("v12a requires CUDA for training; CPU fallback is disabled")
    if frozen:
        required = cfg["train"]["ckpt_path" if args.resume else "init_ckpt_path"]
        if not required or not resolve_from_root(required).is_file():
            raise FileNotFoundError(f"Required {'resume' if args.resume else 'parent'} checkpoint: {required}")
        if cfg.get("decoder_pretrain", {}).get("enabled", False):
            raise ValueError("Frozen-base experiment cannot pretrain base decoders")
    cfg["_config_path"] = args.config
    ensure_output_dirs(cfg)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available()
                          or cfg["device"] == "cpu" else "cpu")

    log(f"[启动] 设备: {device}")
    log("[启动] 加载训练集…")
    validation_cfg = cfg.get("validation", {}) or {}
    train_loader, val_loader = build_loaders(
        cfg, eval_split=validation_cfg.get("split", "val"))
    steps_per_epoch = len(train_loader)

    proto_img = train_loader.dataset.prototype_img.to(device)
    proto_aud = train_loader.dataset.prototype_aud.to(device)
    total_epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    real = train_loader.dataset.use_real_audio
    dataset_kind = cfg.get("data", {}).get("dataset", "mnist_fsdd")
    cc = cfg["corruption"]
    dc = cfg.get("detail_conditioning", {})
    xc = cfg.get("cross_key_conditioning", {})
    xdc = cfg.get("cross_detail_conditioning", {})
    pairing = cfg.get("data", {}).get("pairing", {}) or {}
    log(f"[启动] 训练集 {len(train_loader.dataset)} 样本，"
        f"每 epoch {steps_per_epoch} step，共 {total_epochs} epoch")
    audio_source = ("real paired manifest" if dataset_kind == "paired_manifest"
                    else ("FSDD+log-mel" if real else "toy"))
    log(f"[启动] 音频: {audio_source}  "
        f"enc={cfg['snn'].get('aud_encoder', 'conv')}  "
        f"N_index={cfg['dims']['N_index']} k_wta={cfg['index']['k_wta']}  "
        f"index_schedule={cfg['index'].get('input_schedule', 'simultaneous')}  "
        f"detail_cond={dc.get('enabled', False)}  "
        f"detail_detach={dc.get('detach', True)}  "
        f"cross_key={xc.get('enabled', False)} "
        f"cross_modules={xc.get('enabled', False) or xc.get('build_modules', False)} "
        f"cross_lr_mult={xc.get('lr_mult', 1.0)}  "
        f"cross_detail={xdc.get('enabled', False)} "
        f"pair_alignment={cfg.get('pair_alignment', {}).get('enabled', False)} "
        f"pairing={pairing.get('mode', 'legacy_random')}  "
        f"curriculum={cc.get('curriculum_mode', 'fixed')}  "
        f"binding={cfg['ablation']['use_binding_phase'] and not frozen} "
        f"freeze_base={frozen}")

    model = CrossModalSNN(cfg).to(device)
    init_state = None
    init_source = None
    init_ckpt = cfg["train"].get("init_ckpt_path", "")
    if init_ckpt and not args.resume:
        init_ckpt = str(resolve_from_root(init_ckpt))
        if os.path.isfile(init_ckpt):
            expected_parent_sha = cfg["train"].get("parent_sha256", "")
            if expected_parent_sha and file_sha256(init_ckpt) != expected_parent_sha:
                raise RuntimeError(
                    f"[init] parent checkpoint SHA256 mismatch: {init_ckpt}")
            state = torch.load(init_ckpt, map_location=device)
            if frozen:
                load_frozen_parent(model, state, init_ckpt)
            state_dict = state.get("model", state)
            strict = cfg["train"].get("init_strict", True)
            incompatible = model.load_state_dict(state_dict, strict=strict)
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            allowed_missing = tuple(
                cfg["train"].get("init_allowed_missing_prefixes", []))
            allowed_unexpected = tuple(
                cfg["train"].get("init_allowed_unexpected_prefixes", []))
            bad_missing = [
                key for key in missing
                if not allowed_missing or not key.startswith(allowed_missing)]
            bad_unexpected = [
                key for key in unexpected
                if not allowed_unexpected
                or not key.startswith(allowed_unexpected)]
            if bad_missing or bad_unexpected:
                raise RuntimeError(
                    "[init] incompatible parent checkpoint: "
                    f"missing={bad_missing} unexpected={bad_unexpected}; "
                    f"allowed_missing={allowed_missing} "
                    f"allowed_unexpected={allowed_unexpected}")
            init_state = state if isinstance(state, dict) else None
            init_source = init_ckpt
            log(f"[init] loaded weights from {init_ckpt} strict={strict} "
                f"missing={len(missing)} unexpected={len(unexpected)}")
        else:
            message = f"[init] init_ckpt_path not found: {init_ckpt}"
            if cfg["train"].get("init_required", False):
                raise FileNotFoundError(message)
            log(f"{message}; training from scratch.")

    if args.skip_decoder_pretrain:
        log("[decoder-pretrain] skipped by --skip_decoder_pretrain")
    elif args.resume:
        log("[decoder-pretrain] skipped because --resume was specified")
    else:
        pretrain_decoders(model, train_loader, cfg, device)

    trainable = _apply_trainable_prefixes(model, cfg)
    if trainable is not None:
        log(f"[启动] trainable_prefixes matched {len(trainable)} tensors")
    opt = _build_train_optimizer(model, cfg)

    sched_name = cfg["train"].get("lr_scheduler", "none")
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_epochs,
            eta_min=cfg["train"].get("lr_min", 0.0))
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=cfg["train"].get("lr_step_size", 15),
            gamma=cfg["train"].get("lr_gamma", 0.5))
    else:
        scheduler = None
    log(f"[启动] LR 调度: {sched_name}  初始 lr={cfg['train']['lr']}")
    log(f"[启动] grad_clip={cfg['train'].get('grad_clip', 0.0)}")

    start_epoch = 0
    if init_source is not None:
        configured_start = cfg["train"].get("start_epoch")
        saved_start = None
        if init_state is not None and "epoch" in init_state:
            saved_start = int(init_state["epoch"]) + 1
        if cfg["train"].get("init_load_optimizer", False):
            if init_state is None or "opt" not in init_state:
                raise RuntimeError(
                    "train.init_load_optimizer=true but parent checkpoint "
                    "does not contain optimizer state")
            opt.load_state_dict(init_state["opt"])
            parent_sched = init_state.get("sched")
            if parent_sched is not None:
                if scheduler is None:
                    raise RuntimeError(
                        "parent checkpoint has scheduler state but child "
                        "configuration disables the scheduler")
                scheduler.load_state_dict(parent_sched)
            if saved_start is None:
                raise RuntimeError(
                    "optimizer continuation requires parent epoch metadata")
            start_epoch = saved_start
            if (configured_start is not None
                    and int(configured_start) != start_epoch):
                raise RuntimeError(
                    "train.start_epoch does not match parent checkpoint: "
                    f"configured={configured_start} parent={start_epoch}")
            log(f"[init] restored optimizer/scheduler; start_epoch={start_epoch}")
        else:
            if configured_start is not None:
                start_epoch = int(configured_start)
            elif saved_start is not None:
                start_epoch = saved_start
            log(f"[init] model-only branch; start_epoch={start_epoch}")

    ckpt = str(resolve_from_root(cfg["train"]["ckpt_path"]))
    best_ckpt = str(resolve_from_root(
        cfg["train"].get("best_ckpt_path", ckpt)))
    best_validation_score = float("inf")
    if args.resume and os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        if frozen:
            verify_frozen_resume(model, state)
        if "opt" in state:
            opt.load_state_dict(state["opt"])
        if scheduler is not None and state.get("sched") is not None:
            scheduler.load_state_dict(state["sched"])
        if args.start_epoch is not None:
            start_epoch = args.start_epoch
        elif "epoch" in state:
            start_epoch = int(state["epoch"]) + 1
        else:
            log("[警告] checkpoint 为旧格式（无 epoch 字段），请指定 --start_epoch。")
        best_validation_score = float(
            state.get("best_validation_score", float("inf")))
        log(f"[恢复] 从 {ckpt} 继续，起始 epoch={start_epoch}/{total_epochs - 1}")
    elif args.resume:
        log(f"[警告] --resume 指定但 checkpoint 不存在: {ckpt}，从头训练。")

    verify_audio_normalization(model, cfg)
    log_every = cfg["train"]["log_every"]
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    milestone_epochs = {
        int(value) for value in cfg["train"].get(
            "save_milestone_epochs", [])
    }
    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0
        log(f"[epoch {epoch}/{total_epochs - 1}] 开始 ({steps_per_epoch} steps)")
        for step, batch in enumerate(train_loader):
            x_img, x_aud, labels, pair_ids = unpack_paired_batch(batch)
            x_img = x_img.to(device)
            x_aud = x_aud.to(device)
            labels = labels.to(device)
            if pair_ids is not None:
                pair_ids = pair_ids.to(device)

            cue_mode = sample_cue_mode(cfg)
            loss, logs = compute_losses(
                model, x_img, x_aud, labels, cue_mode, cfg,
                proto_img, proto_aud, pair_ids=pair_ids,
                epoch=epoch, step=step)

            opt.zero_grad()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"[train] non-finite loss at epoch={epoch} "
                    f"step={step} cue={cue_mode}: {logs}")
            if loss.requires_grad:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        grad_clip, error_if_nonfinite=True)
                opt.step()
            elif not frozen:
                raise RuntimeError("Training loss has no gradient path")
            logs["adapter_step"] = float(loss.requires_grad)

            epoch_loss += loss.item()

            if step % log_every == 0 or step == steps_per_epoch - 1:
                parts = " ".join(
                    f"{k}={v:.4f}" if k != "sev" else f"{k}={v:.2f}"
                    for k, v in logs.items())
                log(f"epoch {epoch} step {step}/{steps_per_epoch - 1} "
                    f"cue={cue_mode} loss={loss.item():.4f} | {parts}")

        cur_lr = opt.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / max(steps_per_epoch, 1)
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        completed_epochs = epoch + 1
        validation_metrics = None
        validate_every = int(validation_cfg.get("every_epochs", 1))
        if (validation_cfg.get("enabled", False)
                and validate_every > 0
                and completed_epochs % validate_every == 0):
            validation_metrics = _validate_model(
                model, val_loader, cfg, device)
            log(
                f"[val epoch {epoch}] score={validation_metrics['score']:.6f} "
                f"acc={validation_metrics['acc']*100:.1f}% "
                f"img_mse={validation_metrics['img_mse']:.6f} "
                f"aud_mse={validation_metrics['aud_mse']:.6f} "
                f"pair_i2a={validation_metrics['pair_i2a_acc']*100:.1f}% "
                f"pair_a2i={validation_metrics['pair_a2i_acc']*100:.1f}%")
        is_best = (validation_metrics is not None
                   and validation_metrics["score"] < best_validation_score)
        if is_best:
            best_validation_score = validation_metrics["score"]
        payload = _checkpoint_payload(model, opt, scheduler, cfg, epoch)
        payload["best_validation_score"] = best_validation_score
        if validation_metrics is not None:
            payload["validation"] = validation_metrics
        torch.save(payload, ckpt)
        if is_best:
            os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)
            torch.save(payload, best_ckpt)
            log(f"[val epoch {epoch}] best checkpoint 已保存 -> {best_ckpt}")
        if completed_epochs in milestone_epochs:
            milestone_path = _milestone_checkpoint_path(
                ckpt, completed_epochs)
            torch.save(payload, milestone_path)
            log(f"[epoch {epoch}] milestone 已保存 -> {milestone_path}")
        log(f"[epoch {epoch}] 平均 loss={avg_loss:.4f}  lr={cur_lr:.6f}  "
            f"checkpoint 已保存 -> {ckpt}")


if __name__ == "__main__":
    main()
