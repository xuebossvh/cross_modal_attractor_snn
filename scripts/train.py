"""训练跨模态 SNN 联想记忆网络（binding + readout 两阶段）。

用法（在项目根目录）：
    python -u scripts/train.py --config configs/v9a.yaml
    python -u scripts/train.py --epochs 30
"""

import bootstrap  # noqa: F401

import argparse
import os

import torch
import torch.nn.functional as F

from common import (fix_console_encoding, log, load_config, set_seed,
                    sample_cue_mode, sample_train_severity, build_cue,
                    select_targets, is_aud_only_mode, spike_reg,
                    resolve_train_corrupt_modes)
from paths import ensure_output_dirs, resolve_from_root
from data.dataset import build_loaders
from models.network import CrossModalSNN
from models.lif import rate


def _img_edge_loss(prob, target):
    """图像水平/垂直一阶差分 L1，锐化重建边缘。"""
    dx_p = prob[..., :, 1:] - prob[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = prob[..., 1:, :] - prob[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def _img_recon_loss(rec, x_img, lc):
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


def _aud_recon_loss(rec, target, lc):
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
    return loss


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


def _safe_target_value_state(value_layer, enc_spikes, delay=2, clamp=20.0):
    """Pretrain-only target Value rate without surrogate autograd kernels."""
    T, B, _ = enc_spikes.shape
    device, dtype = enc_spikes.device, enc_spikes.dtype
    v = torch.zeros((B, value_layer.n_value), device=device, dtype=dtype)
    out = []
    use_clamp = clamp is not None and float(clamp) > 0
    clamp = float(clamp) if use_clamp else 0.0
    for t in range(T):
        ts = t - delay
        if ts < 0:
            cur = torch.zeros((B, value_layer.n_value), device=device, dtype=dtype)
        else:
            cur = value_layer.W_enc_to_V(enc_spikes[ts])
        if use_clamp:
            cur = torch.nan_to_num(cur, nan=0.0, posinf=clamp, neginf=-clamp)
            cur = cur.clamp(-clamp, clamp)
        v = value_layer.neuron.beta * v + cur
        if use_clamp:
            v = torch.nan_to_num(v, nan=0.0, posinf=clamp, neginf=-clamp)
            v = v.clamp(-clamp, clamp)
        spikes = (v - value_layer.neuron.v_threshold >= 0).to(dtype)
        v = v * (1.0 - spikes)
        out.append(spikes)
    return rate(torch.stack(out, dim=0))


def _pretrain_decoder_states(model, x_img, x_aud, detail_dropout, target_mode,
                             value_clamp):
    spike_img = model.img_encoder(x_img)
    spike_aud = model.aud_encoder(model._normalize_audio_for_encoder(x_aud))

    delay = model.memory.binding_delay if model.memory.use_delayed_target else 0
    if target_mode == "safe_lif":
        img_state = _safe_target_value_state(
            model.memory.V_img, spike_img, delay=delay, clamp=value_clamp)
        aud_state = _safe_target_value_state(
            model.memory.V_aud, spike_aud, delay=delay, clamp=value_clamp)
    elif target_mode == "linear_sigmoid":
        img_state = torch.sigmoid(model.memory.V_img.W_enc_to_V(rate(spike_img)))
        aud_state = torch.sigmoid(model.memory.V_aud.W_enc_to_V(rate(spike_aud)))
    else:
        raise ValueError(f"Unknown decoder_pretrain.target_value_mode: {target_mode}")

    if model.use_detail_conditioning:
        img_detail = _drop_detail_state(rate(spike_img).detach(), detail_dropout)
        aud_detail = _drop_detail_state(rate(spike_aud).detach(), detail_dropout)
        img_state = torch.cat([img_state.detach(), img_detail], dim=1)
        aud_state = torch.cat([aud_state.detach(), aud_detail], dim=1)
    return img_state.detach(), aud_state.detach()


def _set_decoder_pretrain_requires_grad(model, freeze_non_decoders=True):
    previous = []
    decoder_prefixes = ("image_decoder.", "audio_decoder.")
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

    params = list(model.image_decoder.parameters()) + list(model.audio_decoder.parameters())
    opt = torch.optim.Adam(
        params,
        lr=pc.get("lr", cfg["train"]["lr"]),
        weight_decay=pc.get("weight_decay", 0.0),
    )
    lc = cfg["loss"]
    lam_img = pc.get("lambda_img", lc.get("lambda_img", 1.0))
    lam_aud = pc.get("lambda_aud", lc.get("lambda_aud", 1.0))
    lam_act = pc.get("lambda_aud_active", lc.get("lambda_aud_active", 0.0))
    detail_dropout = pc.get("detail_dropout", 0.0)
    target_mode = pc.get("target_value_mode", "safe_lif")
    value_clamp = pc.get("value_clamp", 20.0)
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
        model, pc.get("freeze_non_decoders", True))
    steps_per_epoch = len(train_loader)
    log("[decoder-pretrain] start "
        f"epochs={epochs} lr={pc.get('lr', cfg['train']['lr'])} "
        f"target_value_mode={target_mode} "
        f"detail_dropout={float(detail_dropout):.2f} "
        f"freeze_non_decoders={pc.get('freeze_non_decoders', True)}")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0
            for step, (x_img, x_aud, labels) in enumerate(train_loader):
                del labels
                x_img = x_img.to(device)
                x_aud = x_aud.to(device)

                with torch.no_grad():
                    img_state, aud_state = _pretrain_decoder_states(
                        model, x_img, x_aud, detail_dropout,
                        target_mode=target_mode, value_clamp=value_clamp)

                rec_img = model.image_decoder(img_state)
                rec_aud = model.audio_decoder(aud_state)

                loss_img = _img_recon_loss(rec_img, x_img, lc)
                loss_aud = _aud_recon_loss(rec_aud, x_aud, lc)
                loss = lam_img * loss_img + lam_aud * loss_aud
                logs = {
                    "img": loss_img.item(),
                    "aud": loss_aud.item(),
                }
                if lam_act > 0:
                    loss_act = _aud_active_loss(rec_aud, x_aud)
                    loss = loss + lam_act * loss_act
                    logs["aud_act"] = loss_act.item()

                opt.zero_grad()
                loss.backward()
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


def _mode_enabled(cue_mode, modes):
    return cue_mode in set(modes or [])


def _soft_cls_loss(student_logits, teacher_logits, temperature=2.0):
    t = max(float(temperature), 1e-6)
    log_p = F.log_softmax(student_logits / t, dim=1)
    q = F.softmax(teacher_logits.detach() / t, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (t * t)


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
            out_clean = model(x_img_cue=clean_img, x_aud_cue=clean_aud,
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


def compute_losses(model, clean_img, clean_aud, labels, cue_mode, cfg,
                   proto_img, proto_aud, epoch=0):
    """返回 (总损失, 日志字典)。"""
    lc = cfg["loss"]
    ab = cfg.get("ablation", {})
    use_binding = ab.get("use_binding_phase", True)

    severity = sample_train_severity(cfg, epoch)
    img_mode, aud_mode = resolve_train_corrupt_modes(cfg, epoch)
    img_cue, aud_cue = build_cue(clean_img, clean_aud, cue_mode, cfg,
                                 severity=severity,
                                 img_mode=img_mode, aud_mode=aud_mode)
    tgt_img, tgt_aud, img_kind, aud_kind = select_targets(
        cue_mode, clean_img, clean_aud, proto_img, proto_aud, labels)
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
                  training_mode=True, phase="readout")

    loss_align, align_logs = _alignment_losses(
        model, out_r, clean_img, clean_aud, labels, cue_mode, cfg)
    total = total + loss_align
    logs.update(align_logs)

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

    loss_img = _img_recon_loss(out_r["recovered_img"], tgt_img, lc)
    total = total + lc["lambda_img"] * loss_img
    logs[f"img({img_kind[:3]})"] = loss_img.item()

    loss_aud = _aud_recon_loss(out_r["recovered_aud"], tgt_aud, lc)
    total = total + lc["lambda_aud"] * loss_aud
    logs[f"aud({aud_kind[:3]})"] = loss_aud.item()

    if aud_kind == "sample":
        lam_act = lc.get("lambda_aud_active", 0.0)
        if lam_act > 0:
            loss_act = _aud_active_loss(out_r["recovered_aud"], tgt_aud)
            total = total + lam_act * loss_act
            logs["aud_act"] = loss_act.item()

    total = total + lc["lambda_reg"] * spike_reg(out_r)

    return total, logs


def main():
    fix_console_encoding()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v9a.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start_epoch", type=int, default=None)
    ap.add_argument("--skip_decoder_pretrain", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config
    ensure_output_dirs(cfg)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available()
                          or cfg["device"] == "cpu" else "cpu")

    log(f"[启动] 设备: {device}")
    log("[启动] 加载训练集…")
    train_loader, _ = build_loaders(cfg)
    steps_per_epoch = len(train_loader)

    proto_img = train_loader.dataset.prototype_img.to(device)
    proto_aud = train_loader.dataset.prototype_aud.to(device)
    total_epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    real = train_loader.dataset.use_real_audio
    cc = cfg["corruption"]
    dc = cfg.get("detail_conditioning", {})
    log(f"[启动] 训练集 {len(train_loader.dataset)} 样本，"
        f"每 epoch {steps_per_epoch} step，共 {total_epochs} epoch")
    log(f"[启动] 音频: {'FSDD+log-mel' if real else 'toy'}  "
        f"enc={cfg['snn'].get('aud_encoder', 'conv')}  "
        f"N_index={cfg['dims']['N_index']} k_wta={cfg['index']['k_wta']}  "
        f"index_schedule={cfg['index'].get('input_schedule', 'simultaneous')}  "
        f"detail_cond={dc.get('enabled', False)}  "
        f"detail_detach={dc.get('detach', True)}  "
        f"curriculum={cc.get('curriculum_mode', 'fixed')}  "
        f"binding={cfg['ablation']['use_binding_phase']}")

    model = CrossModalSNN(cfg).to(device)
    init_ckpt = cfg["train"].get("init_ckpt_path", "")
    if init_ckpt and not args.resume:
        init_ckpt = str(resolve_from_root(init_ckpt))
        if os.path.isfile(init_ckpt):
            state = torch.load(init_ckpt, map_location=device)
            state_dict = state.get("model", state)
            strict = cfg["train"].get("init_strict", False)
            incompatible = model.load_state_dict(state_dict, strict=strict)
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            log(f"[init] loaded weights from {init_ckpt} strict={strict} "
                f"missing={len(missing)} unexpected={len(unexpected)}")
        else:
            log(f"[init] init_ckpt_path not found: {init_ckpt}; training from scratch.")

    if args.skip_decoder_pretrain:
        log("[decoder-pretrain] skipped by --skip_decoder_pretrain")
    elif args.resume:
        log("[decoder-pretrain] skipped because --resume was specified")
    else:
        pretrain_decoders(model, train_loader, cfg, device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])

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

    start_epoch = 0
    ckpt = str(resolve_from_root(cfg["train"]["ckpt_path"]))
    if args.resume and os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
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
        log(f"[恢复] 从 {ckpt} 继续，起始 epoch={start_epoch}/{total_epochs - 1}")
    elif args.resume:
        log(f"[警告] --resume 指定但 checkpoint 不存在: {ckpt}，从头训练。")

    log_every = cfg["train"]["log_every"]
    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0
        log(f"[epoch {epoch}/{total_epochs - 1}] 开始 ({steps_per_epoch} steps)")
        for step, (x_img, x_aud, labels) in enumerate(train_loader):
            x_img = x_img.to(device)
            x_aud = x_aud.to(device)
            labels = labels.to(device)

            cue_mode = sample_cue_mode(cfg)
            loss, logs = compute_losses(
                model, x_img, x_aud, labels, cue_mode, cfg,
                proto_img, proto_aud, epoch=epoch)

            opt.zero_grad()
            loss.backward()
            opt.step()

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
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": scheduler.state_dict() if scheduler is not None else None,
            "cfg": cfg,
            "epoch": epoch,
        }, ckpt)
        log(f"[epoch {epoch}] 平均 loss={avg_loss:.4f}  lr={cur_lr:.6f}  "
            f"checkpoint 已保存 -> {ckpt}")


if __name__ == "__main__":
    main()
