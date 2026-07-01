"""训练跨模态 SNN 联想记忆网络（binding + readout 两阶段）。

用法（在项目根目录）：
    python -u scripts/train.py --config configs/v6b.yaml
    python -u scripts/train.py --epochs 30
"""

import bootstrap  # noqa: F401

import argparse
import os

import torch
import torch.nn.functional as F

from common import (fix_console_encoding, log, load_config, set_seed,
                    sample_cue_mode, sample_train_severity, build_cue,
                    select_targets, is_aud_only_mode, spike_reg)
from paths import ensure_output_dirs, resolve_from_root
from data.dataset import build_loaders
from models.network import CrossModalSNN


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
    return loss


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


def compute_losses(model, clean_img, clean_aud, labels, cue_mode, cfg,
                   proto_img, proto_aud, epoch=0):
    """返回 (总损失, 日志字典)。"""
    lc = cfg["loss"]
    ab = cfg.get("ablation", {})
    use_binding = ab.get("use_binding_phase", True)

    severity = sample_train_severity(cfg, epoch)
    img_cue, aud_cue = build_cue(clean_img, clean_aud, cue_mode, cfg,
                                 severity=severity)
    tgt_img, tgt_aud, img_kind, aud_kind = select_targets(
        cue_mode, clean_img, clean_aud, proto_img, proto_aud, labels)

    total = 0.0
    logs = {"sev": severity}

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
    ap.add_argument("--config", default="configs/v6b.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start_epoch", type=int, default=None)
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
    log(f"[启动] 训练集 {len(train_loader.dataset)} 样本，"
        f"每 epoch {steps_per_epoch} step，共 {total_epochs} epoch")
    log(f"[启动] 音频: {'FSDD+log-mel' if real else 'toy'}  "
        f"enc={cfg['snn'].get('aud_encoder', 'conv')}  "
        f"N_index={cfg['dims']['N_index']} k_wta={cfg['index']['k_wta']}  "
        f"curriculum={cc.get('curriculum_mode', 'fixed')}  "
        f"binding={cfg['ablation']['use_binding_phase']}")

    model = CrossModalSNN(cfg).to(device)
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
