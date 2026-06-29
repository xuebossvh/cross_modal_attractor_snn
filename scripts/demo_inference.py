"""跨模态 SNN 联想记忆推理 demo。

输出三张图（外行可读，默认写入 outputs/outputs_v5/figures/）：
  demo_aud_only.png  — 只输入残缺语音
  demo_img_only.png  — 只输入残缺图像
  demo_both.png      — 双模态残缺输入

用法：
    python -u scripts/demo_inference.py --num 8 --severity 0.5
"""

import bootstrap  # noqa: F401

import argparse

import torch
import torch.nn.functional as F

from paths import (ensure_output_dirs, resolve_from_root,
                   figures_dir, tables_dir)
from common import (fix_console_encoding, log, load_config, select_targets,
                    setup_matplotlib_chinese, batch_ssim, format_table_row,
                    aud_collapse_stats)
from data.corruption import corrupt_audio, corrupt_image, AUD_MODES, IMG_MODES
from data.dataset import build_loaders
from models.network import CrossModalSNN


def _clean_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)


def _draw_footnote(ax, foot):
    """在专用标签 axes 内水平/垂直居中绘制脚注。"""
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if isinstance(foot, tuple) and foot[0] == "wrong":
        _, is_mel, pred, correct = foot
        prefix = "audio=" if is_mel else "digit="
        box = HPacker(
            children=[
                TextArea(prefix, textprops=dict(color="#CC0000", fontsize=10)),
                TextArea(str(int(pred)), textprops=dict(color="#CC0000", fontsize=10)),
                TextArea(f"({int(correct)})", textprops=dict(color="#333333", fontsize=10)),
            ],
            align="center", pad=0, sep=0,
        )
        ab = AnchoredOffsetbox(
            loc="center", child=box, frameon=False,
            bbox_to_anchor=(0.5, 0.5), bbox_transform=ax.transAxes,
            borderpad=0, pad=0,
        )
        ax.add_artist(ab)
    else:
        ax.text(0.5, 0.5, foot, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#333333")


def _display_audio(rec_aud, i, aud_kind, pred, proto_aud):
    """demo 展示：仅 image-only（aud=category）走类别原型；sample 必须显示 decoder 输出。"""
    if aud_kind == "category":
        return proto_aud[int(pred[i])]
    return rec_aud[i]


def _render_image(ax, tensor, is_mel, title=None):
    """单元格只画图像（占满 axes），脚注由下方独立 label 行绘制。"""
    _clean_ax(ax)
    if title:
        ax.set_title(title, fontsize=10, pad=3, loc="center")
    if is_mel:
        t = tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
        vmax = max(float(t.max()), 0.1)
        ax.imshow(t, cmap="magma", aspect="equal", origin="lower", vmin=0, vmax=vmax)
    else:
        t = tensor[0] if tensor.dim() == 3 else tensor
        ax.imshow(t, cmap="gray", aspect="equal")


def _foot(is_mel, y):
    y = int(y)
    return f"audio={y}" if is_mel else f"digit={y}"


# kind ∈ {"sample","category"} -> 评估表标注
_KIND_TAG = {"sample": "sample", "category": "category"}


def _recovered_foot(is_mel, i, labels, pred):
    """恢复列脚注：正确 digit=N/audio=N；错误时 ('wrong', is_mel, 预测, 真值)。"""
    y = int(labels[i])
    p = int(pred[i])
    if p == y:
        return _foot(is_mel, y)
    return ("wrong", is_mel, p, y)


def _rec_col_title(is_mel, kind):
    """恢复列标题：类别级 target 用 recovered category image/audio。"""
    if kind == "category":
        return "recovered category audio" if is_mel else "recovered category image"
    return "recovered audio" if is_mel else "recovered image"


def _build_columns(input_specs, rec_img, rec_aud, tgt_img, tgt_aud, labels, pred,
                   img_kind="sample", aud_kind="sample", retrieval_img=None,
                   proto_aud=None):
    """input_specs: [(tensor, is_mel, col_title), ...]"""
    cols = []
    for tensor, is_mel, title in input_specs:
        cols.append({
            "title": title,
            "is_mel": is_mel,
            "data": lambda i, t=tensor: t[i],
            "foot_fn": lambda i, m=is_mel, y=labels: _foot(m, y[i]),
        })
    cols.append({
        "title": _rec_col_title(False, img_kind),
        "is_mel": False,
        "data": lambda i, r=rec_img: r[i, 0],
        "foot_fn": lambda i, y=labels, p=pred: _recovered_foot(False, i, y, p),
    })
    if retrieval_img is not None:
        cols.append({
            "title": "retrieved category image",
            "is_mel": False,
            "data": lambda i, r=retrieval_img: r[i, 0],
            "foot_fn": lambda i, y=labels, p=pred: _recovered_foot(False, i, y, p),
        })
    pa = proto_aud
    cols.extend([
        {
            "title": _rec_col_title(True, aud_kind),
            "is_mel": True,
            "data": lambda i, r=rec_aud, k=aud_kind, p=pred, pa=pa: (
                _display_audio(r, i, k, p, pa) if pa is not None else r[i]),
            "foot_fn": lambda i, y=labels, p=pred: _recovered_foot(True, i, y, p),
        },
        {
            "title": "target image",
            "is_mel": False,
            "data": lambda i, c=tgt_img: c[i, 0],
            "foot_fn": lambda i, y=labels: _foot(False, y[i]),
        },
        {
            "title": "target audio",
            "is_mel": True,
            "data": lambda i, c=tgt_aud: c[i],
            "foot_fn": lambda i, y=labels: _foot(True, y[i]),
        },
    ])
    return cols


def _plot_demo(k, labels, tgt_img, tgt_aud, pred, out_path, suptitle,
               input_specs, outputs, img_kind="sample", aud_kind="sample",
               retrieval_img=None):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    rec_img = torch.sigmoid(outputs["img"]).cpu()
    rec_aud = outputs["aud"].cpu()
    tgt_img = tgt_img.cpu()
    tgt_aud = tgt_aud.cpu()
    cols = _build_columns(input_specs, rec_img, rec_aud, tgt_img, tgt_aud,
                          labels, pred, img_kind, aud_kind, retrieval_img,
                          proto_aud=outputs.get("proto_aud"))
    ncols = len(cols)

    # 布局：每样本占「大图行 + 标签行」；标签行 = 上下两图之间的间隙
    img_h, lbl_h = 5.0, 0.55
    height_ratios = []
    for _ in range(k):
        height_ratios.extend([img_h, lbl_h])

    col_w = 2.0
    fig_h = k * (img_h + lbl_h) * 0.38 + 0.6
    fig = plt.figure(figsize=(col_w * ncols, fig_h), facecolor="white")
    gs = GridSpec(2 * k, ncols, figure=fig,
                  height_ratios=height_ratios, hspace=0.08, wspace=0.02)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98)

    for i in range(k):
        for j, spec in enumerate(cols):
            ax_img = fig.add_subplot(gs[2 * i, j])
            col_title = spec["title"] if i == 0 else None
            _render_image(ax_img, spec["data"](i), spec["is_mel"], col_title)

            ax_lbl = fig.add_subplot(gs[2 * i + 1, j])
            _draw_footnote(ax_lbl, spec["foot_fn"](i))

    _save_demo_figure(fig, out_path)


def _save_demo_figure(fig, path):
    fig.savefig(path, dpi=140, facecolor="white")
    log(f"[demo] -> {path}")


def _cue_quality(clean, corrupt):
    diff = F.mse_loss(corrupt, clean).item()
    e_clean = clean.abs().sum().item() + 1e-8
    keep = corrupt.abs().sum().item() / e_clean
    return diff, keep


def _make_visible_cue(clean, corrupt_fn, prefer_modes, min_mse=0.003,
                      min_keep=0.30, max_severity=0.5):
    """逐样本生成可见但不全空的残缺 cue。

    severity 扫描上限对齐评估/训练口径（max_severity，默认 0.5），
    使 demo 表与 evaluate 的腐蚀强度可比。
    """
    sweep = tuple(s for s in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
                  if s <= max_severity + 1e-6) or (max_severity,)
    out = clean.clone()
    for i in range(clean.size(0)):
        sample = clean[i:i + 1]
        chosen, fallback, fallback_keep = None, None, -1.0
        for mode in prefer_modes:
            for s in sweep:
                cand = corrupt_fn(sample, mode=mode, severity=s)[0]
                diff, keep = _cue_quality(sample, cand.unsqueeze(0))
                if diff >= min_mse and keep >= min_keep:
                    chosen = cand
                    break
                if diff >= min_mse and keep > fallback_keep:
                    fallback, fallback_keep = cand, keep
            if chosen is not None:
                break
        out[i] = chosen if chosen is not None else (
            fallback if fallback is not None else sample[0])
    return out


def _make_visible_aud_cue(clean, prefer_mode="time_mask", max_severity=0.5):
    modes = [m for m in AUD_MODES if m != "gaussian"]
    if prefer_mode in modes:
        modes = [prefer_mode] + [m for m in modes if m != prefer_mode]
    return _make_visible_cue(clean, corrupt_audio, modes,
                             max_severity=max_severity)


def _make_visible_img_cue(clean, prefer_mode="occlusion", max_severity=0.5):
    modes = [m for m in IMG_MODES if m != "gaussian"]
    if prefer_mode in modes:
        modes = [prefer_mode] + [m for m in modes if m != prefer_mode]
    return _make_visible_cue(clean, corrupt_image, modes,
                             max_severity=max_severity)


def _pred_conf(logits):
    """返回 (pred[B], confidence[B]) ，confidence = softmax 最大概率。"""
    prob = torch.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    return pred.cpu(), conf.cpu()


def _mode_metrics(rec_img_logits, rec_aud, tgt_img, tgt_aud, labels, pred):
    """单 cue 模式下的分类 / 图像 / 音频指标。"""
    rec_img = torch.sigmoid(rec_img_logits).cpu()
    rec_aud = rec_aud.cpu()
    tgt_img = tgt_img.cpu()
    tgt_aud = tgt_aud.cpu()
    acc = (pred == labels).float().mean().item()
    img_mse = F.mse_loss(rec_img, tgt_img).item()
    img_ssim = batch_ssim(rec_img, tgt_img).item()
    aud_mse = F.mse_loss(rec_aud, tgt_aud).item()
    aud_ssim = batch_ssim(rec_aud.unsqueeze(1), tgt_aud.unsqueeze(1)).item()
    return {
        "acc": acc, "img_mse": img_mse, "img_ssim": img_ssim,
        "aud_mse": aud_mse, "aud_ssim": aud_ssim,
    }


def _format_eval_table(rows, k):
    """生成评估表文本（汇总 + 逐样本），中英文混排按显示宽度对齐。"""
    sum_w = [14, 10, 10, 10, 10, 10, 12, 12]
    sum_a = ["l", "r", "r", "r", "r", "r", "r", "r"]
    sum_hdr = ["模式", "分类ACC", "图像SSIM", "图像MSE", "音频SSIM", "音频MSE",
               "img目标", "aud目标"]
    sep = "-" * sum(sum_w)

    lines = [
        "跨模态 SNN Demo 评估表",
        f"样本数: {k}",
        "",
        "【汇总】",
        format_table_row(sum_hdr, sum_w, sum_a),
        sep,
    ]
    for r in rows:
        lines.append(format_table_row([
            r["mode"], f"{r['acc']*100:.1f}%",
            f"{r['img_ssim']:.3f}", f"{r['img_mse']:.4f}",
            f"{r['aud_ssim']:.3f}", f"{r['aud_mse']:.4f}",
            r["img_tgt"], r["aud_tgt"],
        ], sum_w, sum_a))

    samp_w = [3, 5, 12, 12, 12]
    samp_a = ["r", "r", "r", "r", "r"]
    lines.extend(["", "【逐样本分类】",
                  format_table_row(["#", "真值", "audio-only", "image-only", "img+aud"],
                                   samp_w, samp_a)])
    for i in range(k):

        def _cell(p, y):
            mark = "✓" if int(p) == int(y) else "✗"
            return f"{int(p)}{mark}"

        lines.append(format_table_row([
            str(i), str(int(rows[0]["labels"][i])),
            _cell(rows[0]["pred"][i], rows[0]["labels"][i]),
            _cell(rows[1]["pred"][i], rows[1]["labels"][i]),
            _cell(rows[2]["pred"][i], rows[2]["labels"][i]),
        ], samp_w, samp_a))
    return "\n".join(lines)


def _log_demo_audio_diag(rows):
    """音频塌缩诊断：rec/target mean/std/max + top15% 能量召回（近黑图一眼可辨）。"""
    dw = [14, 9, 9, 9, 9, 9, 9, 10]
    da = ["l", "r", "r", "r", "r", "r", "r", "r"]
    hdr = ["模式", "rec均值", "rec标准差", "rec最大",
           "tgt均值", "tgt标准差", "tgt最大", "top15%召回"]
    log("")
    log("【音频塌缩诊断】recovered_aud vs target_aud")
    log(format_table_row(hdr, dw, da))
    log("-" * sum(dw))
    for name, rec, tgt in rows:
        d = aud_collapse_stats(rec, tgt)
        log(format_table_row([
            name,
            f"{d['rec_mean']:.4f}", f"{d['rec_std']:.4f}", f"{d['rec_max']:.4f}",
            f"{d['tgt_mean']:.4f}", f"{d['tgt_std']:.4f}", f"{d['tgt_max']:.4f}",
            f"{d['topk_recall']*100:.1f}%",
        ], dw, da))


def _save_eval_table(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    log(f"[demo] 评估表 -> {path}")


def _plot_aud_only(k, labels, aud_cue, rec_img, rec_aud, tgt_img, tgt_aud,
                   pred, img_kind, aud_kind, out_path, retrieval_img=None,
                   proto_aud=None):
    _plot_demo(
        k, labels, tgt_img, tgt_aud, pred, out_path,
        suptitle="corrupted audio → recovered image & audio",
        input_specs=[(aud_cue, True, "corrupted audio")],
        outputs={"img": rec_img, "aud": rec_aud, "proto_aud": proto_aud},
        img_kind=img_kind, aud_kind=aud_kind,
        retrieval_img=retrieval_img,
    )


def _plot_img_only(k, labels, img_cue, rec_img, rec_aud, tgt_img, tgt_aud,
                   pred, img_kind, aud_kind, out_path, proto_aud=None):
    _plot_demo(
        k, labels, tgt_img, tgt_aud, pred, out_path,
        suptitle="corrupted image → recovered image & audio",
        input_specs=[(img_cue, False, "corrupted image")],
        outputs={"img": rec_img, "aud": rec_aud, "proto_aud": proto_aud},
        img_kind=img_kind, aud_kind=aud_kind,
    )


def _plot_both(k, labels, img_cue, aud_cue, rec_img, rec_aud, tgt_img, tgt_aud,
               pred, img_kind, aud_kind, out_path, proto_aud=None):
    _plot_demo(
        k, labels, tgt_img, tgt_aud, pred, out_path,
        suptitle="corrupted image & audio → recovered image & audio",
        input_specs=[
            (img_cue, False, "corrupted image"),
            (aud_cue, True, "corrupted audio"),
        ],
        outputs={"img": rec_img, "aud": rec_aud, "proto_aud": proto_aud},
        img_kind=img_kind, aud_kind=aud_kind,
    )


def main():
    fix_console_encoding()
    setup_matplotlib_chinese()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v6c.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--num", type=int, default=8, help="可视化样本数（默认 8）")
    ap.add_argument("--severity", type=float, default=0.5)
    ap.add_argument("--aud_corrupt_mode", default="time_mask")
    ap.add_argument("--img_corrupt_mode", default="occlusion")
    ap.add_argument("--out_aud", default=None)
    ap.add_argument("--out_img", default=None)
    ap.add_argument("--out_both", default=None)
    ap.add_argument("--eval_table", default=None, help="评估表输出路径")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config
    ensure_output_dirs(cfg)
    fig_dir = figures_dir(cfg)
    tbl_dir = tables_dir(cfg)
    args.out_aud = args.out_aud or str(fig_dir / "demo_aud_only.png")
    args.out_img = args.out_img or str(fig_dir / "demo_img_only.png")
    args.out_both = args.out_both or str(fig_dir / "demo_both.png")
    args.eval_table = args.eval_table or str(tbl_dir / "demo_eval_table.txt")
    device = torch.device("cuda" if (cfg["device"] == "cuda"
                          and torch.cuda.is_available()) else "cpu")
    ckpt_path = str(resolve_from_root(args.ckpt or cfg["train"]["ckpt_path"]))

    log(f"[demo] 设备: {device}  加载 checkpoint: {ckpt_path}")
    model = CrossModalSNN(cfg).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    except FileNotFoundError:
        log(f"[警告] 未找到 {ckpt_path}，使用随机权重做 demo。")
    except RuntimeError as e:
        log(f"[警告] checkpoint 结构不匹配，使用随机权重做 demo。\n  {e}")
    model.eval()

    _, test_loader = build_loaders(cfg)
    # 类别代表原型（class medoid），单模态 cue 的跨模态类别级 target 来源
    proto_img = test_loader.dataset.prototype_img.to(device)
    proto_aud = test_loader.dataset.prototype_aud.to(device)

    x_img, x_aud, labels = next(iter(test_loader))
    k = min(args.num, x_img.size(0))
    x_img = x_img[:k].to(device)
    x_aud = x_aud[:k].to(device)
    labels = labels[:k].to(device)

    log(f"[demo] 可视化 {k} 个样本")

    img_cue = _make_visible_img_cue(x_img, prefer_mode=args.img_corrupt_mode,
                                    max_severity=args.severity)
    aud_cue = _make_visible_aud_cue(x_aud, prefer_mode=args.aud_corrupt_mode,
                                    max_severity=args.severity)

    with torch.no_grad():
        out_aud = model(x_aud_cue=aud_cue, training_mode=False)
        out_img = model(x_img_cue=img_cue, training_mode=False)
        out_both = model(x_img_cue=img_cue, x_aud_cue=aud_cue, training_mode=False)

    # 按 cue 模式选择 target（展示列与 loss 评估一致）
    tgt_img_a, tgt_aud_a, img_k_a, aud_k_a = select_targets(
        "clean_aud_only", x_img, x_aud, proto_img, proto_aud, labels)
    tgt_img_i, tgt_aud_i, img_k_i, aud_k_i = select_targets(
        "clean_img_only", x_img, x_aud, proto_img, proto_aud, labels)
    tgt_img_b, tgt_aud_b, img_k_b, aud_k_b = select_targets(
        "clean_both", x_img, x_aud, proto_img, proto_aud, labels)

    pred_a, _ = _pred_conf(out_aud["logits"])
    pred_i, _ = _pred_conf(out_img["logits"])
    pred_b, _ = _pred_conf(out_both["logits"])
    labels_cpu = labels.cpu()

    rec_aud_a = out_aud["recovered_aud"].cpu()
    rec_aud_i = out_img["recovered_aud"].cpu()
    rec_aud_b = out_both["recovered_aud"].cpu()

    m_a = _mode_metrics(out_aud["recovered_img"], rec_aud_a,
                        tgt_img_a.cpu(), tgt_aud_a.cpu(), labels_cpu, pred_a)
    m_i = _mode_metrics(out_img["recovered_img"], rec_aud_i,
                        tgt_img_i.cpu(), tgt_aud_i.cpu(), labels_cpu, pred_i)
    m_b = _mode_metrics(out_both["recovered_img"], rec_aud_b,
                        tgt_img_b.cpu(), tgt_aud_b.cpu(), labels_cpu, pred_b)

    eval_rows = [
        {"mode": "audio-only", "labels": labels_cpu, "pred": pred_a,
         "img_tgt": _KIND_TAG[img_k_a], "aud_tgt": _KIND_TAG[aud_k_a], **m_a},
        {"mode": "image-only", "labels": labels_cpu, "pred": pred_i,
         "img_tgt": _KIND_TAG[img_k_i], "aud_tgt": _KIND_TAG[aud_k_i], **m_i},
        {"mode": "image+audio", "labels": labels_cpu, "pred": pred_b,
         "img_tgt": _KIND_TAG[img_k_b], "aud_tgt": _KIND_TAG[aud_k_b], **m_b},
    ]
    table_text = _format_eval_table(eval_rows, k)
    log(table_text)
    _save_eval_table(table_text, args.eval_table)

    _log_demo_audio_diag([
        ("audio-only", rec_aud_a, tgt_aud_a.cpu()),
        ("image-only", rec_aud_i, tgt_aud_i.cpu()),
        ("image+audio", rec_aud_b, tgt_aud_b.cpu()),
    ])

    img_cue_np = img_cue.cpu()
    aud_cue_np = aud_cue.cpu()

    # audio-only 类别图像：按预测标签检索类别原型（联想记忆按地址取内容）
    ret_img_a = proto_img[pred_a].cpu()
    proto_aud_cpu = proto_aud.cpu()

    _plot_aud_only(k, labels_cpu, aud_cue_np,
                   out_aud["recovered_img"], out_aud["recovered_aud"],
                   tgt_img_a.cpu(), tgt_aud_a.cpu(), pred_a,
                   img_k_a, aud_k_a, args.out_aud, retrieval_img=ret_img_a,
                   proto_aud=proto_aud_cpu)
    _plot_img_only(k, labels_cpu, img_cue_np,
                   out_img["recovered_img"], out_img["recovered_aud"],
                   tgt_img_i.cpu(), tgt_aud_i.cpu(), pred_i,
                   img_k_i, aud_k_i, args.out_img, proto_aud=proto_aud_cpu)
    _plot_both(k, labels_cpu, img_cue_np, aud_cue_np,
               out_both["recovered_img"], out_both["recovered_aud"],
               tgt_img_b.cpu(), tgt_aud_b.cpu(), pred_b,
               img_k_b, aud_k_b, args.out_both, proto_aud=proto_aud_cpu)


if __name__ == "__main__":
    main()
