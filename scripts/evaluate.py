"""评估跨模态 SNN 联想记忆网络。

对 6 种 cue 模式分别评估（推理时禁用 target，decoder 只读 v_*_from_A）：
    corrupt_img_only / corrupt_aud_only / corrupt_both
    clean_img_only   / clean_aud_only   / clean_both

指标：
    分类   accuracy
    图像   MSE / PSNR / SSIM（recovered_img vs clean_img）
    音频   MSE（recovered log-mel vs clean log-mel，[B,n_mels,n_frames]）
    多样性 像素方差 / 样本间 L2（检测是否塌缩成同一张图）

可选：--severity_curve 对 corrupt_* 模式扫描 severity，输出退化曲线。

用法：
    python -u scripts/evaluate.py --config configs/v4.yaml
    python -u scripts/evaluate.py --max_batches 20 --severity_curve
"""

import bootstrap  # noqa: F401

import argparse
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import (fix_console_encoding, log, load_config,
                    batch_ssim, batch_psnr, build_cue, select_targets,
                    batch_reconstruction_variance, format_table_row)
from paths import resolve_from_root
from data.dataset import build_loaders
from models.network import CrossModalSNN

EVAL_MODES = ["corrupt_img_only", "corrupt_aud_only", "corrupt_both",
              "clean_img_only", "clean_aud_only", "clean_both"]


@torch.no_grad()
def eval_mode(model, loader, cfg, mode, device, severity, proto_img, proto_aud,
              max_batches=None):
    """按 cue 模式对应的恢复粒度 target 计算指标。

    图像/音频指标均对照 select_targets 选出的 target（区分样本级/类别级）：
        audio-only : 图像 vs 类别代表原型      音频 vs 本样本 clean
        image-only : 图像 vs 本样本 clean       音频 vs 类别代表原型
        both       : 图像/音频均 vs 本样本 clean
    """
    model.eval()
    n = 0
    correct = 0
    sum_img_mse = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_aud_mse = 0.0
    nb = 0
    all_rec = []
    img_kind = aud_kind = "?"

    iterator = enumerate(loader)
    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    pbar = tqdm(iterator, total=total, desc=mode, unit="batch",
                file=sys.stdout, ascii=True)
    for bi, (x_img, x_aud, labels) in pbar:
        if max_batches is not None and bi >= max_batches:
            break
        x_img = x_img.to(device)
        x_aud = x_aud.to(device)
        labels = labels.to(device)

        img_cue, aud_cue = build_cue(x_img, x_aud, mode, cfg, severity=severity)
        tgt_img, tgt_aud, img_kind, aud_kind = select_targets(
            mode, x_img, x_aud, proto_img, proto_aud, labels)
        out = model(x_img_cue=img_cue, x_aud_cue=aud_cue,
                    training_mode=False, phase="readout")

        pred = out["logits"].argmax(dim=1)
        correct += (pred == labels).sum().item()
        n += labels.size(0)

        rec_img = torch.sigmoid(out["recovered_img"])
        sum_img_mse += F.mse_loss(rec_img, tgt_img).item()
        sum_psnr += batch_psnr(rec_img, tgt_img).item()
        sum_ssim += batch_ssim(rec_img, tgt_img).item()

        rec_aud = out["recovered_aud"]
        sum_aud_mse += F.mse_loss(rec_aud, tgt_aud).item()   # log-mel [B,M,T]

        all_rec.append(rec_img.cpu())
        nb += 1

    acc = correct / max(n, 1)
    rec_all = torch.cat(all_rec, dim=0) if all_rec else torch.zeros(1, 1, 28, 28)
    pix_var, pair_l2 = batch_reconstruction_variance(rec_all)
    return {
        "acc": acc,
        "img_mse": sum_img_mse / max(nb, 1),
        "psnr": sum_psnr / max(nb, 1),
        "ssim": sum_ssim / max(nb, 1),
        "aud_mse": sum_aud_mse / max(nb, 1),
        "pix_var": pix_var,
        "pair_l2": pair_l2,
        "img_kind": img_kind,
        "aud_kind": aud_kind,
    }


def main():
    fix_console_encoding()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v4.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--severity", type=float, default=0.5)
    ap.add_argument("--severity_curve", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if (cfg["device"] == "cuda"
                          and torch.cuda.is_available()) else "cpu")
    ckpt_path = str(resolve_from_root(args.ckpt or cfg["train"]["ckpt_path"]))

    log(f"[评估] 设备: {device}  加载 checkpoint: {ckpt_path}")
    model = CrossModalSNN(cfg).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    except FileNotFoundError:
        log(f"[警告] 未找到 checkpoint {ckpt_path}，使用随机初始化权重评估。")
    except RuntimeError as e:
        log(f"[警告] checkpoint 结构不匹配（可能是旧架构），使用随机权重评估。\n  {e}")

    _, test_loader = build_loaders(cfg)
    proto_img = test_loader.dataset.prototype_img.to(device)
    proto_aud = test_loader.dataset.prototype_aud.to(device)

    eval_w = [18, 7, 9, 8, 7, 9, 10, 9, 16]
    eval_a = ["l", "r", "r", "r", "r", "r", "r", "r", "r"]
    eval_hdr = ["cue模式", "acc", "imgMSE", "PSNR", "SSIM", "audMSE",
                "像素方差", "样本L2", "tgt(img/aud)"]
    log("=" * sum(eval_w))
    log(f"[评估] 6 种 cue 模式  (corrupt severity={args.severity})")
    log("  指标按恢复粒度对照 target：img/aud 列后缀 (smp)=样本级  (cat)=类别代表原型")
    log("=" * sum(eval_w))
    log(format_table_row(eval_hdr, eval_w, eval_a))
    for mode in EVAL_MODES:
        r = eval_mode(model, test_loader, cfg, mode, device,
                      args.severity, proto_img, proto_aud, args.max_batches)
        tgt = f"{r['img_kind'][:3]}/{r['aud_kind'][:3]}"
        log(format_table_row([
            mode, f"{r['acc']*100:.1f}%",
            f"{r['img_mse']:.4f}", f"{r['psnr']:.2f}", f"{r['ssim']:.3f}",
            f"{r['aud_mse']:.4f}", f"{r['pix_var']:.4f}", f"{r['pair_l2']:.4f}",
            tgt,
        ], eval_w, eval_a))

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
