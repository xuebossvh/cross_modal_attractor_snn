"""公共工具：配置加载、随机种子、模态 dropout、评估指标、matplotlib 中文。"""

import random
import sys

import numpy as np
import torch
import yaml


def fix_console_encoding():
    """修复 Windows 终端中文乱码（UTF-8 输出）。"""
    if sys.platform != "win32":
        return
    try:
        import os
        os.system("chcp 65001 >nul 2>&1")
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def log(msg):
    """带 flush 的打印，配合 fix_console_encoding 使用。"""
    print(msg, flush=True)


def text_width(s):
    """字符串在等宽终端中的显示宽度（CJK 全角=2，其余=1）。"""
    import unicodedata
    w = 0
    for ch in str(s):
        w += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
    return w


def pad_left(s, width):
    """右对齐到 display width。"""
    s = str(s)
    return " " * max(0, width - text_width(s)) + s


def pad_right(s, width):
    """左对齐到 display width。"""
    s = str(s)
    return s + " " * max(0, width - text_width(s))


def format_table_row(values, widths, aligns):
    """按显示宽度对齐一行表格（supports 中英文混排）。"""
    return "".join(
        pad_right(v, w) if a == "l" else pad_left(v, w)
        for v, w, a in zip(values, widths, aligns)
    )


def load_config(path="configs/v10d.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 6 种 cue 模式（残缺/干净 × 图像/音频/双模态）
CUE_MODES = [
    "corrupt_img_only", "corrupt_aud_only", "corrupt_both",
    "clean_img_only", "clean_aud_only", "clean_both",
]


def sample_cue_mode(cfg):
    """按 cue_modes 概率采样一个 cue 模式。

    若 ablation.use_modality_dropout=False，则只在「双模态」模式中采样
    （corrupt_both / clean_both），即不做单模态屏蔽。
    """
    use_md = cfg.get("ablation", {}).get("use_modality_dropout", True)
    cm = cfg["cue_modes"]
    if not use_md:
        # 仅双模态：按 corrupt_both / clean_both 的相对比例
        pc = cm["p_corrupt_both"]
        pcl = cm["p_clean_both"]
        r = random.random() * (pc + pcl)
        return "corrupt_both" if r < pc else "clean_both"

    weights = [
        cm["p_corrupt_img_only"], cm["p_corrupt_aud_only"], cm["p_corrupt_both"],
        cm["p_clean_img_only"], cm["p_clean_aud_only"], cm["p_clean_both"],
    ]
    return random.choices(CUE_MODES, weights=weights, k=1)[0]


def sample_train_severity(cfg, epoch):
    """训练用 corruption severity（支持 fixed / random / staged curriculum）。"""
    cc = cfg["corruption"]
    mode = cc.get("curriculum_mode", "fixed")
    if mode == "fixed":
        return cc.get("train_severity", 0.5)
    if mode == "random":
        lo = cc.get("severity_min", 0.1)
        hi = cc.get("severity_max", 0.5)
        return random.uniform(lo, hi)
    # staged：前期轻腐蚀，后期接近 train_severity
    e1 = cc.get("stage1_end", 20)
    e2 = cc.get("stage2_end", 50)
    if epoch < e1:
        return random.uniform(0.1, 0.2)
    if epoch < e2:
        return random.uniform(0.3, 0.4)
    return cc.get("train_severity", 0.5)


def resolve_train_corrupt_modes(cfg, epoch, step=None):
    """训练用残缺 family 选择（受控池 + 可选课程式分阶段）。

    返回 (img_mode, aud_mode)。图像从 img_train_modes 采样；音频优先按
    aud_family_curriculum（early/mid/late）分阶段，否则从 aud_train_modes 采样。
    """
    import random as _r
    from data.corruption import AUD_TRAIN_MODES, IMG_TRAIN_MODES
    cc = cfg["corruption"]

    def choose(pool, strategy, fallback):
        if pool:
            if isinstance(strategy, dict):
                strategy = strategy.get("mode", "random")
            if str(strategy).lower() == "balanced":
                step_idx = 0 if step is None else int(step)
                return pool[step_idx % len(pool)]
            return _r.choice(pool)
        return fallback

    img_pool = cc.get("img_train_modes", IMG_TRAIN_MODES)
    img_mode = choose(
        img_pool,
        cc.get("img_family_sampling", cc.get("family_sampling", "random")),
        cc.get("img_mode", "random"),
    )

    sched = cc.get("aud_family_curriculum") or {}
    if sched.get("enabled", False):
        e1 = cc.get("stage1_end", 20)
        e2 = cc.get("stage2_end", 35)
        if epoch < e1:
            pool = sched.get("early")
        elif epoch < e2:
            pool = sched.get("mid")
        else:
            pool = sched.get("late")
    else:
        pool = cc.get("aud_train_modes", AUD_TRAIN_MODES)

    aud_mode = choose(
        pool,
        cc.get("aud_family_sampling", cc.get("family_sampling", "random")),
        cc.get("aud_mode", "random"),
    )
    return img_mode, aud_mode


def build_cue(clean_img, clean_aud, mode, cfg, severity=None,
              img_mode=None, aud_mode=None, return_masks=False):
    """根据 cue 模式构造 (img_cue, aud_cue)。clean_* 为干净输入。

    返回的 cue 可能是残缺的、干净的，或某一模态为 None（单模态）。
    return_masks=True 时返回 (img_cue, aud_cue, masks)，其中
    masks["img"] / masks["aud"] 为对应模态缺失 mask 或 None。
    target 永远是 clean_img / clean_aud（在训练循环里单独传）。

    img_mode / aud_mode：残缺 family 覆盖；为 None 时回退到 corruption.* 配置。
    （训练用课程式 family、评估 fixed_mask 用固定 family 时显式传入。）
    """
    from data.corruption import corrupt_image, corrupt_audio
    cc = cfg["corruption"]
    s = cc["train_severity"] if severity is None else severity
    im = cc["img_mode"] if img_mode is None else img_mode
    am = cc["aud_mode"] if aud_mode is None else aud_mode
    masks = {"img": None, "aud": None}

    def pack(img_cue, aud_cue):
        if return_masks:
            return img_cue, aud_cue, masks
        return img_cue, aud_cue

    def ci(x):
        if return_masks:
            y, mask = corrupt_image(x, im, s, return_mask=True)
            masks["img"] = mask
            return y
        return corrupt_image(x, im, s)

    def ca(x):
        if return_masks:
            y, mask = corrupt_audio(x, am, s, return_mask=True)
            masks["aud"] = mask
            return y
        return corrupt_audio(x, am, s)

    if mode == "corrupt_img_only":
        return pack(ci(clean_img), None)
    if mode == "corrupt_aud_only":
        return pack(None, ca(clean_aud))
    if mode == "corrupt_both":
        return pack(ci(clean_img), ca(clean_aud))
    if mode == "clean_img_only":
        return pack(clean_img, None)
    if mode == "clean_aud_only":
        return pack(None, clean_aud)
    return pack(clean_img, clean_aud)   # clean_both


def is_aud_only_mode(mode):
    """是否为 audio-only cue 模式（含 corrupt / clean）。"""
    return mode in ("corrupt_aud_only", "clean_aud_only")


def cue_modalities(mode):
    """cue 模式 -> (has_img_cue, has_aud_cue)。"""
    if mode in ("corrupt_img_only", "clean_img_only"):
        return True, False
    if mode in ("corrupt_aud_only", "clean_aud_only"):
        return False, True
    return True, True   # corrupt_both / clean_both


def select_targets(cue_mode, clean_img, clean_aud, proto_img, proto_aud, labels):
    """按 cue 模式选择 value target，区分恢复粒度（核心策略）。

    原则：cue 只携带「类别 + 本模态细节」，缺失模态无法唯一确定具体样本，
          只能恢复类别代表原型(class medoid)；本模态可恢复具体样本。

      * audio-only : 图像目标 = 类别代表原型(medoid)   音频目标 = 本样本 clean
      * image-only : 图像目标 = 本样本 clean           音频目标 = 类别代表原型(medoid)
      * both       : 图像/音频目标均 = 本样本 clean

    返回 (x_img_target, x_aud_target, img_kind, aud_kind)，kind ∈ {"sample","category"}。
    """
    has_img, has_aud = cue_modalities(cue_mode)
    if has_img and has_aud:                       # both：双模态 → 均样本级
        return clean_img, clean_aud, "sample", "sample"
    if has_aud and not has_img:                   # audio-only
        return proto_img[labels], clean_aud, "category", "sample"
    # image-only
    return clean_img, proto_aud[labels], "sample", "category"


def aud_collapse_stats(rec, target, top_fraction=0.15):
    """音频塌缩诊断：输出/目标的能量统计 + top-k 能量召回。

    rec / target: log-mel [B, n_mels, n_frames]，值域 ~[0,1]。
    近黑图（能量塌缩）时 rec_std / rec_max 会显著低于 target，topk_recall 也会很低。

    topk_recall：target 能量前 top_fraction 的位置，有多少同时落在 rec 的
    前 top_fraction 内（逐样本取交集 / k 后平均），衡量"能量是否放对地方"。
    """
    rec = rec.float()
    target = target.float()
    flat_t = target.flatten(1)
    flat_r = rec.flatten(1)
    k = max(1, int(round(float(top_fraction) * flat_t.size(1))))
    t_idx = flat_t.topk(k, dim=1).indices
    r_idx = flat_r.topk(k, dim=1).indices
    t_mask = torch.zeros_like(flat_t, dtype=torch.bool).scatter(1, t_idx, True)
    r_mask = torch.zeros_like(flat_r, dtype=torch.bool).scatter(1, r_idx, True)
    inter = (t_mask & r_mask).sum(dim=1).float()
    recall = (inter / k).mean().item()
    return {
        "rec_mean": rec.mean().item(),
        "rec_std": rec.std().item(),
        "rec_max": rec.max().item(),
        "tgt_mean": target.mean().item(),
        "tgt_std": target.std().item(),
        "tgt_max": target.max().item(),
        "topk_recall": recall,
    }


def spike_reg(out):
    """Index + 编码器脉冲的平均活动正则。"""
    reg = 0.0
    n = 0
    for key in ("index_spikes", "spike_img_cue", "spike_aud_cue",
                "spike_img", "spike_aud"):
        s = out.get(key)
        if s is not None:
            reg = reg + s.mean()
            n += 1
    return reg / max(n, 1)


def batch_psnr(pred, target, eps=1e-8):
    """batch 平均 PSNR（dB），pred/target 值域 [0,1]，形状 [B,1,H,W]。"""
    mse = ((pred - target) ** 2).mean(dim=tuple(range(1, pred.dim())))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.mean()


def setup_matplotlib_chinese():
    """设置 matplotlib 中文字体，避免标题乱码（Windows 优先微软雅黑）。"""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi",
        "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def batch_ssim(pred, target, C1=0.01 ** 2, C2=0.03 ** 2):
    """简化 batch SSIM（全局均值/方差，值域 [0,1]）。

    pred, target: [B, 1, H, W]，像素已在 [0, 1]。
    返回 batch 平均 SSIM（越高越好，1 为完全一致）。
    """
    pred = pred.float()
    target = target.float()
    mu_p = pred.mean(dim=(2, 3), keepdim=True)
    mu_t = target.mean(dim=(2, 3), keepdim=True)
    sigma_p = ((pred - mu_p) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_t = ((target - mu_t) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_pt = ((pred - mu_p) * (target - mu_t)).mean(dim=(2, 3), keepdim=True)
    num = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2)
    return (num / den).mean()


def batch_reconstruction_variance(reconstructions, max_pairs=2000):
    """跨样本重建方差：若所有输出相同则接近 0。

    reconstructions: [B, 1, H, W]
    返回 (像素级跨 batch 方差均值, 样本间平均 L2 距离)

    注意：全量两两比较为 O(n^2)，n=10000 时会极慢；L2 用随机配对抽样估计。
    """
    flat = reconstructions.view(reconstructions.size(0), -1)
    pixel_var = flat.var(dim=0).mean().item()
    n = flat.size(0)
    if n < 2:
        return pixel_var, 0.0

    k = min(max_pairs, n * (n - 1) // 2)
    idx_i = torch.randint(0, n, (k,))
    idx_j = torch.randint(0, n, (k,))
    same = idx_i == idx_j
    while same.any():
        idx_j[same] = torch.randint(0, n, (int(same.sum().item()),))
        same = idx_i == idx_j
    diff = flat[idx_i] - flat[idx_j]
    pairwise_l2 = diff.pow(2).mean(dim=1).sqrt().mean().item()
    return pixel_var, float(pairwise_l2)


def per_class_reconstruction_variance(reconstructions, labels, num_classes=10):
    """按类别统计重建方差（类内跨样本方差均值）。

    若某类重建全部相同，该类方差接近 0。
    """
    flat = reconstructions.view(reconstructions.size(0), -1)
    labels = labels.view(-1)
    class_vars = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum().item() < 2:
            continue
        class_vars.append(flat[mask].var(dim=0).mean().item())
    if not class_vars:
        return 0.0
    return float(np.mean(class_vars))
