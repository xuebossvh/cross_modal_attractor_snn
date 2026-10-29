"""图像 / 音频 cue 的损坏（corruption）函数。

用于把干净输入退化成「残缺 cue」，检验吸引子网络的补全/联想能力。
所有函数都是 batched torch 操作，不修改输入（返回新张量）。

severity ∈ [0, 1]：0 = 不损坏，1 = 最大损坏。
"""

import math
import random

import torch

IMG_TRAIN_MODES = [
    "occlusion", "pixel_delete", "mask_vertical",
    "mask_horizontal", "salt_mask",
]
LOCAL_NOISE_MODES = ("local_gaussian", "local_impulse")

# Keep the legacy random distribution unchanged when introducing new families.
_IMG_RANDOM_MODES = IMG_TRAIN_MODES + [
    "gaussian", "mask_left", "mask_right", "mask_top", "mask_bottom",
]
IMG_MODES = _IMG_RANDOM_MODES + list(LOCAL_NOISE_MODES)
_AUD_RANDOM_MODES = ["gaussian", "time_mask", "freq_mask",
                     "feature_dropout", "partial_temporal", "time_freq_block"]
AUD_MODES = _AUD_RANDOM_MODES + list(LOCAL_NOISE_MODES)
AUD_FAMILY_GROUPS = {
    "paper_aligned_time_gap": ["time_mask", "partial_temporal"],
    "spectrogram_occlusion": ["time_freq_block", "freq_mask", "feature_dropout"],
    "noise": ["gaussian"],
    "local_contamination": list(LOCAL_NOISE_MODES),
}

# 受控训练残缺池（避免 full-random 让模型放弃样本级恢复）。
# 默认保留 5 个带 mask 的谱图缺失 family，与 v10f 主协议一致。
AUD_TRAIN_MODES = [
    "time_mask", "freq_mask", "feature_dropout",
    "partial_temporal", "time_freq_block",
]


def _resolve(mode, pool):
    return random.choice(pool) if mode == "random" else mode


def _local_contamination(x, mode, severity, noise_std):
    """Inject noise into one random rectangle per sample, returning its support.

    ``severity`` is the approximate fraction of spatial positions selected, not
    the noise amplitude. Integer rectangle dimensions quantize the coverage.
    ``noise_std`` controls additive Gaussian noise; zero disables either local
    family. Impulse corruption replaces every selected value with random 0/1.
    Support labels mark injection, including pixels unchanged by clipping or
    impulse replacement. They are supervision, not observations for inference.
    """
    strength = float(noise_std)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("noise_std must be finite and nonnegative")
    coverage = float(severity)
    if not math.isfinite(coverage):
        raise ValueError("severity must be finite")
    coverage = max(0.0, min(1.0, coverage))
    mask = torch.zeros_like(x)
    if coverage == 0.0 or strength == 0.0:
        return x.clone(), mask
    height, width = x.shape[-2:]
    block_h = min(height, max(1, int(round(height * math.sqrt(coverage)))))
    block_w = min(width, max(1, int(round(coverage * height * width / block_h))))
    for index in range(x.size(0)):
        top = random.randint(0, height - block_h)
        left = random.randint(0, width - block_w)
        mask[index, ..., top:top + block_h, left:left + block_w] = 1.0
    if mode == "local_gaussian":
        contaminated = (x + strength * torch.randn_like(x)).clamp(0.0, 1.0)
    else:
        contaminated = (torch.rand_like(x) < 0.5).to(x.dtype)
    return torch.where(mask.bool(), contaminated, x), mask


def corrupt_image(x_img, mode="random", severity=0.5, return_mask=False,
                  noise_std=0.2):
    """损坏图像 cue。

    输入 : x_img [B, 1, H, W]，值域 [0, 1]
    模式 : occlusion / pixel_delete / mask_vertical / mask_horizontal /
           salt_mask / gaussian / mask_left / mask_right /
           mask_top / mask_bottom / local_gaussian / local_impulse / random
    local_*: severity controls region coverage; noise_std controls Gaussian
    amplitude (zero disables local noise). Legacy modes ignore noise_std.
    """
    x = x_img.clone()
    B, C, H, W = x.shape
    m = _resolve(mode, _IMG_RANDOM_MODES)
    s = float(max(0.0, min(1.0, severity)))
    mask = None

    if m in LOCAL_NOISE_MODES:
        x, mask = _local_contamination(x, m, severity, noise_std)
    elif m == "occlusion":
        # 在随机位置挖掉一个边长 ~ s*H 的方块
        bh = max(1, int(round(s * H)))
        bw = max(1, int(round(s * W)))
        mask = torch.zeros_like(x)
        for i in range(B):
            top = random.randint(0, max(0, H - bh))
            left = random.randint(0, max(0, W - bw))
            x[i, :, top:top + bh, left:left + bw] = 0.0
            mask[i, :, top:top + bh, left:left + bw] = 1.0
    elif m == "pixel_delete":
        # 随机删除 s 比例像素（置 0）
        keep = (torch.rand_like(x) > s).float()
        x = x * keep
        mask = 1.0 - keep
    elif m == "mask_vertical":
        # 随机遮挡左侧或右侧连续竖向区域
        w = int(round(s * W))
        mask = torch.zeros_like(x)
        if w > 0:
            if random.random() < 0.5:
                x[:, :, :, :w] = 0.0
                mask[:, :, :, :w] = 1.0
            else:
                x[:, :, :, W - w:] = 0.0
                mask[:, :, :, W - w:] = 1.0
    elif m == "mask_horizontal":
        # 随机遮挡上侧或下侧连续横向区域
        h = int(round(s * H))
        mask = torch.zeros_like(x)
        if h > 0:
            if random.random() < 0.5:
                x[:, :, :h, :] = 0.0
                mask[:, :, :h, :] = 1.0
            else:
                x[:, :, H - h:, :] = 0.0
                mask[:, :, H - h:, :] = 1.0
    elif m == "salt_mask":
        # 白色缺失：随机像素置 1，同时把这些位置标为 missing。
        mask = (torch.rand_like(x) < s).float()
        x = x * (1.0 - mask) + mask
    elif m == "gaussian":
        # 叠加高斯噪声后裁剪
        x = x + s * torch.randn_like(x)
        x = x.clamp(0.0, 1.0)
    elif m == "mask_left":
        w = int(round(s * W))
        mask = torch.zeros_like(x)
        x[:, :, :, :w] = 0.0
        mask[:, :, :, :w] = 1.0
    elif m == "mask_right":
        w = int(round(s * W))
        mask = torch.zeros_like(x)
        if w > 0:
            x[:, :, :, W - w:] = 0.0
            mask[:, :, :, W - w:] = 1.0
    elif m == "mask_top":
        h = int(round(s * H))
        mask = torch.zeros_like(x)
        x[:, :, :h, :] = 0.0
        mask[:, :, :h, :] = 1.0
    elif m == "mask_bottom":
        h = int(round(s * H))
        mask = torch.zeros_like(x)
        if h > 0:
            x[:, :, H - h:, :] = 0.0
            mask[:, :, H - h:, :] = 1.0
    if return_mask:
        return x, mask
    return x


def _active_trailing_window(sample, width, threshold_ratio=0.1):
    """Return a fixed-width window ending at the last active speech frame."""
    total_frames = sample.size(-1)
    width = max(0, min(int(width), total_frames))
    if width == 0:
        return total_frames, total_frames

    frame_energy = sample.detach().abs().mean(dim=0)
    peak = frame_energy.max()
    if not torch.isfinite(peak) or peak.item() <= 1e-8:
        return total_frames - width, total_frames

    active = torch.nonzero(
        frame_energy >= peak * float(threshold_ratio), as_tuple=False
    ).flatten()
    if active.numel() == 0:
        return total_frames - width, total_frames

    end = min(total_frames, int(active[-1].item()) + 1)
    end = max(width, end)
    start = end - width
    return start, end


def corrupt_audio(x_aud, mode="random", severity=0.5, return_mask=False,
                  noise_std=0.2):
    """损坏音频 cue（作用于 2D log-mel 特征 [B, n_mels, n_frames]）。

    模式 : gaussian / time_mask / freq_mask / feature_dropout /
           partial_temporal / time_freq_block / local_gaussian /
           local_impulse / random
    local_*: severity controls region coverage; noise_std controls Gaussian
    amplitude (zero disables local noise). Legacy modes ignore noise_std.
    """
    x = x_aud.clone()
    if x.dim() == 2:                       # [B, F] -> 不可做时间/频率 mask，仅噪声/dropout
        if mode in LOCAL_NOISE_MODES:
            raise ValueError("Local audio contamination requires [B, n_mels, n_frames]")
        return _corrupt_audio_flat(x, mode, severity, return_mask=return_mask)
    B, M, Tf = x.shape
    m = _resolve(mode, _AUD_RANDOM_MODES)
    s = float(max(0.0, min(1.0, severity)))
    mask = None

    if m in LOCAL_NOISE_MODES:
        x, mask = _local_contamination(x, m, severity, noise_std)
    elif m == "gaussian":
        x = (x + s * torch.randn_like(x)).clamp(0.0, 1.0)
    elif m == "time_mask":
        # 随机遮挡一段连续帧
        w = int(round(s * Tf))
        mask = torch.zeros_like(x)
        if w > 0:
            for i in range(B):
                t0 = random.randint(0, max(0, Tf - w))
                x[i, :, t0:t0 + w] = 0.0
                mask[i, :, t0:t0 + w] = 1.0
    elif m == "freq_mask":
        # 随机遮挡一段连续频带
        w = int(round(s * M))
        mask = torch.zeros_like(x)
        if w > 0:
            for i in range(B):
                f0 = random.randint(0, max(0, M - w))
                x[i, f0:f0 + w, :] = 0.0
                mask[i, f0:f0 + w, :] = 1.0
    elif m == "feature_dropout":
        keep = (torch.rand_like(x) > s).float()
        x = x * keep
        mask = 1.0 - keep
    elif m == "partial_temporal":
        # 遮挡真实语音的 trailing segment，避免固定尾部只命中 padding/silence。
        w = int(round(s * Tf))
        mask = torch.zeros_like(x)
        if w > 0:
            for i in range(B):
                t0, t1 = _active_trailing_window(x_aud[i], w)
                x[i, :, t0:t1] = 0.0
                mask[i, :, t0:t1] = 1.0
    elif m == "time_freq_block":
        # 同时遮挡一段连续帧 × 一段连续频带（二维块遮挡）
        wt = int(round(s * Tf))
        wf = int(round(s * M))
        mask = torch.zeros_like(x)
        if wt > 0 and wf > 0:
            for i in range(B):
                t0 = random.randint(0, max(0, Tf - wt))
                f0 = random.randint(0, max(0, M - wf))
                x[i, f0:f0 + wf, t0:t0 + wt] = 0.0
                mask[i, f0:f0 + wf, t0:t0 + wt] = 1.0
    if return_mask:
        return x, mask
    return x


def _corrupt_audio_flat(x, mode, severity, return_mask=False):
    s = float(max(0.0, min(1.0, severity)))
    m = _resolve(mode, ["gaussian", "feature_dropout"])
    if m == "gaussian":
        out = (x + s * torch.randn_like(x)).clamp(0.0, 1.0)
        return (out, None) if return_mask else out
    keep = (torch.rand_like(x) > s).float()
    out = x * keep
    return (out, None) if return_mask else out
