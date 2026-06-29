"""图像 / 音频 cue 的损坏（corruption）函数。

用于把干净输入退化成「残缺 cue」，检验吸引子网络的补全/联想能力。
所有函数都是 batched torch 操作，不修改输入（返回新张量）。

severity ∈ [0, 1]：0 = 不损坏，1 = 最大损坏。
"""

import random

import torch

IMG_MODES = ["occlusion", "pixel_delete", "gaussian",
             "mask_left", "mask_right", "mask_top", "mask_bottom"]
AUD_MODES = ["gaussian", "time_mask", "freq_mask",
             "feature_dropout", "partial_temporal", "time_freq_block"]

# v7：受控训练残缺池（避免 full-random 让模型放弃样本级恢复）。
# 仅保留结构化遮挡 + 少量噪声，不与 block 过度叠加。
AUD_TRAIN_MODES = ["time_mask", "freq_mask", "time_freq_block", "gaussian"]


def _resolve(mode, pool):
    return random.choice(pool) if mode == "random" else mode


def corrupt_image(x_img, mode="random", severity=0.5):
    """损坏图像 cue。

    输入 : x_img [B, 1, H, W]，值域 [0, 1]
    模式 : occlusion / pixel_delete / gaussian /
           mask_left / mask_right / mask_top / mask_bottom / random
    """
    x = x_img.clone()
    B, C, H, W = x.shape
    m = _resolve(mode, IMG_MODES)
    s = float(max(0.0, min(1.0, severity)))

    if m == "occlusion":
        # 在随机位置挖掉一个边长 ~ s*H 的方块
        bh = max(1, int(round(s * H)))
        bw = max(1, int(round(s * W)))
        for i in range(B):
            top = random.randint(0, max(0, H - bh))
            left = random.randint(0, max(0, W - bw))
            x[i, :, top:top + bh, left:left + bw] = 0.0
    elif m == "pixel_delete":
        # 随机删除 s 比例像素（置 0）
        keep = (torch.rand_like(x) > s).float()
        x = x * keep
    elif m == "gaussian":
        # 叠加高斯噪声后裁剪
        x = x + s * torch.randn_like(x)
        x = x.clamp(0.0, 1.0)
    elif m == "mask_left":
        w = int(round(s * W))
        x[:, :, :, :w] = 0.0
    elif m == "mask_right":
        w = int(round(s * W))
        if w > 0:
            x[:, :, :, W - w:] = 0.0
    elif m == "mask_top":
        h = int(round(s * H))
        x[:, :, :h, :] = 0.0
    elif m == "mask_bottom":
        h = int(round(s * H))
        if h > 0:
            x[:, :, H - h:, :] = 0.0
    return x


def corrupt_audio(x_aud, mode="random", severity=0.5):
    """损坏音频 cue（作用于 2D log-mel 特征 [B, n_mels, n_frames]）。

    模式 : gaussian / time_mask / freq_mask /
           feature_dropout / partial_temporal / random
    """
    x = x_aud.clone()
    if x.dim() == 2:                       # [B, F] -> 不可做时间/频率 mask，仅噪声/dropout
        return _corrupt_audio_flat(x, mode, severity)
    B, M, Tf = x.shape
    m = _resolve(mode, AUD_MODES)
    s = float(max(0.0, min(1.0, severity)))

    if m == "gaussian":
        x = (x + s * torch.randn_like(x)).clamp(0.0, 1.0)
    elif m == "time_mask":
        # 随机遮挡一段连续帧
        w = max(1, int(round(s * Tf)))
        for i in range(B):
            t0 = random.randint(0, max(0, Tf - w))
            x[i, :, t0:t0 + w] = 0.0
    elif m == "freq_mask":
        # 随机遮挡一段连续频带
        w = max(1, int(round(s * M)))
        for i in range(B):
            f0 = random.randint(0, max(0, M - w))
            x[i, f0:f0 + w, :] = 0.0
    elif m == "feature_dropout":
        keep = (torch.rand_like(x) > s).float()
        x = x * keep
    elif m == "partial_temporal":
        # 只保留前 (1-s) 比例的帧，后段全遮（部分时序 cue）
        w = int(round(s * Tf))
        if w > 0:
            x[:, :, Tf - w:] = 0.0
    elif m == "time_freq_block":
        # 同时遮挡一段连续帧 × 一段连续频带（二维块遮挡）
        wt = max(1, int(round(s * Tf)))
        wf = max(1, int(round(s * M)))
        for i in range(B):
            t0 = random.randint(0, max(0, Tf - wt))
            f0 = random.randint(0, max(0, M - wf))
            x[i, f0:f0 + wf, t0:t0 + wt] = 0.0
    return x


def _corrupt_audio_flat(x, mode, severity):
    s = float(max(0.0, min(1.0, severity)))
    m = _resolve(mode, ["gaussian", "feature_dropout"])
    if m == "gaussian":
        return (x + s * torch.randn_like(x)).clamp(0.0, 1.0)
    keep = (torch.rand_like(x) > s).float()
    return x * keep
