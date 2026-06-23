"""输出 head / decoder 模块。

ClassifierHead : Index state -> logits（MLP）
ImageDecoder   : V_img state -> 28x28（CNN 上采样）
AudioDecoder   : V_aud state -> log-mel [n_mels, n_frames]（CNN 上采样）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    """Index state -> 类别 logits [B, num_classes]。"""

    def __init__(self, n_in, num_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, state):
        return self.net(state)


class ImageDecoder(nn.Module):
    """V_img state -> 重建图像 [B, 1, 28, 28]（CNN decoder，输出 logits）。"""

    def __init__(self, n_value_img, out_hw=28, base_ch=128):
        super().__init__()
        self.out_hw = out_hw
        self.base_ch = base_ch
        self.fc = nn.Linear(n_value_img, base_ch * 7 * 7)
        self.cnn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, value_state):
        x = self.fc(value_state)
        x = x.view(-1, self.base_ch, 7, 7)
        return self.cnn(x)


def _audio_decoder_stages(out_hw, start_hw=4):
    """4×4 起算，stride-2 上采样到 out_hw（32→3 层，64→4 层）。"""
    if out_hw < start_hw or (out_hw & (out_hw - 1)) != 0:
        raise ValueError(f"out_hw 须为 >= {start_hw} 的 2 的幂，得到 {out_hw}")
    n_up = int(round(math.log2(out_hw / start_hw)))
    return n_up


class AudioDecoder(nn.Module):
    """V_aud state -> 重建 log-mel [B, n_mels, n_frames]（2D ConvTranspose decoder）。"""

    def __init__(self, n_value_aud, n_mels, n_frames, base_ch=128, start_hw=4):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.base_ch = base_ch
        self.start_hw = start_hw
        out_hw = max(n_mels, n_frames)
        self.out_hw = out_hw
        n_up = _audio_decoder_stages(out_hw, start_hw)

        self.fc = nn.Linear(n_value_aud, base_ch * start_hw * start_hw)

        ch_schedule = [base_ch]
        c = base_ch
        for i in range(n_up - 1):
            c = max(c // 2, 32)
            ch_schedule.append(c)
        ch_schedule.append(16)

        layers = []
        for i in range(n_up):
            cin, cout = ch_schedule[i], ch_schedule[i + 1]
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                cin, cout, kernel_size=4, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(16, 1, kernel_size=3, padding=1))
        self.cnn = nn.Sequential(*layers)
        nn.init.constant_(self.cnn[-1].bias, 0.5)

    def forward(self, value_state):
        x = self.fc(value_state)
        x = x.view(-1, self.base_ch, self.start_hw, self.start_hw)
        x = self.cnn(x)
        x = x[..., :self.n_mels, :self.n_frames]
        return F.softplus(x.squeeze(1)).clamp(0.0, 1.0)
