"""输出 head / decoder 模块。

ClassifierHead : Index state -> logits（MLP）
ImageDecoder   : V_img state -> 28x28（CNN 上采样）
AudioDecoder   : V_aud state -> log-mel [n_mels, n_frames]（CNN 上采样）
"""

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


class AudioDecoder(nn.Module):
    """V_aud state -> 重建 log-mel [B, n_mels, n_frames]（2D ConvTranspose decoder）。"""

    def __init__(self, n_value_aud, n_mels, n_frames, base_ch=128):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.base_ch = base_ch
        self.fc = nn.Linear(n_value_aud, base_ch * 4 * 4)
        # 4×4 -> 8×8 -> 16×16 -> 32×32（默认 n_mels=n_frames=32）
        self.cnn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        nn.init.constant_(self.cnn[-1].bias, 0.5)

    def forward(self, value_state):
        x = self.fc(value_state)
        x = x.view(-1, self.base_ch, 4, 4)
        x = self.cnn(x)
        # softplus 避免 Sigmoid+负 logits 塌缩成全零；再 clamp 到 log-mel 范围
        return F.softplus(x.squeeze(1)).clamp(0.0, 1.0)
