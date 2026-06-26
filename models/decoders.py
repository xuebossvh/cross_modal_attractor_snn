"""Output heads and decoders."""

import math

import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    """Index state -> class logits [B, num_classes]."""

    def __init__(self, n_in, num_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, state):
        return self.net(state)


class ImageDecoder(nn.Module):
    """V_img state -> image logits [B, 1, 28, 28]."""

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
    """Count stride-2 upsampling stages from start_hw to out_hw."""
    if out_hw < start_hw or out_hw % start_hw != 0:
        raise ValueError(
            f"out_hw must be an integer multiple of start_hw, got {out_hw}/{start_hw}"
        )
    ratio = out_hw // start_hw
    if ratio < 1 or (ratio & (ratio - 1)) != 0:
        raise ValueError(
            f"out_hw/start_hw must be a power of two, got {out_hw}/{start_hw}"
        )
    return int(round(math.log2(ratio)))


class AudioDecoder(nn.Module):
    """V_aud state -> log-mel reconstruction [B, n_mels, n_frames]."""

    def __init__(self, n_value_aud, n_mels, n_frames, base_ch=128, start_hw=4,
                 refine_blocks=0):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.base_ch = base_ch
        self.start_hw = start_hw
        self.refine_blocks = refine_blocks

        out_hw = max(n_mels, n_frames)
        self.out_hw = out_hw
        n_up = _audio_decoder_stages(out_hw, start_hw)
        self.fc = nn.Linear(n_value_aud, base_ch * start_hw * start_hw)

        layers = []
        cur_ch = base_ch
        for i in range(n_up):
            next_ch = 16 if i == n_up - 1 else max(cur_ch // 2, 32)
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                cur_ch, next_ch, kernel_size=4, stride=2, padding=1))
            cur_ch = next_ch

        for _ in range(refine_blocks):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(cur_ch, cur_ch, kernel_size=3, padding=1))

        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(cur_ch, 1, kernel_size=3, padding=1))
        self.cnn = nn.Sequential(*layers)
        nn.init.constant_(self.cnn[-1].bias, 0.5)

    def forward(self, value_state):
        x = self.fc(value_state)
        x = x.view(-1, self.base_ch, self.start_hw, self.start_hw)
        x = self.cnn(x)
        x = x[..., :self.n_mels, :self.n_frames]
        return F.softplus(x.squeeze(1)).clamp(0.0, 1.0)
