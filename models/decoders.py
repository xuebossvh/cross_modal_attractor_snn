"""Output heads and decoders."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv2d(nn.Module):
    """Gated convolution: feat(x) * sigmoid(gate(x)) (F3a refine unit)."""

    def __init__(self, ch, kernel_size=3, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.feat = nn.Conv2d(ch, ch, kernel_size, padding=pad, dilation=dilation)
        self.gate = nn.Conv2d(ch, ch, kernel_size, padding=pad, dilation=dilation)

    def forward(self, x):
        return self.feat(x) * torch.sigmoid(self.gate(x))


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
    """Image decoder input state -> image logits [B, 1, 28, 28]."""

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


class ImageRefiner(nn.Module):
    """Image-space refiner (optional).

    Input channels: [coarse_prob, img_cue, mask]. Output: delta [B,1,H,W].
    The caller decides how to paste visible pixels back; this module only
    predicts a bounded residual for the image space.
    """

    def __init__(self, hidden_ch=32, blocks=3, delta_scale=1.0,
                 max_dilation=4):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.in_proj = nn.Conv2d(3, hidden_ch, kernel_size=3, padding=1)
        body = []
        for bi in range(max(1, blocks)):
            dilation = min(2 ** bi, int(max_dilation))
            body.append(nn.ReLU(inplace=True))
            body.append(GatedConv2d(hidden_ch, kernel_size=3, dilation=dilation))
        self.body = nn.Sequential(*body)
        self.out_proj = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, coarse_prob, cue, mask):
        x = torch.cat([coarse_prob, cue, mask], dim=1)
        x = self.in_proj(x)
        x = self.body(x)
        delta = self.out_proj(x)
        return self.delta_scale * torch.tanh(delta)


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
    """Audio decoder input state -> log-mel reconstruction [B, n_mels, n_frames]."""

    def __init__(self, n_value_aud, n_mels, n_frames, base_ch=128, start_hw=4,
                 refine_blocks=0, refine_type="plain"):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.base_ch = base_ch
        self.start_hw = start_hw
        self.refine_blocks = refine_blocks
        self.refine_type = refine_type

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

        dilations = [1, 2, 4]
        for bi in range(refine_blocks):
            layers.append(nn.ReLU(inplace=True))
            if refine_type == "gated_dilated":
                layers.append(GatedConv2d(
                    cur_ch, kernel_size=3, dilation=dilations[bi % len(dilations)]))
            elif refine_type == "plain":
                layers.append(nn.Conv2d(cur_ch, cur_ch, kernel_size=3, padding=1))
            else:
                raise ValueError(f"Unknown aud_refine_type: {refine_type}")

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


class AudioRefiner(nn.Module):
    """Spectrogram-space refiner (F3b, optional).

    Input channels: [coarse_rec, aud_cue, mask]. Output: delta [B, n_mels, n_frames].
    Dilation grows as 2**block_index (capped by ``max_dilation``) so that stacking
    more blocks enlarges the temporal receptive field to cover long time gaps. The
    caller decides how to combine delta with coarse/cue (see ``network.forward``);
    visible regions must never be overwritten.
    """

    def __init__(self, n_mels, n_frames, hidden_ch=32, blocks=2, delta_scale=1.0,
                 max_dilation=16):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.delta_scale = float(delta_scale)
        self.in_proj = nn.Conv2d(3, hidden_ch, kernel_size=3, padding=1)
        body = []
        for bi in range(max(1, blocks)):
            dilation = min(2 ** bi, int(max_dilation))
            body.append(nn.ReLU(inplace=True))
            body.append(GatedConv2d(hidden_ch, kernel_size=3, dilation=dilation))
        self.body = nn.Sequential(*body)
        self.out_proj = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, coarse, cue, mask):
        x = torch.stack([coarse, cue, mask], dim=1)
        x = self.in_proj(x)
        x = self.body(x)
        delta = self.out_proj(x).squeeze(1)
        return self.delta_scale * torch.tanh(delta)
