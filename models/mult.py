"""Directional cross-modal attention from log-mel segments to image patches."""

import torch
from torch import nn
from torch.nn import functional as F


class AudioToImageAttention(nn.Module):
    def __init__(self, n_mels, n_frames, n_index, width, heads):
        super().__init__()
        if width < 1 or heads < 1 or width % heads or n_frames % 8:
            raise ValueError("MulT requires a width divisible by heads and eight equal audio segments")
        self.segment_frames = n_frames // 8
        self.image_tokens = nn.Conv2d(1, width, kernel_size=7, stride=7)
        self.audio_tokens = nn.Linear(n_mels * self.segment_frames, width)
        self.image_position = nn.Parameter(torch.zeros(1, 16, width))
        self.audio_position = nn.Parameter(torch.zeros(1, 8, width))
        self.image_norm = nn.LayerNorm(width)
        self.audio_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Linear(width, n_index, bias=False)

    def forward(self, image, audio, image_missing=None, audio_missing=None):
        if image is None or audio is None:
            return None
        padding = None
        has_image = None
        if image_missing is not None:
            image = image.masked_fill(image_missing.bool(), 0)
            padding = F.avg_pool2d(image_missing.to(image.dtype), 7, 7).flatten(1).eq(1)
            has_image = (~padding).any(dim=1, keepdim=True)
            # Keep attention finite when every patch is masked, then zero its output.
            padding = padding & has_image
        if audio_missing is not None:
            audio = audio.masked_fill(audio_missing.bool(), 0)
        patches = self.image_tokens(image).flatten(2).transpose(1, 2)
        segments = audio.transpose(1, 2).reshape(audio.size(0), 8, -1)
        visual = self.image_norm(patches + self.image_position)
        query = self.audio_norm(self.audio_tokens(segments) + self.audio_position)
        context, _ = self.attention(query, visual, visual, key_padding_mask=padding,
                                    need_weights=False)
        current = self.output(context.mean(dim=1))
        return current if has_image is None else current * has_image
