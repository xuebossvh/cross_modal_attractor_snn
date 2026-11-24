"""Cross-modal attention between log-mel segments and image patches."""

import torch
from torch import nn
from torch.nn import functional as F


class _CrossAttentionTokens(nn.Module):
    def __init__(self, n_mels, n_frames, width, heads):
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

    def _encode_tokens(self, image, audio):
        patches = self.image_tokens(image).flatten(2).transpose(1, 2)
        segments = audio.transpose(1, 2).reshape(audio.size(0), 8, -1)
        return (self.image_norm(patches + self.image_position),
                self.audio_norm(self.audio_tokens(segments) + self.audio_position))


class AudioToImageAttention(_CrossAttentionTokens):
    def __init__(self, n_mels, n_frames, n_index, width, heads):
        super().__init__(n_mels, n_frames, width, heads)
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
        visual, query = self._encode_tokens(image, audio)
        context, _ = self.attention(query, visual, visual, key_padding_mask=padding,
                                    need_weights=False)
        current = self.output(context.mean(dim=1))
        return current if has_image is None else current * has_image


class BidirectionalCrossAttention(_CrossAttentionTokens):
    """Route image queries to Image Value and audio queries to Audio Value."""

    def __init__(self, n_mels, n_frames, n_value_img, n_value_aud, width, heads):
        super().__init__(n_mels, n_frames, width, heads)
        self.image_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.audio_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.image_current = nn.Linear(width, n_value_img, bias=False)
        self.audio_current = nn.Linear(width, n_value_aud, bias=False)

    @staticmethod
    def _pool(attention, query, source, query_missing, source_missing):
        if source_missing is not None:
            has_source = (~source_missing).any(dim=1, keepdim=True)
            # An all-masked softmax is undefined. Zero those samples after attention.
            source_missing = source_missing & has_source
        context, _ = attention(query, source, source, key_padding_mask=source_missing,
                               need_weights=False)
        if query_missing is None:
            pooled = context.mean(dim=1)
        else:
            valid = (~query_missing).unsqueeze(-1)
            pooled = (context * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return pooled if source_missing is None else pooled * has_source

    def forward(self, image, audio, image_missing=None, audio_missing=None):
        if image is None or audio is None:
            return None, None
        image_padding = audio_padding = None
        if image_missing is not None:
            image = image.masked_fill(image_missing.bool(), 0)
            image_padding = F.avg_pool2d(image_missing.to(image.dtype), 7, 7).flatten(1).eq(1)
        if audio_missing is not None:
            audio = audio.masked_fill(audio_missing.bool(), 0)
            audio_padding = audio_missing.transpose(1, 2).reshape(audio.size(0), 8, -1).bool().all(dim=-1)
        image_tokens, audio_tokens = self._encode_tokens(image, audio)
        from_audio = self._pool(self.image_attention, image_tokens, audio_tokens,
                                image_padding, audio_padding)
        from_image = self._pool(self.audio_attention, audio_tokens, image_tokens,
                                audio_padding, image_padding)
        return self.image_current(from_audio), self.audio_current(from_image)
