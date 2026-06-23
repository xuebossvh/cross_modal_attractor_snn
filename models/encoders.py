"""图像与音频模态的 SNN 编码器。

ImageSNNEncoder：首脉冲时间编码 + 指数 trace，再送入 LIF。
AudioSNNEncoder：Conv2d+LIF（默认）或 flatten Linear+LIF（legacy）。
"""

import torch
import torch.nn as nn

from .lif import LIFLayer, LIFNeuron, spike_fn


def _to_time(x, T):
    return x.unsqueeze(0).expand(T, *x.shape).contiguous()


def _to_time_4d(x, T):
    """[B, C, H, W] -> [T, B, C, H, W]。"""
    return x.unsqueeze(0).expand(T, *x.shape).contiguous()


def _poisson(x, T):
    x = x.clamp(0.0, 1.0)
    xt = x.unsqueeze(0).expand(T, *x.shape)
    return (torch.rand_like(xt) < xt).to(x.dtype)


def _first_spike_encode(x, T):
    """首脉冲时间编码：像素越亮，发放时间越早。"""
    x = x.clamp(0.0, 1.0)
    t_first = ((1.0 - x) * max(T - 1, 1)).round().long().clamp(0, T - 1)
    time_idx = torch.arange(T, device=x.device).view(T, 1, 1)
    return (time_idx == t_first.unsqueeze(0)).to(x.dtype)


def _exponential_trace(spikes, trace_decay):
    T, B, F = spikes.shape
    trace = torch.zeros(B, F, device=spikes.device, dtype=spikes.dtype)
    out = []
    for t in range(T):
        trace = trace_decay * trace + spikes[t]
        out.append(trace)
    return torch.stack(out, dim=0)


def _conv_out_hw(h, w, kernel, stride, padding):
    oh = (h + 2 * padding - kernel) // stride + 1
    ow = (w + 2 * padding - kernel) // stride + 1
    return oh, ow


class LIFConv2dStage(nn.Module):
    """Conv2d 电流 + 逐像素展平 LIF：[T,B,Cin,H,W] -> [T,B,Cout,H',W'] 脉冲（0/1）。"""

    def __init__(self, in_ch, out_ch, in_h, in_w, kernel_size=3, stride=1,
                 padding=1, beta=0.9, v_threshold=1.0, surrogate_alpha=2.0):
        super().__init__()
        self.out_ch = out_ch
        self.out_h, self.out_w = _conv_out_hw(
            in_h, in_w, kernel_size, stride, padding)
        self.flat_dim = out_ch * self.out_h * self.out_w
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding)
        self.neuron = LIFNeuron(beta, v_threshold, surrogate_alpha)

    def forward(self, x):
        """x: [T, B, Cin, H, W] -> spikes [T, B, Cout, H', W']"""
        T, B, _, _, _ = x.shape
        v = self.neuron.init_state((B, self.flat_dim), x.device, x.dtype)
        spikes = []
        for t in range(T):
            cur = self.conv(x[t]).flatten(1)
            v = self.neuron.beta * v + cur
            s = spike_fn(v - self.neuron.v_threshold, self.neuron.surrogate_alpha)
            v = v * (1.0 - s.detach())
            spikes.append(s.view(B, self.out_ch, self.out_h, self.out_w))
        return torch.stack(spikes, dim=0)


class ImageSNNEncoder(nn.Module):
    """图像 -> 脉冲特征（首脉冲编码 + trace -> LIF）。"""

    def __init__(self, img_in=784, hidden=256, D_img=128, T=20,
                 beta=0.9, v_threshold=1.0, surrogate_alpha=2.0,
                 encoding="first_spike_trace", trace_decay=0.9):
        super().__init__()
        self.T = T
        self.encoding = encoding
        self.trace_decay = trace_decay
        self.l1 = LIFLayer(img_in, hidden, beta, v_threshold, surrogate_alpha)
        self.l2 = LIFLayer(hidden, D_img, beta, v_threshold, surrogate_alpha)

    def _encode_input(self, x):
        if self.encoding == "first_spike_trace":
            spikes = _first_spike_encode(x, self.T)
            trace = _exponential_trace(spikes, self.trace_decay)
            return spikes + trace
        if self.encoding == "poisson":
            return _poisson(x, self.T)
        return _to_time(x, self.T)

    def forward(self, x_img):
        x = x_img.reshape(x_img.shape[0], -1)
        xt = self._encode_input(x)
        s1, _ = self.l1(xt)
        s2, _ = self.l2(s1)
        return s2


class AudioSNNEncoder(nn.Module):
    """log-mel -> 脉冲特征。

    encoder_type=conv（默认）：
        [B,1,H,W] -> Conv+LIF -> Conv+LIF -> flatten -> LIF -> LIF -> [T,B,D_aud]
    encoder_type=linear：legacy flatten 1024 + LIF。
    """

    def __init__(self, aud_in=1024, hidden=128, D_aud=128, T=20,
                 beta=0.9, v_threshold=1.0, surrogate_alpha=2.0,
                 encoding="current", encoder_type="conv",
                 n_mels=32, n_frames=32, conv_ch1=16, conv_ch2=32):
        super().__init__()
        self.T = T
        self.encoding = encoding
        self.encoder_type = encoder_type

        if encoder_type == "conv":
            h, w = n_mels, n_frames
            self.conv1 = LIFConv2dStage(
                1, conv_ch1, h, w, kernel_size=3, stride=2, padding=1,
                beta=beta, v_threshold=v_threshold, surrogate_alpha=surrogate_alpha)
            h2, w2 = self.conv1.out_h, self.conv1.out_w
            self.conv2 = LIFConv2dStage(
                conv_ch1, conv_ch2, h2, w2, kernel_size=3, stride=2, padding=1,
                beta=beta, v_threshold=v_threshold, surrogate_alpha=surrogate_alpha)
            flat = self.conv2.flat_dim
            self.l3 = LIFLayer(flat, hidden, beta, v_threshold, surrogate_alpha)
            self.l4 = LIFLayer(hidden, D_aud, beta, v_threshold, surrogate_alpha)
        else:
            self.l1 = LIFLayer(aud_in, hidden, beta, v_threshold, surrogate_alpha)
            self.l2 = LIFLayer(hidden, D_aud, beta, v_threshold, surrogate_alpha)

    def _encode_input_4d(self, x):
        """x: [B, 1, H, W] -> [T, B, 1, H, W]"""
        if self.encoding == "poisson":
            return _poisson(x, self.T)
        return _to_time_4d(x, self.T)

    def forward(self, x_aud):
        if self.encoder_type == "conv":
            x = x_aud.unsqueeze(1)
            xt = self._encode_input_4d(x)
            s1 = self.conv1(xt)
            s2 = self.conv2(s1)
            flat = s2.reshape(self.T, x_aud.size(0), -1)
            s3, _ = self.l3(flat)
            s4, _ = self.l4(s3)
            return s4

        x = x_aud.reshape(x_aud.shape[0], -1)
        if self.encoding == "poisson":
            xt = _poisson(x, self.T)
        else:
            xt = _to_time(x, self.T)
        s1, _ = self.l1(xt)
        s2, _ = self.l2(s1)
        return s2
