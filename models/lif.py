"""LIF 神经元基础模块（含 surrogate gradient）。

本模块中所有层均处理时间优先的脉冲序列，形状为 [T, B, D]。

架构对应：
    全网络各 SNN 层（编码器、Key 层、Index/吸引子层、Value 层）的基础构件。
"""

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    """前向：Heaviside 阶跃发放；反向：atan surrogate gradient。

    forward:  s = 1 if (v - v_th) >= 0 else 0
    backward: ds/dv = alpha / (2 * (1 + (pi/2 * alpha * (v - v_th))**2))
    """

    @staticmethod
    def forward(ctx, v_minus_thresh, alpha):
        ctx.save_for_backward(v_minus_thresh)
        ctx.alpha = alpha
        return (v_minus_thresh >= 0).to(v_minus_thresh.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        # (1/pi) * atan(pi/2 * alpha * x) + 0.5 的导数
        sg = alpha / (2.0 * (1.0 + (torch.pi / 2.0 * alpha * x) ** 2))
        return grad_output * sg, None


def spike_fn(v_minus_thresh, alpha=2.0):
    return SurrogateSpike.apply(v_minus_thresh, alpha)


class LIFNeuron(nn.Module):
    """有状态的 Leaky-Integrate-and-Fire 神经元（不含突触权重）。

    每次 `step` 调用执行一次膜电位更新：
        v_t = beta * v_{t-1} + I_t
        s_t = Theta(v_t - v_th)

    采用 reset-by-zero：发放后膜电位归零。
    """

    def __init__(self, beta=0.9, v_threshold=1.0, surrogate_alpha=2.0):
        super().__init__()
        self.beta = beta
        self.v_threshold = v_threshold
        self.surrogate_alpha = surrogate_alpha

    def init_state(self, shape, device, dtype=torch.float32):
        return torch.zeros(shape, device=device, dtype=dtype)

    def step(self, v, input_current):
        v = self.beta * v + input_current
        spikes = spike_fn(v - self.v_threshold, self.surrogate_alpha)
        # reset-by-zero（detach 避免与 surrogate gradient 冲突）
        v = v * (1.0 - spikes.detach())
        return v, spikes


class LIFLayer(nn.Module):
    """线性突触 + 整段 time window 上的 LIF 动态。

    输入 : 脉冲/电流 [T, B, in_features]
    输出 : (spikes [T, B, out_features], membrane_trace [T, B, out_features])
    """

    def __init__(self, in_features, out_features, beta=0.9,
                 v_threshold=1.0, surrogate_alpha=2.0, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.neuron = LIFNeuron(beta, v_threshold, surrogate_alpha)
        self.out_features = out_features

    def forward(self, x):
        T, B, _ = x.shape
        v = self.neuron.init_state((B, self.out_features), x.device, x.dtype)
        spikes_out, mem_out = [], []
        for t in range(T):
            current = self.fc(x[t])
            v, s = self.neuron.step(v, current)
            spikes_out.append(s)
            mem_out.append(v)
        return torch.stack(spikes_out, dim=0), torch.stack(mem_out, dim=0)


def rate(spikes):
    """沿时间维求平均发放率：[T, B, D] -> [B, D]。"""
    return spikes.mean(dim=0)
