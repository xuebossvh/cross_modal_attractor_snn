"""跨模态循环吸引子记忆层（binding / readout 两阶段版）。

架构对应（架构图中央大框）：
    KeyLayer            -> "K_img" 与 "K_aud" 子层
    RecurrentIndexLayer -> "Index 层 A"（循环吸引子核心）
    ValueLayer          -> "V_img" 与 "V_aud" 子层
    CrossModalAttractorMemory -> 将 Key + Index + Value 封装为一体

设计要点（本版重点修改）：
    * 区分两个阶段：
        - binding 阶段：cue -> Index A 收敛；clean target -> Encoder -> target Value
          神经元群（可延迟）。A 与 target Value 共同放电，用于学习 A->V（bind loss）。
          这一阶段不把混合 Value 送 decoder 计算重建损失。
        - readout 阶段：关闭 target Encoder->Value，返回 A 驱动的 Value
          (v_*_from_A)，供 network 层 decoder 使用。
    * decoder 的 Value 主输入永远来自 v_*_from_A，绝不读「A + Encoder」
      混合 Value；v9 的 cue detail 条件拼接发生在 network 层。
    * 推理阶段 == readout 阶段，禁止 target。
    * 消融开关：use_recurrent / use_kwta / use_delayed_value_target。

硬约束：Key 在记忆内部；K_img/K_aud、V_img/V_aud 不共享权重；
        Key 以独立权重电流求和进入 Index（不拼接）；Index 非 Hopfield。
"""

import torch
import torch.nn as nn

from .lif import LIFNeuron, LIFLayer, spike_fn, rate


# ----------------------------------------------------------------------------
# Key 层（每个模态独立实例）
# ----------------------------------------------------------------------------
class KeyLayer(nn.Module):
    """单模态 spiking Key 层（编码器脉冲的 LIF 投射）。"""

    def __init__(self, d_enc, n_key, beta=0.9, v_threshold=1.0, surrogate_alpha=2.0):
        super().__init__()
        self.layer = LIFLayer(d_enc, n_key, beta, v_threshold, surrogate_alpha)

    def forward(self, enc_spikes):
        spikes, _ = self.layer(enc_spikes)
        return spikes


# ----------------------------------------------------------------------------
# Index / 吸引子层（唯一的循环层）
# ----------------------------------------------------------------------------
class RecurrentIndexLayer(nn.Module):
    """带 k-WTA 竞争的循环 LIF 吸引子层。

    每步电流：I_t = alpha_img*W_img_to_A(key_img_t) + alpha_aud*W_aud_to_A(key_aud_t)
                    + [use_recurrent] W_rec(prev_spikes) - [use_kwta] 竞争抑制
    """

    def __init__(self, n_key_img, n_key_aud, n_index,
                 beta=0.9, v_threshold=1.0, surrogate_alpha=2.0,
                 alpha_img=1.0, alpha_aud=1.0, recurrent_scale=0.2,
                 wta_mode="kwta", k_wta=32, inhibition_strength=1.0,
                 use_recurrent=True, use_kwta=True,
                 input_schedule="simultaneous", phase_split=0.5,
                 phase_current_scale=1.0, phase_on_bimodal_only=True):
        super().__init__()
        self.n_index = n_index
        self.alpha_img = alpha_img
        self.alpha_aud = alpha_aud
        self.wta_mode = wta_mode
        self.k_wta = k_wta
        self.inhibition_strength = inhibition_strength
        self.use_recurrent = use_recurrent
        self.use_kwta = use_kwta
        self.input_schedule = input_schedule
        self.phase_split = phase_split
        self.phase_current_scale = float(phase_current_scale)
        self.phase_on_bimodal_only = phase_on_bimodal_only

        self.W_img_to_A = nn.Linear(n_key_img, n_index, bias=True)
        self.W_aud_to_A = nn.Linear(n_key_aud, n_index, bias=True)

        # 循环 E->E（无对称约束 -> 非 Hopfield）
        self.W_rec = nn.Linear(n_index, n_index, bias=False)
        with torch.no_grad():
            self.W_rec.weight.mul_(recurrent_scale)

        # 方案 B（抑制池）预留
        self.to_inhib = nn.Linear(n_index, 1, bias=False)
        with torch.no_grad():
            self.to_inhib.weight.fill_(1.0 / n_index)

        self.neuron = LIFNeuron(beta, v_threshold, surrogate_alpha)

    def _competition(self, v, raw_spikes):
        if self.wta_mode == "inhibition_pool":
            pool = self.to_inhib(raw_spikes)
            inhib = self.inhibition_strength * pool
            return spike_fn(v - self.neuron.v_threshold - inhib,
                            self.neuron.surrogate_alpha)
        # 方案 A：硬 top-k（按膜电位）
        k = min(self.k_wta, self.n_index)
        topk_idx = torch.topk(v, k, dim=-1).indices
        mask = torch.zeros_like(raw_spikes)
        mask.scatter_(-1, topk_idx, 1.0)
        return raw_spikes * mask

    def _input_gates(self, t, T, has_img, has_aud):
        mode = self.input_schedule
        if mode in ("simultaneous", "both", None):
            return 1.0, 1.0

        both_present = has_img and has_aud
        if self.phase_on_bimodal_only and not both_present:
            return 1.0, 1.0

        split = int(round(T * float(self.phase_split)))
        split = max(1, min(T - 1, split))
        scale = self.phase_current_scale

        if mode in ("phased_img_first", "img_first"):
            return (scale, 0.0) if t < split else (0.0, scale)
        if mode in ("phased_aud_first", "aud_first"):
            return (0.0, scale) if t < split else (scale, 0.0)
        if mode == "interleave_img_first":
            return (scale, 0.0) if (t % 2 == 0) else (0.0, scale)
        if mode == "interleave_aud_first":
            return (0.0, scale) if (t % 2 == 0) else (scale, 0.0)
        raise ValueError(f"Unknown index.input_schedule: {mode}")

    def forward(self, key_img_spikes=None, key_aud_spikes=None):
        assert (key_img_spikes is not None) or (key_aud_spikes is not None), \
            "Index 层至少需要一种模态输入。"
        ref = key_img_spikes if key_img_spikes is not None else key_aud_spikes
        T, B, _ = ref.shape
        device, dtype = ref.device, ref.dtype
        has_img = key_img_spikes is not None
        has_aud = key_aud_spikes is not None

        v = self.neuron.init_state((B, self.n_index), device, dtype)
        prev_spikes = torch.zeros((B, self.n_index), device=device, dtype=dtype)

        spikes_out = []
        for t in range(T):
            img_gate, aud_gate = self._input_gates(t, T, has_img, has_aud)
            if self.use_recurrent:
                current = self.W_rec(prev_spikes)
            else:
                current = torch.zeros((B, self.n_index), device=device, dtype=dtype)
            if has_img and img_gate != 0.0:
                current = current + img_gate * self.alpha_img * self.W_img_to_A(key_img_spikes[t])
            if has_aud and aud_gate != 0.0:
                current = current + aud_gate * self.alpha_aud * self.W_aud_to_A(key_aud_spikes[t])

            v = self.neuron.beta * v + current
            raw_spikes = spike_fn(v - self.neuron.v_threshold,
                                  self.neuron.surrogate_alpha)
            s = self._competition(v, raw_spikes) if self.use_kwta else raw_spikes
            v = v * (1.0 - s.detach())
            prev_spikes = s
            spikes_out.append(s)

        index_spikes = torch.stack(spikes_out, dim=0)
        return index_spikes, rate(index_spikes)


# ----------------------------------------------------------------------------
# Value 层（每个模态独立实例）
# ----------------------------------------------------------------------------
class ValueLayer(nn.Module):
    """单模态 spiking Value 层，分离两路：

        * A 路（始终）   : Index A -> Value      (W_A_to_V)，decoder 的 Value 主输入读它
        * target 路（仅 binding）: clean target Encoder -> Value (W_enc_to_V)，
                                   可延迟 delay 步，仅用于 bind loss 的 teacher

    forward 返回：
        a_spikes      [T,B,N]   A 驱动的 value 脉冲
        a_state       [B,N]     A 驱动的 value rate（=v_*_from_A）
        target_state  [B,N]或None  target 驱动的 value rate（仅 binding）
    """

    def __init__(self, n_index, d_enc, n_value,
                 beta=0.9, v_threshold=1.0, surrogate_alpha=2.0):
        super().__init__()
        self.n_value = n_value
        self.W_A_to_V = nn.Linear(n_index, n_value, bias=True)
        self.W_enc_to_V = nn.Linear(d_enc, n_value, bias=True)   # 仅 target 路
        self.neuron = LIFNeuron(beta, v_threshold, surrogate_alpha)

    def _run_from_A(self, index_spikes):
        T, B, _ = index_spikes.shape
        v = self.neuron.init_state((B, self.n_value),
                                   index_spikes.device, index_spikes.dtype)
        out = []
        for t in range(T):
            v, s = self.neuron.step(v, self.W_A_to_V(index_spikes[t]))
            out.append(s)
        sp = torch.stack(out, dim=0)
        return sp, rate(sp)

    def _run_target(self, enc_spikes, delay):
        T, B, _ = enc_spikes.shape
        v = self.neuron.init_state((B, self.n_value),
                                   enc_spikes.device, enc_spikes.dtype)
        out = []
        for t in range(T):
            ts = t - delay
            if ts < 0:
                cur = torch.zeros((B, self.n_value),
                                  device=enc_spikes.device, dtype=enc_spikes.dtype)
            else:
                cur = self.W_enc_to_V(enc_spikes[ts])
            v, s = self.neuron.step(v, cur)
            out.append(s)
        sp = torch.stack(out, dim=0)
        return sp, rate(sp)

    def forward(self, index_spikes, target_enc_spikes=None, phase="readout",
                use_delayed_target=True, delay=2):
        a_spikes, a_state = self._run_from_A(index_spikes)
        target_state = None
        if phase == "binding" and target_enc_spikes is not None:
            d = delay if use_delayed_target else 0
            _, target_state = self._run_target(target_enc_spikes, d)
        return a_spikes, a_state, target_state


# ----------------------------------------------------------------------------
# 完整记忆模块 = Key + Index + Value
# ----------------------------------------------------------------------------
class CrossModalAttractorMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["dims"]
        s = cfg["snn"]
        ix = cfg["index"]
        ab = cfg.get("ablation", {})
        beta, vth, alpha = s["beta"], s["v_threshold"], s["surrogate_alpha"]

        self.binding_delay = cfg.get("value", {}).get("binding_delay", 2)
        self.use_delayed_target = ab.get("use_delayed_value_target", True)

        self.K_img = KeyLayer(d["D_img"], d["N_key_img"], beta, vth, alpha)
        self.K_aud = KeyLayer(d["D_aud"], d["N_key_aud"], beta, vth, alpha)

        self.index = RecurrentIndexLayer(
            d["N_key_img"], d["N_key_aud"], d["N_index"],
            beta, vth, alpha,
            alpha_img=ix["alpha_img"], alpha_aud=ix["alpha_aud"],
            recurrent_scale=ix["recurrent_scale"], wta_mode=ix["wta_mode"],
            k_wta=ix["k_wta"], inhibition_strength=ix["inhibition_strength"],
            use_recurrent=ab.get("use_recurrent", True),
            use_kwta=ab.get("use_kwta", True),
            input_schedule=ix.get("input_schedule", "simultaneous"),
            phase_split=ix.get("phase_split", 0.5),
            phase_current_scale=ix.get("phase_current_scale", 1.0),
            phase_on_bimodal_only=ix.get("phase_on_bimodal_only", True),
        )

        self.V_img = ValueLayer(d["N_index"], d["D_img"], d["N_value_img"],
                                beta, vth, alpha)
        self.V_aud = ValueLayer(d["N_index"], d["D_aud"], d["N_value_aud"],
                                beta, vth, alpha)

    def forward(self, spike_img_cue=None, spike_aud_cue=None,
                spike_img_target=None, spike_aud_target=None, phase="readout"):
        key_img = self.K_img(spike_img_cue) if spike_img_cue is not None else None
        key_aud = self.K_aud(spike_aud_cue) if spike_aud_cue is not None else None

        index_spikes, index_state = self.index(key_img, key_aud)

        v_img_sp, v_img_from_A, v_img_target = self.V_img(
            index_spikes, target_enc_spikes=spike_img_target, phase=phase,
            use_delayed_target=self.use_delayed_target, delay=self.binding_delay)
        v_aud_sp, v_aud_from_A, v_aud_target = self.V_aud(
            index_spikes, target_enc_spikes=spike_aud_target, phase=phase,
            use_delayed_target=self.use_delayed_target, delay=self.binding_delay)

        return {
            "key_img": key_img, "key_aud": key_aud,
            "index_spikes": index_spikes, "index_state": index_state,
            "v_img_spikes": v_img_sp, "v_img_from_A": v_img_from_A,
            "v_img_target": v_img_target,
            "v_aud_spikes": v_aud_sp, "v_aud_from_A": v_aud_from_A,
            "v_aud_target": v_aud_target,
        }
