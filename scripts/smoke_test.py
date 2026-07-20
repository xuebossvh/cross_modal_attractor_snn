"""随机张量上的快速端到端检查（无需下载数据）。

用法：python -u scripts/smoke_test.py
"""

import bootstrap  # noqa: F401

import torch

from common import load_config, CUE_MODES
from data.corruption import corrupt_image, corrupt_audio
from models.network import CrossModalSNN
from train import (compute_losses, _masked_audio_weighted_mse,
                   _audio_coarse_loss)

cfg = load_config("configs/v11b.yaml")
cfg["device"] = "cpu"
cfg["cross_key_conditioning"]["causal_training"]["batch_probability"] = 1.0
model = CrossModalSNN(cfg)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

B = 4
n_mels = cfg["audio"]["n_mels"]
n_frames = cfg["audio"]["n_frames"]
x_img = torch.rand(B, 1, 28, 28)
x_aud = torch.rand(B, n_mels, n_frames)
labels = torch.arange(B) % cfg["dims"]["num_classes"]

# 类别代表原型（dummy，仅冒烟用）：测试单/双模态 cue 的 target 选择路径
C = cfg["dims"]["num_classes"]
proto_img = torch.rand(C, 1, 28, 28)
proto_aud = torch.rand(C, n_mels, n_frames)


def _grad_sum(parameters):
    return sum(p.grad.abs().sum().item() for p in parameters
               if p.grad is not None)


print("--- Cross-Key 零初始化、梯度与 Decoder-only 路径检查 ---")
dims = cfg["dims"]
cross_model = CrossModalSNN(cfg)
base_img = torch.rand(B, dims["N_value_img"])
base_aud = torch.rand(B, dims["N_value_aud"])
detail_img = torch.rand(B, dims["D_img"])
detail_aud = torch.rand(B, dims["D_aud"])
key_img = torch.rand(B, dims["N_key_img"])
key_aud = torch.rand(B, dims["N_key_aud"])

fused_img, img_stats = cross_model._fuse_decoder_state(
    base_img, detail_img, "img", cross_key_rate=key_aud,
    return_cross_stats=True)
fused_img_zero, _ = cross_model._fuse_decoder_state(
    base_img, detail_img, "img", cross_key_rate=key_aud,
    disable_cross_key=True, return_cross_stats=True)
fused_aud, aud_stats = cross_model._fuse_decoder_state(
    base_aud, detail_aud, "aud", cross_key_rate=key_img,
    return_cross_stats=True)
fused_aud_zero, _ = cross_model._fuse_decoder_state(
    base_aud, detail_aud, "aud", cross_key_rate=key_img,
    disable_cross_key=True, return_cross_stats=True)
assert torch.allclose(fused_img, fused_img_zero, atol=1e-7)
assert torch.allclose(fused_aud, fused_aud_zero, atol=1e-7)
assert img_stats["residual_norm"].max().item() == 0.0
assert aud_stats["residual_norm"].max().item() == 0.0

cross_projectors = list(cross_model.aud_to_img_cross_proj.parameters()) + list(
    cross_model.img_to_aud_cross_proj.parameters())
cross_gates = list(cross_model.aud_to_img_cross_gate.parameters()) + list(
    cross_model.img_to_aud_cross_gate.parameters())
cross_opt = torch.optim.Adam(cross_projectors + cross_gates, lr=1e-2)
cross_opt.zero_grad()
(fused_img.square().mean() + fused_aud.square().mean()).backward()
first_proj_grad = _grad_sum(cross_projectors)
first_gate_grad = _grad_sum(cross_gates)
assert first_proj_grad > 0.0
assert first_gate_grad == 0.0
cross_opt.step()

cross_opt.zero_grad()
fused_img_2 = cross_model._fuse_decoder_state(
    base_img, detail_img, "img", cross_key_rate=key_aud)
fused_aud_2 = cross_model._fuse_decoder_state(
    base_aud, detail_aud, "aud", cross_key_rate=key_img)
(fused_img_2.square().mean() + fused_aud_2.square().mean()).backward()
second_gate_grad = _grad_sum(cross_gates)
assert second_gate_grad > 0.0

cross_model.eval()
with torch.no_grad():
    normal = cross_model(x_img_cue=x_img, x_aud_cue=x_aud)
    conditioned = cross_model(
        x_img_cue=x_img, x_aud_cue=x_aud,
        cross_key_img_rate_override=key_img,
        cross_key_aud_rate_override=key_aud)
    wrong = cross_model(
        x_img_cue=x_img, x_aud_cue=x_aud,
        cross_key_img_rate_override=key_img.roll(1, 0),
        cross_key_aud_rate_override=key_aud.roll(1, 0))
    zero = cross_model(
        x_img_cue=x_img, x_aud_cue=x_aud,
        disable_img_to_aud_cross=True, disable_aud_to_img_cross=True)
assert torch.allclose(normal["index_state"], conditioned["index_state"], atol=1e-7)
assert torch.allclose(conditioned["index_state"], wrong["index_state"], atol=1e-7)
assert (not torch.allclose(conditioned["recovered_img_coarse"],
                           wrong["recovered_img_coarse"], atol=1e-7)
        or not torch.allclose(conditioned["recovered_aud_coarse"],
                              wrong["recovered_aud_coarse"], atol=1e-7))
assert zero["aud_to_img_cross_ratio"].max().item() == 0.0
assert zero["img_to_aud_cross_ratio"].max().item() == 0.0
print(f"first projector grad={first_proj_grad:.6f}, "
      f"first gate grad={first_gate_grad:.6f}, "
      f"second gate grad={second_gate_grad:.6f}")

print("--- v11b weighted normalization 与 coarse loss 检查 ---")
mask = torch.ones_like(x_aud)
constant_rec = x_aud + 0.2
wmse_0 = _masked_audio_weighted_mse(
    constant_rec, x_aud, mask, gamma=0.0)
wmse_5 = _masked_audio_weighted_mse(
    constant_rec, x_aud, mask, gamma=5.0)
assert torch.allclose(wmse_0, wmse_5, atol=1e-6)
assert _audio_coarse_loss(constant_rec, x_aud, mask).item() > 0.0

print("--- 8 种 cue 模式 binding+readout 前向/反向 ---")
for mode in CUE_MODES:
    loss, logs = compute_losses(model, x_img, x_aud, labels, mode, cfg,
                                proto_img, proto_aud, epoch=0)
    assert "aud_coarse" in logs
    if mode in ("clean_img_corrupt_aud", "corrupt_img_clean_aud",
                "corrupt_both"):
        assert "cross_pair" in logs
    opt.zero_grad()
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in model.parameters()
                if p.grad is not None)
    opt.step()
    parts = " ".join(f"{k}={v:.3f}" for k, v in logs.items())
    print(f"cue={mode:18s} loss={loss.item():.4f} grad_sum={gnorm:.1f} | {parts}")

print("--- 损坏函数检查 ---")
print("corrupt_image:", corrupt_image(x_img, "random", 0.5).shape)
print("corrupt_audio:", corrupt_audio(x_aud, "random", 0.5).shape)

print("--- 推理（target 路径强制关闭） ---")
out = model.infer(x_aud_cue=x_aud)
print("音频cue -> 图像:", out["recovered_img"].shape,
      " 音频:", out["recovered_aud"].shape, " logits:", out["logits"].shape)
out = model.infer(x_img_cue=x_img)
print("图像cue -> 图像:", out["recovered_img"].shape,
      " 音频:", out["recovered_aud"].shape)
print("OK")
