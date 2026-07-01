"""随机张量上的快速端到端检查（无需下载数据）。

用法：python -u scripts/smoke_test.py
"""

import bootstrap  # noqa: F401

import torch

from common import load_config, CUE_MODES
from data.corruption import corrupt_image, corrupt_audio
from models.network import CrossModalSNN
from train import compute_losses

cfg = load_config("configs/v6c.yaml")
cfg["device"] = "cpu"
model = CrossModalSNN(cfg)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

B = 4
n_mels = cfg["audio"]["n_mels"]
n_frames = cfg["audio"]["n_frames"]
x_img = torch.rand(B, 1, 28, 28)
x_aud = torch.rand(B, n_mels, n_frames)
labels = torch.randint(0, cfg["dims"]["num_classes"], (B,))

# 类别代表原型（dummy，仅冒烟用）：测试单/双模态 cue 的 target 选择路径
C = cfg["dims"]["num_classes"]
proto_img = torch.rand(C, 1, 28, 28)
proto_aud = torch.rand(C, n_mels, n_frames)

print("--- 6 种 cue 模式 binding+readout 前向/反向 ---")
for mode in CUE_MODES:
    loss, logs = compute_losses(model, x_img, x_aud, labels, mode, cfg,
                                proto_img, proto_aud, epoch=0)
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
