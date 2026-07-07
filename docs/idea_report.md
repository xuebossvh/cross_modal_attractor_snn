# 研究设计记录：Cross-Modal Attractor SNN
> 创建时间：2026-07-07 | 当前用途：F 阶段迭代补充记录
> 说明：本文件先记录当前 D-F 迭代中已经确认的实验设计，不回填完整 A/B/C 阶段长报告。

---

## F 阶段补充：v10c 实验设计

### 背景诊断

v10b 已将音频残缺方式收窄到与 long audio inpainting 更一致的连续时间片段缺失，并通过 `fixed_mask` 协议完成评估。完整评估显示，v10b 的分类与图像恢复链路仍可工作，但音频恢复出现明显能量塌缩：`recovered_aud` 均值接近 0，`audSSIM` 约 0.13-0.14，top15% 能量召回接近随机水平。

因此 v10c 不改变核心 CrossModalSNN 架构，而是修正训练协议，让训练任务与 fixed-mask 论文主评估更加一致，并降低 1 秒 FSDD 数字语音上的长缺失难度。

### v10c 目标

1. 图像残缺训练与评估统一为 `occlusion`，不再训练时随机切换多个图像 family。
2. 图像和音频主残缺强度的后期上限均设为 `0.4`，避免 `0.5` 在短语音和 28x28 图像上过难。
3. decoder pretrain 从 8 轮增加到 25 轮，并加入固定 family 的 corrupt detail 输入，使 decoder 在进入主训练前见过 `occlusion` 图像 cue detail 和 `time_mask` 音频 cue detail。
4. 主训练轮数设为 70 轮，给 25 轮预训练后的 binding/readout 阶段足够时间适应固定残缺协议。

### 主要实验设置

| 项目 | v10b | v10c |
|------|------|------|
| 图像训练 family | `random` | `occlusion` |
| 图像 fixed eval family | `occlusion` | `occlusion` |
| 音频训练 family | `time_mask` | `time_mask` |
| 音频 fixed eval family | `time_mask` | `time_mask` |
| 后期 severity | `0.5` | `0.4` |
| decoder pretrain | 8 轮 clean target Value + detail dropout | 25 轮 clean target Value + fixed corrupt detail + masked audio loss |
| 主训练 | 配置默认 50 轮，实际可覆盖 | 配置默认 70 轮 |

### 预期观察指标

v10c 的优先目标不是单纯提高分类 ACC，而是修复音频恢复：

- `audSSIM`：应显著高于 v10b 的约 0.13-0.14。
- `rec_mean/std/max`：应远离近零输出，接近 target 能量统计。
- `top15%召回`：应高于 v10b 的约 15%-20%。
- `aud_masked_mse`：在 `corrupt_aud_only` 和 `corrupt_both` 下应下降。
- `ACC` 与图像 SSIM：应尽量保持 v10b 的可用水平。

### 评估协议

v10c 主结论仍使用：

```bash
python -u scripts/evaluate.py --config configs/v10c.yaml --protocol fixed_mask --family_breakdown
```

`legacy_random` 仅作为额外鲁棒性检查，不作为论文主对齐结论。
