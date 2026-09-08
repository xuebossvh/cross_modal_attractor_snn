# 研究设计记录：Cross-Modal Attractor SNN
> 创建时间：2026-07-07 | 当前用途：F 阶段迭代补充记录
> 说明：本文件先记录当前 D-F 迭代中已经确认的实验设计，不回填完整 A/B/C 阶段长报告。
> **纪律**：每次 F 阶段版本迭代（新配置、结构改动、消融设计）须同步更新本文件，与 `docs/implementation.md`（先写）和 `docs/dev_log.md`（后写）配套。

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

---

## F 阶段补充：v10d 实验设计

### 背景诊断

v10c 在 v10b 基础上修正了训练协议（固定 `occlusion` 图像 family、severity 上限 0.4、25 轮 corrupt-aware decoder pretrain），但**未改动音频恢复模块结构**。对照《Deep Long Audio Inpainting》，v10c 仍缺少：

1. **F3a**：decoder refine 块仅为普通 3×3 Conv2d，感受野小，难以覆盖长缺失区。
2. **F3b**：无谱图空间 refiner；decoder 从不直接利用残缺 cue 的空间结构，只能靠 Value + 128 维 detail 从头生成。
3. **F4**：无 SSIM / masked TF-grad / 冻结 encoder feature loss 等附加质量项（代码与消融入口）。

v10d 在 v10c 训练协议不变的前提下，首次落地 F3a/F3b/F4 框架，并做保守 cue 比例调整。v10d 修订版进一步对齐文献式 inpainting：**可见区 paste-back 到 aud_cue**，refiner 感受野加深，并补充归因消融配置。

### v10d 目标

1. **F3a**：`AudioDecoder` refine 块改为 gated conv + dilation(1,2,4)（`snn.aud_refine_type: gated_dilated`），扩大缺失区感受野。
2. **F3b**：启用谱图空间 `AudioRefiner`，输入 `[coarse_rec, aud_cue, mask]`，仅在缺失区修正；`visible_paste_back: true` 时 `final = mask*pred + (1-mask)*aud_cue`（文献式：缺失区补全 + 可见区原样保留 cue）。
3. **F4**：代码就绪、`lambda_aud_ssim` / `lambda_aud_masked_grad` / `lambda_aud_feat` 默认 0，避免与 refiner 提升混淆归因。
4. **cue 比例**：相对 v10c 保守小幅调整（corrupt 60% / clean 40%），提高 corrupt-audio 训练占比。
5. **归因可分离**：通过消融配置把「结构增益」与「cue 比例增益」分开测量。

### 主要实验设置

| 项目 | v10c | v10d |
|------|------|------|
| AudioDecoder refine | plain Conv2d ×2 | gated_dilated（dilation 1,2,4）|
| 谱图 refiner | 无 | `audio_refiner.enabled: true`，blocks=4，dilation 1,2,4,8，RF ~31 帧 |
| 可见区合成 | decoder 重建全谱 | `visible_paste_back: true` → 可见区 = aud_cue |
| F4 质量 loss | 无 | 代码就绪，权重默认 0 |
| cue 比例 corrupt/clean | 50% / 50% | 60% / 40%（见下表） |
| refiner 与 pretrain | — | refiner **不入** decoder pretrain，主训练从零学 |
| checkpoint 兼容 | — | 与 v10c 不兼容，需从头训练 |

**v10d cue_modes（合计 1.0）**

| 模式 | v10c | v10d |
|------|------|------|
| corrupt_img_only | 0.10 | 0.15 |
| corrupt_aud_only | 0.20 | 0.20 |
| corrupt_both | 0.20 | 0.25 |
| clean_img_only | 0.10 | 0.10 |
| clean_aud_only | 0.20 | 0.15 |
| clean_both | 0.20 | 0.15 |

其余训练协议与 v10c 相同：`occlusion` 图像、`time_mask` 音频、severity 0.4、decoder pretrain 25 轮、主训练 70 轮、`lambda_aud_masked: 1.0`。

### 归因消融设计

v10d 同时改了结构与 cue 比例，**主结论不能单独归因于 refiner**。须跑以下对照：

| 配置 | 相对 v10d 的差异 | 隔离的问题 |
|------|------------------|------------|
| `configs/v10d.yaml` | 完整 v10d（结构 + 新比例 + paste-back + refiner） | 主实验 |
| `configs/v10d_ablation_v10c_ratio.yaml` | v10d 结构，cue 比例换回 v10c | v10d vs 本配置 → **结构在旧比例下的净效果**；本配置 vs v10c → 同比例下结构改动 |
| `configs/v10d_ablation_refiner_off.yaml` | v10d 比例与 gated_dilated decoder，`audio_refiner.enabled=false`；由于 paste-back 位于 refiner 分支内，该配置同时关闭 refiner + paste-back 后处理 | v10d vs 本配置 → **refiner/paste-back 后处理路径的合并贡献**，不能解读为只关 refiner |

建议解读顺序：

1. `v10c` → `v10d_ablation_v10c_ratio`：在**相同 cue 比例**下看 F3a+F3b 结构是否改善音频恢复。
2. `v10d_ablation_refiner_off` → `v10d`：在**相同比例与 decoder** 下看 refiner + paste-back 后处理路径的合并增量。
3. `v10c` → `v10d`：端到端总增益（结构 + 比例 + refiner，**不可单独归因**）。

### 预期观察指标

v10d 优先改善**缺失区**音频恢复（paste-back 会使可见区指标趋近完美，不宜作为主对照）：

- `aud_masked_mse` / `aud_masked_l1`（paper_aligned `time_mask` 组）：应较 v10c 显著下降。
- `aud_visible_mse`：paste-back 开启时应接近 0（sanity check，非主结论）。
- `audSSIM`：全谱指标会受可见区 paste 抬高，解读时以 masked 指标为主。
- `rec_mean/std/max`：应远离近零塌缩，接近 target 能量。
- `top15%召回`：应高于 v10c。
- **分类 ACC**（fixed_mask）：底线不破——`corrupt_aud ≥95%`、`corrupt_both ≥98%`、`clean_both ≥99%`。

### 风险与边界（实验表述口径）

1. **不是完整文献 inpainting 系统**：仍是 cross-modal attractor SNN；refiner 只修正缺失区，记忆主干（Key/Index/Value）不变。
2. **refiner 未预训练**：主训练阶段从零学习，不是「预训练好的 inpainting refiner」。
3. **demo 与 evaluate 对齐（v10e P0-A）**：`fixed_mask` / `legacy_random` 均透传 `aud_mask` 至 `forward(aud_cue_mask=...)`；demo 指标基于最终 `recovered_aud`；图内展示 `audio mask` 与 `coarse audio` 列。
4. **F4 默认关闭**：后续若打开 `lambda_aud_feat` 等，须单独消融，且 encoder 已严格冻结（梯度不进 aud_encoder）。

### 评估协议

v10d 主结论使用：

```bash
python -u scripts/evaluate.py --config configs/v10d.yaml --protocol fixed_mask --family_breakdown
```

归因消融：

```bash
python -u scripts/evaluate.py --config configs/v10d_ablation_v10c_ratio.yaml --protocol fixed_mask --family_breakdown
python -u scripts/evaluate.py --config configs/v10d_ablation_refiner_off.yaml --protocol fixed_mask --family_breakdown
```

## F 阶段补充：v10e 实验设计

### 背景诊断

v10d 只把音频分支改成 inpainting 形式：音频有残缺 cue 和 `aud_mask` 时走 `AudioRefiner + visible paste-back`，而图像分支仍由 `ImageDecoder` 整图重建。这样在 `corrupt_img_only` 和 `corrupt_both` 下，图像可见区域没有被显式保留，图像恢复口径与音频不对称。

### v10e 目标

1. 图像分支新增 `ImageRefiner`，输入 `[coarse_img, img_cue, img_mask]`，只在图像 cue 残缺且 mask 存在时启用。
2. 图像最终输出采用 `final_img = img_mask * pred_img + (1-img_mask) * img_cue`，使可见区原样保留，缺失区由模型补洞。
3. clean cue 不启用 refiner：没有 mask 时不 paste-back，仍用普通 decoder reconstruction。
4. 缺失模态不启用 refiner：该模态没有 cue 和 mask，只能由 attractor memory 联想/生成。
5. 评估和可视化同时给出图像/音频 masked 指标，避免全图/全谱指标被 paste-back 抬高后误解。

### v10e 关键对照口径

| cue mode | 图像路径 | 音频路径 |
|------|------|------|
| `corrupt_img_only` | 图像有 `img_mask`，启用 ImageRefiner + paste-back | 音频无 cue，只能 category-level 联想 |
| `corrupt_aud_only` | 图像无 cue，只能 category-level 联想 | 音频有 `aud_mask`，启用 AudioRefiner + paste-back |
| `corrupt_both` | 图像和音频都启用对应 refiner + paste-back | 图像和音频都启用对应 refiner + paste-back |
| `clean_img_only` | 图像 clean cue，无 mask，不启用 refiner | 音频无 cue，只能 category-level 联想 |
| `clean_aud_only` | 图像无 cue，只能 category-level 联想 | 音频 clean cue，无 mask，不启用 refiner |
| `clean_both` | 双模态 clean cue，无 mask，不启用 refiner | 双模态 clean cue，无 mask，不启用 refiner |

### 预期观察指标

- `img_masked_mse`：v10e 图像补洞主指标，应优先用于评价 `corrupt_img_only/corrupt_both`。
- `img_visible_mse`：paste-back sanity check，开启后应接近 0；不作为主结论。
- `aud_masked_mse`：沿用 v10d 音频补洞主指标。
- 全图 `imgMSE/SSIM` 与全谱 `audMSE/audSSIM` 会受 paste-back 抬高，论文表述时必须注明。

`legacy_random` 仅作泛化参考，不作为论文主对齐结论。论文主对照指标只引用 `paper_aligned_time_gap`（`time_mask`）组的 masked 指标。

## F 阶段补充：v10f 实验设计（归因补齐）

### 背景

v10e 已具备对称 Image/Audio Refiner + paste-back，但 v10d 消融显示全谱指标可能被 paste-back 抬高；且 refiner 在主训阶段从零学习、pretrain 未对齐 refiner 路径，难以拆分「贴回 vs delta 修正 vs pretrain vs lr_mult」的贡献。

### v10f 目标

1. 抽出 refiner helper，统一 train/eval/pretrain 的 final 合成路径。
2. refiner-aware decoder pretrain（可选，主配置开启）；关闭该开关的消融必须直接使用 coarse decoder 输出，不能把冻结 refiner 放进 pretrain 前向；主训 refiner 使用 `lr_mult=0.5`。
3. evaluate 默认输出 coarse→final masked/visible 指标 + `[归因]` 表。
4. paste-back 分离消融：`pasteback_off`（无可见区贴回）、`pasteback_only`（仅贴回无 delta）、`no_refiner_pretrain`（拆 pretrain 贡献）。

### 关键解读口径（锁定）

| 指标 | 用途 |
|------|------|
| `*_coarse_masked_mse` | decoder 洞内重建基线 |
| `*_masked_mse`（final） | refiner/paste-back 后洞内误差；主结论 |
| `*_coarse_visible_mse` | decoder 对可见区的改写程度 |
| `*_visible_mse`（final） | paste-back sanity；≈0 **不代表**可见区学习 |
| full vs pasteback_only masked 差距 | delta refiner 的洞内贡献 |

### 消融矩阵

| 配置 | 作用 |
|------|------|
| `v10f.yaml` | 主实验：refiner + paste-back + refiner pretrain + lr_mult |
| `v10f_ab_audio_pasteback_off` | 音频 refiner 开、但不贴回可见 cue |
| `v10f_ab_audio_pasteback_only` | 音频仅 paste-back、无 refiner 模块 |
| `v10f_ab_image_pasteback_off` | 图像同上（推荐） |
| `v10f_ab_image_pasteback_only` | 图像仅 paste-back（推荐） |
| `v10f_ab_no_refiner_pretrain` | 结构/lr_mult 同主配置，关 refiner pretrain；pretrain 阶段直接以 coarse decoder 输出计算重建损失 |

### v10f 鲁棒性协议补丁：5-family corruption

为增强鲁棒性，v10f 在保留 refiner/paste-back 归因结构的前提下，把主训练和 fixed-mask 主评估从单一 `occlusion/time_mask` 扩展为 image/audio 各 5 种带 mask 的缺失 family。

| 模态 | family |
|------|--------|
| image | `occlusion`, `pixel_delete`, `mask_vertical`, `mask_horizontal`, `salt_mask` |
| audio | `time_mask`, `freq_mask`, `feature_dropout`, `partial_temporal`, `time_freq_block` |

实验口径调整：

1. 训练阶段 balanced sampling：同一套 cue mode 概率不变，但 corrupt image/audio cue 的残缺 family 在上述 5 种内轮换。
2. fixed-mask 主评估按 5 个 family pair 展开，仍使用固定 seed，保证跨版本可比。
3. demo fixed-mask 可视化使用 10 个样本，每个 family 2 个样本，不允许音频或图像残缺退回 clean cue。
4. Gaussian 噪声暂不纳入主 5 family。原因是当前 Image/AudioRefiner 与 paste-back 依赖 `mask=1` 的缺失语义；Gaussian 是全域连续扰动，不天然对应“洞”。`salt_mask` 已作为带 mask 的白色缺失 family 纳入主协议，但它不是严格的 Gaussian denoising；若研究 noisy restoration，应作为单独 denoising 扩展实验。

## F 阶段补充：v11a 实验设计（跨模态辅助恢复基线）

### 背景诊断

v10f 的 `time_mask` 评估中，`corrupt_aud_only` 到 `corrupt_both` 的分类 ACC 明显提高，但 `audMaskMSE` 基本不变。该结果不能直接解释为“图像对音频恢复无效”，因为原 `corrupt_both` 同时破坏图像和音频，而且两种模式未显式保证复用同一音频 mask。与此同时，v10f 五个音频 family 平衡采样使 `time_mask` 只占全部主训练 batch 的约 9%，其专项结果较 v10e 退化；`partial_temporal` 固定遮挡张量尾部，又可能只遮到 FSDD 的 padding/silence。

### v11a 目标

1. 新增 `clean_img_corrupt_aud`，在完全相同的残缺音频上比较“无图像”与“增加 clean 图像”对分类、coarse audio 与 final audio 的影响。
2. 对称新增 `corrupt_img_clean_aud`，比较 clean 音频是否帮助残缺图像恢复。
3. 保留五 family 鲁棒性，同时用最后 25 轮 `time_mask` 权重 0.60 的专项训练恢复长时间缺失能力。
4. 将 `partial_temporal` 改为 active-aware trailing mask，使缺失区覆盖真实语音尾部而不是固定长度张量的静音尾部。
5. 在既有 masked L1/MSE 上增加缺失区 energy-weighted MSE，防止大量低能量像素主导优化。

### 8 种 cue mode 与目标

| cue mode | image cue | audio cue | image target | audio target | 用途 |
|---|---|---|---|---|---|
| `corrupt_img_only` | corrupt | absent | sample | category | 图像单模态补洞 |
| `corrupt_aud_only` | absent | corrupt | category | sample | 音频单模态补洞 |
| `corrupt_both` | corrupt | corrupt | sample | sample | 双残缺鲁棒性 |
| `clean_img_corrupt_aud` | clean | corrupt | sample | sample | 图像辅助音频恢复的核心对照 |
| `corrupt_img_clean_aud` | corrupt | clean | sample | sample | 音频辅助图像恢复的核心对照 |
| `clean_img_only` | clean | absent | sample | category | 图像单模态 clean 基线 |
| `clean_aud_only` | absent | clean | category | sample | 音频单模态 clean 基线 |
| `clean_both` | clean | clean | sample | sample | 双模态 clean 上限参考 |

### 关键比较与判据

1. 图像是否帮助音频：比较 `corrupt_aud_only` 与 `clean_img_corrupt_aud`，要求同一音频样本、同一 audio family、同一 `aud_mask`。主指标为 `aud_coarse_masked_mse`、`aud_masked_mse` 与 ACC。
2. 音频是否帮助图像：比较 `corrupt_img_only` 与 `corrupt_img_clean_aud`，要求同一图像样本、同一 image family、同一 `img_mask`。主指标为 `img_coarse_masked_mse`、`img_masked_mse` 与 ACC。
3. `corrupt_both` 继续作为双残缺鲁棒性实验，但不再承担“clean 第二模态是否提供帮助”的单独归因。
4. `partial_temporal` 的 MaskMSE 必须结合缺失区 target energy 解读；若 active-aware 后仍接近 0，需检查样本本身是否近静音，不能直接宣称恢复完美。

### 本轮边界

v11a 基线首轮只建立公平输入与训练基线，不把 image key/index 直接注入 AudioRefiner，也不把 audio key/index 直接注入 ImageRefiner。后续 0.8 补丁仍不注入 Refiner，而是在 Decoder 前加入对侧 Key 条件 residual；是否有效必须由下方 normal/zero/wrong 配对 maskedMSE 判据决定。

### v11a 后续补丁：对侧 Key 条件化 Decoder

用户确认在现有 v11a 分支继续实现跨模态条件注入。结构选择为 **Key-conditioned Value residual**，而不是 Key 直接输入 Refiner：

```text
图像辅助音频：K_img -> rate/detach -> projector/gate -> V_aud residual -> AudioDecoder
音频辅助图像：K_aud -> rate/detach -> projector/gate -> V_img residual -> ImageDecoder
```

该设计的研究假设是：共享 Index/Value 已提供类别原型，但可能在 attractor bottleneck 中损失部分对侧模态语义；对侧 Key residual 在 Decoder 前显式补回类别条件，同模态 detail 继续提供笔迹/音色等局部信息，Refiner 仍只负责 mask 区补洞。

第一版采用以下归因边界：

1. Key rate 与原始 Value 均按既有隔离策略 detach；重建 loss 只训练 cross projector/gate、Decoder/detail 与 Refiner，不直接扰动 Key/Index/Value。
2. normal/zero/wrong 干预仅替换 Decoder 条件副本，原始 cue 驱动的 Key->Index->Value 全程不变。
3. 有效证据必须同时满足 `correct maskedMSE < zero maskedMSE` 与 `wrong maskedMSE > correct maskedMSE`；只有 wrong 退化而 correct≈zero，不能宣称正确 Key 有益。
4. 主要方向化模式为：image->audio 使用 `clean_img_corrupt_aud` / `corrupt_both`；audio->image 使用 `corrupt_img_clean_aud` / `corrupt_both`。无对侧 Key 或无目标 mask 时归因指标记为 N/A。
5. 全谱 MSE/SSIM 继续受 paste-back 影响；主结论使用 coarse/final maskedMSE、`correct_gain`、`wrong_damage` 与 ACC 路径隔离检查。

公平实验采用同版本 control：`v11a.yaml` 开启 cross conditioning，`v11a_control.yaml` 关闭前向条件路径但仍构造相同 projector/gate 参数；两者必须使用相同 seed、模型构造顺序、训练预算和除 cross path 开关外的全部设置。这样可避免“少构造四个 Linear 导致后续 Decoder/Refiner 初始化随机数错位”的混杂。若使用已有 v11a checkpoint 做快速续训，control 与 enabled 必须从同一 checkpoint 出发并训练相同轮数；论文主结果仍需同预算训练。

## F 阶段补充：v11b 恢复稳定化与 Cross-Key 因果训练

### v11a 结果诊断

同为 120 轮的 v11a 与同构 control 在音频 family 上得到几乎逐项相同的 `audMaskMSE`，并同时出现 clean audio SSIM 约 0.14--0.17、coarse audio 低能量的问题。由此不能把音频退化归因于 Cross-Key；共同训练目标才是优先排查对象。当前 `energy-weighted masked MSE` 用 mask 像素数而非加权 mask 总和作分母，会同时改变洞内关注与整体梯度尺度；主训练又只监督 paste-back 后的 final audio，没有直接约束 `recovered_aud_coarse`。Cross-Key residual 虽有非零 gate 与 norm ratio，但 correct/zero/wrong 配对 maskedMSE 基本不变，说明 Decoder 可忽略该条件。

### v11b 目标与边界

1. 先在 Cross-Key 关闭时恢复可用 coarse audio，避免将“基础重建塌缩”误判为“跨模态条件无效”。
2. 将 weighted masked MSE 改为按 `sum(weight * mask)` 归一化，使其只重分配缺失区权重；主恢复配置先关闭该项，再用独立分支验证。
3. 在主训练中对 `[B,64,64]` 的 `recovered_aud_coarse` 增加直接监督，禁止 AudioDecoder 依赖 Refiner/paste-back 逃避生成。
4. 基线恢复后，从同一 checkpoint 分叉 control、无因果 loss 的 Cross-Key、带配对因果 loss 的 Cross-Key，隔离结构存在与因果训练目标的贡献。
5. 不在 v11b 同时引入 FiLM、Cross-Attention、F4 质量 loss 或 Key-to-Refiner 新路径；现有 Key-conditioned Value residual 保持不变。

MNIST 与 FSDD 配对只共享数字标签，不共享说话人、音高、时序或笔迹风格。v11b 对 Cross-Key 的合理主张是“在严重缺失下提供类别语义消歧”，而不是从图像恢复特定说话人细节、或从音频恢复特定 MNIST 笔迹。sample-level MaskMSE 仍是恢复主指标，但必须同时报告 correct/zero/wrong Key 的差值；只有 gate 非零不能构成跨模态贡献证据。

### 五实验分叉

共享 decoder pretrain 25 轮；主训练先完成 100 轮 Recovery trunk，再按下表分叉。所有分支复用同一恢复 checkpoint，不重复预训练。

| 配置 | 从何处开始 | 新增轮数 | 累计主训练轮数 | 目的 |
|---|---|---:|---:|---|
| `v11b_recovery` | Recovery trunk | 20 | 120 | weighted off 的恢复基线 |
| `v11b_weighted` | 同一 100 轮 trunk | 20 | 120 | 归一化 weighted loss 的净贡献 |
| `v11b_control` | 前两者中恢复更好的 120 轮 checkpoint | 30 | 150 | Cross-Key 关闭的同预算终点 |
| `v11b_cross_no_causal` | 同一选定 checkpoint | 30 | 150 | Cross-Key 开启、无配对因果 loss |
| `v11b` | 同一选定 checkpoint | 30 | 150 | 主实验：Cross-Key 开启、带配对因果 loss |

恢复 checkpoint 的“选择”必须在训练前固定判据：优先比较 4-family audio MaskMSE、time-mask MaskMSE、clean audio SSIM 与 coarse energy；不得看到最终 Cross-Key 结果后反向选择。若希望论文级严格比较，三条 30 轮分支使用相同 seed、batch 顺序、mask 和 scheduler 位置，并记录父 checkpoint 哈希。

### 损失设计

归一化 weighted masked MSE：

```text
w = 1 + gamma * target
L_weighted = sum(w * mask * (pred-target)^2) / sum(w * mask)
```

主训练 coarse audio 辅助监督不经过 paste-back：有结构 mask 时使用 masked L1 + masked MSE；clean cue、无 mask 或 category target 时使用 full L1 + full MSE。默认 `lambda_aud_coarse=0.5`，日志独立记录 `aud_coarse_l1`、`aud_coarse_mse` 与 `aud_coarse_mask_mse`，不得混入 final 指标。

Cross-Key 因果训练只在有目标 mask 且有对侧 Key 的方向化模式生效：image->audio 使用 `clean_img_corrupt_aud` / `corrupt_both`，audio->image 使用 `corrupt_img_clean_aud` / `corrupt_both`。对有效 batch 构造 zero 与 wrong-class Key，并以 coarse maskedMSE 定义：

```text
L_cross = relu(E_correct - E_zero + margin)
        + relu(E_correct - E_wrong + margin)
```

默认仅以 `causal_batch_probability=0.25` 采样配对分支，`causal_loss_weight=0.1`，margin 取 detached zero error 的 `0.05` 倍。zero/wrong reference error 均 detach，梯度只推动 correct coarse error 下降，禁止通过主动恶化 zero/wrong 输出满足 margin。正常恢复 loss 锚定 correct 输出，Key rate 保持 detach；因果 loss 不得作用于全谱 SSIM 或 paste-back 后的可见区。

Recovery 在第 100 轮额外保存包含 model、optimizer、scheduler 与 epoch 的里程碑 checkpoint。`v11b_weighted` 必须从该完整状态继续 20 轮，使其与 recovery 后 20 轮只相差 weighted loss。三个 Cross-Key 分支从选定的第 120 轮 checkpoint 只加载 model，统一重建固定低学习率 optimizer，并冻结 Encoder、Memory 与 Refiner；仅 Decoder 和 Cross-Key adapter 可训练。这样三分支共享完全相同的父模型与微调预算。

### 验收标准

1. Recovery 阶段第 30 轮检查：clean audio SSIM 应超过 0.5、time-mask MaskMSE 应低于 0.018、coarse audio 不得持续近黑；否则停止长训练并回到 F-1。
2. 完整 Recovery 目标：time-mask MaskMSE 不高于 0.010，4-family audio MaskMSE 接近 v10f 的 0.005--0.007，clean audio SSIM 高于 0.80。
3. Cross-Key 目标：enabled 优于同预算 control；同一 checkpoint 内 `E_correct < E_zero` 且 `E_correct < E_wrong`，建议至少达到 5% 相对 maskedMSE 改善。评估额外加入 same-class different-sample Key：若其结果接近 correct，说明路径主要提供类别语义；若显著更差，才支持 Key 含有可迁移的实例信息。
4. `partial_temporal` 保留 active-aware 语义并单独报告，不与 v10f 静音尾部版本直接作强纵向比较。

## F 阶段补充：v11c AudioRefiner-free 稳定基线

### 结果动机

v11b 五组实验共同表明：新增 coarse audio 监督已经把四个非
`partial_temporal` family 的 coarse MaskMSE 恢复到约 `0.0071`，但外置
AudioRefiner 将 final 恶化到约 `0.0225`；五 family 则约为
`0.0130 -> 0.0297`。该恶化在 recovery、weighted、control、无因果 Cross-Key 和
causal Cross-Key 中一致，demo 也显示多个缺失区的 final 比 coarse 更接近零能量。
因此下一轮首先移除这一后处理混杂，再判断对侧 Key 是否影响 Decoder coarse。

### v11c 假设

1. AudioDecoder 内部 gated/dilated refinement 已能形成可用 coarse audio；外置
   AudioRefiner 在当前训练目标下产生负增益，不应继续作为默认 final 路径。
2. visible-region paste-back 仍然必要，它只保护已知 cue，不修改缺失区，不能与
   AudioRefiner 的 delta 贡献混为一谈。
3. 旁路 AudioRefiner 后，final 缺失区严格等于 coarse，Cross-Key 对 Decoder 的
   任何真实影响都不会再被后处理抵消。
4. 本轮结论是“当前 AudioRefiner 暂时停用”，不是证明该结构永久无效；历史 v10f
   仍观察到过正向 refiner 增益，后续可在稳定 coarse 上单独重训。

### 实验设计

为保持与 v11b 的严格可比性，AudioRefiner 模块仍构造以 strict 加载共同父模型，
但通过 `bypass=true` 禁止前向并冻结参数。`v11c_control` 与 `v11c` 都从
`v11b_recovery` 第 120 轮 checkpoint 开始，用相同 seed 和 30 轮预算微调同一组
Decoder/Cross-Key 参数；两者仅差 Cross-Key/causal 开关。

| 配置 | 目的 | 主要判据 |
|---|---|---|
| `v11c_control` | AudioRefiner bypass、Cross-Key 关闭 | 建立 coarse+paste-back 同预算基线 |
| `v11c` | AudioRefiner bypass、Cross-Key causal 开启 | 判断 image/audio Key 是否真正降低对侧 coarse/final MaskMSE |

验收顺序：先确认每个带 mask 的音频模式满足 `aud_final_mask_mse ==
aud_coarse_mask_mse` 且 final visible MSE 约为 0；再要求 v11c 相对 control 的
方向化 MaskMSE 改善，并在同 checkpoint sweep 中形成
`E_correct < E_zero`、`E_correct < E_wrong`。若仍无差异，应停止扩大 Cross-Key，
把结果表述为当前配对数据只提供类别语义、不能恢复音频实例细节。

## F 阶段补充：v11c / v11d / v11e 仓库评价与后续路线

### 评价边界

本节根据仓库中的远端分支 `origin/v11c`、`origin/v11d` 与 `origin/v11e`
进行方法与实验设计评价。当前本地 `outputs/` 未包含 v11c/v11d/v11e 的评估
日志或 checkpoint，因此以下内容不是数值实验结论，而是下一轮实验前的设计
诊断与归因口径整理。

### v11c：AudioRefiner-free 稳定基线

v11c 的合理性在于先去掉 v11b 中已经被诊断为负贡献的外置 AudioRefiner：
配置仍构造 `AudioRefiner` 以兼容父 checkpoint，但 `audio_refiner.bypass=true`
使音频 final 直接来自 `AudioDecoder`。这样可以把问题重新收敛到
`Value_from_A + cue detail + Cross-Key` 的 decoder 输入融合本身。

v11c 还保留 `detail_conditioning.detach_value_for_recon=true`，避免恢复 loss
通过 `V_from_A` 反向改变 Index/Value。这个边界对本项目很重要：分类与联想
主干应先维持稳定，decoder 侧的样本细节补偿由 detail 和 cross adapter 承担。

v11c 的实验主问题不是“恢复是否好看”，而是：

```text
AudioRefiner 旁路后，audio final MaskMSE 是否严格等于 coarse MaskMSE？
Cross-Key enabled 是否优于同父 checkpoint 的 disabled control？
同一 checkpoint sweep 中是否满足 correct < zero 且 correct < wrong/same？
```

如果 v11c 仍然不能形成稳定 Cross-Key 贡献，则说明对侧 Key 在 MNIST/FSDD
配对任务中主要提供类别语义，而不是可迁移的实例细节。

### v11d：固定伪配对与 Cross-Detail

v11d 在 v11c 的基础上新增两项：固定一一伪配对和 Cross-Detail。固定配对使每个
MNIST item 对应一个确定的 FSDD 基础录音及确定性增强种子，并返回稳定 `pair_id`。
这让缺失模态的 sample target 在工程上变得明确，避免旧版训练中同类别音频目标
随 epoch 变化。

Cross-Detail 的结构路径是：

```text
对侧 encoder pre-key instance spikes
  -> rate()
  -> Linear projector
  -> vector gate(base_value, projected_detail, missing_ratio)
  -> 加到目标模态 detail channel
  -> Decoder
```

它与 Cross-Key 分工不同：Cross-Key residual 加到 `Value` 侧，主要补类别语义；
Cross-Detail residual 加到 `detail` 侧，试图补实例细节。v11d_control 只关闭
Cross-Detail，仍保留 Cross-Key，因此可以相对干净地隔离 Cross-Detail 的贡献。

但 v11d 的根本限制来自数据，而不是模块本身：MNIST 手写图像和 FSDD 语音并非
同一真实事件，二者只共享数字类别。所谓 fixed one-to-one pair 是人为绑定，
不能提供真实的说话人、发音时序、笔迹风格或视觉-语音同步关系。因此 v11d 若
出现弱增益甚至无增益，优先解释应是“伪配对不可泛化”，不应简单归因于
Cross-Detail 结构无效。

v11d 的结果解释必须遵守以下口径：

1. 只把 v11d 当作伪配对压力测试与机制审计。
2. 只有 `correct_gain > 0` 且 `same-class wrong` 明显劣于 correct 时，才说明
   Cross-Detail 携带了超越类别的实例信息。
3. 若 same-class wrong 与 correct 接近，则说明模型主要使用类别线索，不能宣称
   恢复了配对实例细节。

### v11e：真实 GRID 视听配对主线

v11e 是当前更值得作为主线的版本。它把数据域改为 GRID，同一 manifest row 包含
一次真实 utterance 的音频与视频帧；图像输入是 28x28 rank-pooled mouth-motion
dynamic image，音频输入是同一 utterance 的 log-mel。manifest 强制：

```text
source_id == image_source_id == audio_source_id
```

并要求 `pair_id` 唯一、speaker-disjoint train/val/test、验证和测试不做增强。
这使跨模态恢复从“按类别伪配对”转为“真实同源事件配对”，科学问题更清楚。

v11e 从头训练是合理的，因为 MNIST/FSDD checkpoint 与 GRID 的输入域不兼容：
图像不再是手写数字，而是嘴部运动动态图；音频采样率、时长和归一化统计也不同。
继续加载 v11c/v11d 权重会把旧域偏置带入新任务。

v11e 新增 `pair_alignment`，从 encoder pre-key instance rate 投影得到
image/audio pair embedding，用同类不同 source 作为 hard negatives。这个损失和
evaluate 中的 exact-pair Recall@1 是判断实例级跨模态对齐是否成立的关键指标，
比只看分类 ACC 或全谱 MSE 更符合 v11e 的问题定义。

v11e 需要特别注意一个实验设计风险：主配置的 validation score 包含 pair retrieval
项，而 control 关闭 pair alignment 后 `lambda_pair=0`。这会导致 best checkpoint
的选择标准不完全一致。正式比较时应至少采用一种补救：

1. 同时报告 final checkpoint 与 best checkpoint 的结果；
2. 或统一 best checkpoint 的选择指标，再比较 main/control；
3. 或明确声明 best 选择标准不同，仅将其作为模型内选择，不作强因果对照。

### 后续路线建议

v11c 是必须补齐的稳定基线；v11d 适合作为伪配对负结果或机制审计；v11e 才是
回答真实跨模态实例联想问题的主线。建议后续优先顺序为：

1. 补跑并归档 v11c main/control 与 Cross-Key sweep，确认 AudioRefiner 旁路后的
   coarse/final 口径稳定。
2. 若 v11d 已训练，仅把它作为伪配对 Cross-Detail 的中间证据，重点看
   same-class wrong-pair 是否伤害恢复。
3. 把主要 GPU 预算投入 v11e：先跑 smoke、数据 summary、短训 sanity，再跑
   v11e_control 和 v11e。
4. v11e 正式评估必须同时报告 fixed-mask family breakdown、Cross-Detail sweep、
   pair Recall@1、ACC 与 masked MSE，并在 `docs/dev_log.md` 为每个配置写独立
   结论。

## F 阶段修订：v11e 回归 MNIST/FSDD 类别级绑定

> 本节覆盖上文“v11e 真实 GRID 视听配对主线”的当前实现建议。GRID 方案保留为
> 独立的真实配对后续实验，不再占用 `v11e` 当前版本号。

### 修订原因

项目的原始研究对象是 MNIST 手写数字与 FSDD spoken digit 的跨模态类别联想。
两种数据只共享 digit label，不包含天然的一一实例对应。将任意 MNIST 样本与任意
FSDD 样本固定为 `sample/sample`，会把笔迹、说话人、语速和音色之间不存在的关系
写入监督目标；这种映射可以被训练集记忆，却不能对未见样本泛化。

因此 v11e 改为同类 many-to-many 绑定：MNIST 标签为 `c` 的样本，只与 FSDD
标签为 `c` 的训练录音组合；训练时音频从同类池随机抽取，不建立稳定 `pair_id`。
绑定对象是共享的类别吸引子，而不是人工指定的跨模态实例。

### 恢复目标

| cue 类型 | image target | audio target | 科学含义 |
|---|---|---|---|
| image-only | 当前 clean image (`sample`) | 训练集 audio class medoid (`category`) | 图像实例可恢复，缺失音频只可确定类别 |
| audio-only | 训练集 image class medoid (`category`) | 当前 clean audio (`sample`) | 音频实例可恢复，缺失图像只可确定类别 |
| image+audio | 当前 clean image (`sample`) | 当前 clean audio (`sample`) | 两侧实例均由各自 cue 提供 |

模型保留 `V_from_A + same-modal cue detail` decoder 输入与 Cross-Key 类别条件。
`detach_value_for_recon=true` 继续阻止恢复 loss 经 Value 直接改写 Index。v11e 当前
配置关闭 Cross-Detail、exact-pair alignment 及其 causal objective，因为这些模块
原先依赖真实实例对应；`v11e_control` 改为关闭 Cross-Key 的同预算对照。

### 数据量与解释边界

标准 MNIST 包含 60,000 个训练样本和 10,000 个测试样本。当前完整 FSDD 包含
3,000 条录音；按文件 index 规则划分为 2,700 条训练录音和 300 条测试录音，即
训练每类 270 条、测试每类 30 条。每个 MNIST 训练样本会从对应类别的 270 条音频
中随机抽取，因此每 epoch 仍有约 60,000 次跨模态组合曝光，但独立音频样本数仍是
2,700，不能把重复组合称为 60,000 条音频数据。

该规模足以验证十类关联记忆的可行性，但音频侧仅有 6 位 speaker，且当前
train/test 按 utterance index 而非 speaker 隔离。正式结论应限制为 FSDD 范围内的
类别联想；若要强调说话人泛化，需要另设 speaker-disjoint 实验或引入更大的 spoken
digit 数据集。
