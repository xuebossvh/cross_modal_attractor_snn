# 实现指南：Cross-Modal Attractor SNN
> 生成时间：2026-07-06 18:43 | 生成策略：基于当前代码反向整理 | 状态：ACTIVE_F_STAGE
> 关联版本：历史 v10a … v11e；当前实现分支为 `v11f`
> 关联设计：`docs/idea_report.md`（F 阶段实验设计记录，每次版本迭代须同步更新）
> 扩展说明：本项目已按 ResearchPilot F 阶段管理，后续所有代码修改都走 D-F 迭代。

---

## 0 F 阶段迭代约定

### 当前版本 v11f：冻结基线的 Masked Feature Cross-Key

本节优先于下文历史方案。先设计、后编码；完整协议见
`docs/V11F_MASKED_CROSS_KEY_PROTOCOL.md`。

1. 父模型是已训练的 `cross_modal_snn_v11e_control.pt`。保留 Encoder、Key、
   Index、Value、Classifier、own-detail fusion、原 Decoder/Refiner 的权重和 eval
   状态，仅训练 `img_cross_adapter.*` / `aud_cross_adapter.*`。旧全局 Cross-Key
   参数保留以加载父 checkpoint，但不参与前向调制。
2. `Value + gated own detail -> Decoder features F` 不变；在末层卷积前加入
   `F' = F + M * gate(F, cue, M, source_quality) * delta(F, K_other)`。
   使用目标 mask 的空间/时频结构，Key 只调制同一个 decoder 内的特征，不建立
   独立输出 residual decoder。输出端再次按 mask 选择，保证未损坏位置与冻结基线一致。
3. 新 adapter 末层零初始化；对侧缺失、目标无损或显式 zero intervention 均保持
   父模型输出。父模型在所有训练步骤保持 eval，防止冻结参数但更新 running buffers。
4. 正确 Key 的绝对重建 loss + 相对 zero/wrong 的正向 margin loss；reference
   no_grad，按有效 batch 的基线误差及可配置 floor 归一化，不用单样本近零误差作分母。
   same-class 不作负样本。门控不保证改善；必须验证正确 Key 在部分残缺上的真实收益。
5. 分类保留 Index ACC；额外使用冻结编码器/Index/Classifier 对恢复内容进行单模态
   再分类，标注为内部一致性代理，不冒充独立识别器。按 family 导出 paired CSV，
   包含 normal/zero/wrong/same-class、有效样本数、绝对误差与改善比例。
6. main/no_causal 从同一父模型各训练 30 轮；control 仅评估冻结父模型（0 额外优化），
   不伪称等预算重训。配置仅包含 v11f 族；suite 可选运行 no_causal，并包含 fixed、
   random、family breakdown、Cross-Key sweep 和两种可视化。

父 checkpoint 必须存在且 SHA256 符合配置；只能缺少新 adapter 参数。resume
必须有 v11f checkpoint 并保持父模型摘要一致；不允许静默随机初始化或 CPU 训练回退。
保存基础 state 摘要并在每次 checkpoint 前检查，确认冻结权重及 buffers 未改变。

本仓库不是空白项目，而是已有多轮实验输出的研究原型。后续每次修改代码时，必须遵守下面的顺序：

```text
F-1 诊断问题或确认需求
  -> F-2 判断回溯范围
  -> F-3 先更新设计文档
  -> F-4 再修改代码
  -> F-5 做最小必要验证
  -> 追加 docs/dev_log.md
```

| 改动类型 | 改代码前必须更新 | 改代码后必须更新 |
|----------|------------------|------------------|
| 只改超参或运行命令 | 若命令、输出或语义变化，更新 `docs/implementation.md`；配置文件本身也作为设计记录 | 追加 `docs/dev_log.md`；必要时同步 `运行说明` |
| 改模型结构 | 更新 `docs/implementation.md`；更新 `docs/idea_report.md`（实验目标与消融） | 追加 `docs/dev_log.md`，写清预期效果与验证结果 |
| 改实验设计 | 更新 `docs/implementation.md`；更新 `docs/idea_report.md` | 追加 `docs/dev_log.md`，同步结果格式和输出路径 |
| 改数据管线 | 更新本文档的数据流和对应文件说明 | 追加 `docs/dev_log.md`，同步数据准备命令 |
| 改评估、demo 或可视化 | 更新本文档的脚本说明和结果文件格式 | 追加 `docs/dev_log.md`，同步 `运行说明` |

本次文档创建属于 F 阶段补档：此前仓库没有 `docs/implementation.md` 和 `docs/dev_log.md`，因此先从当前代码反向生成实现指南。

### 0.1 v10b 迭代范围

`v10b` 是在 `v10a` 基础上面向《Deep Long Audio Inpainting》的音频恢复对齐版本，目标是先把音频缺失方式收窄到连续时间片段缺失，并补齐 mask-aware 评估与训练入口。`v10a` 保留为通用鲁棒性基线，不直接覆盖。

本轮只实现 F0-F2：

1. 新增 `configs/v10b.yaml`，从 `v10a.yaml` 派生，训练和 fixed-mask 评估的音频腐蚀均使用 `time_mask`。
2. `data/corruption.py` 支持 `corrupt_audio(..., return_mask=True)`，同次生成音频缺失 mask。
3. `common.build_cue(..., return_masks=True)` 可向 train/evaluate 透传 `aud` mask，默认行为保持向后兼容。
4. `scripts/evaluate.py` 增加 `aud_ssim`、masked/visible audio MSE/L1，并在 family breakdown 中标记论文对齐组、谱图遮挡扩展组和噪声组。
5. `scripts/train.py` 在 `cue_mode` 为 corrupt-audio、`aud_kind=="sample"`、`aud_mode` 属于 `loss.aud_masked_families` 且 mask 存在时追加 masked audio loss。

暂不在 `v10b` 中引入 gated/dilated decoder 或谱图空间 refiner；这些属于后续 `v10c+` 或单独消融，避免同时改变训练目标与模型结构。

### 0.2 v10c 迭代范围

`v10c` 是在 `v10b` 结果诊断后的训练协议修正版本。v10b fixed-mask 评估显示分类和图像恢复仍可工作，但 recovered audio 能量接近零，说明主要问题集中在音频 decoder/reconstruction 训练目标。v10c 不改 CrossModalSNN 主架构，优先对齐训练与评估残缺 family，并加强 decoder pretrain。

本轮实现范围：

1. 新增 `configs/v10c.yaml`，从 `v10b.yaml` 派生，保留 `time_mask` 音频主协议。
2. 图像训练 family 从 `random` 改为 `occlusion`，与 fixed-mask 评估保持一致。
3. 训练后期 `train_severity` 从 `0.5` 降为 `0.4`；staged curriculum 的最高强度同步限制为 `0.4`。
4. decoder pretrain 从 8 轮增加到 25 轮。
5. `scripts/train.py` 的 decoder pretrain 支持 `decoder_pretrain.corrupt_detail=true`：Value state 仍来自 clean target encoder，decoder detail state 可来自 fixed corrupt cue，并可在音频 pretrain 中复用 masked audio loss。
6. 主训练默认轮数设为 70，checkpoint 和输出目录切到 v10c。

v10c 仍属于训练协议修正，不引入新的模型层、数据集或评估指标。

### 0.3 v10d 迭代范围

`v10d` 首次落地 F 阶段计划中尚未实现的**音频恢复模块结构改动**（F3a / F3b）以及**附加质量 loss 代码**（F4）。所有新增能力均为 config 开关、默认关闭，旧 checkpoint 与旧配置行为完全不变；`v10d.yaml` 才是唯一默认启用 refiner 相关结构的配置。

本轮实现范围：

1. **F3a — gated/dilated decoder refinement**：`models/decoders.py` 新增 `GatedConv2d`，`AudioDecoder` 增加 `refine_type` 参数（`plain` | `gated_dilated`）。`plain` 与既有普通 Conv2d refine 块完全一致；`gated_dilated` 用门控卷积 + dilation (1,2,4) 序列扩大缺失区感受野。由 `snn.aud_refine_type` 选择，默认 `plain`。
2. **F3b — 谱图空间 refiner 支路**：`models/decoders.py` 新增 `AudioRefiner`，输入通道 `[coarse_rec, aud_cue, mask]`，输出 delta；`models/network.py` 在 `forward(..., aud_cue_mask=None)` 中仅当 refiner 启用、存在音频 cue、且 mask 非空时执行 refiner 后处理。`visible_paste_back=true` 时使用 `recovered_aud = mask * pred + (1 - mask) * aud_cue`；`visible_paste_back=false` 时使用 `recovered_aud = clamp(coarse + mask * delta)`。image-only 或 mask=None 时完全旁路。由 `audio_refiner.enabled` 开关，默认关闭。
3. **F4 — 附加质量 loss（代码就绪，默认权重 0）**：`scripts/train.py` 的 `_aud_recon_loss` 增加 `lambda_aud_ssim`（`1 - SSIM`）、`lambda_aud_masked_grad`（缺失区时频梯度 L1），`compute_losses` 增加 `lambda_aud_feat`（冻结 `aud_encoder` 特征 L1）。三项独立开关，`v10d.yaml` 中权重均设 0，仅作为后续消融入口，避免与 refiner 提升混淆归因。
4. **cue mode 比例保守调整**：`v10d.yaml` 的 `cue_modes` 相对 v10c 做小幅调整（见 §7.5），保持主实验对 refiner 结构改动的可归因性。

约定与风险控制：

- refiner / gated_dilated 属于 decoder 侧结构，均走 `detach_value_for_recon=true` 路径，不回传 Value/Index，沿用 v9b 已验证的「分类/恢复不互拖」机制。
- `scripts/demo_inference.py`：`fixed_mask` / `legacy_random` 均透传 `aud_mask` 至 `forward(aud_cue_mask=...)`；demo 指标基于最终 `recovered_aud`；图内展示 `audio mask` 与 `coarse audio` 列。
- F3b 属于计划中的高风险项（refiner 直接看到 corrupted cue，可能被质疑绕过 attractor memory）；因此严格限制为「只在 mask 区加 delta、可见区不改写、image-only 完全旁路」，并在 dev_log 记录消融对照口径。

### 0.4 v10e 迭代范围

`v10e` 将 v10d 的音频 inpainting 后处理逻辑扩展到图像分支，使图像和音频都遵循同一条判定规则：

1. **有残缺 cue 的模态**：若该模态 cue 存在且 `mask` 非空，则先由对应 decoder 产生 coarse reconstruction，再由对应 refiner 输入 `[coarse, cue, mask]` 预测 delta，并执行 visible-region paste-back。
2. **完全缺失的模态**：若该模态 cue 为 `None`，则没有上下文和 mask，不启用 refiner；只能由 memory/value/detail 路径生成或联想。
3. **clean cue 的模态**：clean cue 没有缺失 mask，不启用 refiner/paste-back；仍按普通 decoder reconstruction 或跨模态联想处理。

本轮实现范围：

1. `data/corruption.py` 的 `corrupt_image(..., return_mask=True)` 返回图像缺失 mask（`1=missing`），与 `corrupt_audio` 对齐；`common.build_cue(..., return_masks=True)` 同时返回 `masks["img"]` 和 `masks["aud"]`。
2. `models/decoders.py` 新增 `ImageRefiner`，输入 `[coarse_img_prob, img_cue, img_mask]`，输出图像空间 delta；`models/network.py` 增加 `forward(..., img_cue_mask=None, aud_cue_mask=None)`。
3. 图像恢复输出保留 `recovered_img_coarse`；若 `image_refiner.enabled=true` 且存在 `x_img_cue/img_cue_mask`，则 `pred_img = clamp(coarse_prob + delta)`，`final_prob = mask * pred_img + (1-mask) * img_cue`，再转换为 logits 存回 `recovered_img`，兼容现有 BCE-with-logits 训练。
4. `scripts/train.py` / `scripts/evaluate.py` / `scripts/demo_inference.py` 透传 image mask，记录/展示 `img_masked_*` 与 `img_visible_*` 指标；`scripts/plot_eval_summary.py` 支持 v10e 新增的 `imgMaskMSE` / `audMaskMSE` 评估列。
5. 新增 `configs/v10e.yaml`，从 v10d 派生，启用 `image_refiner`，checkpoint/output_version 切换至 v10e。

### 0.5 v10f 迭代范围（归因补齐）

`v10f` 在 v10e 对称 refiner 基础上补齐实验归因能力，不改变主训练目标，只增加可解释性与消融开关：

1. **`models/network.py`**：抽出 `_apply_image_refiner` / `_apply_audio_refiner`；支持 `pasteback_only`（`enabled=false` 时仍可做 `mask*coarse+(1-mask)*cue`）；图像在 prob 空间 paste-back 再转 logits。
2. **`scripts/train.py`**：`decoder_pretrain` 可选训练 refiner（`train_*_refiner`）；仅当对应 `train_*_refiner=true` 时 pretrain 才走 refiner helper 并以 **final** 为主，保留 `lambda_coarse_aux`（默认 0）；`train_*_refiner=false` 的消融必须直接用 coarse decoder 输出，避免冻结随机 refiner 混入 pretrain 目标；`pasteback_only=true` 时强制 `train_*_refiner=false`，但允许单独走 paste-back helper；主训 Adam 按 `audio/image_refiner.lr_mult`（默认 0.5）拆 param group。
3. **`scripts/evaluate.py`**：同 checkpoint 输出 `coarse/final` 的 `*_masked_mse` 与 `*_visible_mse`；追加 `[归因]` 表，显式列出 masked 与 visible 两组 coarse→final 指标。解读口径：**`final_visible_mse≈0` 是 paste-back 机制，不代表可见区学习**；主看 `coarse_mask_mse → final_mask_mse` 是否下降，以及 `pasteback_only` vs full refiner 的 masked 区差距。
4. **配置**：`configs/v10f.yaml` + 5 个消融（audio/image pasteback_off/only、`no_refiner_pretrain`），各自独立 `output_version` 与 ckpt 路径。

### 0.6 v10f 鲁棒性协议补丁（multi-family corruption）

在保留 v10f refiner / paste-back 归因结构的基础上，本补丁把训练、fixed-mask 主评估和 demo 可视化的残缺 family 从单一 `occlusion/time_mask` 扩展为 image/audio 各 5 种**有 mask 的缺失式 corruption**，目标是提升对不同残缺形态的鲁棒性。这里的 corruption 是“带缺失 mask 的输入退化”，不是 MNIST/FSDD 的基础预处理。

**图像 5 family**：

| family | 含义 | mask 语义 |
|------|------|----------|
| `occlusion` | 随机位置方块遮挡；当前实现中方块边长约为 `severity * H/W` | 方块区域为 `1=missing` |
| `pixel_delete` | 随机删除 `severity` 比例像素并置 0 | 被删像素为 `1=missing` |
| `mask_vertical` | 随机选择左侧或右侧连续竖向区域遮挡 | 被遮挡竖条为 `1=missing` |
| `mask_horizontal` | 随机选择上侧或下侧连续横向区域遮挡 | 被遮挡横条为 `1=missing` |
| `salt_mask` | 随机选择 `severity` 比例像素并置为 1，形成白色缺失/盐点遮挡 | 被置白像素为 `1=missing` |

**音频 5 family**：

| family | 含义 | mask 语义 |
|------|------|----------|
| `time_mask` | 连续时间帧缺失 | 被遮挡时间帧为 `1=missing` |
| `freq_mask` | 连续频带缺失 | 被遮挡频带为 `1=missing` |
| `feature_dropout` | 随机谱点缺失 | 被删谱点为 `1=missing` |
| `partial_temporal` | 后段时序缺失 | 被遮挡后段为 `1=missing` |
| `time_freq_block` | 连续时间 × 连续频带二维块缺失 | 被遮挡时频块为 `1=missing` |

`gaussian` 暂不进入主 5 family：它是连续加噪，不是“缺失区补洞”，没有天然 `mask=1` 的可见/缺失边界。`salt_mask` 虽然视觉上像盐噪声，但其语义是“这些像素被白色遮挡并标记为 missing”，不是严格的 Gaussian denoising；若后续要做真正 noisy restoration，应单独设计 denoising cue loss 或噪声估计任务。

训练侧 `common.resolve_train_corrupt_modes` 同时支持 `img_train_modes` 与 `aud_train_modes`，默认 balanced sampling；`data/corruption.py` 中的 `IMG_TRAIN_MODES` 与 `AUD_TRAIN_MODES` 默认常量也固定为上述各 5 个 family，配置文件只是显式覆盖，不再依赖旧默认。decoder pretrain 可通过 `decoder_pretrain.img_modes/aud_modes` 使用同一组 family。评估侧 `evaluate.py --protocol fixed_mask` 读取 `corruption.eval_fixed.img_modes/aud_modes`，按 5 个 family pair 逐组评估。demo fixed-mask 可视化默认 `--num 10`，每个 family 2 个样本，不再允许因筛选失败退回 clean cue。

### 0.7 v11a 迭代范围（非对称双模态对照 + time-mask 专项训练）

v11a 从 v10f 主实验派生，本轮不改 Key/Index/Value、decoder 或 refiner 结构，只修正实验协议与音频缺失训练目标，为后续讨论“图像是否真正帮助音频恢复、音频是否真正帮助图像恢复”建立可解释基线。

1. **8 种 cue mode**：在原 6 种模式之外新增 `clean_img_corrupt_aud` 与 `corrupt_img_clean_aud`。两者均有双模态 cue、目标均为 sample/sample；前者只生成 `aud_mask`，后者只生成 `img_mask`。主训练、smoke test、fixed-mask evaluate 与评估表解析都必须覆盖 8 种模式。
2. **公平的跨模态对照 mask**：fixed-mask 评估中，`clean_img_corrupt_aud` 与 `corrupt_aud_only` 复用同一音频 corruption seed；`corrupt_img_clean_aud` 与 `corrupt_img_only` 复用同一图像 corruption seed。这样 only 与“增加另一条 clean cue”的差异只来自新增模态，而不是随机 mask。
3. **最后 25 轮 time-mask 专项训练**：`configs/v11a.yaml` 主训练设为 120 轮；`corruption.aud_time_mask_finetune.start_epoch: 95` 后，音频 family 改为加权采样，`time_mask` 权重 0.60，其余 family 保留小概率，避免专项微调破坏多 family 鲁棒性。该阶段仍使用 severity 0.4。
4. **active-aware `partial_temporal`**：不再机械遮挡固定 64 帧张量的最后 `round(severity*T)` 帧。对每个样本按 frame energy 找到最后有效语音帧，将等宽 trailing mask 对齐到该位置；全静音样本才回退到固定尾部窗口。这样 maskedMSE 不会因只遮到 padding/silence 而虚假接近 0。
5. **masked energy-weighted MSE**：新增 `loss.lambda_aud_masked_weighted` 与 `loss.aud_masked_weight_gamma`。仅在带有效音频 mask 的 sample-level 恢复模式中，对缺失区计算 `(1 + gamma*target) * error^2`，提高有效语音能量相对静音像素的权重；保留既有 masked L1/MSE，不用该项替代 mask 位置修复。
6. **版本卫生**：v11a 分支的 `configs/` 只保留当前版本配置：主实验 `configs/v11a.yaml` 与同预算 `configs/v11a_control.yaml`；默认入口、checkpoint 与版本化输出均使用 v11a 命名。v10f 主配置与五个消融保留在 main/v10f 分支中。

v11a 基线提交当时未实现跨模态 decoder/refiner 条件注入；随后经用户确认，按下方 0.8 补丁在同一 v11a 分支加入对侧 Key 条件化 Decoder，Refiner 接口仍保持不变。

### 0.8 v11a 迭代补丁（跨模态 Key 条件化 Decoder）

按用户确认，原计划中的下一版本结构直接落在现有 `v11a` 分支。本补丁不改变 `memory.py`、Decoder/Refiner 本体或 8 种 cue mode，而是在 `models/network.py` 的 Decoder 输入融合阶段加入**对侧 Key 的标量门控 Value residual**：

```text
K_img [T,B,128] -> rate/detach [B,128]
  -> img_to_aud projector [B,768]
  -> scalar gate [B,1]
  -> 加到 detached V_aud_from_A [B,768]
  -> concat gated audio detail [B,256]
  -> AudioDecoder [B,1024] -> coarse audio

K_aud [T,B,128] -> rate/detach [B,128]
  -> aud_to_img projector [B,384]
  -> scalar gate [B,1]
  -> 加到 detached V_img_from_A [B,384]
  -> concat gated image detail [B,128]
  -> ImageDecoder [B,512] -> coarse image
```

关键实现约束：

1. `_fuse_decoder_state` 先只 detach 原始 Value，再加入可训练 cross residual，禁止对融合后的 Value 二次 detach。
2. cue 原有三条路径全部保留：`cue -> Key/Index/Value`、`cue -> same-modal detail -> Decoder`、`cue+mask -> Refiner/paste-back`。对侧 Key residual 是新增的第四条显式跨模态条件路径。
3. projector 零初始化、gate bias=0，使初始 residual 严格为 0；missing Key 或 `zero` 干预直接旁路 projector/gate。
4. `decoder_pretrain.train_cross_key_conditioning=false`：第一版 pretrain 不训练 cross 模块，cross-aware pretrain 留作独立消融。
5. `forward` 支持 `[B,128]` Key-rate override 与方向化 disable 开关；override 只影响 Decoder 条件副本，不得改变原始 Key、Index 或 Value。
6. `scripts/evaluate.py --cross_key sweep` 在相同 cue/mask 下执行 normal/zero/wrong 三组确定性前向；wrong permutation 必须保证类别不同，并报告方向化 `correct_gain` / `wrong_damage`。
7. `scripts/plot_eval_summary.py` 解析新增 `[Cross-Key归因]` 表；`scripts/smoke_test.py` 验证初始等价、zero 旁路、两步梯度和 wrong-key 路径隔离。
8. 当前主配置仍为 `configs/v11a.yaml`（cross enabled）；新增 `configs/v11a_control.yaml`（cross path disabled、独立 checkpoint/output）作为同版本公平对照。control 仍通过 `cross_key_conditioning.build_modules=true` 构造同一组 projector/gate，使参数规模与模块构造的 RNG 消耗一致，但前向严格返回零 residual；否则完全不构造 cross 层会改变后续 Decoder/Refiner 的随机初始化，污染归因。

### 0.9 v11d 迭代范围（固定伪配对 + Cross-Detail）

`v11d` 从 `v11c` 的 AudioRefiner-free 基线继续，不改变 Key/Index/Value 主干，
而是在数据定义和 decoder 条件通路上补齐实例级可审计机制。

1. **固定一一伪配对**：`data/dataset.py` 为每个 MNIST item 生成稳定 `pair_id`，
   并把该 item 绑定到一个确定的 FSDD 基础录音和确定性增强种子。
   这使缺失模态的 sample target 不再随 epoch 随机变化。
2. **Cross-Detail**：`models/network.py` 新增对侧 Key 前实例特征路径。
   `rate(spike_*_instance)` 经 projector 与 vector gate 后加到目标模态 detail
   channel，再与 `fused_value` 拼接送入 decoder。
3. **归因控制**：`configs/v11d_control.yaml` 仍启用 Cross-Key，但关闭
   Cross-Detail；因此 v11d main-control 主要隔离 Cross-Detail 的增量贡献。
4. **checkpoint 继承**：v11d 加载 `outputs/checkpoints/cross_modal_snn_v11c.pt`，
   `init_strict=false`，仅允许新增 Cross-Detail 参数缺失；训练前缀只开放
   decoder、ImageRefiner 与 Cross-Detail adapter。

重要边界：v11d 的 MNIST/FSDD 配对仍是同类别伪配对，不是真实同源事件。
结果若不能形成 `correct < zero/same-class wrong`，优先解释为数据不提供
可泛化实例关系，而不是单独否定 Cross-Detail 结构。

### 0.10 v11e 历史方案（GRID 真实视听配对，已废止）

> 本节仅保留设计演进记录，已由 0.11 与 7.10 节的 MNIST/FSDD 类别级绑定方案取代；
> 下列 GRID 文件和配置不属于当前 v11e 分支交付物。

`v11e` 将实验主线从 MNIST/FSDD 伪配对切换到 GRID 真实音视频同源事件。
每一行 manifest 对应一次真实 utterance：图像为同源视频帧的 28x28
rank-pooled mouth-motion dynamic image，音频为同源 WAV 的 64x64 log-mel。

1. **真实配对数据集**：新增 `TruePairedManifestDataset`，强制
   `source_id == image_source_id == audio_source_id`、`pair_id` 唯一、
   speaker-disjoint train/val/test。
2. **GRID 预处理脚本**：新增 `scripts/prepare_grid_v11e.py`，从 GRID audio
   与 frame archive 生成 `_data/grid_v11e/pairs.csv`、预计算 image/audio tensor
   和 training-only audio norm stats。
3. **Pair Alignment**：`models/network.py` 从 image/audio pre-key instance rate
   产生归一化 pair embedding；`scripts/train.py` 使用同类不同 source 作为
   hard negatives 计算对称 InfoNCE。
4. **评估指标**：`scripts/evaluate.py` 在 paired manifest 上报告同类候选内
   exact-pair Recall@1，并把 target 明确标记为 `paired-sample`，不再使用
   category medoid target 解释 v11e。
5. **训练策略**：v11e 从头训练，不继承 v11c/v11d checkpoint；默认用
   `train_samples_per_epoch=60000` 做 optimizer exposure matching，但 unique
   pair 数以 `dataset_summary.json` 为准。

v11e 的 main/control 归因需注意：主模型的 validation score 包含 pair retrieval，
control 的 `lambda_pair=0`。正式报告时应同时比较 best 与 final checkpoint，
或统一 best 选择规则后再做强因果判断。

### 0.11 v11e 当前方案（MNIST/FSDD 类别级绑定）

`v11e` 当前使用 MNIST 图像 `[1,28,28]` 与 FSDD `64x64` log-mel。两种模态
只共享 digit label，因此在同一类别内进行 many-to-many 随机组合，不建立或
监督人工的一一实例配对。

1. **恢复目标遵循可辨识性**：image-only 使用 `sample/category`，即恢复当前
   MNIST 样本和训练集音频 class medoid；audio-only 使用 `category/sample`；
   只有双模态 cue 使用 `sample/sample`。
2. **类别原型无测试泄漏**：缺失模态的 category target 是仅由训练 split 构建的
   class medoid，evaluate/demo 复用训练原型，不从测试集重新估计。
3. **保留当前 decoder 结构**：Decoder 输入仍为
   `V_from_A + same-modal gated cue detail`；`detach_value_for_recon=true`，
   重建 loss 不通过 `V_from_A` 反向改写 Index。
4. **保留 Cross-Key**：主实验启用 Cross-Key，为目标 Decoder 注入对侧类别语义；
   `v11e_control` 在数据、target、seed、batch size 和训练预算相同的前提下只关闭
   Cross-Key，用于测量该路径的增量贡献。
5. **关闭实例级伪监督**：Cross-Detail、pair alignment、pair Recall@1 和
   same-class exact-pair 因果目标均关闭，因为 MNIST/FSDD 不提供真实实例对应。
6. **独立版本交付**：`configs/` 只保留自包含的 `v11e.yaml` 和
   `v11e_control.yaml`；不继承或携带 v11c/v11d YAML。正式评估读取 final
   checkpoint，不用 test split 选择 best checkpoint。

完整配置字段、target 选择和评估协议见 7.10、8.2 节及
`docs/V11E_CATEGORY_BINDING_PROTOCOL.md`。

---

## 1 项目结构

### 1.1 当前真实目录树

```text
cross_modal_attractor_snn/
├── configs/
│   └── v11f.yaml / v11f_control.yaml / v11f_no_causal.yaml
├── data/
│   ├── audio_features.py
│   ├── corruption.py
│   ├── dataset.py
│   └── fsdd.py
├── docs/
│   ├── implementation.md
│   ├── dev_log.md
│   ├── user_requirements.md
│   ├── idea_report.md
│   ├── V11F_MASKED_CROSS_KEY_PROTOCOL.md
│   └── V11E_CATEGORY_BINDING_PROTOCOL.md
├── models/
│   ├── decoders.py
│   ├── encoders.py
│   ├── lif.py
│   ├── memory.py
│   ├── network.py
│   ├── frozen_base.py
│   └── __init__.py
├── scripts/
│   ├── bootstrap.py
│   ├── demo_inference.py
│   ├── evaluate.py
│   ├── mkdir_outputs.py
│   ├── plot_eval_summary.py
│   ├── smoke_test.py
│   ├── smoke_test_v11f.py
│   ├── run_v11f_suite.py
│   └── train.py
├── _data/
│   ├── MNIST/...
│   └── fsdd/...
├── outputs/
│   ├── checkpoints/
│   └── outputs_v*/
├── common.py
├── paths.py
├── README.md
└── requirements.txt
```

> 本项目当前采用研究原型式根目录布局，而不是 ResearchPilot 默认的 `code/src` 布局。除非明确要求重构，否则后续改动应保留当前可运行路径。

### 1.2 文件职责表

| 文件 | 职责 | 主要输入 | 主要输出 | 调用方 |
|------|------|----------|----------|--------|
| `configs/v10a.yaml` | 集中管理模型、数据、损失、训练、输出与消融配置 | 手动配置项 | 运行时超参 | 所有脚本通过 `common.load_config` 读取 |
| `configs/v10b.yaml` | v10b 文献对齐主配置：基于 v10a，仅将音频缺失主协议收窄为 `time_mask` 并启用 masked audio loss | 手动配置项 | v10b 运行时超参 | train/evaluate/demo |
| `configs/v10c.yaml` | v10c fixed-family 训练协议配置：图像 `occlusion`、音频 `time_mask`、后期 severity 0.4、25 轮 corrupt-aware decoder pretrain、70 轮主训练 | 手动配置项 | v10c 运行时超参 | train/evaluate/demo |
| `common.py` | 公共工具：配置加载、cue 构造、target 选择、指标、表格格式化 | `cfg`、clean image/audio、labels | cue、target、指标、格式化表格行 | train/evaluate/demo |
| `paths.py` | 项目根目录与版本化输出路径工具 | 配置中的版本号或路径字符串 | `Path` 对象 | 脚本和数据工具 |
| `data/audio_features.py` | FSDD wav 转 log-mel 特征，并维护全局归一化统计 | wav 路径、audio 配置 | `[n_mels,n_frames]` 特征、norm stats | `data/dataset.py` |
| `data/corruption.py` | 图像和音频 cue 的残缺生成 | clean cue、mode、severity | corrupted cue | `common.build_cue`、demo |
| `data/dataset.py` | MNIST + FSDD 配对数据集与 class medoid 原型 | 配置、本地或下载数据 | train/test loader、原型 | train/evaluate/demo |
| `data/fsdd.py` | FSDD 路径检测与自动下载 | audio 配置 | recordings 路径 | dataset/audio_features |
| `models/lif.py` | surrogate spike、LIF 神经元、LIF 层、rate readout | 时间优先的电流或脉冲张量 | spike 序列和 rate | encoders/memory |
| `models/encoders.py` | 图像与音频 SNN 编码器 | image `[B,1,28,28]`、log-mel `[B,M,F]` | `[T,B,D_img]`、`[T,B,D_aud]` | `models/network.py` |
| `models/memory.py` | Key、循环 Index attractor、Value、binding/readout 记忆模块 | cue/target encoder spikes | Index state、A 驱动 Value、target Value | `models/network.py` |
| `models/decoders.py` | 分类头、图像 decoder、音频 decoder | index state、value/detail state | logits、image logits、log-mel reconstruction | `models/network.py` |
| `models/network.py` | 顶层 `CrossModalSNN` 接口，负责 cue/phase 路由 | 可选图像/音频 cue 和 target | logits、重建、内部状态字典 | train/evaluate/demo/smoke |
| `scripts/train.py` | decoder pretrain 与 binding/readout 训练 | config、train loader | checkpoint、stdout 日志 | CLI、suite |
| `scripts/evaluate.py` | 8 种 cue 模式评估，支持 fixed/random 协议 | checkpoint、test loader | 终端表格、family CSV | CLI |
| `scripts/demo_inference.py` | 可视化 demo 与 demo 指标表 | checkpoint、测试样本 | demo PNG、文本表格 | CLI |
| `scripts/run_v10a_suite.py` | 主训练和消融训练编排 | base config、命令参数 | 每个实验的日志和 checkpoint | CLI |
| `scripts/make_v10a_ablations.py` | 生成 v10a 三个消融配置 | base config | `outputs/ablations_v10a/configs/*.yaml` | CLI、suite |
| `scripts/mkdir_outputs.py` | 创建 checkpoint 和版本化输出目录 | config path | 输出目录 | README 命令、suite |
| `scripts/plot_eval_summary.py` | 将 eval/demo 文本日志渲染为 PNG 和 CSV 表格 | eval/demo log | PNG + CSV | CLI |
| `scripts/smoke_test.py` | 随机张量端到端冒烟测试 | 随机输入，无需数据集 | 终端验证结果 | 手动验证 |
| `scripts/bootstrap.py` | 设置 cwd 和 `sys.path` 到项目根目录 | 脚本执行上下文 | import 路径 | 所有脚本 |

### 1.3 目录级约束

| 路径 | 关键约束 |
|------|----------|
| `models/` | 只放模型结构和神经元模块，不放数据加载或命令行解析 |
| `data/` | 只负责数据、特征、残缺函数、下载工具 |
| `scripts/` | 只负责命令行入口、训练评估流程、实验编排 |
| `configs/` | 新增实验开关和超参优先放入配置文件，不在代码里硬编码 |
| `outputs/` | 运行产物目录，不能作为源代码依赖 |
| `_data/` | 原始或缓存数据目录，包括 FSDD 和 audio norm stats |
| `docs/` | ResearchPilot 文档、图、交接材料 |

---

## 2 数据流与 target 语义

### 2.1 数据集构造

```text
MNIST image [1,28,28]
  + 同数字 label 的 FSDD wav
  -> log-mel [n_mels=64, n_frames=64]
  -> paired sample (image, audio, label)
  -> train/test DataLoader
  -> 从训练集构建 class medoid 原型
```

核心类是 `data.dataset.PairedAudioVisualDataset(cfg, train=True)`。

- 图像来源：优先使用 `torchvision.datasets.MNIST`；不可用时回退到 `_SyntheticImages`。
- 音频来源：FSDD wav，通过 `data/audio_features.py` 转换成 log-mel。
- 配对方式：按 digit label 配对，不要求图像样本和音频样本存在一一对应的样本身份。
- 类别原型：`build_prototypes()` 从训练集构建 class medoid，并复用于 test/demo。

### 2.2 核心 tensor shape

| Tensor | Shape | 含义 |
|--------|-------|------|
| `x_img` | `[B,1,28,28]` | clean MNIST 灰度图 |
| `x_aud` | `[B,64,64]` | 归一化 FSDD log-mel |
| `labels` | `[B]` | digit 类别 |
| image encoder spikes | `[T,B,D_img]`，默认 `[20,B,128]` | 图像 cue 脉冲编码 |
| audio encoder spikes | `[T,B,D_aud]`，默认 `[20,B,128]` | 音频 cue 脉冲编码 |
| key spikes | `[T,B,N_key_*]`，默认 `[20,B,128]` | 模态独立 Key 活动 |
| index spikes | `[T,B,N_index]`，默认 `[20,B,512]` | 循环吸引子 Index 活动 |
| index state | `[B,N_index]`，默认 `[B,512]` | 时间平均发放率 |
| image Value state | `[B,N_value_img]`，默认 `[B,384]` | A 驱动图像 Value |
| audio Value state | `[B,N_value_aud]`，默认 `[B,768]` | A 驱动音频 Value |
| image reconstruction logits | `[B,1,28,28]` | 指标和显示前需要 sigmoid |
| audio reconstruction | `[B,64,64]` | `[0,1]` 范围 log-mel 重建 |
| audio corruption mask | `[B,64,64]` 或 `None` | `v10b` 中 `1=被遮挡/缺失`、`0=可见`；`gaussian` 等无结构缺失时为 `None` |
| class logits | `[B,10]` | digit 分类 logits |

### 2.3 cue 模式

`common.CUE_MODES` 定义 8 种训练和评估模式：

| cue mode | 图像 cue | 音频 cue |
|----------|----------|----------|
| `corrupt_img_only` | 残缺图像 | 缺失 |
| `corrupt_aud_only` | 缺失 | 残缺音频 |
| `corrupt_both` | 残缺图像 | 残缺音频 |
| `clean_img_corrupt_aud` | 干净图像 | 残缺音频 |
| `corrupt_img_clean_aud` | 残缺图像 | 干净音频 |
| `clean_img_only` | 干净图像 | 缺失 |
| `clean_aud_only` | 缺失 | 干净音频 |
| `clean_both` | 干净图像 | 干净音频 |

`common.sample_cue_mode(cfg)` 按当前实验配置（默认
`configs/v11b_recovery.yaml::cue_modes`）中的概率采样。若
`ablation.use_modality_dropout=false`，则只采样四种双模态 cue。

### 2.4 target 选择规则

`common.select_targets(cue_mode, clean_img, clean_aud, proto_img, proto_aud, labels)` 实现恢复粒度策略。

| cue 家族 | 图像 target | 音频 target | 理由 |
|----------|-------------|-------------|------|
| audio-only | 类别图像 medoid | 当前样本 clean audio | 音频只能确定 digit，不能确定具体手写笔迹 |
| image-only | 当前样本 clean image | 类别音频 medoid | 图像只能确定 digit，不能确定具体说话人细节 |
| image+audio | 当前样本 clean image | 当前样本 clean audio | 双模态 cue 同时提供样本级细节 |

这个规则在训练、评估和 demo 中保持一致，是避免跨模态恢复目标不适定的重要约束。

---

## 3 模型架构

### 3.1 端到端数据路径

```text
image cue -> ImageSNNEncoder -> K_img \
                                      -> Recurrent Index A -> V_img_from_A -> ImageDecoder -> recovered image
audio cue -> AudioSNNEncoder -> K_aud /                        \
                                                               -> V_aud_from_A -> AudioDecoder -> recovered audio

Index state -> ClassifierHead -> digit logits

cue detail states -> gated concat -> image/audio decoders
```

核心设计约束：decoder 的 Value 主输入只能来自 `v_img_from_A` 和 `v_aud_from_A`。clean target 只允许在 binding 阶段作为 teacher 构建 target Value，推理和 readout 阶段不得读取 target。

### 3.2 `models/lif.py`

**`SurrogateSpike(torch.autograd.Function)`**

- `forward(ctx, v_minus_thresh, alpha) -> Tensor`
- 输入：
  - `v_minus_thresh`：膜电位减阈值。
  - `alpha`：surrogate gradient 斜率。
- 输出：二值 spike tensor。
- 实现逻辑：前向使用 Heaviside 阶跃函数，反向使用 atan 风格 surrogate derivative。

**`spike_fn(v_minus_thresh, alpha=2.0) -> Tensor`**

- `SurrogateSpike.apply` 的封装。

**`LIFNeuron(nn.Module)`**

- 构造函数：`LIFNeuron(beta=0.9, v_threshold=1.0, surrogate_alpha=2.0)`。
- `init_state(shape, device, dtype=torch.float32) -> Tensor`：返回零膜电位。
- `step(v, input_current) -> tuple[Tensor, Tensor]`：执行 `v = beta*v + I`，产生 spike，再用 detach 后的 spike mask 做 reset-by-zero。

**`LIFLayer(nn.Module)`**

- 构造函数：`LIFLayer(in_features, out_features, beta=0.9, v_threshold=1.0, surrogate_alpha=2.0, bias=True)`。
- `forward(x) -> tuple[Tensor, Tensor]`
- 输入：`[T,B,in_features]`。
- 输出：spikes `[T,B,out_features]`，membrane trace `[T,B,out_features]`。
- 实现逻辑：每个时间步先通过线性层生成电流，再调用 `LIFNeuron.step`。

**`rate(spikes) -> Tensor`**

- 输入：`[T,B,D]`。
- 输出：`[B,D]`。
- 实现逻辑：沿时间维求平均发放率。

### 3.3 `models/encoders.py`

**私有辅助函数**

| 函数 | 职责 |
|------|------|
| `_to_time(x, T)` | 将 flat input 复制到时间维 |
| `_to_time_4d(x, T)` | 将 `[B,C,H,W]` 复制到时间维 |
| `_poisson(x, T)` | 按输入强度采样 Poisson/Bernoulli spike |
| `_first_spike_encode(x, T)` | 首脉冲时间编码，像素越亮越早发放 |
| `_exponential_trace(spikes, trace_decay)` | 生成指数衰减 trace |
| `_conv_out_hw(h, w, kernel, stride, padding)` | 计算 Conv2d 输出尺寸 |

**`LIFConv2dStage(nn.Module)`**

- 构造函数：`LIFConv2dStage(in_ch, out_ch, in_h, in_w, kernel_size=3, stride=1, padding=1, beta=0.9, v_threshold=1.0, surrogate_alpha=2.0)`。
- `forward(x) -> Tensor`
- 输入：`[T,B,Cin,H,W]`。
- 输出：`[T,B,Cout,H',W']`。
- 实现逻辑：每个时间步执行 Conv2d 电流投射，展平后做 LIF 更新，再恢复为 feature map。

**`ImageSNNEncoder(nn.Module)`**

- 构造函数：`ImageSNNEncoder(img_in=784, hidden=256, D_img=128, T=20, beta=0.9, v_threshold=1.0, surrogate_alpha=2.0, encoding="first_spike_trace", trace_decay=0.9)`。
- `_encode_input(x) -> Tensor`：支持 first-spike+trace、poisson 和 repeated current。
- `forward(x_img) -> Tensor`
- 输入：`[B,1,28,28]`。
- 输出：`[T,B,D_img]`。
- 实现逻辑：图像 flatten 后进行时间编码，依次通过两层 `LIFLayer`。

**`AudioSNNEncoder(nn.Module)`**

- 构造函数：`AudioSNNEncoder(aud_in=1024, hidden=128, D_aud=128, T=20, beta=0.9, v_threshold=1.0, surrogate_alpha=2.0, encoding="current", encoder_type="conv", n_mels=32, n_frames=32, conv_ch1=16, conv_ch2=32)`。
- `_encode_input_4d(x) -> Tensor`：将 `[B,1,H,W]` 编码到时间维。
- `forward(x_aud) -> Tensor`
- 输入：`[B,n_mels,n_frames]`，v10a 默认 `[B,64,64]`。
- 输出：`[T,B,D_aud]`。
- 实现逻辑：
  1. `encoder_type="conv"` 时：增加 channel 维，两层 `LIFConv2dStage`，flatten，再接两层 `LIFLayer`。
  2. legacy linear 模式：flatten log-mel，再通过两层 `LIFLayer`。

### 3.4 `models/memory.py`

**`KeyLayer(nn.Module)`**

- 构造函数：`KeyLayer(d_enc, n_key, beta=0.9, v_threshold=1.0, surrogate_alpha=2.0)`。
- `forward(enc_spikes) -> Tensor`
- 输入：encoder spikes `[T,B,D_enc]`。
- 输出：key spikes `[T,B,n_key]`。
- 实现逻辑：用一个 `LIFLayer` 将模态编码投射到 Key 空间。

**`RecurrentIndexLayer(nn.Module)`**

- 构造函数包含 key 维度、`n_index`、LIF 参数、模态输入权重、循环强度、WTA 参数和输入调度策略。
- `_competition(v, raw_spikes) -> Tensor`
  - `kwta`：按膜电位保留 top-k spike。
  - `inhibition_pool`：使用全局抑制池反馈。
- `_input_gates(t, T, has_img, has_aud) -> tuple[float,float]`
  - 支持 simultaneous、image/audio first phased schedule、interleave schedule。
  - 可通过 `phase_on_bimodal_only` 限制只在双模态输入时启用分阶段输入。
- `forward(key_img_spikes=None, key_aud_spikes=None) -> tuple[Tensor, Tensor]`
- 输入：一种或两种模态的 key spike。
- 输出：`index_spikes [T,B,N_index]`，`index_state [B,N_index]`。
- 实现逻辑：
  1. 初始化膜电位和上一时间步 spike。
  2. 每个时间步累加 image current、audio current 和可选 recurrent current。
  3. 用 surrogate threshold 产生 raw spikes。
  4. 按配置执行 k-WTA 或 inhibition competition。
  5. reset 已发放神经元。
  6. 返回完整时间序列和 rate state。

**`ValueLayer(nn.Module)`**

- 构造函数：`ValueLayer(n_index, d_enc, n_value, beta=0.9, v_threshold=1.0, surrogate_alpha=2.0)`。
- `_run_from_A(index_spikes) -> tuple[Tensor, Tensor]`
  - A 驱动路径，始终启用。
  - 其 state 是 decoder 的 Value 主输入。
- `_run_target(enc_spikes, delay) -> tuple[Tensor, Tensor]`
  - target 路径，只在 binding 阶段启用。
  - 可按 `value.binding_delay` 延迟 target spikes。
- `forward(index_spikes, target_enc_spikes=None, phase="readout", use_delayed_target=True, delay=2) -> tuple[Tensor, Tensor, Tensor|None]`
- 输出：
  - A 驱动 value spikes。
  - A 驱动 value state。
  - target value state 或 `None`。

**`CrossModalAttractorMemory(nn.Module)`**

- 构造函数：`CrossModalAttractorMemory(cfg)`。
- `forward(spike_img_cue=None, spike_aud_cue=None, spike_img_target=None, spike_aud_target=None, phase="readout") -> dict`
- 实现逻辑：
  1. cue spikes 分别进入 `K_img` 和 `K_aud`。
  2. Key 电流驱动循环 Index attractor。
  3. Index spikes 分别驱动 image/audio Value。
  4. binding 阶段额外计算 clean target Value。
  5. 返回 key spikes、index spikes/state、A-driven Value 和 target Value。

### 3.5 `models/decoders.py`

**`ClassifierHead(nn.Module)`**

- 构造函数：`ClassifierHead(n_in, num_classes, hidden=128)`。
- `forward(state) -> Tensor`
- 输入：`[B,n_in]`。
- 输出：`[B,num_classes]` logits。

**`ImageDecoder(nn.Module)`**

- 构造函数：`ImageDecoder(n_value_img, out_hw=28, base_ch=128)`。
- `forward(value_state) -> Tensor`
- 输入：`[B,n_value_img + optional_detail_dim]`。
- 输出：image logits `[B,1,28,28]`。
- 实现逻辑：FC 到 `[B,base_ch,7,7]`，两层 stride-2 ConvTranspose2d，上采样到 28x28，再用 Conv2d 输出单通道 logits。

**`_audio_decoder_stages(out_hw, start_hw=4) -> int`**

- 检查 `out_hw/start_hw` 是否为 2 的幂，并返回上采样层数。

**`GatedConv2d(nn.Module)`（v10d 新增，F3a）**

- 构造函数：`GatedConv2d(ch, kernel_size=3, dilation=1)`。
- `forward(x) -> Tensor`：`feat(x) * sigmoid(gate(x))`，两个同形状 Conv2d 分别产生特征与门控，`padding=dilation*(kernel_size-1)//2` 保持分辨率。
- 用途：`gated_dilated` refine 块的基本单元，配合 dilation (1,2,4) 扩大缺失区感受野。

**`AudioDecoder(nn.Module)`**

- 构造函数：`AudioDecoder(n_value_aud, n_mels, n_frames, base_ch=128, start_hw=4, refine_blocks=0, refine_type="plain")`。
- `forward(value_state) -> Tensor`
- 输入：`[B,n_value_aud + optional_detail_dim]`。
- 输出：`[B,n_mels,n_frames]`，范围 `[0,1]`。
- 实现逻辑：FC 到方形 feature map，多层 ConvTranspose2d 上采样，可选 refinement，最后 softplus 并 clamp。
- `refine_type="plain"`（默认）：refine 块为普通 `Conv2d(3x3)`，与 v10c 及更早版本逐层一致，旧 checkpoint 可加载。
- `refine_type="gated_dilated"`：refine 块换成 `GatedConv2d`，dilation 依次取 `[1,2,4]` 循环；参数量略增，感受野更大。切换会改变 `cnn` 子模块结构，需重训。

**`AudioRefiner(nn.Module)`（v10d 新增，F3b，默认关闭）**

- 构造函数：`AudioRefiner(n_mels, n_frames, hidden_ch=32, blocks=2, delta_scale=1.0, max_dilation=16)`。
- `forward(coarse, cue, mask) -> Tensor`：输入三通道 `[coarse_rec, aud_cue, mask]`（各 `[B,n_mels,n_frames]`），经小型 gated conv 栈输出 `delta = delta_scale * tanh(...)`，形状 `[B,n_mels,n_frames]`。
- **感受野**：第 `bi` 个 gated conv 块 dilation = `min(2**bi, max_dilation)`（指数增长）。`blocks=4` 时 dilation 依次 `1,2,4,8`，时间感受野约 31 帧，足以覆盖 `severity=0.4`（约 26 帧）的连续缺失；增大 `blocks` 可进一步扩大缺失区上下文覆盖。
- refiner 只产生 delta，最终如何与 coarse/cue 合成由 `network.forward` 决定（见 §3.6），两种合成方式都保证可见区（mask=0）不被 refiner 改写。
- v11c 增加 `audio_refiner.bypass`。`enabled=true,bypass=true` 时仍构造并保留
  `AudioRefiner` 参数，使 v11b checkpoint 可以 strict 加载，但前向不调用 refiner。
  若同时 `pasteback_only=true`，有 mask 时输出
  `mask*coarse + (1-mask)*cue`；无 mask/cue 时直接输出 coarse。bypass 状态下
  AudioRefiner 参数强制冻结，也不得加入 decoder pretrain。

### 3.6 `models/network.py`

**`CrossModalSNN(nn.Module)`**

- 构造函数：`CrossModalSNN(cfg)`。
- 组件：
  - `ImageSNNEncoder`
  - `AudioSNNEncoder`
  - `CrossModalAttractorMemory`
  - `ClassifierHead`
  - 可选 `aux_aud_classifier`
  - 可选 image/audio detail projector 和 gate
  - `ImageDecoder`
  - `AudioDecoder`

**`_normalize_audio_for_encoder(x_aud) -> Tensor|None`**

- 支持 `global`、`dataset`、`none`、`per_sample`、`hybrid`。
- `hybrid` 模式将全局归一化输入与逐样本 min-max 输入混合。

**`_cue_detail_state(spikes, dim, batch, device, dtype) -> Tensor|None`**

- 将 cue spikes 转换为 rate detail state。
- 若某模态 cue 缺失且 `detail_conditioning.zero_missing=true`，返回零向量。
- 若 `detail_conditioning.detach=true`，切断 detail 梯度。

**`_fuse_decoder_state(value_state, raw_detail, modality) -> Tensor`**

- 若 `detach_value_for_recon=true`，先 detach Value state。
- 将 raw detail 投射到配置维度。
- `fusion="gated_concat"` 时，用 `[value_state, detail]` 预测 gate，再对 detail 加权。
- 返回 `[value_state, gated_detail]` 拼接结果。

**v11d+ Cross-Key / Cross-Detail / Pair-Embedding 补充**

- `_cross_key_residual(base_value, cross_key_rate, modality, disabled=False)`：
  将对侧 Key rate `[B,128]` 投影为目标 Value residual：
  `K_img -> V_aud[768]`，`K_aud -> V_img[384]`。gate 为标量 `[B,1]`，
  residual 与 detached `V_from_A` 相加，形成 `fused_value`。
- `_cross_detail_residual(base_value, cross_detail_rate, modality,
  target_missing_ratio=None, disabled=False)`：将对侧 encoder pre-key
  instance rate 投影到目标 detail channel。音频到图像：
  `[B,aud_hidden=256] -> [B,img_detail_dim=128]`；图像到音频：
  `[B,img_hidden=256] -> [B,aud_detail_dim=256]`。gate 为逐维 vector gate，
  输入包含 `base_value`、projected detail 和目标模态缺失比例 `[B,1]`。
- `_pair_embedding(detail, modality)`：v11e 中把 image/audio instance detail
  投影到共享 pair embedding 空间，默认 `[B,128]`，并执行 `F.normalize`。
  该 embedding 用于 pair-alignment loss 与 exact-pair Recall@1，不直接参与
  Index 动力学。
- `_missing_ratio(cue, mask, batch, device, dtype)`：为 Cross-Detail gate 提供
  目标模态缺失比例；模态缺席记为 1，clean cue 且无 mask 记为 0。

**`_fuse_decoder_state(value_state, raw_detail, modality, cross_key_rate=None, disable_cross_key=False, raw_cross_detail=None, target_missing_ratio=None, disable_cross_detail=False, return_cross_stats=False) -> Tensor|tuple`（v11d+）**

融合顺序为：

```text
base_value = detach(V_from_A) if detach_value_for_recon else V_from_A
fused_value = base_value + CrossKeyResidual(opposite key rate)
own_detail = DetailProjector(rate(own cue spikes))
own_detail = own_detail * DetailGate(fused_value, own_detail)
detail = own_detail + CrossDetailResidual(opposite pre-key instance rate)
decoder_state = concat(fused_value, detail)
```

该设计保证恢复 loss 默认不通过 `V_from_A` 回写 Index/Value；Cross-Key 和
Cross-Detail 的 override/disable 只影响 decoder 条件副本，不改变原始
Key、Index 或 Value。

**`forward(x_img_cue=None, x_aud_cue=None, x_img_target=None, x_aud_target=None, training_mode=False, phase="readout", img_cue_mask=None, aud_cue_mask=None, cross_key_img_rate_override=None, cross_key_aud_rate_override=None, cross_detail_img_rate_override=None, cross_detail_aud_rate_override=None, disable_img_to_aud_cross=False, disable_aud_to_img_cross=False, disable_img_to_aud_detail=False, disable_aud_to_img_detail=False) -> dict`**

- 至少需要一种 cue 模态。
- 推理时强制 `phase="readout"`，并清空 target，防止答案泄漏。
- `img_cue_mask` / `aud_cue_mask`：图像和音频缺失 mask，`1=missing`，
  用于 refiner/paste-back、masked metric 和 missing-ratio gate。
- `aud_cue_mask`（v10d，F3b）：音频缺失 mask `[B,n_mels,n_frames]`（1=缺失，0=可见）。仅当 `audio_refiner.enabled=true`、存在音频 cue 且 mask 非空时启用 refiner；其余情况一律忽略、行为与旧版一致。
- 流程：
  1. 对存在的 image/audio cue 编码。
  2. binding 阶段对 clean targets 编码。
  3. 调用 memory。
  4. 从 `index_state` 分类。
  5. 若启用 audio aux classifier，从 `key_aud` rate 辅助分类。
  6. 将 A-driven Value 和 cue detail 融合。
  7. 解码 recovered image 和 coarse recovered audio。
  8. 若 refiner 启用且满足条件（音频 cue 存在 + mask 非空），按 `audio_refiner.visible_paste_back` 二选一合成，否则 `recovered_aud = coarse`。coarse 结果始终保留在 `recovered_aud_coarse`：
     - `visible_paste_back=false`（默认，兼容早期 v10d）：`recovered_aud = clamp(coarse + mask * delta, 0, 1)`；可见区沿用 coarse（decoder 仍负责重建可见区）。
     - `visible_paste_back=true`（文献式 inpainting，v10d.yaml 采用）：`pred = clamp(coarse + delta, 0, 1)`，`recovered_aud = mask * pred + (1 - mask) * aud_cue`；缺失区用网络预测，**可见区原样保留输入 cue**（time_mask 下可见区即 clean 帧）。两种方式可见区都不被 refiner 改写。
- v11d/v11e 额外返回 Cross-Key/Cross-Detail 的 gate、residual norm、residual/value
  ratio，以及 v11e pair embedding。
- 返回字典包含：`index_spikes`、`index_state`、cue spikes、key spikes、
  `v_*_from_A`、`v_*_target`、detail states、`logits`、`aux_aud_logits`、
  `recovered_img`、`recovered_img_coarse`、`recovered_aud`、Cross-Key /
  Cross-Detail 诊断量和 pair embedding。

**`infer(x_img_cue=None, x_aud_cue=None, aud_cue_mask=None) -> dict`**

- eval 模式下执行 readout 推理，不允许 target 输入；可选透传 `aud_cue_mask` 供 refiner 使用。

---

## 4 数据与公共工具实现

### 4.1 `data/audio_features.py`

| 函数 | 输出 | 实现逻辑 |
|------|------|----------|
| `normalize_feature_per_sample(feat)` | 归一化 tensor | 单样本 min-max |
| `normalize_feature_global(feat, lo, hi)` | 归一化 tensor | 使用全局分位数范围缩放并 clamp |
| `_load_wav_mono(path)` | `(wav, sr)` | 优先 soundfile，回退 torchaudio |
| `log_mel_raw(path, sample_rate, n_mels, n_frames, duration_sec, n_fft=512)` | raw log-mel | 重采样、pad/crop、MelSpectrogram、log1p |
| `log_mel_from_wav(...)` | 归一化 log-mel | raw log-mel 加指定 norm |
| `audio_feature_shape(cfg)` | `(n_mels,n_frames)` | 从配置读取音频尺寸 |
| `_fsdd_train_wav_paths(cfg)` | wav 路径列表 | 只返回 FSDD train split，index >= 5 |
| `compute_audio_norm_stats(cfg)` | stats dict | 在训练 wav 上统计 p1-p99 或配置分位数 |
| `save_audio_norm_stats(stats,path)` | 文件 | `torch.save` |
| `load_audio_norm_stats(path)` | stats dict | `torch.load` |
| `ensure_audio_norm_stats(cfg)` | stats dict 或 `None` | 加载缓存，shape 不匹配时重算 |

### 4.2 `data/fsdd.py`

| 函数 | 输出 | 实现逻辑 |
|------|------|----------|
| `fsdd_recordings_dir(cfg)` | recordings 路径 | 规范化 `audio.fsdd_root` |
| `count_wav_files(recordings_dir)` | int | 统计 `*.wav` |
| `_clone_fsdd(parent, verbose=True)` | bool | git shallow clone |
| `_download_zip_fsdd(parent, verbose=True)` | bool | GitHub zip 下载回退 |
| `ensure_fsdd(cfg, verbose=True)` | recordings path 或 `None` | 检查已有 wav，必要时自动下载 |

### 4.3 `data/corruption.py`

图像残缺模式：

```text
occlusion, pixel_delete, gaussian, mask_left, mask_right, mask_top, mask_bottom
```

音频残缺模式：

```text
gaussian, time_mask, freq_mask, feature_dropout, partial_temporal, time_freq_block
```

`v10b` 将音频残缺分成三组：

| 分组 | family | 用途 |
|------|--------|------|
| `paper_aligned_time_gap` | `time_mask`, `partial_temporal` | 与 long audio gap 最接近；`v10b` 主配置只使用 `time_mask` |
| `spectrogram_occlusion` | `time_freq_block`, `freq_mask`, `feature_dropout` | 谱图遮挡鲁棒性扩展，不作为论文主对齐设定 |
| `noise` | `gaussian` | 噪声鲁棒性；无结构缺失，mask 为 `None` |

`time_mask` 的缺失宽度为 `round(severity * n_frames)`。在当前 `audio.duration_sec=1.0`、`n_frames=64` 时，1 帧约为 `15.6ms`，`severity=0.5` 约对应 `500ms` 的 log-mel 时间缺失。该换算是谱图帧比例近似，不等同于论文中直接按 waveform 毫秒/秒设置 gap。

| 函数 | 输出 | 实现逻辑 |
|------|------|----------|
| `_resolve(mode, pool)` | 具体 mode | `mode="random"` 时从 pool 随机采样 |
| `corrupt_image(x_img, mode="random", severity=0.5)` | corrupted image | 方块遮挡、像素删除、噪声或方向 mask |
| `corrupt_audio(x_aud, mode="random", severity=0.5, return_mask=False)` | corrupted log-mel，或 `(corrupted, mask)` | 噪声、时间/频率 mask、dropout、时间截断、二维块遮挡；`return_mask=True` 时必须在同一次随机采样中返回真实缺失 mask |
| `_corrupt_audio_flat(x, mode, severity, return_mask=False)` | corrupted flat audio，或 `(corrupted, None)` | 2D 输入时只支持 noise/dropout；`v10b` 暂不为 flat 输入提供结构化 mask |

mask 约定：

- shape 与输入 log-mel 相同：`[B,n_mels,n_frames]`。
- `1` 表示被遮挡/缺失区域，`0` 表示可见区域。
- `time_mask`、`freq_mask`、`time_freq_block`、`partial_temporal`、`feature_dropout` 返回 tensor mask。
- `gaussian` 返回 `None`，masked loss/metric 自动跳过。

### 4.4 `data/dataset.py`

**`_SyntheticImages`**

- 构造函数：`_SyntheticImages(num_samples, num_classes, seed=0)`。
- MNIST 不可用时的图像数据回退。
- `__getitem__(i)` 返回 synthetic blob image `[1,28,28]` 和 label。

**数据集辅助函数**

| 函数 | 职责 |
|------|------|
| `_make_audio_prototypes(num_classes, n_mels, n_frames, seed=0)` | 生成 toy audio prototypes |
| `_parse_fsdd_name(path)` | 解析 `{digit}_{speaker}_{index}.wav` |
| `_load_fsdd_by_digit(cfg, train)` | 按 digit 和 train/test split 加载 FSDD log-mel |
| `_deterministic_audio_augment(feature, seed, augment_cfg)` | v11d 固定伪配对的音频实例增强：time/freq shift、gain、noise，输出仍 clamp 到 `[0,1]` |
| `_shift_feature(x, amount, dim)` | v11d 音频特征平移辅助函数，移出区域补 0 |
| `_load_saved_tensor(path, field)` | v11e 读取 `.pt/.pth/.npy` 预处理图像或音频 tensor |

**`PairedAudioVisualDataset(Dataset)`**

- 构造函数：`PairedAudioVisualDataset(cfg, train=True)`。
- 关键字段：
  - `self._base`：MNIST 或 synthetic 图像数据集。
  - `self._fsdd`：按 digit 分组的真实音频。
  - `self.prototype_img`：`[C,1,28,28]` 图像 medoid。
  - `self.prototype_aud`：`[C,n_mels,n_frames]` 音频 medoid。
- `__len__() -> int`：返回有效索引数。
- `build_prototypes() -> tuple[Tensor,Tensor]`：构建 image/audio medoids。
- `_medoid(stacked) -> Tensor`：返回距离类均值最近的真实样本。
- `_build_image_prototypes() -> Tensor`：从图像训练样本构建 medoid。
- `_build_audio_prototypes() -> Tensor`：从 FSDD/toy 音频构建 medoid。
- `_make_audio(label) -> Tensor`：为某个 digit 随机采样一个音频特征。
- `__getitem__(idx) -> tuple[Tensor,Tensor,int]`：返回 `(img, aud, label)`。
- v11d 固定配对模式：
  - 当 `data.pairing.enabled=true` 且
    `data.pairing.mode="fixed_augmented_one_to_one"` 时，`_make_audio`
    不再按 epoch 随机采样 FSDD，而是由 `(item_index,label,pair_seed)`
    确定一个基础录音和增强种子。
  - `_pair_id(item_index)` 为 train/test 使用不同命名空间，避免 split 间
    pair id 冲突。
  - `export_pair_manifest(path)` 可输出审计表：
    `pair_id, split, image_index, label, base_audio_index, augmentation_seed`。
  - 若 `return_pair_id=true`，`__getitem__` 返回
    `(img, aud, label, pair_id)`。

**`TruePairedManifestDataset(Dataset)`（v11e）**

- 构造函数：`TruePairedManifestDataset(cfg, split="train")`。
- 输入：`data.manifest_path` 指向 `_data/grid_v11e/pairs.csv`。
- 必需列：
  `pair_id, source_id, image_source_id, audio_source_id, speaker_id, split,
  label, image_path, audio_path`。
- `_validate_manifest(rows)` 严格检查：
  - `pair_id` 全局唯一；
  - `source_id == image_source_id == audio_source_id`；
  - `source_id` 不跨 split 重复；
  - `require_speaker_disjoint=true` 时 speaker 不跨 train/val/test；
  - `require_unique_modalities=true` 时同 split 内 image/audio path 不复用；
  - `validate_paths=true` 时文件必须存在。
- `_load_image(path)`：读取 `.pt/.pth/.npy` 或图片文件，统一为
  `[1,28,28]`、范围 `[0,1]`。
- `_load_audio(path)`：读取预计算 tensor 或 WAV 并转 log-mel，统一为
  `[n_mels,n_frames]`、范围 `[0,1]`。
- `build_prototypes()`：仅为兼容旧接口构建 class medoid；v11e 的 target
  语义为 paired-sample，正式评估不应把 category medoid 当作主目标。
- `__getitem__(idx)` 返回 `(image, audio, label, pair_id)`，其中 image/audio
  来自同一真实 source row。

**`build_loaders(cfg, eval_split=None, train_required=True) -> tuple[DataLoader|None,DataLoader]`**

- `data.dataset="mnist_fsdd"`：沿用 `PairedAudioVisualDataset`，real audio
  启用时先保证 FSDD audio norm stats。
- `data.dataset="paired_manifest"`：使用 `TruePairedManifestDataset`，
  默认评估 split 来自 `data.eval_split`；若 `train_required=false`，只构造
  eval loader，并用零 tensor 占位旧 prototype 接口。
- 若 `data.train_samples_per_epoch>0`，train loader 使用 replacement
  `RandomSampler` 做 optimizer exposure matching；这不会增加 unique pair 数。
- train loader 默认 drop_last；test/val loader 不 shuffle。

### 4.5 `common.py`

| 函数 | 职责 |
|------|------|
| `fix_console_encoding()` | Windows 终端 UTF-8 输出 |
| `log(msg)` | flush print |
| `load_config(path="configs/v11f.yaml")` | 读取 YAML；支持相对路径 `extends` 并做递归 deep merge |
| `set_seed(seed)` | 设置 random、numpy、torch seed |
| `unpack_paired_batch(batch)` | 兼容三元组 `(img,aud,label)` 与四元组 `(img,aud,label,pair_id)` |
| `sample_cue_mode(cfg)` | 按概率采样 cue mode |
| `sample_train_severity(cfg, epoch)` | fixed/random/staged severity |
| `resolve_train_corrupt_modes(cfg, epoch, step=None)` | 训练时采样 image/audio corruption family |
| `build_cue(clean_img, clean_aud, mode, cfg, severity=None, img_mode=None, aud_mode=None, return_masks=False)` | 构造 clean/corrupt/single-modal cue；`return_masks=True` 时返回 `(img_cue,aud_cue,{"img":img_mask,"aud":aud_mask})` |
| `is_aud_only_mode(mode)` | 判断 audio-only cue |
| `cue_modalities(mode)` | 返回 cue 是否含 image/audio |
| `select_targets(cue_mode, clean_img, clean_aud, proto_img, proto_aud, labels, paired_missing_targets=False, paired_target_kind="sample")` | 按恢复粒度选择 target；v11d/v11e 可让缺失模态也使用当前 paired sample，而不是 class medoid |
| `aud_collapse_stats(rec, target, top_fraction=0.15)` | 音频输出能量塌缩诊断 |
| `spike_reg(out)` | spike 活动正则 |
| `batch_psnr(pred, target, eps=1e-8)` | 图像 PSNR |
| `batch_ssim(pred, target, C1=..., C2=...)` | 简化全局 SSIM |
| `batch_reconstruction_variance(reconstructions, max_pairs=2000)` | 图像重建多样性诊断 |
| `per_class_reconstruction_variance(reconstructions, labels, num_classes=10)` | 类内重建方差 |

---

## 5 训练实现

### 5.1 `scripts/train.py`

常用命令：

```bash
python -u scripts/train.py --config configs/v10a.yaml
python -u scripts/train.py --config configs/v10a.yaml --epochs 30
python -u scripts/train.py --config configs/v10a.yaml --resume
```

命令参数：

| 参数 | 含义 |
|------|------|
| `--config` | YAML 配置路径 |
| `--epochs` | 覆盖 `train.epochs` |
| `--resume` | 从 `train.ckpt_path` 恢复 |
| `--start_epoch` | checkpoint 缺少 epoch 字段时指定恢复起点 |
| `--skip_decoder_pretrain` | 跳过 decoder pretraining |

主要损失和辅助函数：

| 函数 | 职责 |
|------|------|
| `_img_edge_loss(prob, target)` | 图像水平/垂直一阶差分 L1 |
| `_img_recon_loss(rec, x_img, lc)` | BCE 或 MSE，加可选 L1 和 edge loss |
| `_aud_tf_grad_loss(rec, target)` | 音频时频方向一阶差分 L1 |
| `_aud_active_loss(rec, target)` | 防止音频输出低方差塌缩 |
| `_aud_foreground_loss(rec, target, top_fraction=0.15)` | 前景能量区域加权误差 |
| `_aud_marginal_loss(rec, target)` | 匹配时间和频率边缘均值 |
| `_masked_audio_error(rec, target, mask, power=2)` | 对每个样本在 mask 区域归一化后再 batch 平均 |
| `_aud_recon_loss(rec, target, lc, mask=None)` | L1 + MSE + weighted MSE + 可选音频项；`v10b` 可追加 masked L1/MSE |
| `_drop_detail_state(detail, drop_prob)` | decoder pretrain 的 detail dropout |
| `_target_value_state(value_layer, enc_spikes, delay)` | 构建 target Value teacher state |
| `_pretrain_decoder_states(model, x_img, x_aud, detail_dropout, x_img_detail=None, x_aud_detail=None)` | 从 clean target Value 构建 decoder pretrain 输入；v10c 可用 corrupt cue 生成 detail state |
| `_set_decoder_pretrain_requires_grad(model, freeze_non_decoders=True)` | 冻结非 decoder/detail fusion 参数 |
| `_restore_requires_grad(previous)` | 恢复参数 requires_grad |
| `_save_decoder_pretrain_ckpt(model, cfg, pre_ckpt, epoch, epochs)` | 保存 decoder pretrain checkpoint |
| `pretrain_decoders(model, train_loader, cfg, device)` | 训练 image/audio decoders；v10c 可启用 fixed corrupt detail 和 masked audio pretrain loss |
| `_soft_cls_loss(student_logits, teacher_logits, temperature=2.0)` | soft classification consistency |
| `_teacher_cues_for_mode(cue_mode, clean_img, clean_aud, match_modality)` | 构建 clean teacher cue |
| `_class_key_alignment_loss(key_img, key_aud, labels, temperature=0.1)` | 监督式 image/audio key 对齐 |
| `_audio_detail_consistency_loss(model, out_r, clean_aud, cue_mode, cfg)` | noisy audio detail 与 clean detail 一致性 |
| `_alignment_losses(model, out_r, clean_img, clean_aud, labels, cue_mode, cfg)` | index/soft-class/key alignment |
| `_apply_audio_target_curriculum(...)` | sample audio target 与 class medoid curriculum 混合 |
| `compute_losses(...)` | 单 batch 的 binding + readout 总损失 |
| `main()` | 训练入口 |

### 5.2 单 batch 训练逻辑

`compute_losses(model, clean_img, clean_aud, labels, cue_mode, cfg, proto_img, proto_aud, epoch=0, step=0)` 的流程：

1. 采样 severity 和 corruption families。
2. 用 `common.build_cue(..., return_masks=True)` 构造 cue 并取得 `aud_mask`；旧配置下若不需要 mask，可保持默认返回 `(img_cue,aud_cue)`。
3. 用 `common.select_targets` 选择 image/audio target。
4. 按配置可选执行 audio target curriculum。
5. 若 `ablation.use_binding_phase=true`，执行 binding 阶段：
   - cue 驱动 Index 和 A-driven Value。
   - clean target 驱动 target Value。
   - bind loss 将 `v_*_from_A` 对齐到 detach 后的 `v_*_target`。
6. 执行 readout 阶段，不传 target。
7. 加入 alignment losses：
   - index consistency。
   - soft class consistency。
   - key alignment。
   - audio detail consistency。
8. 加分类 loss；audio-only 可使用 `lambda_cls_aud_only_mult` 加权。
9. 若启用 audio auxiliary classifier 且 audio cue 存在，加入辅助分类 loss。
10. 加图像恢复 loss。
11. 加音频恢复 loss。`v10b` 中 masked audio loss 只有在以下条件全部满足时追加：
   - `cue_mode in {"corrupt_aud_only","corrupt_both"}`。
   - `aud_kind == "sample"`。
   - 当前 `aud_mode in loss.aud_masked_families`。
   - `aud_mask is not None`。
   - `loss.lambda_aud_masked > 0`。
   `v10d`（F4，默认权重 0）在 `_aud_recon_loss` 内追加两项可选质量 loss：`lambda_aud_ssim`（`1 - batch_ssim(rec, target)`）与 `lambda_aud_masked_grad`（缺失区时频一阶差分 L1，仅在 mask 非空时生效）；`compute_losses` 追加 `lambda_aud_feat`（sample-level audio target 时，冻结 `aud_encoder` 对 rec/target 提取 spike rate 特征的 L1）。三项默认 0，不改变现有行为。
   `v10d`（F3b）readout 前向把 `aud_mask` 透传给 `model.forward(..., aud_cue_mask=aud_mask)`，供 refiner 使用；refiner 关闭或无 mask 时无影响。
12. sample-level audio target 时加入 audio active loss。
13. 加 spike activity regularization。

### 5.3 v11b Recovery 与 Cross-Key 因果训练

v11b 保持 `models/network.py` 中既有三路 Decoder 输入：`Value + 对侧 Key residual + 本模态 cue detail`。本轮不改变模型前向结构，训练侧新增以下约束。

**归一化 energy-weighted masked MSE**：`_masked_audio_weighted_mse(rec,target,mask,gamma)` 的张量均为 `[B,64,64]`。权重 `w=1+gamma*target`，逐样本分母必须为 `(w*mask).flatten(1).sum(1).clamp_min(eps)`，而不是 `mask.sum()`；这样 gamma 只改变洞内相对关注，不随 target energy 任意放大 batch loss。

**主训练 coarse audio loss**：`compute_losses` 从 `out_r["recovered_aud_coarse"]` 读取 `[B,64,64]` coarse prediction。配置 `loss.lambda_aud_coarse` 默认 0，保持 v11a/旧配置兼容；v11b 设为 0.5。若 `mask_for_loss` 非空，计算逐样本 masked L1 + masked MSE；否则计算 full L1 + full MSE。该 loss 不调用 `_apply_audio_refiner`，也不做 visible-region paste-back。日志键为 `aud_coarse`，有 mask 时补 `aud_coarse_mask_mse`。

**decoder pretrain**：继续使用已有 `decoder_pretrain.lambda_coarse_aux`；v11b 设为 0.5。pretrain 的 final helper 与 coarse auxiliary 分开计算，禁止把 paste-back 后的 final 误当作 coarse 监督。

**配对因果 loss**：新增训练配置 `cross_key_conditioning.causal_training`。仅当 Cross-Key enabled、当前 cue 有目标 mask/对侧 Key、且 Bernoulli 采样命中时，基于同一正常前向的 Key rate 构造 zero 与 wrong-class override，再执行两个额外 readout 前向。`scripts/train.py` 复用与 evaluate 相同的异类置换约束，保证有效样本满足 `labels[wrong_perm] != labels`。误差只取方向对应的 coarse maskedMSE：

```text
margin = margin_ratio * E_zero.detach()
L_pair = relu(E_correct - E_zero.detach() + margin)
       + relu(E_correct - E_wrong.detach() + margin)
total += causal_loss_weight * L_pair
```

Key override 只改变 Decoder 条件副本；原 Key/Index/Value 和分类 logits 不变。zero/wrong 前向作为 detached reference，不允许 margin loss 通过恶化对照输出取巧。若 batch 无法构造有效 wrong-class 子集、mask 为空或无对侧 cue，则本项严格跳过并记录 N/A/零计数。日志至少包含 `cross_pair`、方向化 `correct/zero/wrong` coarse mask MSE 和有效样本数。

**分叉与续训**：五个 v11b 配置使用独立 `train.ckpt_path` / `output_version`。Recovery 用 `train.save_milestone_epochs: [100]` 额外保存 `_ep100.pt`，内容与常规 checkpoint 相同。`train.init_ckpt_path` 支持两种显式语义：`init_load_optimizer=true` 时恢复父 checkpoint 的 optimizer/scheduler/epoch，用于 weighted 与 recovery 的严格 20 轮对照；`false` 时只加载 model，并从 `train.start_epoch` 开始重建 optimizer，用于三个统一低学习率的 Cross-Key 分支。`init_ckpt_path` 与 `--resume` 互斥：当前配置 checkpoint 存在且指定 `--resume` 时优先 resume。所有加载默认 strict，禁止静默忽略参数。

Cross-Key 三分支配置 `train.trainable_prefixes`，只保留 `image_decoder.`、`audio_decoder.` 与四个 Cross-Key projector/gate 前缀可训练；Encoder、Memory、detail projector、Refiner 和分类头冻结。三分支 `train.lr` 相同，Cross-Key adapter 通过 `cross_key_conditioning.lr_mult` 使用更高学习率；control 虽构造相同 adapter，但前向关闭且不会得到梯度。

五个配置的训练预算为：Recovery trunk 100 轮；`recovery` / `weighted` 各续训 20 轮；选定 checkpoint 后 `control` / `cross_no_causal` / `cross` 各续训 30 轮。共享 decoder pretrain 只运行一次；分支配置默认跳过 pretrain。

### 5.4 v11d / v11e Cross-Detail、Pair Alignment 与验证

**`_same_class_indices(labels, pair_ids=None)`**

- 用于 v11d/v11e 的 same-class wrong-pair 构造。
- 若提供 `pair_ids`，必须保证置换后的样本同类但不同 `pair_id`；batch 内某类别
  只有一个样本时，该样本在对应 causal/sweep 指标中记为无效。

**`_cross_detail_causal_loss(...)`（v11d+）**

- 仅在 `cross_detail_conditioning.causal_training.enabled=true` 且当前 cue 有对侧
  instance detail 时启用。
- 构造三组 decoder 条件：correct 为正常对侧 instance detail，zero 为关闭
  Cross-Detail，same 为同类不同 `pair_id` 的 instance detail。
- 误差只在目标模态缺失区域上计算 masked MSE；模态完全缺失时，metric mask 为全 1。
- loss 使用相对 margin：

```text
correct_rel = E_correct / stopgrad(E_zero)
same_rel    = E_same / stopgrad(E_zero)
L_detail    = relu(correct_rel - 1 + margin)
            + relu(correct_rel - same_rel + margin)
```

- zero/same reference 前向在 `torch.no_grad()` 下计算，避免模型通过恶化参考输出
  来满足 margin。

**`_pair_alignment_loss(out, labels, pair_ids, cfg)`（v11e）**

- 读取 `out["img_pair_embedding"]` 与 `out["aud_pair_embedding"]`，二者形状均为
  `[B,pair_embed_dim]`。
- 正样本为 batch 对角线的同一真实 `pair_id`；负样本默认限制在同 digit class，
  并排除同 source 的重复视图。
- 计算 image->audio 与 audio->image 两个方向的 masked cross entropy，返回平均
  loss，同时记录 `pair_i2a_acc`、`pair_a2i_acc` 和有效样本数。
- 若 `pair_alignment.enabled=false`，返回 0 loss；若开启但 batch 不含 `pair_id`，
  直接报错，防止伪配对误用。

**`_validate_model(model, loader, cfg, device)`（v11e）**

- 每个验证 epoch 在 validation split 上使用 deterministic fixed-mask
  `corrupt_both`。
- 指标包括 `img_mse`、`aud_mse`、`acc`、`pair_i2a_acc`、`pair_a2i_acc`。
- 组合 score：

```text
score = lambda_img * img_mse
      + lambda_aud * aud_mse
      + lambda_pair * (1 - pair_acc)
      + lambda_cls * (1 - acc)
```

- v11e main 默认 `lambda_pair=0.5`，control 为 0；因此 best checkpoint 比较时
  必须声明选择标准差异，或另行统一选择口径。

### 5.5 checkpoint 格式

主训练保存到：

```text
outputs/checkpoints/cross_modal_snn_v10a.pt
```

checkpoint 字典字段：

| 字段 | 含义 |
|------|------|
| `model` | 模型 `state_dict` |
| `opt` | optimizer state |
| `sched` | scheduler state 或 `None` |
| `cfg` | 运行配置 |
| `epoch` | 已完成 epoch |

decoder pretrain checkpoint：

```text
outputs/checkpoints/cross_modal_snn_v10a_decoder_pretrain.pt
```

v10c 对应 checkpoint：

```text
outputs/checkpoints/cross_modal_snn_v10c.pt
outputs/checkpoints/cross_modal_snn_v10c_decoder_pretrain.pt
```

---

## 6 评估、demo 与消融

### 6.1 `scripts/evaluate.py`

常用命令：

```bash
python -u scripts/evaluate.py --config configs/v10a.yaml --protocol fixed_mask --family_breakdown
python -u scripts/evaluate.py --config configs/v10a.yaml --protocol legacy_random
python -u scripts/evaluate.py --config configs/v10a.yaml --max_batches 20 --severity_curve
python -u scripts/evaluate.py --config configs/v11d.yaml --protocol fixed_mask --family_breakdown --cross_detail sweep
python -u scripts/evaluate.py --config configs/v11e.yaml --protocol fixed_mask --family_breakdown --cross_key sweep --cross_detail sweep
```

参数：

| 参数 | 含义 |
|------|------|
| `--config` | 配置路径 |
| `--ckpt` | 覆盖 checkpoint 路径 |
| `--max_batches` | 快速子集评估 |
| `--severity` | corruption severity |
| `--severity_curve` | 扫描 severity 曲线 |
| `--protocol` | `fixed_mask` 或 `legacy_random` |
| `--family_breakdown` | 对每个 audio corruption family 分开评估 |
| `--cross_key` | v11+ Cross-Key 归因评估；`sweep` 时比较 correct/zero/wrong-class/same-class |
| `--cross_detail` | v11d+ Cross-Detail 归因评估；`sweep` 时比较 correct/zero/wrong/same-class detail |

主要函数：

| 函数 | 职责 |
|------|------|
| `_reseed(seed)` | 固定 corruption mask |
| `_fixed_eval_families(cfg)` | 论文主对照的固定 image/audio corruption family |
| `_audio_masked_metrics(rec, target, mask)` | 计算 audio masked/visible MSE 和 L1，mask 不适用时返回 NaN |
| `_image_masked_metrics(rec, target, mask)` | v11d+ 图像 masked/visible MSE 和 L1，mask 不适用时返回 NaN |
| `_log_audio_diag(diag_rows)` | 打印音频塌缩诊断 |
| `_same_class_indices(labels)` | 为 Cross-Key/Cross-Detail sweep 构造 same-class wrong-pair 对照 |
| `_global_pair_retrieval(...)` | v11e 用 image/audio pair embedding 计算全局 retrieval 指标 |
| `_paired_cross_metrics(...)` | v11e 真配对场景下输出跨模态 sample target 的重建指标 |
| `_paired_detail_metrics(...)` | v11d/v11e Cross-Detail correct/zero/wrong/same 的 masked 区域归因 |
| `eval_mode(...)` | 评估单个 cue mode |
| `eval_audio_family_breakdown(...)` | 输出带 family group 的 `audio_family_breakdown_fixed.csv` |
| `main()` | 6 模式评估入口 |

评估指标：

| 指标 | 含义 | 方向 |
|------|------|------|
| `acc` | digit 分类准确率 | 越高越好 |
| `img_mse` | recovered image 与对应 image target 的 MSE | 越低越好 |
| `psnr` | 图像 PSNR | 越高越好 |
| `ssim` | 简化图像 SSIM | 越高越好 |
| `aud_mse` | recovered log-mel 与对应 audio target 的 MSE | 越低越好 |
| `aud_ssim` | recovered log-mel 与 audio target 的简化 SSIM | 越高越好 |
| `aud_masked_mse/l1` | 音频 mask 区域的逐样本归一误差；无 mask 或非 corrupt-audio 场景为 NaN | 越低越好 |
| `aud_visible_mse/l1` | 音频可见区域的逐样本归一误差；无 mask 或非 corrupt-audio 场景为 NaN | 诊断项 |
| `img_masked_mse/l1` | 图像 mask 区域的逐样本归一误差；无 mask 或非 corrupt-image 场景为 NaN | 越低越好 |
| `img_visible_mse/l1` | 图像可见区域的逐样本归一误差；无 mask 或非 corrupt-image 场景为 NaN | 诊断项 |
| `pairR1_i2a/a2i` | v11e image->audio / audio->image pair retrieval top-1 | 越高越好 |
| `correct_gain` | Cross-Key/Cross-Detail correct 相对 zero 的 masked 区域收益 | 越高越好 |
| `wrong_damage/same_damage` | wrong-class 或 same-class wrong-pair 相对 correct 的损伤 | 归因项 |
| `pix_var` | 图像重建多样性诊断 | 诊断项 |
| `pair_l2` | 随机样本对重建距离 | 诊断项 |
| `rec_mean/std/max`、`topk_recall` | 音频塌缩诊断 | 诊断项 |

v11e 使用 `paired_manifest` 数据集时，缺失模态的评估 target 默认为当前真实
paired sample，而不是 class medoid。`fixed_mask` 表示每个 cue mode 使用固定 seed、
固定 corruption family 和固定 severity，适合作为论文主表；`legacy_random` 每个
batch 重新随机采样 corruption family/mask，更适合作为鲁棒性附表。Cross-Detail
归因只看 masked 区域：correct detail 应优于 zero detail，wrong/same detail 不应
无代价提升，否则说明 decoder 在偷用不该用的实例细节。

### 6.2 `scripts/demo_inference.py`

常用命令：

```bash
python -u scripts/demo_inference.py --config configs/v10d.yaml --num 8 --severity 0.4
python -u scripts/demo_inference.py --config configs/v10d.yaml --num 8 --severity 0.4 --protocol legacy_random
python -u scripts/demo_inference.py --config configs/v11e.yaml --num 8 --severity 0.4 --protocol fixed_mask
python -u scripts/demo_inference.py --config configs/v11e.yaml --num 8 --severity 0.4 --protocol legacy_random
```

**与 `evaluate.py` 的路径对齐（v10d+）**：

- `fixed_mask`：`_make_visible_aud_cue(..., return_mask=True)` 在同一次 `corrupt_audio` 采样中保留 `aud_mask`；`legacy_random`：`build_cue(..., return_masks=True)` 取 `masks["aud"]`。
- `audio-only` / `both` 前向必须传 `aud_cue_mask=aud_mask`（与 evaluate 相同），使 `AudioRefiner` + `visible_paste_back` 生效；`image-only` 不传 mask。
- demo 汇总指标（`aud_mse` / `aud_ssim`、塌缩诊断）一律基于最终 `recovered_aud`，与 evaluate 同口径；**不是** `recovered_aud_coarse`。
- 可视化列（含音频 cue 的模式）：在 `corrupted audio` 与 `recovered audio` 之间插入 `audio mask`（1=缺失）与 `coarse audio`（decoder 输出，refiner 前），便于区分 paste-back 前后。

默认输出：

| 输出 | 含义 |
|------|------|
| `outputs/outputs_v10a/figures/demo_aud_only.png` | audio-only cue -> category image + sample audio |
| `outputs/outputs_v10a/figures/demo_img_only.png` | image-only cue -> sample image + category audio |
| `outputs/outputs_v10a/figures/demo_both.png` | bimodal cue -> sample image + sample audio |
| `outputs/outputs_v10a/tables/demo_eval_table.txt` | demo 汇总表与逐样本分类表 |

v11d/v11e 的 demo 保留 fixed 与 random 两套可视化入口。`fixed_mask` 文件名保持
`demo_*.png`；`legacy_random` 输出 `demo_*_random.png`，用于检查同一 checkpoint
在随机 mask/family 下是否出现只适配固定缺失模板的现象。v11e 表格额外标记
sample/paired-sample target，避免把真实配对恢复误读成类别 medoid 恢复。

### 6.3 `scripts/make_v10a_ablations.py`

生成临时消融配置到：

```text
outputs/ablations_v10a/configs/
```

预设变体：

| 变体 | 关键变化 |
|------|----------|
| `v10a_ablate_A_simultaneous` | detail concat/pretrain baseline，去掉 phased input |
| `v10a_ablate_B_detach_false_only` | v9 风格 detail concat，无 decoder pretrain，detach false |
| `v10a_ablate_C_pretrain_only` | v9 风格 detail concat，启用 decoder pretrain，detach true |

### 6.4 `scripts/run_v10a_suite.py`

常用命令：

```bash
python -u scripts/run_v10a_suite.py --config configs/v10a.yaml --with_ablations
python -u scripts/run_v10a_suite.py --config configs/v10a.yaml --ablations_only
```

流程：

1. 除非指定 `--ablations_only`，先跑主训练。
2. 按需生成消融配置。
3. 逐个顺序跑消融。
4. 每个实验写单独日志。
5. 任一实验失败即停止。

### 6.5 工具脚本

| 脚本 | 用途 |
|------|------|
| `scripts/mkdir_outputs.py` | 创建 `outputs/checkpoints` 和版本化 `figures/logs/tables` |
| `scripts/plot_eval_summary.py` | 解析 demo/eval 日志并渲染 PNG + CSV 表 |
| `scripts/smoke_test.py` | 随机张量前向/反向检查，不需要下载数据 |
| `scripts/bootstrap.py` | 将 cwd 和 import path 设置为项目根目录 |

---

## 7 配置约定

### 7.1 `configs/v10a.yaml` 模块

| 配置块 | 用途 |
|--------|------|
| `seed`, `device` | 随机种子与设备 |
| `snn` | 时间步、LIF 参数、编码方式、音频 decoder 容量 |
| `detail_conditioning` | decoder detail fusion、detach 行为、zero-missing 策略 |
| `decoder_pretrain` | decoder-only 预训练设置 |
| `dims` | 主要模型维度 |
| `index` | Index 输入调度、k-WTA、抑制设置 |
| `value` | target Value delay |
| `ablation` | 核心模块消融开关 |
| `audio` | FSDD/log-mel 与归一化设置 |
| `cue_modes` | 8 种 cue mode 采样概率 |
| `corruption` | severity、corruption family、课程学习、固定评估 mask |
| `loss` | 分类、重建、binding、alignment、正则 loss 权重 |
| `data` | 数据根目录、MNIST 开关、batch size、workers、subset |
| `train` | 输出版本、epoch、optimizer、scheduler、checkpoint 路径 |

### 7.2 v10a 关键默认值

| 参数 | 值 | 含义 |
|------|----|------|
| `snn.T` | `20` | SNN 时间窗 |
| `dims.N_index` | `512` | Index attractor 尺寸 |
| `index.k_wta` | `96` | 每步保留的 Index 活跃神经元 |
| `detail_conditioning.fusion` | `gated_concat` | gated cue detail 融合 |
| `detail_conditioning.detach_value_for_recon` | `true` | 重建 loss 不反传拖动 Value/Index |
| `audio.n_mels`, `audio.n_frames` | `64`, `64` | log-mel 网格 |
| `train.output_version` | `v10a` | 版本化输出目录 |
| `train.ckpt_path` | `outputs/checkpoints/cross_modal_snn_v10a.pt` | 主 checkpoint |

### 7.3 v10b 关键变化

`configs/v10b.yaml` 完整复制 `configs/v10a.yaml` 后做定向修改，不覆盖 v10a 基线。

| 参数 | v10b 值 | 含义 |
|------|---------|------|
| `corruption.aud_mode` | `time_mask` | 训练默认音频 family 对齐连续时间片段缺失 |
| `corruption.aud_train_modes` | `["time_mask"]` | 训练池只包含论文对齐主 family |
| `corruption.aud_family_curriculum.enabled` | `false` | 关闭 v10a 的多 family 课程调度 |
| `corruption.eval_fixed.aud_mode` | `time_mask` | fixed-mask 评估也使用连续时间片段缺失 |
| `loss.lambda_aud_masked` | `1.0` | 在满足白名单和 sample-level 条件时追加 masked audio loss |
| `loss.aud_masked_families` | `["time_mask"]` | v10b 主实验只对 `time_mask` 加 masked loss |
| `train.output_version` | `v10b` | 版本化输出目录 |
| `train.ckpt_path` | `outputs/checkpoints/cross_modal_snn_v10b.pt` | v10b 主 checkpoint |
| `decoder_pretrain.ckpt_path` | `outputs/checkpoints/cross_modal_snn_v10b_decoder_pretrain.pt` | v10b decoder pretrain checkpoint |

`v10b` 的核心比较口径：

- `v10a`：通用鲁棒性基线，多 family 音频 corruption。
- `v10b`：论文对齐主实验，训练与 fixed-mask 评估均优先看 `time_mask`。
- `legacy_random` 仍可用于 `v10b` 的未见 family 泛化检查，但不作为论文主对齐指标。

### 7.4 v10c 关键变化

`configs/v10c.yaml` 完整复制 `configs/v10b.yaml` 后做训练协议修正。

| 参数 | v10c 值 | 含义 |
|------|---------|------|
| `corruption.img_mode` | `occlusion` | 训练图像 family 固定为遮挡块，与 fixed eval 对齐 |
| `corruption.aud_mode` | `time_mask` | 音频仍使用连续时间片段缺失 |
| `corruption.train_severity` | `0.4` | 后期训练最大残缺强度 |
| `corruption.severity_max` | `0.4` | staged curriculum 的上限 |
| `decoder_pretrain.epochs` | `25` | decoder warmup 延长到 25 轮 |
| `decoder_pretrain.corrupt_detail` | `true` | pretrain 时用固定 family 残缺 cue 生成 detail state |
| `decoder_pretrain.img_mode` | `occlusion` | pretrain 图像 detail 残缺方式 |
| `decoder_pretrain.aud_mode` | `time_mask` | pretrain 音频 detail 残缺方式 |
| `decoder_pretrain.corrupt_severity` | `0.4` | pretrain 残缺强度 |
| `decoder_pretrain.use_masked_audio_loss` | `true` | 音频 pretrain 复用 masked audio loss |
| `train.epochs` | `70` | 主训练默认 70 轮 |
| `train.output_version` | `v10c` | 版本化输出目录 |

v10c 的预期是缓解 v10b 的音频近黑图塌缩，同时保持 v10b 在 fixed-mask 分类和图像恢复上的优势。

### 7.5 v10d 关键变化

`configs/v10d.yaml` 从 `configs/v10c.yaml` 派生，落地 F3a/F3b 结构改动、F4 loss 代码（默认权重 0）与保守 cue 比例调整。

| 参数 | v10d 值 | 含义 |
|------|---------|------|
| `snn.aud_refine_type` | `gated_dilated` | F3a：AudioDecoder refine 块改为门控 + dilation(1,2,4) |
| `audio_refiner.enabled` | `true` | F3b：启用谱图空间 refiner 支路 |
| `audio_refiner.hidden_ch` | `32` | refiner 隐藏通道数 |
| `audio_refiner.blocks` | `4` | refiner gated conv 块数（dilation 1,2,4,8，RF ~31 帧）|
| `audio_refiner.max_dilation` | `16` | dilation 上限 |
| `audio_refiner.delta_scale` | `1.0` | refiner delta 幅度（tanh 后缩放）|
| `audio_refiner.visible_paste_back` | `true` | 文献式 inpainting：缺失区用预测、可见区原样保留 aud_cue |
| `loss.lambda_aud_ssim` | `0.0` | F4：`1-SSIM` loss（代码就绪，默认关闭）|
| `loss.lambda_aud_masked_grad` | `0.0` | F4：缺失区时频梯度 loss（默认关闭）|
| `loss.lambda_aud_feat` | `0.0` | F4：冻结 aud_encoder 特征 loss（默认关闭）|
| `cue_modes.p_corrupt_img_only` | `0.15` | 保守调整（v10c 为 0.10）|
| `cue_modes.p_corrupt_aud_only` | `0.20` | 与 v10c 相同 |
| `cue_modes.p_corrupt_both` | `0.25` | 保守调整（v10c 为 0.20）|
| `cue_modes.p_clean_img_only` | `0.10` | 与 v10c 相同 |
| `cue_modes.p_clean_aud_only` | `0.15` | 保守调整（v10c 为 0.20）|
| `cue_modes.p_clean_both` | `0.15` | 保守调整（v10c 为 0.20）|
| `train.output_version` | `v10d` | 版本化输出目录 |
| `train.ckpt_path` | `outputs/checkpoints/cross_modal_snn_v10d.pt` | 主 checkpoint |
| `decoder_pretrain.ckpt_path` | `outputs/checkpoints/cross_modal_snn_v10d_decoder_pretrain.pt` | pretrain checkpoint |

cue 比例合计仍为 1.0（0.15+0.20+0.25+0.10+0.15+0.15）。为保证「提升可归因于 refiner 而非采样比例」，cue 比例仅小幅调整，F4 三项质量 loss 默认权重为 0，仅作为后续独立消融入口。v10d checkpoint 与 v10c/更早版本不兼容（decoder 结构变化 + 新增 refiner 参数），需从头训练；`aud_refine_type: plain` + `audio_refiner.enabled: false` 时结构退回 v10c，可加载旧 checkpoint。

**v10d 归因消融配置**（隔离结构改动与 cue 比例调整）：

| 配置 | 相对 v10d 的差异 | 用途 |
|------|------------------|------|
| `configs/v10d_ablation_v10c_ratio.yaml` | cue 比例换回 v10c（0.10/0.20/0.20/0.10/0.20/0.20），结构不变 | 与 v10c 对比 → 隔离「结构改动」在旧比例下的净效果 |
| `configs/v10d_ablation_refiner_off.yaml` | `audio_refiner.enabled: false`，cue 比例与 gated/dilated decoder 同 v10d；由于 paste-back 写在 refiner 分支内，该配置同时关闭 refiner + paste-back 后处理 | 与 v10d 对比 → 隔离「refiner/paste-back 后处理路径」的合并贡献，不能解读为只关 refiner |

**关于 refiner 与 decoder pretrain**：`decoder_pretrain` 阶段的冻结白名单只含 `image_decoder.` / `audio_decoder.`，不含 `audio_refiner.`，且 pretrain 直接调用 `model.audio_decoder(...)` 不经过 refiner。因此 v10d 的 refiner 是**主训练阶段从零学习**的，不是预训练好的 inpainting refiner；这是有意的设计取舍，记录于此以免后续误读实验归因。

**关于 `lambda_aud_feat` 冻结语义**：`_aud_feature_loss` 提取 rec 特征时临时把 `aud_encoder` 参数 `requires_grad` 置 false，backward 只更新 decoder/refiner（rec 侧），不改写 audio encoder 权重；target 侧在 `no_grad` 下提取。默认权重仍为 0。

### 7.6 v11b 关键变化

`configs/v11b_*.yaml` 从 v11a 派生，但不覆盖 v11a checkpoint。配置族共同启用主训练 coarse audio 监督、修正后的 weighted loss 语义和显式父 checkpoint 初始化。

v11b 成为当前唯一运行配置族后，根配置目录不再保留
`configs/v11a.yaml` 与 `configs/v11a_control.yaml`。两份旧配置的历史内容仍可从
Git 的 v11a 提交恢复；当前训练、评估和 demo 禁止继续引用已经退役的 v11a YAML，
避免误用旧损失与旧实验预算。

| 配置 | Cross-Key | weighted masked loss | causal loss | 累计主训练预算 |
|---|---:|---:|---:|---:|
| `v11b_recovery.yaml` | 关（模块保留） | 0 | 关 | 120 |
| `v11b_weighted.yaml` | 关（模块保留） | 0.25 | 关 | 120 |
| `v11b_control.yaml` | 关（模块保留） | 采用选定 Recovery 设置 | 关 | 150 |
| `v11b_cross_no_causal.yaml` | 开 | 同 control | 关 | 150 |
| `v11b.yaml` | 开 | 同 control | 开 | 150 |

所有分支必须使用独立 checkpoint，禁止多个配置写入同一路径。`scripts/mkdir_outputs.py` 按各自 `output_version` 建目录。正式评估继续使用 fixed-mask、severity 0.4、5-family breakdown；Cross-Key 两个配置额外运行 `--cross_key sweep`。

v11b 的 sweep 在 v11a normal/zero/wrong-class 基础上增加 same-class different-sample Key。每个类别在 batch 内循环置换，类别样本数不足 2 时该样本记为无效；有效置换不得指向自身。输出增加方向化 coarse/final `same_damage = same_mask_mse - correct_mask_mse`，`scripts/plot_eval_summary.py` 同步增加对应列。same damage 接近 0 表示对侧 Key 主要提供类别语义，不得据此声称恢复了另一模态的实例细节。

### 7.7 v11c AudioRefiner bypass

v11b 正式评估显示，AudioDecoder coarse 的五 family MaskMSE 约为 `0.0130`，
AudioRefiner 后的 final 反而约为 `0.0297`；四个非 `partial_temporal` family 的
coarse/final 约为 `0.0071/0.0225`。因此 v11c 暂时旁路外置 AudioRefiner，保留
AudioDecoder 内部 `gated_dilated` refinement、ImageRefiner、visible paste-back 与
既有 Cross-Key Decoder 条件路径。

v11c 不从 state dict 删除 AudioRefiner。配置采用
`audio_refiner.enabled=true`、`bypass=true`、`pasteback_only=true`：

```text
coarse_aud = AudioDecoder(Value + cross-Key residual + audio detail)
final_aud  = aud_mask * coarse_aud + (1-aud_mask) * aud_cue  # 有残缺 cue
final_aud  = coarse_aud                                      # 无 mask/cue
```

保留模块参数是为了 strict 加载共同父 checkpoint
`cross_modal_snn_v11b_recovery.pt`，不代表模块参与前向或训练。构造后将
`audio_refiner.*.requires_grad=false`；`decoder_pretrain.train_audio_refiner=false`。
由于 final 缺失区与 coarse 完全相同，v11c 将 `loss.lambda_aud_coarse` 设为 0，
避免 final masked loss 与 coarse auxiliary 对同一张量重复计权。

| 配置 | AudioRefiner | Cross-Key | causal loss | 父 checkpoint | 累计预算 |
|---|---|---|---|---|---:|
| `configs/v11c_control.yaml` | 构造但 bypass | 关 | 关 | v11b recovery ep120 | 150 |
| `configs/v11c.yaml` | 构造但 bypass | 开 | 开 | v11b recovery ep120 | 150 |

两者使用相同 seed、父 checkpoint、30 轮微调、可训练前缀和评估 mask。主验收先看
audio final MaskMSE 是否等于对应 coarse MaskMSE，再比较 v11c 与 control 的
correct/zero/wrong/same Cross-Key sweep；全谱 SSIM 仍只作辅助指标。

### 7.8 v11d 固定伪配对与 Cross-Detail

`configs/v11d.yaml` 从 v11c 继续，保留 AudioRefiner bypass、Cross-Key 与
visible paste-back，并新增固定一对一伪配对和 Cross-Detail decoder 条件。

| 配置块 | 关键字段 | 含义 |
|---|---|---|
| `data.pairing` | `enabled=true`, `mode=fixed_augmented_one_to_one`, `return_pair_id=true` | MNIST/FSDD 同类样本固定配对；每个样本获得稳定 `pair_id` |
| `data.pairing` | `sample_targets_for_missing=true`, `paired_target_kind=sample` | 缺失模态恢复 target 从 class medoid 改为当前 paired sample |
| `cross_detail_conditioning` | `enabled=true`, `build_modules=true` | 为 decoder 提供对侧 cue detail residual |
| `cross_detail_conditioning` | `detach_source=true` | 归因优先：Cross-Detail 本身可训，但不让对侧 encoder 被恢复 loss 直接拖动 |
| `cross_detail_conditioning` | `causal_training=true` | correct/zero/wrong/same-class detail 对照训练 |
| `train` | `init_ckpt_path=outputs/checkpoints/cross_modal_snn_v11c.pt`, `init_strict=false` | 继承 v11c 权重，新模块按允许缺失前缀初始化 |

`configs/v11d_control.yaml` 的原则是只关闭 Cross-Detail 与相关 causal loss，
保留 v11c 主干和固定伪配对数据协议。这样 main-control 差异主要归因于
Cross-Detail，而不是数据采样或父 checkpoint。

### 7.9 v11e GRID 真配对与 Pair Alignment（已废止方案）

> 该方案是 v11e 的首次设计，已被 7.10 的 MNIST/FSDD 类别级协议覆盖。
> `paired_manifest` 的通用加载能力仍保留；GRID 版本专属文档和准备脚本留在
> 历史提交中，不再由当前 v11e 分支携带或启用。

`configs/v11e.yaml` 继承 v11d 的 decoder 融合与评估能力，但把数据源切到
GRID 真实视听配对，并从头训练，不继承 MNIST/FSDD checkpoint。

| 配置块 | 关键字段 | 含义 |
|---|---|---|
| `data.dataset` | `paired_manifest` | 使用 manifest 驱动的真实 image/audio pair |
| `data.manifest_path` | `_data/grid_v11e/pairs.csv` | `scripts/prepare_grid_v11e.py` 生成的 pair 清单 |
| `data` | `use_mnist=false`, `train_required=true` | v11e 不再依赖 MNIST/FSDD 伪配对训练 |
| `audio_refiner` | `enabled=false` 或 bypass | 避免旧 refiner 对真配对音频恢复造成额外混淆 |
| `pair_alignment` | `enabled=true`, `embed_dim=128`, `temperature=0.07`, `lambda_pair=0.5` | image/audio detail embedding 的同源配对对齐 |
| `cross_key_conditioning` | `detach_source=false` | 允许真配对训练中跨模态条件端到端适配 |
| `cross_detail_conditioning` | `detach_source=false` | 允许对侧 detail 在真配对目标下被恢复 loss 校准 |
| `train` | `output_version=v11e`, `epochs=100`, `init_ckpt_path=null` | 独立训练与独立 checkpoint |

`configs/v11e_control.yaml` 应关闭 Cross-Detail 和 pair alignment，但保留
GRID manifest、同样的 cue mode、severity、batch size 与训练预算。v11e 的主结论
不能再用 digit ACC 单独描述；需要同时报告分类、masked recovery、pair retrieval
和 Cross-Detail 归因。

### 7.10 v11e 当前方案：MNIST/FSDD 类别级绑定

`configs/v11e.yaml` 将 v11c 的网络和损失基线展开为自包含配置，重新使用 MNIST 图像与
FSDD `64 mel x 64 frames` log-mel。类别内配对为动态 many-to-many：dataset item
保留 MNIST 实例，但训练音频从相同 digit 的 FSDD train pool 随机抽取。

| 配置块 | 关键字段 | 含义 |
|---|---|---|
| `data.dataset` | `mnist_fsdd` | 使用 MNIST + FSDD，不读取 GRID manifest |
| `data.pairing` | `mode=category_many_to_many`, `return_pair_id=false` | 同类随机组合，不建立人工固定实例对 |
| `data.pairing` | `sample_targets_for_missing=false` | 单模态缺失侧使用 train class medoid |
| `detail_conditioning` | `detach_value_for_recon=true` | decoder 仍读取 Value，但恢复梯度不经 Value 改写 Index |
| `cross_key_conditioning` | main 开、control 关，`detach_key=true` | 只检验对侧 Key 的类别条件贡献 |
| `cross_detail_conditioning` | 关闭 | 不把无真实对应的笔迹/音色细节注入对侧 decoder |
| `pair_alignment` | 关闭 | 不执行 exact-pair InfoNCE 或 pair Recall@1 |

`common.select_targets` 是唯一目标选择入口：image-only 返回
`sample/category`，audio-only 返回 `category/sample`，所有双模态模式返回
`sample/sample`。图像与音频 class medoid 都只由 train split 构建，evaluate/demo
复用训练原型。

`v11e_control` 使用相同数据、target、初始化和训练预算，只关闭 Cross-Key 前向与
causal loss；Cross-Key 模块仍构造，以保持 main/control checkpoint 结构一致。

---

## 8 数据下载与准备

### 8.1 数据集

| 数据集 | 类型 | 来源 | 路径 |
|--------|------|------|------|
| MNIST | 图像 digit 数据集 | `torchvision.datasets.MNIST` | `_data/MNIST` |
| FSDD | spoken digit wav | Jakobovski free-spoken-digit-dataset | `_data/fsdd/recordings` |
| GRID | 真实视听配对语音数据 | 本地 GRID audio/frame archive | `_data/grid_v11e` |

### 8.2 准备逻辑

1. `data.dataset.build_loaders(cfg)` 在 real audio 启用时调用 `ensure_audio_norm_stats(cfg)`。
2. `ensure_audio_norm_stats` 需要 FSDD train wav。
3. 若 FSDD wav 不存在且 `audio.auto_download=true`，`data.fsdd.ensure_fsdd` 先尝试 git clone，再尝试 zip 下载。
4. MNIST 由 torchvision 按需下载。
5. 音频归一化统计保存到 `_data/audio_norm_stats.pt`。

v11d 的固定伪配对仍使用 MNIST + FSDD，但 `PairedAudioVisualDataset` 会在构造时
为每个图像样本绑定一个稳定同类音频样本，并把 `pair_id` 随 batch 返回。该协议只
解决“训练中同一图像是否总配同一条音频”的问题，不代表图像和音频来自同一个真实事件。

历史 GRID 版 v11e 使用 `paired_manifest`；其准备工具保留在历史提交中，不属于
当前分支的运行入口。

脚本应生成：

```text
_data/grid_v11e/
├── pairs.csv
├── images/
└── audio/
```

`pairs.csv` 至少包含 pair id、split、label、image path、audio path。该可选数据集加载时
会校验文件存在、split 不交叉、可选 speaker/disjoint 约束，以及 image/audio tensor
shape 是否满足当前配置。

当前类别级 v11e 不执行 GRID 准备步骤，只需 MNIST、FSDD 与训练集音频归一化统计。

### 8.3 自动下载失败时的手动处理

将 FSDD wav 放到以下目录，文件名应类似 `{digit}_{speaker}_{index}.wav`：

```text
_data/fsdd/recordings/
```

---

## 9 结果与产物格式

### 9.1 Checkpoints

| 路径 | 格式 | 含义 |
|------|------|------|
| `outputs/checkpoints/cross_modal_snn_v10a.pt` | PyTorch checkpoint dict | v10a 主模型 |
| `outputs/checkpoints/cross_modal_snn_v10a_decoder_pretrain.pt` | PyTorch checkpoint dict | decoder pretrain |
| `outputs/checkpoints/cross_modal_snn_v10b.pt` | PyTorch checkpoint dict | v10b 主模型 |
| `outputs/checkpoints/cross_modal_snn_v10b_decoder_pretrain.pt` | PyTorch checkpoint dict | v10b decoder pretrain |
| `outputs/checkpoints/cross_modal_snn_v10c.pt` | PyTorch checkpoint dict | v10c 主模型 |
| `outputs/checkpoints/cross_modal_snn_v10c_decoder_pretrain.pt` | PyTorch checkpoint dict | v10c decoder pretrain |
| `outputs/checkpoints/cross_modal_snn_v11c.pt` | PyTorch checkpoint dict | v11c AudioRefiner bypass 基线 |
| `outputs/checkpoints/cross_modal_snn_v11d.pt` | PyTorch checkpoint dict | v11d 固定伪配对 + Cross-Detail 主模型 |
| `outputs/checkpoints/cross_modal_snn_v11d_control.pt` | PyTorch checkpoint dict | v11d Cross-Detail control |
| `outputs/checkpoints/cross_modal_snn_v11e.pt` | PyTorch checkpoint dict | v11e MNIST/FSDD 类别级主模型 |
| `outputs/checkpoints/cross_modal_snn_v11e_control.pt` | PyTorch checkpoint dict | v11e 关闭 Cross-Key 的类别级对照 |
| `outputs/checkpoints/cross_modal_snn_v9*.pt` | 历史 checkpoint | 早期 F 阶段迭代 |

### 9.2 版本化输出目录

`train.output_version: v10a` 时，产物应写入：

```text
outputs/outputs_v10a/
├── figures/
├── logs/
└── tables/
```

创建本文档时尚未检测到 `outputs/outputs_v10a/`，因此正式运行 v10a 前需要先运行 `scripts/mkdir_outputs.py` 或由相关脚本创建目录。

`train.output_version: v10b` 时，产物写入：

```text
outputs/outputs_v10b/
├── figures/
├── logs/
└── tables/
```

`train.output_version: v10c` 时，产物写入：

```text
outputs/outputs_v10c/
├── figures/
├── logs/
└── tables/
```

v11c/v11d/v11e 遵循相同版本化目录规则：

```text
outputs/outputs_v11c/
outputs/outputs_v11d/
outputs/outputs_v11d_control/
outputs/outputs_v11e/
outputs/outputs_v11e_control/
```

当前 v11e 依赖 `_data/MNIST/`、`_data/fsdd/recordings/` 与
`_data/audio_norm_stats.pt`；这些数据文件不属于 `outputs/` 产物，不应混入
checkpoint 仓库。

### 9.3 日志

| 日志 | 建议路径 | 内容 |
|------|----------|------|
| 主训练日志 | `outputs/outputs_v10a/logs/train_v10a_50ep.log` | epoch/step loss 与 checkpoint 保存记录 |
| fixed eval 日志 | `outputs/outputs_v10a/logs/eval_v10a_fixed_mask.log` | 6 模式指标与音频诊断 |
| random eval 日志 | `outputs/outputs_v10a/logs/eval_v10a_legacy_random.log` | 随机 family 鲁棒性检查 |
| suite 日志 | `outputs/outputs_v10a/logs/suite_v10a_with_ablations.log` | 长实验编排输出 |
| v10b fixed eval 日志 | `outputs/outputs_v10b/logs/eval_v10b_fixed_mask.log` | 含 masked audio 指标的 time-mask 对齐评估 |
| v10c fixed eval 日志 | `outputs/outputs_v10c/logs/eval_v10c_fixed_mask.log` | fixed occlusion + time-mask 下的 v10c 主评估 |
| v11d fixed eval 日志 | `outputs/outputs_v11d/logs/eval_v11d_fixed_mask.log` | 固定伪配对 + Cross-Detail sweep 主评估 |
| v11d control eval 日志 | `outputs/outputs_v11d_control/logs/eval_v11d_control_fixed_mask.log` | 关闭 Cross-Detail 的对照评估 |
| v11e fixed eval 日志 | `outputs/outputs_v11e/logs/eval_v11e_fixed_mask.log` | MNIST/FSDD 类别级 fixed-mask 主评估 |
| v11e random demo 日志 | `outputs/outputs_v11e/logs/demo_v11e_legacy_random.log` | random 可视化与逐样本诊断 |

### 9.4 表格与图像

| 产物 | 路径 | 格式 |
|------|------|------|
| demo figures | `outputs/outputs_v10a/figures/demo_*.png` | PNG |
| demo table | `outputs/outputs_v10a/tables/demo_eval_table*.txt` | 对齐文本 |
| audio family breakdown | `outputs/outputs_v10a/tables/audio_family_breakdown_fixed.csv` | CSV |
| v10b audio family breakdown | `outputs/outputs_v10b/tables/audio_family_breakdown_fixed.csv` | CSV，包含 `family_group` 和 masked audio 指标 |
| v10c audio family breakdown | `outputs/outputs_v10c/tables/audio_family_breakdown_fixed.csv` | CSV，包含 `family_group` 和 masked audio 指标 |
| v11d Cross-Detail sweep | `outputs/outputs_v11d/tables/*cross_detail*.csv` | CSV，记录 correct/zero/wrong/same 归因指标 |
| v11e Cross-Key sweep | `outputs/outputs_v11e/tables/*cross_key*.csv` | CSV，记录 correct/zero/wrong/same Key 归因 |
| v11e random demo figures | `outputs/outputs_v11e/figures/demo_*_random.png` | PNG，legacy_random 可视化 |
| 渲染后的评估表 | `outputs/outputs_v10a/tables/*.png` 和 `.csv` | PNG + CSV |

---

## 10 后续 F 阶段修改顺序

未来任何代码修改都按以下顺序执行：

1. 读取 `docs/user_requirements.md`。
2. 读取本文档。
3. 读取 `docs/dev_log.md` 与 `docs/idea_report.md`。
4. 判断改动类型：配置、模型结构、数据管线、训练、评估、demo、实验设计或文档。
5. 若行为、API、tensor shape、输出格式或命令变化，先更新本文档；若涉及新版本或消融设计，同步更新 `docs/idea_report.md`。
6. 再修改代码。
7. 运行最小必要验证：
   - 随机张量 API/shape：`python -u scripts/smoke_test.py`
   - 训练入口：短 epoch 或 subset run
   - 评估入口：`--max_batches`
   - demo 渲染：低 `--num`
8. 在 `docs/dev_log.md` 追加记录：
   - 改动原因。
   - 修改文件。
   - 预期效果。
   - 文档同步情况。
   - 验证结果。
9. 若命令或输出变化，更新 `docs/dev_log.md` 末尾 `运行说明`。

---

## 11 本实现指南校验

### 11.1 实验要求覆盖

| 项目 | 状态 | 说明 |
|------|------|------|
| 8 种 cue mode | 通过 | 已覆盖 `common.py`、train/evaluate/smoke；demo 保留三种主展示视图 |
| 跨模态恢复 target 语义 | 通过 | 已在 2.4 节说明 |
| binding/readout 两阶段 | 通过 | 已在模型和训练章节说明 |
| detail-conditioned decoders | 通过 | 已在架构和 `network.py` 章节说明 |
| v10a 消融 | 通过 | 已在 6.3 节说明 |
| v11c AudioRefiner bypass | 通过 | 已在 7.7 节记录 coarse/final 与 paste-back 语义 |
| v11d Cross-Detail | 通过 | 已在 0.9、3.6、4.4、5.4、7.8 节记录 |
| v11e 类别级绑定 | 通过 | 已在 0.11、7.10、8.2 节记录目标粒度、数据与对照定义 |
| 评估协议 | 通过 | fixed_mask 和 legacy_random 均已说明 |
| 音频塌缩诊断 | 通过 | 已在指标和结果文件中说明 |

### 11.2 逻辑一致性

| 检查项 | 状态 | 说明 |
|--------|------|------|
| tensor shape 链路 | 通过 | image/audio encoder -> Key -> Index -> Value -> decoder 主链路与 v10 系一致；v11e 当前保留 Cross-Key，不启用 Cross-Detail/pair embedding |
| 防止 target 泄漏 | 通过 | inference 强制清空 target，decoder 主输入为 A-driven Value |
| 输出路径 | 通过 | v10/v11 版本化输出、checkpoint、eval/demo 表格路径已记录 |
| requirements 规则 | 警告 | 当前 `requirements.txt` 包含 torch/torchvision/torchaudio，与 ResearchPilot 默认“不写 torch 系”规则不同；本文档保留当前项目现实状态 |

### 11.3 完整性

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 源码文件覆盖 | 通过 | 当前 root/data/models/scripts/config 文件均已覆盖 |
| 函数级说明 | 通过，部分压缩 | 核心类和函数含签名、职责与逻辑；绘图辅助函数按角色汇总 |
| D-F 迭代约束 | 通过 | 第 0 节和第 10 节已规定 |
| A-C/F 文档衔接 | 通过 | `docs/idea_report.md` 已存在并记录 v11c/v11d/v11e 评价与后续路线 |
