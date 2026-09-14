# 开发日志：Cross-Modal Attractor SNN

> 创建时间：2026-07-06 18:43 | 当前活动版本：v13pro | 最后整理：2026-09-15
> 关联实现指南：`docs/implementation.md`
> 当前阶段：ResearchPilot F 阶段补档与迭代
> 本文件原则上只追加，不删除。每次代码修改都必须追加新的日志条目。

## 项目概览


| 项目              | 内容                                            |
| --------------- | --------------------------------------------- |
| 研究方向            | 跨模态 attractor SNN 联想记忆                        |
| 当前阶段            | F：代码迭代                                        |
| 当前配置            | `configs/v13pro.yaml` 套件模板；逐实验配置生成到输出目录 |
| 代码结构            | 根目录下的 `data/`、`models/`、`scripts/`、`configs/` |
| 主要任务            | MNIST 图像 + FSDD 音频 cue -> digit 分类 + 图像/音频恢复  |
| 框架              | PyTorch                                       |
| 主 checkpoint 目标 | `outputs/v13pro/seed_<seed>/<experiment>/best.pt` |
| 版本化输出目标         | `outputs/v13pro/`                       |
| 硬性工作流           | 先改文档，再改代码；每次改代码后追加本日志                         |

## 当前版本导航

当前活动版本为 `v13pro`。按文档职责定位内容：

| 内容 | 入口 |
|---|---|
| v13pro 初始方案、研究问题、预期验收 | `docs/idea_report.md` 顶部 v13pro 条目 |
| v13pro 实际结构、配置、张量和命令 | `docs/implementation.md` |
| 统一评估入口（按版本顺序） | [统一汇总导航](#evaluation-format-20260912) |
| v11c 两组实验 | [v11c 完整指标](#evaluation-v11c) |
| v11d 两组实验 | [v11d 完整指标](#evaluation-v11d) |
| v11e 两组实验 | [v11e 完整指标](#evaluation-v11e) |
| v11f 三组实验 | [v11f 完整指标](#evaluation-v11f) |
| v11g 三组实验 | [v11g 完整指标](#evaluation-v11g) |
| v12a 三组实验 | [v12a 评估归档](#evaluation-v12a) |
| v12b 三组实验 | [v12b 完整指标与结论](#evaluation-v12b) |
| v13pro 实现与验证 | [2026-09-15 实现记录](#implementation-v13pro) |
| v13pro 当前运行命令 | `docs/implementation.md` 第 6 节 |

本文件前部的 v10a 状态表和中部各版本条目是历史记录；其中出现的旧配置、旧路径或
已删除文档名不能作为当前运行入口。历史日志只追加，不通过移动或删除旧条目修订。

目录规则：开发记录以日期为一级目录，具体版本或事项作为日期下的子条目；统一评估
使用对应日期入口，日期内按版本顺序展开。以后新增版本时，只在对应日期下新增
该版本子条目，不重命名已有日期或版本目录，也不创建含首尾版本范围的合并目录。


## F 阶段规则

1. 每次代码修改前，先判断改动范围：只改配置、模型结构、数据管线、训练逻辑、评估逻辑、demo，还是实验设计。
2. 若改动会影响行为、tensor shape、函数接口、命令或输出格式，必须先更新 `docs/implementation.md`。
3. 若改动涉及 Method 或实验设计，必须先更新 `docs/idea_report.md`。如果该文件尚不存在，则先创建或补充相关设计说明。
4. 每次代码修改后，都必须在本文件追加日志条目。
5. 本文件末尾固定保留 `运行说明` 章节；命令、参数、输出文件或输出路径变化时必须同步更新。
6. `configs/` **版本卫生（每次切到新版本分支必做）**：`configs/` 只保留**当前活动版本**的主配置 `<current_version>.yaml` 及其必要的 control/ablation 配置；删除其他版本遗留 yaml。同步把 `paths.py`、`common.py`、`scripts/`* 默认 `--config` 改为当前版本；在本文档追加清理记录。历史配置留在对应 Git 分支或本地 `outputs/`，不堆在活跃分支里。



## 项目架构

```text
image cue -> ImageSNNEncoder -> K_img -+
                                      +-> simultaneous recurrent Index A -> V_img/V_aud
audio cue -> AudioSNNEncoder -> K_aud -+                         |
                                                                +-> ClassifierHead

V_img + gated own image detail -> ImageDecoder feature
K_aud -> masked missing-region adapter -> image output
V_aud + gated own audio detail -> AudioDecoder feature
K_img -> masked missing-region adapter -> audio output
```



## 实现进度

状态说明：`已有` 表示代码已存在但本次文档补档未重新完整验证；`文档完成` 表示本轮已生成或中文化；`历史` 表示该条目只用于追溯；当前活动版本以“当前版本导航”为准。


| 模块         | 文件                                                            | 状态   | 时间               | 备注                                     |
| ---------- | ------------------------------------------------------------- | ---- | ---------------- | -------------------------------------- |
| 用户需求记录     | `docs/user_requirements.md`                                   | 文档完成 | 2026-07-06 18:58 | 已中文化，记录 D-F 硬规则                        |
| 实现指南       | `docs/implementation.md`                                      | 文档完成 | 2026-07-06 18:58 | 已中文化，基于当前代码整理                          |
| 开发日志       | `docs/dev_log.md`                                             | 文档完成 | 2026-07-06 18:58 | 当前文件                                   |
| 历史补档基线配置 | `configs/v10a.yaml`                                         | 历史   | 补档前              | 创建文档时的旧版基线，不代表当前配置            |
| 数据管线       | `data/*.py`                                                   | 已有   | 补档前              | MNIST + FSDD log-mel + medoids         |
| 残缺 cue 管线  | `data/corruption.py`, `common.py`                             | 已有   | 补档前              | 6 种 cue mode 和 corruption family       |
| SNN 基础模块   | `models/lif.py`                                               | 已有   | 补档前              | LIF + surrogate gradient               |
| 编码器        | `models/encoders.py`                                          | 已有   | 补档前              | image/audio SNN encoders               |
| 记忆模块       | `models/memory.py`                                            | 已有   | 补档前              | Key/Index/Value + binding/readout      |
| 解码器        | `models/decoders.py`                                          | 已有   | 补档前              | classifier、image decoder、audio decoder |
| 顶层模型       | `models/network.py`                                           | 已有   | 补档前              | `CrossModalSNN`                        |
| 训练脚本       | `scripts/train.py`                                            | 已有   | 补档前              | decoder pretrain + full training       |
| 评估脚本       | `scripts/evaluate.py`                                         | 已有   | 补档前              | fixed/random protocols                 |
| Demo 脚本    | `scripts/demo_inference.py`                                   | 已有   | 补档前              | demo figures 和 table                   |
| 消融 suite   | `scripts/make_v10a_ablations.py`, `scripts/run_v10a_suite.py` | 已有   | 补档前              | 三个 v10a 消融变体                           |
| 历史 v10a 输出 | `outputs/outputs_v10a/`                                       | 历史   | 2026-07-06 18:43 | 创建旧版文档时目录不存在                         |




## 开发日志条目




### 2026-07-06 18:43：基于现有代码补建 D/F 文档

**改动类型**：仅文档，不改代码。

**原因**：项目已经处于 F 阶段，但此前缺少 `docs/implementation.md` 和 `docs/dev_log.md`。用户要求后续代码修改必须先更新文档，并在每次代码修改后追加开发日志。

**完成内容**：

- 创建 `docs/user_requirements.md`，记录用户硬性规则。
- 创建 `docs/implementation.md`，基于当前代码生成实现指南。
- 创建 `docs/dev_log.md`，作为 F 阶段只追加日志基线。
- 记录当前 v10a 命令与输出约定。

**涉及文件**：

- `docs/user_requirements.md`
- `docs/implementation.md`
- `docs/dev_log.md`

**是否修改代码**：否。

**验证情况**：

- 已静态检查当前项目结构和源码文件。
- 该条记录未运行训练或评估命令。

**已知观察**：

- `outputs/outputs_v10a/` 尚不存在。
- 历史输出已存在到 v9/v9c。
- `docs/idea_report.md` 尚不存在。
- 当前 `requirements.txt` 包含 torch 系依赖，这与 ResearchPilot 默认“不在 requirements 中写 torch 系”的规则不同，但符合当前项目 README 和代码需求。



### 2026-07-06 18:49：尝试运行 smoke test，但因缺少 torch 被阻塞

**改动类型**：仅验证，不改代码。

**原因**：创建 D/F 文档后，尝试用随机张量 smoke test 检查当前代码路径能否执行。

**尝试命令**：

```bash
C:\Users\26835\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -u scripts/smoke_test.py
```

**结果**：在进入模型逻辑前被阻塞。

**错误信息**：

```text
ModuleNotFoundError: No module named 'torch'
```

**结论**：

- smoke test 未完成。
- Codex 桌面自带 Python runtime 当前没有安装 PyTorch。
- 需要使用项目预期的 Python/PyTorch 环境运行 `python -u scripts/smoke_test.py`。



### 2026-07-06 18:58：将三份 ResearchPilot 文档中文化

**改动类型**：仅文档，不改代码。

**原因**：用户要求将 `docs/user_requirements.md`、`docs/implementation.md`、`docs/dev_log.md` 改为中文。

**完成内容**：

- 将 `docs/user_requirements.md` 改为中文。
- 将 `docs/implementation.md` 改为中文，保留函数名、路径、命令、tensor shape 等英文技术标识。
- 将 `docs/dev_log.md` 改为中文，并保留此前两条日志事实。

**涉及文件**：

- `docs/user_requirements.md`
- `docs/implementation.md`
- `docs/dev_log.md`

**是否修改代码**：否。

**验证情况**：

- 待执行文档检查。



### 2026-07-06 20:57：v10b 文献对齐音频 time-mask 恢复迭代

**改动类型**：配置 + 数据管线 + 训练逻辑 + 评估逻辑。

**改动原因**：基于《Deep Long Audio Inpainting》的 long audio gap 设定，将 v10a 的多 family 音频 corruption 先收窄为论文对齐的连续时间片段缺失，并加入 mask-aware 评估和训练入口。

**改动内容**：

- `docs/implementation.md`：先补充 v10b 设计范围、mask API、masked audio loss、masked 指标、v10b 配置与输出路径。
- `configs/v10b.yaml`：从 v10a 派生，保留模型结构，改为 `time_mask` 训练/fixed 评估，并启用 `lambda_aud_masked: 1.0`。
- `data/corruption.py`：新增 `AUD_FAMILY_GROUPS`，`corrupt_audio(..., return_mask=True)` 同次返回真实音频缺失 mask；音频结构化 mask 在 `severity=0` 时不再强制遮挡 1 帧/频带。
- `common.py`：`build_cue(..., return_masks=True)` 向训练/评估透传 `{"aud": aud_mask}`，默认返回值保持向后兼容。
- `scripts/train.py`：新增 masked audio error helper；仅在 corrupt-audio、sample-level、白名单 family 且 mask 存在时追加 masked L1/MSE。
- `scripts/evaluate.py`：新增 `aud_ssim`、`aud_masked_mse/l1`、`aud_visible_mse/l1`，family breakdown 增加 `family_group`。

**预期效果**：

- v10b 主实验的音频缺失方式与 long time-gap inpainting 更一致。
- fixed-mask 评估能区分缺失区和可见区误差，避免只看全谱 `aud_mse`。
- masked loss 优先改善 `time_mask` 缺失区恢复，同时通过 `detach_value_for_recon=true` 降低对 Value/Index 的反向拉扯。

**文档同步**：`implementation.md` 是 | `dev_log.md` 是 | `configs/` 是。

**验证情况**：

- 通过语法编译：
`python -m py_compile data/corruption.py common.py scripts/train.py scripts/evaluate.py`
（使用绝对路径运行）。
- 通过配置关键项检查：`v10b.yaml` 包含 `output_version: v10b`、`aud_mode: "time_mask"`、`aud_train_modes: ["time_mask"]`、`lambda_aud_masked: 1.0`、v10b checkpoint 路径。
- `scripts/smoke_test.py` 未完成：当前 Codex bundled Python 缺少 `torch`，报错 `ModuleNotFoundError: No module named 'torch'`。



### 2026-07-07：v10c fixed-family 与 corrupt-aware decoder pretrain 迭代

**改动类型**：实验设计补充 + 配置 + 训练逻辑。

**改动原因**：v10b 完整 fixed-mask 评估显示分类与图像恢复可用，但音频恢复出现近黑图能量塌缩；同时训练时图像 family 仍为 `random`，与 fixed eval 的 `occlusion` 不一致。用户确认 v10c 采用图像后期 severity 0.4、decoder pretrain 25 轮、主训练 70 轮。

**改动内容**：

- `docs/idea_report.md`：新增 F 阶段 v10c 实验设计记录，说明 v10c 不改核心架构，只修正训练协议。
- `docs/implementation.md`：先补充 v10c 迭代范围、配置含义、训练函数接口、checkpoint 与输出路径。
- `configs/v10c.yaml`：从 v10b 派生，设定 `corruption.img_mode: "occlusion"`、`corruption.aud_mode: "time_mask"`、`train_severity: 0.4`、`severity_max: 0.4`、`decoder_pretrain.epochs: 25`、`train.epochs: 70`，并切换到 v10c checkpoint/output 路径。
- `scripts/train.py`：`_pretrain_decoder_states(...)` 增加可选 `x_img_detail` / `x_aud_detail`；`pretrain_decoders(...)` 支持 `decoder_pretrain.corrupt_detail=true` 时用 fixed corrupt cue 生成 detail state，并在音频预训练中复用 masked audio loss。

**预期效果**：

- 训练图像残缺 family 与 fixed eval 对齐，减少 random family 带来的训练/评估错位。
- 将后期 severity 从 0.5 降到 0.4，降低 1 秒 FSDD 长片段缺失的难度。
- 25 轮 corrupt-aware decoder pretrain 让 decoder 在主训练前见过 `occlusion` / `time_mask` detail 输入，预期缓解 v10b 的 recovered audio 近黑图塌缩。

**文档同步**：`idea_report.md` 是 | `implementation.md` 是 | `configs/` 是。

**验证情况**：

- 通过语法编译：
`C:\Users\26835\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m py_compile D:\Documents\Documents\Literature\SNN_Literature\Codes\cross_modal_attractor_snn\scripts\train.py`
- 通过 v10c 配置关键项文本检查：`output_version: v10c`、`train.epochs: 70`、`decoder_pretrain.epochs: 25`、`train_severity: 0.4`、`severity_max: 0.4`、`img_mode: "occlusion"`、`aud_mode: "time_mask"`、v10c checkpoint 路径均存在。
- 未运行完整 smoke test / 训练：当前 Codex bundled Python 缺少 `torch`；且本轮主要提供 v10c 代码与配置，完整训练应在项目 PyTorch/CUDA 环境执行。



### 2026-07-07：v10d 落地 F3a/F3b 结构改动与 F4 质量 loss 代码

**改动类型**：模型结构 + 训练逻辑 + 评估接线 + 配置。

**改动原因**：F 阶段计划中 v10c 已实现 F0/F1/F2（论文对齐 time_mask、masked 指标、masked audio loss、corrupt-aware pretrain），但尚未落地 F3a（gated/dilated decoder refinement）、F3b（谱图空间 refiner 支路）与 F4（附加质量 loss）。本轮把这些「未使用项」以 config 开关、默认关闭的方式加入代码，并新建 `v10d.yaml` 首次启用 refiner 相关结构；cue 比例仅做保守小幅调整，保证提升可归因于 refiner 而非采样比例。

**改动内容**：

- `docs/implementation.md`：先补充 §0.3 v10d 迭代范围、§3.5 `GatedConv2d` / `AudioDecoder.refine_type` / `AudioRefiner`、§3.6 `forward(..., aud_cue_mask=...)` 与 refiner 合成规则、§5.2 F4 loss、§7.5 v10d 配置表。
- `models/decoders.py`：新增 `GatedConv2d`；`AudioDecoder` 增加 `refine_type`（`plain` 默认 / `gated_dilated` 用门控卷积 + dilation(1,2,4)）；新增 `AudioRefiner`（输入 `[coarse, cue, mask]`，输出 tanh 缩放 delta，`out_proj` 零初始化）。
- `models/network.py`：按 `snn.aud_refine_type` 构造 decoder；按 `audio_refiner.enabled` 构造 `self.audio_refiner`；`forward` 增加 `aud_cue_mask`，仅当 refiner 启用、存在音频 cue 且 mask 非空时执行 `recovered_aud = clamp(coarse + mask*delta, 0, 1)`，coarse 结果保留在 `recovered_aud_coarse`；`infer` 透传 mask。
- `scripts/train.py`：`_aud_recon_loss` 追加 `lambda_aud_ssim`（`1-SSIM`）与 `lambda_aud_masked_grad`（缺失区时频梯度，新增 `_masked_tf_grad_loss`）；新增 `_aud_feature_loss`（冻结 aud_encoder 特征 L1，target 侧 detach）由 `lambda_aud_feat` 触发；readout 前向透传 `aud_cue_mask=aud_mask`。
- `scripts/evaluate.py`：`eval_mode` 前向透传 `aud_cue_mask=aud_mask`，使评估与训练一致使用 refiner。
- `configs/v10d.yaml`：从 v10c 派生；`snn.aud_refine_type: gated_dilated`、`audio_refiner.enabled: true`；F4 三项权重设 0（就绪、默认关闭）；cue 比例保守调整为 corrupt_img_only 0.15 / corrupt_aud_only 0.20 / corrupt_both 0.25 / clean_img_only 0.10 / clean_aud_only 0.15 / clean_both 0.15（合计 1.0）；切换 v10d checkpoint/输出路径。

**预期效果**：

- gated/dilated refine 扩大 decoder 缺失区感受野；refiner 在缺失区叠加修正 delta，仅作用于 mask 区、可见区逐位不变。
- F4 代码就绪但默认关闭，作为后续独立消融入口，避免与 refiner 提升混淆归因。
- 所有恢复侧改动走 `detach_value_for_recon=true`，不回传 Value/Index，沿用 v9b 分类/恢复不互拖机制。

**兼容性**：`aud_refine_type: plain` + `audio_refiner.enabled: false` 时结构退回 v10c，可加载旧 checkpoint；v10d 默认结构与旧 checkpoint 不兼容，需从头训练。旧 `forward` 调用（不传 `aud_cue_mask`）行为不变。

**文档同步**：`implementation.md` 是 | `dev_log.md` 是 | `configs/` 是 | `idea_report.md` 否（v10d 首版遗漏，已在同日复订补写）。

**验证情况**：

- 向后兼容：`python -u scripts/smoke_test.py`（v10a，plain decoder、refiner 关闭）6 种 cue 模式前向/反向通过。
- v10d 结构：临时脚本加载 `configs/v10d.yaml`（CPU，F3a+F3b 开启 + F4 三项临时置正）跑 6 种 cue 模式 `compute_losses` 前向/反向均通过；refiner 可见区 `max|rec-coarse|=0.00e+00`、缺失区 delta 生效（`7.79e-02`）、`mask=None` 完全旁路（`0.00e+00`）。临时脚本验证后删除。

**demo 说明（已过时，见 2026-07-08 P0-A 条目）**：v10d 首版 demo 曾旁路 refiner；P0-A 已修复。

### 2026-07-07（修订）：v10d 对齐文献 inpainting + 归因消融配置

**改动类型**：模型逻辑 + 训练逻辑 + 配置（对 review 反馈的修订）。

**改动原因**：v10d 首版收到 review，指出与文献式 inpainting 的偏差：①可见区未真正 paste-back 到 aud_cue；②refiner 时间感受野偏小（`blocks:2` 只用到 dilation 1,2）；③`lambda_aud_feat` 未严格冻结 encoder；④refiner 未参与 pretrain（属实需明确）；⑤同时改结构与 cue 比例导致归因混淆。本轮修订 ①②③⑤，并把 ④ 显式写入文档。

**改动内容**：

- `models/network.py`：新增 `audio_refiner.visible_paste_back` 开关。为 true 时 `pred = clamp(coarse + delta)`，`recovered_aud = mask * pred + (1 - mask) * aud_cue`（文献式：缺失区预测 + 可见区原样保留 cue）；为 false 时保持原 `coarse + mask * delta`。两者可见区都不被 refiner 改写。
- `models/decoders.py`：`AudioRefiner` dilation 改为 `min(2**block_index, max_dilation)` 指数增长（新增 `max_dilation` 参数），`blocks=4` 时 dilation 1,2,4,8、时间感受野约 31 帧，覆盖 severity=0.4 的约 26 帧缺失。
- `scripts/train.py`：`_aud_feature_loss` 提取 rec 特征时临时关闭 `aud_encoder` 参数 `requires_grad`（target 侧 `no_grad`），实现真正「冻结 encoder feature loss」，梯度只回传 decoder/refiner。默认权重仍 0。
- `configs/v10d.yaml`：`audio_refiner.blocks: 4`、`max_dilation: 16`、`visible_paste_back: true`。
- 新增 `configs/v10d_ablation_v10c_ratio.yaml`（v10d 结构 + v10c cue 比例，隔离 cue 比例）与 `configs/v10d_ablation_refiner_off.yaml`（v10d 比例与结构 + refiner 关闭，隔离 refiner 贡献）。
- `docs/implementation.md`：更新 §3.5（AudioRefiner 感受野）、§3.6（paste-back 合成规则）、§7.5（新 refiner key + 消融配置表 + refiner 不入 pretrain 说明 + aud_feat 冻结说明）。
- `docs/idea_report.md`：补充 §「F 阶段补充：v10d 实验设计」（目标、设置对比、归因消融、预期指标、风险边界、评估协议）；文件头增加「每次 F 阶段迭代须同步更新本文件」纪律。

**预期效果**：

- paste-back 让 v10d 真正符合「缺失区补全 + 可见区保留」的 inpainting 形式，decoder/refiner 精力集中在缺失区。
- 更深 refiner 覆盖长缺失区上下文。
- 消融配置支持把「结构增益」与「cue 比例增益」分开归因。

**关于 ④（refiner 不入 pretrain）**：保持现状（主训练从零学 refiner），已在 §7.5 显式说明，非缺陷。

**文档同步**：`implementation.md` 是 | `dev_log.md` 是 | `configs/` 是 | `idea_report.md` 是。

**验证情况**：

- `python -u scripts/smoke_test.py`（v10a，refiner 关闭）通过，向后兼容。
- 临时脚本（CPU）加载 `configs/v10d.yaml`（paste_back + blocks=4）跑 6 种 cue 模式前向/反向通过；断言可见区 `recovered_aud == aud_cue`（paste-back 生效）、`mask=None` 完全旁路、F4（含冻结 aud_feat）开启时 `aud_encoder` 无梯度累积。验证后删除临时脚本。



### 2026-07-08：v10d 消融口径与训练 visible 指标小修

**改动类型**：训练日志与消融配置说明小修，不改变模型结构、loss 权重或评估协议。

**改动原因**：`v10d_ablation_refiner_off.yaml` 中 `audio_refiner.enabled=false` 会使 `models/network.py` 的 refiner 分支整体旁路，而 `visible_paste_back` 也写在该分支内；因此该消融实际关闭的是 refiner + paste-back 后处理路径，不应表述为只关 refiner。同时主训练日志已记录 `aud_mask_l1/mse`，但缺少与 evaluate 对齐的可见区 sanity 指标。

**改动内容**：

- `docs/implementation.md`：修正 F3b 合成公式与 `v10d_ablation_refiner_off.yaml` 的归因口径。
- `docs/idea_report.md`：修正消融表述为“refiner/paste-back 后处理路径的合并贡献”。
- `docs/dev_log.md`：更新 v10d demo 运行说明，去掉“refiner 旁路”的过时描述。
- `configs/v10d_ablation_refiner_off.yaml`：注释改为关闭 refiner + paste-back 后处理路径。
- `scripts/train.py`：在有 `aud_mask` 的训练/decoder pretrain 日志中追加 `aud_visible_l1` 与 `aud_visible_mse`，用于确认 visible-region paste-back 是否保持可见区一致。

**预期效果**：不改变训练数值目标；训练 stdout 中可直接看到 missing/visible 两类区域误差，消融归因描述更准确。

**验证情况**：

- `python -m py_compile scripts/train.py` 通过。
- `git diff --check -- scripts/train.py configs/v10d_ablation_refiner_off.yaml` 通过（仅提示 Windows 工作区 CRLF 替换警告）。
- 本次提交只应包含 `scripts/train.py` 与 `configs/v10d_ablation_refiner_off.yaml`；`docs/`、`.gitignore` 与本地 `scripts/plot_eval_summary.py` 不进入 Git 提交。



### 2026-07-08：v10e 图像 inpainting 对称化设计

**改动类型**：模型结构 + 数据 mask + 训练/评估/demo/绘图脚本同步。

**改动原因**：v10d 只在音频分支启用 decoder 后 `AudioRefiner + visible paste-back`。为了让图像和音频遵循同一条 inpainting 规则，v10e 需要在图像有残缺 cue 时启用补洞逻辑；缺失模态和 clean cue 模态不启用补洞。

**设计内容**：

- `data/corruption.py`：`corrupt_image(..., return_mask=True)` 返回 `img_mask`（`1=missing`）。
- `common.py`：`build_cue(..., return_masks=True)` 同时返回 `masks["img"]` 与 `masks["aud"]`。
- `models/decoders.py`：新增 `ImageRefiner`。
- `models/network.py`：新增 `img_cue_mask`；图像 decoder 输出 `recovered_img_coarse`，满足 `image_refiner.enabled and x_img_cue and img_cue_mask` 时执行 `ImageRefiner + visible paste-back`。
- `scripts/train.py`：图像 masked/visible loss 日志与音频对齐。
- `scripts/evaluate.py`：主表新增 `imgMaskMSE` 和 `audMaskMSE`。
- `scripts/demo_inference.py`：图像 demo 展示 `image mask` 与 `coarse image`。
- `scripts/plot_eval_summary.py`：正式纳入仓库，支持 v10e 新增评估列。
- `configs/v10e.yaml`：从 v10d 派生，启用 `image_refiner`，版本输出切到 v10e。

**预期效果**：`corrupt_img_only` 与 `corrupt_both` 的图像可见区保持输入 cue，缺失区由模型补洞；`clean_`* 和缺失模态路径保持普通重建/联想，不做 paste-back。

**验证情况**：

- `python -m py_compile` 使用 Codex bundled Python 对 `common.py`、`data/corruption.py`、`models/decoders.py`、`models/network.py`、`scripts/train.py`、`scripts/evaluate.py`、`scripts/demo_inference.py`、`scripts/plot_eval_summary.py` 通过。
- `git diff --check` 对 v10e 代码/配置范围通过（仅提示 Windows 工作区 CRLF 替换警告）。
- 本地 bundled Python 未安装 `torch` / `yaml` / `matplotlib`，因此本机未跑模型前向 smoke 与绘图渲染；训练环境有 torch 后建议先跑 `scripts/evaluate.py --config configs/v10e.yaml --protocol fixed_mask --severity 0.4 --max_batches 1`。



### 2026-07-08：P0-A demo 与 evaluate 路径对齐（v10d 分支）

**改动类型**：demo 脚本 only（不改训练 / 主模型前向训练路径）。

**改动原因**：v10e 计划 P0-A；v10d demo 未传 `aud_cue_mask`，refiner + paste-back 被旁路，可视化与 `evaluate.py` 不一致。

**改动内容**：

- `docs/implementation.md` §6.2：补 demo 与 evaluate 对齐约定（mask 获取、前向透传、指标口径、可视化列）。
- `docs/idea_report.md`：更新 demo 展示口径说明。
- `scripts/demo_inference.py`：
  - `_make_visible_aud_cue_with_mask`：`fixed_mask` 下同次 `corrupt_audio(..., return_mask=True)` 保留 mask。
  - `legacy_random`：`build_cue(..., return_masks=True)` 取 `aud_mask`。
  - `audio-only` / `both` 前向传 `aud_cue_mask`；指标仍基于最终 `recovered_aud`。
  - 可视化新增 `audio mask`、`coarse audio` 列（在 recovered audio 之前）。

**预期效果**：demo 图与 fixed_mask 评估同走 refiner + paste-back；可肉眼对比 coarse 与 final。

**文档同步**：`implementation.md` 是 | `idea_report.md` 是 | `dev_log.md` 是。

**验证情况**：

- `python -m py_compile scripts/demo_inference.py` 通过。
- `python -u scripts/demo_inference.py --config configs/v10d.yaml --num 8 --severity 0.4`（CPU）成功，输出三张 demo 图与 `demo_eval_table.txt`；audio-only audSSIM≈0.906（final 路径）。



### 2026-07-09：v10f 归因补齐实现

**改动类型**：模型 helper + 训练 pretrain/lr_mult + 评估 coarse/final 指标 + 配置消融。

**改动原因**：v10e 已有对称 refiner，但难以拆分 paste-back、delta refiner、pretrain 与 lr_mult 的贡献；需在同一 checkpoint 上对比 coarse→final masked MSE，并支持 pasteback_only/off 消融。

**修改文件（Git 提交范围）**：

- `models/network.py`：`_apply_image_refiner` / `_apply_audio_refiner` + `pasteback_only`
- `scripts/train.py`：refiner-aware pretrain、`lambda_coarse_aux`、`lr_mult` param groups
- `scripts/evaluate.py`：coarse/final masked 指标 + `[归因]` 表（注明 final_visible≈0 是机制而非学习）
- `scripts/plot_eval_summary.py`：解析/渲染归因表
- `configs/v10f.yaml` + 5 个 `v10f_ab_*` 消融配置

**文档同步（本地，不入 Git）**：`docs/implementation.md` §0.5、`docs/idea_report.md` v10f 实验设计、`docs/dev_log.md` 本条。

**预期效果**：主配置 pretrain 走 final 路径并可训 refiner；评估主看 `coarse_mask_mse → final_mask_mse`；`pasteback_only` 与 full refiner 的 masked 差距量化 delta 贡献。

**验证情况**：`python -m py_compile` 对 `models/network.py`、`scripts/train.py`、`scripts/evaluate.py`、`scripts/plot_eval_summary.py` 通过；已提交并推送 `origin/v10f`（commit `ce6018f`）。

### 2026-07-09：v10f 5-family 鲁棒性协议与文档位置修正

**改动原因**：v10f 原本仍以单一 `occlusion/time_mask` 为主协议，demo 中还可能出现无 mask 的音频样本，不符合“主评估和可视化评估都覆盖 5 种残缺 family、每种 2 个样本”的鲁棒性目标。同时修正此前把“运行说明补充”直接写到文档最后的问题：开发日志条目只放在本章，命令说明只合并到 `## 运行说明` 对应小节。

**改动内容**：

- `docs/implementation.md`：补充 v10f multi-family corruption 设计，说明 image/audio 各 5 种带 mask 的残缺方式，以及 Gaussian/noise 不纳入当前主协议的原因。
- `docs/idea_report.md`：同步 v10f 鲁棒性实验口径，明确训练、fixed-mask 主评估、demo 都使用 5-family。
- `data/corruption.py`：新增 `mask_vertical`、`mask_horizontal`、`block_corner` 图像残缺；保留 `occlusion` 和 `pixel_delete` 组成图像 5-family。
- `common.py`：训练 cue 构造支持 image/audio family balanced sampling。
- `scripts/train.py`：decoder pretrain 的 corrupt detail 支持 image/audio family 列表采样。
- `scripts/evaluate.py`：fixed-mask 主评估支持按配置的 5 个 image/audio family pair 顺序评估，并保留 family breakdown 与归因表。
- `scripts/demo_inference.py`：fixed-mask demo 默认 `--num 10`，按 5-family 扩展为每种 2 个样本；不再把未找到合适 mask 的音频退回 clean。
- `configs/v10f*.yaml`：主配置和 5 个消融配置全部补齐 `img_modes`、`aud_modes`、`img_train_modes`、`aud_train_modes`、masked family 白名单和 fixed eval family 列表。
- `docs/dev_log.md`：移除末尾错误新增的“运行说明补充”章节，后续运行命令只维护在 `### v10f 运行命令` 内。

**预期效果**：

- v10f 主训练、主评估和可视化都不再只围绕单一 `occlusion/time_mask`，而是覆盖图像和音频各 5 种带 mask 的缺失形态。
- demo 图每次默认 10 行，可按 family label 检查每类残缺，避免出现“音频完全没处理”的可视化样本。
- 论文口径继续以 maskedMSE、coarse→final masked 改变量和 ACC 为核心；全谱指标仍需注明会被 paste-back 抬高。

**验证**：

```bash
python -m py_compile common.py data/corruption.py scripts/train.py scripts/evaluate.py scripts/demo_inference.py scripts/plot_eval_summary.py
git diff --check -- common.py data/corruption.py scripts/train.py scripts/evaluate.py scripts/demo_inference.py scripts/plot_eval_summary.py configs/v10f*.yaml
```

`py_compile` 已通过；`git diff --check` 通过，仅提示 Windows 工作区 LF/CRLF 替换警告。配置关键项检查确认 6 个 `configs/v10f*.yaml` 均包含新的 image/audio family 列表。

**文档同步**：idea_report.md 是 | implementation.md 是 | dev_log.md 是 | configs/ 是

### 2026-07-09 20:21 — 迭代：v10f P1/P2 归因口径修正

**改动原因**：v10f code review 发现两个归因风险：`v10f_ab_no_refiner_pretrain` 虽然关闭 `train_*_refiner`，但 decoder pretrain 仍会经过 frozen refiner helper；`evaluate.py` 已计算 visible coarse/final 指标，但 `[归因]` 表和 `plot_eval_summary.py` 只输出 masked 指标。

**改动内容**：

- `docs/implementation.md`：补充 v10f pretrain helper 的开关语义；明确 `train_*_refiner=false` 时 pretrain 直接使用 coarse decoder 输出，`pasteback_only=true` 才单独走 paste-back helper。
- `docs/idea_report.md`：更新 v10f 实验设计口径，加入 coarse/final visible 指标，并澄清 `no_refiner_pretrain` 的 pretrain 目标。
- `scripts/train.py`：新增 `use_img_final_helper/use_aud_final_helper`，使 refiner-aware pretrain 只在对应开关启用时进入 helper；训练日志输出 `img_final_path/aud_final_path`。
- `scripts/evaluate.py`：`[归因]` 表从 masked-only 扩展为 masked + visible 两组 coarse/final MSE。
- `scripts/plot_eval_summary.py`：归因表解析与渲染支持 masked + visible 八列，并兼容旧的四列 masked-only 日志。

**预期效果**：

- `v10f_ab_no_refiner_pretrain` 可干净隔离 refiner-aware pretrain 贡献，不再混入 frozen refiner 前向。
- 评估日志和归因图可直接检查 `final_visible_mse≈0` 是否来自 paste-back，同时仍以 `masked_mse` 作为真实补洞结论。

**验证**：

```bash
python -m py_compile scripts/train.py scripts/evaluate.py scripts/plot_eval_summary.py
```

**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-07-10：v10f 分支清理遗留 configs

**改动原因**：`v10f` 分支 `configs/` 仍保留 v10d（含 2 个消融）和 v10e 共 4 个旧版 yaml，与「当前分支只服务当前版本」冲突，易造成误用旧配置训练/评估。

**删除文件**：

- `configs/v10d.yaml`
- `configs/v10d_ablation_refiner_off.yaml`
- `configs/v10d_ablation_v10c_ratio.yaml`
- `configs/v10e.yaml`

**保留文件（v10f 共 6 个）**：

- `configs/v10f.yaml`
- `configs/v10f_ab_audio_pasteback_off.yaml`
- `configs/v10f_ab_audio_pasteback_only.yaml`
- `configs/v10f_ab_image_pasteback_off.yaml`
- `configs/v10f_ab_image_pasteback_only.yaml`
- `configs/v10f_ab_no_refiner_pretrain.yaml`

**同步默认配置**：`paths.py`、`common.py`、`scripts/train.py`、`scripts/evaluate.py`、`scripts/demo_inference.py`、`scripts/mkdir_outputs.py`、`scripts/smoke_test.py` 默认改为 `configs/v10f.yaml`。

**规则**：已在「F 阶段规则」第 6 条写入——每次新版本/新分支，必须删旧版 yaml 并改默认 config，避免重复堆积。

**文档同步**：`dev_log.md` 是 | `README.md` 是（训练/评估命令改 v10f）

### 2026-07-10：demo 评估表增加 mask MSE + v10f 日志出表

**改动原因**：demo 评估表仅有全谱 MSE/SSIM，缺少与 evaluate 一致的 mask 区域 MSE；v10f 已有 eval/demo 日志需一键生成 PNG+CSV 表格。

**代码**：

- `scripts/demo_inference.py`：`_mode_metrics` 增加 `img_masked_mse` / `aud_masked_mse`（与 evaluate `_region_error` 同口径）；`【汇总】` 表新增两列。
- `scripts/plot_eval_summary.py`：demo 表解析/渲染支持 mask MSE；全量 eval / 归因日志对 5-family 结果取平均；默认输入改 `outputs_v10f`。

**已生成表格（本地 outputs）**：

- `outputs/outputs_v10f/tables/full_eval_table.png` + `.csv`（来自 `eval_v10f_fixed_mask_sev04.log`，5-family 平均）
- `outputs/outputs_v10f/tables/full_eval_table_attribution.png` + `.csv`
- `outputs/outputs_v10f/tables/full_eval_table_aud_diag.png` + `.csv`
- `outputs/outputs_v10f/figures/demo_eval_summary_table.png` + `.csv`（来自 `demo_v10f_sev04.log`；旧日志无 mask 列，需重跑 demo 后才有）

**重跑 demo（带 mask MSE）**：

```bash
python -u scripts/demo_inference.py --config configs/v10f.yaml --num 10 --severity 0.4 --protocol fixed_mask
python scripts/plot_eval_summary.py outputs/outputs_v10f/tables/demo_eval_table.txt
```



### 2026-07-10：全量 eval 表改为按 family 分表（取消平均）

**改动原因**：5-family fixed_mask 评估应逐 family 出主表，跨 family 平均会掩盖单 family 差异。

**代码**：`scripts/plot_eval_summary.py` 按 `[评估 family i/n]` 切分日志；每个 family 独立生成主表 + 归因 + aud_diag（PNG+CSV）。输出命名：`full_eval_table_family01_occlusion_time_mask.png` 等。

**重跑命令**：

```bash
python scripts/plot_eval_summary.py outputs/outputs_v10f/logs/eval_v10f_fixed_mask_sev04.log --out outputs/outputs_v10f/tables/full_eval_table.png --title "v10f Full Eval (fixed_mask, sev=0.4)"
```



### 2026-07-10：tables 按 family 子目录输出

**改动原因**：`tables/` 根目录堆满 `full_eval_table_familyXX_...` 长文件名，不便浏览。

**目录结构**：

```text
tables/
  demo_eval_table.txt
  audio_family_breakdown_fixed.csv
  family01_occlusion_time_mask/
    full_eval.png + .csv
    attribution.png + .csv
    aud_diag.png + .csv
  family02_.../
  ...
```

**代码**：`plot_eval_summary.py` 用 `_family_dir` / `_family_artifact_path` 写入子目录；README 已补充生成说明。

**文档同步**：`dev_log.md` 是 | `README.md` 是

### 2026-07-10：v11a 非对称 cue 与 time-mask 专项训练

**改动原因**：v10f 中 `corrupt_both` 相对 `corrupt_aud_only` 的 ACC 明显提高，但 audio maskedMSE 基本不变；原比较同时改变了第二模态是否存在、第二模态是否残缺和随机 mask，不能单独归因。v10f 五个音频 family 平衡采样还降低了 `time_mask` 的专项训练占比，`partial_temporal` 固定遮挡张量尾部时可能只命中 FSDD 静音 padding。

**改动内容**：

- Git：将 v10f 提交快进至 `main`，从同一提交创建 `v11a` 分支；v11a 的 `configs/` 只保留 `configs/v11a.yaml`。
- `common.py`：cue mode 从 6 种扩展为 8 种，新增 `clean_img_corrupt_aud` 与 `corrupt_img_clean_aud`；新增最后阶段音频 family 加权采样。
- `data/corruption.py`：`partial_temporal` 改为 active-aware trailing mask，窗口结束位置对齐每条样本的最后有效语音帧，全静音时才回退固定尾部。
- `scripts/train.py`：两种非对称 cue 纳入对应 masked loss；新增 `lambda_aud_masked_weighted`，按 `1 + gamma*target` 提高缺失区高能语音位置权重，并记录 `aud_mask_wmse`。
- `scripts/evaluate.py`：主评估覆盖 8 种 cue；only 与新增 clean-assist 模式复用对应 corruption seed；音频 family breakdown 增加 `clean_img_corrupt_aud`。
- `scripts/plot_eval_summary.py`：主表、归因表解析支持两个新模式。
- `configs/v11a.yaml`：120 轮主训练；第 95 轮起 25 轮 time-mask-heavy 阶段（`time_mask=0.60`）；新增 cue 概率、weighted masked loss 与 v11a checkpoint/output 路径。
- `README.md`、默认 config 入口和 smoke test：统一切换至 v11a。

**本轮边界**：未修改 Key/Index/Value、decoder/refiner 结构；图像条件如何进入音频恢复、音频条件如何进入图像恢复留待非对称 cue 基线结果后决定。

**验证情况**：

- Python 语法检查通过：`common.py`、`data/corruption.py`、`scripts/train.py`、`scripts/evaluate.py`、`scripts/demo_inference.py`、`scripts/mkdir_outputs.py`、`scripts/plot_eval_summary.py`、`scripts/smoke_test.py`、`paths.py`。
- `git diff --check` 通过，仅有 Windows LF/CRLF 提示。
- 配置检查通过：cue 概率和为 1.0；`output_version=v11a`、120 轮、fine-tune 起点 95、`time_mask=0.60`、新 cue、新 weighted loss 与 v11a checkpoint 均存在。
- `scripts/mkdir_outputs.py --config configs/v11a.yaml` 成功创建 `outputs/outputs_v11a/`。
- 当前 Codex bundled Python 无 `torch`，未在本机执行模型前向 smoke；需在 CUDA/PyTorch 环境运行快速评估。

**文档同步**：`implementation.md` 是 | `idea_report.md` 是 | `dev_log.md` 是 | `README.md` 是。

### 2026-07-12：v11a 对侧 Key 条件化 Decoder 与配对归因

**改动原因**：v11a 非对称 cue 已能公平比较“增加 clean 对侧模态”前后的恢复，但原结构中对侧模态只经共享 Index/Value 间接影响目标 Decoder，无法判断其 Key 语义是否真正改善缺失区。为避免把 Index 再次送入 Refiner造成冗余，本轮按正式计划把 detached 对侧 Key 作为 Decoder 前的门控 Value residual，并用 correct/zero/wrong 配对干预验证因果贡献。

**结构与训练**：

- `models/network.py`：新增 `K_img -> AudioDecoder`、`K_aud -> ImageDecoder` 两个方向的 projector + 标量 gate；先对原 `V_*_from_A` detach，再加 cross residual，最后拼接同模态 cue detail。projector 零初始化，缺失 Key、control 或 zero 干预严格返回零 residual；Refiner 仍只接收 coarse、同模态 cue 和同模态 mask。
- `forward/infer`：新增两个 Key-rate override 与两个方向化 disable 参数；override 仅改变 Decoder 条件副本，不改原始 Key、共享 Index 或 Value。
- `scripts/train.py`：Cross-Key 参数使用真实模块名前缀建立独立 `lr_mult` optimizer group；decoder pretrain 统一复用 fusion helper，但固定 `cross_key_rate=None` 且禁止 `train_cross_key_conditioning=true`。
- `configs/v11a.yaml`：启用 Cross-Key，`detach_key=true`、`lr_mult=1.0`。
- `configs/v11a_control.yaml`：前向 Cross-Key 关闭，但通过 `build_modules=true` 保留同构参数和相同模块构造顺序；使用独立 pretrain/main checkpoint 与 output_version。重置相同 seed 后，主实验与 control 的 120 个 state tensor 逐项一致，参数量均为 22,759,946。

**评估与汇总**：

- `scripts/evaluate.py`：新增 `--cross_key normal|zero|shuffle_wrong|sweep`；sweep 在同一 cue/mask 下配对执行 normal/zero/wrong-class Key，逐 batch 检查 `index_state` 与 ACC 路径不变，并输出方向化 coarse/final `correct_gain=zero-normal`、`wrong_damage=wrong-normal`、gate mean 和 residual/value norm ratio。checkpoint 缺失或结构不匹配时直接失败，禁止随机权重伪评估。
- `scripts/plot_eval_summary.py`：按 corruption family 解析 `[Cross-Key归因]`，生成 `cross_key_attribution.png/.csv`。
- `scripts/smoke_test.py`：增加零初始化等价、zero 旁路、首步 projector/次步 gate 梯度、合成 wrong Key 仅改变 Decoder 与 Index 隔离检查。
- `README.md`：补充完整三路 Decoder 输入、control、sweep 与严格 checkpoint 说明。

**验证情况**：

- Python 语法检查、`git diff --check` 通过；`docs/` 与 `.gitignore` 均未被 Git 跟踪。
- 随机张量 smoke 覆盖 8 种 cue mode，前向/反向通过；独立 Cross-Key 测试模型的首步 projector 梯度 `45.220241`、gate 梯度 `0`，更新后第二步 gate 梯度 `32.874781`，未污染后续 8 模式 smoke 模型。
- decoder pretrain 检查通过：Cross-Key residual 为零，Cross-Key 参数无梯度。
- wrong-class permutation 检查通过；优先构造一对一完整异类置换，多数类超过半批时使用最大有效子集；所有有效样本均满足 `labels[wrong_perm] != labels` 且不复用 Key index。
- 使用本地 `torch 1.13.1+cpu`、临时 `T=2` toy 配置完成 `max_batches=1 --cross_key sweep`；日志正确输出 N/A/零初始化归因，plot 成功生成主表、音频诊断、coarse/final 归因与 Cross-Key 归因 PNG/CSV。正式结果仍须使用 CUDA、FSDD 和训练后的 v11a checkpoint。

**文档同步**：`implementation.md` 是 | `idea_report.md` 是 | `dev_log.md` 是 | `README.md` 是 | 正式 Cursor plan 是。

### 2026-07-20：退役 v11a YAML 并统一 v11b 默认入口

**改动原因**：`configs/v11a.yaml` 与 `configs/v11a_control.yaml` 已被 v11b 五实验
配置族替代。继续保留会造成默认入口、实验预算和损失口径混用。

**改动内容**：

- 删除 tracked 的 `configs/v11a.yaml` 与 `configs/v11a_control.yaml`；历史配置仍可
  从 v11a Git 提交恢复。
- `paths.py::DEFAULT_CONFIG` 改为 `configs/v11b_recovery.yaml`。
- `scripts/train.py`、`scripts/evaluate.py`、`scripts/plot_eval_summary.py` 中仍作为
  当前说明出现的 v11a 文案统一改为 v11b。
- `docs/implementation.md` 先行增加配置退役规则，并移除当前实现对 v11a YAML 的
  默认引用。
- `运行说明` 增加五组 v11b 实验的单后台、失败即停止顺序命令。

**验证情况**：Python 语法检查通过；`paths.DEFAULT_CONFIG` 指向存在的
`configs/v11b_recovery.yaml`；tracked 文件中已无 `v11a` 字符串；
`git diff --check` 通过，仅保留 Windows LF/CRLF 提示。

**文档同步**：`implementation.md` 是 | `idea_report.md` 否（未改变模型或实验设计） |
`dev_log.md` 是 | `README.md` 否（现有 v11b 说明无需改变）。

### 2026-07-20：v11b 音频恢复稳定化与 Cross-Key 因果归因

**问题来源**：v11a 主实验与同预算 control 的音频恢复都发生塌缩，二者的
audio family 重建指标几乎一致；已经学到非零 Cross-Key gate/residual，但
correct、zero、wrong-class Key 对洞内误差几乎没有影响。因此 v11b 不把扩大
Cross-Key 模块当作第一步，而是先恢复 coarse audio 的直接监督，再用同父
checkpoint 的三路微调区分普通条件化与显式因果约束。

**文档先行**：代码修改前已更新 `docs/implementation.md`；模型训练路径、五实验
设计、同类异样本 Key 的解释边界与验收口径同步写入 `docs/idea_report.md`。

**训练与损失**：

- `scripts/train.py` 修正 energy-weighted masked MSE 的归一化分母，由
  `sum(mask)` 改为 `sum(weight*mask)`，使 `gamma` 只改变区域相对权重，不再隐式
  放大损失总尺度。
- 主训练新增 `lambda_aud_coarse`，直接监督 paste-back/refiner 之前的 coarse
  audio；训练日志新增 `aud_coarse`、`aud_coarse_mse` 与
  `aud_coarse_mask_mse`。
- 新增 Cross-Key causal pair loss：在相同 cue/mask 下比较 correct、zero 与
  wrong-class 对侧 Key；zero/wrong 只作为 detached reference，梯度仅要求
  correct Key 的 coarse 洞内 MSE 优于参考。
- 新增严格父 checkpoint、完整 optimizer/scheduler 续训、指定可训练参数前缀、
  epoch milestone 保存。结构或 epoch 不匹配时立即失败，避免静默从错误状态启动。

**五组实验配置**：

1. `v11b_recovery`：Cross-Key 关闭、weighted loss 关闭，训练 120 轮；保存包含
   model/optimizer/scheduler/epoch 的第 100 轮 milestone。
2. `v11b_weighted`：从 recovery 第 100 轮完整状态续训到 120 轮，仅开启归一化
   weighted masked MSE，用于判断高能区域加权是否优于普通 coarse 恢复。
3. `v11b_control`：从选定的同一个第 120 轮父模型做 30 轮低学习率微调，
   Cross-Key 前向关闭但保留同构模块。
4. `v11b_cross_no_causal`：与 control 同父模型、同预算，开启 Cross-Key，不加
   causal pair loss。
5. `v11b`：与前两路同父模型、同预算，开启 Cross-Key 与 causal pair loss。

默认后三路都指向 `cross_modal_snn_v11b_recovery.pt`。如果 recovery 与 weighted
的预注册比较最终选择 weighted，必须同步修改三份配置的 `init_ckpt_path`，不能
混用不同父模型。

**评估与可视化**：

- `scripts/evaluate.py` 的 Cross-Key sweep 扩展为
  correct/zero/wrong-class/same-class-different-sample 四路配对；每批检查干预不改变
  Index 与 ACC 路径。same-class 对照用于区分“类别语义”与“实例细节”，不把
  MNIST/FSDD 类内样本误述为同一实例。
- `scripts/plot_eval_summary.py` 支持新的 `Cwrong/Fwrong/Csame/Fsame` 列，并兼容
  v11a 旧 8 列归因日志。
- `scripts/smoke_test.py` 增加 weighted loss 尺度不变、coarse loss 存在和因果
  loss 前向/反向检查；README 与默认入口同步到 v11b。

**验证情况**：

- Python 语法检查通过；五份 YAML 均可解析，cue 概率均为 1，checkpoint 和
  output_version 互不冲突。
- 随机张量完整 smoke 通过：8 种 cue 的前向/反向、Cross-Key 梯度、coarse loss
  与 causal loss 均有覆盖。
- 第 100 轮状态装载单元检查通过：`epoch=99` 正确映射到 `start_epoch=100`，
  optimizer 参数组和 scheduler 状态可恢复。
- 三路最终微调的 `trainable_prefixes` 检查通过，共匹配 34 个 Decoder/Cross-Key
  参数张量，其他参数保持冻结。
- same-class 置换与新旧 Cross-Key 日志解析检查通过；v11a 旧 sweep 日志可由新版
  `plot_eval_summary.py` 完整渲染。
- `docs/` 与 `.gitignore` 保持不跟踪、不提交；`models/network.py` 工作区内容哈希
  与 HEAD 完全相同，不纳入本轮提交。

**结论状态**：本条只记录实现与验证，不预写实验结论。五组训练完成后，每一组
必须在 `dev_log.md` 写独立结论，至少记录训练轮数、父 checkpoint、maskedMSE、
coarse-to-final 变化、ACC、Cross-Key 配对归因及可解释边界，避免后续遗忘或把
不同预算实验直接混比。

**文档同步**：`implementation.md` 是 | `idea_report.md` 是 | `dev_log.md` 是 |
`README.md` 是。

### 2026-07-21：v11b 五组实验正式评估与消融结论

**统一评估口径**：五组 checkpoint 均使用 `seed=1234`、MNIST/FSDD test
`n=10000`、`fixed_mask`、`severity=0.4` 和相同五组 image/audio corruption family。
表中 ACC 与 MaskMSE 均为五 family 宏平均；`aud C/F` 分别表示 AudioDecoder
coarse 与 AudioRefiner/paste-back final 的缺失区 MSE。所有日志均以“评估完成”
结束，demo 与 Cross-Key sweep 也已生成。当前只有一个训练 seed，且 CSV 保留四位
小数，因此小于约 `1e-4` 的差异不能作显著性结论。

| 实验 | 训练预算 | img-only ACC | aud-only ACC | both ACC | img Final MaskMSE | aud Coarse MaskMSE | aud Final MaskMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v11b_recovery` | 0-119，共 120 轮 | 95.50% | 86.04% | 97.74% | 0.01752 | 0.01320 | 0.02970 |
| `v11b_weighted` | recovery ep100 -> ep119，共 20 轮续训 | 95.56% | 86.06% | 97.76% | 0.01758 | 0.01314 | 0.02970 |
| `v11b_control` | recovery ep120 -> ep149，共 30 轮 | 95.50% | 86.04% | 97.74% | 0.01742 | 0.01298 | 0.02970 |
| `v11b_cross_no_causal` | 与 control 同父模型、同 30 轮预算 | 95.50% | 86.04% | 97.74% | 0.01742 | 0.01298 | 0.02970 |
| `v11b` causal | 与 control 同父模型、同 30 轮预算 | 95.50% | 86.04% | 97.74% | 0.01740 | 0.01300 | 0.02970 |

**实验 1 - `v11b_recovery` 独立结论**：对照为 v11a 与 v10f 主实验。新增 coarse
audio 直接监督有效：五 family coarse MaskMSE 降至 `0.01320`；排除已改为有效
语音区遮挡、难度显著升高的 `partial_temporal` 后为 `0.00708`，已经接近 v10f
final 的 `0.00520`。但 AudioRefiner 将五 family MaskMSE 从 `0.01320` 恶化到
`0.02970`（+125%）；排除 `partial_temporal` 后从约 `0.00708` 恶化到
`0.02248`。demo 中也可见 coarse audio 保留了待补能量，而 final 在多个连续
缺失区接近黑图。结论是 Decoder 恢复方案有效，当前主要故障已定位到 AudioRefiner，
不能把 `0.02970` 解释成 Decoder 没学会。ImageRefiner 仍有效，图像 coarse/final
宏平均约 `0.06130 -> 0.01752`。aud-only ACC 的宏平均下降主要由
`partial_temporal` 的 47.0% 拉低；其余四 family 宏平均约 95.8%。

**实验 2 - `v11b_weighted` 独立结论**：对照为共享 recovery ep100 状态、相同
optimizer/scheduler 和 seed 的 `v11b_recovery` 后 20 轮。归一化 energy-weighted
masked MSE 仅把 audio coarse 宏平均从 `0.01320` 变为 `0.01314`，final 仍为
`0.02970`；time_mask coarse 还从 `0.0101` 轻微变差到 `0.0102`。ACC 和图像指标
也无稳定收益。该权重在当前 `lambda=0.25` 下没有解决高能语音补洞或 Refiner
塌缩，不应选作后三路 Cross-Key 实验的父 checkpoint。该比较预算和起点干净，
但仍只有单 seed。

**实验 3 - `v11b_control` 独立结论**：对照为 `v11b_recovery`，从同一 ep120
父 checkpoint 以 `lr=1e-4` 微调 Decoder/Cross-Key 同构参数 30 轮，Cross-Key
前向关闭。audio coarse 宏平均小幅改善到 `0.01298`，但 final 固定在 `0.02970`；
图像 final 为 `0.01742`。它证明额外低学习率 Decoder 微调只能继续改善 coarse，
不能自行修复被冻结/未被当前可训练前缀覆盖的 AudioRefiner；同时它是后两项
Cross-Key 实验的有效同预算因果对照。

**实验 4 - `v11b_cross_no_causal` 独立结论**：对照为同父 checkpoint、同 seed、
同 30 轮预算的 `v11b_control`，唯一关键差异是启用 Cross-Key 条件通路、不启用
causal pair loss。主指标在 CSV 精度下与 control 完全相同。Cross-Key sweep 中
`image->audio` gate/residual-to-Value 均值约为 `0.00087/0.00482`，但 coarse 与
final gain 均为 `0.00000`；`audio->image` 约为 `0.00643/0.06130`，只产生
`0.00014` coarse gain，经过 ImageRefiner 后仅剩 `0.00001`。结论是普通重建损失
虽能让条件模块产生非零残差，却没有证明图像参与音频补洞；音频对图像只有极小的
coarse 语义修正，final 贡献近零。

**实验 5 - `v11b` causal 独立结论**：对照为 `v11b_control` 与
`v11b_cross_no_causal`，三者共用 recovery ep120 父模型、相同 seed、相同 30 轮
预算。causal pair loss 将 `image->audio` gate/residual-to-Value 提高到约
`0.00117/0.00714`，将 `audio->image` 提高到约 `0.00784/0.07088`；但
`image->audio` 的 correct-vs-zero coarse/final gain 仍为 `0.00000`，
`audio->image` 也只有 `0.00018/0.00001`。wrong-class 与 same-class-different-sample
干预几乎不改变误差，没有形成预期的 correct < same/zero < wrong 因果排序。
因此该 causal loss 只放大了条件路径幅度，没有带来可测的跨模态缺失区恢复收益，
v11b 不能宣称已经实现有效的双向 Cross-Key 补洞。

**跨模式与跨版本判断**：`clean_img_corrupt_aud` 的 audio final MaskMSE 在所有
五组实验、所有 family 中都与 `corrupt_aud_only` 相同；`corrupt_both` 的 audio
宏平均反而约差 `0.00020`。`corrupt_img_clean_aud` 相比 `corrupt_img_only` 的
image final 宏平均约改善 `0.00052`，但该现象在 Cross-Key 关闭的 recovery/control
中已经存在，应归因于原共享 Index/Value 通路，不能归因于新增 Cross-Key。
与 v10f 的四个有效 audio family 比较，v11b final `0.02248` 约为 v10f
`0.00520` 的 4.3 倍；但 v11b coarse 已到约 `0.00710`，说明下一轮应先旁路、
冻结后重训或重新约束 AudioRefiner，而不是继续扩大 Cross-Key。图像 final
MaskMSE `0.01740` 比 v10f 的 `0.01846` 约低 5.7%，图像恢复路径保持有效。

**归档结论**：五项实验均已完成结果归档。v11b 的可靠正结论是 coarse audio
监督有效、ImageRefiner 有效、active-aware `partial_temporal` 不再利用尾部静音；
可靠负结论是 energy weighting、普通 Cross-Key 和当前 causal pair loss 均未改善
最终音频补洞，且 AudioRefiner 明显破坏了已经改善的 coarse audio。后续版本的
首要对照应是同一 recovery checkpoint 上的 audio refiner bypass/paste-back-only，
并以 coarse-vs-final MaskMSE 为验收门槛。

### 2026-07-21 21:30 - v11c AudioRefiner 旁路基线与 v11b 配置归档

**阶段**：F 代码迭代。

**背景**：v11b 五 family 的 audio coarse/final MaskMSE 约为
`0.0130/0.0297`；排除 `partial_temporal` 后约为 `0.0071/0.0225`。
AudioDecoder coarse 已具备恢复能力，外置 AudioRefiner 反而系统性放大缺失区误差。

**文档先行**：编码前已更新 `docs/implementation.md` 与
`docs/idea_report.md`，明确 v11c 为 AudioRefiner bypass + coarse paste-back 的稳定基线。

**实现**：

- `models/network.py` 新增 `audio_refiner.bypass`。旁路时仍构造 AudioRefiner
  以保持 v11b state dict 键和形状兼容，但冻结参数并禁止前向调用。
- 有 audio cue 且有 mask 时，最终输出为
  `mask * coarse_aud + (1-mask) * aud_cue`；无 cue/mask 时直接返回 coarse。
- `scripts/train.py` 在 bypass 时禁止 AudioRefiner 进入 decoder pretrain。
  `loss.lambda_aud_coarse=0`，避免 final 缺失区等于 coarse 时重复计权。
- 新增 `configs/v11c.yaml` 与 `configs/v11c_control.yaml`；两者共用
  `cross_modal_snn_v11b_recovery.pt` 第 120 轮父权重、seed 和 30 轮预算，
  唯一核心差异是 Cross-Key causal 路径开/关。
- 删除已完成实验的五份 `configs/v11b*.yaml`。v11b 结果和独立消融
  结论仍保留在本日志的历史条目中。
- README、默认脚本路径、smoke test 和 `scripts/plot_eval_summary.py`
  已同步到 v11c。

**验证**：

- Python 语法检查通过。
- v11c/main-control 配置关键项、cue 概率和独立输出路径检查通过。
- v11b 形状对照与 v11c 的 state dict 共 120 个 tensor，键顺序与形状完全一致。
- 本地归档的 `checkpoint-v11b/cross_modal_snn_v11b_recovery.pt` 已对
  v11c 模型执行 `strict=True` 实际加载，所有键完全匹配。
- `scripts/smoke_test.py` 通过：旁路公式、参数冻结、Cross-Key 梯度、
  8 种 cue 前向/反向和 target-free infer 均正常。
- `docs/` 和 `.gitignore` 仍不纳入 Git 跟踪或提交。

**待实验**：v11c 与 v11c_control 完成评估后，必须在本日志分别
追加独立结论；主验收以 audio coarse/final MaskMSE 是否一致、以及
v11c 相对 control 的 correct/zero/wrong Cross-Key 差值为核心。

### 2026-09-08 11:55 - v11c / v11d / v11e 仓库评价归档

**阶段**：F-1 诊断分析与文档归档。

**触发原因**：用户要求严格按 F 阶段流程评价仓库中的 `v11d`、`v11e`，
随后补充必须同时评价 `v11c`。本次仅做仓库与文档诊断，不修改代码。

**检查范围**：

- 已读取 `docs/user_requirements.md`、`docs/dev_log.md` 与
  `docs/idea_report.md`。
- 已检查本地分支状态：当前工作区在 `v11c`，落后 `origin/v11c` 一个提交；
  远端存在 `origin/v11c`、`origin/v11d`、`origin/v11e`。
- 已读取远端配置：`configs/v11c.yaml`、`configs/v11c_control.yaml`、
  `configs/v11d.yaml`、`configs/v11d_control.yaml`、`configs/v11e.yaml`、
  `configs/v11e_control.yaml`。
- 已检查关键代码路径：`common.py`、`data/dataset.py`、
  `models/network.py`、`scripts/train.py`、`scripts/evaluate.py`、
  `scripts/demo_inference.py`，以及 `docs/V11E_REAL_PAIR_PROTOCOL.md`。
- 本地 `outputs/` 中未检测到 `outputs_v11c`、`outputs_v11d`、
  `outputs_v11e`；`outputs/checkpoints/` 中未检测到 v11c/v11d/v11e
  checkpoint。因此本条目不能写成数值实验结论，只能写成架构、数据协议
  与评估流程诊断。

**v11c 评价**：

- `v11c` 是必要的稳定基线：保留 `AudioRefiner` 模块以兼容 v11b
  checkpoint，但设置 `audio_refiner.bypass=true`，音频 final 等价于
  `AudioDecoder` 输出。
- `detail_conditioning.detach_value_for_recon=true` 保留了此前确认的隔离策略：
  重建 loss 不通过 `V_from_A` 反向拉动 Index/Value，使分类与联想主干
  不被 decoder 目标直接改写。
- `v11c_control` 与 `v11c` 共享父 checkpoint、seed 和 30 轮预算，核心差异是
  Cross-Key/causal path 是否启用，适合验证“对侧 Key 是否对 decoder
  maskedMSE 有真实贡献”。
- 本地尚无 v11c 评估结果；v11c 完成后必须先确认带 mask 的音频模式满足
  `aud_final_mask_mse == aud_coarse_mask_mse`，再比较 main/control 与
  `--cross_key sweep`。

**v11d 评价**：

- `v11d` 在 v11c 基础上引入固定一一伪配对与 `Cross-Detail`。数据集中每个
  MNIST item 固定对应一个 FSDD 录音实例及确定性增强种子，并返回稳定
  `pair_id`。
- Cross-Detail 实现为对侧 Key 前实例特征 `rate(spike_*_instance)`，经独立
  projector 与 vector gate 后加到目标模态 detail channel；decoder 输入尺寸
  不改变，仍是 `fused_value + detail`。
- `v11d_control` 只关闭 Cross-Detail，仍保留 Cross-Key，因此它可以隔离
  Cross-Detail 的增量贡献。
- 主要限制是数据定义：MNIST 图像与 FSDD 音频只共享数字类别，不共享真实
  物理事件、说话人或笔迹因果信息。固定伪配对最多能检验网络是否记住
  人工 pair id，不应被解释为真实跨模态实例恢复。
- 若 v11d 的 correct/zero/same-class Cross-Detail 差异不明显，首要解释应是
  伪配对数据不提供可泛化实例关系，而不是继续扩大 Cross-Detail 模块。

**v11e 评价**：

- `v11e` 是比 v11d 更合理的主线：数据切换为 GRID 真实音视频同源事件，
  manifest 强制 `source_id == image_source_id == audio_source_id`，并按
  speaker 划分 train/val/test。
- v11e 从头训练，不继承 MNIST/FSDD 权重；这是必要的，因为图像域从 MNIST
  数字图变为 28x28 mouth-motion dynamic image，音频采样率和时长也切换到
  GRID 协议。
- v11e 加入 `pair_alignment`，用同类不同 source 作为 hard negatives，
  并在 evaluate 中报告同类候选内 exact-pair Recall@1。这是验证“实例级
  跨模态对齐”的关键补充。
- 需要注意一个归因风险：`v11e.yaml` 的 validation score 包含 pair retrieval，
  而 `v11e_control.yaml` 关闭 pair loss 后 `validation.score.lambda_pair=0`。
  因此 best checkpoint 的选择标准不同。正式比较时应同时报告 best 与 final，
  或统一 checkpoint 选择口径后再做 main/control 对比。
- v11e 的 `prepare_grid_v11e.py` 明确不把增强视图计为新 pair，并输出
  `dataset_summary.json`，后续报告必须同时写清 unique pair 数与每 epoch
  optimizer exposures。

**综合判断**：

- `v11c` 是 AudioRefiner-free 的稳定恢复基线，必须先补评估。
- `v11d` 是伪配对与 Cross-Detail 的过渡验证，适合做负结果解释与机制审计，
  不适合作为最终科学主线。
- `v11e` 才真正回答“能否在真实同源视听事件上做实例级跨模态联想恢复”。
  后续优先投入 v11e，但必须用 control、Cross-Detail sweep、pair Recall@1
  与 speaker-disjoint split 共同支撑结论。

**后续归档要求**：

- v11c、v11d、v11e 每个主实验与 control 完成后，都必须在本日志追加独立
  结果条目。
- 每个结果条目至少写清：训练轮数、seed、checkpoint、评估协议、fixed/random
  mask 口径、主要 cue 模式指标、main-control 差异、sweep 因果指标、异常与
  比较限制。
- 未跑出本地 `outputs_v11c/v11d/v11e` 前，不得把本条目的架构诊断当作实验结论。

### 2026-09-08 12:09 - implementation.md 补齐 v11c / v11d / v11e 实现说明

**阶段**：F-0/F-1 文档同步修复。

**触发原因**：用户指出 `implementation` 文档未填写。检查确认此前只补了
`dev_log.md` 与 `idea_report.md`，但 `docs/implementation.md` 尚未完整覆盖
v11c、v11d、v11e 的实现入口、配置、评估、数据与产物路径。

**修改文件**：

- `docs/implementation.md`
- `docs/dev_log.md`

**补齐内容**：

- 更新文档头部版本范围为 v10a...v11e，并标记当前本地分支 `v11c`、
  远端含 `origin/v11d` / `origin/v11e`。
- 新增 v11d 固定伪配对 + Cross-Detail、v11e GRID 真配对 + Pair Alignment
  的迭代范围说明。
- 更新目录结构，补充 `configs/v11d*.yaml`、`configs/v11e*.yaml`、
  `scripts/smoke_test_v11d.py`、`scripts/smoke_test_v11e.py`、
  `scripts/prepare_grid_v11e.py` 与 `_data/grid_v11e`。
- 补充 `models/network.py` 中 Cross-Key、Cross-Detail、pair embedding、
  `_fuse_decoder_state` 扩展参数与 `forward` 新 mask/override 参数。
- 补充 `data/dataset.py` 中固定伪配对、`TruePairedManifestDataset`、
  manifest 校验和 `build_loaders` 新分支。
- 补充 `common.py` 中 `extends` deep merge、`unpack_paired_batch`、
  image/audio mask 返回与 paired sample target 选择。
- 补充 `scripts/train.py` 中 Cross-Detail causal loss、pair alignment loss、
  v11e validation score 与 best checkpoint 语义。
- 补充 `scripts/evaluate.py` / `scripts/demo_inference.py` 的 v11d/v11e
  fixed/random、Cross-Key/Cross-Detail sweep、pair retrieval、random demo
  可视化口径。
- 补充 v11d/v11e 配置约定、GRID 数据准备流程、checkpoint、日志、表格与
  图像产物路径。
- 修正实现指南校验表中“`docs/idea_report.md` 尚不存在”的过期警告。

**验证结果**：

- 本次只改文档，未修改代码，未运行训练或评估。
- 已通过 `rg` 检查 v11d/v11e/Cross-Detail/paired manifest 等关键段落存在。
- 注意：`docs/` 目录当前被 Git 忽略，本次文档修复不会出现在普通 `git diff`
  中；需要保留时应另外处理文档跟踪策略或手动备份。

### 2026-09-08 13:07 - v11c / v11d 本地评估归档与 checkpoint 仓库同步

**阶段**：F-1/F-5 结果评价与产物归档。

**触发原因**：用户说明本地已有 v11c、v11d 评估结果，要求开始评价，并将
本地存在但 checkpoint 仓库缺失的 checkpoint 上传到仓库。

**评估来源**：

- v11c：`v11cnew_outputs_with_ckpt/outputs/outputs_v11c/`
- v11c control：`v11cnew_outputs_with_ckpt/outputs/outputs_v11c_control/`
- v11d：`v11d_outputs_with_ckpt/outputs/outputs_v11d/`
- v11d control：`v11d_outputs_with_ckpt/outputs/outputs_v11d_control/`
- 评估协议：fixed-mask，severity=0.4，5 个 image/audio family pair，8 种 cue mode。

**五 family 平均主指标**：

| 实验 | cue | ACC | ImgMaskMSE | AudMaskMSE | AudSSIM | target |
|---|---|---:|---:|---:|---:|---|
| v11c | corrupt_img_only | 0.9550 | 0.0174 | 0.0003 | 0.966 | sample/category |
| v11c | corrupt_aud_only | 0.8588 | 0.0098 | 0.0123 | 0.780 | category/sample |
| v11c | corrupt_both | 0.9762 | 0.0169 | 0.0120 | 0.784 | sample/sample |
| v11c | clean_img_corrupt_aud | 0.9870 | N/A | 0.0119 | 0.785 | sample/sample |
| v11c | corrupt_img_clean_aud | 0.9906 | 0.0169 | N/A | 0.876 | sample/sample |
| v11c | clean_both | 0.9940 | N/A | N/A | 0.876 | sample/sample |
| v11c_control | corrupt_aud_only | 0.8588 | 0.0098 | 0.0120 | 0.779 | category/sample |
| v11c_control | corrupt_both | 0.9762 | 0.0169 | 0.0117 | 0.782 | sample/sample |
| v11d | corrupt_img_only | 0.9550 | 0.0173 | 0.0194 | 0.254 | sample/sample |
| v11d | corrupt_aud_only | 0.5224 | 0.0603 | 0.0144 | 0.724 | sample/sample |
| v11d | corrupt_both | 0.8760 | 0.0170 | 0.0133 | 0.739 | sample/sample |
| v11d | clean_img_corrupt_aud | 0.9446 | N/A | 0.0133 | 0.737 | sample/sample |
| v11d | corrupt_img_clean_aud | 0.8994 | 0.0170 | N/A | 0.836 | sample/sample |
| v11d | clean_both | 0.9570 | N/A | N/A | 0.838 | sample/sample |
| v11d_control | corrupt_aud_only | 0.5224 | 0.0620 | 0.0144 | 0.727 | sample/sample |
| v11d_control | corrupt_both | 0.8760 | 0.0170 | 0.0133 | 0.739 | sample/sample |

**v11c 结论**：

- v11c main 与 control 的 ACC 完全重合；主要 cue mode 的 MaskMSE 差异在
  `0.0000~0.0004` 量级，AudSSIM 只有约 `+0.001` 的微小波动。
- 因此 v11c 的可靠结论是：AudioRefiner bypass 后恢复链路稳定，分类也保持高位；
  但 Cross-Key 在该评估下没有形成可报告的实质增益。
- v11c 可作为 AudioRefiner-free baseline，但不能声称对侧 Key 已带来显著实例级
  补全能力。

**v11d 结论**：

- v11d 把缺失模态 target 改成 `sample/sample`，任务比 v11c 的
  `category/sample` 或 `sample/category` 更难，因此 v11d 与 v11c 不能作同难度
  直接横比。
- v11d main 与 control 的 ACC 仍完全重合；主要 MaskMSE 差异基本在
  `0.0000~0.0008`，说明 Cross-Detail 对主评估没有稳定净增益。
- v11d 的显著退化集中在 audio-only 到 image/sample 的链路：
  `corrupt_aud_only ACC=0.5224`、`ImgMaskMSE=0.0603`，`clean_aud_only`
  也只有 `ACC=0.6470`。这符合“MNIST/FSDD 固定伪配对缺少真实实例对应”的预期。
- Cross-Key sweep 中 gate 和 residual/value 很小，平均 gain 基本为 0；
  对侧 Key 仍主要不是恢复指标的决定因素。
- Cross-Detail sweep 中，`aud->img` 在 `corrupt_aud_only` 上有
  `gain≈+0.0159`，但 same-class wrong detail 的 damage 接近 0；这说明它学到的是
  同类图像补偿或类别级先验，而不是正确 pair 的实例级信息。
- `img->aud` 的 Cross-Detail gain 约为 0 或轻微为负，说明 MNIST 图像无法为
  FSDD 音频实例提供可泛化的细节恢复线索。

**综合判断**：

- v11c 是当前更稳的 baseline：分类高、恢复稳定、但跨模态条件贡献弱。
- v11d 是有价值的负结果/机制审计：它验证了固定伪配对无法支撑实例级
  Cross-Detail，不能作为最终主线。
- 后续若继续追求实例级跨模态恢复，应优先使用 v11e 的真实视听配对，而不是继续
  在 MNIST/FSDD 伪配对上扩大 Cross-Detail 容量。

**checkpoint 仓库同步**：

- 目标仓库：`git@github.com:xuebossvh/cross_modal_attractor_snn_checkpoints.git`
- 已有且 hash 一致：`cross_modal_snn_v11c.pt`、`cross_modal_snn_v11c_control.pt`、
  `cross_modal_snn_v11d.pt`、`cross_modal_snn_v11d_control.pt` 等。
- 本次新增上传 14 个本地存在但 checkpoint 仓库缺失的 LFS checkpoint：
  `cross_modal_snn_v10a.pt`、`v10b.pt`、`v10c.pt`、`v10d.pt`、`v10e.pt`、
  `v10f.pt`、`v11a.pt`、`v11a_control.pt`、`v11b.pt`、`v11b_control.pt`、
  `v11b_cross_no_causal.pt`、`v11b_decoder_pretrain.pt`、
  `v11b_recovery_ep100.pt`、`v11b_weighted.pt`。
- 本次未上传 `_mismatch.pt`、`_smoke_v11a.pt` 和 `_data/audio_norm_stats.pt`，
  因为它们不是正式实验 checkpoint。
- checkpoint 仓库提交：`a48d8c1 Add missing v10-v11b checkpoints`。
- `git lfs ls-files` 已确认新增文件均为 LFS 对象；远端 `refs/heads/main`
  已核验为 `a48d8c1f3eccbd86535592d84ea41b6a614df423`。

#### 已知问题

- [ ] `docs/idea_report.md` 已建立；v10c / v10d 实验设计已写入；后续 F 阶段迭代须同步更新。
- [ ] `outputs/outputs_v10a/` 尚不存在。重定向 v10a 日志前，应先运行 `python scripts/mkdir_outputs.py --config configs/v10a.yaml`。
- [ ] 创建日志时未在 `outputs/checkpoints/` 下检测到 v10a checkpoint。
- [ ] 当前 `requirements.txt` 包含 `torch`、`torchvision`、`torchaudio`；后续需决定保留当前项目实用约定，还是改为只在 README 中说明 PyTorch 安装。
- [x] 当前项目 Python 已提供 `torch 1.13.1+cpu`；v11a Cross-Key smoke 与 toy `max_batches=1` 已在本机通过。
- [ ] v10b 尚未在带 PyTorch 的项目环境中完成 smoke test、训练或评估。

---



### 2026-09-08 13:32 - v11c / v11d 详细评估补充

**触发原因**：用户要求详细评价本地已有的 v11c、v11d 结果，并要求当前工作区
后续切换到 `v11e` 分支。

**评估依据**：

- v11c fixed-mask：
  `v11cnew_outputs_with_ckpt/outputs/outputs_v11c/tables/family*/full_eval.csv`
- v11c control fixed-mask：
  `v11cnew_outputs_with_ckpt/outputs/outputs_v11c_control/tables/family*/full_eval.csv`
- v11d fixed-mask：
  `v11d_outputs_with_ckpt/outputs/outputs_v11d/tables/main_eval/family*/full_eval.csv`
- v11d control fixed-mask：
  `v11d_outputs_with_ckpt/outputs/outputs_v11d_control/tables/main_eval/family*/full_eval.csv`
- v11d Cross-Key attribution：
  `v11d_outputs_with_ckpt/outputs/outputs_v11d/tables/cross_key_eval/family*/cross_key_attribution.csv`
- v11d Cross-Detail attribution：
  `v11d_outputs_with_ckpt/outputs/outputs_v11d/logs/eval_v11d_cross_detail_sweep_sev04.log`

**总体指标复核**：

| 版本 | cue | ACC | img mask MSE | aud mask MSE | aud SSIM | target |
|---|---:|---:|---:|---:|---:|---|
| v11c | corrupt_img_only | 0.9550 | 0.0174 | 0.0003 | 0.966 | sample/category |
| v11c | corrupt_aud_only | 0.8588 | 0.0098 | 0.0123 | 0.780 | category/sample |
| v11c | corrupt_both | 0.9762 | 0.0169 | 0.0120 | 0.784 | sample/sample |
| v11c | clean_img_corrupt_aud | 0.9870 | N/A | 0.0119 | 0.785 | sample/sample |
| v11c | corrupt_img_clean_aud | 0.9906 | 0.0169 | N/A | 0.876 | sample/sample |
| v11c | clean_both | 0.9940 | N/A | N/A | 0.876 | sample/sample |
| v11d | corrupt_img_only | 0.9550 | 0.0173 | 0.0194 | 0.254 | sample/sample |
| v11d | corrupt_aud_only | 0.5224 | 0.0603 | 0.0144 | 0.724 | sample/sample |
| v11d | corrupt_both | 0.8760 | 0.0170 | 0.0133 | 0.739 | sample/sample |
| v11d | clean_img_corrupt_aud | 0.9446 | N/A | 0.0133 | 0.737 | sample/sample |
| v11d | corrupt_img_clean_aud | 0.8994 | 0.0170 | N/A | 0.836 | sample/sample |
| v11d | clean_both | 0.9570 | N/A | N/A | 0.838 | sample/sample |

**v11c 判断**：

- v11c 是目前 MNIST/FSDD 伪配对设定下更稳的 v11 系基线。它保留
  `sample/category` 与 `category/sample` 的不对称目标：缺失方向允许退回类别
  原型，存在同模态 cue 时再恢复样本细节。因此分类 basin 很稳，
  `clean_both` ACC=0.9940，`corrupt_both` ACC=0.9762。
- v11c 的主要失败点不是整体塌缩，而是音频 `partial_temporal` 遮挡。该 family
  下 `corrupt_aud_only` ACC=0.4700、aud mask MSE=0.0336、aud SSIM=0.513；
  其余四类音频遮挡的 `corrupt_aud_only` 平均 ACC 约 0.956，aud mask MSE
  约 0.0070。也就是说，v11c 对随机/频域/局部块类遮挡可恢复，但对连续时间段
  缺失仍明显脆弱。
- 当图像干净、音频残缺时，`partial_temporal` 的 ACC 可从 0.4700 提升到
  0.9670，说明干净图像 cue 能把 Index 类别 basin 拉回来；但 aud mask MSE
  仍约 0.0321，说明图像 cue 主要修正类别，不提供说话人、节奏和局部能量等
  音频样本细节。
- v11c main 与 v11c control 的 ACC 基本完全重合，MaskMSE/SSIM 只在 1e-4 到
  1e-3 量级波动。本地 v11c 包未发现 Cross-Key sweep 归因日志，因此 v11c 的
  Cross-Key 贡献只能由 main/control 近似判断：目前看它不是主要性能来源。

**v11d 判断**：

- v11d 将目标统一为 `sample/sample`，并加入 Cross-Detail；这让它和 v11c 不再是
  同难度比较。MNIST 图像与 FSDD 音频只是固定一一伪配对，不是真实同源事件，
  因此“只给图像恢复对应音频样本”或“只给音频恢复对应图像样本”在信息论上
  缺少可观测依据。
- 这个难度变化直接体现在评估：`clean_img_only` 的分类仍可达 0.9860，但音频
  aud SSIM 只有 0.255、aud mask MSE=0.0194；`clean_aud_only` ACC 只有 0.6470，
  img mask MSE=0.0567、img SSIM=0.562。分类可以借由可见模态判断类别，但
  另一个模态的样本细节无法从伪配对中推断。
- v11d 后期训练日志平均 loss 仍在 9 到 10 左右，明显高于 v11c 后期约 0.96 到
  0.98 的区间。高 loss 主要来自 `clean_aud_only`、`corrupt_aud_only` 等
  audio-only 到 image/sample 的不可识别目标；日志中这些 batch 的 `cls`、
  `aux_aud`、`soft_cls` 项经常显著偏高。
- v11d 与 v11d_control 的 ACC 完全重合，主要 MSE/SSIM 差异也很小，说明
  Cross-Detail 没有带来可见主指标增益。v11d 的退化主要来自 sample/sample
  目标和固定伪配对，而不是 control 是否关闭 Cross-Detail。
- v11d Cross-Key attribution 很弱：`aud->img` gate 约 0.011 到 0.015，
  `img->aud` gate 约 0.043 到 0.055；`gain` 基本为 0 到 2e-4。Cross-Key
  当前几乎没有承担有效恢复。
- v11d Cross-Detail attribution 更强但方向不对称：`aud->img` 在
  `clean_aud_only` / `corrupt_aud_only` 下 gain 约 +0.016，但 same-class
  damage 约 0，表示同类音频 detail 与正确配对 detail 几乎等价，学到的是类别
  或能量统计，而不是实例级配对。`img->aud` 的 gain 约为 0 或略负，说明图像
  detail 对音频样本恢复基本无帮助。

**结论**：

- v11c 可以作为 MNIST/FSDD 伪配对阶段的主结果候选。它的科学表述应是：
  循环吸引子负责稳定类别联想，decoder/detail 负责在同模态 cue 可用时补样本
  细节；连续时间缺失的音频恢复仍是主要短板。
- v11d 不适合作为性能主线，应作为负结果或机制审计：它验证了固定伪配对不足以
  支撑跨模态实例级恢复。继续在 v11d 上调 `lambda`、gate 或 decoder 宽度，
  很可能只会让网络记忆伪相关，而不是解决根因。
- 若坚持 sample/sample 跨模态恢复，后续应转向 v11e 这类真实配对数据，例如
  GRID；若仍使用 MNIST/FSDD，则建议保持 v11c 的类别级缺失目标，不再强迫
  image-only 预测音频样本、audio-only 预测图像样本。

**checkpoint 同步状态**：

- 已确认 v11c/v11d 正式 checkpoint 在
  `cross_modal_attractor_snn_checkpoints` 仓库中存在，且本地与仓库版本哈希一致。
- 已补传本地存在而 checkpoint 仓库缺失的 v10a-v10f、v11a、v11b 及相关 control/
  weighted/recovery/pretrain checkpoint。
- checkpoint 仓库提交：`a48d8c1 Add missing v10-v11b checkpoints`。

### 2026-09-08 15:10 - v11e 改为 MNIST/FSDD 类别级绑定

**改动原因**：MNIST 手写实例与 FSDD 语音实例只共享 digit label，不存在天然
一一对应；旧 v11e 的 GRID `paired-sample/paired-sample` 任务偏离原始
MNIST+FSDD 类别联想问题。用户确认将 v11e 改为类别级绑定，并要求版本分支不再
携带无关旧版 YAML。

**改动内容**：

- `configs/v11e.yaml`：改为自包含 MNIST/FSDD 配置；同类 many-to-many 随机组合，
  `sample_targets_for_missing=false`；保留 Cross-Key 与同模态 detail，关闭
  Cross-Detail、pair alignment 和测试集 best-checkpoint 选择。
- `configs/v11e_control.yaml`：同数据、同 target、同预算，只关闭 Cross-Key；模块
  仍构造，main/control state dict 可 strict 互载。
- `configs/v11c*.yaml`、`configs/v11d*.yaml`：从 v11e 分支工作树移除，历史配置留在
  对应旧分支；v11e 不再通过 `extends` 依赖旧版 YAML。
- `scripts/prepare_grid_v11e.py`、`scripts/smoke_test_v11d.py` 与旧 GRID 协议文档：
  从当前 v11e 分支移除，仍可从历史提交获取。
- `scripts/smoke_test_v11e.py`：改为类别协议测试，覆盖 image-only
  `sample/category`、audio-only `category/sample`、双模态 `sample/sample`，并检查
  main/control 结构一致。
- `README.md`、`docs/idea_report.md`、`docs/implementation.md`、
  `docs/V11E_CATEGORY_BINDING_PROTOCOL.md`：同步当前数据、目标、命令与评估口径。
- `docs/user_requirements.md`：新增 commit/push 文件清单规范，禁止 `git add .`，新
  版本分支默认只保留当前版本必要配置。

**验证**：

- `python -m py_compile ...`：通过。
- `python -u scripts/smoke_test_v11e.py`：通过，输出
  `v11e category-binding smoke test passed`。
- YAML 合并检查：dataset=`mnist_fsdd`，main=`Cross-Key on / Cross-Detail off /
  pair alignment off`，control 仅 `Cross-Key off`，audio shape=`64x64`。
- `git diff --check`：通过，仅有 Windows LF/CRLF 提示。

**当前结论**：代码协议已经按类别级绑定修正；尚未进行正式 GPU 训练，因此没有
新增性能结论。FSDD 扩充与额外数据增强本轮按用户要求不处理。

### 2026-09-08 16:32 - 将 v11e 当前方案提升到实现指南顶层目录

**改动原因**：`docs/implementation.md` 虽已在 7.10 节记录当前类别级方案，但
顶部迭代目录只显示 0.10 的废止 GRID 方案，容易被误读为新方案尚未填写。

**改动内容**：新增 0.11“v11e 当前方案（MNIST/FSDD 类别级绑定）”，集中列出
数据形状、同类 many-to-many、三类 target 粒度、Decoder 输入、Value detach、
Cross-Key main/control、关闭实例级伪监督以及版本配置清理规则；0.10 明确链接到
0.11 和 7.10，完整技术细节仍由 7.10、8.2 节承载，避免互相矛盾。

**验证**：检查 Markdown 标题后，0.11 已位于“项目结构”之前并会直接出现在文档
顶部目录；本次仅调整文档结构，不改变代码、配置或运行命令。

### 2026-09-09 v11f：冻结父模型的缺失区域 Cross-Key

#### 需求与回溯

用户同意四项修改并最终指定 `v11f` 分支。不是 `v11`，也不覆盖 v11e。
按 F 阶段先更新需求、implementation 与实验设计，再实现和测试。
诊断是 v11e 的部分残缺收益有限且联合训练存在分类代价；本轮通过冻结基线和
局部调制检验改进假设，不预先认定之前差异的因果来源已经完全确定。

#### 逐文件修改

- `configs/v11f.yaml`：从自包含 v11e 配置迁移；batch=128、simultaneous、
  类别绑定、Value + own detail、detach_value_for_recon 保留。父模型为
  v11e_control，固定 severity=0.4，额外训练 30 轮，仅 adapter 可训练。
- `configs/v11f_control.yaml` / `v11f_no_causal.yaml`：固定父模型评估参考与同父、
  同 seed/额外预算的去因果项实验。删除新分支旧 v11e YAML，不影响旧分支。
- `models/decoders.py`：原 decoder 拆分 feature/head 接口（不改旧 state key）；
  新 MaskedCrossKeyAdapter 使用上下文、Key 乘性条件、逐位置/通道 gate，输出层零初始化。
- `models/network.py`：v11f 禁用旧全局 Value residual，保留参数只为父权重兼容；
  在缺失区作特征调制，head/refiner 后按 mask 再选择，保护可见基线输出。
  冻结参数与模块运行状态；新增恢复内容单模态再分类，不走 decoder 递归。
- `models/frozen_base.py`：校验父文件 SHA256、关键前向配置、音频归一化统计；
  只允许缺少新 adapter 参数；每次保存检查全部基础 state（含 buffers）摘要，
  resume 校验 provenance 和前向配置。
- `scripts/train.py`：跳过冻结基础网络无效的 binding/teacher 计算；只优化 adapter；
  保留绝对重建项，因果项改用 batch floor，zero/wrong reference 不反传。
  单类别 batch 仍用 zero 监督；clean-only 无梯度时不做 optimizer step。
  父权重/续训权重缺失直接报错，正式训练禁止静默 CPU 回退。
- `scripts/evaluate.py`：增加 normal/zero/wrong/same-class 绝对 masked MSE、
  win_zero/win_wrong/win_both 与有效样本数；新增恢复内容类别一致性代理。
  按 family 写长表 CSV；v11f random 明确随机抽取 family，独立 seed 可复现。
- `scripts/demo_inference.py`：保留 fixed/random 六图，严格加载 checkpoint；
  禁止缺失或不兼容时以随机权重继续绘图。原 pred 注释仍是 Index 分类，未冒充
  恢复内容分类。
- `scripts/run_v11f_suite.py`：顺序执行 train/eval/demo，逐阶段日志；支持可选
  no_causal、eval_only、resume、dry_run。control 不训练；错误停止后续阶段。
- `scripts/smoke_test_v11f.py` / `scripts/smoke_test.py`：替换旧版本冒烟入口，
  覆盖模型、损失、严格加载、续训、评估 CSV 与两套 demo。
- `common.py`、`paths.py`、`scripts/mkdir_outputs.py`：默认配置改为 v11f。
- `README.md`、`docs/implementation.md`、`docs/idea_report.md`、
  `docs/user_requirements.md`、`docs/V11F_MASKED_CROSS_KEY_PROTOCOL.md`：同步版本、
  架构、维度、四点目的、对照定义、运行与验收方式。

#### 验证结果与限制

本地 Python 3.10 / torch 1.13.1+cpu：

1. compileall 通过；`git diff --check` 通过（仅 Windows 换行提示）。
2. 离线 smoke：8 种 cue 的前后向与 finite 梯度通过；新两个 adapter 的输出层、
   Key 投影及 gate 可获得梯度；基础参数没有梯度，优化前后摘要不变。
3. 新模块初始输出和干预 zero 输出与同一父模型严格一致；训练后保持可见位置不变，
   Index state/logits 不因 normal/wrong/zero 干预改变；零 Key 不产生特征残差。
4. 使用真实 `cross_modal_snn_v11e_control.pt`（epoch=99）执行同样的模型回归通过。
   父文件 SHA256 为 `5a792f10e57a95947c8e51bd915b09897baee01475103010826f25275b71501b`；
   基础 state 摘要为 `92b2f7d6d2127376d59ba093ab86c787a34d6efb038b7e0054ebdbe5489fc02a`。
   实际 control 配置严格加载通过，可训练参数数为 0。
5. CLI 合成数据 train 1 轮 + resume 至第 2 轮通过；两个评估协议的 CSV 字段和
   有效样本数通过；生成六张 demo 并检查尺寸、像素非空；缺失 checkpoint 被拒绝。
6. no_causal 套件 dry-run 顺序及独立日志检查通过；父权重未被改写，测试产物在临时目录。

这些是实现正确性/回归检查，不能当作 v11f 已收敛或 Cross-Key 已改善的实验结论。
本机无 CUDA，未执行 RTX 3080、batch=128 的完整 GPU 训练或显存稳定性验证。
恢复内容分类使用冻结原模型，只是内部一致性，不是外部独立识别准确率。
正式验收必须看部分残缺各 family 的 normal 相对固定 zero/control 是否改善，
同时检查 wrong/same-class；不能只看 gate、总 loss 或 wrong 变差。

#### 运行说明

以下为当前 v11f 命令；以上旧版本命令仅保留为历史记录。
在 v11f 项目根目录、可用 GPU 环境运行，先准备
`outputs/checkpoints/cross_modal_snn_v11e_control.pt` 以及同一数据/归一化统计。
无需重新训练父模型，不得用 v11e main 替代 control。

```bash
# main -> eval/demo -> frozen control eval/demo -> optional no_causal train/eval/demo
nohup python -u scripts/run_v11f_suite.py --with_ablations > v11f_suite.log 2>&1 < /dev/null &

tail -f v11f_suite.log

# All selected checkpoints must exist for eval_only/resume.
python -u scripts/run_v11f_suite.py --eval_only --with_ablations
python -u scripts/run_v11f_suite.py --resume

# Independent entries
python -u scripts/train.py --config configs/v11f.yaml
python -u scripts/train.py --config configs/v11f_no_causal.yaml
python -u scripts/evaluate.py --config configs/v11f_control.yaml --protocol fixed_mask --severity 0.4
python -u scripts/evaluate.py --config configs/v11f.yaml --protocol fixed_mask --severity 0.4 --cross_key sweep
python -u scripts/evaluate.py --config configs/v11f.yaml --protocol legacy_random --severity 0.4 --cross_key sweep
python -u scripts/demo_inference.py --config configs/v11f.yaml --protocol fixed_mask --severity 0.4
python -u scripts/demo_inference.py --config configs/v11f.yaml --protocol legacy_random --severity 0.4

# Offline regression only
python -u scripts/smoke_test_v11f.py --cli
python -u scripts/smoke_test_v11f.py --parent outputs/checkpoints/cross_modal_snn_v11e_control.pt
```

去掉 `--with_ablations` 只跑 main + 固定 control。新权重写入
`outputs/checkpoints/cross_modal_snn_v11f.pt` 和 `cross_modal_snn_v11f_no_causal.pt`；
其他产物进入 `outputs/outputs_v11f{,_control,_no_causal}/`。
fixed/random 的每个评估过程都会写各自 CSV，不互相覆盖；随机可视化带 `_random` 后缀。
套件结束标志是 `[suite] ALL STAGES COMPLETED`，`tail -f` 无新输出不代表仍在训练。

### 2026-09-09 - v11e 类别级绑定正式评估与 checkpoint 归档

**评估对象与可比条件**：

- 主实验 `v11e` 与 `v11e_control` 均从头训练 100 epoch，`seed=1234`、
  `batch_size=128`、`T=20`，使用同一 MNIST/FSDD 数据、同一 cue 概率、同一
  balanced corruption-family 采样和同一 cosine 学习率预算。两者唯一目标差异是
  main 启用 Cross-Key 与对应 causal loss，control 关闭该路径；control 仍构造同构
  模块，因此这是本轮主要的同预算消融对照。
- 正式评估采用 `severity=0.4`。fixed-mask 使用 `seed=1234`，覆盖五组 family pair；
  main 另有 legacy-random、Cross-Key normal/zero/wrong/same-class sweep，以及
  fixed/random 各三张 demo。当前只有一个训练 seed。
- 结果来自 `v11e_outputs_with_ckpt/outputs/outputs_v11e/` 与
  `v11e_outputs_with_ckpt/outputs/outputs_v11e_control/`。两个训练日志均完整到
  epoch 99，未发现 NaN、CUDA 异常或中断；末轮平均 loss 分别为 `1.2580` 与
  `1.2135`。

**五类音频残缺的 main/control 宏平均**：

| cue mode | ACC main/control | aud MaskMSE main/control | 相对变化 | aud SSIM main/control |
| --- | ---: | ---: | ---: | ---: |
| corrupt_aud_only | 83.62% / 84.47% | 0.010795 / 0.011125 | -2.97% | 0.8111 / 0.8044 |
| clean_img_corrupt_aud | 98.75% / 98.88% | 0.010255 / 0.010642 | -3.64% | 0.8180 / 0.8073 |
| corrupt_both | 96.69% / 97.09% | 0.010392 / 0.010762 | -3.44% | 0.8165 / 0.8064 |

Cross-Key main 在三种受损音频场景都带来约 `3%--4%` 的 masked-MSE 相对改善，
SSIM 与 top-15% 能量召回也小幅提高；代价是 ACC 下降 `0.13--0.84` 个百分点。
这些差异来自同预算 main/control，方向可信，但单 seed 下小幅 ACC 差异仍需多 seed
复验。

**图像侧宏平均与 refiner 归因**：

- 五个图像 family 上，`corrupt_img_only` 的 ACC main/control 都为 `92.02%`，
  img MaskMSE 为 `0.02750 / 0.02664`；`corrupt_both` 为 `97.10% / 97.56%`、
  `0.02592 / 0.02592`；`corrupt_img_clean_aud` 为 `98.96% / 99.14%`、
  `0.02550 / 0.02594`。Cross-Key 对部分残缺图像的平均增益很小，没有形成全面
  优于 control 的图像恢复结果。
- ImageRefiner 在 occlusion family 的洞内 MSE 从 coarse `0.0955` 降至 final
  `0.0624`，在 corrupt-both 从 `0.0878` 降至 `0.0572`，约改善 `35%`；
  visible paste-back 后可见区 MSE 为 0。图像 refiner 的作用清楚且稳定。
- AudioRefiner 处于 bypass，音频 final 即单一 decoder 输出；恢复音频标准差约
  `0.122--0.155`，目标约 `0.152`，top-15% 召回约 `68.8%--82.5%`，没有重新出现
  v11a/v11b 的低能量塌缩。

**Cross-Key 同 checkpoint 因果扫描**：

- 在目标模态完全缺失时，Cross-Key 证据很强。`aud->img` 的
  `corrupt_aud_only` 五 family 中，zero-normal gain 为 `0.0131--0.0236`，错误类别
  damage 为 `0.0213--0.0742`；`clean_aud_only` 的 gain/damage 为
  `0.0228 / 0.0766`。`img->aud` 的 `corrupt_img_only` gain 为
  `0.0003--0.0005`、wrong damage 为 `0.0005--0.0008`，相对其约
  `0.0005--0.0010` 的类别音频 MSE 仍是实质贡献。
- same-class different-sample 替换几乎不伤害结果，部分场景还略优于 normal；而
  wrong-class 替换显著恶化。这说明路径学到的是可替换的数字类别语义，符合
  many-to-many 类别级绑定，不能据此宣称实例级说话人、笔迹或时序迁移。
- 当目标模态仍有残缺 cue 时，Cross-Key 增益明显变小：`img->aud` 多数不超过
  `0.0007`，`aud->img` 多数不超过 `0.0048`。此时同模态 detail 与 paste-back 已
  提供主要信息，Cross-Key 更像缺失模态的类别兜底，而不是通用细节补全器。
- sweep 逐 batch 验证干预前后的 `index_state` 与分类预测完全一致，因此
  Cross-Key 在推理时只改变 decoder，不直接改变 Index/ACC。main/control 的 ACC
  差异来自训练阶段共享 encoder/detail 梯度的间接影响。当前没有
  “Cross-Key 开启但 causal loss 关闭”的第三组同预算实验，因而还不能拆分前向
  条件结构和 causal loss 各自的训练贡献。
- 两个方向并不等强：audio Key 对缺失图像的绝对作用远大于 image Key 对缺失音频；
  这也受两个输出空间的 MSE 尺度和类别 medoid 能量不同影响，不能只按绝对 gain
  直接比较方向强弱。

**鲁棒性、瓶颈与跨版本边界**：

- main 的 fixed/random 结果接近：`corrupt_img_only` ACC `89.6% / 89.4%`、
  `corrupt_aud_only` `89.4% / 88.7%`、`corrupt_both` 均为 `97.4%`；核心 MSE
  变化也在 `1e-4--5e-4` 量级。结论不是单张固定 mask 的偶然结果，但 random 仍
  只有一次评估，不能替代多 seed。
- 最大瓶颈是 `partial_temporal`：main 的 audio-only ACC `45.5%`、aud MaskMSE
  `0.0294`、SSIM `0.548`，显著差于其余四类；加入干净图像后 ACC 恢复到
  `97.6%`，说明 Index 能用图像语义救回类别，但音频样本细节仍无法由类别 Key
  代替。
- main 的 clean single-modality ACC 为 image `97.4%`、audio `96.7%`；control 为
  `97.7%`、`98.8%`。尤其 clean-audio 下降 `2.1` 个百分点，提示 Cross-Key/causal
  训练通过共享 audio encoder/detail 梯度带来分类与恢复权衡，后续需多 seed 确认。
- 相对 v11c，v11e 在第一 family 的音频 MaskMSE 有改善，例如 audio-only
  `0.0099 -> 0.0090`，clean audio MSE `0.0051 -> 0.0037`；但图像 occlusion
  MaskMSE `0.0398 -> 0.0624`，clean image MSE `0.0091 -> 0.0129`，分类也下降。
  v11c 使用不同训练起点/预算，v11e 又重新进行类别 many-to-many 训练，因此该
  跨版本比较只能描述取舍，不能把变化单独归因于某个模块。
- 日志中的 test `n=10000` 是 MNIST 条目数，不是 10000 条独立音频。FSDD test
  只有约 300 条录音并在同类 MNIST 条目间确定性复用；当前划分也不是
  speaker-independent。结果可以支持类别级联想机制，但不能外推为大规模或未知
  说话人泛化。

**结论**：v11e 已按预期实现并验证 MNIST/FSDD 类别级 many-to-many 联想。
Cross-Key 不是无效装饰：在一侧完全缺失时，zero/wrong/same-class 扫描给出了明确
的类别因果证据；但它尚未成为全面性能增益，部分残缺场景只获得小幅音频恢复改善，
同时有轻微分类代价，图像恢复也未整体胜过 control。下一步优先级应为：至少 3 个
训练 seed 复验 main/control；针对 `partial_temporal` 做专门建模或课程；用更大且
speaker-disjoint 的语音集合验证泛化；在此之前不继续扩大 Cross-Key 宽度或提高
causal loss 权重。

**checkpoint 归档**：本地正式权重的 Git LFS SHA-256 为
`e0e0cf0d52fff95599135d2dce154d1fa9395b54319a500cc164fa26716d89f4`
（main）与
`5a792f10e57a95947c8e51bd915b09897baee01475103010826f25275b71501b`
（control）；仅这两个 `.pt` 文件提交到独立 checkpoint 仓库，提交为
`f091a70 Add v11e main and control checkpoints`。

#### 运行说明

本章固定放在文件末尾。凡代码修改影响命令、参数、输出文件或输出格式时，必须在同一轮迭代中更新本章。

#### 环境准备

```bash
pip install -r requirements.txt
```

- 安装当前项目依赖。
- 当前 `requirements.txt` 包含 PyTorch 系依赖，因为当前 README 说明这些依赖是必需的。
- 若使用 CUDA，建议先按 PyTorch 官网命令安装匹配 CUDA 版本的 PyTorch，再安装其余依赖。



#### 创建输出目录

```bash
python scripts/mkdir_outputs.py --config configs/v10a.yaml
```

- 读取 config 中的 `train.output_version`。
- 创建：
  - `outputs/checkpoints/`
  - `outputs/outputs_v10a/figures/`
  - `outputs/outputs_v10a/logs/`
  - `outputs/outputs_v10a/tables/`



#### 主训练

```bash
python -u scripts/train.py --config configs/v10a.yaml
```

- 加载 MNIST + FSDD。
- 计算或加载 audio norm stats。
- 构建 `CrossModalSNN`。
- 除非跳过或 resume，否则先执行 decoder pretraining。
- 按 `train.epochs` 执行 binding/readout 训练。
- 保存 checkpoint 到 `outputs/checkpoints/cross_modal_snn_v10a.pt`。

短训练：

```bash
python -u scripts/train.py --config configs/v10a.yaml --epochs 30
```

恢复训练：

```bash
python -u scripts/train.py --config configs/v10a.yaml --resume
```

跳过 decoder pretraining：

```bash
python -u scripts/train.py --config configs/v10a.yaml --skip_decoder_pretrain
```



#### 后台训练并写日志

```bash
nohup env PYTHONUNBUFFERED=1 python -u scripts/train.py --config configs/v10a.yaml > outputs/outputs_v10a/logs/train_v10a_50ep.log 2>&1 < /dev/null &
```

- 运行前需要先创建输出目录。
- stdout/stderr 写入 `outputs/outputs_v10a/logs/train_v10a_50ep.log`。



#### 主训练加消融 suite

```bash
python -u scripts/run_v10a_suite.py --config configs/v10a.yaml --with_ablations
```

- 先跑主训练。
- 在 `outputs/ablations_v10a/configs/` 下生成三个消融配置。
- 顺序运行各消融。
- 任一实验失败时停止。

只跑消融：

```bash
python -u scripts/run_v10a_suite.py --config configs/v10a.yaml --ablations_only
```



#### 评估

```bash
python -u scripts/evaluate.py --config configs/v10a.yaml --protocol fixed_mask --family_breakdown
```

- 评估 6 种 cue mode。
- 使用确定性 fixed corruption masks 和配置中的 fixed families。
- 输出 accuracy、image MSE、PSNR、SSIM、audio MSE、多样性诊断和 target 粒度。
- 加 `--family_breakdown` 时，输出 `outputs/outputs_v10a/tables/audio_family_breakdown_fixed.csv`。

随机协议鲁棒性检查：

```bash
python -u scripts/evaluate.py --config configs/v10a.yaml --protocol legacy_random
```

快速评估：

```bash
python -u scripts/evaluate.py --config configs/v10a.yaml --protocol fixed_mask --max_batches 5
```



#### Demo 图

```bash
python -u scripts/demo_inference.py --config configs/v10a.yaml --num 10 --severity 0.5
```

- 生成 fixed-mask demo 图和文本指标表。
- 默认输出：
  - `outputs/outputs_v10a/figures/demo_aud_only.png`
  - `outputs/outputs_v10a/figures/demo_img_only.png`
  - `outputs/outputs_v10a/figures/demo_both.png`
  - `outputs/outputs_v10a/tables/demo_eval_table.txt`

随机 family demo：

```bash
python -u scripts/demo_inference.py --config configs/v10a.yaml --num 10 --severity 0.5 --protocol legacy_random
```



#### 渲染评估表

```bash
python scripts/plot_eval_summary.py outputs/outputs_v10a/tables/demo_eval_table.txt
```

- 解析 demo table 文本。
- 在版本化输出目录附近生成 PNG 和 CSV 汇总表。

解析完整评估日志：

```bash
python scripts/plot_eval_summary.py outputs/outputs_v10a/logs/eval_v10a_fixed_mask.log
```



#### Smoke Test

```bash
python -u scripts/smoke_test.py
```

- 只使用随机张量。
- 检查 6 种 cue mode 的 binding/readout 前向和反向。
- 检查 corruption 函数。
- 检查 audio-only 和 image-only 推理。
- 不需要下载 MNIST 或 FSDD。



#### v10b 运行命令

创建 v10b 输出目录：

```bash
python scripts/mkdir_outputs.py --config configs/v10b.yaml
```

训练 v10b：

```bash
python -u scripts/train.py --config configs/v10b.yaml
```

后台训练并写日志：

```bash
nohup env PYTHONUNBUFFERED=1 python -u scripts/train.py --config configs/v10b.yaml > outputs/outputs_v10b/logs/train_v10b_50ep.log 2>&1 < /dev/null &
```

fixed-mask 论文对齐评估：

```bash
python -u scripts/evaluate.py --config configs/v10b.yaml --protocol fixed_mask --family_breakdown
```

快速评估：

```bash
python -u scripts/evaluate.py --config configs/v10b.yaml --protocol fixed_mask --max_batches 5
```

v10b 默认输出：

- checkpoint：`outputs/checkpoints/cross_modal_snn_v10b.pt`
- decoder pretrain checkpoint：`outputs/checkpoints/cross_modal_snn_v10b_decoder_pretrain.pt`
- 版本化输出目录：`outputs/outputs_v10b/`
- family breakdown：`outputs/outputs_v10b/tables/audio_family_breakdown_fixed.csv`



#### v10c 运行命令

创建 v10c 输出目录：

```bash
python scripts/mkdir_outputs.py --config configs/v10c.yaml
```

训练 v10c：

```bash
python -u scripts/train.py --config configs/v10c.yaml
```

后台训练并写日志：

```bash
nohup env PYTHONUNBUFFERED=1 python -u scripts/train.py --config configs/v10c.yaml > outputs/outputs_v10c/logs/train_v10c_70ep.log 2>&1 < /dev/null &
```

fixed-mask 主评估：

```bash
python -u scripts/evaluate.py --config configs/v10c.yaml --protocol fixed_mask --family_breakdown > outputs/outputs_v10c/logs/eval_v10c_fixed_mask.log 2>&1
```

快速评估：

```bash
python -u scripts/evaluate.py --config configs/v10c.yaml --protocol fixed_mask --max_batches 5
```

v10c demo：

```bash
python -u scripts/demo_inference.py --config configs/v10c.yaml --num 8 --severity 0.4
```

v10c 默认输出：

- checkpoint：`outputs/checkpoints/cross_modal_snn_v10c.pt`
- decoder pretrain checkpoint：`outputs/checkpoints/cross_modal_snn_v10c_decoder_pretrain.pt`
- 版本化输出目录：`outputs/outputs_v10c/`
- family breakdown：`outputs/outputs_v10c/tables/audio_family_breakdown_fixed.csv`



#### v10d 运行命令

创建 v10d 输出目录：

```bash
python scripts/mkdir_outputs.py --config configs/v10d.yaml
```

训练 v10d（含 gated/dilated decoder + 谱图空间 refiner）：

```bash
python -u scripts/train.py --config configs/v10d.yaml
```

后台训练并写日志：

```bash
nohup env PYTHONUNBUFFERED=1 python -u scripts/train.py --config configs/v10d.yaml > outputs/outputs_v10d/logs/train_v10d_70ep.log 2>&1 < /dev/null &
```

fixed-mask 主评估（refiner 与训练一致启用）：

```bash
python -u scripts/evaluate.py --config configs/v10d.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown \
  2>&1 | tee outputs/outputs_v10d/logs/eval_v10d_fixed_mask_sev04.log
```

快速验证：

```bash
python -u scripts/smoke_test.py
python -u scripts/evaluate.py --config configs/v10d.yaml --protocol fixed_mask --max_batches 5
```

v10d demo（已与 evaluate 对齐，展示 audio mask / coarse audio / final recovered audio）：

```bash
python -u scripts/demo_inference.py --config configs/v10d.yaml --num 8 --severity 0.4
```

v10d 默认输出：

- checkpoint：`outputs/checkpoints/cross_modal_snn_v10d.pt`
- decoder pretrain checkpoint：`outputs/checkpoints/cross_modal_snn_v10d_decoder_pretrain.pt`
- 版本化输出目录：`outputs/outputs_v10d/`
- family breakdown：`outputs/outputs_v10d/tables/audio_family_breakdown_fixed.csv`

消融开关：`configs/v10d.yaml` 中 `snn.aud_refine_type`（plain|gated_dilated）、`audio_refiner.enabled`（true|false）、`loss.lambda_aud_ssim` / `lambda_aud_masked_grad` / `lambda_aud_feat`（默认 0）可单独开关做消融。

#### v10e 运行命令

创建 v10e 输出目录：

```bash
python scripts/mkdir_outputs.py --config configs/v10e.yaml
```

训练 v10e（对称 image/audio refiner + paste-back）：

```bash
python -u scripts/train.py --config configs/v10e.yaml
```

后台训练并写日志：

```bash
nohup env PYTHONUNBUFFERED=1 python -u scripts/train.py --config configs/v10e.yaml > outputs/outputs_v10e/logs/train_v10e_70ep.log 2>&1 < /dev/null &
```

fixed-mask 主评估：

```bash
python -u scripts/evaluate.py --config configs/v10e.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown \
  2>&1 | tee outputs/outputs_v10e/logs/eval_v10e_fixed_mask_sev04.log
```

快速验证：

```bash
python -u scripts/evaluate.py --config configs/v10e.yaml --protocol fixed_mask --max_batches 5
```

v10e demo：

```bash
python -u scripts/demo_inference.py --config configs/v10e.yaml --num 8 --severity 0.4
```

v10e 默认输出：

- checkpoint：`outputs/checkpoints/cross_modal_snn_v10e.pt`
- decoder pretrain checkpoint：`outputs/checkpoints/cross_modal_snn_v10e_decoder_pretrain.pt`
- 版本化输出目录：`outputs/outputs_v10e/`



#### v10f 运行命令

创建 v10f 输出目录：

```bash
python scripts/mkdir_outputs.py --config configs/v10f.yaml
```

训练 v10f（refiner-aware pretrain + lr_mult + 5-family corruption + 归因评估）：

```bash
python -u scripts/train.py --config configs/v10f.yaml
```

后台训练并写日志：

```bash
nohup env PYTHONUNBUFFERED=1 python -u scripts/train.py --config configs/v10f.yaml > outputs/outputs_v10f/logs/train_v10f_70ep.log 2>&1 < /dev/null &
```

fixed-mask 主评估（按 5 个 image/audio family pair 依次评估，含 `[归因]` coarse/final 表）：

```bash
python -u scripts/evaluate.py --config configs/v10f.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown \
  2>&1 | tee outputs/outputs_v10f/logs/eval_v10f_fixed_mask_sev04.log
```

v10f demo（默认 10 个样本；5 个 family 各 2 个样本；无 clean fallback）：

```bash
python -u scripts/demo_inference.py --config configs/v10f.yaml --num 10 --severity 0.4
```

归因消融（示例）：

```bash
python -u scripts/train.py --config configs/v10f_ab_audio_pasteback_only.yaml
python -u scripts/train.py --config configs/v10f_ab_no_refiner_pretrain.yaml
```

渲染评估表（含归因表）：

```bash
python scripts/plot_eval_summary.py outputs/outputs_v10f/logs/eval_v10f_fixed_mask_sev04.log
```

P1/P2 修正版检查：

- `v10f_ab_no_refiner_pretrain` 的 decoder pretrain 日志应包含 `train_image_refiner=False train_audio_refiner=False img_final_path=coarse aud_final_path=coarse`。
- `v10f.yaml` 主实验的 decoder pretrain 日志应包含 `train_image_refiner=True train_audio_refiner=True img_final_path=helper aud_final_path=helper`。
- `plot_eval_summary.py` 生成的 attribution CSV/PNG 应包含 masked 与 visible 两组 coarse/final MSE。

v10f 默认输出：

- checkpoint：`outputs/checkpoints/cross_modal_snn_v10f.pt`
- decoder pretrain checkpoint：`outputs/checkpoints/cross_modal_snn_v10f_decoder_pretrain.pt`
- 版本化输出目录：`outputs/outputs_v10f/`
- 消融配置：`configs/v10f_ab_*.yaml`（各自独立 ckpt 路径）



#### v11a 运行命令

创建输出目录：

```bash
python scripts/mkdir_outputs.py --config configs/v11a.yaml
```

后台训练 120 轮（最后 25 轮自动进入 time-mask-heavy 阶段）：

```bash
nohup env OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 python -u scripts/train.py \
  --config configs/v11a.yaml \
  > outputs/outputs_v11a/logs/train_v11a_120ep.log 2>&1 < /dev/null &
```

同预算 control（同构 Cross-Key 模块保留、条件路径关闭）：

```bash
python scripts/mkdir_outputs.py --config configs/v11a_control.yaml
nohup env OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 python -u scripts/train.py \
  --config configs/v11a_control.yaml \
  > outputs/outputs_v11a_control/logs/train_v11a_control_120ep.log 2>&1 < /dev/null &
```

fixed-mask 主评估（8 种 cue、5 个 family pair、含 clean-assist 对照）：

```bash
python -u scripts/evaluate.py --config configs/v11a.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown \
  2>&1 | tee outputs/outputs_v11a/logs/eval_v11a_fixed_mask_sev04.log
```

对侧 Key 配对归因（normal/zero/wrong-class，共享同一 cue 与 mask）：

```bash
python -u scripts/evaluate.py --config configs/v11a.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown --cross_key sweep \
  2>&1 | tee outputs/outputs_v11a/logs/eval_v11a_cross_key_sweep.log
python scripts/plot_eval_summary.py \
  outputs/outputs_v11a/logs/eval_v11a_cross_key_sweep.log
```

快速验证：

```bash
python -u scripts/smoke_test.py
python -u scripts/evaluate.py --config configs/v11a.yaml \
  --protocol fixed_mask --severity 0.4 --max_batches 1 --cross_key sweep
```

v11a 默认输出：

- checkpoint：`outputs/checkpoints/cross_modal_snn_v11a.pt`
- decoder pretrain checkpoint：`outputs/checkpoints/cross_modal_snn_v11a_decoder_pretrain.pt`
- 版本化输出目录：`outputs/outputs_v11a/`
- control checkpoint：`outputs/checkpoints/cross_modal_snn_v11a_control.pt`
- control decoder pretrain checkpoint：`outputs/checkpoints/cross_modal_snn_v11a_control_decoder_pretrain.pt`
- control 输出目录：`outputs/outputs_v11a_control/`


#### v11c 运行命令

v11b 五组实验已完成并归档，其 YAML 已删除。v11c 仅保留同父权重、
同 seed 和同 30 轮预算的 control 与 causal-main。两者都依赖已有
`outputs/checkpoints/cross_modal_snn_v11b_recovery.pt`。

顺序后台训练 control -> causal-main：

```bash
PY="$(command -v python)" && mkdir -p outputs && \
nohup env PY="$PY" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 bash -c '
set -euo pipefail

run_exp() {
  cfg="$1"
  out="$2"
  log_name="$3"
  echo "[start] $cfg $(date)"
  "$PY" scripts/mkdir_outputs.py --config "$cfg"
  "$PY" -u scripts/train.py --config "$cfg" \
    > "$out/logs/$log_name" 2>&1
  echo "[done]  $cfg $(date)"
}

run_exp configs/v11c_control.yaml \
  outputs/outputs_v11c_control train_v11c_control_ep120_to_150.log
run_exp configs/v11c.yaml \
  outputs/outputs_v11c train_v11c_causal_ep120_to_150.log

echo "[done] both v11c experiments $(date)"
' > outputs/train_v11c_suite.log 2>&1 < /dev/null &
```

主实验与 control 的 fixed-mask、Cross-Key sweep 与 demo：

```bash
python -u scripts/evaluate.py --config configs/v11c_control.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown \
  2>&1 | tee outputs/outputs_v11c_control/logs/eval_v11c_control_fixed_mask_sev04.log

python -u scripts/evaluate.py --config configs/v11c.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown \
  2>&1 | tee outputs/outputs_v11c/logs/eval_v11c_fixed_mask_sev04.log

python -u scripts/evaluate.py --config configs/v11c.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown --cross_key sweep \
  2>&1 | tee outputs/outputs_v11c/logs/eval_v11c_cross_key_sweep_sev04.log

python -u scripts/demo_inference.py --config configs/v11c.yaml \
  --num 10 --severity 0.4
```

快速验证：

```bash
python -u scripts/smoke_test.py
python -u scripts/evaluate.py --config configs/v11c.yaml \
  --protocol fixed_mask --severity 0.4 --max_batches 1 --cross_key sweep
```

#### v11d / v11e 远端分支评价命令索引

当前本地工作区位于 `v11c`，`configs/v11d*.yaml` 与 `configs/v11e*.yaml`
存在于远端分支 `origin/v11d`、`origin/v11e`。运行这些实验前需先检出对应
分支，或在新的工作目录中 clone/checkout，避免与当前 v11c 文件集混用。

v11d 前置条件：

- 需要 `outputs/checkpoints/cross_modal_snn_v11c.pt`。
- `v11d_control` 与 `v11d` 均从该 checkpoint 继续，区别是是否启用
  Cross-Detail。
- v11d 是 MNIST/FSDD 固定一一伪配对，不是 GRID 真实同源配对。

```bash
git fetch origin
git checkout v11d

python scripts/mkdir_outputs.py --config configs/v11d_control.yaml
python -u scripts/train.py --config configs/v11d_control.yaml

python scripts/mkdir_outputs.py --config configs/v11d.yaml
python -u scripts/train.py --config configs/v11d.yaml

python -u scripts/evaluate.py --config configs/v11d_control.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown \
  2>&1 | tee outputs/outputs_v11d_control/logs/eval_v11d_control_fixed_mask_sev04.log

python -u scripts/evaluate.py --config configs/v11d.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown \
  2>&1 | tee outputs/outputs_v11d/logs/eval_v11d_fixed_mask_sev04.log

python -u scripts/evaluate.py --config configs/v11d.yaml \
  --protocol fixed_mask --severity 0.4 --cross_detail sweep \
  2>&1 | tee outputs/outputs_v11d/logs/eval_v11d_cross_detail_sweep_sev04.log
```

v11e 前置条件：

- 使用 MNIST + FSDD；不再准备 GRID manifest。
- FSDD wav 位于 `_data/fsdd/recordings/`，MNIST 可由 torchvision 自动下载。
- v11e 从头训练，不继承 v11c/v11d checkpoint。
- main 启用 Cross-Key；control 只关闭 Cross-Key。单模态缺失侧使用 train class
  medoid，双模态使用 sample/sample。

```bash
git fetch origin
git checkout v11e

python -u scripts/smoke_test_v11e.py

python scripts/mkdir_outputs.py --config configs/v11e_control.yaml
python -u scripts/train.py --config configs/v11e_control.yaml

python scripts/mkdir_outputs.py --config configs/v11e.yaml
python -u scripts/train.py --config configs/v11e.yaml

python -u scripts/evaluate.py --config configs/v11e_control.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown \
  2>&1 | tee outputs/outputs_v11e_control/logs/eval_v11e_control_fixed_mask_sev04.log

python -u scripts/evaluate.py --config configs/v11e.yaml \
  --protocol fixed_mask --severity 0.4 --family_breakdown \
  2>&1 | tee outputs/outputs_v11e/logs/eval_v11e_fixed_mask_sev04.log

python -u scripts/evaluate.py --config configs/v11e.yaml \
  --protocol fixed_mask --severity 0.4 --cross_key sweep \
  2>&1 | tee outputs/outputs_v11e/logs/eval_v11e_cross_key_sweep_sev04.log

python -u scripts/demo_inference.py --config configs/v11e.yaml \
  --num 10 --severity 0.4

python -u scripts/demo_inference.py --config configs/v11e.yaml \
  --num 10 --severity 0.4 --protocol legacy_random
```

### 2026-09-10 v11f 本地实测评价与 checkpoint 上传

#### 范围与证据

用户要求上传 v11f checkpoint 并详细评价本地结果。本次仅做 F-1 诊断与归档，
没有修改模型、训练配置或评估代码，也没有重跑全量 GPU 训练/推理。
来源是 `v11f_outputs_with_ckpt/outputs/` 下 main、control、no_causal 三组产物：
12 份 normal/sweep CSV、3 份 family breakdown CSV、6 份 demo 表、18 张图及
20 份日志。完整口径、全部方向和限制见 `docs/V11F_EVALUATION.md`。

#### 主模型结论

- v11f 从 v11e_control epoch=99 初始化，seed=1234、batch=128，固定 severity=0.4，
  五 family 均衡采样，额外 30 轮。checkpoint epoch=29，scheduler last_epoch=30。
- 48,736 个 adapter 参数可训练；所有父模型参数/buffers 逐张量相同，数值有限。
  基础 state SHA256 与 provenance 一致，Index ACC 和可见区域误差不变。
- fixed 四个双模态部分残缺方向的 masked MSE 降幅分别为：干净图像助音频 4.45%、
  双残缺音频 3.33%、干净音频助图像 2.55%、双残缺图像 2.16%；random 分别为
  4.47%、3.32%、2.98%、2.19%。这是相对同输入/mask 的固定 zero/control。
- 但 fixed 的上述场景同时胜过 zero/wrong 的样本只占 35.75%–38.22%，错误 Key
  仍保留大部分平均改善。额外 decoder 支路的类别选择性仍弱；不代表原有
  Key -> Index -> Value 完全没有跨模态联系。
- 干净/残缺图像生成类别音频的 MSE 分别退化约 1.68%/0.28%；部分残缺音频的
  内容类别一致性未稳定提升。partial_temporal 仍最弱，不能宣称全部目标达成。

#### 对照结论：v11f_control

- 消融对象：关闭额外 Cross-Key 的固定父模型；无额外训练，不是等预算重训对照。
- 数据与评估：同 seed、severity=0.4，fixed 五 family、random seed=4321。
- 核验：两协议中 main/no_causal 的 paired zero 方向误差与 control 一致，
  normal 与 sweep 的共有主指标也一致（数值比较容差 1e-9）。
- 独立结论：固定父模型比较有效，v11f 的部分残缺收益并非换基线或基础参数漂移。
  不能借此作“局部结构因果优于 v11e 全局结构”的等预算结论。

#### 消融结论：v11f_no_causal

- 消融对象：仅去掉因果项开关/权重，保留两个 masked-feature adapter。
- 对照与预算：同一 v11e_control 父模型、seed=1234、batch=128、额外 30 轮，
  saved cfg 除因果项和路径外一致，optimizer 参数更新步数也一致。
- fixed 部分残缺 masked MSE 降幅为 3.05%、2.21%、1.76%、1.40%；main 为
  4.45%、3.33%、2.55%、2.16%。main 在四方向的全部 20 个 family 任务格都更低。
- 代价：no_causal 的音频 SSIM 和部分音频内容类别一致性反而略好；残缺音频
  生成类别图像时，MSE 改善 6.85%，优于 main 的 3.55%，random 也有同向差异。
- 独立结论：因果项增加了部分残缺正向 MSE 收益，但没有全面改善类别选择性
  或全部恢复指标。wrong reference 不反传，其梯度更直接地加权正常重建误差。
  单训练 seed、无逐样本误差分布和独立重复，不作统计显著性结论。

#### 异常与比较限制

- 所有日志有非法 OMP_NUM_THREADS 警告，但训练/评估结束，未发现 CUDA/NaN/Inf
  报错；不能据此认定结果无效或推断具体耗时影响。
- family breakdown 使用独立 seed 调度和固定 occlusion 图像，不能与主 sweep
  混为同一 mask。partial_temporal audio-only ACC=45.88% 来自该独立 breakdown。
- demo 的 random 分支仍默认 occlusion/time_mask 的随机位置；数值 random 才是
  随机五 family。已在 implementation 和评价报告标注，本轮未改代码/重画图片。
- Index ACC 与恢复内容再分类分开；后者是内部代理，不是外部独立识别或听感。
- 类别绑定下不存在可由 MNIST 唯一确定的说话人/语速实例；全缺失 target 为训练
  medoid，不能用其低 MSE 冒充逐实例跨模态恢复。

#### 上传与归档

- 独立 checkpoint 仓库 main 提交 `4ae02eb9e38b11bce5fd80e51f6a75756835813c`，
  Git LFS 上传成功，远端分支确认，LFS fsck 通过。
- 新增 `cross_modal_snn_v11f.pt`：SHA256
  `9b3c2a6fc47f6cb890c2cfd0a87564ce612e5b8424ef343607da8dfef1d95fc8`。
- 新增 `cross_modal_snn_v11f_no_causal.pt`：SHA256
  `3951fadc830c0583a73e93340bc1906da5c7578544d478680fec990ace01359f`。
- bundle 的 v11e_control 与仓库现有文件哈希相同，不重复覆盖。
- 本轮同步 `V11F_EVALUATION.md`、`idea_report.md`、`implementation.md`、
  `V11F_MASKED_CROSS_KEY_PROTOCOL.md` 和本日志；不把原始产物或临时分析脚本
  加入代码仓库。没有进入新版本设计/实现。

#### 运行说明

本次评价读取已存在的 `v11f_outputs_with_ckpt/outputs/`，无需重训。
详细数字、来源 CSV、逐配置结论和限制见 `docs/V11F_EVALUATION.md`。
要在服务器复跑评估，须先保留原始结果，避免下面套件覆盖同名日志和表。
在 `/root/autodl-tmp/projects/cross_modal_attractor_snn_v11f` 项目根目录与已激活的
GPU 环境中，可用：

```bash
OMP_NUM_THREADS=1 python -u scripts/run_v11f_suite.py --eval_only --with_ablations
```

此命令只评估已有 main/control/no_causal 权重，不继续训练。需要
`outputs/checkpoints/cross_modal_snn_v11f.pt`、`cross_modal_snn_v11f_no_causal.pt`
以及正确 SHA256 的父权重 `cross_modal_snn_v11e_control.pt`。
当前 random demo 的 family 限制仍然存在，不因重跑命令自动修复。

### 2026-09-10 评估指标交付规范补充

用户要求后续每次评价均告知各实验的各项指标。本轮只补文档规范，不改变模型、
配置、损失实现或服务器任务，也没有提交/推送代码。

- 在 `docs/user_requirements.md` 增加硬性规则 12 及“评估报告强制规范”，
  覆盖所有后续版本的主实验、control、各消融和可用评估协议。
- 要求用户回复包含逐实验主要指标表，完整汇总覆盖原始字段、绝对数值、有效 n、
  配对干预、区域误差、内容分类、音频诊断、训练曲线和 demo；缺失项明确说明。
- fixed/random、family breakdown、小样本 demo 不混算，sample/category target
  不混同，不能只给提升百分比或只给主实验指标。
- `docs/implementation.md` 增加规范入口；当前完整指标表为
  `docs/V11F_METRICS_SUMMARY.md`，评价与消融结论见 `docs/V11F_EVALUATION.md`。
- 重新核对 v11f 的 `_cross_key_causal_loss` 及 YAML，确认 reference 不反传、
  margin_ratio=0.05、权重=0.5、图/音 scale floor=0.01/0.005。后续损失修订、
  内容语义监督和受控 Key 对照仅为讨论建议，尚未获得实现授权。

#### 运行说明

本轮仅修改规范，无需重新运行训练或评估，现有命令与 checkpoint 不变。
后续评价前先读取 `docs/user_requirements.md` 的“评估报告强制规范”，按实际
产物字段清点并生成逐实验完整表，再追加各消融的独立结论。
当前 v11f 指标入口：`docs/V11F_METRICS_SUMMARY.md`；解读入口：
`docs/V11F_EVALUATION.md`。服务器复核命令见本日志上一段运行说明；原始结果
应保留，random demo 口径问题尚未在代码中修复。

### 2026-09-10 v11f 结果归档格式修订

用户明确要求：v11f 评估结果不要写在单独文件。此前新增的
`V11F_EVALUATION.md` 与 `V11F_METRICS_SUMMARY.md` 属于重复的结果说明文档，
现将结果统一归档到本 `dev_log.md` 条目；`outputs/` 下的原始 CSV、日志和图片
继续保留，作为可复核的运行证据。后续版本不得再创建同类独立评估 Markdown。
此前条目中指向两个文件的路径属于历史记录，本条目是新的正式入口。

#### v11f 三组实验口径

| 实验 | 权重来源 | 额外训练 | 可训练范围 | seed | batch | severity |
|---|---|---:|---|---:|---:|---:|
| control | v11e_control | 0 轮，仅评估 | 无 | 1234 | 128 | 0.4 |
| main | v11e_control | 30 轮 | 两个 masked Cross-Key adapter | 1234 | 128 | 0.4 |
| no_causal | v11e_control | 30 轮 | 同上，关闭 causal loss | 1234 | 128 | 0.4 |

fixed 使用五个 family 的均衡汇总；random 使用 seed=4321 的一次可复现
family/mask 抽样。下表的主指标均为有效汇总值，主要 normal 指标每格
`n_sum=50000`；这表示重复评估曝光数，不是独立样本数。图像/音频内容 ACC
是恢复结果经过冻结原模型单模态再分类得到的内部一致性代理，不是独立识别器。
MSE 越低越好，SSIM 和 ACC 越高越好；mask MSE 只在对应缺失区域存在时适用。

#### fixed_mask 主指标

| cue | 实验 | Index ACC | img MSE | img SSIM | aud MSE | aud SSIM | img mask MSE | aud mask MSE | img content ACC | aud content ACC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean_both | control | 99.180% | 0.00973233 | 0.942685 | 0.00364344 | 0.911253 | N/A | N/A | 97.790% | 96.040% |
| clean_both | main | 99.180% | 0.00973233 | 0.942685 | 0.00364344 | 0.911253 | N/A | N/A | 97.790% | 96.040% |
| clean_both | no_causal | 99.180% | 0.00973233 | 0.942685 | 0.00364344 | 0.911253 | N/A | N/A | 97.790% | 96.040% |
| clean_img_only | control | 97.700% | 0.00977739 | 0.942646 | 0.00028921 | 0.965964 | N/A | 0.00028921 | 96.820% | 94.320% |
| clean_img_only | main | 97.700% | 0.00977739 | 0.942646 | 0.00029406 | 0.965814 | N/A | 0.00029406 | 96.820% | 94.250% |
| clean_img_only | no_causal | 97.700% | 0.00977739 | 0.942646 | 0.00029571 | 0.966055 | N/A | 0.00029571 | 96.820% | 94.550% |
| clean_aud_only | control | 98.830% | 0.00208493 | 0.985595 | 0.00364809 | 0.911157 | 0.00208493 | N/A | 97.710% | 96.000% |
| clean_aud_only | main | 98.830% | 0.00205180 | 0.985961 | 0.00364809 | 0.911157 | 0.00205180 | N/A | 98.090% | 96.000% |
| clean_aud_only | no_causal | 98.830% | 0.00207669 | 0.985706 | 0.00364809 | 0.911157 | 0.00207669 | N/A | 97.750% | 96.000% |
| corrupt_img_only | control | 92.028% | 0.00768047 | 0.955151 | 0.00066839 | 0.916781 | 0.02666188 | 0.00066839 | 95.514% | 83.518% |
| corrupt_img_only | main | 92.028% | 0.00768047 | 0.955151 | 0.00067026 | 0.916379 | 0.02666188 | 0.00067026 | 95.514% | 83.526% |
| corrupt_img_only | no_causal | 92.028% | 0.00768047 | 0.955151 | 0.00067341 | 0.917347 | 0.02666188 | 0.00067341 | 95.514% | 84.174% |
| corrupt_aud_only | control | 84.572% | 0.01321163 | 0.877703 | 0.00628970 | 0.804689 | 0.01321163 | 0.01107898 | 82.882% | 80.600% |
| corrupt_aud_only | main | 84.572% | 0.01274271 | 0.886050 | 0.00628970 | 0.804689 | 0.01274271 | 0.01107898 | 82.986% | 80.600% |
| corrupt_aud_only | no_causal | 84.572% | 0.01230638 | 0.893063 | 0.00628970 | 0.804689 | 0.01230638 | 0.01107898 | 82.846% | 80.600% |
| clean_img_corrupt_aud | control | 98.934% | 0.00979025 | 0.942597 | 0.00609738 | 0.807590 | N/A | 0.01059938 | 97.520% | 88.970% |
| clean_img_corrupt_aud | main | 98.934% | 0.00979025 | 0.942597 | 0.00592129 | 0.809909 | N/A | 0.01012738 | 97.520% | 88.918% |
| clean_img_corrupt_aud | no_causal | 98.934% | 0.00979025 | 0.942597 | 0.00597777 | 0.810239 | N/A | 0.01027564 | 97.520% | 89.142% |
| corrupt_img_clean_aud | control | 99.144% | 0.00750159 | 0.956305 | 0.00364682 | 0.911207 | 0.02591803 | N/A | 96.720% | 95.968% |
| corrupt_img_clean_aud | main | 99.144% | 0.00731868 | 0.957252 | 0.00364682 | 0.911207 | 0.02525703 | N/A | 96.736% | 95.968% |
| corrupt_img_clean_aud | no_causal | 99.144% | 0.00739533 | 0.956837 | 0.00364682 | 0.911207 | 0.02546277 | N/A | 96.754% | 95.968% |
| corrupt_both | control | 97.536% | 0.00751963 | 0.956189 | 0.00617585 | 0.805881 | 0.02591245 | 0.01077483 | 96.178% | 87.012% |
| corrupt_both | main | 97.536% | 0.00737009 | 0.957024 | 0.00604091 | 0.807620 | 0.02535232 | 0.01041650 | 96.278% | 87.012% |
| corrupt_both | no_causal | 97.536% | 0.00743799 | 0.956651 | 0.00608690 | 0.807978 | 0.02554904 | 0.01053662 | 96.296% | 87.134% |

#### legacy_random 主指标

| cue | 实验 | Index ACC | img MSE | img SSIM | aud MSE | aud SSIM | img mask MSE | aud mask MSE | img content ACC | aud content ACC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean_both | control | 99.180% | 0.00973233 | 0.942685 | 0.00364344 | 0.911253 | N/A | N/A | 97.790% | 96.040% |
| clean_both | main | 99.180% | 0.00973233 | 0.942685 | 0.00364344 | 0.911253 | N/A | N/A | 97.790% | 96.040% |
| clean_both | no_causal | 99.180% | 0.00973233 | 0.942685 | 0.00364344 | 0.911253 | N/A | N/A | 97.790% | 96.040% |
| clean_img_only | control | 97.700% | 0.00977739 | 0.942646 | 0.00028921 | 0.965964 | N/A | 0.00028921 | 96.820% | 94.320% |
| clean_img_only | main | 97.700% | 0.00977739 | 0.942646 | 0.00029406 | 0.965814 | N/A | 0.00029406 | 96.820% | 94.250% |
| clean_img_only | no_causal | 97.700% | 0.00977739 | 0.942646 | 0.00029571 | 0.966055 | N/A | 0.00029571 | 96.820% | 94.550% |
| clean_aud_only | control | 98.830% | 0.00208493 | 0.985595 | 0.00364809 | 0.911157 | 0.00208493 | N/A | 97.710% | 96.000% |
| clean_aud_only | main | 98.830% | 0.00205180 | 0.985961 | 0.00364809 | 0.911157 | 0.00205180 | N/A | 98.090% | 96.000% |
| clean_aud_only | no_causal | 98.830% | 0.00207669 | 0.985706 | 0.00364809 | 0.911157 | 0.00207669 | N/A | 97.750% | 96.000% |
| corrupt_img_only | control | 92.360% | 0.00761429 | 0.955219 | 0.00062749 | 0.921694 | 0.02923265 | 0.00062749 | 95.490% | 84.270% |
| corrupt_img_only | main | 92.360% | 0.00761429 | 0.955219 | 0.00062952 | 0.921340 | 0.02923265 | 0.00062952 | 95.490% | 84.480% |
| corrupt_img_only | no_causal | 92.360% | 0.00761429 | 0.955219 | 0.00063137 | 0.922253 | 0.02923265 | 0.00063137 | 95.490% | 85.160% |
| corrupt_aud_only | control | 85.790% | 0.01220869 | 0.887254 | 0.00605548 | 0.813113 | 0.01220869 | 0.01038064 | 84.090% | 81.770% |
| corrupt_aud_only | main | 85.790% | 0.01179923 | 0.894696 | 0.00605548 | 0.813113 | 0.01179923 | 0.01038064 | 84.250% | 81.770% |
| corrupt_aud_only | no_causal | 85.790% | 0.01141631 | 0.900913 | 0.00605548 | 0.813113 | 0.01141631 | 0.01038064 | 84.010% | 81.770% |
| clean_img_corrupt_aud | control | 98.960% | 0.00978341 | 0.942602 | 0.00587250 | 0.816051 | N/A | 0.00992893 | 97.500% | 89.830% |
| clean_img_corrupt_aud | main | 98.960% | 0.00978341 | 0.942602 | 0.00570647 | 0.818213 | N/A | 0.00948503 | 97.500% | 89.720% |
| clean_img_corrupt_aud | no_causal | 98.960% | 0.00978341 | 0.942602 | 0.00576489 | 0.818253 | N/A | 0.00963809 | 97.500% | 90.060% |
| corrupt_img_clean_aud | control | 99.160% | 0.00741694 | 0.956508 | 0.00364718 | 0.911193 | 0.02826212 | N/A | 96.690% | 96.000% |
| corrupt_img_clean_aud | main | 99.160% | 0.00720425 | 0.957629 | 0.00364718 | 0.911193 | 0.02742097 | N/A | 96.720% | 96.000% |
| corrupt_img_clean_aud | no_causal | 99.160% | 0.00727446 | 0.957239 | 0.00364718 | 0.911193 | 0.02759958 | N/A | 96.780% | 96.000% |
| corrupt_both | control | 97.280% | 0.00771856 | 0.954641 | 0.00592670 | 0.820633 | 0.02618422 | 0.00991035 | 96.340% | 87.310% |
| corrupt_both | main | 97.280% | 0.00756339 | 0.955451 | 0.00580403 | 0.822283 | 0.02561142 | 0.00958097 | 96.390% | 87.530% |
| corrupt_both | no_causal | 97.280% | 0.00762250 | 0.955149 | 0.00584141 | 0.822596 | 0.02577683 | 0.00967829 | 96.380% | 87.670% |

#### Cross-Key 配对指标

以下只替换 decoder 额外支路的 Key，Index/Value 保留原始输入；`zero` 是关闭
额外支路，`wrong` 是错误类别，`same-class` 是同类其它样本。`rel. gain` 为
相对 zero 的 MSE 降幅，`win_both` 为同一样本同时胜 zero 和 wrong 的比例。
control 没有可训练 adapter，所以 normal/zero/wrong 数值相同。

| 协议 | cue/方向 | 实验 | zero MSE | normal MSE | wrong MSE | same-class MSE | rel. gain | win_zero | win_wrong | win_both |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | clean_img_corrupt_aud/img2aud | control | 0.01059938 | 0.01059938 | 0.01059938 | 0.01060036 | 0.000% | 0.000% | 0.000% | 0.000% |
| fixed | clean_img_corrupt_aud/img2aud | main | 0.01059938 | 0.01012738 | 0.01014259 | 0.01012619 | 4.453% | 66.964% | 43.524% | 36.230% |
| fixed | clean_img_corrupt_aud/img2aud | no_causal | 0.01059938 | 0.01027564 | 0.01035881 | 0.01027439 | 3.054% | 54.962% | 45.200% | 32.946% |
| fixed | corrupt_both/img2aud | control | 0.01077483 | 0.01077483 | 0.01077483 | 0.01077508 | 0.000% | 0.000% | 0.000% | 0.000% |
| fixed | corrupt_both/img2aud | main | 0.01077483 | 0.01041650 | 0.01042322 | 0.01041735 | 3.326% | 67.572% | 42.914% | 35.748% |
| fixed | corrupt_both/img2aud | no_causal | 0.01077483 | 0.01053662 | 0.01058506 | 0.01053651 | 2.211% | 55.080% | 44.550% | 32.154% |
| fixed | corrupt_img_clean_aud/aud2img | control | 0.02591803 | 0.02591803 | 0.02591803 | 0.02591619 | 0.000% | 0.000% | 0.000% | 0.000% |
| fixed | corrupt_img_clean_aud/aud2img | main | 0.02591803 | 0.02525703 | 0.02530174 | 0.02525717 | 2.550% | 68.540% | 52.338% | 38.216% |
| fixed | corrupt_img_clean_aud/aud2img | no_causal | 0.02591803 | 0.02546277 | 0.02554611 | 0.02546369 | 1.757% | 61.064% | 52.492% | 35.954% |
| fixed | corrupt_both/aud2img | control | 0.02591245 | 0.02591245 | 0.02591245 | 0.02590695 | 0.000% | 0.000% | 0.000% | 0.000% |
| fixed | corrupt_both/aud2img | main | 0.02591245 | 0.02535232 | 0.02537906 | 0.02533928 | 2.162% | 67.618% | 49.846% | 36.596% |
| fixed | corrupt_both/aud2img | no_causal | 0.02591245 | 0.02554904 | 0.02560886 | 0.02553523 | 1.402% | 60.584% | 50.108% | 35.208% |
| random | clean_img_corrupt_aud/img2aud | control | 0.00992893 | 0.00992893 | 0.00992893 | 0.00992173 | 0.000% | 0.000% | 0.000% | 0.000% |
| random | clean_img_corrupt_aud/img2aud | main | 0.00992893 | 0.00948503 | 0.00949385 | 0.00947565 | 4.471% | 69.320% | 43.870% | 36.930% |
| random | clean_img_corrupt_aud/img2aud | no_causal | 0.00992893 | 0.00963809 | 0.00970579 | 0.00962232 | 2.929% | 56.320% | 44.900% | 32.520% |
| random | corrupt_both/img2aud | control | 0.00991035 | 0.00991035 | 0.00991035 | 0.00990307 | 0.000% | 0.000% | 0.000% | 0.000% |
| random | corrupt_both/img2aud | main | 0.00991035 | 0.00958097 | 0.00958426 | 0.00957182 | 3.324% | 66.230% | 41.860% | 35.290% |
| random | corrupt_both/img2aud | no_causal | 0.00991035 | 0.00967829 | 0.00971518 | 0.00966431 | 2.342% | 53.890% | 42.540% | 30.780% |
| random | corrupt_img_clean_aud/aud2img | control | 0.02826212 | 0.02826212 | 0.02826212 | 0.02824295 | 0.000% | 0.000% | 0.000% | 0.000% |
| random | corrupt_img_clean_aud/aud2img | main | 0.02826212 | 0.02742097 | 0.02743745 | 0.02740010 | 2.976% | 71.240% | 51.800% | 38.860% |
| random | corrupt_img_clean_aud/aud2img | no_causal | 0.02826212 | 0.02759958 | 0.02766430 | 0.02757850 | 2.344% | 63.980% | 52.830% | 37.240% |
| random | corrupt_both/aud2img | control | 0.02618422 | 0.02618422 | 0.02618422 | 0.02615921 | 0.000% | 0.000% | 0.000% | 0.000% |
| random | corrupt_both/aud2img | main | 0.02618422 | 0.02561142 | 0.02563057 | 0.02557729 | 2.188% | 67.280% | 49.940% | 36.370% |
| random | corrupt_both/aud2img | no_causal | 0.02618422 | 0.02577683 | 0.02583198 | 0.02574490 | 1.556% | 59.950% | 51.150% | 34.860% |

#### 其它已计算指标和证据位置

本次没有把所有原始 CSV 字段再复制成第二份独立文档；原始产物中的
`psnr`、`aud_masked_l1`、`aud_visible_mse`、`aud_visible_l1`、`rec_std`、
`tgt_std`、`top15_recall`、`pix_var`、`pair_l2`、训练 loss/LR、family breakdown
和 demo 小样本统计仍按原始文件保留。它们的来源是：

- `v11f_outputs_with_ckpt/outputs/outputs_v11f{,_control,_no_causal}/tables/`
- `v11f_outputs_with_ckpt/outputs/outputs_v11f{,_control,_no_causal}/logs/`
- `v11f_outputs_with_ckpt/outputs/outputs_v11f{,_control,_no_causal}/figures/`

本日志已直接记录逐实验 fixed/random 主表和 Cross-Key normal/zero/wrong/
same-class 对照；若某个字段未在主表中适用，必须按原始 CSV 的 `N/A` 和有效 n
解释，不能把缺失值改写成 0。v11f 的独立消融结论与限制继续沿用本日志上一段
“消融结论：v11f_no_causal”，不再另建结果说明文件。

#### 归档清理

- 删除独立结果说明：`docs/V11F_EVALUATION.md`、`docs/V11F_METRICS_SUMMARY.md`。
- 保留实验协议：`docs/V11F_MASKED_CROSS_KEY_PROTOCOL.md`。
- 评估结果正式入口：本文件的 v11f 评估条目；原始可复核证据入口：`outputs/`。

#### 运行说明

后续评价直接在 `docs/dev_log.md` 追加当前版本条目，按“主实验、control、全部
消融 × fixed/random × cue/family × 已计算指标”展开；不得再创建单独的
`Vxx_EVALUATION.md` 或 `Vxx_METRICS_SUMMARY.md`。原始 CSV、日志和图片仍写入
`outputs/`，但不把它们误称为独立结果报告。

### 2026-09-10 文档目录清理

按用户要求，删除两个版本专用的独立协议文件：

- `docs/V11E_CATEGORY_BINDING_PROTOCOL.md`
- `docs/V11F_MASKED_CROSS_KEY_PROTOCOL.md`

其有效约束已并入 `docs/implementation.md`、`docs/idea_report.md`、
`docs/user_requirements.md`，具体版本过程和评估归档继续写入本文件。以后除非
用户明确要求，不创建版本专用协议或结果 Markdown；原始运行证据仍保留在
`outputs/`。此前日志中对这些文件的路径引用属于历史记录，不改写。

### 2026-09-10 v11f 文档闭环补档

用户进一步明确：每个版本的最初方案和后续评估都必须写入现有核心文档，不能因为
删除版本专用文件而只留下代码或零散路径。v11f 的当前权威入口现明确如下：

- **最初方案与研究假设**：`docs/idea_report.md` 的“F 阶段补充：v11f”；
- **最终实现与真实配置**：`docs/implementation.md` 的“当前版本 v11f”；
- **完整评估与实验结论**：本文件前面的“2026-09-10 v11f 结果归档格式修订”
  条目，其中包含 control、main、no_causal 的 fixed/random 主表、Cross-Key
  normal/zero/wrong/same-class 配对表、有效 n、改善比例和限制；
- **运行与验收规范**：`docs/user_requirements.md` 和上述 implementation 条目。

本次补档没有创建新的 v11f 专用文件。此前日志中提到的已删除协议/评估文件路径
保留为不可变历史记录；从本条起，后续版本只允许把方案写入
`idea_report.md`/`implementation.md`，把验证和评估追加到 `dev_log.md`，并在
进入下一版本前完成三处文档核对。

### 2026-09-11 核心文档结构整理

用户指出核心文档存在重复、旧版本说明混入当前方案和入口不清的问题。本次只整理
文档，没有修改模型、配置、评估脚本或实验结果。

- `docs/implementation.md` 重写为当前 v11f 的单一实现指南：只保留真实目录、当前
  数据/target、tensor shape、前向路径、三组实验、评估口径、运行命令和历史索引。
- `docs/idea_report.md` 增加文档导航，继续保留各版本研究设计；已废止的 GRID 等
  方案只用于研究演进追溯，不作为当前运行入口。
- `docs/GPT_HANDOFF.md` 更新为 v11f 快速交接页，只提供入口和运行要点，不复制独立
  评估结果。
- `docs/dev_log.md` 顶部项目概览改为当前 v11f；历史日志保持只追加原则，不改写
  旧结论。当前评估结果的正式入口仍是本文件已有的 v11f 条目。
- 本次没有创建版本专用 Markdown，也没有暂存 `_data/`、`outputs/`、checkpoint、
  临时备份或其他未跟踪目录。

验证：`git diff --check` 通过；当前实现文档中的配置文件、路径和 v11f 三组实验与
仓库代码/配置一致。此次是文档整理，不应被解释为重新训练或重新评估。

#### 运行说明（当前 v11g）

这是当前活动版本的唯一运行命令入口；前面各版本的“运行说明”只属于对应历史条目。

```bash
# 在项目根目录、已激活 GPU 环境中，顺序运行 main、control 和可选 no-causal
nohup python -u scripts/run_v11g_suite.py --with_ablations > v11g_suite.log 2>&1 < /dev/null &
tail -f v11g_suite.log

# 只评估已有三组 checkpoint
python -u scripts/run_v11g_suite.py --eval_only --with_ablations
```

单独运行、fixed/random、family breakdown 和 demo 命令见
`docs/implementation.md` 第 6 节。当前分支 `configs/` 只使用三份 v11g YAML；本次
没有上传本地数据、outputs、checkpoint、临时文件或旧版本配置。

### 2026-09-11 v11g 版本迁移记录

本次将已确定的 v11f 方案迁移为独立的 v11g 运行入口，不新增模型结构或未经验证
的性能结论。v11g 继续使用 `v11e_control` 冻结父权重、Masked Feature Cross-Key、
same-modal gated cue detail、`detach_value_for_recon=true`、30 轮训练、batch size
128、severity 0.4 和五类 corruption family 均衡采样；main、control、no-causal
三组配置的行为边界与 v11f 保持一致。

当前待执行验证：

- `python -u scripts/smoke_test_v11g.py --config configs/v11g.yaml`
- `python -u scripts/run_v11g_suite.py --with_ablations`
- 训练完成后使用 `python -u scripts/run_v11g_suite.py --eval_only --with_ablations`

本条只记录版本入口和验收要求，尚无 v11g 训练或评估结果。实际完成后，必须在本
文件继续追加 main、control、no-causal 各实验的全部指标、fixed/random 协议、有效 n、
异常、限制、独立结论和 outputs 证据路径；不创建独立版本评估 Markdown。

### 2026-09-12 v11g 完整指标汇总

已整理成 v11g 完整指标汇总表，包含 52 个评估指标字段、fixed/random、逐 family、训练统计和 demo 小样本结果。三组实验分别为：

- **control**：冻结 v11e_control 父模型，额外训练 0 轮。
- **v11g**：局部 Masked Cross-Key + causal margin，额外训练 30 轮。
- **no_causal**：局部 Masked Cross-Key，不加 causal margin，额外训练 30 轮。

以下为 severity=0.4、fixed_mask 的主要指标。MSE 越低越好，SSIM/ACC 越高越好。格式为 Index ACC、图像 MSE、图像 SSIM、音频 MSE、音频 SSIM；每个 cue 有效 n=10000。

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
|---|---|---:|---:|---:|---:|---:|
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11g | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11g | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965833 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966043 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11g | 98.830% | 0.002053 | 0.985947 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002085 | 0.985614 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 98.934% | 0.009790 | 0.942597 | 0.006097 | 0.807590 |
| 图像干净、音频残缺 | v11g | 98.934% | 0.009790 | 0.942597 | 0.005922 | 0.809892 |
| 图像干净、音频残缺 | no_causal | 98.934% | 0.009790 | 0.942597 | 0.005991 | 0.810125 |
| 音频干净、图像残缺 | control | 99.144% | 0.007502 | 0.956305 | 0.003647 | 0.911207 |
| 音频干净、图像残缺 | v11g | 99.144% | 0.007321 | 0.957235 | 0.003647 | 0.911207 |
| 音频干净、图像残缺 | no_causal | 99.144% | 0.007390 | 0.956850 | 0.003647 | 0.911207 |
| 双模态残缺 | control | 97.536% | 0.007520 | 0.956189 | 0.006176 | 0.805881 |
| 双模态残缺 | v11g | 97.536% | 0.007371 | 0.957016 | 0.006041 | 0.807610 |
| 双模态残缺 | no_causal | 97.536% | 0.007431 | 0.956679 | 0.006095 | 0.807901 |
| 仅音频残缺 | control | 84.572% | 0.013212 | 0.877703 | 0.006290 | 0.804689 |
| 仅音频残缺 | v11g | 84.572% | 0.012756 | 0.885779 | 0.006290 | 0.804689 |
| 仅音频残缺 | no_causal | 84.572% | 0.012351 | 0.891893 | 0.006290 | 0.804689 |
| 仅图像残缺 | control | 92.028% | 0.007680 | 0.955151 | 0.000668 | 0.916781 |
| 仅图像残缺 | v11g | 92.028% | 0.007680 | 0.955151 | 0.000670 | 0.916390 |
| 仅图像残缺 | no_causal | 92.028% | 0.007680 | 0.955151 | 0.000674 | 0.917291 |

#### 缺失区域指标

| cue | main 图像 masked MSE | control 图像 masked MSE | no-causal 图像 masked MSE | main 音频 masked MSE | control 音频 masked MSE | no-causal 音频 masked MSE |
|---|---:|---:|---:|---:|---:|---:|
| 图像干净、音频残缺 | NA | NA | NA | 0.010128 | 0.010599 | 0.010311 |
| 仅音频残缺 | 0.012756 | 0.013212 | 0.012351 | 0.011079 | 0.011079 | 0.011079 |
| 音频干净、图像残缺 | 0.025264 | 0.025918 | 0.025441 | NA | NA | NA |
| 仅图像残缺 | 0.026662 | 0.026662 | 0.026662 | 0.000670 | 0.000668 | 0.000674 |
| 双模态残缺 | 0.025356 | 0.025912 | 0.025528 | 0.010416 | 0.010775 | 0.010558 |

#### Random、Cross-Key、family 与训练指标

- random 主结果：main 在 corrupt_aud_only 的 ACC/MSE/SSIM 为 85.79%/0.006055/0.813113，在 corrupt_both 为 97.28%/0.005804/0.822269；control 与 no_causal 的完整 random 指标见对应 CSV。
- main fixed 的 audio→image corrupt_aud_only：normal/zero/wrong MSE 为 0.012756/0.013212/0.012740，gain=0.000455，win_zero=53.99%，win_both=33.56%，gate=0.5358。
- main fixed 的 image→audio clean_img_corrupt_aud：normal/zero/wrong MSE 为 0.010128/0.010599/0.010143，gain=0.000471，win_zero=67.03%，win_both=36.31%，gate=0.9471。
- control 的 Cross-Key gain 和 win 指标全部为 0；no_causal 上述两组 gain 分别为 0.000861 和 0.000288，但 win_both 仅 32.08% 和 32.89%。
- family 中最困难的是 partial_temporal：audio-only ACC=45.88%、audio MSE 约 0.0126、audio SSIM 约 0.60、top15 recall 约 68.6%；其余 family ACC 为 90.39%--97.10%。
- main/no-causal 末轮平均 loss 分别为 0.8213/0.7790；fixed/random demo 均已生成。全部 52 个字段和逐 family 原始值保存在 v11g_outputs/outputs/outputs_v11g*/tables/。
- 结论：v11g 没有改变 Index 分类，Cross-Key 对缺失区域有小幅改善，但 win_both 仍只有约 32%--38%，尚不能证明稳定的类别选择性；causal margin 没有在全部指标上稳定优于 no_causal。

```text
（以上历史条目仅作追溯，当前统一评估从下方开始。）
```


<a id="evaluation-format-20260912"></a>

## 2026-09-12 评估归档（统一版式）

本次按用户给出的截图统一本地评估格式，只重算表格聚合并整理已有 CSV/日志，没有训练或重跑模型。下列五节是各版本结果的当前汇总入口；前面的旧评估条目保留作历史追溯，不再作为最新格式模板。旧条目中的错误在对应版本“结论与比较限制”明确勘误。

| 版本 | 实验组 | 统一入口 |
| --- | --- | ---: |
| v11c | control、v11c | [v11c 完整指标汇总](#evaluation-v11c) |
| v11d | control、v11d | [v11d 完整指标汇总](#evaluation-v11d) |
| v11e | control、v11e | [v11e 完整指标汇总](#evaluation-v11e) |
| v11f | control、v11f、no_causal | [v11f 完整指标汇总](#evaluation-v11f) |
| v11g | control、v11g、no_causal | [v11g 完整指标汇总](#evaluation-v11g) |

**统一读法**：每版先看实验说明及 fixed 主表，再看 random；区域、干预、family、训练和 demo 的数值均在本文件的折叠表中，不需要另开版本评估文档。区域无效/未保存为 N/A，具体原因按版本说明和 n 表判定。历史方法、训练预算与 target 不同，本次没有生成误导性的跨版本总排名。

<a id="evaluation-v11c"></a>

### v11c

本节覆盖 **2 组实验、14 个主评估/干预字段、7 个音频分布诊断字段**，以及独立 family、训练统计和本地已有 demo。字段按实际产物统计，含明确标注的不适用项；不把其他版本的缺项补成零。

**实验说明**

| 实验 | 权重起点 | 本轮训练范围 | 实际轮数 |
| --- | --- | ---: | ---: |
| control | v11b_recovery（epoch 119 后） | 关闭 Cross-Key；训练 decoder | 30（120–149） |
| v11c | 同一 v11b_recovery | decoder + Cross-Key/causal | 30（120–149） |

训练配置 seed=1234、batch_size=128；fixed_mask seed=1234、severity=0.4。轮数以上表和完成日志为准，不把父模型历史轮数计成本轮新增训练。历史分支配置不是当前分支的运行入口。

**恢复目标：仅图像为 sample/category，仅音频为 category/sample，双模态为 sample/sample。** category 使用训练集 medoid；“仅音频残缺”表示图像全缺失、音频部分残缺，反向同理。

| 实验 | fixed / random | Cross-Key 扫描 | demo fixed / random |
| --- | --- | ---: | ---: |
| control | 有 / 未找到 | 未找到 | 有 / 未找到 |
| v11c | 有 / 未找到 | 未找到 | 有 / 未找到 |

**数据来源**：
- `control`：[本地产物](../v11cnew_outputs_with_ckpt/outputs/outputs_v11c_control/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。
- `v11c`：[本地产物](../v11cnew_outputs_with_ckpt/outputs/outputs_v11c/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。

MSE/L1 越低越好，ACC/SSIM/PSNR 越高越好；gate、res/V 和多样性仅是诊断。宏平均为五个 family 等权均值，不是把 normal、sweep、独立音频 family 重复叠加。N/A 表示未保存或在该场景不适用，详见字段覆盖说明。

**精度与缺项**：主表源 CSV/日志已舍入（ACC 常到 0.1%、MSE 常到 4 位、SSIM 常到 3 位）；本节六位显示只是统一排版，不恢复丢失精度。未保存逐指标有效 n、区域 L1、内容 ACC、干预绝对误差/win 等字段不编造。独立音频 family CSV 中实际存在的 L1 仍在其专表报告。

**分类与恢复：fixed 五类等权宏平均**

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.400% | 0.008800 | 0.950000 | 0.005200 | 0.875000 |
| 干净双模态 | v11c | 99.400% | 0.009100 | 0.946000 | 0.005100 | 0.876000 |
| 仅干净图像 | control | 98.600% | 0.008800 | 0.950000 | 0.000100 | 0.990000 |
| 仅干净图像 | v11c | 98.600% | 0.009100 | 0.946000 | 0.000100 | 0.990000 |
| 仅干净音频 | control | 97.700% | 0.001600 | 0.988000 | 0.005200 | 0.875000 |
| 仅干净音频 | v11c | 97.700% | 0.001600 | 0.988000 | 0.005100 | 0.876000 |
| 图像干净、音频残缺 | control | 98.700% | 0.008820 | 0.949800 | 0.007180 | 0.784200 |
| 图像干净、音频残缺 | v11c | 98.700% | 0.009120 | 0.946000 | 0.007340 | 0.785200 |
| 音频干净、图像残缺 | control | 99.060% | 0.004820 | 0.972600 | 0.005200 | 0.875000 |
| 音频干净、图像残缺 | v11c | 99.060% | 0.004820 | 0.972600 | 0.005100 | 0.876000 |
| 双模态残缺 | control | 97.620% | 0.004840 | 0.972800 | 0.007240 | 0.782400 |
| 双模态残缺 | v11c | 97.620% | 0.004840 | 0.972800 | 0.007420 | 0.783600 |
| 仅音频残缺 | control | 85.880% | 0.009820 | 0.919400 | 0.007360 | 0.778600 |
| 仅音频残缺 | v11c | 85.880% | 0.009820 | 0.920800 | 0.007500 | 0.779600 |
| 仅图像残缺 | control | 95.500% | 0.004920 | 0.972000 | 0.000320 | 0.966600 |
| 仅图像残缺 | v11c | 95.500% | 0.004920 | 0.972000 | 0.000320 | 0.966400 |


<details>
<summary>fixed 五类等权宏平均：区域、内容、多样性与音频分布完整指标</summary>

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.410000 | N/A | N/A |
| 干净双模态 | v11c | 21.180000 | N/A | N/A |
| 仅干净图像 | control | 21.410000 | N/A | N/A |
| 仅干净图像 | v11c | 21.180000 | N/A | N/A |
| 仅干净音频 | control | 58.330000 | 0.001600 | 0.001600 |
| 仅干净音频 | v11c | 54.910000 | 0.001600 | 0.001600 |
| 图像干净、音频残缺 | control | 21.402000 | N/A | N/A |
| 图像干净、音频残缺 | v11c | 21.164000 | N/A | N/A |
| 音频干净、图像残缺 | control | 25.760000 | 0.059660 | 0.016880 |
| 音频干净、图像残缺 | v11c | 25.804000 | 0.059380 | 0.016860 |
| 双模态残缺 | control | 25.768000 | 0.059720 | 0.016900 |
| 双模态残缺 | v11c | 25.798000 | 0.059460 | 0.016920 |
| 仅音频残缺 | control | 50.044000 | 0.009820 | 0.009820 |
| 仅音频残缺 | v11c | 47.406000 | 0.009820 | 0.009820 |
| 仅图像残缺 | control | 25.702000 | 0.060600 | 0.017360 |
| 仅图像残缺 | v11c | 25.736000 | 0.060380 | 0.017380 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11c | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11c | N/A | N/A |
| 音频干净、图像残缺 | control | 0.033000 | 0.000000 |
| 音频干净、图像残缺 | v11c | 0.033660 | 0.000000 |
| 双模态残缺 | control | 0.033080 | 0.000000 |
| 双模态残缺 | v11c | 0.033760 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11c | N/A | N/A |
| 仅图像残缺 | control | 0.033340 | 0.000000 |
| 仅图像残缺 | v11c | 0.034020 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | 0.000100 | N/A |
| 仅干净图像 | v11c | 0.000100 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | 0.011520 | 0.004320 |
| 图像干净、音频残缺 | v11c | 0.011880 | 0.004320 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11c | N/A | N/A |
| 双模态残缺 | control | 0.011660 | 0.004300 |
| 双模态残缺 | v11c | 0.012040 | 0.004340 |
| 仅音频残缺 | control | 0.011980 | 0.004320 |
| 仅音频残缺 | v11c | 0.012280 | 0.004340 |
| 仅图像残缺 | control | 0.000320 | N/A |
| 仅图像残缺 | v11c | 0.000320 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.060900 | 0.344780 |
| 干净双模态 | v11c | 0.058700 | 0.338060 |
| 仅干净图像 | control | 0.060900 | 0.344480 |
| 仅干净图像 | v11c | 0.058600 | 0.337660 |
| 仅干净音频 | control | 0.047400 | 0.289240 |
| 仅干净音频 | v11c | 0.047400 | 0.289320 |
| 图像干净、音频残缺 | control | 0.060840 | 0.344160 |
| 图像干净、音频残缺 | v11c | 0.058540 | 0.337460 |
| 音频干净、图像残缺 | control | 0.063560 | 0.352280 |
| 音频干净、图像残缺 | v11c | 0.063740 | 0.352740 |
| 双模态残缺 | control | 0.063500 | 0.352540 |
| 双模态残缺 | v11c | 0.063640 | 0.353020 |
| 仅音频残缺 | control | 0.043140 | 0.273660 |
| 仅音频残缺 | v11c | 0.043100 | 0.273560 |
| 仅图像残缺 | control | 0.063480 | 0.352000 |
| 仅图像残缺 | v11c | 0.063620 | 0.352420 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043000 | 0.155600 | 1.000000 |
| 干净双模态 | v11c | 0.043600 | 0.155500 | 1.000000 |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.994300 |
| 仅干净图像 | v11c | 0.015700 | 0.070400 | 0.993400 |
| 仅干净音频 | control | 0.043000 | 0.155700 | 1.000000 |
| 仅干净音频 | v11c | 0.043400 | 0.155400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.041060 | 0.149160 | 1.000000 |
| 图像干净、音频残缺 | v11c | 0.040380 | 0.147140 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043000 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11c | 0.043600 | 0.155600 | 1.000000 |
| 双模态残缺 | control | 0.041100 | 0.149160 | 1.000000 |
| 双模态残缺 | v11c | 0.040380 | 0.147160 | 1.000000 |
| 仅音频残缺 | control | 0.040980 | 0.149000 | 1.000000 |
| 仅音频残缺 | v11c | 0.040220 | 0.146960 | 1.000000 |
| 仅图像残缺 | control | 0.015560 | 0.070300 | 0.995380 |
| 仅图像残缺 | v11c | 0.015680 | 0.070120 | 0.994360 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 干净双模态 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 92.300% |
| 仅干净图像 | v11c | 0.015600 | 0.070500 | 0.987900 | 92.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 81.800% |
| 仅干净音频 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.560% |
| 图像干净、音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 78.460% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 音频干净、图像残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.480% |
| 双模态残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 78.400% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.060% |
| 仅音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 77.940% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 90.640% |
| 仅图像残缺 | v11c | 0.015600 | 0.070500 | 0.987900 | 90.680% |

</details>


<details>
<summary>fixed 五类等权宏平均：各字段有效样本数</summary>

| 实验 | 指标字段（同一计数口径） | 每 family 的 n |
| --- | --- | ---: |
| control | `acc`、`aud_masked_mse`、`aud_mse` | 未逐指标保存 |
| control | `aud_ssim`、`aud_visible_mse`、`img_coarse_masked_mse` | 未逐指标保存 |
| control | `img_coarse_visible_mse`、`img_masked_mse`、`img_mse` | 未逐指标保存 |
| control | `img_visible_mse`、`pair_l2`、`pix_var` | 未逐指标保存 |
| control | `psnr`、`rec_max`、`rec_mean` | 未逐指标保存 |
| control | `rec_std`、`ssim`、`tgt_max` | 未逐指标保存 |
| control | `tgt_mean`、`tgt_std`、`top15_recall` | 未逐指标保存 |
| v11c | `acc`、`aud_masked_mse`、`aud_mse` | 未逐指标保存 |
| v11c | `aud_ssim`、`aud_visible_mse`、`img_coarse_masked_mse` | 未逐指标保存 |
| v11c | `img_coarse_visible_mse`、`img_masked_mse`、`img_mse` | 未逐指标保存 |
| v11c | `img_visible_mse`、`pair_l2`、`pix_var` | 未逐指标保存 |
| v11c | `psnr`、`rec_max`、`rec_mean` | 未逐指标保存 |
| v11c | `rec_std`、`ssim`、`tgt_max` | 未逐指标保存 |
| v11c | `tgt_mean`、`tgt_std`、`top15_recall` | 未逐指标保存 |

</details>

**分类与恢复：random 单次评估**

本地产物未找到此协议结果，不将 fixed 复制成 random。

**Cross-Key / Cross-Detail 干预**

同一 cue/mask 内，gain=zero−normal，wrong damage=wrong−normal，same damage 为有效同类替换上的配对差；正 gain 表示改善。gate 非零本身不代表恢复有效。

- fixed 五类等权宏平均：未找到干预结果。

- random：未找到干预结果。

**逐 family 完整分项**

以下是与主表同源的五组 image/audio family pair，每组内仍按输入模式、实验排列。每组约 10000 个 MNIST 测试条目；音频会复用，旧版有效 n 未逐指标保存。


<details>
<summary>family 1：occlusion/time_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.400% | 0.008800 | 0.950000 | 0.005200 | 0.875000 |
| 干净双模态 | v11c | 99.400% | 0.009100 | 0.946000 | 0.005100 | 0.876000 |
| 仅干净图像 | control | 98.600% | 0.008800 | 0.950000 | 0.000100 | 0.990000 |
| 仅干净图像 | v11c | 98.600% | 0.009100 | 0.946000 | 0.000100 | 0.990000 |
| 仅干净音频 | control | 97.700% | 0.001600 | 0.988000 | 0.005200 | 0.875000 |
| 仅干净音频 | v11c | 97.700% | 0.001600 | 0.988000 | 0.005100 | 0.876000 |
| 图像干净、音频残缺 | control | 99.100% | 0.008800 | 0.950000 | 0.007200 | 0.812000 |
| 图像干净、音频残缺 | v11c | 99.100% | 0.009100 | 0.946000 | 0.007400 | 0.806000 |
| 音频干净、图像残缺 | control | 99.100% | 0.005900 | 0.966000 | 0.005200 | 0.875000 |
| 音频干净、图像残缺 | v11c | 99.100% | 0.005900 | 0.966000 | 0.005100 | 0.876000 |
| 双模态残缺 | control | 98.100% | 0.005900 | 0.966000 | 0.007400 | 0.806000 |
| 双模态残缺 | v11c | 98.100% | 0.005900 | 0.966000 | 0.007600 | 0.799000 |
| 仅音频残缺 | control | 92.800% | 0.006300 | 0.954000 | 0.007300 | 0.810000 |
| 仅音频残缺 | v11c | 92.800% | 0.006300 | 0.954000 | 0.007500 | 0.804000 |
| 仅图像残缺 | control | 93.500% | 0.006100 | 0.964000 | 0.000500 | 0.953000 |
| 仅图像残缺 | v11c | 93.500% | 0.006100 | 0.964000 | 0.000500 | 0.953000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.410000 | N/A | N/A |
| 干净双模态 | v11c | 21.180000 | N/A | N/A |
| 仅干净图像 | control | 21.410000 | N/A | N/A |
| 仅干净图像 | v11c | 21.180000 | N/A | N/A |
| 仅干净音频 | control | 58.330000 | 0.001600 | 0.001600 |
| 仅干净音频 | v11c | 54.910000 | 0.001600 | 0.001600 |
| 图像干净、音频残缺 | control | 21.410000 | N/A | N/A |
| 图像干净、音频残缺 | v11c | 21.170000 | N/A | N/A |
| 音频干净、图像残缺 | control | 25.310000 | 0.103800 | 0.038200 |
| 音频干净、图像残缺 | v11c | 25.350000 | 0.103400 | 0.038200 |
| 双模态残缺 | control | 25.320000 | 0.103200 | 0.038100 |
| 双模态残缺 | v11c | 25.330000 | 0.102800 | 0.038100 |
| 仅音频残缺 | control | 53.750000 | 0.006300 | 0.006300 |
| 仅音频残缺 | v11c | 50.810000 | 0.006300 | 0.006300 |
| 仅图像残缺 | control | 25.180000 | 0.106300 | 0.039800 |
| 仅图像残缺 | v11c | 25.210000 | 0.105900 | 0.039800 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11c | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11c | N/A | N/A |
| 音频干净、图像残缺 | control | 0.015600 | 0.000000 |
| 音频干净、图像残缺 | v11c | 0.016400 | 0.000000 |
| 双模态残缺 | control | 0.015500 | 0.000000 |
| 双模态残缺 | v11c | 0.016400 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11c | N/A | N/A |
| 仅图像残缺 | control | 0.015700 | 0.000000 |
| 仅图像残缺 | v11c | 0.016600 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | 0.000100 | N/A |
| 仅干净图像 | v11c | 0.000100 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | 0.009300 | 0.005800 |
| 图像干净、音频残缺 | v11c | 0.009700 | 0.005900 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11c | N/A | N/A |
| 双模态残缺 | control | 0.009800 | 0.005800 |
| 双模态残缺 | v11c | 0.010300 | 0.005900 |
| 仅音频残缺 | control | 0.009600 | 0.005800 |
| 仅音频残缺 | v11c | 0.009900 | 0.005900 |
| 仅图像残缺 | control | 0.000500 | N/A |
| 仅图像残缺 | v11c | 0.000500 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.060900 | 0.345600 |
| 干净双模态 | v11c | 0.058700 | 0.338900 |
| 仅干净图像 | control | 0.060900 | 0.343300 |
| 仅干净图像 | v11c | 0.058600 | 0.336500 |
| 仅干净音频 | control | 0.047400 | 0.288600 |
| 仅干净音频 | v11c | 0.047400 | 0.288700 |
| 图像干净、音频残缺 | control | 0.060900 | 0.345200 |
| 图像干净、音频残缺 | v11c | 0.058600 | 0.338400 |
| 音频干净、图像残缺 | control | 0.062800 | 0.350700 |
| 音频干净、图像残缺 | v11c | 0.062900 | 0.351000 |
| 双模态残缺 | control | 0.062700 | 0.349200 |
| 双模态残缺 | v11c | 0.062800 | 0.349600 |
| 仅音频残缺 | control | 0.046400 | 0.292400 |
| 仅音频残缺 | v11c | 0.046400 | 0.292500 |
| 仅图像残缺 | control | 0.062600 | 0.350100 |
| 仅图像残缺 | v11c | 0.062700 | 0.350400 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043000 | 0.155600 | 1.000000 |
| 干净双模态 | v11c | 0.043600 | 0.155500 | 1.000000 |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.994300 |
| 仅干净图像 | v11c | 0.015700 | 0.070400 | 0.993400 |
| 仅干净音频 | control | 0.043000 | 0.155700 | 1.000000 |
| 仅干净音频 | v11c | 0.043400 | 0.155400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.041000 | 0.150500 | 1.000000 |
| 图像干净、音频残缺 | v11c | 0.040900 | 0.149200 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043000 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11c | 0.043600 | 0.155600 | 1.000000 |
| 双模态残缺 | control | 0.041000 | 0.150300 | 1.000000 |
| 双模态残缺 | v11c | 0.040800 | 0.149000 | 1.000000 |
| 仅音频残缺 | control | 0.041000 | 0.150500 | 1.000000 |
| 仅音频残缺 | v11c | 0.040700 | 0.148900 | 1.000000 |
| 仅图像残缺 | control | 0.015500 | 0.070100 | 0.995200 |
| 仅图像残缺 | v11c | 0.015600 | 0.069900 | 0.994700 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 干净双模态 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 92.300% |
| 仅干净图像 | v11c | 0.015600 | 0.070500 | 0.987900 | 92.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 81.800% |
| 仅干净音频 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.200% |
| 图像干净、音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 78.000% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 音频干净、图像残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.000% |
| 双模态残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 77.800% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.000% |
| 仅音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 77.800% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 90.000% |
| 仅图像残缺 | v11c | 0.015600 | 0.070500 | 0.987900 | 90.000% |

</details>


<details>
<summary>family 2：pixel_delete/freq_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.400% | 0.008800 | 0.950000 | 0.005200 | 0.875000 |
| 干净双模态 | v11c | 99.400% | 0.009100 | 0.946000 | 0.005100 | 0.876000 |
| 仅干净图像 | control | 98.600% | 0.008800 | 0.950000 | 0.000100 | 0.990000 |
| 仅干净图像 | v11c | 98.600% | 0.009100 | 0.946000 | 0.000100 | 0.990000 |
| 仅干净音频 | control | 97.700% | 0.001600 | 0.988000 | 0.005200 | 0.875000 |
| 仅干净音频 | v11c | 97.700% | 0.001600 | 0.988000 | 0.005100 | 0.876000 |
| 图像干净、音频残缺 | control | 99.100% | 0.008800 | 0.950000 | 0.005800 | 0.840000 |
| 图像干净、音频残缺 | v11c | 99.100% | 0.009100 | 0.946000 | 0.005800 | 0.843000 |
| 音频干净、图像残缺 | control | 99.200% | 0.001800 | 0.990000 | 0.005200 | 0.875000 |
| 音频干净、图像残缺 | v11c | 99.200% | 0.001800 | 0.990000 | 0.005100 | 0.876000 |
| 双模态残缺 | control | 98.500% | 0.001800 | 0.990000 | 0.005800 | 0.840000 |
| 双模态残缺 | v11c | 98.500% | 0.001800 | 0.990000 | 0.005800 | 0.843000 |
| 仅音频残缺 | control | 94.900% | 0.004800 | 0.966000 | 0.005900 | 0.839000 |
| 仅音频残缺 | v11c | 94.900% | 0.004800 | 0.966000 | 0.005800 | 0.841000 |
| 仅图像残缺 | control | 97.200% | 0.001800 | 0.990000 | 0.000200 | 0.980000 |
| 仅图像残缺 | v11c | 97.200% | 0.001800 | 0.990000 | 0.000200 | 0.980000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.410000 | N/A | N/A |
| 干净双模态 | v11c | 21.180000 | N/A | N/A |
| 仅干净图像 | control | 21.410000 | N/A | N/A |
| 仅干净图像 | v11c | 21.180000 | N/A | N/A |
| 仅干净音频 | control | 58.330000 | 0.001600 | 0.001600 |
| 仅干净音频 | v11c | 54.910000 | 0.001600 | 0.001600 |
| 图像干净、音频残缺 | control | 21.410000 | N/A | N/A |
| 图像干净、音频残缺 | v11c | 21.170000 | N/A | N/A |
| 音频干净、图像残缺 | control | 28.030000 | 0.034300 | 0.004500 |
| 音频干净、图像残缺 | v11c | 28.070000 | 0.035600 | 0.004500 |
| 双模态残缺 | control | 28.030000 | 0.034400 | 0.004500 |
| 双模态残缺 | v11c | 28.070000 | 0.035600 | 0.004500 |
| 仅音频残缺 | control | 54.100000 | 0.004800 | 0.004800 |
| 仅音频残缺 | v11c | 51.250000 | 0.004800 | 0.004800 |
| 仅图像残缺 | control | 28.020000 | 0.034700 | 0.004500 |
| 仅图像残缺 | v11c | 28.060000 | 0.036000 | 0.004500 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11c | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11c | N/A | N/A |
| 音频干净、图像残缺 | control | 0.026100 | 0.000000 |
| 音频干净、图像残缺 | v11c | 0.027700 | 0.000000 |
| 双模态残缺 | control | 0.026100 | 0.000000 |
| 双模态残缺 | v11c | 0.027700 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11c | N/A | N/A |
| 仅图像残缺 | control | 0.026300 | 0.000000 |
| 仅图像残缺 | v11c | 0.028000 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | 0.000100 | N/A |
| 仅干净图像 | v11c | 0.000100 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | 0.006800 | 0.005200 |
| 图像干净、音频残缺 | v11c | 0.006700 | 0.005100 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11c | N/A | N/A |
| 双模态残缺 | control | 0.006900 | 0.005100 |
| 双模态残缺 | v11c | 0.006800 | 0.005100 |
| 仅音频残缺 | control | 0.006900 | 0.005200 |
| 仅音频残缺 | v11c | 0.006800 | 0.005200 |
| 仅图像残缺 | control | 0.000200 | N/A |
| 仅图像残缺 | v11c | 0.000200 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.060900 | 0.345600 |
| 干净双模态 | v11c | 0.058700 | 0.338900 |
| 仅干净图像 | control | 0.060900 | 0.344400 |
| 仅干净图像 | v11c | 0.058600 | 0.337500 |
| 仅干净音频 | control | 0.047400 | 0.290800 |
| 仅干净音频 | v11c | 0.047400 | 0.290800 |
| 图像干净、音频残缺 | control | 0.060900 | 0.345500 |
| 图像干净、音频残缺 | v11c | 0.058600 | 0.338900 |
| 音频干净、图像残缺 | control | 0.066400 | 0.359700 |
| 音频干净、图像残缺 | v11c | 0.066500 | 0.359900 |
| 双模态残缺 | control | 0.066400 | 0.360400 |
| 双模态残缺 | v11c | 0.066400 | 0.360600 |
| 仅音频残缺 | control | 0.047100 | 0.290900 |
| 仅音频残缺 | v11c | 0.047100 | 0.290800 |
| 仅图像残缺 | control | 0.066400 | 0.359700 |
| 仅图像残缺 | v11c | 0.066400 | 0.359800 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043000 | 0.155600 | 1.000000 |
| 干净双模态 | v11c | 0.043600 | 0.155500 | 1.000000 |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.994300 |
| 仅干净图像 | v11c | 0.015700 | 0.070400 | 0.993400 |
| 仅干净音频 | control | 0.043000 | 0.155700 | 1.000000 |
| 仅干净音频 | v11c | 0.043400 | 0.155400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.042500 | 0.154400 | 1.000000 |
| 图像干净、音频残缺 | v11c | 0.043000 | 0.154100 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043000 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11c | 0.043600 | 0.155600 | 1.000000 |
| 双模态残缺 | control | 0.042500 | 0.154300 | 1.000000 |
| 双模态残缺 | v11c | 0.043000 | 0.154100 | 1.000000 |
| 仅音频残缺 | control | 0.042500 | 0.154500 | 1.000000 |
| 仅音频残缺 | v11c | 0.042900 | 0.154000 | 1.000000 |
| 仅图像残缺 | control | 0.015600 | 0.070400 | 0.994900 |
| 仅图像残缺 | v11c | 0.015700 | 0.070200 | 0.994300 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 干净双模态 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 92.300% |
| 仅干净图像 | v11c | 0.015600 | 0.070500 | 0.987900 | 92.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 81.800% |
| 仅干净音频 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.500% |
| 图像干净、音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 80.800% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 音频干净、图像残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.500% |
| 双模态残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 80.800% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.300% |
| 仅音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 80.600% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 91.600% |
| 仅图像残缺 | v11c | 0.015600 | 0.070500 | 0.987900 | 91.600% |

</details>


<details>
<summary>family 3：mask_vertical/feature_dropout</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.400% | 0.008800 | 0.950000 | 0.005200 | 0.875000 |
| 干净双模态 | v11c | 99.400% | 0.009100 | 0.946000 | 0.005100 | 0.876000 |
| 仅干净图像 | control | 98.600% | 0.008800 | 0.950000 | 0.000100 | 0.990000 |
| 仅干净图像 | v11c | 98.600% | 0.009100 | 0.946000 | 0.000100 | 0.990000 |
| 仅干净音频 | control | 97.700% | 0.001600 | 0.988000 | 0.005200 | 0.875000 |
| 仅干净音频 | v11c | 97.700% | 0.001600 | 0.988000 | 0.005100 | 0.876000 |
| 图像干净、音频残缺 | control | 99.300% | 0.008800 | 0.950000 | 0.005100 | 0.868000 |
| 图像干净、音频残缺 | v11c | 99.300% | 0.009100 | 0.946000 | 0.005000 | 0.871000 |
| 音频干净、图像残缺 | control | 99.200% | 0.005700 | 0.969000 | 0.005200 | 0.875000 |
| 音频干净、图像残缺 | v11c | 99.200% | 0.005700 | 0.969000 | 0.005100 | 0.876000 |
| 双模态残缺 | control | 99.000% | 0.005600 | 0.970000 | 0.005100 | 0.868000 |
| 双模态残缺 | v11c | 99.000% | 0.005600 | 0.970000 | 0.005000 | 0.872000 |
| 仅音频残缺 | control | 97.800% | 0.001900 | 0.987000 | 0.005100 | 0.867000 |
| 仅音频残缺 | v11c | 97.800% | 0.001900 | 0.987000 | 0.005000 | 0.871000 |
| 仅图像残缺 | control | 96.500% | 0.005800 | 0.969000 | 0.000200 | 0.975000 |
| 仅图像残缺 | v11c | 96.500% | 0.005800 | 0.969000 | 0.000200 | 0.975000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.410000 | N/A | N/A |
| 干净双模态 | v11c | 21.180000 | N/A | N/A |
| 仅干净图像 | control | 21.410000 | N/A | N/A |
| 仅干净图像 | v11c | 21.180000 | N/A | N/A |
| 仅干净音频 | control | 58.330000 | 0.001600 | 0.001600 |
| 仅干净音频 | v11c | 54.910000 | 0.001600 | 0.001600 |
| 图像干净、音频残缺 | control | 21.410000 | N/A | N/A |
| 图像干净、音频残缺 | v11c | 21.180000 | N/A | N/A |
| 音频干净、图像残缺 | control | 26.110000 | 0.033500 | 0.014400 |
| 音频干净、图像残缺 | v11c | 26.170000 | 0.032900 | 0.014400 |
| 双模态残缺 | control | 26.200000 | 0.033400 | 0.014300 |
| 双模态残缺 | v11c | 26.260000 | 0.032800 | 0.014300 |
| 仅音频残缺 | control | 57.250000 | 0.001900 | 0.001900 |
| 仅音频残缺 | v11c | 54.030000 | 0.001900 | 0.001900 |
| 仅图像残缺 | control | 26.050000 | 0.033800 | 0.014800 |
| 仅图像残缺 | v11c | 26.090000 | 0.033300 | 0.014800 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11c | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11c | N/A | N/A |
| 音频干净、图像残缺 | control | 0.017600 | 0.000000 |
| 音频干净、图像残缺 | v11c | 0.018500 | 0.000000 |
| 双模态残缺 | control | 0.017700 | 0.000000 |
| 双模态残缺 | v11c | 0.018600 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11c | N/A | N/A |
| 仅图像残缺 | control | 0.017800 | 0.000000 |
| 仅图像残缺 | v11c | 0.018600 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | 0.000100 | N/A |
| 仅干净图像 | v11c | 0.000100 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | 0.005200 | 0.005100 |
| 图像干净、音频残缺 | v11c | 0.005100 | 0.005000 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11c | N/A | N/A |
| 双模态残缺 | control | 0.005200 | 0.005100 |
| 双模态残缺 | v11c | 0.005100 | 0.005000 |
| 仅音频残缺 | control | 0.005200 | 0.005100 |
| 仅音频残缺 | v11c | 0.005100 | 0.005000 |
| 仅图像残缺 | control | 0.000200 | N/A |
| 仅图像残缺 | v11c | 0.000200 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.060900 | 0.343500 |
| 干净双模态 | v11c | 0.058700 | 0.336800 |
| 仅干净图像 | control | 0.060900 | 0.345800 |
| 仅干净图像 | v11c | 0.058600 | 0.338900 |
| 仅干净音频 | control | 0.047400 | 0.288500 |
| 仅干净音频 | v11c | 0.047400 | 0.288600 |
| 图像干净、音频残缺 | control | 0.060900 | 0.343000 |
| 图像干净、音频残缺 | v11c | 0.058600 | 0.336200 |
| 音频干净、图像残缺 | control | 0.062400 | 0.348500 |
| 音频干净、图像残缺 | v11c | 0.062700 | 0.349300 |
| 双模态残缺 | control | 0.062400 | 0.349000 |
| 双模态残缺 | v11c | 0.062700 | 0.349800 |
| 仅音频残缺 | control | 0.047400 | 0.286400 |
| 仅音频残缺 | v11c | 0.047400 | 0.286500 |
| 仅图像残缺 | control | 0.062300 | 0.348300 |
| 仅图像残缺 | v11c | 0.062600 | 0.349000 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043000 | 0.155600 | 1.000000 |
| 干净双模态 | v11c | 0.043600 | 0.155500 | 1.000000 |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.994300 |
| 仅干净图像 | v11c | 0.015700 | 0.070400 | 0.993400 |
| 仅干净音频 | control | 0.043000 | 0.155700 | 1.000000 |
| 仅干净音频 | v11c | 0.043400 | 0.155400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.041200 | 0.152200 | 1.000000 |
| 图像干净、音频残缺 | v11c | 0.041800 | 0.152100 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043000 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11c | 0.043600 | 0.155600 | 1.000000 |
| 双模态残缺 | control | 0.041200 | 0.152200 | 1.000000 |
| 双模态残缺 | v11c | 0.041800 | 0.152100 | 1.000000 |
| 仅音频残缺 | control | 0.041300 | 0.152300 | 1.000000 |
| 仅音频残缺 | v11c | 0.041700 | 0.152000 | 1.000000 |
| 仅图像残缺 | control | 0.015500 | 0.070400 | 0.995400 |
| 仅图像残缺 | v11c | 0.015700 | 0.070300 | 0.994500 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 干净双模态 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 92.300% |
| 仅干净图像 | v11c | 0.015600 | 0.070500 | 0.987900 | 92.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 81.800% |
| 仅干净音频 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.300% |
| 图像干净、音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 音频干净、图像残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.300% |
| 双模态残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 81.800% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.300% |
| 仅音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 91.300% |
| 仅图像残缺 | v11c | 0.015600 | 0.070500 | 0.987900 | 91.300% |

</details>


<details>
<summary>family 4：mask_horizontal/partial_temporal</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.400% | 0.008800 | 0.950000 | 0.005200 | 0.875000 |
| 干净双模态 | v11c | 99.400% | 0.009100 | 0.946000 | 0.005100 | 0.876000 |
| 仅干净图像 | control | 98.600% | 0.008800 | 0.950000 | 0.000100 | 0.990000 |
| 仅干净图像 | v11c | 98.600% | 0.009100 | 0.946000 | 0.000100 | 0.990000 |
| 仅干净音频 | control | 97.700% | 0.001600 | 0.988000 | 0.005200 | 0.875000 |
| 仅干净音频 | v11c | 97.700% | 0.001600 | 0.988000 | 0.005100 | 0.876000 |
| 图像干净、音频残缺 | control | 96.700% | 0.008900 | 0.949000 | 0.012500 | 0.533000 |
| 图像干净、音频残缺 | v11c | 96.700% | 0.009200 | 0.946000 | 0.013300 | 0.536000 |
| 音频干净、图像残缺 | control | 99.200% | 0.008800 | 0.949000 | 0.005200 | 0.875000 |
| 音频干净、图像残缺 | v11c | 99.200% | 0.008800 | 0.949000 | 0.005100 | 0.876000 |
| 双模态残缺 | control | 94.100% | 0.009000 | 0.949000 | 0.012600 | 0.530000 |
| 双模态残缺 | v11c | 94.100% | 0.009000 | 0.949000 | 0.013400 | 0.535000 |
| 仅音频残缺 | control | 47.000% | 0.033600 | 0.708000 | 0.013200 | 0.509000 |
| 仅音频残缺 | v11c | 47.000% | 0.033600 | 0.715000 | 0.013900 | 0.513000 |
| 仅图像残缺 | control | 95.700% | 0.009000 | 0.948000 | 0.000300 | 0.968000 |
| 仅图像残缺 | v11c | 95.700% | 0.009000 | 0.948000 | 0.000300 | 0.968000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.410000 | N/A | N/A |
| 干净双模态 | v11c | 21.180000 | N/A | N/A |
| 仅干净图像 | control | 21.410000 | N/A | N/A |
| 仅干净图像 | v11c | 21.180000 | N/A | N/A |
| 仅干净音频 | control | 58.330000 | 0.001600 | 0.001600 |
| 仅干净音频 | v11c | 54.910000 | 0.001600 | 0.001600 |
| 图像干净、音频残缺 | control | 21.370000 | N/A | N/A |
| 图像干净、音频残缺 | v11c | 21.130000 | N/A | N/A |
| 音频干净、图像残缺 | control | 21.560000 | 0.044800 | 0.022500 |
| 音频干净、图像残缺 | v11c | 21.620000 | 0.044400 | 0.022500 |
| 双模态残缺 | control | 21.500000 | 0.045700 | 0.022800 |
| 双模态残缺 | v11c | 21.530000 | 0.045400 | 0.022900 |
| 仅音频残缺 | control | 27.910000 | 0.033600 | 0.033600 |
| 仅音频残缺 | v11c | 26.960000 | 0.033600 | 0.033600 |
| 仅图像残缺 | control | 21.480000 | 0.045300 | 0.022900 |
| 仅图像残缺 | v11c | 21.530000 | 0.045100 | 0.023000 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11c | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11c | N/A | N/A |
| 音频干净、图像残缺 | control | 0.023800 | 0.000000 |
| 音频干净、图像残缺 | v11c | 0.025100 | 0.000000 |
| 双模态残缺 | control | 0.024200 | 0.000000 |
| 双模态残缺 | v11c | 0.025500 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11c | N/A | N/A |
| 仅图像残缺 | control | 0.024000 | 0.000000 |
| 仅图像残缺 | v11c | 0.025300 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | 0.000100 | N/A |
| 仅干净图像 | v11c | 0.000100 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | 0.030400 | 0.000300 |
| 图像干净、音频残缺 | v11c | 0.032100 | 0.000500 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11c | N/A | N/A |
| 双模态残缺 | control | 0.030600 | 0.000300 |
| 双模态残缺 | v11c | 0.032300 | 0.000500 |
| 仅音频残缺 | control | 0.032100 | 0.000300 |
| 仅音频残缺 | v11c | 0.033600 | 0.000500 |
| 仅图像残缺 | control | 0.000300 | N/A |
| 仅图像残缺 | v11c | 0.000300 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.060900 | 0.345000 |
| 干净双模态 | v11c | 0.058700 | 0.338100 |
| 仅干净图像 | control | 0.060900 | 0.344200 |
| 仅干净图像 | v11c | 0.058600 | 0.337400 |
| 仅干净音频 | control | 0.047400 | 0.292300 |
| 仅干净音频 | v11c | 0.047400 | 0.292500 |
| 图像干净、音频残缺 | control | 0.060600 | 0.343300 |
| 图像干净、音频残缺 | v11c | 0.058300 | 0.336600 |
| 音频干净、图像残缺 | control | 0.059900 | 0.343100 |
| 音频干净、图像残缺 | v11c | 0.060300 | 0.344100 |
| 双模态残缺 | control | 0.059700 | 0.342200 |
| 双模态残缺 | v11c | 0.060000 | 0.343200 |
| 仅音频残缺 | control | 0.027400 | 0.205900 |
| 仅音频残缺 | v11c | 0.027200 | 0.205200 |
| 仅图像残缺 | control | 0.059800 | 0.342600 |
| 仅图像残缺 | v11c | 0.060100 | 0.343600 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043000 | 0.155600 | 1.000000 |
| 干净双模态 | v11c | 0.043600 | 0.155500 | 1.000000 |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.994300 |
| 仅干净图像 | v11c | 0.015700 | 0.070400 | 0.993400 |
| 仅干净音频 | control | 0.043000 | 0.155700 | 1.000000 |
| 仅干净音频 | v11c | 0.043400 | 0.155400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.037800 | 0.133500 | 1.000000 |
| 图像干净、音频残缺 | v11c | 0.032800 | 0.125200 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043000 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11c | 0.043600 | 0.155600 | 1.000000 |
| 双模态残缺 | control | 0.037900 | 0.133500 | 1.000000 |
| 双模态残缺 | v11c | 0.032800 | 0.125300 | 1.000000 |
| 仅音频残缺 | control | 0.037200 | 0.132400 | 1.000000 |
| 仅音频残缺 | v11c | 0.032500 | 0.124900 | 1.000000 |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.995000 |
| 仅图像残缺 | v11c | 0.015700 | 0.070300 | 0.993900 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 干净双模态 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 92.300% |
| 仅干净图像 | v11c | 0.015600 | 0.070500 | 0.987900 | 92.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 81.800% |
| 仅干净音频 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 71.300% |
| 图像干净、音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 70.000% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 音频干净、图像残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 71.100% |
| 双模态残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 69.800% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 69.300% |
| 仅音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 67.900% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 90.800% |
| 仅图像残缺 | v11c | 0.015600 | 0.070500 | 0.987900 | 90.900% |

</details>


<details>
<summary>family 5：salt_mask/time_freq_block</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.400% | 0.008800 | 0.950000 | 0.005200 | 0.875000 |
| 干净双模态 | v11c | 99.400% | 0.009100 | 0.946000 | 0.005100 | 0.876000 |
| 仅干净图像 | control | 98.600% | 0.008800 | 0.950000 | 0.000100 | 0.990000 |
| 仅干净图像 | v11c | 98.600% | 0.009100 | 0.946000 | 0.000100 | 0.990000 |
| 仅干净音频 | control | 97.700% | 0.001600 | 0.988000 | 0.005200 | 0.875000 |
| 仅干净音频 | v11c | 97.700% | 0.001600 | 0.988000 | 0.005100 | 0.876000 |
| 图像干净、音频残缺 | control | 99.300% | 0.008800 | 0.950000 | 0.005300 | 0.868000 |
| 图像干净、音频残缺 | v11c | 99.300% | 0.009100 | 0.946000 | 0.005200 | 0.870000 |
| 音频干净、图像残缺 | control | 98.600% | 0.001900 | 0.989000 | 0.005200 | 0.875000 |
| 音频干净、图像残缺 | v11c | 98.600% | 0.001900 | 0.989000 | 0.005100 | 0.876000 |
| 双模态残缺 | control | 98.400% | 0.001900 | 0.989000 | 0.005300 | 0.868000 |
| 双模态残缺 | v11c | 98.400% | 0.001900 | 0.989000 | 0.005300 | 0.869000 |
| 仅音频残缺 | control | 96.900% | 0.002500 | 0.982000 | 0.005300 | 0.868000 |
| 仅音频残缺 | v11c | 96.900% | 0.002500 | 0.982000 | 0.005300 | 0.869000 |
| 仅图像残缺 | control | 94.600% | 0.001900 | 0.989000 | 0.000400 | 0.957000 |
| 仅图像残缺 | v11c | 94.600% | 0.001900 | 0.989000 | 0.000400 | 0.956000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.410000 | N/A | N/A |
| 干净双模态 | v11c | 21.180000 | N/A | N/A |
| 仅干净图像 | control | 21.410000 | N/A | N/A |
| 仅干净图像 | v11c | 21.180000 | N/A | N/A |
| 仅干净音频 | control | 58.330000 | 0.001600 | 0.001600 |
| 仅干净音频 | v11c | 54.910000 | 0.001600 | 0.001600 |
| 图像干净、音频残缺 | control | 21.410000 | N/A | N/A |
| 图像干净、音频残缺 | v11c | 21.170000 | N/A | N/A |
| 音频干净、图像残缺 | control | 27.790000 | 0.081900 | 0.004800 |
| 音频干净、图像残缺 | v11c | 27.810000 | 0.080600 | 0.004700 |
| 双模态残缺 | control | 27.790000 | 0.081900 | 0.004800 |
| 双模态残缺 | v11c | 27.800000 | 0.080700 | 0.004800 |
| 仅音频残缺 | control | 57.210000 | 0.002500 | 0.002500 |
| 仅音频残缺 | v11c | 53.980000 | 0.002500 | 0.002500 |
| 仅图像残缺 | control | 27.780000 | 0.082900 | 0.004800 |
| 仅图像残缺 | v11c | 27.790000 | 0.081600 | 0.004800 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11c | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11c | N/A | N/A |
| 音频干净、图像残缺 | control | 0.081900 | 0.000000 |
| 音频干净、图像残缺 | v11c | 0.080600 | 0.000000 |
| 双模态残缺 | control | 0.081900 | 0.000000 |
| 双模态残缺 | v11c | 0.080600 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11c | N/A | N/A |
| 仅图像残缺 | control | 0.082900 | 0.000000 |
| 仅图像残缺 | v11c | 0.081600 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11c | N/A | N/A |
| 仅干净图像 | control | 0.000100 | N/A |
| 仅干净图像 | v11c | 0.000100 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11c | N/A | N/A |
| 图像干净、音频残缺 | control | 0.005900 | 0.005200 |
| 图像干净、音频残缺 | v11c | 0.005800 | 0.005100 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11c | N/A | N/A |
| 双模态残缺 | control | 0.005800 | 0.005200 |
| 双模态残缺 | v11c | 0.005700 | 0.005200 |
| 仅音频残缺 | control | 0.006100 | 0.005200 |
| 仅音频残缺 | v11c | 0.006000 | 0.005100 |
| 仅图像残缺 | control | 0.000400 | N/A |
| 仅图像残缺 | v11c | 0.000400 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.060900 | 0.344200 |
| 干净双模态 | v11c | 0.058700 | 0.337600 |
| 仅干净图像 | control | 0.060900 | 0.344700 |
| 仅干净图像 | v11c | 0.058600 | 0.338000 |
| 仅干净音频 | control | 0.047400 | 0.286000 |
| 仅干净音频 | v11c | 0.047400 | 0.286000 |
| 图像干净、音频残缺 | control | 0.060900 | 0.343800 |
| 图像干净、音频残缺 | v11c | 0.058600 | 0.337200 |
| 音频干净、图像残缺 | control | 0.066300 | 0.359400 |
| 音频干净、图像残缺 | v11c | 0.066300 | 0.359400 |
| 双模态残缺 | control | 0.066300 | 0.361900 |
| 双模态残缺 | v11c | 0.066300 | 0.361900 |
| 仅音频残缺 | control | 0.047400 | 0.292700 |
| 仅音频残缺 | v11c | 0.047400 | 0.292800 |
| 仅图像残缺 | control | 0.066300 | 0.359300 |
| 仅图像残缺 | v11c | 0.066300 | 0.359300 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043000 | 0.155600 | 1.000000 |
| 干净双模态 | v11c | 0.043600 | 0.155500 | 1.000000 |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.994300 |
| 仅干净图像 | v11c | 0.015700 | 0.070400 | 0.993400 |
| 仅干净音频 | control | 0.043000 | 0.155700 | 1.000000 |
| 仅干净音频 | v11c | 0.043400 | 0.155400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.042800 | 0.155200 | 1.000000 |
| 图像干净、音频残缺 | v11c | 0.043400 | 0.155100 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043000 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11c | 0.043600 | 0.155600 | 1.000000 |
| 双模态残缺 | control | 0.042900 | 0.155500 | 1.000000 |
| 双模态残缺 | v11c | 0.043500 | 0.155300 | 1.000000 |
| 仅音频残缺 | control | 0.042900 | 0.155300 | 1.000000 |
| 仅音频残缺 | v11c | 0.043300 | 0.155000 | 1.000000 |
| 仅图像残缺 | control | 0.015600 | 0.070100 | 0.996400 |
| 仅图像残缺 | v11c | 0.015700 | 0.069900 | 0.994400 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 干净双模态 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 92.300% |
| 仅干净图像 | v11c | 0.015600 | 0.070500 | 0.987900 | 92.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 81.800% |
| 仅干净音频 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.500% |
| 图像干净、音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 81.800% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 音频干净、图像残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.500% |
| 双模态残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 81.800% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.400% |
| 仅音频残缺 | v11c | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 89.500% |
| 仅图像残缺 | v11c | 0.015600 | 0.070500 | 0.987900 | 89.600% |

</details>

**独立音频 family breakdown**

此处来自各实验的 `tables/audio_family_breakdown_fixed.csv`，单独改变音频 family；不能用它替代上面的双 family-pair 主表或再次计入宏平均。该 CSV 不含逐指标 n。


<details>
<summary>time_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.060% | 0.008800 | 0.949637 | 0.007293 | 0.810528 |
| 图像干净、音频残缺 | v11c | 99.060% | 0.009146 | 0.946225 | 0.007497 | 0.804174 |
| 双模态残缺 | control | 98.210% | 0.005810 | 0.966421 | 0.007337 | 0.808858 |
| 双模态残缺 | v11c | 98.210% | 0.005805 | 0.966476 | 0.007541 | 0.802648 |
| 仅音频残缺 | control | 92.730% | 0.006378 | 0.953292 | 0.007395 | 0.808418 |
| 仅音频残缺 | v11c | 92.730% | 0.006370 | 0.953826 | 0.007577 | 0.802188 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 21.404885 | 0.009355 | 0.023940 |
| 图像干净、音频残缺 | v11c | 21.170320 | 0.009730 | 0.024074 |
| 双模态残缺 | control | 25.417149 | 0.009568 | 0.024452 |
| 双模态残缺 | v11c | 25.442569 | 0.009948 | 0.024564 |
| 仅音频残缺 | control | 53.529128 | 0.009584 | 0.024265 |
| 仅音频残缺 | v11c | 50.572641 | 0.009940 | 0.024343 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.005882 | 0.020372 |
| 图像干净、音频残缺 | v11c | 0.005969 | 0.020794 |
| 双模态残缺 | control | 0.005810 | 0.020174 |
| 双模态残缺 | v11c | 0.005893 | 0.020592 |
| 仅音频残缺 | control | 0.005897 | 0.020393 |
| 仅音频残缺 | v11c | 0.005960 | 0.020752 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.150601 | 0.152054 | 78.139% |
| 图像干净、音频残缺 | v11c | 0.149307 | 0.152054 | 77.997% |
| 双模态残缺 | control | 0.150682 | 0.152054 | 78.041% |
| 双模态残缺 | v11c | 0.149359 | 0.152054 | 77.867% |
| 仅音频残缺 | control | 0.150579 | 0.152054 | 77.940% |
| 仅音频残缺 | v11c | 0.149045 | 0.152054 | 77.751% |

</details>


<details>
<summary>freq_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.120% | 0.008798 | 0.949663 | 0.005887 | 0.839445 |
| 图像干净、音频残缺 | v11c | 99.120% | 0.009146 | 0.946252 | 0.005806 | 0.842421 |
| 双模态残缺 | control | 98.280% | 0.005899 | 0.965737 | 0.005946 | 0.838180 |
| 双模态残缺 | v11c | 98.280% | 0.005888 | 0.965826 | 0.005871 | 0.841262 |
| 仅音频残缺 | control | 94.870% | 0.004828 | 0.965359 | 0.005960 | 0.837740 |
| 仅音频残缺 | v11c | 94.870% | 0.004882 | 0.965040 | 0.005875 | 0.840310 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 21.405810 | 0.006988 | 0.021697 |
| 图像干净、音频残缺 | v11c | 21.169405 | 0.006852 | 0.021592 |
| 双模态残缺 | control | 25.308196 | 0.007080 | 0.021860 |
| 双模态残缺 | v11c | 25.333129 | 0.006958 | 0.021780 |
| 仅音频残缺 | control | 54.065979 | 0.007119 | 0.021907 |
| 仅音频残缺 | v11c | 51.191365 | 0.006996 | 0.021819 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.005134 | 0.018169 |
| 图像干净、音频残缺 | v11c | 0.005090 | 0.018255 |
| 双模态残缺 | control | 0.005171 | 0.018268 |
| 双模态残缺 | v11c | 0.005128 | 0.018359 |
| 仅音频残缺 | control | 0.005167 | 0.018221 |
| 仅音频残缺 | v11c | 0.005108 | 0.018266 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.154459 | 0.152054 | 80.470% |
| 图像干净、音频残缺 | v11c | 0.154234 | 0.152054 | 80.785% |
| 双模态残缺 | control | 0.154647 | 0.152054 | 80.413% |
| 双模态残缺 | v11c | 0.154399 | 0.152054 | 80.708% |
| 仅音频残缺 | control | 0.154597 | 0.152054 | 80.340% |
| 仅音频残缺 | v11c | 0.154080 | 0.152054 | 80.600% |

</details>


<details>
<summary>feature_dropout</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.320% | 0.008791 | 0.949700 | 0.005103 | 0.868530 |
| 图像干净、音频残缺 | v11c | 99.320% | 0.009135 | 0.946308 | 0.005008 | 0.872205 |
| 双模态残缺 | control | 99.030% | 0.005831 | 0.966026 | 0.005117 | 0.868352 |
| 双模态残缺 | v11c | 99.030% | 0.005827 | 0.966085 | 0.005023 | 0.872035 |
| 仅音频残缺 | control | 97.600% | 0.001944 | 0.986024 | 0.005126 | 0.868132 |
| 仅音频残缺 | v11c | 97.600% | 0.001962 | 0.985931 | 0.005026 | 0.871340 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 21.410037 | 0.005158 | 0.018051 |
| 图像干净、音频残缺 | v11c | 21.175723 | 0.005061 | 0.017986 |
| 双模态残缺 | control | 25.468283 | 0.005170 | 0.018072 |
| 双模态残缺 | v11c | 25.495747 | 0.005074 | 0.018013 |
| 仅音频残缺 | control | 57.224148 | 0.005180 | 0.018085 |
| 仅音频残缺 | v11c | 53.992903 | 0.005079 | 0.018004 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.005066 | 0.017880 |
| 图像干净、音频残缺 | v11c | 0.004973 | 0.017821 |
| 双模态残缺 | control | 0.005082 | 0.017921 |
| 双模态残缺 | v11c | 0.004988 | 0.017866 |
| 仅音频残缺 | control | 0.005090 | 0.017913 |
| 仅音频残缺 | v11c | 0.004991 | 0.017836 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.152062 | 0.152054 | 81.327% |
| 图像干净、音频残缺 | v11c | 0.151929 | 0.152054 | 81.744% |
| 双模态残缺 | control | 0.152288 | 0.152054 | 81.324% |
| 双模态残缺 | v11c | 0.152149 | 0.152054 | 81.743% |
| 仅音频残缺 | control | 0.152141 | 0.152054 | 81.308% |
| 仅音频残缺 | v11c | 0.151776 | 0.152054 | 81.697% |

</details>


<details>
<summary>partial_temporal</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 96.670% | 0.008851 | 0.949169 | 0.012546 | 0.532745 |
| 图像干净、音频残缺 | v11c | 96.670% | 0.009211 | 0.945622 | 0.013312 | 0.536216 |
| 双模态残缺 | control | 92.260% | 0.005971 | 0.965323 | 0.012661 | 0.529119 |
| 双模态残缺 | v11c | 92.260% | 0.005996 | 0.965231 | 0.013416 | 0.534311 |
| 仅音频残缺 | control | 46.990% | 0.033557 | 0.707553 | 0.013236 | 0.509147 |
| 仅音频残缺 | v11c | 46.990% | 0.033601 | 0.715109 | 0.013931 | 0.512972 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 21.374938 | 0.030419 | 0.076276 |
| 图像干净、音频残缺 | v11c | 21.133121 | 0.032087 | 0.074700 |
| 双模态残缺 | control | 25.270874 | 0.030699 | 0.076751 |
| 双模态残缺 | v11c | 25.284814 | 0.032338 | 0.075096 |
| 仅音频残缺 | control | 27.911753 | 0.032112 | 0.078523 |
| 仅音频残缺 | v11c | 26.962645 | 0.033590 | 0.076838 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.000318 | 0.001568 |
| 图像干净、音频残缺 | v11c | 0.000466 | 0.002116 |
| 双模态残缺 | control | 0.000319 | 0.001574 |
| 双模态残缺 | v11c | 0.000469 | 0.002128 |
| 仅音频残缺 | control | 0.000321 | 0.001580 |
| 仅音频残缺 | v11c | 0.000480 | 0.002173 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.133493 | 0.152054 | 71.288% |
| 图像干净、音频残缺 | v11c | 0.125240 | 0.152054 | 70.012% |
| 双模态残缺 | control | 0.133467 | 0.152054 | 71.022% |
| 双模态残缺 | v11c | 0.125318 | 0.152054 | 69.731% |
| 仅音频残缺 | control | 0.132352 | 0.152054 | 69.255% |
| 仅音频残缺 | v11c | 0.124863 | 0.152054 | 67.912% |

</details>


<details>
<summary>time_freq_block</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.230% | 0.008788 | 0.949732 | 0.005301 | 0.868744 |
| 图像干净、音频残缺 | v11c | 99.230% | 0.009133 | 0.946330 | 0.005238 | 0.869787 |
| 双模态残缺 | control | 98.920% | 0.005868 | 0.966053 | 0.005307 | 0.868567 |
| 双模态残缺 | v11c | 98.920% | 0.005865 | 0.966124 | 0.005244 | 0.869735 |
| 仅音频残缺 | control | 97.080% | 0.002365 | 0.982709 | 0.005332 | 0.868372 |
| 仅音频残缺 | v11c | 97.080% | 0.002350 | 0.983011 | 0.005258 | 0.869695 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 21.410563 | 0.005950 | 0.018714 |
| 图像干净、音频残缺 | v11c | 21.175552 | 0.005840 | 0.018639 |
| 双模态残缺 | control | 25.452266 | 0.005859 | 0.018448 |
| 双模态残缺 | v11c | 25.498128 | 0.005758 | 0.018377 |
| 仅音频残缺 | control | 57.294569 | 0.006055 | 0.018866 |
| 仅音频残缺 | v11c | 54.045673 | 0.005943 | 0.018782 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.005173 | 0.018325 |
| 图像干净、音频残缺 | v11c | 0.005119 | 0.018404 |
| 双模态残缺 | control | 0.005198 | 0.018389 |
| 双模态残缺 | v11c | 0.005142 | 0.018462 |
| 仅音频残缺 | control | 0.005189 | 0.018348 |
| 仅音频残缺 | v11c | 0.005122 | 0.018388 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.155251 | 0.152054 | 81.482% |
| 图像干净、音频残缺 | v11c | 0.155131 | 0.152054 | 81.787% |
| 双模态残缺 | control | 0.155352 | 0.152054 | 81.465% |
| 双模态残缺 | v11c | 0.155256 | 0.152054 | 81.780% |
| 仅音频残缺 | control | 0.155301 | 0.152054 | 81.441% |
| 仅音频残缺 | v11c | 0.154977 | 0.152054 | 81.712% |

</details>

**训练统计**

仅统计每个完整 epoch 的平均训练 loss；不同 loss 定义不可横比，最低训练 loss 也不是测试最优 checkpoint。

| 实验 | 完成 epoch | 本轮轮数 | 首轮 loss | 末轮 loss | 最低 loss | 末轮 LR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| control | 120–149 | 30 | 0.9865 | 0.9678 | 0.9094 | 0.000100 |
| v11c | 120–149 | 30 | 1.0538 | 0.9754 | 0.9567 | 0.000100 |


<details>
<summary>逐 epoch loss / LR</summary>

| epoch | 实验 | 平均 loss | LR |
| --- | --- | ---: | ---: |
| 120 | control | 0.9865 | 0.000100 |
| 120 | v11c | 1.0538 | 0.000100 |
| 121 | control | 1.0295 | 0.000100 |
| 121 | v11c | 1.0737 | 0.000100 |
| 122 | control | 1.0818 | 0.000100 |
| 122 | v11c | 1.1303 | 0.000100 |
| 123 | control | 0.9637 | 0.000100 |
| 123 | v11c | 1.0110 | 0.000100 |
| 124 | control | 1.0233 | 0.000100 |
| 124 | v11c | 1.0722 | 0.000100 |
| 125 | control | 0.9692 | 0.000100 |
| 125 | v11c | 1.0189 | 0.000100 |
| 126 | control | 1.0477 | 0.000100 |
| 126 | v11c | 1.1032 | 0.000100 |
| 127 | control | 1.0351 | 0.000100 |
| 127 | v11c | 1.0891 | 0.000100 |
| 128 | control | 0.9646 | 0.000100 |
| 128 | v11c | 1.0182 | 0.000100 |
| 129 | control | 0.9862 | 0.000100 |
| 129 | v11c | 1.0026 | 0.000100 |
| 130 | control | 0.9859 | 0.000100 |
| 130 | v11c | 1.0497 | 0.000100 |
| 131 | control | 0.9397 | 0.000100 |
| 131 | v11c | 1.1065 | 0.000100 |
| 132 | control | 0.9691 | 0.000100 |
| 132 | v11c | 0.9923 | 0.000100 |
| 133 | control | 1.0247 | 0.000100 |
| 133 | v11c | 1.0553 | 0.000100 |
| 134 | control | 0.9317 | 0.000100 |
| 134 | v11c | 1.0038 | 0.000100 |
| 135 | control | 0.9668 | 0.000100 |
| 135 | v11c | 1.0869 | 0.000100 |
| 136 | control | 0.9111 | 0.000100 |
| 136 | v11c | 1.0742 | 0.000100 |
| 137 | control | 0.9128 | 0.000100 |
| 137 | v11c | 1.0039 | 0.000100 |
| 138 | control | 0.9103 | 0.000100 |
| 138 | v11c | 1.0274 | 0.000100 |
| 139 | control | 0.9357 | 0.000100 |
| 139 | v11c | 1.0261 | 0.000100 |
| 140 | control | 0.9276 | 0.000100 |
| 140 | v11c | 0.9834 | 0.000100 |
| 141 | control | 0.9322 | 0.000100 |
| 141 | v11c | 1.0130 | 0.000100 |
| 142 | control | 0.9426 | 0.000100 |
| 142 | v11c | 1.0721 | 0.000100 |
| 143 | control | 0.9816 | 0.000100 |
| 143 | v11c | 0.9795 | 0.000100 |
| 144 | control | 0.9802 | 0.000100 |
| 144 | v11c | 1.0144 | 0.000100 |
| 145 | control | 0.9489 | 0.000100 |
| 145 | v11c | 0.9567 | 0.000100 |
| 146 | control | 0.9734 | 0.000100 |
| 146 | v11c | 0.9594 | 0.000100 |
| 147 | control | 0.9324 | 0.000100 |
| 147 | v11c | 0.9568 | 0.000100 |
| 148 | control | 0.9094 | 0.000100 |
| 148 | v11c | 0.9839 | 0.000100 |
| 149 | control | 0.9678 | 0.000100 |
| 149 | v11c | 0.9754 | 0.000100 |

</details>

训练证据：
- `control`：[train_v11c_control.log](../v11cnew_outputs_with_ckpt/outputs/outputs_v11c_control/logs/train_v11c_control.log)
- `v11c`：[train_v11c.log](../v11cnew_outputs_with_ckpt/outputs/outputs_v11c/logs/train_v11c.log)

**Demo 小样本**

每份 demo 为 10 条样本，不是全测试集分数；fixed/random 分开。原始图片与逐样本预测留在对应产物目录，本节不宣称重新进行了图像质量评审。


<details>
<summary>fixed_mask demo (n=10)</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 90.000% | 0.008400 | 0.929000 | 0.012100 | 0.693000 |
| 仅音频残缺 | v11c | 90.000% | 0.006800 | 0.942000 | 0.012600 | 0.665000 |
| 仅图像残缺 | control | 100.000% | 0.004900 | 0.968000 | 0.000000 | 0.999000 |
| 仅图像残缺 | v11c | 100.000% | 0.005400 | 0.965000 | 0.000000 | 1.000000 |
| 双模态残缺 | control | 100.000% | 0.004800 | 0.969000 | 0.011700 | 0.709000 |
| 双模态残缺 | v11c | 100.000% | 0.005300 | 0.966000 | 0.012100 | 0.682000 |

| 输入模式 | 实验 | 图像缺失区 MSE | 音频缺失区 MSE |
| --- | --- | ---: | ---: |
| 仅音频残缺 | control | 0.008400 | 0.024500 |
| 仅音频残缺 | v11c | 0.006800 | 0.025900 |
| 仅图像残缺 | control | 0.015200 | 0.000000 |
| 仅图像残缺 | v11c | 0.016600 | 0.000000 |
| 双模态残缺 | control | 0.014900 | 0.023300 |
| 双模态残缺 | v11c | 0.016300 | 0.024400 |

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.042500 | 0.155700 | 1.000000 |
| 仅音频残缺 | v11c | 0.041400 | 0.153400 | 1.000000 |
| 仅图像残缺 | control | 0.020700 | 0.084100 | 1.000000 |
| 仅图像残缺 | v11c | 0.020800 | 0.084000 | 1.000000 |
| 双模态残缺 | control | 0.043800 | 0.157700 | 1.000000 |
| 双模态残缺 | v11c | 0.042600 | 0.154600 | 1.000000 |

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.047400 | 0.168700 | 1.000000 | 77.200% |
| 仅音频残缺 | v11c | 0.047400 | 0.168700 | 1.000000 | 77.300% |
| 仅图像残缺 | control | 0.020700 | 0.084100 | 0.987900 | 94.500% |
| 仅图像残缺 | v11c | 0.020700 | 0.084100 | 0.987900 | 94.600% |
| 双模态残缺 | control | 0.047400 | 0.168700 | 1.000000 | 77.100% |
| 双模态残缺 | v11c | 0.047400 | 0.168700 | 1.000000 | 77.100% |

</details>

- legacy_random demo：未找到。

**各实验结论与比较限制**

- **仅音频残缺**：主实验相对 control，音频缺失区 MSE 变化 +2.50%，Index ACC 变化 +0.000 个百分点。
- **双模态残缺**：主实验相对 control，音频缺失区 MSE 变化 +3.26%，Index ACC 变化 +0.000 个百分点。
- **control**：同一 v11b_recovery 起点额外训练 30 轮，关闭 Cross-Key，是同预算参考。
- **v11c**：保留 Cross-Key，但现有结果不能证明它全面优于 control；缺少同 checkpoint 的 zero/wrong/same-class 扫描，不能从主表推断通路因果作用。
- control 的 `train_v11c_control_ep120_to_150.log` 是缺少 `libcudart.so.13` 的失败尝试；完成训练的依据是 `train_v11c_control.log`，不是失败文件名。
- 本节仅汇总已有结果，没有重跑训练或推理。单 seed、重复音频曝光和旧日志舍入限制仍在，不报告统计显著性。

[返回统一评估导航](#evaluation-format-20260912)

<a id="evaluation-v11d"></a>

### v11d

本节覆盖 **2 组实验、32 个主评估/干预字段、7 个音频分布诊断字段**，以及独立 family、训练统计和本地已有 demo。字段按实际产物统计，含明确标注的不适用项；不把其他版本的缺项补成零。

**实验说明**

| 实验 | 权重起点 | 本轮训练范围 | 实际轮数 |
| --- | --- | ---: | ---: |
| control | v11c（epoch 149 后） | 伪配对 + Cross-Key；无 Cross-Detail | 50（150–199） |
| v11d | 同一 v11c | 伪配对 + Cross-Key + Cross-Detail | 50（150–199） |

训练配置 seed=1234、batch_size=128；fixed_mask seed=1234、severity=0.4。轮数以上表和完成日志为准，不把父模型历史轮数计成本轮新增训练。历史分支配置不是当前分支的运行入口。

**恢复目标：全部模式为 sample/sample，使用固定伪实例配对。** 它不等同于其它四版的类别级绑定。

| 实验 | fixed / random | Cross-Key 扫描 | demo fixed / random |
| --- | --- | ---: | ---: |
| control | 有 / 未找到 | fixed | 有 / 未找到 |
| v11d | 有 / 未找到 | fixed | 有 / 未找到 |

**数据来源**：
- `control`：[本地产物](../v11d_outputs_with_ckpt/outputs/outputs_v11d_control/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。
- `v11d`：[本地产物](../v11d_outputs_with_ckpt/outputs/outputs_v11d/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。

MSE/L1 越低越好，ACC/SSIM/PSNR 越高越好；gate、res/V 和多样性仅是诊断。宏平均为五个 family 等权均值，不是把 normal、sweep、独立音频 family 重复叠加。N/A 表示未保存或在该场景不适用，详见字段覆盖说明。

**精度与缺项**：主表源 CSV/日志已舍入（ACC 常到 0.1%、MSE 常到 4 位、SSIM 常到 3 位）；本节六位显示只是统一排版，不恢复丢失精度。未保存逐指标有效 n、区域 L1、内容 ACC、干预绝对误差/win 等字段不编造。独立音频 family CSV 中实际存在的 L1 仍在其专表报告。

**分类与恢复：fixed 五类等权宏平均**

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 95.700% | 0.009100 | 0.947000 | 0.005700 | 0.837000 |
| 干净双模态 | v11d | 95.700% | 0.009300 | 0.945000 | 0.005700 | 0.838000 |
| 仅干净图像 | control | 98.600% | 0.009100 | 0.947000 | 0.018600 | 0.261000 |
| 仅干净图像 | v11d | 98.600% | 0.009300 | 0.945000 | 0.019400 | 0.255000 |
| 仅干净音频 | control | 64.700% | 0.058900 | 0.535000 | 0.006000 | 0.828000 |
| 仅干净音频 | v11d | 64.700% | 0.056700 | 0.562000 | 0.006100 | 0.826000 |
| 图像干净、音频残缺 | control | 94.460% | 0.009140 | 0.946800 | 0.008300 | 0.741400 |
| 图像干净、音频残缺 | v11d | 94.460% | 0.009320 | 0.945000 | 0.008400 | 0.737200 |
| 音频干净、图像残缺 | control | 89.940% | 0.004820 | 0.972600 | 0.005720 | 0.836000 |
| 音频干净、图像残缺 | v11d | 89.940% | 0.004820 | 0.972600 | 0.005760 | 0.836000 |
| 双模态残缺 | control | 87.600% | 0.004840 | 0.972800 | 0.008400 | 0.739000 |
| 双模态残缺 | v11d | 87.600% | 0.004840 | 0.972800 | 0.008380 | 0.739400 |
| 仅音频残缺 | control | 52.240% | 0.062000 | 0.505200 | 0.008960 | 0.726600 |
| 仅音频残缺 | v11d | 52.240% | 0.060260 | 0.524800 | 0.009000 | 0.724400 |
| 仅图像残缺 | control | 95.500% | 0.004880 | 0.972200 | 0.018640 | 0.260200 |
| 仅图像残缺 | v11d | 95.500% | 0.004900 | 0.972200 | 0.019440 | 0.253600 |


<details>
<summary>fixed 五类等权宏平均：区域、内容、多样性与音频分布完整指标</summary>

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.200000 | N/A | N/A |
| 干净双模态 | v11d | 21.090000 | N/A | N/A |
| 仅干净图像 | control | 21.220000 | N/A | N/A |
| 仅干净图像 | v11d | 21.110000 | N/A | N/A |
| 仅干净音频 | control | 12.580000 | 0.058900 | 0.058900 |
| 仅干净音频 | v11d | 12.770000 | 0.056700 | 0.056700 |
| 图像干净、音频残缺 | control | 21.192000 | N/A | N/A |
| 图像干净、音频残缺 | v11d | 21.082000 | N/A | N/A |
| 音频干净、图像残缺 | control | 25.938000 | 0.056140 | 0.016960 |
| 音频干净、图像残缺 | v11d | 25.948000 | 0.056540 | 0.016960 |
| 双模态残缺 | control | 25.950000 | 0.056140 | 0.016980 |
| 双模态残缺 | v11d | 25.956000 | 0.056800 | 0.016960 |
| 仅音频残缺 | control | 12.356000 | 0.062000 | 0.062000 |
| 仅音频残缺 | v11d | 12.492000 | 0.060260 | 0.060260 |
| 仅图像残缺 | control | 25.914000 | 0.056300 | 0.017240 |
| 仅图像残缺 | v11d | 25.906000 | 0.056840 | 0.017300 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.031820 | 0.000000 |
| 音频干净、图像残缺 | v11d | 0.032140 | 0.000000 |
| 双模态残缺 | control | 0.031820 | 0.000000 |
| 双模态残缺 | v11d | 0.032320 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.031800 | 0.000000 |
| 仅图像残缺 | v11d | 0.032260 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.018600 | N/A |
| 仅干净图像 | v11d | 0.019400 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.013180 | 0.005140 |
| 图像干净、音频残缺 | v11d | 0.013340 | 0.005100 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.013320 | 0.005100 |
| 双模态残缺 | v11d | 0.013320 | 0.005140 |
| 仅音频残缺 | control | 0.014380 | 0.005380 |
| 仅音频残缺 | v11d | 0.014380 | 0.005440 |
| 仅图像残缺 | control | 0.018640 | N/A |
| 仅图像残缺 | v11d | 0.019440 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.059200 | 0.339520 |
| 干净双模态 | v11d | 0.058700 | 0.338240 |
| 仅干净图像 | control | 0.059300 | 0.339800 |
| 仅干净图像 | v11d | 0.058300 | 0.336840 |
| 仅干净音频 | control | 0.007900 | 0.115140 |
| 仅干净音频 | v11d | 0.011300 | 0.140140 |
| 图像干净、音频残缺 | control | 0.059140 | 0.339060 |
| 图像干净、音频残缺 | v11d | 0.058400 | 0.336940 |
| 音频干净、图像残缺 | control | 0.063920 | 0.353320 |
| 音频干净、图像残缺 | v11d | 0.064000 | 0.353440 |
| 双模态残缺 | control | 0.063900 | 0.353680 |
| 双模态残缺 | v11d | 0.063900 | 0.353740 |
| 仅音频残缺 | control | 0.006580 | 0.104300 |
| 仅音频残缺 | v11d | 0.008480 | 0.119700 |
| 仅图像残缺 | control | 0.063920 | 0.353320 |
| 仅图像残缺 | v11d | 0.063860 | 0.353160 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.047200 | 0.146200 | 1.000000 |
| 干净双模态 | v11d | 0.047200 | 0.145900 | 1.000000 |
| 仅干净图像 | control | 0.021300 | 0.041000 | 0.358400 |
| 仅干净图像 | v11d | 0.017900 | 0.035000 | 0.342300 |
| 仅干净音频 | control | 0.048300 | 0.147600 | 1.000000 |
| 仅干净音频 | v11d | 0.048500 | 0.147800 | 1.000000 |
| 图像干净、音频残缺 | control | 0.042900 | 0.132320 | 0.999940 |
| 图像干净、音频残缺 | v11d | 0.042000 | 0.130440 | 0.999940 |
| 音频干净、图像残缺 | control | 0.047240 | 0.146260 | 1.000000 |
| 音频干净、图像残缺 | v11d | 0.047460 | 0.146280 | 1.000000 |
| 双模态残缺 | control | 0.042880 | 0.132080 | 0.999900 |
| 双模态残缺 | v11d | 0.043020 | 0.132140 | 0.999940 |
| 仅音频残缺 | control | 0.044200 | 0.134000 | 0.999960 |
| 仅音频残缺 | v11d | 0.044240 | 0.133880 | 0.999960 |
| 仅图像残缺 | control | 0.021400 | 0.040920 | 0.362380 |
| 仅图像残缺 | v11d | 0.017860 | 0.034140 | 0.340080 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 干净双模态 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 仅干净图像 | control | 0.042600 | 0.148600 | 1.000000 | 48.400% |
| 仅干净图像 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.300% |
| 仅干净音频 | control | 0.042600 | 0.148600 | 1.000000 | 62.900% |
| 仅干净音频 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.800% |
| 图像干净、音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 59.820% |
| 图像干净、音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 59.820% |
| 音频干净、图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 63.200% |
| 音频干净、图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.220% |
| 双模态残缺 | control | 0.042600 | 0.148600 | 1.000000 | 59.680% |
| 双模态残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 59.680% |
| 仅音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 58.840% |
| 仅音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 58.800% |
| 仅图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 48.200% |
| 仅图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.100% |

</details>


<details>
<summary>fixed 五类等权宏平均：各字段有效样本数</summary>

| 实验 | 指标字段（同一计数口径） | 每 family 的 n |
| --- | --- | ---: |
| control | `acc`、`aud_masked_mse`、`aud_mse` | 未逐指标保存 |
| control | `aud_ssim`、`aud_visible_mse`、`img_coarse_masked_mse` | 未逐指标保存 |
| control | `img_coarse_visible_mse`、`img_masked_mse`、`img_mse` | 未逐指标保存 |
| control | `img_visible_mse`、`pair_l2`、`pix_var` | 未逐指标保存 |
| control | `psnr`、`rec_max`、`rec_mean` | 未逐指标保存 |
| control | `rec_std`、`ssim`、`tgt_max` | 未逐指标保存 |
| control | `tgt_mean`、`tgt_std`、`top15_recall` | 未逐指标保存 |
| v11d | `acc`、`aud_masked_mse`、`aud_mse` | 未逐指标保存 |
| v11d | `aud_ssim`、`aud_visible_mse`、`img_coarse_masked_mse` | 未逐指标保存 |
| v11d | `img_coarse_visible_mse`、`img_masked_mse`、`img_mse` | 未逐指标保存 |
| v11d | `img_visible_mse`、`pair_l2`、`pix_var` | 未逐指标保存 |
| v11d | `psnr`、`rec_max`、`rec_mean` | 未逐指标保存 |
| v11d | `rec_std`、`ssim`、`tgt_max` | 未逐指标保存 |
| v11d | `tgt_mean`、`tgt_std`、`top15_recall` | 未逐指标保存 |

</details>

**分类与恢复：random 单次评估**

本地产物未找到此协议结果，不将 fixed 复制成 random。

**Cross-Key / Cross-Detail 干预**

同一 cue/mask 内，gain=zero−normal，wrong damage=wrong−normal，same damage 为有效同类替换上的配对差；正 gain 表示改善。gate 非零本身不代表恢复有效。


<details>
<summary>fixed 五类等权宏平均：干预完整指标</summary>

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A | N/A |
| 仅干净音频 | control | 0.000100 | 0.000100 | -0.000300 |
| 仅干净音频 | v11d | 0.000000 | 0.000000 | -0.000200 |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.000020 | 0.000020 | -0.000020 |
| 音频干净、图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | control | 0.000000 | 0.000000 | -0.000020 |
| 双模态残缺 | v11d | 0.000000 | 0.000000 | -0.000020 |
| 仅音频残缺 | control | -0.000040 | 0.000040 | -0.000260 |
| 仅音频残缺 | v11d | 0.000000 | 0.000000 | -0.000180 |
| 仅图像残缺 | control | N/A | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | 0.011000 | 0.028400 |
| 仅干净音频 | v11d | 0.011000 | 0.028400 |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.013780 | 0.034340 |
| 音频干净、图像残缺 | v11d | 0.013780 | 0.034340 |
| 双模态残缺 | control | 0.015300 | 0.035980 |
| 双模态残缺 | v11d | 0.015300 | 0.035980 |
| 仅音频残缺 | control | 0.012060 | 0.030300 |
| 仅音频残缺 | v11d | 0.012060 | 0.030300 |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.000180 | 0.000300 | 0.000000 |
| 图像干净、音频残缺 | v11d | 0.000140 | 0.000260 | 0.000000 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A | N/A |
| 双模态残缺 | control | 0.000200 | 0.000300 | 0.000000 |
| 双模态残缺 | v11d | 0.000160 | 0.000240 | -0.000020 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.042900 | 0.149700 |
| 仅干净图像 | v11d | 0.042900 | 0.149700 |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.047740 | 0.179240 |
| 图像干净、音频残缺 | v11d | 0.047740 | 0.179240 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.055080 | 0.169240 |
| 双模态残缺 | v11d | 0.055080 | 0.169240 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.048100 | 0.140720 |
| 仅图像残缺 | v11d | 0.048100 | 0.140720 |

**音频 Cross-Detail → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.016000 | 0.000000 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.000120 | 0.000000 |
| 双模态残缺 | v11d | 0.000160 | 0.000020 |
| 仅音频残缺 | v11d | 0.015920 | -0.000180 |
| 仅图像残缺 | v11d | N/A | N/A |

**音频 Cross-Detail → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.212200 | 0.930400 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.165440 | 0.710020 |
| 双模态残缺 | v11d | 0.154720 | 0.740500 |
| 仅音频残缺 | v11d | 0.198520 | 1.000300 |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Cross-Detail → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | -0.000200 | 0.000000 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | -0.000020 | 0.000000 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.000160 | 0.000000 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | -0.000200 | 0.000000 |

**图像 Cross-Detail → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | 0.164900 | 0.569800 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | 0.173020 | 0.604920 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.211220 | 0.589280 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | 0.198600 | 0.556800 |

</details>


<details>
<summary>fixed 五类等权宏平均：干预各字段有效 n</summary>

| 实验 | 指标字段（同一计数口径） | 每 family 的 n |
| --- | --- | ---: |
| control | `aud2img_correct_gain`、`aud2img_gate`、`aud2img_ratio` | 未逐指标保存 |
| control | `aud2img_same_damage`、`aud2img_wrong_damage`、`img2aud_correct_gain` | 未逐指标保存 |
| control | `img2aud_gate`、`img2aud_ratio`、`img2aud_same_damage` | 未逐指标保存 |
| control | `img2aud_wrong_damage` | 未逐指标保存 |
| v11d | `aud2img_correct_gain`、`aud2img_gate`、`aud2img_ratio` | 未逐指标保存 |
| v11d | `aud2img_same_damage`、`aud2img_wrong_damage`、`detail_aud2img_correct_gain` | 未逐指标保存 |
| v11d | `detail_aud2img_gate`、`detail_aud2img_ratio`、`detail_aud2img_same_damage` | 未逐指标保存 |
| v11d | `detail_img2aud_correct_gain`、`detail_img2aud_gate`、`detail_img2aud_ratio` | 未逐指标保存 |
| v11d | `detail_img2aud_same_damage`、`img2aud_correct_gain`、`img2aud_gate` | 未逐指标保存 |
| v11d | `img2aud_ratio`、`img2aud_same_damage`、`img2aud_wrong_damage` | 未逐指标保存 |

</details>

- random：未找到干预结果。

Cross-Detail 只有主实验的 fixed 扫描；control 关闭该通路，没有对应扫描，不能将其缺项填成 0。旧日志未保存干预绝对误差和 win 比例，不从舍入 gain 反推出假精度的绝对值。

**逐 family 完整分项**

以下是与主表同源的五组 image/audio family pair，每组内仍按输入模式、实验排列。每组约 10000 个 MNIST 测试条目；音频会复用，旧版有效 n 未逐指标保存。


<details>
<summary>family 1：occlusion/time_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 95.700% | 0.009100 | 0.947000 | 0.005700 | 0.837000 |
| 干净双模态 | v11d | 95.700% | 0.009300 | 0.945000 | 0.005700 | 0.838000 |
| 仅干净图像 | control | 98.600% | 0.009100 | 0.947000 | 0.018600 | 0.261000 |
| 仅干净图像 | v11d | 98.600% | 0.009300 | 0.945000 | 0.019400 | 0.255000 |
| 仅干净音频 | control | 64.700% | 0.058900 | 0.535000 | 0.006000 | 0.828000 |
| 仅干净音频 | v11d | 64.700% | 0.056700 | 0.562000 | 0.006100 | 0.826000 |
| 图像干净、音频残缺 | control | 95.600% | 0.009100 | 0.947000 | 0.008800 | 0.722000 |
| 图像干净、音频残缺 | v11d | 95.600% | 0.009300 | 0.945000 | 0.008900 | 0.720000 |
| 音频干净、图像残缺 | control | 90.600% | 0.006000 | 0.965000 | 0.005700 | 0.836000 |
| 音频干净、图像残缺 | v11d | 90.600% | 0.006000 | 0.965000 | 0.005800 | 0.836000 |
| 双模态残缺 | control | 90.400% | 0.006000 | 0.965000 | 0.008900 | 0.717000 |
| 双模态残缺 | v11d | 90.400% | 0.006000 | 0.965000 | 0.009000 | 0.714000 |
| 仅音频残缺 | control | 57.700% | 0.061800 | 0.509000 | 0.009300 | 0.711000 |
| 仅音频残缺 | v11d | 57.700% | 0.059900 | 0.529000 | 0.009300 | 0.709000 |
| 仅图像残缺 | control | 93.500% | 0.006100 | 0.964000 | 0.018700 | 0.260000 |
| 仅图像残缺 | v11d | 93.500% | 0.006200 | 0.964000 | 0.019400 | 0.254000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.200000 | N/A | N/A |
| 干净双模态 | v11d | 21.090000 | N/A | N/A |
| 仅干净图像 | control | 21.220000 | N/A | N/A |
| 仅干净图像 | v11d | 21.110000 | N/A | N/A |
| 仅干净音频 | control | 12.580000 | 0.058900 | 0.058900 |
| 仅干净音频 | v11d | 12.770000 | 0.056700 | 0.056700 |
| 图像干净、音频残缺 | control | 21.200000 | N/A | N/A |
| 图像干净、音频残缺 | v11d | 21.080000 | N/A | N/A |
| 音频干净、图像残缺 | control | 25.410000 | 0.098500 | 0.038800 |
| 音频干净、图像残缺 | v11d | 25.430000 | 0.099300 | 0.038800 |
| 双模态残缺 | control | 25.390000 | 0.097900 | 0.038900 |
| 双模态残缺 | v11d | 25.390000 | 0.099100 | 0.038800 |
| 仅音频残缺 | control | 12.390000 | 0.061800 | 0.061800 |
| 仅音频残缺 | v11d | 12.530000 | 0.059900 | 0.059900 |
| 仅图像残缺 | control | 25.350000 | 0.099300 | 0.039700 |
| 仅图像残缺 | v11d | 25.330000 | 0.099900 | 0.040000 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.016000 | 0.000000 |
| 音频干净、图像残缺 | v11d | 0.016100 | 0.000000 |
| 双模态残缺 | control | 0.015900 | 0.000000 |
| 双模态残缺 | v11d | 0.016200 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.015900 | 0.000000 |
| 仅图像残缺 | v11d | 0.016200 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.018600 | N/A |
| 仅干净图像 | v11d | 0.019400 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.012000 | 0.006700 |
| 图像干净、音频残缺 | v11d | 0.012200 | 0.006700 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.012300 | 0.006600 |
| 双模态残缺 | v11d | 0.012400 | 0.006700 |
| 仅音频残缺 | control | 0.012700 | 0.007000 |
| 仅音频残缺 | v11d | 0.012700 | 0.007000 |
| 仅图像残缺 | control | 0.018700 | N/A |
| 仅图像残缺 | v11d | 0.019400 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.059200 | 0.340500 |
| 干净双模态 | v11d | 0.058700 | 0.339200 |
| 仅干净图像 | control | 0.059300 | 0.338600 |
| 仅干净图像 | v11d | 0.058300 | 0.335600 |
| 仅干净音频 | control | 0.007900 | 0.114900 |
| 仅干净音频 | v11d | 0.011300 | 0.139900 |
| 图像干净、音频残缺 | control | 0.059100 | 0.339800 |
| 图像干净、音频残缺 | v11d | 0.058400 | 0.337800 |
| 音频干净、图像残缺 | control | 0.063000 | 0.351200 |
| 音频干净、图像残缺 | v11d | 0.063100 | 0.351400 |
| 双模态残缺 | control | 0.063000 | 0.350000 |
| 双模态残缺 | v11d | 0.063000 | 0.350000 |
| 仅音频残缺 | control | 0.007300 | 0.112600 |
| 仅音频残缺 | v11d | 0.008900 | 0.125200 |
| 仅图像残缺 | control | 0.063000 | 0.351100 |
| 仅图像残缺 | v11d | 0.063000 | 0.351100 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.047200 | 0.146200 | 1.000000 |
| 干净双模态 | v11d | 0.047200 | 0.145900 | 1.000000 |
| 仅干净图像 | control | 0.021300 | 0.041000 | 0.358400 |
| 仅干净图像 | v11d | 0.017900 | 0.035000 | 0.342300 |
| 仅干净音频 | control | 0.048300 | 0.147600 | 1.000000 |
| 仅干净音频 | v11d | 0.048500 | 0.147800 | 1.000000 |
| 图像干净、音频残缺 | control | 0.042000 | 0.132500 | 1.000000 |
| 图像干净、音频残缺 | v11d | 0.041300 | 0.131400 | 1.000000 |
| 音频干净、图像残缺 | control | 0.047200 | 0.146200 | 1.000000 |
| 音频干净、图像残缺 | v11d | 0.047600 | 0.146400 | 1.000000 |
| 双模态残缺 | control | 0.041800 | 0.132000 | 1.000000 |
| 双模态残缺 | v11d | 0.041600 | 0.131600 | 1.000000 |
| 仅音频残缺 | control | 0.043100 | 0.133800 | 1.000000 |
| 仅音频残缺 | v11d | 0.043200 | 0.133700 | 1.000000 |
| 仅图像残缺 | control | 0.021400 | 0.041000 | 0.363400 |
| 仅图像残缺 | v11d | 0.018000 | 0.034500 | 0.346400 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 干净双模态 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 仅干净图像 | control | 0.042600 | 0.148600 | 1.000000 | 48.400% |
| 仅干净图像 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.300% |
| 仅干净音频 | control | 0.042600 | 0.148600 | 1.000000 | 62.900% |
| 仅干净音频 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.800% |
| 图像干净、音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 59.000% |
| 图像干净、音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 58.900% |
| 音频干净、图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 63.200% |
| 音频干净、图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.200% |
| 双模态残缺 | control | 0.042600 | 0.148600 | 1.000000 | 58.700% |
| 双模态残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 58.600% |
| 仅音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 58.300% |
| 仅音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 58.300% |
| 仅图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 48.100% |
| 仅图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.000% |

**本 family 干预**

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A | N/A |
| 仅干净音频 | control | 0.000100 | 0.000100 | -0.000300 |
| 仅干净音频 | v11d | 0.000000 | 0.000000 | -0.000200 |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.000100 | 0.000100 | -0.000100 |
| 音频干净、图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | control | 0.000000 | 0.000000 | -0.000100 |
| 双模态残缺 | v11d | 0.000000 | 0.000000 | -0.000100 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | -0.000200 |
| 仅音频残缺 | v11d | 0.000000 | 0.000000 | -0.000200 |
| 仅图像残缺 | control | N/A | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | 0.011000 | 0.028400 |
| 仅干净音频 | v11d | 0.011000 | 0.028400 |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.013900 | 0.034000 |
| 音频干净、图像残缺 | v11d | 0.013900 | 0.034000 |
| 双模态残缺 | control | 0.014300 | 0.033500 |
| 双模态残缺 | v11d | 0.014300 | 0.033500 |
| 仅音频残缺 | control | 0.010900 | 0.027800 |
| 仅音频残缺 | v11d | 0.010900 | 0.027800 |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.000100 | 0.000200 | 0.000000 |
| 图像干净、音频残缺 | v11d | 0.000100 | 0.000100 | 0.000000 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A | N/A |
| 双模态残缺 | control | 0.000100 | 0.000200 | 0.000000 |
| 双模态残缺 | v11d | 0.000100 | 0.000100 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.042900 | 0.149700 |
| 仅干净图像 | v11d | 0.042900 | 0.149700 |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.041100 | 0.151600 |
| 图像干净、音频残缺 | v11d | 0.041100 | 0.151600 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.045500 | 0.144000 |
| 双模态残缺 | v11d | 0.045500 | 0.144000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.046700 | 0.143400 |
| 仅图像残缺 | v11d | 0.046700 | 0.143400 |

**音频 Cross-Detail → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.016000 | 0.000000 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.000400 | 0.000000 |
| 双模态残缺 | v11d | 0.000500 | 0.000100 |
| 仅音频残缺 | v11d | 0.016200 | 0.000000 |
| 仅图像残缺 | v11d | N/A | N/A |

**音频 Cross-Detail → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.212200 | 0.930400 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.159800 | 0.675800 |
| 双模态残缺 | v11d | 0.137400 | 0.714100 |
| 仅音频残缺 | v11d | 0.182900 | 1.010400 |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Cross-Detail → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | -0.000200 | 0.000000 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | -0.000100 | 0.000000 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.000000 | 0.000000 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | -0.000200 | 0.000000 |

**图像 Cross-Detail → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | 0.164900 | 0.569800 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | 0.167900 | 0.585100 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.180600 | 0.562700 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | 0.176200 | 0.554300 |

</details>


<details>
<summary>family 2：pixel_delete/freq_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 95.700% | 0.009100 | 0.947000 | 0.005700 | 0.837000 |
| 干净双模态 | v11d | 95.700% | 0.009300 | 0.945000 | 0.005700 | 0.838000 |
| 仅干净图像 | control | 98.600% | 0.009100 | 0.947000 | 0.018600 | 0.261000 |
| 仅干净图像 | v11d | 98.600% | 0.009300 | 0.945000 | 0.019400 | 0.255000 |
| 仅干净音频 | control | 64.700% | 0.058900 | 0.535000 | 0.006000 | 0.828000 |
| 仅干净音频 | v11d | 64.700% | 0.056700 | 0.562000 | 0.006100 | 0.826000 |
| 图像干净、音频残缺 | control | 92.100% | 0.009200 | 0.947000 | 0.007500 | 0.761000 |
| 图像干净、音频残缺 | v11d | 92.100% | 0.009400 | 0.945000 | 0.007500 | 0.764000 |
| 音频干净、图像残缺 | control | 91.600% | 0.001700 | 0.990000 | 0.005700 | 0.836000 |
| 音频干净、图像残缺 | v11d | 91.600% | 0.001700 | 0.990000 | 0.005700 | 0.837000 |
| 双模态残缺 | control | 85.700% | 0.001700 | 0.990000 | 0.007600 | 0.761000 |
| 双模态残缺 | v11d | 85.700% | 0.001700 | 0.990000 | 0.007500 | 0.763000 |
| 仅音频残缺 | control | 50.200% | 0.062600 | 0.502000 | 0.008200 | 0.744000 |
| 仅音频残缺 | v11d | 50.200% | 0.061500 | 0.515000 | 0.008300 | 0.741000 |
| 仅图像残缺 | control | 97.200% | 0.001700 | 0.990000 | 0.018600 | 0.261000 |
| 仅图像残缺 | v11d | 97.200% | 0.001700 | 0.990000 | 0.019400 | 0.255000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.200000 | N/A | N/A |
| 干净双模态 | v11d | 21.090000 | N/A | N/A |
| 仅干净图像 | control | 21.220000 | N/A | N/A |
| 仅干净图像 | v11d | 21.110000 | N/A | N/A |
| 仅干净音频 | control | 12.580000 | 0.058900 | 0.058900 |
| 仅干净音频 | v11d | 12.770000 | 0.056700 | 0.056700 |
| 图像干净、音频残缺 | control | 21.180000 | N/A | N/A |
| 图像干净、音频残缺 | v11d | 21.080000 | N/A | N/A |
| 音频干净、图像残缺 | control | 28.350000 | 0.033600 | 0.004200 |
| 音频干净、图像残缺 | v11d | 28.370000 | 0.033700 | 0.004200 |
| 双模态残缺 | control | 28.350000 | 0.033600 | 0.004200 |
| 双模态残缺 | v11d | 28.370000 | 0.034100 | 0.004200 |
| 仅音频残缺 | control | 12.300000 | 0.062600 | 0.062600 |
| 仅音频残缺 | v11d | 12.390000 | 0.061500 | 0.061500 |
| 仅图像残缺 | control | 28.350000 | 0.033600 | 0.004200 |
| 仅图像残缺 | v11d | 28.360000 | 0.033900 | 0.004200 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.026200 | 0.000000 |
| 音频干净、图像残缺 | v11d | 0.026400 | 0.000000 |
| 双模态残缺 | control | 0.026200 | 0.000000 |
| 双模态残缺 | v11d | 0.026700 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.026200 | 0.000000 |
| 仅图像残缺 | v11d | 0.026600 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.018600 | N/A |
| 仅干净图像 | v11d | 0.019400 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.009100 | 0.006500 |
| 图像干净、音频残缺 | v11d | 0.009000 | 0.006500 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.009200 | 0.006400 |
| 双模态残缺 | v11d | 0.009200 | 0.006400 |
| 仅音频残缺 | control | 0.010100 | 0.006900 |
| 仅音频残缺 | v11d | 0.010200 | 0.007000 |
| 仅图像残缺 | control | 0.018600 | N/A |
| 仅图像残缺 | v11d | 0.019400 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.059200 | 0.340200 |
| 干净双模态 | v11d | 0.058700 | 0.338900 |
| 仅干净图像 | control | 0.059300 | 0.339700 |
| 仅干净图像 | v11d | 0.058300 | 0.336700 |
| 仅干净音频 | control | 0.007900 | 0.115500 |
| 仅干净音频 | v11d | 0.011300 | 0.140200 |
| 图像干净、音频残缺 | control | 0.059100 | 0.340200 |
| 图像干净、音频残缺 | v11d | 0.058300 | 0.337900 |
| 音频干净、图像残缺 | control | 0.066400 | 0.359700 |
| 音频干净、图像残缺 | v11d | 0.066500 | 0.360000 |
| 双模态残缺 | control | 0.066400 | 0.360300 |
| 双模态残缺 | v11d | 0.066500 | 0.360700 |
| 仅音频残缺 | control | 0.006700 | 0.108300 |
| 仅音频残缺 | v11d | 0.008200 | 0.121800 |
| 仅图像残缺 | control | 0.066400 | 0.359700 |
| 仅图像残缺 | v11d | 0.066500 | 0.360000 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.047200 | 0.146200 | 1.000000 |
| 干净双模态 | v11d | 0.047200 | 0.145900 | 1.000000 |
| 仅干净图像 | control | 0.021300 | 0.041000 | 0.358400 |
| 仅干净图像 | v11d | 0.017900 | 0.035000 | 0.342300 |
| 仅干净音频 | control | 0.048300 | 0.147600 | 1.000000 |
| 仅干净音频 | v11d | 0.048500 | 0.147800 | 1.000000 |
| 图像干净、音频残缺 | control | 0.043900 | 0.137900 | 1.000000 |
| 图像干净、音频残缺 | v11d | 0.044000 | 0.137800 | 1.000000 |
| 音频干净、图像残缺 | control | 0.047200 | 0.146300 | 1.000000 |
| 音频干净、图像残缺 | v11d | 0.047400 | 0.146300 | 1.000000 |
| 双模态残缺 | control | 0.044000 | 0.137900 | 1.000000 |
| 双模态残缺 | v11d | 0.044300 | 0.138200 | 1.000000 |
| 仅音频残缺 | control | 0.045100 | 0.139300 | 1.000000 |
| 仅音频残缺 | v11d | 0.045300 | 0.139500 | 1.000000 |
| 仅图像残缺 | control | 0.021300 | 0.041000 | 0.361800 |
| 仅图像残缺 | v11d | 0.018000 | 0.035000 | 0.348400 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 干净双模态 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 仅干净图像 | control | 0.042600 | 0.148600 | 1.000000 | 48.400% |
| 仅干净图像 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.300% |
| 仅干净音频 | control | 0.042600 | 0.148600 | 1.000000 | 62.900% |
| 仅干净音频 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.800% |
| 图像干净、音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 61.000% |
| 图像干净、音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 61.100% |
| 音频干净、图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 63.200% |
| 音频干净、图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 双模态残缺 | control | 0.042600 | 0.148600 | 1.000000 | 60.900% |
| 双模态残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 61.000% |
| 仅音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 60.100% |
| 仅音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 60.000% |
| 仅图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 48.300% |
| 仅图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.200% |

**本 family 干预**

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A | N/A |
| 仅干净音频 | control | 0.000100 | 0.000100 | -0.000300 |
| 仅干净音频 | v11d | 0.000000 | 0.000000 | -0.000200 |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | control | -0.000100 | 0.000000 | -0.000300 |
| 仅音频残缺 | v11d | 0.000000 | 0.000000 | -0.000200 |
| 仅图像残缺 | control | N/A | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | 0.011000 | 0.028400 |
| 仅干净音频 | v11d | 0.011000 | 0.028400 |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.013500 | 0.033600 |
| 音频干净、图像残缺 | v11d | 0.013500 | 0.033600 |
| 双模态残缺 | control | 0.015200 | 0.036700 |
| 双模态残缺 | v11d | 0.015200 | 0.036700 |
| 仅音频残缺 | control | 0.012400 | 0.031600 |
| 仅音频残缺 | v11d | 0.012400 | 0.031600 |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.000200 | 0.000300 | 0.000000 |
| 图像干净、音频残缺 | v11d | 0.000200 | 0.000300 | 0.000000 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A | N/A |
| 双模态残缺 | control | 0.000200 | 0.000300 | 0.000000 |
| 双模态残缺 | v11d | 0.000200 | 0.000300 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.042900 | 0.149700 |
| 仅干净图像 | v11d | 0.042900 | 0.149700 |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.050700 | 0.189900 |
| 图像干净、音频残缺 | v11d | 0.050700 | 0.189900 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.057000 | 0.182700 |
| 双模态残缺 | v11d | 0.057000 | 0.182700 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.047700 | 0.144200 |
| 仅图像残缺 | v11d | 0.047700 | 0.144200 |

**音频 Cross-Detail → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.016000 | 0.000000 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.000000 | 0.000000 |
| 双模态残缺 | v11d | 0.000000 | 0.000000 |
| 仅音频残缺 | v11d | 0.015500 | -0.000500 |
| 仅图像残缺 | v11d | N/A | N/A |

**音频 Cross-Detail → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.212200 | 0.930400 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.167200 | 0.715000 |
| 双模态残缺 | v11d | 0.159700 | 0.712900 |
| 仅音频残缺 | v11d | 0.202700 | 0.934200 |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Cross-Detail → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | -0.000200 | 0.000000 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | 0.000200 | 0.000000 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.000200 | 0.000000 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | -0.000200 | 0.000000 |

**图像 Cross-Detail → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | 0.164900 | 0.569800 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | 0.174800 | 0.608000 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.228500 | 0.558400 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | 0.214000 | 0.524000 |

</details>


<details>
<summary>family 3：mask_vertical/feature_dropout</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 95.700% | 0.009100 | 0.947000 | 0.005700 | 0.837000 |
| 干净双模态 | v11d | 95.700% | 0.009300 | 0.945000 | 0.005700 | 0.838000 |
| 仅干净图像 | control | 98.600% | 0.009100 | 0.947000 | 0.018600 | 0.261000 |
| 仅干净图像 | v11d | 98.600% | 0.009300 | 0.945000 | 0.019400 | 0.255000 |
| 仅干净音频 | control | 64.700% | 0.058900 | 0.535000 | 0.006000 | 0.828000 |
| 仅干净音频 | v11d | 64.700% | 0.056700 | 0.562000 | 0.006100 | 0.826000 |
| 图像干净、音频残缺 | control | 96.000% | 0.009100 | 0.947000 | 0.005800 | 0.827000 |
| 图像干净、音频残缺 | v11d | 96.000% | 0.009300 | 0.945000 | 0.005800 | 0.829000 |
| 音频干净、图像残缺 | control | 92.600% | 0.005700 | 0.969000 | 0.005700 | 0.837000 |
| 音频干净、图像残缺 | v11d | 92.600% | 0.005700 | 0.969000 | 0.005700 | 0.838000 |
| 双模态残缺 | control | 93.600% | 0.005700 | 0.970000 | 0.005800 | 0.826000 |
| 双模态残缺 | v11d | 93.600% | 0.005700 | 0.970000 | 0.005800 | 0.828000 |
| 仅音频残缺 | control | 64.300% | 0.059400 | 0.530000 | 0.006100 | 0.818000 |
| 仅音频残缺 | v11d | 64.300% | 0.057500 | 0.555000 | 0.006100 | 0.817000 |
| 仅图像残缺 | control | 96.500% | 0.005800 | 0.969000 | 0.018600 | 0.261000 |
| 仅图像残缺 | v11d | 96.500% | 0.005800 | 0.969000 | 0.019400 | 0.255000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.200000 | N/A | N/A |
| 干净双模态 | v11d | 21.090000 | N/A | N/A |
| 仅干净图像 | control | 21.220000 | N/A | N/A |
| 仅干净图像 | v11d | 21.110000 | N/A | N/A |
| 仅干净音频 | control | 12.580000 | 0.058900 | 0.058900 |
| 仅干净音频 | v11d | 12.770000 | 0.056700 | 0.056700 |
| 图像干净、音频残缺 | control | 21.200000 | N/A | N/A |
| 图像干净、音频残缺 | v11d | 21.090000 | N/A | N/A |
| 音频干净、图像残缺 | control | 26.230000 | 0.031100 | 0.014600 |
| 音频干净、图像残缺 | v11d | 26.240000 | 0.031100 | 0.014600 |
| 双模态残缺 | control | 26.340000 | 0.031100 | 0.014400 |
| 双模态残缺 | v11d | 26.350000 | 0.031200 | 0.014400 |
| 仅音频残缺 | control | 12.550000 | 0.059400 | 0.059400 |
| 仅音频残缺 | v11d | 12.720000 | 0.057500 | 0.057500 |
| 仅图像残缺 | control | 26.210000 | 0.031100 | 0.014800 |
| 仅图像残缺 | v11d | 26.200000 | 0.031700 | 0.014800 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.018100 | 0.000000 |
| 音频干净、图像残缺 | v11d | 0.018300 | 0.000000 |
| 双模态残缺 | control | 0.018100 | 0.000000 |
| 双模态残缺 | v11d | 0.018400 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.018100 | 0.000000 |
| 仅图像残缺 | v11d | 0.018600 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.018600 | N/A |
| 仅干净图像 | v11d | 0.019400 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.005900 | 0.005800 |
| 图像干净、音频残缺 | v11d | 0.005900 | 0.005700 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.005900 | 0.005800 |
| 双模态残缺 | v11d | 0.005900 | 0.005800 |
| 仅音频残缺 | control | 0.006200 | 0.006000 |
| 仅音频残缺 | v11d | 0.006200 | 0.006100 |
| 仅图像残缺 | control | 0.018600 | N/A |
| 仅图像残缺 | v11d | 0.019400 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.059200 | 0.338100 |
| 干净双模态 | v11d | 0.058700 | 0.336900 |
| 仅干净图像 | control | 0.059300 | 0.341100 |
| 仅干净图像 | v11d | 0.058300 | 0.338200 |
| 仅干净音频 | control | 0.007900 | 0.114500 |
| 仅干净音频 | v11d | 0.011300 | 0.140000 |
| 图像干净、音频残缺 | control | 0.059200 | 0.337700 |
| 图像干净、音频残缺 | v11d | 0.058700 | 0.336100 |
| 音频干净、图像残缺 | control | 0.063200 | 0.350700 |
| 音频干净、图像残缺 | v11d | 0.063300 | 0.350900 |
| 双模态残缺 | control | 0.063200 | 0.351100 |
| 双模态残缺 | v11d | 0.063200 | 0.351300 |
| 仅音频残缺 | control | 0.007900 | 0.113300 |
| 仅音频残缺 | v11d | 0.011100 | 0.138400 |
| 仅图像残缺 | control | 0.063200 | 0.350700 |
| 仅图像残缺 | v11d | 0.063000 | 0.350300 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.047200 | 0.146200 | 1.000000 |
| 干净双模态 | v11d | 0.047200 | 0.145900 | 1.000000 |
| 仅干净图像 | control | 0.021300 | 0.041000 | 0.358400 |
| 仅干净图像 | v11d | 0.017900 | 0.035000 | 0.342300 |
| 仅干净音频 | control | 0.048300 | 0.147600 | 1.000000 |
| 仅干净音频 | v11d | 0.048500 | 0.147800 | 1.000000 |
| 图像干净、音频残缺 | control | 0.044100 | 0.140100 | 1.000000 |
| 图像干净、音频残缺 | v11d | 0.044200 | 0.140100 | 1.000000 |
| 音频干净、图像残缺 | control | 0.047200 | 0.146200 | 1.000000 |
| 音频干净、图像残缺 | v11d | 0.047300 | 0.146000 | 1.000000 |
| 双模态残缺 | control | 0.044200 | 0.140100 | 1.000000 |
| 双模态残缺 | v11d | 0.044400 | 0.140400 | 1.000000 |
| 仅音频残缺 | control | 0.045100 | 0.141400 | 1.000000 |
| 仅音频残缺 | v11d | 0.045300 | 0.141600 | 1.000000 |
| 仅图像残缺 | control | 0.021300 | 0.040900 | 0.361100 |
| 仅图像残缺 | v11d | 0.017900 | 0.034700 | 0.341800 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 干净双模态 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 仅干净图像 | control | 0.042600 | 0.148600 | 1.000000 | 48.400% |
| 仅干净图像 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.300% |
| 仅干净音频 | control | 0.042600 | 0.148600 | 1.000000 | 62.900% |
| 仅干净音频 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.800% |
| 图像干净、音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 62.800% |
| 图像干净、音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.900% |
| 音频干净、图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 63.200% |
| 音频干净、图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 双模态残缺 | control | 0.042600 | 0.148600 | 1.000000 | 62.800% |
| 双模态残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.800% |
| 仅音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 62.400% |
| 仅音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.300% |
| 仅图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 48.300% |
| 仅图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.200% |

**本 family 干预**

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A | N/A |
| 仅干净音频 | control | 0.000100 | 0.000100 | -0.000300 |
| 仅干净音频 | v11d | 0.000000 | 0.000000 | -0.000200 |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | control | 0.000000 | 0.000100 | -0.000300 |
| 仅音频残缺 | v11d | 0.000000 | 0.000000 | -0.000200 |
| 仅图像残缺 | control | N/A | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | 0.011000 | 0.028400 |
| 仅干净音频 | v11d | 0.011000 | 0.028400 |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.013400 | 0.033000 |
| 音频干净、图像残缺 | v11d | 0.013400 | 0.033000 |
| 双模态残缺 | control | 0.014300 | 0.034100 |
| 双模态残缺 | v11d | 0.014300 | 0.034100 |
| 仅音频残缺 | control | 0.011700 | 0.029700 |
| 仅音频残缺 | v11d | 0.011700 | 0.029700 |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.000100 | 0.000100 | 0.000000 |
| 图像干净、音频残缺 | v11d | 0.000100 | 0.000100 | 0.000000 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A | N/A |
| 双模态残缺 | control | 0.000100 | 0.000100 | 0.000000 |
| 双模态残缺 | v11d | 0.000100 | 0.000100 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.042900 | 0.149700 |
| 仅干净图像 | v11d | 0.042900 | 0.149700 |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.049500 | 0.184800 |
| 图像干净、音频残缺 | v11d | 0.049500 | 0.184800 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.054000 | 0.179300 |
| 双模态残缺 | v11d | 0.054000 | 0.179300 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.046500 | 0.145700 |
| 仅图像残缺 | v11d | 0.046500 | 0.145700 |

**音频 Cross-Detail → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.016000 | 0.000000 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.000100 | 0.000000 |
| 双模态残缺 | v11d | 0.000100 | 0.000000 |
| 仅音频残缺 | v11d | 0.015400 | -0.000300 |
| 仅图像残缺 | v11d | N/A | N/A |

**音频 Cross-Detail → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.212200 | 0.930400 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.165100 | 0.696500 |
| 双模态残缺 | v11d | 0.169600 | 0.702900 |
| 仅音频残缺 | v11d | 0.216800 | 0.940200 |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Cross-Detail → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | -0.000200 | 0.000000 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | 0.000100 | 0.000000 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.000100 | 0.000000 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | -0.000200 | 0.000000 |

**图像 Cross-Detail → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | 0.164900 | 0.569800 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | 0.172000 | 0.599300 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.190100 | 0.573300 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | 0.181000 | 0.548500 |

</details>


<details>
<summary>family 4：mask_horizontal/partial_temporal</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 95.700% | 0.009100 | 0.947000 | 0.005700 | 0.837000 |
| 干净双模态 | v11d | 95.700% | 0.009300 | 0.945000 | 0.005700 | 0.838000 |
| 仅干净图像 | control | 98.600% | 0.009100 | 0.947000 | 0.018600 | 0.261000 |
| 仅干净图像 | v11d | 98.600% | 0.009300 | 0.945000 | 0.019400 | 0.255000 |
| 仅干净音频 | control | 64.700% | 0.058900 | 0.535000 | 0.006000 | 0.828000 |
| 仅干净音频 | v11d | 64.700% | 0.056700 | 0.562000 | 0.006100 | 0.826000 |
| 图像干净、音频残缺 | control | 93.300% | 0.009200 | 0.946000 | 0.013200 | 0.581000 |
| 图像干净、音频残缺 | v11d | 93.300% | 0.009300 | 0.945000 | 0.013600 | 0.555000 |
| 音频干净、图像残缺 | control | 90.900% | 0.008900 | 0.949000 | 0.005700 | 0.836000 |
| 音频干净、图像残缺 | v11d | 90.900% | 0.008900 | 0.949000 | 0.005900 | 0.830000 |
| 双模态残缺 | control | 85.000% | 0.009000 | 0.949000 | 0.013400 | 0.578000 |
| 双模态残缺 | v11d | 85.000% | 0.009000 | 0.949000 | 0.013400 | 0.576000 |
| 仅音频残缺 | control | 26.600% | 0.066200 | 0.459000 | 0.014600 | 0.556000 |
| 仅音频残缺 | v11d | 26.600% | 0.064200 | 0.477000 | 0.014600 | 0.553000 |
| 仅图像残缺 | control | 95.700% | 0.009000 | 0.948000 | 0.018700 | 0.260000 |
| 仅图像残缺 | v11d | 95.700% | 0.009000 | 0.948000 | 0.019300 | 0.255000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.200000 | N/A | N/A |
| 干净双模态 | v11d | 21.090000 | N/A | N/A |
| 仅干净图像 | control | 21.220000 | N/A | N/A |
| 仅干净图像 | v11d | 21.110000 | N/A | N/A |
| 仅干净音频 | control | 12.580000 | 0.058900 | 0.058900 |
| 仅干净音频 | v11d | 12.770000 | 0.056700 | 0.056700 |
| 图像干净、音频残缺 | control | 21.180000 | N/A | N/A |
| 图像干净、音频残缺 | v11d | 21.070000 | N/A | N/A |
| 音频干净、图像残缺 | control | 21.640000 | 0.042600 | 0.022700 |
| 音频干净、图像残缺 | v11d | 21.640000 | 0.042900 | 0.022700 |
| 双模态残缺 | control | 21.620000 | 0.043100 | 0.022900 |
| 双模态残缺 | v11d | 21.610000 | 0.043700 | 0.022900 |
| 仅音频残缺 | control | 12.030000 | 0.066200 | 0.066200 |
| 仅音频残缺 | v11d | 12.160000 | 0.064200 | 0.064200 |
| 仅图像残缺 | control | 21.610000 | 0.042600 | 0.023000 |
| 仅图像残缺 | v11d | 21.590000 | 0.043100 | 0.023000 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.023900 | 0.000000 |
| 音频干净、图像残缺 | v11d | 0.024200 | 0.000000 |
| 双模态残缺 | control | 0.024000 | 0.000000 |
| 双模态残缺 | v11d | 0.024500 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.023900 | 0.000000 |
| 仅图像残缺 | v11d | 0.024300 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.018600 | N/A |
| 仅干净图像 | v11d | 0.019400 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.031700 | 0.000600 |
| 图像干净、音频残缺 | v11d | 0.032500 | 0.000600 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.032000 | 0.000600 |
| 双模态残缺 | v11d | 0.032000 | 0.000700 |
| 仅音频残缺 | control | 0.035000 | 0.000600 |
| 仅音频残缺 | v11d | 0.034900 | 0.000700 |
| 仅图像残缺 | control | 0.018700 | N/A |
| 仅图像残缺 | v11d | 0.019300 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.059200 | 0.339700 |
| 干净双模态 | v11d | 0.058700 | 0.338400 |
| 仅干净图像 | control | 0.059300 | 0.339500 |
| 仅干净图像 | v11d | 0.058300 | 0.336600 |
| 仅干净音频 | control | 0.007900 | 0.116300 |
| 仅干净音频 | v11d | 0.011300 | 0.141200 |
| 图像干净、音频残缺 | control | 0.059100 | 0.338900 |
| 图像干净、音频残缺 | v11d | 0.058000 | 0.335800 |
| 音频干净、图像残缺 | control | 0.060800 | 0.345700 |
| 音频干净、图像残缺 | v11d | 0.060800 | 0.345600 |
| 双模态残缺 | control | 0.060700 | 0.345100 |
| 双模态残缺 | v11d | 0.060600 | 0.344800 |
| 仅音频残缺 | control | 0.003300 | 0.070300 |
| 仅音频残缺 | v11d | 0.003700 | 0.075600 |
| 仅图像残缺 | control | 0.060800 | 0.345800 |
| 仅图像残缺 | v11d | 0.060600 | 0.345100 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.047200 | 0.146200 | 1.000000 |
| 干净双模态 | v11d | 0.047200 | 0.145900 | 1.000000 |
| 仅干净图像 | control | 0.021300 | 0.041000 | 0.358400 |
| 仅干净图像 | v11d | 0.017900 | 0.035000 | 0.342300 |
| 仅干净音频 | control | 0.048300 | 0.147600 | 1.000000 |
| 仅干净音频 | v11d | 0.048500 | 0.147800 | 1.000000 |
| 图像干净、音频残缺 | control | 0.038200 | 0.107200 | 0.999700 |
| 图像干净、音频残缺 | v11d | 0.034200 | 0.099200 | 0.999700 |
| 音频干净、图像残缺 | control | 0.047200 | 0.146200 | 1.000000 |
| 音频干净、图像残缺 | v11d | 0.048700 | 0.148000 | 1.000000 |
| 双模态残缺 | control | 0.038100 | 0.106700 | 0.999500 |
| 双模态残缺 | v11d | 0.039600 | 0.108500 | 0.999700 |
| 仅音频残缺 | control | 0.040300 | 0.110300 | 0.999800 |
| 仅音频残缺 | v11d | 0.039800 | 0.109200 | 0.999800 |
| 仅图像残缺 | control | 0.021400 | 0.041000 | 0.362700 |
| 仅图像残缺 | v11d | 0.018900 | 0.035900 | 0.342200 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 干净双模态 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 仅干净图像 | control | 0.042600 | 0.148600 | 1.000000 | 48.400% |
| 仅干净图像 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.300% |
| 仅干净音频 | control | 0.042600 | 0.148600 | 1.000000 | 62.900% |
| 仅干净音频 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.800% |
| 图像干净、音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 53.700% |
| 图像干净、音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 53.500% |
| 音频干净、图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 63.200% |
| 音频干净、图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.100% |
| 双模态残缺 | control | 0.042600 | 0.148600 | 1.000000 | 53.500% |
| 双模态残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 53.500% |
| 仅音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 51.300% |
| 仅音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 51.300% |
| 仅图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 48.200% |
| 仅图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.100% |

**本 family 干预**

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A | N/A |
| 仅干净音频 | control | 0.000100 | 0.000100 | -0.000300 |
| 仅干净音频 | v11d | 0.000000 | 0.000000 | -0.000200 |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | control | -0.000100 | 0.000000 | -0.000200 |
| 仅音频残缺 | v11d | 0.000000 | 0.000000 | -0.000100 |
| 仅图像残缺 | control | N/A | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | 0.011000 | 0.028400 |
| 仅干净音频 | v11d | 0.011000 | 0.028400 |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.013900 | 0.034300 |
| 音频干净、图像残缺 | v11d | 0.013900 | 0.034300 |
| 双模态残缺 | control | 0.018100 | 0.037900 |
| 双模态残缺 | v11d | 0.018100 | 0.037900 |
| 仅音频残缺 | control | 0.014100 | 0.033600 |
| 仅音频残缺 | v11d | 0.014100 | 0.033600 |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.000400 | 0.000700 | 0.000000 |
| 图像干净、音频残缺 | v11d | 0.000200 | 0.000600 | 0.000000 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A | N/A |
| 双模态残缺 | control | 0.000400 | 0.000700 | 0.000000 |
| 双模态残缺 | v11d | 0.000300 | 0.000500 | -0.000100 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.042900 | 0.149700 |
| 仅干净图像 | v11d | 0.042900 | 0.149700 |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.049900 | 0.193500 |
| 图像干净、音频残缺 | v11d | 0.049900 | 0.193500 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.056900 | 0.191900 |
| 双模态残缺 | v11d | 0.056900 | 0.191900 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.047100 | 0.144300 |
| 仅图像残缺 | v11d | 0.047100 | 0.144300 |

**音频 Cross-Detail → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.016000 | 0.000000 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.000100 | 0.000000 |
| 双模态残缺 | v11d | 0.000200 | 0.000000 |
| 仅音频残缺 | v11d | 0.016900 | 0.000300 |
| 仅图像残缺 | v11d | N/A | N/A |

**音频 Cross-Detail → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.212200 | 0.930400 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.164600 | 0.701900 |
| 双模态残缺 | v11d | 0.141400 | 0.805500 |
| 仅音频残缺 | v11d | 0.184700 | 1.177000 |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Cross-Detail → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | -0.000200 | 0.000000 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | -0.000500 | 0.000000 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.000400 | 0.000000 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | 0.000000 | 0.000000 |

**图像 Cross-Detail → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | 0.164900 | 0.569800 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | 0.177700 | 0.636900 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.244600 | 0.666700 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | 0.224500 | 0.601300 |

</details>


<details>
<summary>family 5：salt_mask/time_freq_block</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 95.700% | 0.009100 | 0.947000 | 0.005700 | 0.837000 |
| 干净双模态 | v11d | 95.700% | 0.009300 | 0.945000 | 0.005700 | 0.838000 |
| 仅干净图像 | control | 98.600% | 0.009100 | 0.947000 | 0.018600 | 0.261000 |
| 仅干净图像 | v11d | 98.600% | 0.009300 | 0.945000 | 0.019400 | 0.255000 |
| 仅干净音频 | control | 64.700% | 0.058900 | 0.535000 | 0.006000 | 0.828000 |
| 仅干净音频 | v11d | 64.700% | 0.056700 | 0.562000 | 0.006100 | 0.826000 |
| 图像干净、音频残缺 | control | 95.300% | 0.009100 | 0.947000 | 0.006200 | 0.816000 |
| 图像干净、音频残缺 | v11d | 95.300% | 0.009300 | 0.945000 | 0.006200 | 0.818000 |
| 音频干净、图像残缺 | control | 84.000% | 0.001800 | 0.990000 | 0.005800 | 0.835000 |
| 音频干净、图像残缺 | v11d | 84.000% | 0.001800 | 0.990000 | 0.005700 | 0.839000 |
| 双模态残缺 | control | 83.300% | 0.001800 | 0.990000 | 0.006300 | 0.813000 |
| 双模态残缺 | v11d | 83.300% | 0.001800 | 0.990000 | 0.006200 | 0.816000 |
| 仅音频残缺 | control | 62.400% | 0.060000 | 0.526000 | 0.006600 | 0.804000 |
| 仅音频残缺 | v11d | 62.400% | 0.058200 | 0.548000 | 0.006700 | 0.802000 |
| 仅图像残缺 | control | 94.600% | 0.001800 | 0.990000 | 0.018600 | 0.259000 |
| 仅图像残缺 | v11d | 94.600% | 0.001800 | 0.990000 | 0.019700 | 0.249000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 21.200000 | N/A | N/A |
| 干净双模态 | v11d | 21.090000 | N/A | N/A |
| 仅干净图像 | control | 21.220000 | N/A | N/A |
| 仅干净图像 | v11d | 21.110000 | N/A | N/A |
| 仅干净音频 | control | 12.580000 | 0.058900 | 0.058900 |
| 仅干净音频 | v11d | 12.770000 | 0.056700 | 0.056700 |
| 图像干净、音频残缺 | control | 21.200000 | N/A | N/A |
| 图像干净、音频残缺 | v11d | 21.090000 | N/A | N/A |
| 音频干净、图像残缺 | control | 28.060000 | 0.074900 | 0.004500 |
| 音频干净、图像残缺 | v11d | 28.060000 | 0.075700 | 0.004500 |
| 双模态残缺 | control | 28.050000 | 0.075000 | 0.004500 |
| 双模态残缺 | v11d | 28.060000 | 0.075900 | 0.004500 |
| 仅音频残缺 | control | 12.510000 | 0.060000 | 0.060000 |
| 仅音频残缺 | v11d | 12.660000 | 0.058200 | 0.058200 |
| 仅图像残缺 | control | 28.050000 | 0.074900 | 0.004500 |
| 仅图像残缺 | v11d | 28.050000 | 0.075600 | 0.004500 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.074900 | 0.000000 |
| 音频干净、图像残缺 | v11d | 0.075700 | 0.000000 |
| 双模态残缺 | control | 0.074900 | 0.000000 |
| 双模态残缺 | v11d | 0.075800 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.074900 | 0.000000 |
| 仅图像残缺 | v11d | 0.075600 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.018600 | N/A |
| 仅干净图像 | v11d | 0.019400 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.007200 | 0.006100 |
| 图像干净、音频残缺 | v11d | 0.007100 | 0.006000 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.007200 | 0.006100 |
| 双模态残缺 | v11d | 0.007100 | 0.006100 |
| 仅音频残缺 | control | 0.007900 | 0.006400 |
| 仅音频残缺 | v11d | 0.007900 | 0.006400 |
| 仅图像残缺 | control | 0.018600 | N/A |
| 仅图像残缺 | v11d | 0.019700 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.059200 | 0.339100 |
| 干净双模态 | v11d | 0.058700 | 0.337800 |
| 仅干净图像 | control | 0.059300 | 0.340100 |
| 仅干净图像 | v11d | 0.058300 | 0.337100 |
| 仅干净音频 | control | 0.007900 | 0.114500 |
| 仅干净音频 | v11d | 0.011300 | 0.139400 |
| 图像干净、音频残缺 | control | 0.059200 | 0.338700 |
| 图像干净、音频残缺 | v11d | 0.058600 | 0.337100 |
| 音频干净、图像残缺 | control | 0.066200 | 0.359300 |
| 音频干净、图像残缺 | v11d | 0.066300 | 0.359300 |
| 双模态残缺 | control | 0.066200 | 0.361900 |
| 双模态残缺 | v11d | 0.066200 | 0.361900 |
| 仅音频残缺 | control | 0.007700 | 0.117000 |
| 仅音频残缺 | v11d | 0.010500 | 0.137500 |
| 仅图像残缺 | control | 0.066200 | 0.359300 |
| 仅图像残缺 | v11d | 0.066200 | 0.359300 |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.047200 | 0.146200 | 1.000000 |
| 干净双模态 | v11d | 0.047200 | 0.145900 | 1.000000 |
| 仅干净图像 | control | 0.021300 | 0.041000 | 0.358400 |
| 仅干净图像 | v11d | 0.017900 | 0.035000 | 0.342300 |
| 仅干净音频 | control | 0.048300 | 0.147600 | 1.000000 |
| 仅干净音频 | v11d | 0.048500 | 0.147800 | 1.000000 |
| 图像干净、音频残缺 | control | 0.046300 | 0.143900 | 1.000000 |
| 图像干净、音频残缺 | v11d | 0.046300 | 0.143700 | 1.000000 |
| 音频干净、图像残缺 | control | 0.047400 | 0.146400 | 1.000000 |
| 音频干净、图像残缺 | v11d | 0.046300 | 0.144700 | 1.000000 |
| 双模态残缺 | control | 0.046300 | 0.143700 | 1.000000 |
| 双模态残缺 | v11d | 0.045200 | 0.142000 | 1.000000 |
| 仅音频残缺 | control | 0.047400 | 0.145200 | 1.000000 |
| 仅音频残缺 | v11d | 0.047600 | 0.145400 | 1.000000 |
| 仅图像残缺 | control | 0.021600 | 0.040700 | 0.362900 |
| 仅图像残缺 | v11d | 0.016500 | 0.030600 | 0.321600 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 干净双模态 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.300% |
| 仅干净图像 | control | 0.042600 | 0.148600 | 1.000000 | 48.400% |
| 仅干净图像 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.300% |
| 仅干净音频 | control | 0.042600 | 0.148600 | 1.000000 | 62.900% |
| 仅干净音频 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.800% |
| 图像干净、音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 62.600% |
| 图像干净、音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.700% |
| 音频干净、图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 63.200% |
| 音频干净、图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 63.200% |
| 双模态残缺 | control | 0.042600 | 0.148600 | 1.000000 | 62.500% |
| 双模态残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.500% |
| 仅音频残缺 | control | 0.042600 | 0.148600 | 1.000000 | 62.100% |
| 仅音频残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 62.100% |
| 仅图像残缺 | control | 0.042600 | 0.148600 | 1.000000 | 48.100% |
| 仅图像残缺 | v11d | 0.042600 | 0.148600 | 1.000000 | 48.000% |

**本 family 干预**

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A | N/A |
| 仅干净音频 | control | 0.000100 | 0.000100 | -0.000300 |
| 仅干净音频 | v11d | 0.000000 | 0.000000 | -0.000200 |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | control | 0.000000 | 0.000100 | -0.000300 |
| 仅音频残缺 | v11d | 0.000000 | 0.000000 | -0.000200 |
| 仅图像残缺 | control | N/A | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | control | 0.011000 | 0.028400 |
| 仅干净音频 | v11d | 0.011000 | 0.028400 |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | control | 0.014200 | 0.036800 |
| 音频干净、图像残缺 | v11d | 0.014200 | 0.036800 |
| 双模态残缺 | control | 0.014600 | 0.037700 |
| 双模态残缺 | v11d | 0.014600 | 0.037700 |
| 仅音频残缺 | control | 0.011200 | 0.028800 |
| 仅音频残缺 | v11d | 0.011200 | 0.028800 |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11d | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.000100 | 0.000200 | 0.000000 |
| 图像干净、音频残缺 | v11d | 0.000100 | 0.000200 | 0.000000 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A | N/A |
| 双模态残缺 | control | 0.000200 | 0.000200 | 0.000000 |
| 双模态残缺 | v11d | 0.000100 | 0.000200 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11d | 0.000000 | 0.000000 | 0.000000 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | control | 0.042900 | 0.149700 |
| 仅干净图像 | v11d | 0.042900 | 0.149700 |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | control | 0.047500 | 0.176400 |
| 图像干净、音频残缺 | v11d | 0.047500 | 0.176400 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | control | 0.062000 | 0.148300 |
| 双模态残缺 | v11d | 0.062000 | 0.148300 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | control | 0.052500 | 0.126000 |
| 仅图像残缺 | v11d | 0.052500 | 0.126000 |

**音频 Cross-Detail → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.016000 | 0.000000 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.000000 | 0.000000 |
| 双模态残缺 | v11d | 0.000000 | 0.000000 |
| 仅音频残缺 | v11d | 0.015600 | -0.000400 |
| 仅图像残缺 | v11d | N/A | N/A |

**音频 Cross-Detail → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | N/A | N/A |
| 仅干净音频 | v11d | 0.212200 | 0.930400 |
| 图像干净、音频残缺 | v11d | N/A | N/A |
| 音频干净、图像残缺 | v11d | 0.170500 | 0.760900 |
| 双模态残缺 | v11d | 0.165500 | 0.767100 |
| 仅音频残缺 | v11d | 0.205500 | 0.939700 |
| 仅图像残缺 | v11d | N/A | N/A |

**图像 Cross-Detail → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | same−normal（配对） |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | -0.000200 | 0.000000 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | 0.000200 | 0.000000 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.000100 | 0.000000 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | -0.000400 | 0.000000 |

**图像 Cross-Detail → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11d | N/A | N/A |
| 仅干净图像 | v11d | 0.164900 | 0.569800 |
| 仅干净音频 | v11d | N/A | N/A |
| 图像干净、音频残缺 | v11d | 0.172700 | 0.595300 |
| 音频干净、图像残缺 | v11d | N/A | N/A |
| 双模态残缺 | v11d | 0.212300 | 0.585300 |
| 仅音频残缺 | v11d | N/A | N/A |
| 仅图像残缺 | v11d | 0.197300 | 0.555900 |

</details>

**独立音频 family breakdown**

此处来自各实验的 `tables/audio_family_breakdown_fixed.csv`，单独改变音频 family；不能用它替代上面的双 family-pair 主表或再次计入宏平均。该 CSV 不含逐指标 n。


<details>
<summary>time_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 95.740% | 0.009131 | 0.946754 | 0.008777 | 0.721565 |
| 图像干净、音频残缺 | v11d | 95.740% | 0.009336 | 0.945202 | 0.008861 | 0.718910 |
| 双模态残缺 | control | 90.000% | 0.005953 | 0.965617 | 0.008868 | 0.718962 |
| 双模态残缺 | v11d | 90.000% | 0.005942 | 0.965691 | 0.008938 | 0.716434 |
| 仅音频残缺 | control | 57.600% | 0.061886 | 0.509040 | 0.009232 | 0.709561 |
| 仅音频残缺 | v11d | 57.600% | 0.060006 | 0.528310 | 0.009269 | 0.707398 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 21.198114 | 0.011825 | 0.032886 |
| 图像干净、音频残缺 | v11d | 21.082964 | 0.011998 | 0.032370 |
| 双模态残缺 | control | 25.493052 | 0.012070 | 0.033384 |
| 双模态残缺 | v11d | 25.495064 | 0.012175 | 0.032952 |
| 仅音频残缺 | control | 12.390107 | 0.012532 | 0.034531 |
| 仅音频残缺 | v11d | 12.524009 | 0.012557 | 0.034677 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.006692 | 0.027873 |
| 图像干净、音频残缺 | v11d | 0.006715 | 0.028001 |
| 双模态残缺 | control | 0.006677 | 0.027838 |
| 双模态残缺 | v11d | 0.006723 | 0.028056 |
| 仅音频残缺 | control | 0.006974 | 0.028615 |
| 仅音频残缺 | v11d | 0.007020 | 0.028767 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.132980 | 0.148571 | 59.048% |
| 图像干净、音频残缺 | v11d | 0.131910 | 0.148571 | 58.887% |
| 双模态残缺 | control | 0.132322 | 0.148571 | 58.880% |
| 双模态残缺 | v11d | 0.131880 | 0.148571 | 58.740% |
| 仅音频残缺 | control | 0.134211 | 0.148571 | 58.376% |
| 仅音频残缺 | v11d | 0.134106 | 0.148571 | 58.349% |

</details>


<details>
<summary>freq_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 92.020% | 0.009157 | 0.946567 | 0.007522 | 0.762156 |
| 图像干净、音频残缺 | v11d | 92.020% | 0.009349 | 0.945073 | 0.007469 | 0.764970 |
| 双模态残缺 | control | 84.170% | 0.006045 | 0.964890 | 0.007637 | 0.757635 |
| 双模态残缺 | v11d | 84.170% | 0.006042 | 0.964903 | 0.007626 | 0.759554 |
| 仅音频残缺 | control | 49.610% | 0.062624 | 0.501700 | 0.008194 | 0.744206 |
| 仅音频残缺 | v11d | 49.610% | 0.061419 | 0.514941 | 0.008244 | 0.741614 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 21.182674 | 0.009180 | 0.030366 |
| 图像干净、音频残缺 | v11d | 21.076488 | 0.009079 | 0.030231 |
| 双模态残缺 | control | 25.353146 | 0.009462 | 0.030867 |
| 双模态残缺 | v11d | 25.357925 | 0.009393 | 0.030907 |
| 仅音频残缺 | control | 12.298081 | 0.010252 | 0.032495 |
| 仅音频残缺 | v11d | 12.393634 | 0.010272 | 0.032609 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.006388 | 0.026399 |
| 图像干净、音频残缺 | v11d | 0.006367 | 0.026356 |
| 双模态残缺 | control | 0.006389 | 0.026452 |
| 双模态残缺 | v11d | 0.006417 | 0.026573 |
| 仅音频残缺 | control | 0.006786 | 0.027412 |
| 仅音频残缺 | v11d | 0.006856 | 0.027675 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.138160 | 0.148571 | 61.013% |
| 图像干净、音频残缺 | v11d | 0.138040 | 0.148571 | 61.131% |
| 双模态残缺 | control | 0.137955 | 0.148571 | 60.896% |
| 双模态残缺 | v11d | 0.138519 | 0.148571 | 60.990% |
| 仅音频残缺 | control | 0.139555 | 0.148571 | 60.084% |
| 仅音频残缺 | v11d | 0.139796 | 0.148571 | 60.006% |

</details>


<details>
<summary>feature_dropout</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 96.010% | 0.009124 | 0.946799 | 0.005787 | 0.827544 |
| 图像干净、音频残缺 | v11d | 96.010% | 0.009330 | 0.945336 | 0.005774 | 0.829228 |
| 双模态残缺 | control | 91.280% | 0.005931 | 0.965488 | 0.005826 | 0.826515 |
| 双模态残缺 | v11d | 91.280% | 0.005928 | 0.965513 | 0.005852 | 0.827234 |
| 仅音频残缺 | control | 64.470% | 0.059434 | 0.530340 | 0.006064 | 0.818714 |
| 仅音频残缺 | v11d | 64.470% | 0.057539 | 0.554241 | 0.006104 | 0.817755 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 21.201446 | 0.005866 | 0.024930 |
| 图像干净、音频残缺 | v11d | 21.092883 | 0.005848 | 0.024948 |
| 双模态残缺 | control | 25.550541 | 0.005915 | 0.025067 |
| 双模态残缺 | v11d | 25.556520 | 0.005940 | 0.025210 |
| 仅音频残缺 | control | 12.549016 | 0.006147 | 0.025706 |
| 仅音频残缺 | v11d | 12.716800 | 0.006186 | 0.025863 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.005735 | 0.024706 |
| 图像干净、音频残缺 | v11d | 0.005724 | 0.024733 |
| 双模态残缺 | control | 0.005766 | 0.024792 |
| 双模态残缺 | v11d | 0.005793 | 0.024941 |
| 仅音频残缺 | control | 0.006008 | 0.025471 |
| 仅音频残缺 | v11d | 0.006050 | 0.025631 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.140158 | 0.148571 | 62.842% |
| 图像干净、音频残缺 | v11d | 0.140167 | 0.148571 | 62.878% |
| 双模态残缺 | control | 0.140123 | 0.148571 | 62.761% |
| 双模态残缺 | v11d | 0.140709 | 0.148571 | 62.767% |
| 仅音频残缺 | control | 0.141495 | 0.148571 | 62.448% |
| 仅音频残缺 | v11d | 0.141609 | 0.148571 | 62.354% |

</details>


<details>
<summary>partial_temporal</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 93.350% | 0.009167 | 0.946488 | 0.013235 | 0.581393 |
| 图像干净、音频残缺 | v11d | 93.350% | 0.009336 | 0.945016 | 0.013571 | 0.554801 |
| 双模态残缺 | control | 82.670% | 0.006021 | 0.965093 | 0.013411 | 0.576608 |
| 双模态残缺 | v11d | 82.670% | 0.006010 | 0.965145 | 0.013670 | 0.553746 |
| 仅音频残缺 | control | 26.570% | 0.066168 | 0.458636 | 0.014582 | 0.555665 |
| 仅音频残缺 | v11d | 26.570% | 0.064167 | 0.477217 | 0.014562 | 0.553423 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 21.178982 | 0.031669 | 0.083916 |
| 图像干净、音频残缺 | v11d | 21.071336 | 0.032497 | 0.082163 |
| 双模态残缺 | control | 25.400256 | 0.032109 | 0.084655 |
| 双模态残缺 | v11d | 25.408011 | 0.032737 | 0.083017 |
| 仅音频残缺 | control | 12.029531 | 0.034953 | 0.090854 |
| 仅音频残缺 | v11d | 12.156998 | 0.034894 | 0.090497 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.000621 | 0.007633 |
| 图像干净、音频残缺 | v11d | 0.000621 | 0.007653 |
| 双模态残缺 | control | 0.000618 | 0.007608 |
| 双模态残缺 | v11d | 0.000624 | 0.007674 |
| 仅音频残缺 | control | 0.000645 | 0.007721 |
| 仅音频残缺 | v11d | 0.000651 | 0.007762 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.107219 | 0.148571 | 53.749% |
| 图像干净、音频残缺 | v11d | 0.099185 | 0.148571 | 53.512% |
| 双模态残缺 | control | 0.106595 | 0.148571 | 53.437% |
| 双模态残缺 | v11d | 0.099875 | 0.148571 | 53.269% |
| 仅音频残缺 | control | 0.110287 | 0.148571 | 51.311% |
| 仅音频残缺 | v11d | 0.109154 | 0.148571 | 51.305% |

</details>


<details>
<summary>time_freq_block</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 95.020% | 0.009131 | 0.946763 | 0.006213 | 0.815572 |
| 图像干净、音频残缺 | v11d | 95.020% | 0.009330 | 0.945324 | 0.006185 | 0.817586 |
| 双模态残缺 | control | 89.780% | 0.005995 | 0.965327 | 0.006304 | 0.813835 |
| 双模态残缺 | v11d | 89.780% | 0.005989 | 0.965368 | 0.006317 | 0.814762 |
| 仅音频残缺 | control | 61.910% | 0.060140 | 0.525117 | 0.006609 | 0.803524 |
| 仅音频残缺 | v11d | 61.910% | 0.058327 | 0.546707 | 0.006646 | 0.801813 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 21.198715 | 0.007043 | 0.025690 |
| 图像干净、音频残缺 | v11d | 21.090033 | 0.006958 | 0.025554 |
| 双模态残缺 | control | 25.522866 | 0.007471 | 0.026526 |
| 双模态残缺 | v11d | 25.532791 | 0.007425 | 0.026563 |
| 仅音频残缺 | control | 12.498258 | 0.007734 | 0.027165 |
| 仅音频残缺 | v11d | 12.654100 | 0.007713 | 0.027255 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.006049 | 0.025904 |
| 图像干净、音频残缺 | v11d | 0.006033 | 0.025893 |
| 双模态残缺 | control | 0.006074 | 0.025937 |
| 双模态残缺 | v11d | 0.006098 | 0.026065 |
| 仅音频残缺 | control | 0.006387 | 0.026788 |
| 仅音频残缺 | v11d | 0.006435 | 0.026963 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.143682 | 0.148571 | 62.595% |
| 图像干净、音频残缺 | v11d | 0.143441 | 0.148571 | 62.652% |
| 双模态残缺 | control | 0.143816 | 0.148571 | 62.534% |
| 双模态残缺 | v11d | 0.144145 | 0.148571 | 62.586% |
| 仅音频残缺 | control | 0.145005 | 0.148571 | 62.104% |
| 仅音频残缺 | v11d | 0.145120 | 0.148571 | 62.057% |

</details>

**训练统计**

仅统计每个完整 epoch 的平均训练 loss；不同 loss 定义不可横比，最低训练 loss 也不是测试最优 checkpoint。

| 实验 | 完成 epoch | 本轮轮数 | 首轮 loss | 末轮 loss | 最低 loss | 末轮 LR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| control | 150–199 | 50 | 9.8300 | 10.2570 | 8.6495 | 0.000100 |
| v11d | 150–199 | 50 | 9.8345 | 10.2892 | 8.6846 | 0.000100 |


<details>
<summary>逐 epoch loss / LR</summary>

| epoch | 实验 | 平均 loss | LR |
| --- | --- | ---: | ---: |
| 150 | control | 9.8300 | 0.000100 |
| 150 | v11d | 9.8345 | 0.000100 |
| 151 | control | 10.3606 | 0.000100 |
| 151 | v11d | 10.3866 | 0.000100 |
| 152 | control | 10.9675 | 0.000100 |
| 152 | v11d | 10.9915 | 0.000100 |
| 153 | control | 9.9119 | 0.000100 |
| 153 | v11d | 9.9355 | 0.000100 |
| 154 | control | 10.1601 | 0.000100 |
| 154 | v11d | 10.1835 | 0.000100 |
| 155 | control | 10.0570 | 0.000100 |
| 155 | v11d | 10.0827 | 0.000100 |
| 156 | control | 9.8895 | 0.000100 |
| 156 | v11d | 9.9198 | 0.000100 |
| 157 | control | 9.4665 | 0.000100 |
| 157 | v11d | 9.4926 | 0.000100 |
| 158 | control | 9.5434 | 0.000100 |
| 158 | v11d | 9.5722 | 0.000100 |
| 159 | control | 9.5440 | 0.000100 |
| 159 | v11d | 9.5711 | 0.000100 |
| 160 | control | 9.4741 | 0.000100 |
| 160 | v11d | 9.5007 | 0.000100 |
| 161 | control | 9.3795 | 0.000100 |
| 161 | v11d | 9.4102 | 0.000100 |
| 162 | control | 9.6523 | 0.000100 |
| 162 | v11d | 9.6822 | 0.000100 |
| 163 | control | 9.5649 | 0.000100 |
| 163 | v11d | 9.5959 | 0.000100 |
| 164 | control | 9.1173 | 0.000100 |
| 164 | v11d | 9.1458 | 0.000100 |
| 165 | control | 9.1032 | 0.000100 |
| 165 | v11d | 9.1349 | 0.000100 |
| 166 | control | 9.2910 | 0.000100 |
| 166 | v11d | 9.3217 | 0.000100 |
| 167 | control | 9.2397 | 0.000100 |
| 167 | v11d | 9.2688 | 0.000100 |
| 168 | control | 9.8440 | 0.000100 |
| 168 | v11d | 9.8700 | 0.000100 |
| 169 | control | 9.4167 | 0.000100 |
| 169 | v11d | 9.4440 | 0.000100 |
| 170 | control | 9.7290 | 0.000100 |
| 170 | v11d | 9.7601 | 0.000100 |
| 171 | control | 9.0857 | 0.000100 |
| 171 | v11d | 9.1167 | 0.000100 |
| 172 | control | 9.2577 | 0.000100 |
| 172 | v11d | 9.2901 | 0.000100 |
| 173 | control | 10.4258 | 0.000100 |
| 173 | v11d | 10.4559 | 0.000100 |
| 174 | control | 9.6558 | 0.000100 |
| 174 | v11d | 9.6856 | 0.000100 |
| 175 | control | 9.9500 | 0.000100 |
| 175 | v11d | 9.9781 | 0.000100 |
| 176 | control | 9.7017 | 0.000100 |
| 176 | v11d | 9.7320 | 0.000100 |
| 177 | control | 9.4412 | 0.000100 |
| 177 | v11d | 9.4723 | 0.000100 |
| 178 | control | 10.0218 | 0.000100 |
| 178 | v11d | 10.0502 | 0.000100 |
| 179 | control | 9.8342 | 0.000100 |
| 179 | v11d | 9.8682 | 0.000100 |
| 180 | control | 9.7331 | 0.000100 |
| 180 | v11d | 9.7675 | 0.000100 |
| 181 | control | 8.6495 | 0.000100 |
| 181 | v11d | 8.6846 | 0.000100 |
| 182 | control | 9.4295 | 0.000100 |
| 182 | v11d | 9.4608 | 0.000100 |
| 183 | control | 9.2305 | 0.000100 |
| 183 | v11d | 9.2612 | 0.000100 |
| 184 | control | 9.6476 | 0.000100 |
| 184 | v11d | 9.6771 | 0.000100 |
| 185 | control | 9.7560 | 0.000100 |
| 185 | v11d | 9.7890 | 0.000100 |
| 186 | control | 9.8138 | 0.000100 |
| 186 | v11d | 9.8440 | 0.000100 |
| 187 | control | 9.6294 | 0.000100 |
| 187 | v11d | 9.6626 | 0.000100 |
| 188 | control | 9.5826 | 0.000100 |
| 188 | v11d | 9.6159 | 0.000100 |
| 189 | control | 9.6813 | 0.000100 |
| 189 | v11d | 9.7153 | 0.000100 |
| 190 | control | 9.9143 | 0.000100 |
| 190 | v11d | 9.9487 | 0.000100 |
| 191 | control | 9.5900 | 0.000100 |
| 191 | v11d | 9.6230 | 0.000100 |
| 192 | control | 10.1978 | 0.000100 |
| 192 | v11d | 10.2313 | 0.000100 |
| 193 | control | 9.9765 | 0.000100 |
| 193 | v11d | 10.0095 | 0.000100 |
| 194 | control | 9.7949 | 0.000100 |
| 194 | v11d | 9.8274 | 0.000100 |
| 195 | control | 9.1949 | 0.000100 |
| 195 | v11d | 9.2289 | 0.000100 |
| 196 | control | 9.0240 | 0.000100 |
| 196 | v11d | 9.0591 | 0.000100 |
| 197 | control | 9.5696 | 0.000100 |
| 197 | v11d | 9.6023 | 0.000100 |
| 198 | control | 9.1441 | 0.000100 |
| 198 | v11d | 9.1764 | 0.000100 |
| 199 | control | 10.2570 | 0.000100 |
| 199 | v11d | 10.2892 | 0.000100 |

</details>

训练证据：
- `control`：[train_v11d_suite.log](../v11d_outputs_with_ckpt/outputs/train_v11d_suite.log)
- `v11d`：[train_v11d_suite.log](../v11d_outputs_with_ckpt/outputs/train_v11d_suite.log)

**Demo 小样本**

每份 demo 为 10 条样本，不是全测试集分数；fixed/random 分开。原始图片与逐样本预测留在对应产物目录，本节不宣称重新进行了图像质量评审。


<details>
<summary>fixed_mask demo (n=10)</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 50.000% | 0.056400 | 0.518000 | 0.015400 | 0.669000 |
| 仅音频残缺 | v11d | 50.000% | 0.052300 | 0.555000 | 0.015700 | 0.665000 |
| 仅图像残缺 | control | 100.000% | 0.005000 | 0.967000 | 0.028500 | 0.196000 |
| 仅图像残缺 | v11d | 100.000% | 0.005000 | 0.968000 | 0.028900 | 0.224000 |
| 双模态残缺 | control | 100.000% | 0.004900 | 0.969000 | 0.013100 | 0.712000 |
| 双模态残缺 | v11d | 100.000% | 0.004700 | 0.970000 | 0.013100 | 0.720000 |

| 输入模式 | 实验 | 图像缺失区 MSE | 音频缺失区 MSE |
| --- | --- | ---: | ---: |
| 仅音频残缺 | control | 0.056400 | 0.029700 |
| 仅音频残缺 | v11d | 0.052300 | 0.030300 |
| 仅图像残缺 | control | 0.015000 | 0.028500 |
| 仅图像残缺 | v11d | 0.015300 | 0.028900 |
| 双模态残缺 | control | 0.014500 | 0.024600 |
| 双模态残缺 | v11d | 0.014300 | 0.024700 |

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.059200 | 0.157700 | 1.000000 |
| 仅音频残缺 | v11d | 0.059500 | 0.157700 | 1.000000 |
| 仅图像残缺 | control | 0.025600 | 0.048000 | 0.354800 |
| 仅图像残缺 | v11d | 0.022300 | 0.042900 | 0.351500 |
| 双模态残缺 | control | 0.056900 | 0.155500 | 1.000000 |
| 双模态残缺 | v11d | 0.057400 | 0.156100 | 1.000000 |

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.060800 | 0.189100 | 1.000000 | 60.800% |
| 仅音频残缺 | v11d | 0.060800 | 0.189100 | 1.000000 | 60.500% |
| 仅图像残缺 | control | 0.060800 | 0.189100 | 1.000000 | 50.400% |
| 仅图像残缺 | v11d | 0.060800 | 0.189100 | 1.000000 | 50.000% |
| 双模态残缺 | control | 0.060800 | 0.189100 | 1.000000 | 64.000% |
| 双模态残缺 | v11d | 0.060800 | 0.189100 | 1.000000 | 63.700% |

</details>

- legacy_random demo：未找到。

**各实验结论与比较限制**

- **仅音频残缺**：主实验相对 control，音频缺失区 MSE 变化 +0.00%，Index ACC 变化 +0.000 个百分点。
- **双模态残缺**：主实验相对 control，音频缺失区 MSE 变化 +0.00%，Index ACC 变化 +0.000 个百分点。
- **control**：固定伪配对且保留 Cross-Key，关闭 Cross-Detail。它不是 Cross-Key-off 对照。
- **v11d**：在同样的伪配对条件加入 Cross-Detail；同类替换损害很小，不能证明模型恢复了绑定实例的细节。
- 本版所有 cue 使用 sample/sample；与 v11c/e/f/g 单模态缺失时的类别 medoid 目标不同，跨版本误差不能直接排名。
- 本节仅汇总已有结果，没有重跑训练或推理。单 seed、重复音频曝光和旧日志舍入限制仍在，不报告统计显著性。

[返回统一评估导航](#evaluation-format-20260912)

<a id="evaluation-v11e"></a>

### v11e

本节覆盖 **2 组实验、26 个主评估/干预字段、7 个音频分布诊断字段**，以及独立 family、训练统计和本地已有 demo。字段按实际产物统计，含明确标注的不适用项；不把其他版本的缺项补成零。

**实验说明**

| 实验 | 权重起点 | 本轮训练范围 | 实际轮数 |
| --- | --- | ---: | ---: |
| control | 从头训练 | 无 Cross-Key/causal | 100（0–99） |
| v11e | 从头训练 | Cross-Key + causal | 100（0–99） |

训练配置 seed=1234、batch_size=128；fixed_mask seed=1234、severity=0.4。轮数以上表和完成日志为准，不把父模型历史轮数计成本轮新增训练。历史分支配置不是当前分支的运行入口。

**恢复目标：仅图像为 sample/category，仅音频为 category/sample，双模态为 sample/sample。** category 使用训练集 medoid；“仅音频残缺”表示图像全缺失、音频部分残缺，反向同理。

| 实验 | fixed / random | Cross-Key 扫描 | demo fixed / random |
| --- | --- | ---: | ---: |
| control | 有 / 未找到 | 未找到 | 未找到 / 未找到 |
| v11e | 有 / 有 | fixed | 有 / 有 |

**数据来源**：
- `control`：[本地产物](../v11e_outputs_with_ckpt/outputs/outputs_v11e_control/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。
- `v11e`：[本地产物](../v11e_outputs_with_ckpt/outputs/outputs_v11e/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。

MSE/L1 越低越好，ACC/SSIM/PSNR 越高越好；gate、res/V 和多样性仅是诊断。宏平均为五个 family 等权均值，不是把 normal、sweep、独立音频 family 重复叠加。N/A 表示未保存或在该场景不适用，详见字段覆盖说明。

**精度与缺项**：主表源 CSV/日志已舍入（ACC 常到 0.1%、MSE 常到 4 位、SSIM 常到 3 位）；本节六位显示只是统一排版，不恢复丢失精度。未保存逐指标有效 n、区域 L1、内容 ACC、干预绝对误差/win 等字段不编造。独立音频 family CSV 中实际存在的 L1 仍在其专表报告。

**分类与恢复：fixed 五类等权宏平均**

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.200% | 0.009700 | 0.943000 | 0.003600 | 0.911000 |
| 干净双模态 | v11e | 99.500% | 0.012900 | 0.922000 | 0.003700 | 0.911000 |
| 仅干净图像 | control | 97.700% | 0.009800 | 0.943000 | 0.000300 | 0.966000 |
| 仅干净图像 | v11e | 97.400% | 0.013200 | 0.921000 | 0.000300 | 0.968000 |
| 仅干净音频 | control | 98.800% | 0.002100 | 0.986000 | 0.003600 | 0.911000 |
| 仅干净音频 | v11e | 96.700% | 0.003100 | 0.976000 | 0.003700 | 0.911000 |
| 图像干净、音频残缺 | control | 98.940% | 0.009780 | 0.942800 | 0.006100 | 0.807400 |
| 图像干净、音频残缺 | v11e | 98.720% | 0.012960 | 0.921800 | 0.005920 | 0.818000 |
| 音频干净、图像残缺 | control | 99.140% | 0.007500 | 0.956200 | 0.003620 | 0.911000 |
| 音频干净、图像残缺 | v11e | 98.960% | 0.007360 | 0.957200 | 0.003700 | 0.911000 |
| 双模态残缺 | control | 97.560% | 0.007500 | 0.956200 | 0.006160 | 0.805800 |
| 双模态残缺 | v11e | 97.100% | 0.007500 | 0.956200 | 0.006060 | 0.815800 |
| 仅音频残缺 | control | 84.580% | 0.013220 | 0.877600 | 0.006300 | 0.804600 |
| 仅音频残缺 | v11e | 83.620% | 0.012460 | 0.895200 | 0.006160 | 0.811200 |
| 仅图像残缺 | control | 92.020% | 0.007700 | 0.955200 | 0.000660 | 0.916800 |
| 仅图像残缺 | v11e | 92.020% | 0.007860 | 0.954200 | 0.000680 | 0.918800 |


<details>
<summary>fixed 五类等权宏平均：区域、内容、多样性与音频分布完整指标</summary>

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 20.790000 | N/A | N/A |
| 干净双模态 | v11e | 19.600000 | N/A | N/A |
| 仅干净图像 | control | 20.770000 | N/A | N/A |
| 仅干净图像 | v11e | 19.480000 | N/A | N/A |
| 仅干净音频 | control | 37.620000 | 0.002100 | 0.002100 |
| 仅干净音频 | v11e | 39.510000 | 0.003100 | 0.003100 |
| 图像干净、音频残缺 | control | 20.768000 | N/A | N/A |
| 图像干净、音频残缺 | v11e | 19.564000 | N/A | N/A |
| 音频干净、图像残缺 | control | 22.994000 | 0.054820 | 0.025940 |
| 音频干净、图像残缺 | v11e | 23.216000 | 0.049420 | 0.025500 |
| 双模态残缺 | control | 22.976000 | 0.054680 | 0.025920 |
| 双模态残缺 | v11e | 23.152000 | 0.049860 | 0.025920 |
| 仅音频残缺 | control | 30.892000 | 0.013220 | 0.013220 |
| 仅音频残缺 | v11e | 32.010000 | 0.012460 | 0.012460 |
| 仅图像残缺 | control | 22.906000 | 0.056340 | 0.026640 |
| 仅图像残缺 | v11e | 22.962000 | 0.052700 | 0.027500 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | 0.030720 | 0.000000 |
| 音频干净、图像残缺 | v11e | 0.029740 | 0.000000 |
| 双模态残缺 | control | 0.030560 | 0.000000 |
| 双模态残缺 | v11e | 0.029860 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | 0.031260 | 0.000000 |
| 仅图像残缺 | v11e | 0.030700 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | 0.000300 | N/A |
| 仅干净图像 | v11e | 0.000300 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | 0.010620 | 0.003140 |
| 图像干净、音频残缺 | v11e | 0.010200 | 0.003140 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | 0.010780 | 0.003160 |
| 双模态残缺 | v11e | 0.010400 | 0.003140 |
| 仅音频残缺 | control | 0.011060 | 0.003140 |
| 仅音频残缺 | v11e | 0.010720 | 0.003140 |
| 仅图像残缺 | control | 0.000660 | N/A |
| 仅图像残缺 | v11e | 0.000680 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.058900 | 0.338700 |
| 干净双模态 | v11e | 0.053500 | 0.322420 |
| 仅干净图像 | control | 0.059400 | 0.340040 |
| 仅干净图像 | v11e | 0.054700 | 0.326040 |
| 仅干净音频 | control | 0.046300 | 0.287560 |
| 仅干净音频 | v11e | 0.046000 | 0.286560 |
| 图像干净、音频残缺 | control | 0.059320 | 0.339480 |
| 图像干净、音频残缺 | v11e | 0.053580 | 0.322520 |
| 音频干净、图像残缺 | control | 0.061060 | 0.345200 |
| 音频干净、图像残缺 | v11e | 0.061240 | 0.345600 |
| 双模态残缺 | control | 0.061060 | 0.345660 |
| 双模态残缺 | v11e | 0.061160 | 0.345880 |
| 仅音频残缺 | control | 0.040040 | 0.265080 |
| 仅音频残缺 | v11e | 0.039000 | 0.260540 |
| 仅图像残缺 | control | 0.060820 | 0.344480 |
| 仅图像残缺 | v11e | 0.060840 | 0.344500 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11e | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11e | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11e | 0.042900 | 0.154500 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11e | 0.015200 | 0.068400 | 1.000000 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11e | 0.042800 | 0.154400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.040800 | 0.147600 | 1.000000 |
| 图像干净、音频残缺 | v11e | 0.040420 | 0.146200 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155680 | 1.000000 |
| 音频干净、图像残缺 | v11e | 0.043000 | 0.154640 | 1.000000 |
| 双模态残缺 | control | 0.040800 | 0.147500 | 1.000000 |
| 双模态残缺 | v11e | 0.040540 | 0.146220 | 1.000000 |
| 仅音频残缺 | control | 0.040340 | 0.146620 | 1.000000 |
| 仅音频残缺 | v11e | 0.039840 | 0.145020 | 1.000000 |
| 仅图像残缺 | control | 0.015280 | 0.066280 | 0.999900 |
| 仅图像残缺 | v11e | 0.014920 | 0.064460 | 0.999520 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11e | 0.015600 | 0.070500 | 0.987900 | 82.700% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.400% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.340% |
| 图像干净、音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 78.740% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.240% |
| 双模态残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 78.600% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 77.940% |
| 仅音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 78.220% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 78.700% |
| 仅图像残缺 | v11e | 0.015600 | 0.070500 | 0.987900 | 80.120% |

</details>


<details>
<summary>fixed 五类等权宏平均：各字段有效样本数</summary>

| 实验 | 指标字段（同一计数口径） | 每 family 的 n |
| --- | --- | ---: |
| control | `acc`、`aud_masked_mse`、`aud_mse` | 未逐指标保存 |
| control | `aud_ssim`、`aud_visible_mse`、`img_coarse_masked_mse` | 未逐指标保存 |
| control | `img_coarse_visible_mse`、`img_masked_mse`、`img_mse` | 未逐指标保存 |
| control | `img_visible_mse`、`pair_l2`、`pix_var` | 未逐指标保存 |
| control | `psnr`、`rec_max`、`rec_mean` | 未逐指标保存 |
| control | `rec_std`、`ssim`、`tgt_max` | 未逐指标保存 |
| control | `tgt_mean`、`tgt_std`、`top15_recall` | 未逐指标保存 |
| control | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| v11e | `acc`、`aud_masked_mse`、`aud_mse` | 未逐指标保存 |
| v11e | `aud_ssim`、`aud_visible_mse`、`img_coarse_masked_mse` | 未逐指标保存 |
| v11e | `img_coarse_visible_mse`、`img_masked_mse`、`img_mse` | 未逐指标保存 |
| v11e | `img_visible_mse`、`pair_l2`、`pix_var` | 未逐指标保存 |
| v11e | `psnr`、`rec_max`、`rec_mean` | 未逐指标保存 |
| v11e | `rec_std`、`ssim`、`tgt_max` | 未逐指标保存 |
| v11e | `tgt_mean`、`tgt_std`、`top15_recall` | 未逐指标保存 |
| v11e | `pair_a2i_r1`、`pair_i2a_r1` | 0 |

</details>

**分类与恢复：random 单次评估**

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A | N/A |
| 干净双模态 | v11e | 99.500% | 0.012900 | 0.922000 | 0.003700 | 0.911000 |
| 仅干净图像 | control | N/A | N/A | N/A | N/A | N/A |
| 仅干净图像 | v11e | 97.400% | 0.013200 | 0.921000 | 0.000300 | 0.968000 |
| 仅干净音频 | control | N/A | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11e | 96.700% | 0.003100 | 0.976000 | 0.003700 | 0.911000 |
| 图像干净、音频残缺 | control | N/A | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11e | 98.900% | 0.012900 | 0.922000 | 0.006000 | 0.839000 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11e | 99.000% | 0.008700 | 0.949000 | 0.003700 | 0.911000 |
| 双模态残缺 | control | N/A | N/A | N/A | N/A | N/A |
| 双模态残缺 | v11e | 97.400% | 0.008900 | 0.947000 | 0.006100 | 0.837000 |
| 仅音频残缺 | control | N/A | N/A | N/A | N/A | N/A |
| 仅音频残缺 | v11e | 88.700% | 0.009800 | 0.920000 | 0.006200 | 0.833000 |
| 仅图像残缺 | control | N/A | N/A | N/A | N/A | N/A |
| 仅图像残缺 | v11e | 89.400% | 0.009600 | 0.942000 | 0.000800 | 0.902000 |

random 不与 fixed 混算；表中整行 N/A 表示该实验未找到 random 结果，不表示实验得分为 0。


<details>
<summary>random 单次评估：区域、内容、多样性与音频分布完整指标</summary>

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | 19.600000 | N/A | N/A |
| 仅干净图像 | v11e | 19.480000 | N/A | N/A |
| 仅干净音频 | v11e | 39.510000 | 0.003100 | 0.003100 |
| 图像干净、音频残缺 | v11e | 19.570000 | N/A | N/A |
| 音频干净、图像残缺 | v11e | 23.250000 | 0.086600 | 0.056000 |
| 双模态残缺 | v11e | 23.110000 | 0.088800 | 0.058000 |
| 仅音频残缺 | v11e | 34.020000 | 0.009800 | 0.009800 |
| 仅图像残缺 | v11e | 22.890000 | 0.095400 | 0.062500 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.016200 | 0.000000 |
| 双模态残缺 | v11e | 0.016300 | 0.000000 |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | v11e | 0.016800 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | 0.000300 | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.008800 | 0.004100 |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | v11e | 0.008900 | 0.004100 |
| 仅音频残缺 | v11e | 0.009300 | 0.004100 |
| 仅图像残缺 | v11e | 0.000800 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | 0.053500 | 0.324900 |
| 仅干净图像 | v11e | 0.054700 | 0.326500 |
| 仅干净音频 | v11e | 0.046000 | 0.285500 |
| 图像干净、音频残缺 | v11e | 0.053500 | 0.324200 |
| 音频干净、图像残缺 | v11e | 0.060300 | 0.345000 |
| 双模态残缺 | v11e | 0.060200 | 0.342500 |
| 仅音频残缺 | v11e | 0.041800 | 0.278800 |
| 仅图像残缺 | v11e | 0.059700 | 0.341000 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | v11e | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | v11e | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | 0.042900 | 0.154500 | 1.000000 |
| 仅干净图像 | v11e | 0.015200 | 0.068400 | 1.000000 |
| 仅干净音频 | v11e | 0.042800 | 0.154400 | 1.000000 |
| 图像干净、音频残缺 | v11e | 0.040300 | 0.146800 | 1.000000 |
| 音频干净、图像残缺 | v11e | 0.043000 | 0.154600 | 1.000000 |
| 双模态残缺 | v11e | 0.040400 | 0.146700 | 1.000000 |
| 仅音频残缺 | v11e | 0.039600 | 0.145600 | 1.000000 |
| 仅图像残缺 | v11e | 0.014700 | 0.063400 | 1.000000 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 仅干净图像 | v11e | 0.015600 | 0.070500 | 0.987900 | 82.700% |
| 仅干净音频 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.400% |
| 图像干净、音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 78.500% |
| 音频干净、图像残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 双模态残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 78.500% |
| 仅音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 78.200% |
| 仅图像残缺 | v11e | 0.015600 | 0.070500 | 0.987900 | 79.500% |

</details>


<details>
<summary>random 单次评估：各字段有效样本数</summary>

| 实验 | 指标字段（同一计数口径） | n |
| --- | --- | ---: |
| v11e | `acc`、`aud_masked_mse`、`aud_mse` | 未逐指标保存 |
| v11e | `aud_ssim`、`aud_visible_mse`、`img_coarse_masked_mse` | 未逐指标保存 |
| v11e | `img_coarse_visible_mse`、`img_masked_mse`、`img_mse` | 未逐指标保存 |
| v11e | `img_visible_mse`、`pair_l2`、`pix_var` | 未逐指标保存 |
| v11e | `psnr`、`rec_max`、`rec_mean` | 未逐指标保存 |
| v11e | `rec_std`、`ssim`、`tgt_max` | 未逐指标保存 |
| v11e | `tgt_mean`、`tgt_std`、`top15_recall` | 未逐指标保存 |
| v11e | `pair_a2i_r1`、`pair_i2a_r1` | 0 |

</details>

**Cross-Key / Cross-Detail 干预**

同一 cue/mask 内，gain=zero−normal，wrong damage=wrong−normal，same damage 为有效同类替换上的配对差；正 gain 表示改善。gate 非零本身不代表恢复有效。


<details>
<summary>fixed 五类等权宏平均：干预完整指标</summary>

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A | N/A |
| 仅干净音频 | v11e | 0.022800 | 0.076600 | -0.001700 |
| 图像干净、音频残缺 | v11e | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.001640 | 0.002100 | -0.000020 |
| 双模态残缺 | v11e | 0.001260 | 0.001520 | -0.000020 |
| 仅音频残缺 | v11e | 0.020900 | 0.059400 | -0.003720 |
| 仅图像残缺 | v11e | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | v11e | 0.279400 | 0.963500 |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.261400 | 0.887240 |
| 双模态残缺 | v11e | 0.312280 | 0.757560 |
| 仅音频残缺 | v11e | 0.331920 | 0.882760 |
| 仅图像残缺 | v11e | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | 0.000200 | 0.000500 | -0.000100 |
| 仅干净音频 | v11e | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.000200 | 0.000260 | 0.000000 |
| 音频干净、图像残缺 | v11e | N/A | N/A | N/A |
| 双模态残缺 | v11e | 0.000180 | 0.000220 | 0.000000 |
| 仅音频残缺 | v11e | N/A | N/A | N/A |
| 仅图像残缺 | v11e | 0.000340 | 0.000620 | -0.000140 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | 0.209200 | 0.307800 |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.170800 | 0.317840 |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | v11e | 0.187640 | 0.281680 |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | v11e | 0.269800 | 0.370480 |

</details>


<details>
<summary>fixed 五类等权宏平均：干预各字段有效 n</summary>

| 实验 | 指标字段（同一计数口径） | 每 family 的 n |
| --- | --- | ---: |
| v11e | `aud2img_correct_gain`、`aud2img_gate`、`aud2img_ratio` | 未逐指标保存 |
| v11e | `aud2img_same_damage`、`aud2img_wrong_damage`、`img2aud_correct_gain` | 未逐指标保存 |
| v11e | `img2aud_gate`、`img2aud_ratio`、`img2aud_same_damage` | 未逐指标保存 |
| v11e | `img2aud_wrong_damage` | 未逐指标保存 |

</details>

- random：未找到干预结果。

**逐 family 完整分项**

以下是与主表同源的五组 image/audio family pair，每组内仍按输入模式、实验排列。每组约 10000 个 MNIST 测试条目；音频会复用，旧版有效 n 未逐指标保存。


<details>
<summary>family 1：occlusion/time_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.200% | 0.009700 | 0.943000 | 0.003600 | 0.911000 |
| 干净双模态 | v11e | 99.500% | 0.012900 | 0.922000 | 0.003700 | 0.911000 |
| 仅干净图像 | control | 97.700% | 0.009800 | 0.943000 | 0.000300 | 0.966000 |
| 仅干净图像 | v11e | 97.400% | 0.013200 | 0.921000 | 0.000300 | 0.968000 |
| 仅干净音频 | control | 98.800% | 0.002100 | 0.986000 | 0.003600 | 0.911000 |
| 仅干净音频 | v11e | 96.700% | 0.003100 | 0.976000 | 0.003700 | 0.911000 |
| 图像干净、音频残缺 | control | 99.100% | 0.009800 | 0.943000 | 0.006200 | 0.835000 |
| 图像干净、音频残缺 | v11e | 99.000% | 0.012900 | 0.922000 | 0.005900 | 0.841000 |
| 音频干净、图像残缺 | control | 99.100% | 0.008800 | 0.948000 | 0.003600 | 0.911000 |
| 音频干净、图像残缺 | v11e | 99.000% | 0.008700 | 0.948000 | 0.003700 | 0.911000 |
| 双模态残缺 | control | 97.900% | 0.008700 | 0.948000 | 0.006400 | 0.828000 |
| 双模态残缺 | v11e | 97.400% | 0.008800 | 0.947000 | 0.006200 | 0.834000 |
| 仅音频残缺 | control | 90.500% | 0.009000 | 0.925000 | 0.006300 | 0.832000 |
| 仅音频残缺 | v11e | 89.400% | 0.009300 | 0.924000 | 0.006100 | 0.836000 |
| 仅图像残缺 | control | 89.800% | 0.009200 | 0.946000 | 0.000800 | 0.900000 |
| 仅图像残缺 | v11e | 89.600% | 0.009600 | 0.942000 | 0.000800 | 0.903000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 20.790000 | N/A | N/A |
| 干净双模态 | v11e | 19.600000 | N/A | N/A |
| 仅干净图像 | control | 20.770000 | N/A | N/A |
| 仅干净图像 | v11e | 19.480000 | N/A | N/A |
| 仅干净音频 | control | 37.620000 | 0.002100 | 0.002100 |
| 仅干净音频 | v11e | 39.510000 | 0.003100 | 0.003100 |
| 图像干净、音频残缺 | control | 20.780000 | N/A | N/A |
| 图像干净、音频残缺 | v11e | 19.580000 | N/A | N/A |
| 音频干净、图像残缺 | control | 23.030000 | 0.098600 | 0.057000 |
| 音频干净、图像残缺 | v11e | 23.210000 | 0.086900 | 0.056500 |
| 双模态残缺 | control | 22.970000 | 0.098200 | 0.056600 |
| 双模态残缺 | v11e | 23.110000 | 0.087800 | 0.057200 |
| 仅音频残缺 | control | 32.890000 | 0.009000 | 0.009000 |
| 仅音频残缺 | v11e | 34.250000 | 0.009300 | 0.009300 |
| 仅图像残缺 | control | 22.870000 | 0.102800 | 0.059300 |
| 仅图像残缺 | v11e | 22.800000 | 0.095500 | 0.062400 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | 0.015300 | 0.000000 |
| 音频干净、图像残缺 | v11e | 0.016200 | 0.000000 |
| 双模态残缺 | control | 0.015200 | 0.000000 |
| 双模态残缺 | v11e | 0.016200 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | 0.015600 | 0.000000 |
| 仅图像残缺 | v11e | 0.016800 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | 0.000300 | N/A |
| 仅干净图像 | v11e | 0.000300 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | 0.009100 | 0.004200 |
| 图像干净、音频残缺 | v11e | 0.008600 | 0.004100 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | 0.009600 | 0.004200 |
| 双模态残缺 | v11e | 0.009200 | 0.004100 |
| 仅音频残缺 | control | 0.009300 | 0.004200 |
| 仅音频残缺 | v11e | 0.009000 | 0.004100 |
| 仅图像残缺 | control | 0.000800 | N/A |
| 仅图像残缺 | v11e | 0.000800 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.058900 | 0.339400 |
| 干净双模态 | v11e | 0.053500 | 0.323400 |
| 仅干净图像 | control | 0.059400 | 0.338900 |
| 仅干净图像 | v11e | 0.054700 | 0.324800 |
| 仅干净音频 | control | 0.046300 | 0.287100 |
| 仅干净音频 | v11e | 0.046000 | 0.286100 |
| 图像干净、音频残缺 | control | 0.059200 | 0.339700 |
| 图像干净、音频残缺 | v11e | 0.053500 | 0.322900 |
| 音频干净、图像残缺 | control | 0.060200 | 0.343600 |
| 音频干净、图像残缺 | v11e | 0.060300 | 0.343300 |
| 双模态残缺 | control | 0.060200 | 0.341800 |
| 双模态残缺 | v11e | 0.060200 | 0.341700 |
| 仅音频残缺 | control | 0.042500 | 0.283300 |
| 仅音频残缺 | v11e | 0.042000 | 0.280400 |
| 仅图像残缺 | control | 0.059800 | 0.342300 |
| 仅图像残缺 | v11e | 0.059700 | 0.341800 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11e | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11e | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11e | 0.042900 | 0.154500 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11e | 0.015200 | 0.068400 | 1.000000 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11e | 0.042800 | 0.154400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.040000 | 0.147100 | 1.000000 |
| 图像干净、音频残缺 | v11e | 0.040300 | 0.146900 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11e | 0.043000 | 0.154700 | 1.000000 |
| 双模态残缺 | control | 0.039900 | 0.146800 | 1.000000 |
| 双模态残缺 | v11e | 0.040300 | 0.146500 | 1.000000 |
| 仅音频残缺 | control | 0.039800 | 0.146600 | 1.000000 |
| 仅音频残缺 | v11e | 0.039800 | 0.145900 | 1.000000 |
| 仅图像残缺 | control | 0.015000 | 0.065400 | 1.000000 |
| 仅图像残缺 | v11e | 0.014700 | 0.063600 | 1.000000 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11e | 0.015600 | 0.070500 | 0.987900 | 82.700% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.400% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.100% |
| 图像干净、音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 78.700% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 77.800% |
| 双模态残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 77.900% |
| 仅音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 78.300% |
| 仅图像残缺 | v11e | 0.015600 | 0.070500 | 0.987900 | 79.500% |

**本 family 干预**

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A | N/A |
| 仅干净音频 | v11e | 0.022800 | 0.076600 | -0.001700 |
| 图像干净、音频残缺 | v11e | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.004800 | 0.005300 | -0.000100 |
| 双模态残缺 | v11e | 0.004200 | 0.004400 | -0.000100 |
| 仅音频残缺 | v11e | 0.021700 | 0.064400 | -0.004800 |
| 仅图像残缺 | v11e | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | v11e | 0.279400 | 0.963500 |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.260700 | 0.878100 |
| 双模态残缺 | v11e | 0.300700 | 0.789700 |
| 仅音频残缺 | v11e | 0.319900 | 0.905100 |
| 仅图像残缺 | v11e | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | 0.000200 | 0.000500 | -0.000100 |
| 仅干净音频 | v11e | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.000200 | 0.000200 | 0.000000 |
| 音频干净、图像残缺 | v11e | N/A | N/A | N/A |
| 双模态残缺 | v11e | 0.000200 | 0.000200 | 0.000000 |
| 仅音频残缺 | v11e | N/A | N/A | N/A |
| 仅图像残缺 | v11e | 0.000300 | 0.000600 | -0.000200 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | 0.209200 | 0.307800 |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.158600 | 0.286800 |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | v11e | 0.172800 | 0.258900 |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | v11e | 0.258300 | 0.352500 |

</details>


<details>
<summary>family 2：pixel_delete/freq_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.200% | 0.009700 | 0.943000 | 0.003600 | 0.911000 |
| 干净双模态 | v11e | 99.500% | 0.012900 | 0.922000 | 0.003700 | 0.911000 |
| 仅干净图像 | control | 97.700% | 0.009800 | 0.943000 | 0.000300 | 0.966000 |
| 仅干净图像 | v11e | 97.400% | 0.013200 | 0.921000 | 0.000300 | 0.968000 |
| 仅干净音频 | control | 98.800% | 0.002100 | 0.986000 | 0.003600 | 0.911000 |
| 仅干净音频 | v11e | 96.700% | 0.003100 | 0.976000 | 0.003700 | 0.911000 |
| 图像干净、音频残缺 | control | 99.000% | 0.009800 | 0.943000 | 0.004800 | 0.860000 |
| 图像干净、音频残缺 | v11e | 98.500% | 0.012900 | 0.922000 | 0.004700 | 0.870000 |
| 音频干净、图像残缺 | control | 99.200% | 0.004400 | 0.974000 | 0.003600 | 0.911000 |
| 音频干净、图像残缺 | v11e | 99.100% | 0.004000 | 0.977000 | 0.003700 | 0.911000 |
| 双模态残缺 | control | 98.100% | 0.004400 | 0.974000 | 0.004800 | 0.860000 |
| 双模态残缺 | v11e | 97.600% | 0.004100 | 0.976000 | 0.004800 | 0.869000 |
| 仅音频残缺 | control | 92.500% | 0.007800 | 0.941000 | 0.004900 | 0.858000 |
| 仅音频残缺 | v11e | 91.800% | 0.008100 | 0.935000 | 0.004800 | 0.866000 |
| 仅图像残缺 | control | 95.000% | 0.004500 | 0.974000 | 0.000500 | 0.941000 |
| 仅图像残缺 | v11e | 95.000% | 0.004200 | 0.976000 | 0.000500 | 0.943000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 20.790000 | N/A | N/A |
| 干净双模态 | v11e | 19.600000 | N/A | N/A |
| 仅干净图像 | control | 20.770000 | N/A | N/A |
| 仅干净图像 | v11e | 19.480000 | N/A | N/A |
| 仅干净音频 | control | 37.620000 | 0.002100 | 0.002100 |
| 仅干净音频 | v11e | 39.510000 | 0.003100 | 0.003100 |
| 图像干净、音频残缺 | control | 20.780000 | N/A | N/A |
| 图像干净、音频残缺 | v11e | 19.570000 | N/A | N/A |
| 音频干净、图像残缺 | control | 24.100000 | 0.036400 | 0.011100 |
| 音频干净、图像残缺 | v11e | 24.540000 | 0.028500 | 0.010100 |
| 双模态残缺 | control | 24.090000 | 0.036300 | 0.011100 |
| 双模态残缺 | v11e | 24.530000 | 0.028500 | 0.010100 |
| 仅音频残缺 | control | 31.760000 | 0.007800 | 0.007800 |
| 仅音频残缺 | v11e | 32.220000 | 0.008100 | 0.008100 |
| 仅图像残缺 | control | 24.080000 | 0.037200 | 0.011100 |
| 仅图像残缺 | v11e | 24.410000 | 0.029600 | 0.010400 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | 0.028100 | 0.000000 |
| 音频干净、图像残缺 | v11e | 0.022700 | 0.000000 |
| 双模态残缺 | control | 0.028000 | 0.000000 |
| 双模态残缺 | v11e | 0.022700 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | 0.028800 | 0.000000 |
| 仅图像残缺 | v11e | 0.023400 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | 0.000300 | N/A |
| 仅干净图像 | v11e | 0.000300 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | 0.006300 | 0.003800 |
| 图像干净、音频残缺 | v11e | 0.006000 | 0.003800 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | 0.006400 | 0.003800 |
| 双模态残缺 | v11e | 0.006100 | 0.003800 |
| 仅音频残缺 | control | 0.006400 | 0.003800 |
| 仅音频残缺 | v11e | 0.006200 | 0.003800 |
| 仅图像残缺 | control | 0.000500 | N/A |
| 仅图像残缺 | v11e | 0.000500 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.058900 | 0.339200 |
| 干净双模态 | v11e | 0.053500 | 0.323100 |
| 仅干净图像 | control | 0.059400 | 0.339900 |
| 仅干净图像 | v11e | 0.054700 | 0.325800 |
| 仅干净音频 | control | 0.046300 | 0.289800 |
| 仅干净音频 | v11e | 0.046000 | 0.288700 |
| 图像干净、音频残缺 | control | 0.059200 | 0.340300 |
| 图像干净、音频残缺 | v11e | 0.053500 | 0.323200 |
| 音频干净、图像残缺 | control | 0.064100 | 0.353400 |
| 音频干净、图像残缺 | v11e | 0.064700 | 0.355000 |
| 双模态残缺 | control | 0.064100 | 0.354000 |
| 双模态残缺 | v11e | 0.064700 | 0.355500 |
| 仅音频残缺 | control | 0.043300 | 0.281700 |
| 仅音频残缺 | v11e | 0.042200 | 0.277800 |
| 仅图像残缺 | control | 0.064000 | 0.353100 |
| 仅图像残缺 | v11e | 0.064500 | 0.354500 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11e | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11e | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11e | 0.042900 | 0.154500 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11e | 0.015200 | 0.068400 | 1.000000 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11e | 0.042800 | 0.154400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.041700 | 0.151500 | 1.000000 |
| 图像干净、音频残缺 | v11e | 0.042200 | 0.151700 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11e | 0.043000 | 0.154600 | 1.000000 |
| 双模态残缺 | control | 0.041900 | 0.151700 | 1.000000 |
| 双模态残缺 | v11e | 0.042300 | 0.151900 | 1.000000 |
| 仅音频残缺 | control | 0.041700 | 0.151400 | 1.000000 |
| 仅音频残缺 | v11e | 0.042000 | 0.151300 | 1.000000 |
| 仅图像残缺 | control | 0.015500 | 0.067600 | 1.000000 |
| 仅图像残缺 | v11e | 0.015300 | 0.066400 | 1.000000 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11e | 0.015600 | 0.070500 | 0.987900 | 82.700% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.400% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 图像干净、音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 80.600% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 双模态残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 80.600% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.000% |
| 仅音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 80.400% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 79.700% |
| 仅图像残缺 | v11e | 0.015600 | 0.070500 | 0.987900 | 81.200% |

**本 family 干预**

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A | N/A |
| 仅干净音频 | v11e | 0.022800 | 0.076600 | -0.001700 |
| 图像干净、音频残缺 | v11e | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.000300 | 0.000400 | 0.000000 |
| 双模态残缺 | v11e | 0.000200 | 0.000300 | 0.000000 |
| 仅音频残缺 | v11e | 0.023100 | 0.063800 | -0.003900 |
| 仅图像残缺 | v11e | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | v11e | 0.279400 | 0.963500 |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.261900 | 0.891700 |
| 双模态残缺 | v11e | 0.303100 | 0.798700 |
| 仅音频残缺 | v11e | 0.323300 | 0.893200 |
| 仅图像残缺 | v11e | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | 0.000200 | 0.000500 | -0.000100 |
| 仅干净音频 | v11e | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.000100 | 0.000100 | 0.000000 |
| 音频干净、图像残缺 | v11e | N/A | N/A | N/A |
| 双模态残缺 | v11e | 0.000100 | 0.000100 | 0.000000 |
| 仅音频残缺 | v11e | N/A | N/A | N/A |
| 仅图像残缺 | v11e | 0.000300 | 0.000600 | -0.000100 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | 0.209200 | 0.307800 |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.156900 | 0.278600 |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | v11e | 0.172300 | 0.240200 |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | v11e | 0.264500 | 0.346900 |

</details>


<details>
<summary>family 3：mask_vertical/feature_dropout</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.200% | 0.009700 | 0.943000 | 0.003600 | 0.911000 |
| 干净双模态 | v11e | 99.500% | 0.012900 | 0.922000 | 0.003700 | 0.911000 |
| 仅干净图像 | control | 97.700% | 0.009800 | 0.943000 | 0.000300 | 0.966000 |
| 仅干净图像 | v11e | 97.400% | 0.013200 | 0.921000 | 0.000300 | 0.968000 |
| 仅干净音频 | control | 98.800% | 0.002100 | 0.986000 | 0.003600 | 0.911000 |
| 仅干净音频 | v11e | 96.700% | 0.003100 | 0.976000 | 0.003700 | 0.911000 |
| 图像干净、音频残缺 | control | 99.300% | 0.009700 | 0.943000 | 0.003700 | 0.904000 |
| 图像干净、音频残缺 | v11e | 99.300% | 0.012900 | 0.922000 | 0.003700 | 0.906000 |
| 音频干净、图像残缺 | control | 99.100% | 0.007700 | 0.957000 | 0.003600 | 0.911000 |
| 音频干净、图像残缺 | v11e | 99.200% | 0.007900 | 0.956000 | 0.003700 | 0.911000 |
| 双模态残缺 | control | 99.200% | 0.007600 | 0.958000 | 0.003700 | 0.904000 |
| 双模态残缺 | v11e | 99.000% | 0.007900 | 0.957000 | 0.003800 | 0.906000 |
| 仅音频残缺 | control | 97.200% | 0.003000 | 0.979000 | 0.003700 | 0.904000 |
| 仅音频残缺 | v11e | 96.100% | 0.003700 | 0.972000 | 0.003800 | 0.905000 |
| 仅图像残缺 | control | 94.500% | 0.007900 | 0.956000 | 0.000500 | 0.939000 |
| 仅图像残缺 | v11e | 94.400% | 0.008400 | 0.953000 | 0.000500 | 0.941000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 20.790000 | N/A | N/A |
| 干净双模态 | v11e | 19.600000 | N/A | N/A |
| 仅干净图像 | control | 20.770000 | N/A | N/A |
| 仅干净图像 | v11e | 19.480000 | N/A | N/A |
| 仅干净音频 | control | 37.620000 | 0.002100 | 0.002100 |
| 仅干净音频 | v11e | 39.510000 | 0.003100 | 0.003100 |
| 图像干净、音频残缺 | control | 20.790000 | N/A | N/A |
| 图像干净、音频残缺 | v11e | 19.590000 | N/A | N/A |
| 音频干净、图像残缺 | control | 24.350000 | 0.030800 | 0.019500 |
| 音频干净、图像残缺 | v11e | 24.430000 | 0.028800 | 0.020000 |
| 双模态残缺 | control | 24.380000 | 0.030700 | 0.019400 |
| 双模态残缺 | v11e | 24.440000 | 0.028900 | 0.020000 |
| 仅音频残缺 | control | 36.160000 | 0.003000 | 0.003000 |
| 仅音频残缺 | v11e | 37.700000 | 0.003700 | 0.003700 |
| 仅图像残缺 | control | 24.240000 | 0.031500 | 0.020000 |
| 仅图像残缺 | v11e | 24.110000 | 0.030600 | 0.021500 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | 0.019300 | 0.000000 |
| 音频干净、图像残缺 | v11e | 0.021800 | 0.000000 |
| 双模态残缺 | control | 0.019400 | 0.000000 |
| 双模态残缺 | v11e | 0.022000 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | 0.019600 | 0.000000 |
| 仅图像残缺 | v11e | 0.022300 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | 0.000300 | N/A |
| 仅干净图像 | v11e | 0.000300 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | 0.003800 | 0.003700 |
| 图像干净、音频残缺 | v11e | 0.003800 | 0.003700 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | 0.003800 | 0.003700 |
| 双模态残缺 | v11e | 0.003800 | 0.003700 |
| 仅音频残缺 | control | 0.003800 | 0.003700 |
| 仅音频残缺 | v11e | 0.003800 | 0.003700 |
| 仅图像残缺 | control | 0.000500 | N/A |
| 仅图像残缺 | v11e | 0.000500 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.058900 | 0.337400 |
| 干净双模态 | v11e | 0.053500 | 0.321000 |
| 仅干净图像 | control | 0.059400 | 0.341100 |
| 仅干净图像 | v11e | 0.054700 | 0.327000 |
| 仅干净音频 | control | 0.046300 | 0.286000 |
| 仅干净音频 | v11e | 0.046000 | 0.285300 |
| 图像干净、音频残缺 | control | 0.059000 | 0.337200 |
| 图像干净、音频残缺 | v11e | 0.053500 | 0.321300 |
| 音频干净、图像残缺 | control | 0.060400 | 0.342800 |
| 音频干净、图像残缺 | v11e | 0.060500 | 0.343000 |
| 双模态残缺 | control | 0.060400 | 0.343400 |
| 双模态残缺 | v11e | 0.060400 | 0.343400 |
| 仅音频残缺 | control | 0.045500 | 0.283400 |
| 仅音频残缺 | v11e | 0.045200 | 0.281300 |
| 仅图像残缺 | control | 0.060200 | 0.342200 |
| 仅图像残缺 | v11e | 0.060000 | 0.341800 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11e | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11e | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11e | 0.042900 | 0.154500 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11e | 0.015200 | 0.068400 | 1.000000 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11e | 0.042800 | 0.154400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.042100 | 0.153900 | 1.000000 |
| 图像干净、音频残缺 | v11e | 0.041900 | 0.152700 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155600 | 1.000000 |
| 音频干净、图像残缺 | v11e | 0.043000 | 0.154600 | 1.000000 |
| 双模态残缺 | control | 0.042100 | 0.153900 | 1.000000 |
| 双模态残缺 | v11e | 0.042000 | 0.152900 | 1.000000 |
| 仅音频残缺 | control | 0.042100 | 0.153900 | 1.000000 |
| 仅音频残缺 | v11e | 0.041800 | 0.152600 | 1.000000 |
| 仅图像残缺 | control | 0.015300 | 0.067500 | 1.000000 |
| 仅图像残缺 | v11e | 0.015300 | 0.067000 | 1.000000 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11e | 0.015600 | 0.070500 | 0.987900 | 82.700% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.400% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 图像干净、音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.000% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 双模态残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 仅音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 81.800% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 79.800% |
| 仅图像残缺 | v11e | 0.015600 | 0.070500 | 0.987900 | 81.100% |

**本 family 干预**

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A | N/A |
| 仅干净音频 | v11e | 0.022800 | 0.076600 | -0.001700 |
| 图像干净、音频残缺 | v11e | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.001200 | 0.001900 | 0.000000 |
| 双模态残缺 | v11e | 0.001200 | 0.001800 | 0.000000 |
| 仅音频残缺 | v11e | 0.023600 | 0.074200 | -0.002000 |
| 仅图像残缺 | v11e | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | v11e | 0.279400 | 0.963500 |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.261300 | 0.872500 |
| 双模态残缺 | v11e | 0.282800 | 0.845500 |
| 仅音频残缺 | v11e | 0.302000 | 0.950000 |
| 仅图像残缺 | v11e | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | 0.000200 | 0.000500 | -0.000100 |
| 仅干净音频 | v11e | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11e | N/A | N/A | N/A |
| 双模态残缺 | v11e | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11e | N/A | N/A | N/A |
| 仅图像残缺 | v11e | 0.000300 | 0.000500 | -0.000100 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | 0.209200 | 0.307800 |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.141600 | 0.248600 |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | v11e | 0.150100 | 0.226100 |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | v11e | 0.244500 | 0.333900 |

</details>


<details>
<summary>family 4：mask_horizontal/partial_temporal</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.200% | 0.009700 | 0.943000 | 0.003600 | 0.911000 |
| 干净双模态 | v11e | 99.500% | 0.012900 | 0.922000 | 0.003700 | 0.911000 |
| 仅干净图像 | control | 97.700% | 0.009800 | 0.943000 | 0.000300 | 0.966000 |
| 仅干净图像 | v11e | 97.400% | 0.013200 | 0.921000 | 0.000300 | 0.968000 |
| 仅干净音频 | control | 98.800% | 0.002100 | 0.986000 | 0.003600 | 0.911000 |
| 仅干净音频 | v11e | 96.700% | 0.003100 | 0.976000 | 0.003700 | 0.911000 |
| 图像干净、音频残缺 | control | 98.100% | 0.009900 | 0.942000 | 0.011800 | 0.538000 |
| 图像干净、音频残缺 | v11e | 97.600% | 0.013200 | 0.921000 | 0.011300 | 0.572000 |
| 音频干净、图像残缺 | control | 99.200% | 0.011400 | 0.934000 | 0.003600 | 0.911000 |
| 音频干净、图像残缺 | v11e | 99.100% | 0.011300 | 0.935000 | 0.003700 | 0.911000 |
| 双模态残缺 | control | 94.000% | 0.011500 | 0.933000 | 0.011900 | 0.537000 |
| 双模态残缺 | v11e | 93.600% | 0.011800 | 0.931000 | 0.011500 | 0.569000 |
| 仅音频残缺 | control | 45.900% | 0.042700 | 0.569000 | 0.012600 | 0.529000 |
| 仅音频残缺 | v11e | 45.500% | 0.036800 | 0.679000 | 0.012100 | 0.548000 |
| 仅图像残缺 | control | 93.000% | 0.011600 | 0.932000 | 0.000600 | 0.921000 |
| 仅图像残缺 | v11e | 92.900% | 0.012100 | 0.930000 | 0.000600 | 0.926000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 20.790000 | N/A | N/A |
| 干净双模态 | v11e | 19.600000 | N/A | N/A |
| 仅干净图像 | control | 20.770000 | N/A | N/A |
| 仅干净图像 | v11e | 19.480000 | N/A | N/A |
| 仅干净音频 | control | 37.620000 | 0.002100 | 0.002100 |
| 仅干净音频 | v11e | 39.510000 | 0.003100 | 0.003100 |
| 图像干净、音频残缺 | control | 20.700000 | N/A | N/A |
| 图像干净、音频残缺 | v11e | 19.490000 | N/A | N/A |
| 音频干净、图像残缺 | control | 20.220000 | 0.041900 | 0.029000 |
| 音频干净、图像残缺 | v11e | 20.320000 | 0.037600 | 0.028700 |
| 双模态残缺 | control | 20.170000 | 0.041700 | 0.029400 |
| 双模态残缺 | v11e | 20.100000 | 0.038800 | 0.030100 |
| 仅音频残缺 | control | 17.520000 | 0.042700 | 0.042700 |
| 仅音频残缺 | v11e | 18.200000 | 0.036800 | 0.036800 |
| 仅图像残缺 | control | 20.120000 | 0.042800 | 0.029600 |
| 仅图像残缺 | v11e | 20.020000 | 0.040300 | 0.030700 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | 0.024600 | 0.000000 |
| 音频干净、图像残缺 | v11e | 0.022900 | 0.000000 |
| 双模态残缺 | control | 0.023800 | 0.000000 |
| 双模态残缺 | v11e | 0.023200 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | 0.025100 | 0.000000 |
| 仅图像残缺 | v11e | 0.023700 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | 0.000300 | N/A |
| 仅干净图像 | v11e | 0.000300 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | 0.028600 | 0.000300 |
| 图像干净、音频残缺 | v11e | 0.027500 | 0.000300 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | 0.029000 | 0.000300 |
| 双模态残缺 | v11e | 0.027800 | 0.000300 |
| 仅音频残缺 | control | 0.030500 | 0.000300 |
| 仅音频残缺 | v11e | 0.029400 | 0.000300 |
| 仅图像残缺 | control | 0.000600 | N/A |
| 仅图像残缺 | v11e | 0.000600 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.058900 | 0.339400 |
| 干净双模态 | v11e | 0.053500 | 0.322700 |
| 仅干净图像 | control | 0.059400 | 0.340100 |
| 仅干净图像 | v11e | 0.054700 | 0.326300 |
| 仅干净音频 | control | 0.046300 | 0.291400 |
| 仅干净音频 | v11e | 0.046000 | 0.290200 |
| 图像干净、音频残缺 | control | 0.060300 | 0.342500 |
| 图像干净、音频残缺 | v11e | 0.053900 | 0.324000 |
| 音频干净、图像残缺 | control | 0.057500 | 0.335900 |
| 音频干净、图像残缺 | v11e | 0.057500 | 0.336000 |
| 双模态残缺 | control | 0.057600 | 0.336200 |
| 双模态残缺 | v11e | 0.057300 | 0.335500 |
| 仅音频残缺 | control | 0.023300 | 0.188500 |
| 仅音频残缺 | v11e | 0.020500 | 0.176500 |
| 仅图像残缺 | control | 0.057100 | 0.334800 |
| 仅图像残缺 | v11e | 0.057100 | 0.334800 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11e | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11e | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11e | 0.042900 | 0.154500 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11e | 0.015200 | 0.068400 | 1.000000 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11e | 0.042800 | 0.154400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.037500 | 0.131100 | 1.000000 |
| 图像干净、音频残缺 | v11e | 0.035000 | 0.126000 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11e | 0.043000 | 0.154600 | 1.000000 |
| 双模态残缺 | control | 0.037300 | 0.130500 | 1.000000 |
| 双模态残缺 | v11e | 0.035200 | 0.125700 | 1.000000 |
| 仅音频残缺 | control | 0.035400 | 0.126800 | 1.000000 |
| 仅音频残缺 | v11e | 0.033000 | 0.121700 | 1.000000 |
| 仅图像残缺 | control | 0.015400 | 0.067400 | 1.000000 |
| 仅图像残缺 | v11e | 0.015200 | 0.066000 | 0.999700 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11e | 0.015600 | 0.070500 | 0.987900 | 82.700% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.400% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 70.200% |
| 图像干净、音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 70.500% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 70.000% |
| 双模态残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 70.300% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 68.600% |
| 仅音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 68.800% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 79.000% |
| 仅图像残缺 | v11e | 0.015600 | 0.070500 | 0.987900 | 80.300% |

**本 family 干预**

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A | N/A |
| 仅干净音频 | v11e | 0.022800 | 0.076600 | -0.001700 |
| 图像干净、音频残缺 | v11e | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.001700 | 0.002500 | 0.000000 |
| 双模态残缺 | v11e | 0.000500 | 0.000700 | 0.000000 |
| 仅音频残缺 | v11e | 0.013100 | 0.021300 | -0.005500 |
| 仅图像残缺 | v11e | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | v11e | 0.279400 | 0.963500 |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.261200 | 0.881200 |
| 双模态残缺 | v11e | 0.402700 | 0.460600 |
| 仅音频残缺 | v11e | 0.423800 | 0.717200 |
| 仅图像残缺 | v11e | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | 0.000200 | 0.000500 | -0.000100 |
| 仅干净音频 | v11e | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.000700 | 0.000900 | 0.000000 |
| 音频干净、图像残缺 | v11e | N/A | N/A | N/A |
| 双模态残缺 | v11e | 0.000600 | 0.000800 | 0.000000 |
| 仅音频残缺 | v11e | N/A | N/A | N/A |
| 仅图像残缺 | v11e | 0.000300 | 0.000600 | -0.000100 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | 0.209200 | 0.307800 |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.271200 | 0.565000 |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | v11e | 0.302600 | 0.522900 |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | v11e | 0.261300 | 0.348500 |

</details>


<details>
<summary>family 5：salt_mask/time_freq_block</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.200% | 0.009700 | 0.943000 | 0.003600 | 0.911000 |
| 干净双模态 | v11e | 99.500% | 0.012900 | 0.922000 | 0.003700 | 0.911000 |
| 仅干净图像 | control | 97.700% | 0.009800 | 0.943000 | 0.000300 | 0.966000 |
| 仅干净图像 | v11e | 97.400% | 0.013200 | 0.921000 | 0.000300 | 0.968000 |
| 仅干净音频 | control | 98.800% | 0.002100 | 0.986000 | 0.003600 | 0.911000 |
| 仅干净音频 | v11e | 96.700% | 0.003100 | 0.976000 | 0.003700 | 0.911000 |
| 图像干净、音频残缺 | control | 99.200% | 0.009700 | 0.943000 | 0.004000 | 0.900000 |
| 图像干净、音频残缺 | v11e | 99.200% | 0.012900 | 0.922000 | 0.004000 | 0.901000 |
| 音频干净、图像残缺 | control | 99.100% | 0.005200 | 0.968000 | 0.003700 | 0.911000 |
| 音频干净、图像残缺 | v11e | 98.400% | 0.004900 | 0.970000 | 0.003700 | 0.911000 |
| 双模态残缺 | control | 98.600% | 0.005300 | 0.968000 | 0.004000 | 0.900000 |
| 双模态残缺 | v11e | 97.900% | 0.004900 | 0.970000 | 0.004000 | 0.901000 |
| 仅音频残缺 | control | 96.800% | 0.003600 | 0.974000 | 0.004000 | 0.900000 |
| 仅音频残缺 | v11e | 95.300% | 0.004400 | 0.966000 | 0.004000 | 0.901000 |
| 仅图像残缺 | control | 87.800% | 0.005300 | 0.968000 | 0.000900 | 0.883000 |
| 仅图像残缺 | v11e | 88.200% | 0.005000 | 0.970000 | 0.001000 | 0.881000 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 20.790000 | N/A | N/A |
| 干净双模态 | v11e | 19.600000 | N/A | N/A |
| 仅干净图像 | control | 20.770000 | N/A | N/A |
| 仅干净图像 | v11e | 19.480000 | N/A | N/A |
| 仅干净音频 | control | 37.620000 | 0.002100 | 0.002100 |
| 仅干净音频 | v11e | 39.510000 | 0.003100 | 0.003100 |
| 图像干净、音频残缺 | control | 20.790000 | N/A | N/A |
| 图像干净、音频残缺 | v11e | 19.590000 | N/A | N/A |
| 音频干净、图像残缺 | control | 23.270000 | 0.066400 | 0.013100 |
| 音频干净、图像残缺 | v11e | 23.580000 | 0.065300 | 0.012200 |
| 双模态残缺 | control | 23.270000 | 0.066500 | 0.013100 |
| 双模态残缺 | v11e | 23.580000 | 0.065300 | 0.012200 |
| 仅音频残缺 | control | 36.130000 | 0.003600 | 0.003600 |
| 仅音频残缺 | v11e | 37.680000 | 0.004400 | 0.004400 |
| 仅图像残缺 | control | 23.220000 | 0.067400 | 0.013200 |
| 仅图像残缺 | v11e | 23.470000 | 0.067500 | 0.012500 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | 0.066300 | 0.000000 |
| 音频干净、图像残缺 | v11e | 0.065100 | 0.000000 |
| 双模态残缺 | control | 0.066400 | 0.000000 |
| 双模态残缺 | v11e | 0.065200 | 0.000000 |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | 0.067200 | 0.000000 |
| 仅图像残缺 | v11e | 0.067300 | 0.000000 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频可见区 MSE |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | 0.000300 | N/A |
| 仅干净图像 | v11e | 0.000300 | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | 0.005300 | 0.003700 |
| 图像干净、音频残缺 | v11e | 0.005100 | 0.003800 |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | 0.005100 | 0.003800 |
| 双模态残缺 | v11e | 0.005100 | 0.003800 |
| 仅音频残缺 | control | 0.005300 | 0.003700 |
| 仅音频残缺 | v11e | 0.005200 | 0.003800 |
| 仅图像残缺 | control | 0.000900 | N/A |
| 仅图像残缺 | v11e | 0.001000 | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: |
| 干净双模态 | control | 0.058900 | 0.338100 |
| 干净双模态 | v11e | 0.053500 | 0.321900 |
| 仅干净图像 | control | 0.059400 | 0.340200 |
| 仅干净图像 | v11e | 0.054700 | 0.326300 |
| 仅干净音频 | control | 0.046300 | 0.283500 |
| 仅干净音频 | v11e | 0.046000 | 0.282500 |
| 图像干净、音频残缺 | control | 0.058900 | 0.337700 |
| 图像干净、音频残缺 | v11e | 0.053500 | 0.321200 |
| 音频干净、图像残缺 | control | 0.063100 | 0.350300 |
| 音频干净、图像残缺 | v11e | 0.063200 | 0.350700 |
| 双模态残缺 | control | 0.063000 | 0.352900 |
| 双模态残缺 | v11e | 0.063200 | 0.353300 |
| 仅音频残缺 | control | 0.045600 | 0.288500 |
| 仅音频残缺 | v11e | 0.045100 | 0.286700 |
| 仅图像残缺 | control | 0.063000 | 0.350000 |
| 仅图像残缺 | v11e | 0.062900 | 0.349600 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11e | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11e | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11e | 0.042900 | 0.154500 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11e | 0.015200 | 0.068400 | 1.000000 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11e | 0.042800 | 0.154400 | 1.000000 |
| 图像干净、音频残缺 | control | 0.042700 | 0.154400 | 1.000000 |
| 图像干净、音频残缺 | v11e | 0.042700 | 0.153700 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11e | 0.043000 | 0.154700 | 1.000000 |
| 双模态残缺 | control | 0.042800 | 0.154600 | 1.000000 |
| 双模态残缺 | v11e | 0.042900 | 0.154100 | 1.000000 |
| 仅音频残缺 | control | 0.042700 | 0.154400 | 1.000000 |
| 仅音频残缺 | v11e | 0.042600 | 0.153600 | 1.000000 |
| 仅图像残缺 | control | 0.015200 | 0.063500 | 0.999500 |
| 仅图像残缺 | v11e | 0.014100 | 0.059300 | 0.997900 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11e | 0.015600 | 0.070500 | 0.987900 | 82.700% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.400% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 图像干净、音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 82.500% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 双模态残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 81.900% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.500% |
| 仅音频残缺 | v11e | 0.040600 | 0.152100 | 1.000000 | 81.800% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 76.700% |
| 仅图像残缺 | v11e | 0.015600 | 0.070500 | 0.987900 | 78.500% |

**本 family 干预**

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A | N/A |
| 仅干净音频 | v11e | 0.022800 | 0.076600 | -0.001700 |
| 图像干净、音频残缺 | v11e | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.000200 | 0.000400 | 0.000000 |
| 双模态残缺 | v11e | 0.000200 | 0.000400 | 0.000000 |
| 仅音频残缺 | v11e | 0.023000 | 0.073300 | -0.002400 |
| 仅图像残缺 | v11e | N/A | N/A | N/A |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | N/A | N/A |
| 仅干净音频 | v11e | 0.279400 | 0.963500 |
| 图像干净、音频残缺 | v11e | N/A | N/A |
| 音频干净、图像残缺 | v11e | 0.261900 | 0.912700 |
| 双模态残缺 | v11e | 0.272100 | 0.893300 |
| 仅音频残缺 | v11e | 0.290600 | 0.948300 |
| 仅图像残缺 | v11e | N/A | N/A |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A | N/A |
| 仅干净图像 | v11e | 0.000200 | 0.000500 | -0.000100 |
| 仅干净音频 | v11e | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.000000 | 0.000100 | 0.000000 |
| 音频干净、图像残缺 | v11e | N/A | N/A | N/A |
| 双模态残缺 | v11e | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11e | N/A | N/A | N/A |
| 仅图像残缺 | v11e | 0.000500 | 0.000800 | -0.000200 |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 干净双模态 | v11e | N/A | N/A |
| 仅干净图像 | v11e | 0.209200 | 0.307800 |
| 仅干净音频 | v11e | N/A | N/A |
| 图像干净、音频残缺 | v11e | 0.125700 | 0.210200 |
| 音频干净、图像残缺 | v11e | N/A | N/A |
| 双模态残缺 | v11e | 0.140400 | 0.160300 |
| 仅音频残缺 | v11e | N/A | N/A |
| 仅图像残缺 | v11e | 0.320400 | 0.470600 |

</details>

**独立音频 family breakdown**

此处来自各实验的 `tables/audio_family_breakdown_fixed.csv`，单独改变音频 family；不能用它替代上面的双 family-pair 主表或再次计入宏平均。该 CSV 不含逐指标 n。


<details>
<summary>time_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.030% | 0.009772 | 0.942626 | 0.006218 | 0.833523 |
| 图像干净、音频残缺 | v11e | 99.010% | 0.012934 | 0.921994 | 0.005997 | 0.840328 |
| 双模态残缺 | control | 97.710% | 0.008706 | 0.948581 | 0.006304 | 0.830981 |
| 双模态残缺 | v11e | 97.440% | 0.008753 | 0.948174 | 0.006068 | 0.838087 |
| 仅音频残缺 | control | 90.390% | 0.009307 | 0.923568 | 0.006326 | 0.830664 |
| 仅音频残缺 | v11e | 88.860% | 0.009578 | 0.921334 | 0.006171 | 0.834722 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.776195 | 0.009109 | 0.023938 |
| 图像干净、音频残缺 | v11e | 19.574593 | 0.008664 | 0.023296 |
| 双模态残缺 | control | 23.123210 | 0.009356 | 0.024483 |
| 双模态残缺 | v11e | 23.288974 | 0.008870 | 0.023821 |
| 仅音频残缺 | control | 32.712509 | 0.009393 | 0.024228 |
| 仅音频残缺 | v11e | 34.046115 | 0.009106 | 0.023751 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.004240 | 0.017718 |
| 图像干净、音频残缺 | v11e | 0.004173 | 0.017649 |
| 双模态残缺 | control | 0.004215 | 0.017613 |
| 双模态残缺 | v11e | 0.004150 | 0.017577 |
| 仅音频残缺 | control | 0.004228 | 0.017696 |
| 仅音频残缺 | v11e | 0.004163 | 0.017578 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.147264 | 0.152054 | 78.086% |
| 图像干净、音频残缺 | v11e | 0.146936 | 0.152054 | 78.668% |
| 双模态残缺 | control | 0.147181 | 0.152054 | 77.929% |
| 双模态残缺 | v11e | 0.146978 | 0.152054 | 78.451% |
| 仅音频残缺 | control | 0.146734 | 0.152054 | 77.866% |
| 仅音频残缺 | v11e | 0.145974 | 0.152054 | 78.307% |

</details>


<details>
<summary>freq_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 98.840% | 0.009763 | 0.942693 | 0.004856 | 0.860159 |
| 图像干净、音频残缺 | v11e | 98.760% | 0.012947 | 0.921893 | 0.004747 | 0.869650 |
| 双模态残缺 | control | 97.910% | 0.008750 | 0.948192 | 0.004888 | 0.859557 |
| 双模态残缺 | v11e | 97.300% | 0.008848 | 0.947576 | 0.004821 | 0.868622 |
| 仅音频残缺 | control | 92.220% | 0.007925 | 0.938894 | 0.004905 | 0.858229 |
| 仅音频残缺 | v11e | 91.960% | 0.008098 | 0.934803 | 0.004843 | 0.865803 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.781139 | 0.006477 | 0.021425 |
| 图像干净、音频残缺 | v11e | 19.568458 | 0.006128 | 0.020842 |
| 双模态残缺 | control | 23.020364 | 0.006543 | 0.021547 |
| 双模态残缺 | v11e | 23.111733 | 0.006274 | 0.021145 |
| 仅音频残缺 | control | 31.814109 | 0.006583 | 0.021626 |
| 仅音频残缺 | v11e | 31.997705 | 0.006350 | 0.021210 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.003747 | 0.016059 |
| 图像干净、音频残缺 | v11e | 0.003802 | 0.016275 |
| 双模态残缺 | control | 0.003756 | 0.016091 |
| 双模态残缺 | v11e | 0.003827 | 0.016345 |
| 仅音频残缺 | control | 0.003757 | 0.016081 |
| 仅音频残缺 | v11e | 0.003812 | 0.016266 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.151326 | 0.152054 | 80.131% |
| 图像干净、音频残缺 | v11e | 0.151686 | 0.152054 | 80.674% |
| 双模态残缺 | control | 0.151691 | 0.152054 | 80.076% |
| 双模态残缺 | v11e | 0.152114 | 0.152054 | 80.531% |
| 仅音频残缺 | control | 0.151239 | 0.152054 | 80.019% |
| 仅音频残缺 | v11e | 0.151314 | 0.152054 | 80.404% |

</details>


<details>
<summary>feature_dropout</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.320% | 0.009733 | 0.942740 | 0.003738 | 0.904314 |
| 图像干净、音频残缺 | v11e | 99.230% | 0.012887 | 0.922236 | 0.003752 | 0.906152 |
| 双模态残缺 | control | 99.030% | 0.008696 | 0.948433 | 0.003731 | 0.904324 |
| 双模态残缺 | v11e | 98.770% | 0.008666 | 0.948467 | 0.003760 | 0.905792 |
| 仅音频残缺 | control | 97.100% | 0.003009 | 0.978650 | 0.003738 | 0.904178 |
| 仅音频残缺 | v11e | 96.250% | 0.003788 | 0.971642 | 0.003765 | 0.905259 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.793747 | 0.003796 | 0.015939 |
| 图像干净、音频残缺 | v11e | 19.591252 | 0.003814 | 0.016013 |
| 双模态残缺 | control | 23.151188 | 0.003790 | 0.015938 |
| 双模态残缺 | v11e | 23.328911 | 0.003820 | 0.016041 |
| 仅音频残缺 | control | 36.140352 | 0.003797 | 0.015945 |
| 仅音频残缺 | v11e | 37.709363 | 0.003830 | 0.016037 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.003698 | 0.015718 |
| 图像干净、音频残缺 | v11e | 0.003710 | 0.015781 |
| 双模态残缺 | control | 0.003691 | 0.015713 |
| 双模态残缺 | v11e | 0.003720 | 0.015813 |
| 仅音频残缺 | control | 0.003699 | 0.015725 |
| 仅音频残缺 | v11e | 0.003722 | 0.015797 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.153838 | 0.152054 | 81.663% |
| 图像干净、音频残缺 | v11e | 0.152805 | 0.152054 | 81.983% |
| 双模态残缺 | control | 0.153931 | 0.152054 | 81.662% |
| 双模态残缺 | v11e | 0.152942 | 0.152054 | 81.916% |
| 仅音频残缺 | control | 0.153804 | 0.152054 | 81.634% |
| 仅音频残缺 | v11e | 0.152762 | 0.152054 | 81.831% |

</details>


<details>
<summary>partial_temporal</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 98.070% | 0.009949 | 0.942169 | 0.011784 | 0.538387 |
| 图像干净、音频残缺 | v11e | 97.590% | 0.013171 | 0.920787 | 0.011344 | 0.572365 |
| 双模态残缺 | control | 92.030% | 0.008929 | 0.947314 | 0.011935 | 0.536746 |
| 双模态残缺 | v11e | 91.500% | 0.009247 | 0.945054 | 0.011488 | 0.568682 |
| 仅音频残缺 | control | 45.880% | 0.042710 | 0.569261 | 0.012585 | 0.528776 |
| 仅音频残缺 | v11e | 45.500% | 0.036829 | 0.678581 | 0.012110 | 0.548484 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.701504 | 0.028559 | 0.074284 |
| 图像干净、音频残缺 | v11e | 19.487983 | 0.027502 | 0.071020 |
| 双模态残缺 | control | 22.906776 | 0.028924 | 0.074755 |
| 双模态残缺 | v11e | 22.884288 | 0.027852 | 0.071804 |
| 仅音频残缺 | control | 17.523188 | 0.030529 | 0.075829 |
| 仅音频残缺 | v11e | 18.196821 | 0.029405 | 0.073002 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.000306 | 0.001634 |
| 图像干净、音频残缺 | v11e | 0.000289 | 0.001687 |
| 双模态残缺 | control | 0.000311 | 0.001646 |
| 双模态残缺 | v11e | 0.000291 | 0.001696 |
| 仅音频残缺 | control | 0.000308 | 0.001635 |
| 仅音频残缺 | v11e | 0.000276 | 0.001573 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.131051 | 0.152054 | 70.184% |
| 图像干净、音频残缺 | v11e | 0.126014 | 0.152054 | 70.498% |
| 双模态残缺 | control | 0.130508 | 0.152054 | 69.956% |
| 双模态残缺 | v11e | 0.125733 | 0.152054 | 70.203% |
| 仅音频残缺 | control | 0.126849 | 0.152054 | 68.606% |
| 仅音频残缺 | v11e | 0.121733 | 0.152054 | 68.842% |

</details>


<details>
<summary>time_freq_block</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.120% | 0.009738 | 0.942713 | 0.003986 | 0.900341 |
| 图像干净、音频残缺 | v11e | 99.140% | 0.012894 | 0.922192 | 0.003971 | 0.901598 |
| 双模态残缺 | control | 98.770% | 0.008698 | 0.948653 | 0.003986 | 0.900257 |
| 双模态残缺 | v11e | 98.420% | 0.008718 | 0.948381 | 0.003994 | 0.901280 |
| 仅音频残缺 | control | 96.750% | 0.003525 | 0.974426 | 0.003997 | 0.900021 |
| 仅音频残缺 | v11e | 95.550% | 0.004210 | 0.967556 | 0.003993 | 0.901259 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.790810 | 0.005269 | 0.017942 |
| 图像干净、音频残缺 | v11e | 19.588628 | 0.005167 | 0.017800 |
| 双模态残缺 | control | 23.127621 | 0.005198 | 0.017747 |
| 双模态残缺 | v11e | 23.286688 | 0.005144 | 0.017693 |
| 仅音频残缺 | control | 36.145908 | 0.005324 | 0.018041 |
| 仅音频残缺 | v11e | 37.666176 | 0.005283 | 0.017992 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.003733 | 0.015994 |
| 图像干净、音频残缺 | v11e | 0.003734 | 0.016071 |
| 双模态残缺 | control | 0.003747 | 0.016053 |
| 双模态残缺 | v11e | 0.003767 | 0.016171 |
| 仅音频残缺 | control | 0.003734 | 0.015998 |
| 仅音频残缺 | v11e | 0.003738 | 0.016064 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.154400 | 0.152054 | 81.583% |
| 图像干净、音频残缺 | v11e | 0.153828 | 0.152054 | 81.946% |
| 双模态残缺 | control | 0.154617 | 0.152054 | 81.568% |
| 双模态残缺 | v11e | 0.154098 | 0.152054 | 81.873% |
| 仅音频残缺 | control | 0.154387 | 0.152054 | 81.547% |
| 仅音频残缺 | v11e | 0.153716 | 0.152054 | 81.816% |

</details>

**训练统计**

仅统计每个完整 epoch 的平均训练 loss；不同 loss 定义不可横比，最低训练 loss 也不是测试最优 checkpoint。

| 实验 | 完成 epoch | 本轮轮数 | 首轮 loss | 末轮 loss | 最低 loss | 末轮 LR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| control | 0–99 | 100 | 9.6242 | 1.2135 | 1.0775 | 0.000010 |
| v11e | 0–99 | 100 | 9.5262 | 1.2580 | 1.1251 | 0.000010 |


<details>
<summary>逐 epoch loss / LR</summary>

| epoch | 实验 | 平均 loss | LR |
| --- | --- | ---: | ---: |
| 0 | control | 9.6242 | 0.000100 |
| 0 | v11e | 9.5262 | 0.000100 |
| 1 | control | 4.8480 | 0.000100 |
| 1 | v11e | 4.6963 | 0.000100 |
| 2 | control | 3.6840 | 0.000100 |
| 2 | v11e | 3.5585 | 0.000100 |
| 3 | control | 2.9783 | 0.000100 |
| 3 | v11e | 2.8382 | 0.000100 |
| 4 | control | 2.4889 | 0.000100 |
| 4 | v11e | 2.4313 | 0.000100 |
| 5 | control | 2.1177 | 0.000099 |
| 5 | v11e | 2.1370 | 0.000099 |
| 6 | control | 1.9762 | 0.000099 |
| 6 | v11e | 1.9683 | 0.000099 |
| 7 | control | 1.8048 | 0.000099 |
| 7 | v11e | 1.7711 | 0.000099 |
| 8 | control | 1.7056 | 0.000099 |
| 8 | v11e | 1.6748 | 0.000099 |
| 9 | control | 1.6092 | 0.000098 |
| 9 | v11e | 1.6032 | 0.000098 |
| 10 | control | 1.4691 | 0.000098 |
| 10 | v11e | 1.4650 | 0.000098 |
| 11 | control | 1.3629 | 0.000097 |
| 11 | v11e | 1.3727 | 0.000097 |
| 12 | control | 1.3590 | 0.000097 |
| 12 | v11e | 1.3616 | 0.000097 |
| 13 | control | 1.2697 | 0.000096 |
| 13 | v11e | 1.2936 | 0.000096 |
| 14 | control | 1.2325 | 0.000096 |
| 14 | v11e | 1.2470 | 0.000096 |
| 15 | control | 1.1677 | 0.000095 |
| 15 | v11e | 1.2051 | 0.000095 |
| 16 | control | 1.1346 | 0.000094 |
| 16 | v11e | 1.1714 | 0.000094 |
| 17 | control | 1.0830 | 0.000094 |
| 17 | v11e | 1.1251 | 0.000094 |
| 18 | control | 1.0943 | 0.000093 |
| 18 | v11e | 1.1463 | 0.000093 |
| 19 | control | 1.0775 | 0.000092 |
| 19 | v11e | 1.2044 | 0.000092 |
| 20 | control | 2.1178 | 0.000091 |
| 20 | v11e | 2.1344 | 0.000091 |
| 21 | control | 1.8803 | 0.000091 |
| 21 | v11e | 1.9087 | 0.000091 |
| 22 | control | 1.7913 | 0.000090 |
| 22 | v11e | 1.8149 | 0.000090 |
| 23 | control | 1.6584 | 0.000089 |
| 23 | v11e | 1.6762 | 0.000089 |
| 24 | control | 1.5985 | 0.000088 |
| 24 | v11e | 1.6426 | 0.000088 |
| 25 | control | 1.7075 | 0.000087 |
| 25 | v11e | 1.7525 | 0.000087 |
| 26 | control | 1.6061 | 0.000086 |
| 26 | v11e | 1.6563 | 0.000086 |
| 27 | control | 1.5803 | 0.000085 |
| 27 | v11e | 1.6297 | 0.000085 |
| 28 | control | 1.4932 | 0.000084 |
| 28 | v11e | 1.5414 | 0.000084 |
| 29 | control | 1.5053 | 0.000083 |
| 29 | v11e | 1.5693 | 0.000083 |
| 30 | control | 1.5588 | 0.000081 |
| 30 | v11e | 1.6069 | 0.000081 |
| 31 | control | 1.5216 | 0.000080 |
| 31 | v11e | 1.5702 | 0.000080 |
| 32 | control | 1.3809 | 0.000079 |
| 32 | v11e | 1.4385 | 0.000079 |
| 33 | control | 1.4919 | 0.000078 |
| 33 | v11e | 1.5427 | 0.000078 |
| 34 | control | 1.4420 | 0.000077 |
| 34 | v11e | 1.4969 | 0.000077 |
| 35 | control | 1.6082 | 0.000075 |
| 35 | v11e | 1.6604 | 0.000075 |
| 36 | control | 1.6017 | 0.000074 |
| 36 | v11e | 1.6436 | 0.000074 |
| 37 | control | 1.4509 | 0.000073 |
| 37 | v11e | 1.5054 | 0.000073 |
| 38 | control | 1.5575 | 0.000072 |
| 38 | v11e | 1.6013 | 0.000072 |
| 39 | control | 1.5134 | 0.000070 |
| 39 | v11e | 1.5496 | 0.000070 |
| 40 | control | 1.5368 | 0.000069 |
| 40 | v11e | 1.5856 | 0.000069 |
| 41 | control | 1.4400 | 0.000068 |
| 41 | v11e | 1.4932 | 0.000068 |
| 42 | control | 1.4364 | 0.000066 |
| 42 | v11e | 1.4908 | 0.000066 |
| 43 | control | 1.5321 | 0.000065 |
| 43 | v11e | 1.5913 | 0.000065 |
| 44 | control | 1.4207 | 0.000063 |
| 44 | v11e | 1.4728 | 0.000063 |
| 45 | control | 1.4530 | 0.000062 |
| 45 | v11e | 1.5073 | 0.000062 |
| 46 | control | 1.4689 | 0.000061 |
| 46 | v11e | 1.5239 | 0.000061 |
| 47 | control | 1.3716 | 0.000059 |
| 47 | v11e | 1.4246 | 0.000059 |
| 48 | control | 1.3880 | 0.000058 |
| 48 | v11e | 1.4306 | 0.000058 |
| 49 | control | 1.4133 | 0.000056 |
| 49 | v11e | 1.4676 | 0.000056 |
| 50 | control | 1.4376 | 0.000055 |
| 50 | v11e | 1.4868 | 0.000055 |
| 51 | control | 1.2928 | 0.000054 |
| 51 | v11e | 1.3324 | 0.000054 |
| 52 | control | 1.3797 | 0.000052 |
| 52 | v11e | 1.4206 | 0.000052 |
| 53 | control | 1.3155 | 0.000051 |
| 53 | v11e | 1.3598 | 0.000051 |
| 54 | control | 1.3097 | 0.000049 |
| 54 | v11e | 1.3537 | 0.000049 |
| 55 | control | 1.3705 | 0.000048 |
| 55 | v11e | 1.4135 | 0.000048 |
| 56 | control | 1.2378 | 0.000047 |
| 56 | v11e | 1.2876 | 0.000047 |
| 57 | control | 1.4385 | 0.000045 |
| 57 | v11e | 1.4963 | 0.000045 |
| 58 | control | 1.3510 | 0.000044 |
| 58 | v11e | 1.3950 | 0.000044 |
| 59 | control | 1.3359 | 0.000042 |
| 59 | v11e | 1.3707 | 0.000042 |
| 60 | control | 1.3023 | 0.000041 |
| 60 | v11e | 1.3508 | 0.000041 |
| 61 | control | 1.3311 | 0.000040 |
| 61 | v11e | 1.3730 | 0.000040 |
| 62 | control | 1.3014 | 0.000038 |
| 62 | v11e | 1.3626 | 0.000038 |
| 63 | control | 1.2912 | 0.000037 |
| 63 | v11e | 1.3387 | 0.000037 |
| 64 | control | 1.3752 | 0.000036 |
| 64 | v11e | 1.4311 | 0.000036 |
| 65 | control | 1.2502 | 0.000035 |
| 65 | v11e | 1.2905 | 0.000035 |
| 66 | control | 1.2517 | 0.000033 |
| 66 | v11e | 1.2903 | 0.000033 |
| 67 | control | 1.3054 | 0.000032 |
| 67 | v11e | 1.3576 | 0.000032 |
| 68 | control | 1.2910 | 0.000031 |
| 68 | v11e | 1.3434 | 0.000031 |
| 69 | control | 1.2521 | 0.000030 |
| 69 | v11e | 1.2936 | 0.000030 |
| 70 | control | 1.3243 | 0.000029 |
| 70 | v11e | 1.3725 | 0.000029 |
| 71 | control | 1.2828 | 0.000027 |
| 71 | v11e | 1.3280 | 0.000027 |
| 72 | control | 1.2873 | 0.000026 |
| 72 | v11e | 1.3147 | 0.000026 |
| 73 | control | 1.1437 | 0.000025 |
| 73 | v11e | 1.1884 | 0.000025 |
| 74 | control | 1.2973 | 0.000024 |
| 74 | v11e | 1.3412 | 0.000024 |
| 75 | control | 1.1011 | 0.000023 |
| 75 | v11e | 1.1393 | 0.000023 |
| 76 | control | 1.2013 | 0.000022 |
| 76 | v11e | 1.2397 | 0.000022 |
| 77 | control | 1.2436 | 0.000021 |
| 77 | v11e | 1.2858 | 0.000021 |
| 78 | control | 1.2446 | 0.000020 |
| 78 | v11e | 1.2853 | 0.000020 |
| 79 | control | 1.2806 | 0.000019 |
| 79 | v11e | 1.3281 | 0.000019 |
| 80 | control | 1.2498 | 0.000019 |
| 80 | v11e | 1.2911 | 0.000019 |
| 81 | control | 1.2079 | 0.000018 |
| 81 | v11e | 1.2408 | 0.000018 |
| 82 | control | 1.2230 | 0.000017 |
| 82 | v11e | 1.2579 | 0.000017 |
| 83 | control | 1.2388 | 0.000016 |
| 83 | v11e | 1.2915 | 0.000016 |
| 84 | control | 1.2347 | 0.000016 |
| 84 | v11e | 1.2760 | 0.000016 |
| 85 | control | 1.2773 | 0.000015 |
| 85 | v11e | 1.3216 | 0.000015 |
| 86 | control | 1.1769 | 0.000014 |
| 86 | v11e | 1.2197 | 0.000014 |
| 87 | control | 1.2065 | 0.000014 |
| 87 | v11e | 1.2545 | 0.000014 |
| 88 | control | 1.1917 | 0.000013 |
| 88 | v11e | 1.2348 | 0.000013 |
| 89 | control | 1.2156 | 0.000013 |
| 89 | v11e | 1.2602 | 0.000013 |
| 90 | control | 1.2118 | 0.000012 |
| 90 | v11e | 1.2551 | 0.000012 |
| 91 | control | 1.1591 | 0.000012 |
| 91 | v11e | 1.2081 | 0.000012 |
| 92 | control | 1.2480 | 0.000011 |
| 92 | v11e | 1.2928 | 0.000011 |
| 93 | control | 1.1314 | 0.000011 |
| 93 | v11e | 1.1753 | 0.000011 |
| 94 | control | 1.1536 | 0.000011 |
| 94 | v11e | 1.1890 | 0.000011 |
| 95 | control | 1.2962 | 0.000011 |
| 95 | v11e | 1.3410 | 0.000011 |
| 96 | control | 1.1714 | 0.000010 |
| 96 | v11e | 1.2122 | 0.000010 |
| 97 | control | 1.1240 | 0.000010 |
| 97 | v11e | 1.1667 | 0.000010 |
| 98 | control | 1.1306 | 0.000010 |
| 98 | v11e | 1.1769 | 0.000010 |
| 99 | control | 1.2135 | 0.000010 |
| 99 | v11e | 1.2580 | 0.000010 |

</details>

训练证据：
- `control`：[train_v11e_control_100ep.log](../v11e_outputs_with_ckpt/outputs/outputs_v11e_control/logs/train_v11e_control_100ep.log)
- `v11e`：[train_v11e_100ep.log](../v11e_outputs_with_ckpt/outputs/outputs_v11e/logs/train_v11e_100ep.log)

**Demo 小样本**

每份 demo 为 10 条样本，不是全测试集分数；fixed/random 分开。原始图片与逐样本预测留在对应产物目录，本节不宣称重新进行了图像质量评审。


<details>
<summary>fixed_mask demo (n=10)</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 仅音频残缺 | v11e | 50.000% | 0.028000 | 0.757000 | 0.011000 | 0.724000 |
| 仅图像残缺 | v11e | 90.000% | 0.008300 | 0.944000 | 0.001500 | 0.863000 |
| 双模态残缺 | v11e | 100.000% | 0.008200 | 0.946000 | 0.010600 | 0.736000 |

| 输入模式 | 实验 | 图像缺失区 MSE | 音频缺失区 MSE |
| --- | --- | ---: | ---: |
| 仅音频残缺 | v11e | 0.028000 | 0.025700 |
| 仅图像残缺 | v11e | 0.026800 | 0.001500 |
| 双模态残缺 | v11e | 0.026200 | 0.024600 |

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 仅音频残缺 | v11e | 0.042700 | 0.150900 | 1.000000 |
| 仅图像残缺 | v11e | 0.016200 | 0.068700 | 0.922300 |
| 双模态残缺 | v11e | 0.043800 | 0.151900 | 1.000000 |

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅音频残缺 | v11e | 0.047400 | 0.168700 | 1.000000 | 80.100% |
| 仅图像残缺 | v11e | 0.020700 | 0.084100 | 0.987900 | 84.800% |
| 双模态残缺 | v11e | 0.047400 | 0.168700 | 1.000000 | 80.200% |

</details>


<details>
<summary>legacy_random demo (n=10)</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 仅音频残缺 | v11e | 70.000% | 0.021300 | 0.818000 | 0.017100 | 0.628000 |
| 仅图像残缺 | v11e | 90.000% | 0.012100 | 0.923000 | 0.000700 | 0.946000 |
| 双模态残缺 | v11e | 90.000% | 0.009200 | 0.944000 | 0.013500 | 0.713000 |

| 输入模式 | 实验 | 图像缺失区 MSE | 音频缺失区 MSE |
| --- | --- | ---: | ---: |
| 仅音频残缺 | v11e | 0.021300 | 0.035600 |
| 仅图像残缺 | v11e | 0.078200 | 0.000700 |
| 双模态残缺 | v11e | 0.059800 | 0.027300 |

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 仅音频残缺 | v11e | 0.037800 | 0.133900 | 1.000000 |
| 仅图像残缺 | v11e | 0.019100 | 0.073900 | 0.933300 |
| 双模态残缺 | v11e | 0.042800 | 0.148100 | 1.000000 |

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅音频残缺 | v11e | 0.047400 | 0.168700 | 1.000000 | 71.300% |
| 仅图像残缺 | v11e | 0.020700 | 0.084100 | 0.987900 | 84.500% |
| 双模态残缺 | v11e | 0.047400 | 0.168700 | 1.000000 | 73.900% |

</details>

**各实验结论与比较限制**

- **仅音频残缺**：主实验相对 control，音频缺失区 MSE 变化 -3.07%，Index ACC 变化 -0.960 个百分点。
- **双模态残缺**：主实验相对 control，音频缺失区 MSE 变化 -3.53%，Index ACC 变化 -0.460 个百分点。
- **control**：从头训练 100 轮，关闭 Cross-Key/causal，作为同预算类别绑定对照。
- **v11e**：Cross-Key 在完全缺失场景有 zero/wrong 干预效应，但部分残缺恢复收益有限；分类与恢复存在取舍。没有 no_causal 第三组，不能分离模块与因果损失的贡献。
- 更正旧条目的 fixed/random 解释：89.6% 等 fixed 数字来自第一组 family，而不是五类宏平均；本节主表分别给出宏平均与 random，不混淆二者。
- 评估日志含 `libgomp: Invalid value for environment variable OMP_NUM_THREADS` 警告；输出表完整，但不能描述为完全没有环境警告。
- 本节仅汇总已有结果，没有重跑训练或推理。单 seed、重复音频曝光和旧日志舍入限制仍在，不报告统计显著性。

[返回统一评估导航](#evaluation-format-20260912)

<a id="evaluation-v11f"></a>

### v11f

本节覆盖 **3 组实验、52 个主评估/干预字段、7 个音频分布诊断字段**，以及独立 family、训练统计和本地已有 demo。字段按实际产物统计，含明确标注的不适用项；不把其他版本的缺项补成零。

**实验说明**

| 实验 | 权重起点 | 本轮训练范围 | 实际轮数 |
| --- | --- | ---: | ---: |
| control | v11e_control | 冻结父模型；不训练 | 0 |
| v11f | v11e_control | 仅 Masked Cross-Key adapter + causal | 30（0–29） |
| no_causal | v11e_control | 仅 adapter；关闭 causal | 30（0–29） |

训练配置 seed=1234、batch_size=128；fixed_mask seed=1234、severity=0.4。轮数以上表和完成日志为准，不把父模型历史轮数计成本轮新增训练。历史分支配置不是当前分支的运行入口。

**恢复目标：仅图像为 sample/category，仅音频为 category/sample，双模态为 sample/sample。** category 使用训练集 medoid；“仅音频残缺”表示图像全缺失、音频部分残缺，反向同理。

| 实验 | fixed / random | Cross-Key 扫描 | demo fixed / random |
| --- | --- | ---: | ---: |
| control | 有 / 有 | fixed + random | 有 / 有 |
| v11f | 有 / 有 | fixed + random | 有 / 有 |
| no_causal | 有 / 有 | fixed + random | 有 / 有 |

**数据来源**：
- `control`：[本地产物](../v11f_outputs_with_ckpt/outputs/outputs_v11f_control/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。
- `v11f`：[本地产物](../v11f_outputs_with_ckpt/outputs/outputs_v11f/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。
- `no_causal`：[本地产物](../v11f_outputs_with_ckpt/outputs/outputs_v11f_no_causal/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。

MSE/L1 越低越好，ACC/SSIM/PSNR 越高越好；gate、res/V 和多样性仅是诊断。宏平均为五个 family 等权均值，不是把 normal、sweep、独立音频 family 重复叠加。N/A 表示未保存或在该场景不适用，详见字段覆盖说明。

**有效 n**：normal 有效指标每 family 通常 n=10000，不适用区域 n=0；same-class 干预通常 n=9996，其余有效干预 n=10000。五类是同一测试集的重复曝光，不是 50000 条独立音频；精确字段计数见附表。

**分类与恢复：fixed 五类等权宏平均**

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11f | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11f | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965814 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966055 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11f | 98.830% | 0.002052 | 0.985961 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002077 | 0.985706 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 98.934% | 0.009790 | 0.942597 | 0.006097 | 0.807590 |
| 图像干净、音频残缺 | v11f | 98.934% | 0.009790 | 0.942597 | 0.005921 | 0.809909 |
| 图像干净、音频残缺 | no_causal | 98.934% | 0.009790 | 0.942597 | 0.005978 | 0.810239 |
| 音频干净、图像残缺 | control | 99.144% | 0.007502 | 0.956305 | 0.003647 | 0.911207 |
| 音频干净、图像残缺 | v11f | 99.144% | 0.007319 | 0.957252 | 0.003647 | 0.911207 |
| 音频干净、图像残缺 | no_causal | 99.144% | 0.007395 | 0.956837 | 0.003647 | 0.911207 |
| 双模态残缺 | control | 97.536% | 0.007520 | 0.956189 | 0.006176 | 0.805881 |
| 双模态残缺 | v11f | 97.536% | 0.007370 | 0.957024 | 0.006041 | 0.807620 |
| 双模态残缺 | no_causal | 97.536% | 0.007438 | 0.956651 | 0.006087 | 0.807978 |
| 仅音频残缺 | control | 84.572% | 0.013212 | 0.877703 | 0.006290 | 0.804689 |
| 仅音频残缺 | v11f | 84.572% | 0.012743 | 0.886050 | 0.006290 | 0.804689 |
| 仅音频残缺 | no_causal | 84.572% | 0.012306 | 0.893063 | 0.006290 | 0.804689 |
| 仅图像残缺 | control | 92.028% | 0.007680 | 0.955151 | 0.000668 | 0.916781 |
| 仅图像残缺 | v11f | 92.028% | 0.007680 | 0.955151 | 0.000670 | 0.916379 |
| 仅图像残缺 | no_causal | 92.028% | 0.007680 | 0.955151 | 0.000673 | 0.917347 |


<details>
<summary>fixed 五类等权宏平均：区域、内容、多样性与音频分布完整指标</summary>

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11f | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11f | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11f | 37.208255 | 0.002052 | 0.002052 | 0.007433 |
| 仅干净音频 | no_causal | 36.902481 | 0.002077 | 0.002077 | 0.007504 |
| 图像干净、音频残缺 | control | 20.768335 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | 20.768335 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.768335 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 22.993817 | 0.054834 | 0.025918 | 0.062284 |
| 音频干净、图像残缺 | v11f | 23.094712 | 0.055786 | 0.025257 | 0.062744 |
| 音频干净、图像残缺 | no_causal | 23.061578 | 0.054808 | 0.025463 | 0.062059 |
| 双模态残缺 | control | 22.976000 | 0.054686 | 0.025912 | 0.062268 |
| 双模态残缺 | v11f | 23.059602 | 0.054910 | 0.025352 | 0.062799 |
| 双模态残缺 | no_causal | 23.022582 | 0.054205 | 0.025549 | 0.062580 |
| 仅音频残缺 | control | 30.894253 | 0.013212 | 0.013212 | 0.025060 |
| 仅音频残缺 | v11f | 30.743681 | 0.012743 | 0.012743 | 0.025330 |
| 仅音频残缺 | no_causal | 30.480506 | 0.012306 | 0.012306 | 0.025855 |
| 仅图像残缺 | control | 22.904681 | 0.056325 | 0.026662 | 0.063682 |
| 仅图像残缺 | v11f | 22.904681 | 0.056325 | 0.026662 | 0.063682 |
| 仅图像残缺 | no_causal | 22.904681 | 0.056325 | 0.026662 | 0.063682 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.030732 | 8.090e-09 | 0.000081 |
| 音频干净、图像残缺 | v11f | 0.030732 | 8.090e-09 | 0.000081 |
| 音频干净、图像残缺 | no_causal | 0.030732 | 8.090e-09 | 0.000081 |
| 双模态残缺 | control | 0.030551 | 8.089e-09 | 0.000081 |
| 双模态残缺 | v11f | 0.030551 | 8.089e-09 | 0.000081 |
| 双模态残缺 | no_causal | 0.030551 | 8.089e-09 | 0.000081 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.031234 | 8.090e-09 | 0.000081 |
| 仅图像残缺 | v11f | 0.031234 | 8.090e-09 | 0.000081 |
| 仅图像残缺 | no_causal | 0.031234 | 8.090e-09 | 0.000081 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11f | 0.000294 | 0.003618 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003556 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.010599 | 0.030586 | 0.003143 | 0.013437 |
| 图像干净、音频残缺 | v11f | 0.010127 | 0.030618 | 0.003143 | 0.013437 |
| 图像干净、音频残缺 | no_causal | 0.010276 | 0.030690 | 0.003143 | 0.013437 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.010775 | 0.030940 | 0.003137 | 0.013386 |
| 双模态残缺 | v11f | 0.010417 | 0.030931 | 0.003137 | 0.013386 |
| 双模态残缺 | no_causal | 0.010537 | 0.030991 | 0.003137 | 0.013386 |
| 仅音频残缺 | control | 0.011079 | 0.031012 | 0.003143 | 0.013438 |
| 仅音频残缺 | v11f | 0.011079 | 0.031012 | 0.003143 | 0.013438 |
| 仅音频残缺 | no_causal | 0.011079 | 0.031012 | 0.003143 | 0.013438 |
| 仅图像残缺 | control | 0.000668 | 0.005295 | N/A | N/A |
| 仅图像残缺 | v11f | 0.000670 | 0.005429 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000673 | 0.005321 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.338677 |
| 干净双模态 | v11f | 97.790% | 96.040% | 0.058851 | 0.338677 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.338677 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.340023 |
| 仅干净图像 | v11f | 96.820% | 94.250% | 0.059386 | 0.340023 |
| 仅干净图像 | no_causal | 96.820% | 94.550% | 0.059386 | 0.340023 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.287549 |
| 仅干净音频 | v11f | 98.090% | 96.000% | 0.045937 | 0.286380 |
| 仅干净音频 | no_causal | 97.750% | 96.000% | 0.046202 | 0.287337 |
| 图像干净、音频残缺 | control | 97.520% | 88.970% | 0.059321 | 0.339469 |
| 图像干净、音频残缺 | v11f | 97.520% | 88.918% | 0.059321 | 0.339469 |
| 图像干净、音频残缺 | no_causal | 97.520% | 89.142% | 0.059321 | 0.339469 |
| 音频干净、图像残缺 | control | 96.720% | 95.968% | 0.061073 | 0.345199 |
| 音频干净、图像残缺 | v11f | 96.736% | 95.968% | 0.060675 | 0.344028 |
| 音频干净、图像残缺 | no_causal | 96.754% | 95.968% | 0.060881 | 0.344608 |
| 双模态残缺 | control | 96.178% | 87.012% | 0.061083 | 0.345650 |
| 双模态残缺 | v11f | 96.278% | 87.012% | 0.060892 | 0.345118 |
| 双模态残缺 | no_causal | 96.296% | 87.134% | 0.060992 | 0.345393 |
| 仅音频残缺 | control | 82.882% | 80.600% | 0.040052 | 0.265096 |
| 仅音频残缺 | v11f | 82.986% | 80.600% | 0.039601 | 0.263346 |
| 仅音频残缺 | no_causal | 82.846% | 80.600% | 0.039433 | 0.262475 |
| 仅图像残缺 | control | 95.514% | 83.518% | 0.060839 | 0.344495 |
| 仅图像残缺 | v11f | 95.514% | 83.526% | 0.060839 | 0.344495 |
| 仅图像残缺 | no_causal | 95.514% | 84.174% | 0.060839 | 0.344495 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11f | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11f | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11f | 0.015600 | 0.068500 | 0.981600 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.994400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.040800 | 0.147600 | 1.000000 |
| 图像干净、音频残缺 | v11f | 0.041160 | 0.146560 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.041560 | 0.148500 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155680 | 1.000000 |
| 音频干净、图像残缺 | v11f | 0.043200 | 0.155680 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155680 | 1.000000 |
| 双模态残缺 | control | 0.040800 | 0.147500 | 1.000000 |
| 双模态残缺 | v11f | 0.041060 | 0.146780 | 1.000000 |
| 双模态残缺 | no_causal | 0.041400 | 0.148320 | 1.000000 |
| 仅音频残缺 | control | 0.040340 | 0.146620 | 1.000000 |
| 仅音频残缺 | v11f | 0.040340 | 0.146620 | 1.000000 |
| 仅音频残缺 | no_causal | 0.040340 | 0.146620 | 1.000000 |
| 仅图像残缺 | control | 0.015280 | 0.066280 | 0.999900 |
| 仅图像残缺 | v11f | 0.015700 | 0.066220 | 0.994700 |
| 仅图像残缺 | no_causal | 0.015420 | 0.066880 | 0.999300 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11f | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.340% |
| 图像干净、音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 78.420% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.420% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.240% |
| 双模态残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 77.940% |
| 仅音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 77.940% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 77.940% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 78.700% |
| 仅图像残缺 | v11f | 0.015600 | 0.070500 | 0.987900 | 78.640% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 78.640% |

</details>


<details>
<summary>fixed 五类等权宏平均：各字段有效样本数</summary>

| 实验 | 指标字段（同一计数口径） | 每 family 的 n |
| --- | --- | ---: |
| control | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| control | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| control | `pair_l2`、`pix_var`、`psnr` | 10000 |
| control | `ssim` | 10000 |
| control | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| control | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| control | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| control | `img_visible_mse` | 0, 10000 |
| control | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| control | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| control | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| control | `top15_recall` | 未逐指标保存 |
| v11f | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| v11f | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| v11f | `pair_l2`、`pix_var`、`psnr` | 10000 |
| v11f | `ssim` | 10000 |
| v11f | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| v11f | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| v11f | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| v11f | `img_visible_mse` | 0, 10000 |
| v11f | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| v11f | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| v11f | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| v11f | `top15_recall` | 未逐指标保存 |
| no_causal | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| no_causal | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| no_causal | `pair_l2`、`pix_var`、`psnr` | 10000 |
| no_causal | `ssim` | 10000 |
| no_causal | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| no_causal | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| no_causal | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| no_causal | `img_visible_mse` | 0, 10000 |
| no_causal | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| no_causal | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| no_causal | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| no_causal | `top15_recall` | 未逐指标保存 |

</details>

**分类与恢复：random 单次评估**

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11f | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11f | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965814 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966055 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11f | 98.830% | 0.002052 | 0.985961 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002077 | 0.985706 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 98.960% | 0.009783 | 0.942602 | 0.005872 | 0.816051 |
| 图像干净、音频残缺 | v11f | 98.960% | 0.009783 | 0.942602 | 0.005706 | 0.818213 |
| 图像干净、音频残缺 | no_causal | 98.960% | 0.009783 | 0.942602 | 0.005765 | 0.818253 |
| 音频干净、图像残缺 | control | 99.160% | 0.007417 | 0.956508 | 0.003647 | 0.911193 |
| 音频干净、图像残缺 | v11f | 99.160% | 0.007204 | 0.957629 | 0.003647 | 0.911193 |
| 音频干净、图像残缺 | no_causal | 99.160% | 0.007274 | 0.957239 | 0.003647 | 0.911193 |
| 双模态残缺 | control | 97.280% | 0.007719 | 0.954641 | 0.005927 | 0.820633 |
| 双模态残缺 | v11f | 97.280% | 0.007563 | 0.955451 | 0.005804 | 0.822283 |
| 双模态残缺 | no_causal | 97.280% | 0.007623 | 0.955149 | 0.005841 | 0.822596 |
| 仅音频残缺 | control | 85.790% | 0.012209 | 0.887254 | 0.006055 | 0.813113 |
| 仅音频残缺 | v11f | 85.790% | 0.011799 | 0.894696 | 0.006055 | 0.813113 |
| 仅音频残缺 | no_causal | 85.790% | 0.011416 | 0.900913 | 0.006055 | 0.813113 |
| 仅图像残缺 | control | 92.360% | 0.007614 | 0.955219 | 0.000627 | 0.921694 |
| 仅图像残缺 | v11f | 92.360% | 0.007614 | 0.955219 | 0.000630 | 0.921340 |
| 仅图像残缺 | no_causal | 92.360% | 0.007614 | 0.955219 | 0.000631 | 0.922253 |

random 不与 fixed 混算；表中整行 N/A 表示该实验未找到 random 结果，不表示实验得分为 0。


<details>
<summary>random 单次评估：区域、内容、多样性与音频分布完整指标</summary>

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11f | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11f | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11f | 37.208255 | 0.002052 | 0.002052 | 0.007433 |
| 仅干净音频 | no_causal | 36.902481 | 0.002077 | 0.002077 | 0.007504 |
| 图像干净、音频残缺 | control | 20.770485 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | 20.770485 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.770485 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 22.956013 | 0.058904 | 0.028262 | 0.067230 |
| 音频干净、图像残缺 | v11f | 23.074014 | 0.059553 | 0.027421 | 0.067536 |
| 音频干净、图像残缺 | no_causal | 23.039616 | 0.058469 | 0.027600 | 0.066752 |
| 双模态残缺 | control | 22.756852 | 0.054375 | 0.026184 | 0.062785 |
| 双模态残缺 | v11f | 22.834460 | 0.054548 | 0.025611 | 0.063256 |
| 双模态残缺 | no_causal | 22.804034 | 0.053841 | 0.025777 | 0.062783 |
| 仅音频残缺 | control | 31.275545 | 0.012209 | 0.012209 | 0.023570 |
| 仅音频残缺 | v11f | 31.122341 | 0.011799 | 0.011799 | 0.023846 |
| 仅音频残缺 | no_causal | 30.850819 | 0.011416 | 0.011416 | 0.024341 |
| 仅图像残缺 | control | 22.865423 | 0.060709 | 0.029233 | 0.068927 |
| 仅图像残缺 | v11f | 22.865423 | 0.060709 | 0.029233 | 0.068927 |
| 仅图像残缺 | no_causal | 22.865423 | 0.060709 | 0.029233 | 0.068927 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.028630 | 8.143e-09 | 0.000081 |
| 音频干净、图像残缺 | v11f | 0.028630 | 8.143e-09 | 0.000081 |
| 音频干净、图像残缺 | no_causal | 0.028630 | 8.143e-09 | 0.000081 |
| 双模态残缺 | control | 0.030697 | 8.093e-09 | 0.000081 |
| 双模态残缺 | v11f | 0.030697 | 8.093e-09 | 0.000081 |
| 双模态残缺 | no_causal | 0.030697 | 8.093e-09 | 0.000081 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.029101 | 8.143e-09 | 0.000081 |
| 仅图像残缺 | v11f | 0.029101 | 8.143e-09 | 0.000081 |
| 仅图像残缺 | no_causal | 0.029101 | 8.143e-09 | 0.000081 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11f | 0.000294 | 0.003618 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003556 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.009929 | 0.029169 | 0.003206 | 0.013700 |
| 图像干净、音频残缺 | v11f | 0.009485 | 0.029187 | 0.003206 | 0.013700 |
| 图像干净、音频残缺 | no_causal | 0.009638 | 0.029266 | 0.003206 | 0.013700 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.009910 | 0.028544 | 0.003315 | 0.014139 |
| 双模态残缺 | v11f | 0.009581 | 0.028515 | 0.003315 | 0.014139 |
| 双模态残缺 | no_causal | 0.009678 | 0.028561 | 0.003315 | 0.014139 |
| 仅音频残缺 | control | 0.010381 | 0.029567 | 0.003207 | 0.013704 |
| 仅音频残缺 | v11f | 0.010381 | 0.029567 | 0.003207 | 0.013704 |
| 仅音频残缺 | no_causal | 0.010381 | 0.029567 | 0.003207 | 0.013704 |
| 仅图像残缺 | control | 0.000627 | 0.005133 | N/A | N/A |
| 仅图像残缺 | v11f | 0.000630 | 0.005266 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000631 | 0.005160 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.339940 |
| 干净双模态 | v11f | 97.790% | 96.040% | 0.058851 | 0.339940 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.339940 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.339948 |
| 仅干净图像 | v11f | 96.820% | 94.250% | 0.059386 | 0.339948 |
| 仅干净图像 | no_causal | 96.820% | 94.550% | 0.059386 | 0.339948 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.285884 |
| 仅干净音频 | v11f | 98.090% | 96.000% | 0.045937 | 0.284692 |
| 仅干净音频 | no_causal | 97.750% | 96.000% | 0.046202 | 0.285661 |
| 图像干净、音频残缺 | control | 97.500% | 89.830% | 0.059285 | 0.341653 |
| 图像干净、音频残缺 | v11f | 97.500% | 89.720% | 0.059285 | 0.341653 |
| 图像干净、音频残缺 | no_causal | 97.500% | 90.060% | 0.059285 | 0.341653 |
| 音频干净、图像残缺 | control | 96.690% | 96.000% | 0.061193 | 0.346267 |
| 音频干净、图像残缺 | v11f | 96.720% | 96.000% | 0.060845 | 0.345230 |
| 音频干净、图像残缺 | no_causal | 96.780% | 96.000% | 0.061043 | 0.345785 |
| 双模态残缺 | control | 96.340% | 87.310% | 0.060926 | 0.345003 |
| 双模态残缺 | v11f | 96.390% | 87.530% | 0.060677 | 0.344288 |
| 双模态残缺 | no_causal | 96.380% | 87.670% | 0.060841 | 0.344744 |
| 仅音频残缺 | control | 84.090% | 81.770% | 0.041126 | 0.274684 |
| 仅音频残缺 | v11f | 84.250% | 81.770% | 0.040567 | 0.272502 |
| 仅音频残缺 | no_causal | 84.010% | 81.770% | 0.040292 | 0.271363 |
| 仅图像残缺 | control | 95.490% | 84.270% | 0.060948 | 0.345536 |
| 仅图像残缺 | v11f | 95.490% | 84.480% | 0.060948 | 0.345536 |
| 仅图像残缺 | no_causal | 95.490% | 85.160% | 0.060948 | 0.345536 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11f | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11f | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11f | 0.015600 | 0.068500 | 0.981600 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.994400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.041100 | 0.148400 | 1.000000 |
| 图像干净、音频残缺 | v11f | 0.041300 | 0.147400 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.041700 | 0.149200 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11f | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155700 | 1.000000 |
| 双模态残缺 | control | 0.040800 | 0.147900 | 1.000000 |
| 双模态残缺 | v11f | 0.041000 | 0.147300 | 1.000000 |
| 双模态残缺 | no_causal | 0.041300 | 0.148600 | 1.000000 |
| 仅音频残缺 | control | 0.040600 | 0.147500 | 1.000000 |
| 仅音频残缺 | v11f | 0.040600 | 0.147500 | 1.000000 |
| 仅音频残缺 | no_causal | 0.040600 | 0.147500 | 1.000000 |
| 仅图像残缺 | control | 0.015200 | 0.066200 | 1.000000 |
| 仅图像残缺 | v11f | 0.015600 | 0.066200 | 0.993600 |
| 仅图像残缺 | no_causal | 0.015400 | 0.066800 | 0.999100 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11f | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.600% |
| 图像干净、音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 78.700% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.700% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.500% |
| 双模态残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 78.600% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.600% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 仅音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 78.900% |
| 仅图像残缺 | v11f | 0.015600 | 0.070500 | 0.987900 | 78.800% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 78.800% |

</details>


<details>
<summary>random 单次评估：各字段有效样本数</summary>

| 实验 | 指标字段（同一计数口径） | n |
| --- | --- | ---: |
| control | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| control | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| control | `pair_l2`、`pix_var`、`psnr` | 10000 |
| control | `ssim` | 10000 |
| control | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| control | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| control | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| control | `img_visible_mse` | 0, 10000 |
| control | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| control | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| control | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| control | `top15_recall` | 未逐指标保存 |
| v11f | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| v11f | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| v11f | `pair_l2`、`pix_var`、`psnr` | 10000 |
| v11f | `ssim` | 10000 |
| v11f | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| v11f | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| v11f | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| v11f | `img_visible_mse` | 0, 10000 |
| v11f | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| v11f | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| v11f | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| v11f | `top15_recall` | 未逐指标保存 |
| no_causal | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| no_causal | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| no_causal | `pair_l2`、`pix_var`、`psnr` | 10000 |
| no_causal | `ssim` | 10000 |
| no_causal | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| no_causal | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| no_causal | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| no_causal | `img_visible_mse` | 0, 10000 |
| no_causal | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| no_causal | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| no_causal | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| no_causal | `top15_recall` | 未逐指标保存 |

</details>

**Cross-Key / Cross-Detail 干预**

同一 cue/mask 内，gain=zero−normal，wrong damage=wrong−normal，same damage 为有效同类替换上的配对差；正 gain 表示改善。gate 非零本身不代表恢复有效。


<details>
<summary>fixed 五类等权宏平均：干预完整指标</summary>

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11f | 0.002052 | 0.002085 | 0.002049 | 0.002040 |
| 仅干净音频 | no_causal | 0.002077 | 0.002085 | 0.002083 | 0.002071 |
| 音频干净、图像残缺 | control | 0.025918 | 0.025918 | 0.025918 | 0.025916 |
| 音频干净、图像残缺 | v11f | 0.025257 | 0.025918 | 0.025302 | 0.025257 |
| 音频干净、图像残缺 | no_causal | 0.025463 | 0.025918 | 0.025546 | 0.025464 |
| 双模态残缺 | control | 0.025912 | 0.025912 | 0.025912 | 0.025907 |
| 双模态残缺 | v11f | 0.025352 | 0.025912 | 0.025379 | 0.025339 |
| 双模态残缺 | no_causal | 0.025549 | 0.025912 | 0.025609 | 0.025535 |
| 仅音频残缺 | control | 0.013212 | 0.013212 | 0.013212 | 0.013212 |
| 仅音频残缺 | v11f | 0.012743 | 0.013212 | 0.012729 | 0.012723 |
| 仅音频残缺 | no_causal | 0.012306 | 0.013212 | 0.012549 | 0.012523 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11f | 0.000033 | -0.000003 | -0.000013 |
| 仅干净音频 | no_causal | 0.000008 | 0.000006 | -0.000006 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.000661 | 0.000045 | 0.000002 |
| 音频干净、图像残缺 | no_causal | 0.000455 | 0.000083 | 0.000003 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000560 | 0.000027 | -0.000008 |
| 双模态残缺 | no_causal | 0.000363 | 0.000060 | -0.000009 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11f | 0.000469 | -0.000013 | -0.000020 |
| 仅音频残缺 | no_causal | 0.000905 | 0.000243 | 0.000216 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11f | 36.260% | 63.900% | 28.100% |
| 仅干净音频 | no_causal | 30.240% | 63.710% | 24.860% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11f | 68.540% | 52.338% | 38.216% |
| 音频干净、图像残缺 | no_causal | 61.064% | 52.492% | 35.954% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 67.618% | 49.846% | 36.596% |
| 双模态残缺 | no_causal | 60.584% | 50.108% | 35.208% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11f | 54.082% | 57.730% | 33.742% |
| 仅音频残缺 | no_causal | 45.012% | 60.664% | 31.686% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11f | 0.544109 | 0.428402 |
| 仅干净音频 | no_causal | 0.754257 | 0.214882 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.572656 | 0.226294 |
| 音频干净、图像残缺 | no_causal | 0.741362 | 0.108225 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.571634 | 0.181210 |
| 双模态残缺 | no_causal | 0.745631 | 0.101489 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11f | 0.545633 | 0.342391 |
| 仅音频残缺 | no_causal | 0.756886 | 0.204522 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11f | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000323 | 0.000300 |
| 图像干净、音频残缺 | control | 0.010599 | 0.010599 | 0.010599 | 0.010600 |
| 图像干净、音频残缺 | v11f | 0.010127 | 0.010599 | 0.010143 | 0.010126 |
| 图像干净、音频残缺 | no_causal | 0.010276 | 0.010599 | 0.010359 | 0.010274 |
| 双模态残缺 | control | 0.010775 | 0.010775 | 0.010775 | 0.010775 |
| 双模态残缺 | v11f | 0.010417 | 0.010775 | 0.010423 | 0.010417 |
| 双模态残缺 | no_causal | 0.010537 | 0.010775 | 0.010585 | 0.010537 |
| 仅图像残缺 | control | 0.000668 | 0.000668 | 0.000668 | 0.000668 |
| 仅图像残缺 | v11f | 0.000670 | 0.000668 | 0.000675 | 0.000672 |
| 仅图像残缺 | no_causal | 0.000673 | 0.000668 | 0.000686 | 0.000682 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11f | -0.000005 | 0.000010 | 7.096e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000027 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.000472 | 0.000015 | -0.000002 |
| 图像干净、音频残缺 | no_causal | 0.000324 | 0.000083 | -0.000002 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000358 | 0.000007 | 6.460e-07 |
| 双模态残缺 | no_causal | 0.000238 | 0.000048 | -2.367e-07 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11f | -0.000002 | 0.000005 | 0.000001 |
| 仅图像残缺 | no_causal | -0.000005 | 0.000013 | 0.000009 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11f | 34.920% | 69.880% | 26.600% |
| 仅干净图像 | no_causal | 30.950% | 71.780% | 25.460% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11f | 66.964% | 43.524% | 36.230% |
| 图像干净、音频残缺 | no_causal | 54.962% | 45.200% | 32.946% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 67.572% | 42.914% | 35.748% |
| 双模态残缺 | no_causal | 55.080% | 44.550% | 32.154% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11f | 32.692% | 66.630% | 23.652% |
| 仅图像残缺 | no_causal | 41.060% | 65.982% | 29.322% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11f | 0.948699 | 0.042074 |
| 仅干净图像 | no_causal | 0.971705 | 0.029951 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.947719 | 0.015806 |
| 图像干净、音频残缺 | no_causal | 0.970837 | 0.012118 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.945563 | 0.012599 |
| 双模态残缺 | no_causal | 0.969325 | 0.009611 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11f | 0.946901 | 0.033123 |
| 仅图像残缺 | no_causal | 0.970579 | 0.023333 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11f | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11f | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11f | 98.090% | 97.710% | 97.530% | 97.459% |
| 仅干净音频 | no_causal | 97.750% | 97.710% | 97.720% | 97.709% |
| 图像干净、音频残缺 | control | 97.520% | 97.520% | 97.520% | 97.519% |
| 图像干净、音频残缺 | v11f | 97.520% | 97.520% | 97.520% | 97.519% |
| 图像干净、音频残缺 | no_causal | 97.520% | 97.520% | 97.520% | 97.519% |
| 音频干净、图像残缺 | control | 96.720% | 96.720% | 96.720% | 96.719% |
| 音频干净、图像残缺 | v11f | 96.736% | 96.720% | 96.744% | 96.749% |
| 音频干净、图像残缺 | no_causal | 96.754% | 96.720% | 96.680% | 96.659% |
| 双模态残缺 | control | 96.178% | 96.178% | 96.178% | 96.176% |
| 双模态残缺 | v11f | 96.278% | 96.178% | 96.246% | 96.275% |
| 双模态残缺 | no_causal | 96.296% | 96.178% | 96.268% | 96.293% |
| 仅音频残缺 | control | 82.882% | 82.882% | 82.882% | 82.881% |
| 仅音频残缺 | v11f | 82.986% | 82.882% | 82.946% | 82.905% |
| 仅音频残缺 | no_causal | 82.846% | 82.882% | 82.848% | 82.889% |
| 仅图像残缺 | control | 95.514% | 95.514% | 95.514% | 95.512% |
| 仅图像残缺 | v11f | 95.514% | 95.514% | 95.514% | 95.512% |
| 仅图像残缺 | no_causal | 95.514% | 95.514% | 95.514% | 95.512% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11f | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11f | 94.250% | 94.320% | 94.150% | 94.168% |
| 仅干净图像 | no_causal | 94.550% | 94.320% | 94.610% | 94.498% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11f | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 88.970% | 88.970% | 88.970% | 88.968% |
| 图像干净、音频残缺 | v11f | 88.918% | 88.970% | 88.700% | 88.922% |
| 图像干净、音频残缺 | no_causal | 89.142% | 88.970% | 88.882% | 89.220% |
| 音频干净、图像残缺 | control | 95.968% | 95.968% | 95.968% | 95.966% |
| 音频干净、图像残缺 | v11f | 95.968% | 95.968% | 95.968% | 95.966% |
| 音频干净、图像残缺 | no_causal | 95.968% | 95.968% | 95.968% | 95.966% |
| 双模态残缺 | control | 87.012% | 87.012% | 87.012% | 87.009% |
| 双模态残缺 | v11f | 87.012% | 87.012% | 87.020% | 87.113% |
| 双模态残缺 | no_causal | 87.134% | 87.012% | 87.068% | 87.171% |
| 仅音频残缺 | control | 80.600% | 80.600% | 80.600% | 80.600% |
| 仅音频残缺 | v11f | 80.600% | 80.600% | 80.600% | 80.600% |
| 仅音频残缺 | no_causal | 80.600% | 80.600% | 80.600% | 80.600% |
| 仅图像残缺 | control | 83.518% | 83.518% | 83.518% | 83.513% |
| 仅图像残缺 | v11f | 83.526% | 83.518% | 83.372% | 83.493% |
| 仅图像残缺 | no_causal | 84.174% | 83.518% | 84.330% | 84.258% |

</details>


<details>
<summary>fixed 五类等权宏平均：干预各字段有效 n</summary>

| 实验 | 指标字段（同一计数口径） | 每 family 的 n |
| --- | --- | ---: |
| control | `aud2img_correct_gain`、`aud2img_normal_mse`、`aud2img_ratio` | 10000 |
| control | `aud2img_win_both`、`aud2img_win_wrong`、`aud2img_win_zero` | 10000 |
| control | `aud2img_wrong_damage`、`aud2img_wrong_mse`、`aud2img_zero_mse` | 10000 |
| control | `content_aud_normal_acc`、`content_aud_wrong_acc`、`content_aud_zero_acc` | 10000 |
| control | `content_img_normal_acc`、`content_img_wrong_acc`、`content_img_zero_acc` | 10000 |
| control | `img2aud_correct_gain`、`img2aud_normal_mse`、`img2aud_ratio` | 10000 |
| control | `img2aud_win_both`、`img2aud_win_wrong`、`img2aud_win_zero` | 10000 |
| control | `img2aud_wrong_damage`、`img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| control | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| control | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |
| v11f | `aud2img_correct_gain`、`aud2img_gate`、`aud2img_normal_mse` | 10000 |
| v11f | `aud2img_ratio`、`aud2img_win_both`、`aud2img_win_wrong` | 10000 |
| v11f | `aud2img_win_zero`、`aud2img_wrong_damage`、`aud2img_wrong_mse` | 10000 |
| v11f | `aud2img_zero_mse`、`content_aud_normal_acc`、`content_aud_wrong_acc` | 10000 |
| v11f | `content_aud_zero_acc`、`content_img_normal_acc`、`content_img_wrong_acc` | 10000 |
| v11f | `content_img_zero_acc`、`img2aud_correct_gain`、`img2aud_gate` | 10000 |
| v11f | `img2aud_normal_mse`、`img2aud_ratio`、`img2aud_win_both` | 10000 |
| v11f | `img2aud_win_wrong`、`img2aud_win_zero`、`img2aud_wrong_damage` | 10000 |
| v11f | `img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| v11f | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| v11f | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |
| no_causal | `aud2img_correct_gain`、`aud2img_gate`、`aud2img_normal_mse` | 10000 |
| no_causal | `aud2img_ratio`、`aud2img_win_both`、`aud2img_win_wrong` | 10000 |
| no_causal | `aud2img_win_zero`、`aud2img_wrong_damage`、`aud2img_wrong_mse` | 10000 |
| no_causal | `aud2img_zero_mse`、`content_aud_normal_acc`、`content_aud_wrong_acc` | 10000 |
| no_causal | `content_aud_zero_acc`、`content_img_normal_acc`、`content_img_wrong_acc` | 10000 |
| no_causal | `content_img_zero_acc`、`img2aud_correct_gain`、`img2aud_gate` | 10000 |
| no_causal | `img2aud_normal_mse`、`img2aud_ratio`、`img2aud_win_both` | 10000 |
| no_causal | `img2aud_win_wrong`、`img2aud_win_zero`、`img2aud_wrong_damage` | 10000 |
| no_causal | `img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| no_causal | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| no_causal | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |

</details>


<details>
<summary>random：干预完整指标</summary>

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11f | 0.002052 | 0.002085 | 0.002049 | 0.002040 |
| 仅干净音频 | no_causal | 0.002077 | 0.002085 | 0.002083 | 0.002071 |
| 音频干净、图像残缺 | control | 0.028262 | 0.028262 | 0.028262 | 0.028243 |
| 音频干净、图像残缺 | v11f | 0.027421 | 0.028262 | 0.027437 | 0.027400 |
| 音频干净、图像残缺 | no_causal | 0.027600 | 0.028262 | 0.027664 | 0.027578 |
| 双模态残缺 | control | 0.026184 | 0.026184 | 0.026184 | 0.026159 |
| 双模态残缺 | v11f | 0.025611 | 0.026184 | 0.025631 | 0.025577 |
| 双模态残缺 | no_causal | 0.025777 | 0.026184 | 0.025832 | 0.025745 |
| 仅音频残缺 | control | 0.012209 | 0.012209 | 0.012209 | 0.012191 |
| 仅音频残缺 | v11f | 0.011799 | 0.012209 | 0.011791 | 0.011769 |
| 仅音频残缺 | no_causal | 0.011416 | 0.012209 | 0.011629 | 0.011612 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11f | 0.000033 | -0.000003 | -0.000013 |
| 仅干净音频 | no_causal | 0.000008 | 0.000006 | -0.000006 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.000841 | 0.000016 | -0.000003 |
| 音频干净、图像残缺 | no_causal | 0.000663 | 0.000065 | -0.000003 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000573 | 0.000019 | -0.000011 |
| 双模态残缺 | no_causal | 0.000407 | 0.000055 | -0.000011 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11f | 0.000409 | -0.000008 | -0.000014 |
| 仅音频残缺 | no_causal | 0.000792 | 0.000213 | 0.000211 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11f | 36.260% | 63.900% | 28.100% |
| 仅干净音频 | no_causal | 30.240% | 63.710% | 24.860% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11f | 71.240% | 51.800% | 38.860% |
| 音频干净、图像残缺 | no_causal | 63.980% | 52.830% | 37.240% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 67.280% | 49.940% | 36.370% |
| 双模态残缺 | no_causal | 59.950% | 51.150% | 34.860% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11f | 53.260% | 57.720% | 33.610% |
| 仅音频残缺 | no_causal | 43.950% | 59.510% | 31.000% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11f | 0.544109 | 0.428402 |
| 仅干净音频 | no_causal | 0.754257 | 0.214882 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.571370 | 0.213149 |
| 音频干净、图像残缺 | no_causal | 0.735658 | 0.102322 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.571839 | 0.184788 |
| 双模态残缺 | no_causal | 0.744612 | 0.100567 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11f | 0.545408 | 0.345944 |
| 仅音频残缺 | no_causal | 0.756656 | 0.202794 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11f | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000323 | 0.000300 |
| 图像干净、音频残缺 | control | 0.009929 | 0.009929 | 0.009929 | 0.009922 |
| 图像干净、音频残缺 | v11f | 0.009485 | 0.009929 | 0.009494 | 0.009476 |
| 图像干净、音频残缺 | no_causal | 0.009638 | 0.009929 | 0.009706 | 0.009622 |
| 双模态残缺 | control | 0.009910 | 0.009910 | 0.009910 | 0.009903 |
| 双模态残缺 | v11f | 0.009581 | 0.009910 | 0.009584 | 0.009572 |
| 双模态残缺 | no_causal | 0.009678 | 0.009910 | 0.009715 | 0.009664 |
| 仅图像残缺 | control | 0.000627 | 0.000627 | 0.000627 | 0.000628 |
| 仅图像残缺 | v11f | 0.000630 | 0.000627 | 0.000635 | 0.000631 |
| 仅图像残缺 | no_causal | 0.000631 | 0.000627 | 0.000645 | 0.000640 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11f | -0.000005 | 0.000010 | 7.096e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000027 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.000444 | 0.000009 | -0.000002 |
| 图像干净、音频残缺 | no_causal | 0.000291 | 0.000068 | -0.000008 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000329 | 0.000003 | -0.000002 |
| 双模态残缺 | no_causal | 0.000232 | 0.000037 | -0.000006 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11f | -0.000002 | 0.000006 | 0.000002 |
| 仅图像残缺 | no_causal | -0.000004 | 0.000014 | 0.000009 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11f | 34.920% | 69.880% | 26.600% |
| 仅干净图像 | no_causal | 30.950% | 71.780% | 25.460% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11f | 69.320% | 43.870% | 36.930% |
| 图像干净、音频残缺 | no_causal | 56.320% | 44.900% | 32.520% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 66.230% | 41.860% | 35.290% |
| 双模态残缺 | no_causal | 53.890% | 42.540% | 30.780% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11f | 33.060% | 67.200% | 24.100% |
| 仅图像残缺 | no_causal | 40.670% | 66.550% | 30.010% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11f | 0.948699 | 0.042074 |
| 仅干净图像 | no_causal | 0.971705 | 0.029951 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.947044 | 0.017596 |
| 图像干净、音频残缺 | no_causal | 0.969964 | 0.013374 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.947009 | 0.012509 |
| 双模态残缺 | no_causal | 0.971041 | 0.009562 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11f | 0.946987 | 0.033173 |
| 仅图像残缺 | no_causal | 0.970644 | 0.023380 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11f | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11f | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11f | 98.090% | 97.710% | 97.530% | 97.459% |
| 仅干净音频 | no_causal | 97.750% | 97.710% | 97.720% | 97.709% |
| 图像干净、音频残缺 | control | 97.500% | 97.500% | 97.500% | 97.499% |
| 图像干净、音频残缺 | v11f | 97.500% | 97.500% | 97.500% | 97.499% |
| 图像干净、音频残缺 | no_causal | 97.500% | 97.500% | 97.500% | 97.499% |
| 音频干净、图像残缺 | control | 96.690% | 96.690% | 96.690% | 96.689% |
| 音频干净、图像残缺 | v11f | 96.720% | 96.690% | 96.830% | 96.799% |
| 音频干净、图像残缺 | no_causal | 96.780% | 96.690% | 96.830% | 96.699% |
| 双模态残缺 | control | 96.340% | 96.340% | 96.340% | 96.339% |
| 双模态残缺 | v11f | 96.390% | 96.340% | 96.390% | 96.299% |
| 双模态残缺 | no_causal | 96.380% | 96.340% | 96.370% | 96.399% |
| 仅音频残缺 | control | 84.090% | 84.090% | 84.090% | 84.114% |
| 仅音频残缺 | v11f | 84.250% | 84.090% | 84.120% | 84.104% |
| 仅音频残缺 | no_causal | 84.010% | 84.090% | 84.060% | 84.054% |
| 仅图像残缺 | control | 95.490% | 95.490% | 95.490% | 95.488% |
| 仅图像残缺 | v11f | 95.490% | 95.490% | 95.490% | 95.488% |
| 仅图像残缺 | no_causal | 95.490% | 95.490% | 95.490% | 95.488% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11f | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11f | 94.250% | 94.320% | 94.150% | 94.168% |
| 仅干净图像 | no_causal | 94.550% | 94.320% | 94.610% | 94.498% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11f | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 89.830% | 89.830% | 89.830% | 89.826% |
| 图像干净、音频残缺 | v11f | 89.720% | 89.830% | 89.370% | 89.676% |
| 图像干净、音频残缺 | no_causal | 90.060% | 89.830% | 89.540% | 89.866% |
| 音频干净、图像残缺 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 音频干净、图像残缺 | v11f | 96.000% | 96.000% | 96.000% | 95.998% |
| 音频干净、图像残缺 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 双模态残缺 | control | 87.310% | 87.310% | 87.310% | 87.305% |
| 双模态残缺 | v11f | 87.530% | 87.310% | 87.370% | 87.535% |
| 双模态残缺 | no_causal | 87.670% | 87.310% | 87.490% | 87.605% |
| 仅音频残缺 | control | 81.770% | 81.770% | 81.770% | 81.783% |
| 仅音频残缺 | v11f | 81.770% | 81.770% | 81.770% | 81.783% |
| 仅音频残缺 | no_causal | 81.770% | 81.770% | 81.770% | 81.783% |
| 仅图像残缺 | control | 84.270% | 84.270% | 84.270% | 84.264% |
| 仅图像残缺 | v11f | 84.480% | 84.270% | 84.650% | 84.594% |
| 仅图像残缺 | no_causal | 85.160% | 84.270% | 84.930% | 84.834% |

</details>


<details>
<summary>random：干预各字段有效 n</summary>

| 实验 | 指标字段（同一计数口径） | n |
| --- | --- | ---: |
| control | `aud2img_correct_gain`、`aud2img_normal_mse`、`aud2img_ratio` | 10000 |
| control | `aud2img_win_both`、`aud2img_win_wrong`、`aud2img_win_zero` | 10000 |
| control | `aud2img_wrong_damage`、`aud2img_wrong_mse`、`aud2img_zero_mse` | 10000 |
| control | `content_aud_normal_acc`、`content_aud_wrong_acc`、`content_aud_zero_acc` | 10000 |
| control | `content_img_normal_acc`、`content_img_wrong_acc`、`content_img_zero_acc` | 10000 |
| control | `img2aud_correct_gain`、`img2aud_normal_mse`、`img2aud_ratio` | 10000 |
| control | `img2aud_win_both`、`img2aud_win_wrong`、`img2aud_win_zero` | 10000 |
| control | `img2aud_wrong_damage`、`img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| control | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| control | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |
| v11f | `aud2img_correct_gain`、`aud2img_gate`、`aud2img_normal_mse` | 10000 |
| v11f | `aud2img_ratio`、`aud2img_win_both`、`aud2img_win_wrong` | 10000 |
| v11f | `aud2img_win_zero`、`aud2img_wrong_damage`、`aud2img_wrong_mse` | 10000 |
| v11f | `aud2img_zero_mse`、`content_aud_normal_acc`、`content_aud_wrong_acc` | 10000 |
| v11f | `content_aud_zero_acc`、`content_img_normal_acc`、`content_img_wrong_acc` | 10000 |
| v11f | `content_img_zero_acc`、`img2aud_correct_gain`、`img2aud_gate` | 10000 |
| v11f | `img2aud_normal_mse`、`img2aud_ratio`、`img2aud_win_both` | 10000 |
| v11f | `img2aud_win_wrong`、`img2aud_win_zero`、`img2aud_wrong_damage` | 10000 |
| v11f | `img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| v11f | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| v11f | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |
| no_causal | `aud2img_correct_gain`、`aud2img_gate`、`aud2img_normal_mse` | 10000 |
| no_causal | `aud2img_ratio`、`aud2img_win_both`、`aud2img_win_wrong` | 10000 |
| no_causal | `aud2img_win_zero`、`aud2img_wrong_damage`、`aud2img_wrong_mse` | 10000 |
| no_causal | `aud2img_zero_mse`、`content_aud_normal_acc`、`content_aud_wrong_acc` | 10000 |
| no_causal | `content_aud_zero_acc`、`content_img_normal_acc`、`content_img_wrong_acc` | 10000 |
| no_causal | `content_img_zero_acc`、`img2aud_correct_gain`、`img2aud_gate` | 10000 |
| no_causal | `img2aud_normal_mse`、`img2aud_ratio`、`img2aud_win_both` | 10000 |
| no_causal | `img2aud_win_wrong`、`img2aud_win_zero`、`img2aud_wrong_damage` | 10000 |
| no_causal | `img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| no_causal | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| no_causal | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |

</details>

**逐 family 完整分项**

以下是与主表同源的五组 image/audio family pair，每组内仍按输入模式、实验排列。每组约 10000 个 MNIST 测试条目；音频会复用，旧版有效 n 未逐指标保存。


<details>
<summary>family 1：occlusion/time_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11f | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11f | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965814 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966055 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11f | 98.830% | 0.002052 | 0.985961 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002077 | 0.985706 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 99.140% | 0.009767 | 0.942667 | 0.006178 | 0.834596 |
| 图像干净、音频残缺 | v11f | 99.140% | 0.009767 | 0.942667 | 0.006013 | 0.837068 |
| 图像干净、音频残缺 | no_causal | 99.140% | 0.009767 | 0.942667 | 0.006026 | 0.838227 |
| 音频干净、图像残缺 | control | 99.130% | 0.008792 | 0.948068 | 0.003647 | 0.911187 |
| 音频干净、图像残缺 | v11f | 99.130% | 0.008539 | 0.949380 | 0.003647 | 0.911187 |
| 音频干净、图像残缺 | no_causal | 99.130% | 0.008555 | 0.949363 | 0.003647 | 0.911187 |
| 双模态残缺 | control | 97.850% | 0.008731 | 0.948098 | 0.006377 | 0.827549 |
| 双模态残缺 | v11f | 97.850% | 0.008499 | 0.949366 | 0.006236 | 0.829693 |
| 双模态残缺 | no_causal | 97.850% | 0.008532 | 0.949290 | 0.006245 | 0.830808 |
| 仅音频残缺 | control | 90.500% | 0.009049 | 0.925291 | 0.006279 | 0.831989 |
| 仅音频残缺 | v11f | 90.500% | 0.008828 | 0.928792 | 0.006279 | 0.831989 |
| 仅音频残缺 | no_causal | 90.500% | 0.008767 | 0.929644 | 0.006279 | 0.831989 |
| 仅图像残缺 | control | 89.790% | 0.009160 | 0.945520 | 0.000822 | 0.900165 |
| 仅图像残缺 | v11f | 89.790% | 0.009160 | 0.945520 | 0.000822 | 0.900128 |
| 仅图像残缺 | no_causal | 89.790% | 0.009160 | 0.945520 | 0.000830 | 0.900631 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11f | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11f | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11f | 37.208255 | 0.002052 | 0.002052 | 0.007433 |
| 仅干净音频 | no_causal | 36.902481 | 0.002077 | 0.002077 | 0.007504 |
| 图像干净、音频残缺 | control | 20.777420 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | 20.777420 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.777420 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 23.028585 | 0.098624 | 0.056964 | 0.127488 |
| 音频干净、图像残缺 | v11f | 23.113802 | 0.099672 | 0.055324 | 0.128515 |
| 音频干净、图像残缺 | no_causal | 23.141779 | 0.096682 | 0.055430 | 0.126528 |
| 双模态残缺 | control | 22.973645 | 0.098165 | 0.056573 | 0.126785 |
| 双模态残缺 | v11f | 23.054348 | 0.098544 | 0.055068 | 0.127700 |
| 双模态残缺 | no_causal | 23.066614 | 0.095827 | 0.055280 | 0.126080 |
| 仅音频残缺 | control | 32.892612 | 0.009049 | 0.009049 | 0.018964 |
| 仅音频残缺 | v11f | 32.678100 | 0.008828 | 0.008828 | 0.019251 |
| 仅音频残缺 | no_causal | 32.331695 | 0.008767 | 0.008767 | 0.019499 |
| 仅图像残缺 | control | 22.868519 | 0.102796 | 0.059348 | 0.131669 |
| 仅图像残缺 | v11f | 22.868519 | 0.102796 | 0.059348 | 0.131669 |
| 仅图像残缺 | no_causal | 22.868519 | 0.102796 | 0.059348 | 0.131669 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.015330 | 8.420e-09 | 0.000084 |
| 音频干净、图像残缺 | v11f | 0.015330 | 8.420e-09 | 0.000084 |
| 音频干净、图像残缺 | no_causal | 0.015330 | 8.420e-09 | 0.000084 |
| 双模态残缺 | control | 0.015193 | 8.415e-09 | 0.000084 |
| 双模态残缺 | v11f | 0.015193 | 8.415e-09 | 0.000084 |
| 双模态残缺 | no_causal | 0.015193 | 8.415e-09 | 0.000084 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.015612 | 8.420e-09 | 0.000084 |
| 仅图像残缺 | v11f | 0.015612 | 8.420e-09 | 0.000084 |
| 仅图像残缺 | no_causal | 0.015612 | 8.420e-09 | 0.000084 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11f | 0.000294 | 0.003618 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003556 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.009056 | 0.023760 | 0.004209 | 0.017673 |
| 图像干净、音频残缺 | v11f | 0.008648 | 0.023668 | 0.004209 | 0.017673 |
| 图像干净、音频残缺 | no_causal | 0.008681 | 0.023655 | 0.004209 | 0.017673 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.009616 | 0.025095 | 0.004161 | 0.017356 |
| 双模态残缺 | v11f | 0.009269 | 0.025015 | 0.004161 | 0.017356 |
| 双模态残缺 | no_causal | 0.009292 | 0.025001 | 0.004161 | 0.017356 |
| 仅音频残缺 | control | 0.009326 | 0.024048 | 0.004193 | 0.017643 |
| 仅音频残缺 | v11f | 0.009326 | 0.024048 | 0.004193 | 0.017643 |
| 仅音频残缺 | no_causal | 0.009326 | 0.024048 | 0.004193 | 0.017643 |
| 仅图像残缺 | control | 0.000822 | 0.005739 | N/A | N/A |
| 仅图像残缺 | v11f | 0.000822 | 0.005879 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000830 | 0.005787 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.339353 |
| 干净双模态 | v11f | 97.790% | 96.040% | 0.058851 | 0.339353 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.339353 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.338850 |
| 仅干净图像 | v11f | 96.820% | 94.250% | 0.059386 | 0.338850 |
| 仅干净图像 | no_causal | 96.820% | 94.550% | 0.059386 | 0.338850 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.287092 |
| 仅干净音频 | v11f | 98.090% | 96.000% | 0.045937 | 0.285968 |
| 仅干净音频 | no_causal | 97.750% | 96.000% | 0.046202 | 0.286909 |
| 图像干净、音频残缺 | control | 97.610% | 89.110% | 0.059190 | 0.339690 |
| 图像干净、音频残缺 | v11f | 97.610% | 89.120% | 0.059190 | 0.339690 |
| 图像干净、音频残缺 | no_causal | 97.610% | 89.390% | 0.059190 | 0.339690 |
| 音频干净、图像残缺 | control | 95.920% | 95.920% | 0.060249 | 0.343568 |
| 音频干净、图像残缺 | v11f | 96.310% | 95.920% | 0.059821 | 0.342261 |
| 音频干净、图像残缺 | no_causal | 96.250% | 95.920% | 0.060231 | 0.343409 |
| 双模态残缺 | control | 95.140% | 87.600% | 0.060214 | 0.341794 |
| 双模态残缺 | v11f | 95.510% | 87.660% | 0.059891 | 0.340838 |
| 双模态残缺 | no_causal | 95.720% | 87.880% | 0.060272 | 0.341948 |
| 仅音频残缺 | control | 89.060% | 86.090% | 0.042524 | 0.283295 |
| 仅音频残缺 | v11f | 89.380% | 86.090% | 0.042139 | 0.281966 |
| 仅音频残缺 | no_causal | 89.200% | 86.090% | 0.042147 | 0.282006 |
| 仅图像残缺 | control | 93.750% | 82.120% | 0.059824 | 0.342283 |
| 仅图像残缺 | v11f | 93.750% | 82.090% | 0.059824 | 0.342283 |
| 仅图像残缺 | no_causal | 93.750% | 82.390% | 0.059824 | 0.342283 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11f | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11f | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11f | 0.015600 | 0.068500 | 0.981600 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.994400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.040000 | 0.147100 | 1.000000 |
| 图像干净、音频残缺 | v11f | 0.040400 | 0.146600 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.040700 | 0.148000 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11f | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155700 | 1.000000 |
| 双模态残缺 | control | 0.039900 | 0.146800 | 1.000000 |
| 双模态残缺 | v11f | 0.040200 | 0.146300 | 1.000000 |
| 双模态残缺 | no_causal | 0.040500 | 0.147500 | 1.000000 |
| 仅音频残缺 | control | 0.039800 | 0.146600 | 1.000000 |
| 仅音频残缺 | v11f | 0.039800 | 0.146600 | 1.000000 |
| 仅音频残缺 | no_causal | 0.039800 | 0.146600 | 1.000000 |
| 仅图像残缺 | control | 0.015000 | 0.065400 | 1.000000 |
| 仅图像残缺 | v11f | 0.015500 | 0.065400 | 0.992000 |
| 仅图像残缺 | no_causal | 0.015200 | 0.066100 | 0.999100 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11f | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.100% |
| 图像干净、音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 78.200% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.200% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 77.800% |
| 双模态残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 77.900% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 77.800% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 77.900% |
| 仅音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 77.900% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 77.900% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 78.300% |
| 仅图像残缺 | v11f | 0.015600 | 0.070500 | 0.987900 | 78.300% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 78.300% |

**本 family 干预**

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11f | 0.002052 | 0.002085 | 0.002049 | 0.002040 |
| 仅干净音频 | no_causal | 0.002077 | 0.002085 | 0.002083 | 0.002071 |
| 音频干净、图像残缺 | control | 0.056964 | 0.056964 | 0.056964 | 0.056959 |
| 音频干净、图像残缺 | v11f | 0.055324 | 0.056964 | 0.055433 | 0.055327 |
| 音频干净、图像残缺 | no_causal | 0.055430 | 0.056964 | 0.055692 | 0.055443 |
| 双模态残缺 | control | 0.056573 | 0.056573 | 0.056573 | 0.056550 |
| 双模态残缺 | v11f | 0.055068 | 0.056573 | 0.055169 | 0.055025 |
| 双模态残缺 | no_causal | 0.055280 | 0.056573 | 0.055493 | 0.055241 |
| 仅音频残缺 | control | 0.009049 | 0.009049 | 0.009049 | 0.009050 |
| 仅音频残缺 | v11f | 0.008828 | 0.009049 | 0.008771 | 0.008754 |
| 仅音频残缺 | no_causal | 0.008767 | 0.009049 | 0.008850 | 0.008830 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11f | 0.000033 | -0.000003 | -0.000013 |
| 仅干净音频 | no_causal | 0.000008 | 0.000006 | -0.000006 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.001640 | 0.000109 | 0.000006 |
| 音频干净、图像残缺 | no_causal | 0.001534 | 0.000263 | 0.000016 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.001506 | 0.000102 | -0.000021 |
| 双模态残缺 | no_causal | 0.001293 | 0.000213 | -0.000019 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11f | 0.000220 | -0.000057 | -0.000075 |
| 仅音频残缺 | no_causal | 0.000282 | 0.000083 | 0.000062 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11f | 36.260% | 63.900% | 28.100% |
| 仅干净音频 | no_causal | 30.240% | 63.710% | 24.860% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11f | 64.990% | 51.540% | 36.180% |
| 音频干净、图像残缺 | no_causal | 61.880% | 53.800% | 37.490% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 65.620% | 52.230% | 37.120% |
| 双模态残缺 | no_causal | 62.120% | 52.840% | 38.160% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11f | 49.770% | 58.410% | 32.900% |
| 仅音频残缺 | no_causal | 39.820% | 62.940% | 30.270% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11f | 0.544109 | 0.428402 |
| 仅干净音频 | no_causal | 0.754257 | 0.214882 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.568725 | 0.101606 |
| 音频干净、图像残缺 | no_causal | 0.681011 | 0.048088 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.567053 | 0.083491 |
| 双模态残缺 | no_causal | 0.687675 | 0.042373 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11f | 0.544560 | 0.361307 |
| 仅音频残缺 | no_causal | 0.756216 | 0.196419 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11f | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000323 | 0.000300 |
| 图像干净、音频残缺 | control | 0.009056 | 0.009056 | 0.009056 | 0.009057 |
| 图像干净、音频残缺 | v11f | 0.008648 | 0.009056 | 0.008663 | 0.008644 |
| 图像干净、音频残缺 | no_causal | 0.008681 | 0.009056 | 0.008740 | 0.008677 |
| 双模态残缺 | control | 0.009616 | 0.009616 | 0.009616 | 0.009618 |
| 双模态残缺 | v11f | 0.009269 | 0.009616 | 0.009283 | 0.009273 |
| 双模态残缺 | no_causal | 0.009292 | 0.009616 | 0.009351 | 0.009297 |
| 仅图像残缺 | control | 0.000822 | 0.000822 | 0.000822 | 0.000821 |
| 仅图像残缺 | v11f | 0.000822 | 0.000822 | 0.000828 | 0.000824 |
| 仅图像残缺 | no_causal | 0.000830 | 0.000822 | 0.000847 | 0.000843 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11f | -0.000005 | 0.000010 | 7.096e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000027 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.000408 | 0.000015 | -0.000005 |
| 图像干净、音频残缺 | no_causal | 0.000375 | 0.000059 | -0.000005 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000347 | 0.000014 | 0.000002 |
| 双模态残缺 | no_causal | 0.000323 | 0.000059 | 0.000003 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11f | -2.093e-07 | 0.000007 | 0.000003 |
| 仅图像残缺 | no_causal | -0.000008 | 0.000017 | 0.000014 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11f | 34.920% | 69.880% | 26.600% |
| 仅干净图像 | no_causal | 30.950% | 71.780% | 25.460% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11f | 51.890% | 34.500% | 28.480% |
| 图像干净、音频残缺 | no_causal | 45.380% | 35.690% | 27.000% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 53.260% | 35.270% | 29.020% |
| 双模态残缺 | no_causal | 47.420% | 36.020% | 27.580% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11f | 35.490% | 65.200% | 24.840% |
| 仅图像残缺 | no_causal | 39.260% | 65.820% | 28.660% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11f | 0.948699 | 0.042074 |
| 仅干净图像 | no_causal | 0.971705 | 0.029951 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.957848 | 0.012158 |
| 图像干净、音频残缺 | no_causal | 0.982372 | 0.010174 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.956833 | 0.009901 |
| 双模态残缺 | no_causal | 0.981466 | 0.008286 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11f | 0.947956 | 0.034471 |
| 仅图像残缺 | no_causal | 0.971215 | 0.024319 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11f | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11f | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11f | 98.090% | 97.710% | 97.530% | 97.459% |
| 仅干净音频 | no_causal | 97.750% | 97.710% | 97.720% | 97.709% |
| 图像干净、音频残缺 | control | 97.610% | 97.610% | 97.610% | 97.609% |
| 图像干净、音频残缺 | v11f | 97.610% | 97.610% | 97.610% | 97.609% |
| 图像干净、音频残缺 | no_causal | 97.610% | 97.610% | 97.610% | 97.609% |
| 音频干净、图像残缺 | control | 95.920% | 95.920% | 95.920% | 95.918% |
| 音频干净、图像残缺 | v11f | 96.310% | 95.920% | 96.220% | 96.289% |
| 音频干净、图像残缺 | no_causal | 96.250% | 95.920% | 96.220% | 96.208% |
| 双模态残缺 | control | 95.140% | 95.140% | 95.140% | 95.138% |
| 双模态残缺 | v11f | 95.510% | 95.140% | 95.430% | 95.518% |
| 双模态残缺 | no_causal | 95.720% | 95.140% | 95.520% | 95.638% |
| 仅音频残缺 | control | 89.060% | 89.060% | 89.060% | 89.056% |
| 仅音频残缺 | v11f | 89.380% | 89.060% | 89.240% | 89.156% |
| 仅音频残缺 | no_causal | 89.200% | 89.060% | 89.120% | 89.286% |
| 仅图像残缺 | control | 93.750% | 93.750% | 93.750% | 93.747% |
| 仅图像残缺 | v11f | 93.750% | 93.750% | 93.750% | 93.747% |
| 仅图像残缺 | no_causal | 93.750% | 93.750% | 93.750% | 93.747% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11f | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11f | 94.250% | 94.320% | 94.150% | 94.168% |
| 仅干净图像 | no_causal | 94.550% | 94.320% | 94.610% | 94.498% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11f | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 89.110% | 89.110% | 89.110% | 89.116% |
| 图像干净、音频残缺 | v11f | 89.120% | 89.110% | 88.950% | 89.116% |
| 图像干净、音频残缺 | no_causal | 89.390% | 89.110% | 88.970% | 89.506% |
| 音频干净、图像残缺 | control | 95.920% | 95.920% | 95.920% | 95.918% |
| 音频干净、图像残缺 | v11f | 95.920% | 95.920% | 95.920% | 95.918% |
| 音频干净、图像残缺 | no_causal | 95.920% | 95.920% | 95.920% | 95.918% |
| 双模态残缺 | control | 87.600% | 87.600% | 87.600% | 87.595% |
| 双模态残缺 | v11f | 87.660% | 87.600% | 87.770% | 87.775% |
| 双模态残缺 | no_causal | 87.880% | 87.600% | 87.820% | 87.935% |
| 仅音频残缺 | control | 86.090% | 86.090% | 86.090% | 86.094% |
| 仅音频残缺 | v11f | 86.090% | 86.090% | 86.090% | 86.094% |
| 仅音频残缺 | no_causal | 86.090% | 86.090% | 86.090% | 86.094% |
| 仅图像残缺 | control | 82.120% | 82.120% | 82.120% | 82.123% |
| 仅图像残缺 | v11f | 82.090% | 82.120% | 82.040% | 82.193% |
| 仅图像残缺 | no_causal | 82.390% | 82.120% | 82.810% | 82.753% |

</details>


<details>
<summary>family 2：pixel_delete/freq_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11f | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11f | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965814 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966055 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11f | 98.830% | 0.002052 | 0.985961 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002077 | 0.985706 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 98.970% | 0.009764 | 0.942683 | 0.004813 | 0.860118 |
| 图像干净、音频残缺 | v11f | 98.970% | 0.009764 | 0.942683 | 0.004670 | 0.862269 |
| 图像干净、音频残缺 | no_causal | 98.970% | 0.009764 | 0.942683 | 0.004712 | 0.861972 |
| 音频干净、图像残缺 | control | 99.190% | 0.004433 | 0.974220 | 0.003645 | 0.911259 |
| 音频干净、图像残缺 | v11f | 99.190% | 0.004187 | 0.975631 | 0.003645 | 0.911259 |
| 音频干净、图像残缺 | no_causal | 99.190% | 0.004304 | 0.974952 | 0.003645 | 0.911259 |
| 双模态残缺 | control | 98.110% | 0.004434 | 0.974162 | 0.004848 | 0.860361 |
| 双模态残缺 | v11f | 98.110% | 0.004231 | 0.975340 | 0.004737 | 0.861993 |
| 双模态残缺 | no_causal | 98.110% | 0.004324 | 0.974803 | 0.004768 | 0.861782 |
| 仅音频残缺 | control | 92.480% | 0.007755 | 0.941128 | 0.004865 | 0.858251 |
| 仅音频残缺 | v11f | 92.480% | 0.007590 | 0.943099 | 0.004865 | 0.858251 |
| 仅音频残缺 | no_causal | 92.480% | 0.007642 | 0.942386 | 0.004865 | 0.858251 |
| 仅图像残缺 | control | 95.010% | 0.004457 | 0.974026 | 0.000483 | 0.940544 |
| 仅图像残缺 | v11f | 95.010% | 0.004457 | 0.974026 | 0.000486 | 0.939795 |
| 仅图像残缺 | no_causal | 95.010% | 0.004457 | 0.974026 | 0.000487 | 0.940886 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11f | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11f | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11f | 37.208255 | 0.002052 | 0.002052 | 0.007433 |
| 仅干净音频 | no_causal | 36.902481 | 0.002077 | 0.002077 | 0.007504 |
| 图像干净、音频残缺 | control | 20.778868 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | 20.778868 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.778868 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 24.098065 | 0.036370 | 0.011070 | 0.033332 |
| 音频干净、图像残缺 | v11f | 24.343867 | 0.035350 | 0.010457 | 0.032731 |
| 音频干净、图像残缺 | no_causal | 24.221211 | 0.035473 | 0.010749 | 0.032930 |
| 双模态残缺 | control | 24.090888 | 0.036313 | 0.011067 | 0.033377 |
| 双模态残缺 | v11f | 24.294379 | 0.034700 | 0.010559 | 0.032794 |
| 双模态残缺 | no_causal | 24.196031 | 0.035168 | 0.010792 | 0.033004 |
| 仅音频残缺 | control | 31.764547 | 0.007755 | 0.007755 | 0.017812 |
| 仅音频残缺 | v11f | 31.659390 | 0.007590 | 0.007590 | 0.018027 |
| 仅音频残缺 | no_causal | 31.359554 | 0.007642 | 0.007642 | 0.018181 |
| 仅图像残缺 | control | 24.077221 | 0.037197 | 0.011131 | 0.033512 |
| 仅图像残缺 | v11f | 24.077221 | 0.037197 | 0.011131 | 0.033512 |
| 仅图像残缺 | no_causal | 24.077221 | 0.037197 | 0.011131 | 0.033512 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.028126 | 8.145e-09 | 0.000081 |
| 音频干净、图像残缺 | v11f | 0.028126 | 8.145e-09 | 0.000081 |
| 音频干净、图像残缺 | no_causal | 0.028126 | 8.145e-09 | 0.000081 |
| 双模态残缺 | control | 0.028026 | 8.144e-09 | 0.000081 |
| 双模态残缺 | v11f | 0.028026 | 8.144e-09 | 0.000081 |
| 双模态残缺 | no_causal | 0.028026 | 8.144e-09 | 0.000081 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.028769 | 8.145e-09 | 0.000081 |
| 仅图像残缺 | v11f | 0.028769 | 8.145e-09 | 0.000081 |
| 仅图像残缺 | no_causal | 0.028769 | 8.145e-09 | 0.000081 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11f | 0.000294 | 0.003618 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003556 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.006317 | 0.021067 | 0.003784 | 0.016163 |
| 图像干净、音频残缺 | v11f | 0.005964 | 0.021141 | 0.003784 | 0.016163 |
| 图像干净、音频残缺 | no_causal | 0.006067 | 0.021217 | 0.003784 | 0.016163 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.006427 | 0.021316 | 0.003767 | 0.016127 |
| 双模态残缺 | v11f | 0.006155 | 0.021325 | 0.003767 | 0.016127 |
| 双模态残缺 | no_causal | 0.006232 | 0.021382 | 0.003767 | 0.016127 |
| 仅音频残缺 | control | 0.006428 | 0.021270 | 0.003795 | 0.016189 |
| 仅音频残缺 | v11f | 0.006428 | 0.021270 | 0.003795 | 0.016189 |
| 仅音频残缺 | no_causal | 0.006428 | 0.021270 | 0.003795 | 0.016189 |
| 仅图像残缺 | control | 0.000483 | 0.004556 | N/A | N/A |
| 仅图像残缺 | v11f | 0.000486 | 0.004691 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000487 | 0.004582 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.339185 |
| 干净双模态 | v11f | 97.790% | 96.040% | 0.058851 | 0.339185 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.339185 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.339899 |
| 仅干净图像 | v11f | 96.820% | 94.250% | 0.059386 | 0.339899 |
| 仅干净图像 | no_causal | 96.820% | 94.550% | 0.059386 | 0.339899 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.289794 |
| 仅干净音频 | v11f | 98.090% | 96.000% | 0.045937 | 0.288606 |
| 仅干净音频 | no_causal | 97.750% | 96.000% | 0.046202 | 0.289566 |
| 图像干净、音频残缺 | control | 97.510% | 89.560% | 0.059181 | 0.340259 |
| 图像干净、音频残缺 | v11f | 97.510% | 89.900% | 0.059181 | 0.340259 |
| 图像干净、音频残缺 | no_causal | 97.510% | 89.860% | 0.059181 | 0.340259 |
| 音频干净、图像残缺 | control | 97.380% | 96.010% | 0.064142 | 0.353424 |
| 音频干净、图像残缺 | v11f | 97.430% | 96.010% | 0.064136 | 0.353445 |
| 音频干净、图像残缺 | no_causal | 97.300% | 96.010% | 0.064098 | 0.353322 |
| 双模态残缺 | control | 97.330% | 88.800% | 0.064141 | 0.354006 |
| 双模态残缺 | v11f | 97.430% | 88.950% | 0.064324 | 0.354542 |
| 双模态残缺 | no_causal | 97.350% | 88.860% | 0.064205 | 0.354192 |
| 仅音频残缺 | control | 91.530% | 87.110% | 0.043328 | 0.281725 |
| 仅音频残缺 | v11f | 91.590% | 87.110% | 0.042996 | 0.280627 |
| 仅音频残缺 | no_causal | 91.760% | 87.110% | 0.043098 | 0.281030 |
| 仅图像残缺 | control | 97.110% | 87.490% | 0.064040 | 0.353135 |
| 仅图像残缺 | v11f | 97.110% | 87.710% | 0.064040 | 0.353135 |
| 仅图像残缺 | no_causal | 97.110% | 88.510% | 0.064040 | 0.353135 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11f | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11f | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11f | 0.015600 | 0.068500 | 0.981600 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.994400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.041700 | 0.151500 | 1.000000 |
| 图像干净、音频残缺 | v11f | 0.042100 | 0.150300 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.042500 | 0.152200 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11f | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155700 | 1.000000 |
| 双模态残缺 | control | 0.041900 | 0.151700 | 1.000000 |
| 双模态残缺 | v11f | 0.042100 | 0.151000 | 1.000000 |
| 双模态残缺 | no_causal | 0.042500 | 0.152500 | 1.000000 |
| 仅音频残缺 | control | 0.041700 | 0.151400 | 1.000000 |
| 仅音频残缺 | v11f | 0.041700 | 0.151400 | 1.000000 |
| 仅音频残缺 | no_causal | 0.041700 | 0.151400 | 1.000000 |
| 仅图像残缺 | control | 0.015500 | 0.067600 | 1.000000 |
| 仅图像残缺 | v11f | 0.015900 | 0.067500 | 0.996000 |
| 仅图像残缺 | no_causal | 0.015600 | 0.068100 | 0.999700 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11f | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 图像干净、音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 80.200% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 双模态残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.000% |
| 仅音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 80.000% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 80.000% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 79.700% |
| 仅图像残缺 | v11f | 0.015600 | 0.070500 | 0.987900 | 79.600% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 79.600% |

**本 family 干预**

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11f | 0.002052 | 0.002085 | 0.002049 | 0.002040 |
| 仅干净音频 | no_causal | 0.002077 | 0.002085 | 0.002083 | 0.002071 |
| 音频干净、图像残缺 | control | 0.011070 | 0.011070 | 0.011070 | 0.011070 |
| 音频干净、图像残缺 | v11f | 0.010457 | 0.011070 | 0.010468 | 0.010456 |
| 音频干净、图像残缺 | no_causal | 0.010749 | 0.011070 | 0.010765 | 0.010749 |
| 双模态残缺 | control | 0.011067 | 0.011067 | 0.011067 | 0.011066 |
| 双模态残缺 | v11f | 0.010559 | 0.011067 | 0.010569 | 0.010562 |
| 双模态残缺 | no_causal | 0.010792 | 0.011067 | 0.010805 | 0.010791 |
| 仅音频残缺 | control | 0.007755 | 0.007755 | 0.007755 | 0.007756 |
| 仅音频残缺 | v11f | 0.007590 | 0.007755 | 0.007581 | 0.007555 |
| 仅音频残缺 | no_causal | 0.007642 | 0.007755 | 0.007655 | 0.007628 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11f | 0.000033 | -0.000003 | -0.000013 |
| 仅干净音频 | no_causal | 0.000008 | 0.000006 | -0.000006 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.000613 | 0.000011 | -0.000001 |
| 音频干净、图像残缺 | no_causal | 0.000321 | 0.000016 | 2.260e-07 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000508 | 0.000010 | 0.000003 |
| 双模态残缺 | no_causal | 0.000275 | 0.000013 | 8.449e-08 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11f | 0.000165 | -0.000009 | -0.000036 |
| 仅音频残缺 | no_causal | 0.000113 | 0.000014 | -0.000015 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11f | 36.260% | 63.900% | 28.100% |
| 仅干净音频 | no_causal | 30.240% | 63.710% | 24.860% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11f | 85.150% | 52.070% | 45.780% |
| 音频干净、图像残缺 | no_causal | 76.500% | 52.510% | 43.050% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 84.450% | 51.570% | 44.630% |
| 双模态残缺 | no_causal | 76.890% | 51.840% | 42.590% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11f | 52.890% | 57.740% | 35.400% |
| 仅音频残缺 | no_causal | 41.420% | 57.910% | 30.570% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11f | 0.544109 | 0.428402 |
| 仅干净音频 | no_causal | 0.754257 | 0.214882 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.565929 | 0.269952 |
| 音频干净、图像残缺 | no_causal | 0.756193 | 0.131157 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.565042 | 0.217219 |
| 双模态残缺 | no_causal | 0.760049 | 0.109722 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11f | 0.544292 | 0.346065 |
| 仅音频残缺 | no_causal | 0.755258 | 0.179777 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11f | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000323 | 0.000300 |
| 图像干净、音频残缺 | control | 0.006317 | 0.006317 | 0.006317 | 0.006319 |
| 图像干净、音频残缺 | v11f | 0.005964 | 0.006317 | 0.005976 | 0.005964 |
| 图像干净、音频残缺 | no_causal | 0.006067 | 0.006317 | 0.006137 | 0.006069 |
| 双模态残缺 | control | 0.006427 | 0.006427 | 0.006427 | 0.006427 |
| 双模态残缺 | v11f | 0.006155 | 0.006427 | 0.006155 | 0.006155 |
| 双模态残缺 | no_causal | 0.006232 | 0.006427 | 0.006259 | 0.006231 |
| 仅图像残缺 | control | 0.000483 | 0.000483 | 0.000483 | 0.000483 |
| 仅图像残缺 | v11f | 0.000486 | 0.000483 | 0.000491 | 0.000487 |
| 仅图像残缺 | no_causal | 0.000487 | 0.000483 | 0.000500 | 0.000494 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11f | -0.000005 | 0.000010 | 7.096e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000027 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.000353 | 0.000012 | -9.003e-07 |
| 图像干净、音频残缺 | no_causal | 0.000250 | 0.000070 | 8.597e-07 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000272 | 2.698e-08 | 2.595e-07 |
| 双模态残缺 | no_causal | 0.000195 | 0.000027 | -0.000001 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11f | -0.000003 | 0.000005 | 0.000001 |
| 仅图像残缺 | no_causal | -0.000005 | 0.000013 | 0.000006 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11f | 34.920% | 69.880% | 26.600% |
| 仅干净图像 | no_causal | 30.950% | 71.780% | 25.460% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11f | 77.890% | 50.550% | 43.000% |
| 图像干净、音频残缺 | no_causal | 56.690% | 52.620% | 35.780% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 80.130% | 49.440% | 42.860% |
| 双模态残缺 | no_causal | 56.650% | 51.300% | 34.320% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11f | 26.700% | 67.360% | 19.680% |
| 仅图像残缺 | no_causal | 38.470% | 66.820% | 28.010% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11f | 0.948699 | 0.042074 |
| 仅干净图像 | no_causal | 0.971705 | 0.029951 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.945328 | 0.029003 |
| 图像干净、音频残缺 | no_causal | 0.968854 | 0.021473 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.943183 | 0.022046 |
| 双模态残缺 | no_causal | 0.967508 | 0.016314 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11f | 0.946640 | 0.031803 |
| 仅图像残缺 | no_causal | 0.970448 | 0.022658 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11f | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11f | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11f | 98.090% | 97.710% | 97.530% | 97.459% |
| 仅干净音频 | no_causal | 97.750% | 97.710% | 97.720% | 97.709% |
| 图像干净、音频残缺 | control | 97.510% | 97.510% | 97.510% | 97.509% |
| 图像干净、音频残缺 | v11f | 97.510% | 97.510% | 97.510% | 97.509% |
| 图像干净、音频残缺 | no_causal | 97.510% | 97.510% | 97.510% | 97.509% |
| 音频干净、图像残缺 | control | 97.380% | 97.380% | 97.380% | 97.379% |
| 音频干净、图像残缺 | v11f | 97.430% | 97.380% | 97.420% | 97.359% |
| 音频干净、图像残缺 | no_causal | 97.300% | 97.380% | 97.310% | 97.349% |
| 双模态残缺 | control | 97.330% | 97.330% | 97.330% | 97.329% |
| 双模态残缺 | v11f | 97.430% | 97.330% | 97.340% | 97.419% |
| 双模态残缺 | no_causal | 97.350% | 97.330% | 97.390% | 97.299% |
| 仅音频残缺 | control | 91.530% | 91.530% | 91.530% | 91.527% |
| 仅音频残缺 | v11f | 91.590% | 91.530% | 91.600% | 91.687% |
| 仅音频残缺 | no_causal | 91.760% | 91.530% | 91.570% | 91.657% |
| 仅图像残缺 | control | 97.110% | 97.110% | 97.110% | 97.109% |
| 仅图像残缺 | v11f | 97.110% | 97.110% | 97.110% | 97.109% |
| 仅图像残缺 | no_causal | 97.110% | 97.110% | 97.110% | 97.109% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11f | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11f | 94.250% | 94.320% | 94.150% | 94.168% |
| 仅干净图像 | no_causal | 94.550% | 94.320% | 94.610% | 94.498% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11f | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 89.560% | 89.560% | 89.560% | 89.556% |
| 图像干净、音频残缺 | v11f | 89.900% | 89.560% | 89.760% | 89.966% |
| 图像干净、音频残缺 | no_causal | 89.860% | 89.560% | 89.880% | 89.766% |
| 音频干净、图像残缺 | control | 96.010% | 96.010% | 96.010% | 96.008% |
| 音频干净、图像残缺 | v11f | 96.010% | 96.010% | 96.010% | 96.008% |
| 音频干净、图像残缺 | no_causal | 96.010% | 96.010% | 96.010% | 96.008% |
| 双模态残缺 | control | 88.800% | 88.800% | 88.800% | 88.796% |
| 双模态残缺 | v11f | 88.950% | 88.800% | 88.970% | 88.976% |
| 双模态残缺 | no_causal | 88.860% | 88.800% | 88.620% | 88.745% |
| 仅音频残缺 | control | 87.110% | 87.110% | 87.110% | 87.115% |
| 仅音频残缺 | v11f | 87.110% | 87.110% | 87.110% | 87.115% |
| 仅音频残缺 | no_causal | 87.110% | 87.110% | 87.110% | 87.115% |
| 仅图像残缺 | control | 87.490% | 87.490% | 87.490% | 87.485% |
| 仅图像残缺 | v11f | 87.710% | 87.490% | 87.640% | 87.565% |
| 仅图像残缺 | no_causal | 88.510% | 87.490% | 88.550% | 88.365% |

</details>


<details>
<summary>family 3：mask_vertical/feature_dropout</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11f | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11f | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965814 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966055 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11f | 98.830% | 0.002052 | 0.985961 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002077 | 0.985706 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 99.310% | 0.009733 | 0.942750 | 0.003736 | 0.904357 |
| 图像干净、音频残缺 | v11f | 99.310% | 0.009733 | 0.942750 | 0.003659 | 0.905507 |
| 图像干净、音频残缺 | no_causal | 99.310% | 0.009733 | 0.942750 | 0.003679 | 0.905141 |
| 音频干净、图像残缺 | control | 99.130% | 0.007666 | 0.957471 | 0.003646 | 0.911234 |
| 音频干净、图像残缺 | v11f | 99.130% | 0.007544 | 0.958104 | 0.003646 | 0.911234 |
| 音频干净、图像残缺 | no_causal | 99.130% | 0.007616 | 0.957709 | 0.003646 | 0.911234 |
| 双模态残缺 | control | 99.150% | 0.007630 | 0.957875 | 0.003737 | 0.904205 |
| 双模态残缺 | v11f | 99.150% | 0.007509 | 0.958518 | 0.003675 | 0.905153 |
| 双模态残缺 | no_causal | 99.150% | 0.007581 | 0.958149 | 0.003692 | 0.904819 |
| 仅音频残缺 | control | 97.230% | 0.002978 | 0.978846 | 0.003734 | 0.904240 |
| 仅音频残缺 | v11f | 97.230% | 0.002927 | 0.979480 | 0.003734 | 0.904240 |
| 仅音频残缺 | no_causal | 97.230% | 0.002961 | 0.979054 | 0.003734 | 0.904240 |
| 仅图像残缺 | control | 94.530% | 0.007853 | 0.956388 | 0.000504 | 0.939148 |
| 仅图像残缺 | v11f | 94.530% | 0.007853 | 0.956388 | 0.000507 | 0.938745 |
| 仅图像残缺 | no_causal | 94.530% | 0.007853 | 0.956388 | 0.000508 | 0.939476 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11f | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11f | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11f | 37.208255 | 0.002052 | 0.002052 | 0.007433 |
| 仅干净音频 | no_causal | 36.902481 | 0.002077 | 0.002077 | 0.007504 |
| 图像干净、音频残缺 | control | 20.793251 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | 20.793251 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.793251 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 24.345296 | 0.030801 | 0.019513 | 0.045871 |
| 音频干净、图像残缺 | v11f | 24.379481 | 0.031158 | 0.019202 | 0.046712 |
| 音频干净、图像残缺 | no_causal | 24.393627 | 0.030745 | 0.019385 | 0.046058 |
| 双模态残缺 | control | 24.379809 | 0.030731 | 0.019422 | 0.045644 |
| 双模态残缺 | v11f | 24.410835 | 0.031003 | 0.019114 | 0.046463 |
| 双模态残缺 | no_causal | 24.425158 | 0.030559 | 0.019296 | 0.045779 |
| 仅音频残缺 | control | 36.157700 | 0.002978 | 0.002978 | 0.009158 |
| 仅音频残缺 | v11f | 35.874698 | 0.002927 | 0.002927 | 0.009414 |
| 仅音频残缺 | no_causal | 35.509912 | 0.002961 | 0.002961 | 0.009552 |
| 仅图像残缺 | control | 24.244018 | 0.031509 | 0.019989 | 0.046766 |
| 仅图像残缺 | v11f | 24.244018 | 0.031509 | 0.019989 | 0.046766 |
| 仅图像残缺 | no_causal | 24.244018 | 0.031509 | 0.019989 | 0.046766 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.019340 | 7.760e-09 | 0.000078 |
| 音频干净、图像残缺 | v11f | 0.019340 | 7.760e-09 | 0.000078 |
| 音频干净、图像残缺 | no_causal | 0.019340 | 7.760e-09 | 0.000078 |
| 双模态残缺 | control | 0.019353 | 7.757e-09 | 0.000078 |
| 双模态残缺 | v11f | 0.019353 | 7.757e-09 | 0.000078 |
| 双模态残缺 | no_causal | 0.019353 | 7.757e-09 | 0.000078 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.019562 | 7.760e-09 | 0.000078 |
| 仅图像残缺 | v11f | 0.019562 | 7.760e-09 | 0.000078 |
| 仅图像残缺 | no_causal | 0.019562 | 7.760e-09 | 0.000078 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11f | 0.000294 | 0.003618 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003556 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.003793 | 0.015930 | 0.003698 | 0.015719 |
| 图像干净、音频残缺 | v11f | 0.003601 | 0.015715 | 0.003698 | 0.015719 |
| 图像干净、音频残缺 | no_causal | 0.003649 | 0.015749 | 0.003698 | 0.015719 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.003804 | 0.015958 | 0.003692 | 0.015713 |
| 双模态残缺 | v11f | 0.003649 | 0.015779 | 0.003692 | 0.015713 |
| 双模态残缺 | no_causal | 0.003692 | 0.015808 | 0.003692 | 0.015713 |
| 仅音频残缺 | control | 0.003791 | 0.015931 | 0.003695 | 0.015719 |
| 仅音频残缺 | v11f | 0.003791 | 0.015931 | 0.003695 | 0.015719 |
| 仅音频残缺 | no_causal | 0.003791 | 0.015931 | 0.003695 | 0.015719 |
| 仅图像残缺 | control | 0.000504 | 0.004496 | N/A | N/A |
| 仅图像残缺 | v11f | 0.000507 | 0.004626 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000508 | 0.004529 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.337367 |
| 干净双模态 | v11f | 97.790% | 96.040% | 0.058851 | 0.337367 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.337367 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.341075 |
| 仅干净图像 | v11f | 96.820% | 94.250% | 0.059386 | 0.341075 |
| 仅干净图像 | no_causal | 96.820% | 94.550% | 0.059386 | 0.341075 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.285967 |
| 仅干净音频 | v11f | 98.090% | 96.000% | 0.045937 | 0.284773 |
| 仅干净音频 | no_causal | 97.750% | 96.000% | 0.046202 | 0.285741 |
| 图像干净、音频残缺 | control | 97.610% | 94.530% | 0.058961 | 0.337150 |
| 图像干净、音频残缺 | v11f | 97.610% | 94.610% | 0.058961 | 0.337150 |
| 图像干净、音频残缺 | no_causal | 97.610% | 94.580% | 0.058961 | 0.337150 |
| 音频干净、图像残缺 | control | 96.740% | 96.000% | 0.060439 | 0.342839 |
| 音频干净、图像残缺 | v11f | 96.670% | 96.000% | 0.060156 | 0.341991 |
| 音频干净、图像残缺 | no_causal | 96.610% | 96.000% | 0.060345 | 0.342559 |
| 双模态残缺 | control | 96.600% | 94.680% | 0.060428 | 0.343442 |
| 双模态残缺 | v11f | 96.520% | 94.650% | 0.060173 | 0.342805 |
| 双模态残缺 | no_causal | 96.650% | 94.590% | 0.060389 | 0.343373 |
| 仅音频残缺 | control | 96.850% | 93.950% | 0.045526 | 0.283439 |
| 仅音频残缺 | v11f | 96.850% | 93.950% | 0.045236 | 0.282494 |
| 仅音频残缺 | no_causal | 96.850% | 93.950% | 0.045386 | 0.283086 |
| 仅图像残缺 | control | 95.430% | 88.100% | 0.060211 | 0.342225 |
| 仅图像残缺 | v11f | 95.430% | 88.070% | 0.060211 | 0.342225 |
| 仅图像残缺 | no_causal | 95.430% | 88.690% | 0.060211 | 0.342225 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11f | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11f | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11f | 0.015600 | 0.068500 | 0.981600 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.994400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.042100 | 0.153900 | 1.000000 |
| 图像干净、音频残缺 | v11f | 0.042100 | 0.153300 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.042200 | 0.153900 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155600 | 1.000000 |
| 音频干净、图像残缺 | v11f | 0.043200 | 0.155600 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155600 | 1.000000 |
| 双模态残缺 | control | 0.042100 | 0.153900 | 1.000000 |
| 双模态残缺 | v11f | 0.042100 | 0.153400 | 1.000000 |
| 双模态残缺 | no_causal | 0.042200 | 0.153900 | 1.000000 |
| 仅音频残缺 | control | 0.042100 | 0.153900 | 1.000000 |
| 仅音频残缺 | v11f | 0.042100 | 0.153900 | 1.000000 |
| 仅音频残缺 | no_causal | 0.042100 | 0.153900 | 1.000000 |
| 仅图像残缺 | control | 0.015300 | 0.067500 | 1.000000 |
| 仅图像残缺 | v11f | 0.015700 | 0.067400 | 0.992400 |
| 仅图像残缺 | no_causal | 0.015400 | 0.068000 | 0.999200 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11f | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 图像干净、音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 双模态残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 仅音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 79.800% |
| 仅图像残缺 | v11f | 0.015600 | 0.070500 | 0.987900 | 79.800% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 79.800% |

**本 family 干预**

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11f | 0.002052 | 0.002085 | 0.002049 | 0.002040 |
| 仅干净音频 | no_causal | 0.002077 | 0.002085 | 0.002083 | 0.002071 |
| 音频干净、图像残缺 | control | 0.019513 | 0.019513 | 0.019513 | 0.019510 |
| 音频干净、图像残缺 | v11f | 0.019202 | 0.019513 | 0.019244 | 0.019198 |
| 音频干净、图像残缺 | no_causal | 0.019385 | 0.019513 | 0.019441 | 0.019385 |
| 双模态残缺 | control | 0.019422 | 0.019422 | 0.019422 | 0.019419 |
| 双模态残缺 | v11f | 0.019114 | 0.019422 | 0.019147 | 0.019107 |
| 双模态残缺 | no_causal | 0.019296 | 0.019422 | 0.019343 | 0.019291 |
| 仅音频残缺 | control | 0.002978 | 0.002978 | 0.002978 | 0.002979 |
| 仅音频残缺 | v11f | 0.002927 | 0.002978 | 0.002919 | 0.002909 |
| 仅音频残缺 | no_causal | 0.002961 | 0.002978 | 0.002968 | 0.002952 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11f | 0.000033 | -0.000003 | -0.000013 |
| 仅干净音频 | no_causal | 0.000008 | 0.000006 | -0.000006 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.000310 | 0.000041 | -0.000001 |
| 音频干净、图像残缺 | no_causal | 0.000128 | 0.000056 | 0.000004 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000307 | 0.000033 | -0.000004 |
| 双模态残缺 | no_causal | 0.000126 | 0.000047 | -0.000002 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11f | 0.000051 | -0.000008 | -0.000019 |
| 仅音频残缺 | no_causal | 0.000017 | 0.000007 | -0.000009 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11f | 36.260% | 63.900% | 28.100% |
| 仅干净音频 | no_causal | 30.240% | 63.710% | 24.860% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11f | 58.120% | 52.670% | 33.950% |
| 音频干净、图像残缺 | no_causal | 52.780% | 51.480% | 33.130% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 58.190% | 51.220% | 33.180% |
| 双模态残缺 | no_causal | 53.330% | 50.800% | 33.500% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11f | 40.280% | 60.730% | 29.690% |
| 仅音频残缺 | no_causal | 29.750% | 59.990% | 24.140% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11f | 0.544109 | 0.428402 |
| 仅干净音频 | no_causal | 0.754257 | 0.214882 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.570936 | 0.235123 |
| 音频干净、图像残缺 | no_causal | 0.772989 | 0.113092 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.570151 | 0.214600 |
| 双模态残缺 | no_causal | 0.777758 | 0.104092 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11f | 0.543773 | 0.389137 |
| 仅音频残缺 | no_causal | 0.756356 | 0.197014 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11f | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000323 | 0.000300 |
| 图像干净、音频残缺 | control | 0.003793 | 0.003793 | 0.003793 | 0.003793 |
| 图像干净、音频残缺 | v11f | 0.003601 | 0.003793 | 0.003592 | 0.003601 |
| 图像干净、音频残缺 | no_causal | 0.003649 | 0.003793 | 0.003643 | 0.003650 |
| 双模态残缺 | control | 0.003804 | 0.003804 | 0.003804 | 0.003804 |
| 双模态残缺 | v11f | 0.003649 | 0.003804 | 0.003641 | 0.003650 |
| 双模态残缺 | no_causal | 0.003692 | 0.003804 | 0.003687 | 0.003693 |
| 仅图像残缺 | control | 0.000504 | 0.000504 | 0.000504 | 0.000504 |
| 仅图像残缺 | v11f | 0.000507 | 0.000504 | 0.000513 | 0.000508 |
| 仅图像残缺 | no_causal | 0.000508 | 0.000504 | 0.000523 | 0.000515 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11f | -0.000005 | 0.000010 | 7.096e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000027 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.000192 | -0.000008 | -2.068e-07 |
| 图像干净、音频残缺 | no_causal | 0.000143 | -0.000007 | -6.272e-08 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000154 | -0.000008 | 6.887e-08 |
| 双模态残缺 | no_causal | 0.000111 | -0.000005 | 1.123e-07 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11f | -0.000003 | 0.000006 | 0.000001 |
| 仅图像残缺 | no_causal | -0.000004 | 0.000015 | 0.000007 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11f | 34.920% | 69.880% | 26.600% |
| 仅干净图像 | no_causal | 30.950% | 71.780% | 25.460% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11f | 88.090% | 48.780% | 44.560% |
| 图像干净、音频残缺 | no_causal | 77.930% | 49.780% | 42.130% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 88.610% | 48.170% | 44.300% |
| 双模态残缺 | no_causal | 77.520% | 49.840% | 41.690% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11f | 33.150% | 68.240% | 24.430% |
| 仅图像残缺 | no_causal | 36.390% | 68.020% | 26.810% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11f | 0.948699 | 0.042074 |
| 仅干净图像 | no_causal | 0.971705 | 0.029951 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.942745 | 0.027457 |
| 图像干净、音频残缺 | no_causal | 0.964047 | 0.020253 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.940665 | 0.023052 |
| 双模态残缺 | no_causal | 0.962559 | 0.016920 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11f | 0.946637 | 0.035245 |
| 仅图像残缺 | no_causal | 0.970442 | 0.024955 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11f | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11f | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11f | 98.090% | 97.710% | 97.530% | 97.459% |
| 仅干净音频 | no_causal | 97.750% | 97.710% | 97.720% | 97.709% |
| 图像干净、音频残缺 | control | 97.610% | 97.610% | 97.610% | 97.609% |
| 图像干净、音频残缺 | v11f | 97.610% | 97.610% | 97.610% | 97.609% |
| 图像干净、音频残缺 | no_causal | 97.610% | 97.610% | 97.610% | 97.609% |
| 音频干净、图像残缺 | control | 96.740% | 96.740% | 96.740% | 96.739% |
| 音频干净、图像残缺 | v11f | 96.670% | 96.740% | 96.600% | 96.619% |
| 音频干净、图像残缺 | no_causal | 96.610% | 96.740% | 96.600% | 96.499% |
| 双模态残缺 | control | 96.600% | 96.600% | 96.600% | 96.599% |
| 双模态残缺 | v11f | 96.520% | 96.600% | 96.570% | 96.529% |
| 双模态残缺 | no_causal | 96.650% | 96.600% | 96.640% | 96.599% |
| 仅音频残缺 | control | 96.850% | 96.850% | 96.850% | 96.849% |
| 仅音频残缺 | v11f | 96.850% | 96.850% | 96.880% | 96.869% |
| 仅音频残缺 | no_causal | 96.850% | 96.850% | 96.780% | 96.839% |
| 仅图像残缺 | control | 95.430% | 95.430% | 95.430% | 95.428% |
| 仅图像残缺 | v11f | 95.430% | 95.430% | 95.430% | 95.428% |
| 仅图像残缺 | no_causal | 95.430% | 95.430% | 95.430% | 95.428% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11f | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11f | 94.250% | 94.320% | 94.150% | 94.168% |
| 仅干净图像 | no_causal | 94.550% | 94.320% | 94.610% | 94.498% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11f | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 94.530% | 94.530% | 94.530% | 94.528% |
| 图像干净、音频残缺 | v11f | 94.610% | 94.530% | 94.550% | 94.678% |
| 图像干净、音频残缺 | no_causal | 94.580% | 94.530% | 94.690% | 94.648% |
| 音频干净、图像残缺 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 音频干净、图像残缺 | v11f | 96.000% | 96.000% | 96.000% | 95.998% |
| 音频干净、图像残缺 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 双模态残缺 | control | 94.680% | 94.680% | 94.680% | 94.678% |
| 双模态残缺 | v11f | 94.650% | 94.680% | 94.760% | 94.778% |
| 双模态残缺 | no_causal | 94.590% | 94.680% | 94.610% | 94.728% |
| 仅音频残缺 | control | 93.950% | 93.950% | 93.950% | 93.948% |
| 仅音频残缺 | v11f | 93.950% | 93.950% | 93.950% | 93.948% |
| 仅音频残缺 | no_causal | 93.950% | 93.950% | 93.950% | 93.948% |
| 仅图像残缺 | control | 88.100% | 88.100% | 88.100% | 88.095% |
| 仅图像残缺 | v11f | 88.070% | 88.100% | 88.050% | 88.075% |
| 仅图像残缺 | no_causal | 88.690% | 88.100% | 88.700% | 88.675% |

</details>


<details>
<summary>family 4：mask_horizontal/partial_temporal</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11f | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11f | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965814 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966055 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11f | 98.830% | 0.002052 | 0.985961 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002077 | 0.985706 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 98.070% | 0.009949 | 0.942169 | 0.011784 | 0.538387 |
| 图像干净、音频残缺 | v11f | 98.070% | 0.009949 | 0.942169 | 0.011342 | 0.543572 |
| 图像干净、音频残缺 | no_causal | 98.070% | 0.009949 | 0.942169 | 0.011537 | 0.544788 |
| 音频干净、图像残缺 | control | 99.180% | 0.011375 | 0.933588 | 0.003646 | 0.911236 |
| 音频干净、图像残缺 | v11f | 99.180% | 0.011234 | 0.934197 | 0.003646 | 0.911236 |
| 音频干净、图像残缺 | no_causal | 99.180% | 0.011310 | 0.933820 | 0.003646 | 0.911236 |
| 双模态残缺 | control | 93.960% | 0.011546 | 0.932764 | 0.011946 | 0.537024 |
| 双模态残缺 | v11f | 93.960% | 0.011501 | 0.933089 | 0.011620 | 0.540568 |
| 双模态残缺 | no_causal | 93.960% | 0.011547 | 0.932774 | 0.011783 | 0.541861 |
| 仅音频残缺 | control | 45.880% | 0.042710 | 0.569261 | 0.012585 | 0.528776 |
| 仅音频残缺 | v11f | 45.880% | 0.040867 | 0.604162 | 0.012585 | 0.528776 |
| 仅音频残缺 | no_causal | 45.880% | 0.038628 | 0.639875 | 0.012585 | 0.528776 |
| 仅图像残缺 | control | 93.020% | 0.011626 | 0.932092 | 0.000617 | 0.921198 |
| 仅图像残缺 | v11f | 93.020% | 0.011626 | 0.932092 | 0.000617 | 0.921070 |
| 仅图像残缺 | no_causal | 93.020% | 0.011626 | 0.932092 | 0.000621 | 0.921928 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11f | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11f | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11f | 37.208255 | 0.002052 | 0.002052 | 0.007433 |
| 仅干净音频 | no_causal | 36.902481 | 0.002077 | 0.002077 | 0.007504 |
| 图像干净、音频残缺 | control | 20.701504 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | 20.701504 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.701504 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 20.224451 | 0.041941 | 0.028956 | 0.066252 |
| 音频干净、图像残缺 | v11f | 20.239753 | 0.042981 | 0.028596 | 0.067434 |
| 音频干净、图像残缺 | no_causal | 20.241025 | 0.042240 | 0.028790 | 0.066206 |
| 双模态残缺 | control | 20.167815 | 0.041696 | 0.029389 | 0.067064 |
| 双模态残缺 | v11f | 20.152106 | 0.041368 | 0.029275 | 0.068773 |
| 双模态残缺 | no_causal | 20.119359 | 0.041005 | 0.029393 | 0.069511 |
| 仅音频残缺 | control | 17.523188 | 0.042710 | 0.042710 | 0.069514 |
| 仅音频残缺 | v11f | 17.651848 | 0.040867 | 0.040867 | 0.069858 |
| 仅音频残缺 | no_causal | 17.683809 | 0.038628 | 0.038628 | 0.071835 |
| 仅图像残缺 | control | 20.117320 | 0.042756 | 0.029594 | 0.067630 |
| 仅图像残缺 | v11f | 20.117320 | 0.042756 | 0.029594 | 0.067630 |
| 仅图像残缺 | no_causal | 20.117320 | 0.042756 | 0.029594 | 0.067630 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.024606 | 7.982e-09 | 0.000080 |
| 音频干净、图像残缺 | v11f | 0.024606 | 7.982e-09 | 0.000080 |
| 音频干净、图像残缺 | no_causal | 0.024606 | 7.982e-09 | 0.000080 |
| 双模态残缺 | control | 0.023821 | 7.982e-09 | 0.000080 |
| 双模态残缺 | v11f | 0.023821 | 7.982e-09 | 0.000080 |
| 双模态残缺 | no_causal | 0.023821 | 7.982e-09 | 0.000080 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.025068 | 7.982e-09 | 0.000080 |
| 仅图像残缺 | v11f | 0.025068 | 7.982e-09 | 0.000080 |
| 仅图像残缺 | no_causal | 0.025068 | 7.982e-09 | 0.000080 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11f | 0.000294 | 0.003618 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003556 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.028559 | 0.074284 | 0.000306 | 0.001634 |
| 图像干净、音频残缺 | v11f | 0.027472 | 0.074668 | 0.000306 | 0.001634 |
| 图像干净、音频残缺 | no_causal | 0.027952 | 0.074875 | 0.000306 | 0.001634 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.028952 | 0.074811 | 0.000311 | 0.001645 |
| 双模态残缺 | v11f | 0.028149 | 0.075041 | 0.000311 | 0.001645 |
| 双模态残缺 | no_causal | 0.028550 | 0.075221 | 0.000311 | 0.001645 |
| 仅音频残缺 | control | 0.030529 | 0.075829 | 0.000308 | 0.001635 |
| 仅音频残缺 | v11f | 0.030529 | 0.075829 | 0.000308 | 0.001635 |
| 仅音频残缺 | no_causal | 0.030529 | 0.075829 | 0.000308 | 0.001635 |
| 仅图像残缺 | control | 0.000617 | 0.005049 | N/A | N/A |
| 仅图像残缺 | v11f | 0.000617 | 0.005171 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000621 | 0.005073 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.339376 |
| 干净双模态 | v11f | 97.790% | 96.040% | 0.058851 | 0.339376 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.339376 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.340113 |
| 仅干净图像 | v11f | 96.820% | 94.250% | 0.059386 | 0.340113 |
| 仅干净图像 | no_causal | 96.820% | 94.550% | 0.059386 | 0.340113 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.291384 |
| 仅干净音频 | v11f | 98.090% | 96.000% | 0.045937 | 0.290242 |
| 仅干净音频 | no_causal | 97.750% | 96.000% | 0.046202 | 0.291208 |
| 图像干净、音频残缺 | control | 97.040% | 76.880% | 0.060324 | 0.342538 |
| 图像干净、音频残缺 | v11f | 97.040% | 76.110% | 0.060324 | 0.342538 |
| 图像干净、音频残缺 | no_causal | 97.040% | 76.980% | 0.060324 | 0.342538 |
| 音频干净、图像残缺 | control | 96.280% | 95.960% | 0.057461 | 0.335875 |
| 音频干净、图像残缺 | v11f | 96.050% | 95.960% | 0.056749 | 0.333754 |
| 音频干净、图像残缺 | no_causal | 96.200% | 95.960% | 0.057120 | 0.334814 |
| 双模态残缺 | control | 94.510% | 69.490% | 0.057604 | 0.336153 |
| 双模态残缺 | v11f | 94.650% | 69.220% | 0.057454 | 0.335711 |
| 双模态残缺 | no_causal | 94.570% | 69.930% | 0.057447 | 0.335686 |
| 仅音频残缺 | control | 40.770% | 41.520% | 0.023301 | 0.188507 |
| 仅音频残缺 | v11f | 40.670% | 41.520% | 0.022398 | 0.184300 |
| 仅音频残缺 | no_causal | 40.270% | 41.520% | 0.021084 | 0.178191 |
| 仅图像残缺 | control | 94.140% | 84.400% | 0.057135 | 0.334808 |
| 仅图像残缺 | v11f | 94.140% | 84.310% | 0.057135 | 0.334808 |
| 仅图像残缺 | no_causal | 94.140% | 84.930% | 0.057135 | 0.334808 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11f | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11f | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11f | 0.015600 | 0.068500 | 0.981600 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.994400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.037500 | 0.131100 | 1.000000 |
| 图像干净、音频残缺 | v11f | 0.038400 | 0.128600 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.039400 | 0.133700 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11f | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155700 | 1.000000 |
| 双模态残缺 | control | 0.037300 | 0.130500 | 1.000000 |
| 双模态残缺 | v11f | 0.038000 | 0.128800 | 1.000000 |
| 双模态残缺 | no_causal | 0.038800 | 0.132900 | 1.000000 |
| 仅音频残缺 | control | 0.035400 | 0.126800 | 1.000000 |
| 仅音频残缺 | v11f | 0.035400 | 0.126800 | 1.000000 |
| 仅音频残缺 | no_causal | 0.035400 | 0.126800 | 1.000000 |
| 仅图像残缺 | control | 0.015400 | 0.067400 | 1.000000 |
| 仅图像残缺 | v11f | 0.015800 | 0.067300 | 0.996500 |
| 仅图像残缺 | no_causal | 0.015600 | 0.068000 | 0.999800 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11f | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 70.200% |
| 图像干净、音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 70.400% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 70.500% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 70.000% |
| 双模态残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 70.200% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 70.300% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 68.600% |
| 仅音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 68.600% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 68.600% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 79.000% |
| 仅图像残缺 | v11f | 0.015600 | 0.070500 | 0.987900 | 78.900% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 78.900% |

**本 family 干预**

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11f | 0.002052 | 0.002085 | 0.002049 | 0.002040 |
| 仅干净音频 | no_causal | 0.002077 | 0.002085 | 0.002083 | 0.002071 |
| 音频干净、图像残缺 | control | 0.028956 | 0.028956 | 0.028956 | 0.028954 |
| 音频干净、图像残缺 | v11f | 0.028596 | 0.028956 | 0.028648 | 0.028599 |
| 音频干净、图像残缺 | no_causal | 0.028790 | 0.028956 | 0.028864 | 0.028782 |
| 双模态残缺 | control | 0.029389 | 0.029389 | 0.029389 | 0.029388 |
| 双模态残缺 | v11f | 0.029275 | 0.029389 | 0.029259 | 0.029256 |
| 双模态残缺 | no_causal | 0.029393 | 0.029389 | 0.029410 | 0.029366 |
| 仅音频残缺 | control | 0.042710 | 0.042710 | 0.042710 | 0.042704 |
| 仅音频残缺 | v11f | 0.040867 | 0.042710 | 0.040883 | 0.040916 |
| 仅音频残缺 | no_causal | 0.038628 | 0.042710 | 0.039730 | 0.039679 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11f | 0.000033 | -0.000003 | -0.000013 |
| 仅干净音频 | no_causal | 0.000008 | 0.000006 | -0.000006 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.000360 | 0.000052 | 0.000005 |
| 音频干净、图像残缺 | no_causal | 0.000166 | 0.000074 | -0.000006 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000113 | -0.000017 | -0.000018 |
| 双模态残缺 | no_causal | -0.000004 | 0.000017 | -0.000025 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11f | 0.001842 | 0.000016 | 0.000053 |
| 仅音频残缺 | no_causal | 0.004082 | 0.001103 | 0.001056 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11f | 36.260% | 63.900% | 28.100% |
| 仅干净音频 | no_causal | 30.240% | 63.710% | 24.860% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11f | 58.220% | 53.650% | 33.550% |
| 音频干净、图像残缺 | no_causal | 53.250% | 53.060% | 31.510% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 52.380% | 42.890% | 26.300% |
| 双模态残缺 | no_causal | 48.720% | 44.030% | 26.910% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11f | 84.000% | 50.100% | 39.090% |
| 仅音频残缺 | no_causal | 81.180% | 61.030% | 47.470% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11f | 0.544109 | 0.428402 |
| 仅干净音频 | no_causal | 0.754257 | 0.214882 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.571198 | 0.229067 |
| 音频干净、图像残缺 | no_causal | 0.746949 | 0.110235 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.569736 | 0.110445 |
| 双模态残缺 | no_causal | 0.750899 | 0.119191 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11f | 0.551502 | 0.207795 |
| 仅音频残缺 | no_causal | 0.761585 | 0.243473 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11f | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000323 | 0.000300 |
| 图像干净、音频残缺 | control | 0.028559 | 0.028559 | 0.028559 | 0.028559 |
| 图像干净、音频残缺 | v11f | 0.027472 | 0.028559 | 0.027520 | 0.027467 |
| 图像干净、音频残缺 | no_causal | 0.027952 | 0.028559 | 0.028196 | 0.027941 |
| 双模态残缺 | control | 0.028952 | 0.028952 | 0.028952 | 0.028952 |
| 双模态残缺 | v11f | 0.028149 | 0.028952 | 0.028175 | 0.028149 |
| 双模态残缺 | no_causal | 0.028550 | 0.028952 | 0.028695 | 0.028544 |
| 仅图像残缺 | control | 0.000617 | 0.000617 | 0.000617 | 0.000617 |
| 仅图像残缺 | v11f | 0.000617 | 0.000617 | 0.000622 | 0.000619 |
| 仅图像残缺 | no_causal | 0.000621 | 0.000617 | 0.000635 | 0.000629 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11f | -0.000005 | 0.000010 | 7.096e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000027 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.001087 | 0.000049 | -0.000005 |
| 图像干净、音频残缺 | no_causal | 0.000607 | 0.000245 | -0.000010 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000802 | 0.000026 | -7.426e-07 |
| 双模态残缺 | no_causal | 0.000402 | 0.000145 | -0.000005 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11f | -6.373e-07 | 0.000005 | 0.000001 |
| 仅图像残缺 | no_causal | -0.000004 | 0.000014 | 0.000008 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11f | 34.920% | 69.880% | 26.600% |
| 仅干净图像 | no_causal | 30.950% | 71.780% | 25.460% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11f | 74.970% | 53.040% | 42.030% |
| 图像干净、音频残缺 | no_causal | 60.230% | 56.030% | 38.620% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 73.880% | 52.700% | 40.640% |
| 双模态残缺 | no_causal | 59.480% | 56.210% | 37.990% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11f | 37.350% | 68.260% | 28.180% |
| 仅图像残缺 | no_causal | 41.960% | 67.790% | 30.750% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11f | 0.948699 | 0.042074 |
| 仅干净图像 | no_causal | 0.971705 | 0.029951 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.936677 | 0.002647 |
| 图像干净、音频残缺 | no_causal | 0.957263 | 0.002027 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.933061 | 0.002044 |
| 双模态残缺 | no_causal | 0.954380 | 0.001559 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11f | 0.946648 | 0.033230 |
| 仅图像残缺 | no_causal | 0.970425 | 0.023237 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11f | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11f | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11f | 98.090% | 97.710% | 97.530% | 97.459% |
| 仅干净音频 | no_causal | 97.750% | 97.710% | 97.720% | 97.709% |
| 图像干净、音频残缺 | control | 97.040% | 97.040% | 97.040% | 97.039% |
| 图像干净、音频残缺 | v11f | 97.040% | 97.040% | 97.040% | 97.039% |
| 图像干净、音频残缺 | no_causal | 97.040% | 97.040% | 97.040% | 97.039% |
| 音频干净、图像残缺 | control | 96.280% | 96.280% | 96.280% | 96.279% |
| 音频干净、图像残缺 | v11f | 96.050% | 96.280% | 96.180% | 96.279% |
| 音频干净、图像残缺 | no_causal | 96.200% | 96.280% | 95.950% | 96.088% |
| 双模态残缺 | control | 94.510% | 94.510% | 94.510% | 94.508% |
| 双模态残缺 | v11f | 94.650% | 94.510% | 94.620% | 94.608% |
| 双模态残缺 | no_causal | 94.570% | 94.510% | 94.540% | 94.638% |
| 仅音频残缺 | control | 40.770% | 40.770% | 40.770% | 40.776% |
| 仅音频残缺 | v11f | 40.670% | 40.770% | 40.700% | 40.526% |
| 仅音频残缺 | no_causal | 40.270% | 40.770% | 40.470% | 40.366% |
| 仅图像残缺 | control | 94.140% | 94.140% | 94.140% | 94.138% |
| 仅图像残缺 | v11f | 94.140% | 94.140% | 94.140% | 94.138% |
| 仅图像残缺 | no_causal | 94.140% | 94.140% | 94.140% | 94.138% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11f | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11f | 94.250% | 94.320% | 94.150% | 94.168% |
| 仅干净图像 | no_causal | 94.550% | 94.320% | 94.610% | 94.498% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11f | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 76.880% | 76.880% | 76.880% | 76.871% |
| 图像干净、音频残缺 | v11f | 76.110% | 76.880% | 75.360% | 75.950% |
| 图像干净、音频残缺 | no_causal | 76.980% | 76.880% | 76.070% | 77.211% |
| 音频干净、图像残缺 | control | 95.960% | 95.960% | 95.960% | 95.958% |
| 音频干净、图像残缺 | v11f | 95.960% | 95.960% | 95.960% | 95.958% |
| 音频干净、图像残缺 | no_causal | 95.960% | 95.960% | 95.960% | 95.958% |
| 双模态残缺 | control | 69.490% | 69.490% | 69.490% | 69.488% |
| 双模态残缺 | v11f | 69.220% | 69.490% | 69.080% | 69.408% |
| 双模态残缺 | no_causal | 69.930% | 69.490% | 69.800% | 69.998% |
| 仅音频残缺 | control | 41.520% | 41.520% | 41.520% | 41.517% |
| 仅音频残缺 | v11f | 41.520% | 41.520% | 41.520% | 41.517% |
| 仅音频残缺 | no_causal | 41.520% | 41.520% | 41.520% | 41.517% |
| 仅图像残缺 | control | 84.400% | 84.400% | 84.400% | 84.394% |
| 仅图像残缺 | v11f | 84.310% | 84.400% | 84.070% | 84.434% |
| 仅图像残缺 | no_causal | 84.930% | 84.400% | 85.070% | 85.024% |

</details>


<details>
<summary>family 5：salt_mask/time_freq_block</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11f | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11f | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965814 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966055 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11f | 98.830% | 0.002052 | 0.985961 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002077 | 0.985706 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 99.180% | 0.009738 | 0.942714 | 0.003975 | 0.900494 |
| 图像干净、音频残缺 | v11f | 99.180% | 0.009738 | 0.942714 | 0.003922 | 0.901131 |
| 图像干净、音频残缺 | no_causal | 99.180% | 0.009738 | 0.942714 | 0.003935 | 0.901065 |
| 音频干净、图像残缺 | control | 99.090% | 0.005243 | 0.968180 | 0.003650 | 0.911121 |
| 音频干净、图像残缺 | v11f | 99.090% | 0.005090 | 0.968950 | 0.003650 | 0.911121 |
| 音频干净、图像残缺 | no_causal | 99.090% | 0.005193 | 0.968342 | 0.003650 | 0.911121 |
| 双模态残缺 | control | 98.610% | 0.005257 | 0.968044 | 0.003972 | 0.900263 |
| 双模态残缺 | v11f | 98.610% | 0.005111 | 0.968809 | 0.003937 | 0.900695 |
| 双模态残缺 | no_causal | 98.610% | 0.005207 | 0.968237 | 0.003946 | 0.900620 |
| 仅音频残缺 | control | 96.770% | 0.003567 | 0.973989 | 0.003986 | 0.900188 |
| 仅音频残缺 | v11f | 96.770% | 0.003501 | 0.974718 | 0.003986 | 0.900188 |
| 仅音频残缺 | no_causal | 96.770% | 0.003535 | 0.974355 | 0.003986 | 0.900188 |
| 仅图像残缺 | control | 87.790% | 0.005307 | 0.967732 | 0.000917 | 0.882853 |
| 仅图像残缺 | v11f | 87.790% | 0.005307 | 0.967732 | 0.000919 | 0.882155 |
| 仅图像残缺 | no_causal | 87.790% | 0.005307 | 0.967732 | 0.000921 | 0.883812 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11f | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11f | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11f | 37.208255 | 0.002052 | 0.002052 | 0.007433 |
| 仅干净音频 | no_causal | 36.902481 | 0.002077 | 0.002077 | 0.007504 |
| 图像干净、音频残缺 | control | 20.790630 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | 20.790630 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.790630 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 23.272686 | 0.066434 | 0.013088 | 0.038475 |
| 音频干净、图像残缺 | v11f | 23.396656 | 0.069767 | 0.012706 | 0.038328 |
| 音频干净、图像残缺 | no_causal | 23.310246 | 0.068901 | 0.012961 | 0.038573 |
| 双模态残缺 | control | 23.267845 | 0.066524 | 0.013111 | 0.038469 |
| 双模态残缺 | v11f | 23.386343 | 0.068938 | 0.012745 | 0.038267 |
| 双模态残缺 | no_causal | 23.305749 | 0.068468 | 0.012984 | 0.038526 |
| 仅音频残缺 | control | 36.133221 | 0.003567 | 0.003567 | 0.009855 |
| 仅音频残缺 | v11f | 35.854368 | 0.003501 | 0.003501 | 0.010101 |
| 仅音频残缺 | no_causal | 35.517563 | 0.003535 | 0.003535 | 0.010208 |
| 仅图像残缺 | control | 23.216327 | 0.067368 | 0.013248 | 0.038833 |
| 仅图像残缺 | v11f | 23.216327 | 0.067368 | 0.013248 | 0.038833 |
| 仅图像残缺 | no_causal | 23.216327 | 0.067368 | 0.013248 | 0.038833 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.066257 | 8.146e-09 | 0.000081 |
| 音频干净、图像残缺 | v11f | 0.066257 | 8.146e-09 | 0.000081 |
| 音频干净、图像残缺 | no_causal | 0.066257 | 8.146e-09 | 0.000081 |
| 双模态残缺 | control | 0.066359 | 8.145e-09 | 0.000081 |
| 双模态残缺 | v11f | 0.066359 | 8.145e-09 | 0.000081 |
| 双模态残缺 | no_causal | 0.066359 | 8.145e-09 | 0.000081 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.067160 | 8.146e-09 | 0.000081 |
| 仅图像残缺 | v11f | 0.067160 | 8.146e-09 | 0.000081 |
| 仅图像残缺 | no_causal | 0.067160 | 8.146e-09 | 0.000081 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11f | 0.000294 | 0.003618 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003556 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.005272 | 0.017887 | 0.003719 | 0.015995 |
| 图像干净、音频残缺 | v11f | 0.004952 | 0.017898 | 0.003719 | 0.015995 |
| 图像干净、音频残缺 | no_causal | 0.005029 | 0.017954 | 0.003719 | 0.015995 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.005077 | 0.017518 | 0.003754 | 0.016087 |
| 双模态残缺 | v11f | 0.004860 | 0.017497 | 0.003754 | 0.016087 |
| 双模态残缺 | no_causal | 0.004917 | 0.017546 | 0.003754 | 0.016087 |
| 仅音频残缺 | control | 0.005321 | 0.017980 | 0.003722 | 0.016004 |
| 仅音频残缺 | v11f | 0.005321 | 0.017980 | 0.003722 | 0.016004 |
| 仅音频残缺 | no_causal | 0.005321 | 0.017980 | 0.003722 | 0.016004 |
| 仅图像残缺 | control | 0.000917 | 0.006634 | N/A | N/A |
| 仅图像残缺 | v11f | 0.000919 | 0.006778 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000921 | 0.006633 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.338103 |
| 干净双模态 | v11f | 97.790% | 96.040% | 0.058851 | 0.338103 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.338103 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.340177 |
| 仅干净图像 | v11f | 96.820% | 94.250% | 0.059386 | 0.340177 |
| 仅干净图像 | no_causal | 96.820% | 94.550% | 0.059386 | 0.340177 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.283510 |
| 仅干净音频 | v11f | 98.090% | 96.000% | 0.045937 | 0.282308 |
| 仅干净音频 | no_causal | 97.750% | 96.000% | 0.046202 | 0.283260 |
| 图像干净、音频残缺 | control | 97.830% | 94.770% | 0.058946 | 0.337707 |
| 图像干净、音频残缺 | v11f | 97.830% | 94.850% | 0.058946 | 0.337707 |
| 图像干净、音频残缺 | no_causal | 97.830% | 94.900% | 0.058946 | 0.337707 |
| 音频干净、图像残缺 | control | 97.280% | 95.950% | 0.063072 | 0.350291 |
| 音频干净、图像残缺 | v11f | 97.220% | 95.950% | 0.062515 | 0.348688 |
| 音频干净、图像残缺 | no_causal | 97.410% | 95.950% | 0.062610 | 0.348938 |
| 双模态残缺 | control | 97.310% | 94.490% | 0.063027 | 0.352854 |
| 双模态残缺 | v11f | 97.280% | 94.580% | 0.062620 | 0.351693 |
| 双模态残缺 | no_causal | 97.190% | 94.410% | 0.062646 | 0.351765 |
| 仅音频残缺 | control | 96.200% | 94.330% | 0.045583 | 0.288516 |
| 仅音频残缺 | v11f | 96.440% | 94.330% | 0.045238 | 0.287343 |
| 仅音频残缺 | no_causal | 96.150% | 94.330% | 0.045453 | 0.288063 |
| 仅图像残缺 | control | 97.140% | 75.480% | 0.062985 | 0.350024 |
| 仅图像残缺 | v11f | 97.140% | 75.450% | 0.062985 | 0.350024 |
| 仅图像残缺 | no_causal | 97.140% | 76.350% | 0.062985 | 0.350024 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11f | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11f | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11f | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11f | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11f | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11f | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11f | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11f | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11f | 0.015600 | 0.068500 | 0.981600 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.994400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11f | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.042700 | 0.154400 | 1.000000 |
| 图像干净、音频残缺 | v11f | 0.042800 | 0.154000 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.043000 | 0.154700 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11f | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155700 | 1.000000 |
| 双模态残缺 | control | 0.042800 | 0.154600 | 1.000000 |
| 双模态残缺 | v11f | 0.042900 | 0.154400 | 1.000000 |
| 双模态残缺 | no_causal | 0.043000 | 0.154800 | 1.000000 |
| 仅音频残缺 | control | 0.042700 | 0.154400 | 1.000000 |
| 仅音频残缺 | v11f | 0.042700 | 0.154400 | 1.000000 |
| 仅音频残缺 | no_causal | 0.042700 | 0.154400 | 1.000000 |
| 仅图像残缺 | control | 0.015200 | 0.063500 | 0.999500 |
| 仅图像残缺 | v11f | 0.015600 | 0.063500 | 0.996600 |
| 仅图像残缺 | no_causal | 0.015300 | 0.064200 | 0.998700 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11f | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.200% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 图像干净、音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 双模态残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.500% |
| 仅音频残缺 | v11f | 0.040600 | 0.152100 | 1.000000 | 81.500% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.500% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 76.700% |
| 仅图像残缺 | v11f | 0.015600 | 0.070500 | 0.987900 | 76.600% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 76.600% |

**本 family 干预**

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11f | 0.002052 | 0.002085 | 0.002049 | 0.002040 |
| 仅干净音频 | no_causal | 0.002077 | 0.002085 | 0.002083 | 0.002071 |
| 音频干净、图像残缺 | control | 0.013088 | 0.013088 | 0.013088 | 0.013088 |
| 音频干净、图像残缺 | v11f | 0.012706 | 0.013088 | 0.012716 | 0.012706 |
| 音频干净、图像残缺 | no_causal | 0.012961 | 0.013088 | 0.012968 | 0.012959 |
| 双模态残缺 | control | 0.013111 | 0.013111 | 0.013111 | 0.013112 |
| 双模态残缺 | v11f | 0.012745 | 0.013111 | 0.012751 | 0.012746 |
| 双模态残缺 | no_causal | 0.012984 | 0.013111 | 0.012995 | 0.012986 |
| 仅音频残缺 | control | 0.003567 | 0.003567 | 0.003567 | 0.003569 |
| 仅音频残缺 | v11f | 0.003501 | 0.003567 | 0.003493 | 0.003479 |
| 仅音频残缺 | no_causal | 0.003535 | 0.003567 | 0.003543 | 0.003526 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11f | 0.000033 | -0.000003 | -0.000013 |
| 仅干净音频 | no_causal | 0.000008 | 0.000006 | -0.000006 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.000382 | 0.000010 | 6.569e-07 |
| 音频干净、图像残缺 | no_causal | 0.000127 | 0.000007 | -0.000002 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000366 | 0.000005 | 6.235e-07 |
| 双模态残缺 | no_causal | 0.000127 | 0.000011 | 0.000002 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11f | 0.000066 | -0.000009 | -0.000024 |
| 仅音频残缺 | no_causal | 0.000032 | 0.000008 | -0.000011 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11f | 36.260% | 63.900% | 28.100% |
| 仅干净音频 | no_causal | 30.240% | 63.710% | 24.860% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11f | 76.220% | 51.760% | 41.620% |
| 音频干净、图像残缺 | no_causal | 60.910% | 51.610% | 34.590% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 77.450% | 51.320% | 41.750% |
| 双模态残缺 | no_causal | 61.860% | 51.030% | 34.880% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11f | 43.470% | 61.670% | 31.630% |
| 仅音频残缺 | no_causal | 32.890% | 61.450% | 25.980% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11f | 0.544109 | 0.428402 |
| 仅干净音频 | no_causal | 0.754257 | 0.214882 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11f | 0.586491 | 0.295720 |
| 音频干净、图像残缺 | no_causal | 0.749668 | 0.138552 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.586190 | 0.280293 |
| 双模态残缺 | no_causal | 0.751773 | 0.132068 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11f | 0.544040 | 0.407651 |
| 仅音频残缺 | no_causal | 0.755015 | 0.205929 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11f | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000323 | 0.000300 |
| 图像干净、音频残缺 | control | 0.005272 | 0.005272 | 0.005272 | 0.005274 |
| 图像干净、音频残缺 | v11f | 0.004952 | 0.005272 | 0.004962 | 0.004954 |
| 图像干净、音频残缺 | no_causal | 0.005029 | 0.005272 | 0.005078 | 0.005035 |
| 双模态残缺 | control | 0.005077 | 0.005077 | 0.005077 | 0.005075 |
| 双模态残缺 | v11f | 0.004860 | 0.005077 | 0.004861 | 0.004860 |
| 双模态残缺 | no_causal | 0.004917 | 0.005077 | 0.004933 | 0.004918 |
| 仅图像残缺 | control | 0.000917 | 0.000917 | 0.000917 | 0.000917 |
| 仅图像残缺 | v11f | 0.000919 | 0.000917 | 0.000922 | 0.000920 |
| 仅图像残缺 | no_causal | 0.000921 | 0.000917 | 0.000927 | 0.000929 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11f | -0.000005 | 0.000010 | 7.096e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000027 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.000320 | 0.000010 | 4.882e-07 |
| 图像干净、音频残缺 | no_causal | 0.000243 | 0.000049 | 0.000004 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11f | 0.000216 | 9.348e-07 | 0.000002 |
| 双模态残缺 | no_causal | 0.000159 | 0.000016 | 0.000002 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11f | -0.000002 | 0.000003 | 8.985e-07 |
| 仅图像残缺 | no_causal | -0.000004 | 0.000006 | 0.000008 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11f | 34.920% | 69.880% | 26.600% |
| 仅干净图像 | no_causal | 30.950% | 71.780% | 25.460% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11f | 41.980% | 30.750% | 23.080% |
| 图像干净、音频残缺 | no_causal | 34.580% | 31.880% | 21.200% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11f | 41.980% | 28.990% | 21.920% |
| 双模态残缺 | no_causal | 34.330% | 29.380% | 19.190% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11f | 30.770% | 64.090% | 21.130% |
| 仅图像残缺 | no_causal | 49.220% | 61.460% | 32.380% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11f | 0.948699 | 0.042074 |
| 仅干净图像 | no_causal | 0.971705 | 0.029951 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11f | 0.955995 | 0.007767 |
| 图像干净、音频残缺 | no_causal | 0.981649 | 0.006664 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11f | 0.954072 | 0.005952 |
| 双模态残缺 | no_causal | 0.980710 | 0.004976 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11f | 0.946621 | 0.030866 |
| 仅图像残缺 | no_causal | 0.970366 | 0.021495 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11f | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11f | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11f | 98.090% | 97.710% | 97.530% | 97.459% |
| 仅干净音频 | no_causal | 97.750% | 97.710% | 97.720% | 97.709% |
| 图像干净、音频残缺 | control | 97.830% | 97.830% | 97.830% | 97.829% |
| 图像干净、音频残缺 | v11f | 97.830% | 97.830% | 97.830% | 97.829% |
| 图像干净、音频残缺 | no_causal | 97.830% | 97.830% | 97.830% | 97.829% |
| 音频干净、图像残缺 | control | 97.280% | 97.280% | 97.280% | 97.279% |
| 音频干净、图像残缺 | v11f | 97.220% | 97.280% | 97.300% | 97.199% |
| 音频干净、图像残缺 | no_causal | 97.410% | 97.280% | 97.320% | 97.149% |
| 双模态残缺 | control | 97.310% | 97.310% | 97.310% | 97.309% |
| 双模态残缺 | v11f | 97.280% | 97.310% | 97.270% | 97.299% |
| 双模态残缺 | no_causal | 97.190% | 97.310% | 97.250% | 97.289% |
| 仅音频残缺 | control | 96.200% | 96.200% | 96.200% | 96.198% |
| 仅音频残缺 | v11f | 96.440% | 96.200% | 96.310% | 96.289% |
| 仅音频残缺 | no_causal | 96.150% | 96.200% | 96.300% | 96.299% |
| 仅图像残缺 | control | 97.140% | 97.140% | 97.140% | 97.139% |
| 仅图像残缺 | v11f | 97.140% | 97.140% | 97.140% | 97.139% |
| 仅图像残缺 | no_causal | 97.140% | 97.140% | 97.140% | 97.139% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11f | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11f | 94.250% | 94.320% | 94.150% | 94.168% |
| 仅干净图像 | no_causal | 94.550% | 94.320% | 94.610% | 94.498% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11f | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 94.770% | 94.770% | 94.770% | 94.768% |
| 图像干净、音频残缺 | v11f | 94.850% | 94.770% | 94.880% | 94.898% |
| 图像干净、音频残缺 | no_causal | 94.900% | 94.770% | 94.800% | 94.968% |
| 音频干净、图像残缺 | control | 95.950% | 95.950% | 95.950% | 95.948% |
| 音频干净、图像残缺 | v11f | 95.950% | 95.950% | 95.950% | 95.948% |
| 音频干净、图像残缺 | no_causal | 95.950% | 95.950% | 95.950% | 95.948% |
| 双模态残缺 | control | 94.490% | 94.490% | 94.490% | 94.488% |
| 双模态残缺 | v11f | 94.580% | 94.490% | 94.520% | 94.628% |
| 双模态残缺 | no_causal | 94.410% | 94.490% | 94.490% | 94.448% |
| 仅音频残缺 | control | 94.330% | 94.330% | 94.330% | 94.328% |
| 仅音频残缺 | v11f | 94.330% | 94.330% | 94.330% | 94.328% |
| 仅音频残缺 | no_causal | 94.330% | 94.330% | 94.330% | 94.328% |
| 仅图像残缺 | control | 75.480% | 75.480% | 75.480% | 75.470% |
| 仅图像残缺 | v11f | 75.450% | 75.480% | 75.060% | 75.200% |
| 仅图像残缺 | no_causal | 76.350% | 75.480% | 76.520% | 76.471% |

</details>

**独立音频 family breakdown**

此处来自各实验的 `tables/audio_family_breakdown_fixed.csv`，单独改变音频 family；不能用它替代上面的双 family-pair 主表或再次计入宏平均。该 CSV 不含逐指标 n。


<details>
<summary>time_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.030% | 0.009772 | 0.942626 | 0.006218 | 0.833523 |
| 图像干净、音频残缺 | v11f | 99.030% | 0.009772 | 0.942626 | 0.006048 | 0.836068 |
| 图像干净、音频残缺 | no_causal | 99.030% | 0.009772 | 0.942626 | 0.006062 | 0.837266 |
| 双模态残缺 | control | 97.710% | 0.008706 | 0.948581 | 0.006304 | 0.830981 |
| 双模态残缺 | v11f | 97.710% | 0.008474 | 0.949835 | 0.006162 | 0.833106 |
| 双模态残缺 | no_causal | 97.710% | 0.008492 | 0.949849 | 0.006172 | 0.834155 |
| 仅音频残缺 | control | 90.390% | 0.009307 | 0.923568 | 0.006326 | 0.830664 |
| 仅音频残缺 | v11f | 90.390% | 0.009088 | 0.927061 | 0.006326 | 0.830664 |
| 仅音频残缺 | no_causal | 90.390% | 0.009025 | 0.927912 | 0.006326 | 0.830664 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.776195 | 0.009109 | 0.023938 |
| 图像干净、音频残缺 | v11f | 20.776195 | 0.008692 | 0.023846 |
| 图像干净、音频残缺 | no_causal | 20.776195 | 0.008724 | 0.023830 |
| 双模态残缺 | control | 23.123210 | 0.009356 | 0.024483 |
| 双模态残缺 | v11f | 23.200677 | 0.009008 | 0.024394 |
| 双模态残缺 | no_causal | 23.224873 | 0.009033 | 0.024381 |
| 仅音频残缺 | control | 32.712509 | 0.009393 | 0.024228 |
| 仅音频残缺 | v11f | 32.500217 | 0.009393 | 0.024228 |
| 仅音频残缺 | no_causal | 32.168268 | 0.009393 | 0.024228 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.004240 | 0.017718 |
| 图像干净、音频残缺 | v11f | 0.004240 | 0.017718 |
| 图像干净、音频残缺 | no_causal | 0.004240 | 0.017718 |
| 双模态残缺 | control | 0.004215 | 0.017613 |
| 双模态残缺 | v11f | 0.004215 | 0.017613 |
| 双模态残缺 | no_causal | 0.004215 | 0.017613 |
| 仅音频残缺 | control | 0.004228 | 0.017696 |
| 仅音频残缺 | v11f | 0.004228 | 0.017696 |
| 仅音频残缺 | no_causal | 0.004228 | 0.017696 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.147264 | 0.152054 | 78.086% |
| 图像干净、音频残缺 | v11f | 0.146721 | 0.152054 | 78.208% |
| 图像干净、音频残缺 | no_causal | 0.148156 | 0.152054 | 78.158% |
| 双模态残缺 | control | 0.147181 | 0.152054 | 77.929% |
| 双模态残缺 | v11f | 0.146756 | 0.152054 | 78.034% |
| 双模态残缺 | no_causal | 0.147914 | 0.152054 | 77.988% |
| 仅音频残缺 | control | 0.146734 | 0.152054 | 77.866% |
| 仅音频残缺 | v11f | 0.146734 | 0.152054 | 77.866% |
| 仅音频残缺 | no_causal | 0.146734 | 0.152054 | 77.866% |

</details>


<details>
<summary>freq_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 98.840% | 0.009763 | 0.942693 | 0.004856 | 0.860159 |
| 图像干净、音频残缺 | v11f | 98.840% | 0.009763 | 0.942693 | 0.004710 | 0.862357 |
| 图像干净、音频残缺 | no_causal | 98.840% | 0.009763 | 0.942693 | 0.004750 | 0.862228 |
| 双模态残缺 | control | 97.910% | 0.008750 | 0.948192 | 0.004888 | 0.859557 |
| 双模态残缺 | v11f | 97.910% | 0.008521 | 0.949440 | 0.004767 | 0.861325 |
| 双模态残缺 | no_causal | 97.910% | 0.008552 | 0.949384 | 0.004803 | 0.861079 |
| 仅音频残缺 | control | 92.220% | 0.007925 | 0.938894 | 0.004905 | 0.858229 |
| 仅音频残缺 | v11f | 92.220% | 0.007763 | 0.940916 | 0.004905 | 0.858229 |
| 仅音频残缺 | no_causal | 92.220% | 0.007808 | 0.940264 | 0.004905 | 0.858229 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.781139 | 0.006477 | 0.021425 |
| 图像干净、音频残缺 | v11f | 20.781139 | 0.006118 | 0.021510 |
| 图像干净、音频残缺 | no_causal | 20.781139 | 0.006216 | 0.021578 |
| 双模态残缺 | control | 23.020364 | 0.006543 | 0.021547 |
| 双模态残缺 | v11f | 23.096880 | 0.006244 | 0.021590 |
| 双模态残缺 | no_causal | 23.115478 | 0.006332 | 0.021665 |
| 仅音频残缺 | control | 31.814109 | 0.006583 | 0.021626 |
| 仅音频残缺 | v11f | 31.707819 | 0.006583 | 0.021626 |
| 仅音频残缺 | no_causal | 31.402489 | 0.006583 | 0.021626 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.003747 | 0.016059 |
| 图像干净、音频残缺 | v11f | 0.003747 | 0.016059 |
| 图像干净、音频残缺 | no_causal | 0.003747 | 0.016059 |
| 双模态残缺 | control | 0.003756 | 0.016091 |
| 双模态残缺 | v11f | 0.003756 | 0.016091 |
| 双模态残缺 | no_causal | 0.003756 | 0.016091 |
| 仅音频残缺 | control | 0.003757 | 0.016081 |
| 仅音频残缺 | v11f | 0.003757 | 0.016081 |
| 仅音频残缺 | no_causal | 0.003757 | 0.016081 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.151326 | 0.152054 | 80.131% |
| 图像干净、音频残缺 | v11f | 0.150156 | 0.152054 | 80.161% |
| 图像干净、音频残缺 | no_causal | 0.152110 | 0.152054 | 80.138% |
| 双模态残缺 | control | 0.151691 | 0.152054 | 80.076% |
| 双模态残缺 | v11f | 0.150796 | 0.152054 | 80.109% |
| 双模态残缺 | no_causal | 0.152338 | 0.152054 | 80.090% |
| 仅音频残缺 | control | 0.151239 | 0.152054 | 80.019% |
| 仅音频残缺 | v11f | 0.151239 | 0.152054 | 80.019% |
| 仅音频残缺 | no_causal | 0.151239 | 0.152054 | 80.019% |

</details>


<details>
<summary>feature_dropout</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.320% | 0.009733 | 0.942740 | 0.003738 | 0.904314 |
| 图像干净、音频残缺 | v11f | 99.320% | 0.009733 | 0.942740 | 0.003661 | 0.905470 |
| 图像干净、音频残缺 | no_causal | 99.320% | 0.009733 | 0.942740 | 0.003680 | 0.905112 |
| 双模态残缺 | control | 99.030% | 0.008696 | 0.948433 | 0.003731 | 0.904324 |
| 双模态残缺 | v11f | 99.030% | 0.008448 | 0.949732 | 0.003668 | 0.905291 |
| 双模态残缺 | no_causal | 99.030% | 0.008492 | 0.949625 | 0.003684 | 0.904983 |
| 仅音频残缺 | control | 97.100% | 0.003009 | 0.978650 | 0.003738 | 0.904178 |
| 仅音频残缺 | v11f | 97.100% | 0.002965 | 0.979239 | 0.003738 | 0.904178 |
| 仅音频残缺 | no_causal | 97.100% | 0.002993 | 0.978861 | 0.003738 | 0.904178 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.793747 | 0.003796 | 0.015939 |
| 图像干净、音频残缺 | v11f | 20.793747 | 0.003605 | 0.015724 |
| 图像干净、音频残缺 | no_causal | 20.793747 | 0.003652 | 0.015754 |
| 双模态残缺 | control | 23.151188 | 0.003790 | 0.015938 |
| 双模态残缺 | v11f | 23.239851 | 0.003633 | 0.015757 |
| 双模态残缺 | no_causal | 23.258268 | 0.003672 | 0.015784 |
| 仅音频残缺 | control | 36.140352 | 0.003797 | 0.015945 |
| 仅音频残缺 | v11f | 35.851597 | 0.003797 | 0.015945 |
| 仅音频残缺 | no_causal | 35.489989 | 0.003797 | 0.015945 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.003698 | 0.015718 |
| 图像干净、音频残缺 | v11f | 0.003698 | 0.015718 |
| 图像干净、音频残缺 | no_causal | 0.003698 | 0.015718 |
| 双模态残缺 | control | 0.003691 | 0.015713 |
| 双模态残缺 | v11f | 0.003691 | 0.015713 |
| 双模态残缺 | no_causal | 0.003691 | 0.015713 |
| 仅音频残缺 | control | 0.003699 | 0.015725 |
| 仅音频残缺 | v11f | 0.003699 | 0.015725 |
| 仅音频残缺 | no_causal | 0.003699 | 0.015725 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.153838 | 0.152054 | 81.663% |
| 图像干净、音频残缺 | v11f | 0.153231 | 0.152054 | 81.695% |
| 图像干净、音频残缺 | no_causal | 0.153849 | 0.152054 | 81.681% |
| 双模态残缺 | control | 0.153931 | 0.152054 | 81.662% |
| 双模态残缺 | v11f | 0.153452 | 0.152054 | 81.687% |
| 双模态残缺 | no_causal | 0.153938 | 0.152054 | 81.678% |
| 仅音频残缺 | control | 0.153804 | 0.152054 | 81.634% |
| 仅音频残缺 | v11f | 0.153804 | 0.152054 | 81.634% |
| 仅音频残缺 | no_causal | 0.153804 | 0.152054 | 81.634% |

</details>


<details>
<summary>partial_temporal</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 98.070% | 0.009949 | 0.942169 | 0.011784 | 0.538387 |
| 图像干净、音频残缺 | v11f | 98.070% | 0.009949 | 0.942169 | 0.011342 | 0.543572 |
| 图像干净、音频残缺 | no_causal | 98.070% | 0.009949 | 0.942169 | 0.011537 | 0.544788 |
| 双模态残缺 | control | 92.030% | 0.008929 | 0.947314 | 0.011935 | 0.536746 |
| 双模态残缺 | v11f | 92.030% | 0.008741 | 0.948478 | 0.011577 | 0.540696 |
| 双模态残缺 | no_causal | 92.030% | 0.008717 | 0.948595 | 0.011741 | 0.541828 |
| 仅音频残缺 | control | 45.880% | 0.042710 | 0.569261 | 0.012585 | 0.528776 |
| 仅音频残缺 | v11f | 45.880% | 0.040867 | 0.604162 | 0.012585 | 0.528776 |
| 仅音频残缺 | no_causal | 45.880% | 0.038628 | 0.639875 | 0.012585 | 0.528776 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.701504 | 0.028559 | 0.074284 |
| 图像干净、音频残缺 | v11f | 20.701504 | 0.027472 | 0.074668 |
| 图像干净、音频残缺 | no_causal | 20.701504 | 0.027952 | 0.074875 |
| 双模态残缺 | control | 22.906776 | 0.028924 | 0.074755 |
| 双模态残缺 | v11f | 22.957621 | 0.028044 | 0.075042 |
| 双模态残缺 | no_causal | 22.912000 | 0.028448 | 0.075233 |
| 仅音频残缺 | control | 17.523188 | 0.030529 | 0.075829 |
| 仅音频残缺 | v11f | 17.651848 | 0.030529 | 0.075829 |
| 仅音频残缺 | no_causal | 17.683809 | 0.030529 | 0.075829 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.000306 | 0.001634 |
| 图像干净、音频残缺 | v11f | 0.000306 | 0.001634 |
| 图像干净、音频残缺 | no_causal | 0.000306 | 0.001634 |
| 双模态残缺 | control | 0.000311 | 0.001646 |
| 双模态残缺 | v11f | 0.000311 | 0.001646 |
| 双模态残缺 | no_causal | 0.000311 | 0.001646 |
| 仅音频残缺 | control | 0.000308 | 0.001635 |
| 仅音频残缺 | v11f | 0.000308 | 0.001635 |
| 仅音频残缺 | no_causal | 0.000308 | 0.001635 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.131051 | 0.152054 | 70.184% |
| 图像干净、音频残缺 | v11f | 0.128603 | 0.152054 | 70.423% |
| 图像干净、音频残缺 | no_causal | 0.133703 | 0.152054 | 70.494% |
| 双模态残缺 | control | 0.130508 | 0.152054 | 69.956% |
| 双模态残缺 | v11f | 0.128607 | 0.152054 | 70.142% |
| 双模态残缺 | no_causal | 0.132802 | 0.152054 | 70.199% |
| 仅音频残缺 | control | 0.126849 | 0.152054 | 68.606% |
| 仅音频残缺 | v11f | 0.126849 | 0.152054 | 68.606% |
| 仅音频残缺 | no_causal | 0.126849 | 0.152054 | 68.606% |

</details>


<details>
<summary>time_freq_block</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.120% | 0.009738 | 0.942713 | 0.003986 | 0.900341 |
| 图像干净、音频残缺 | v11f | 99.120% | 0.009738 | 0.942713 | 0.003935 | 0.900974 |
| 图像干净、音频残缺 | no_causal | 99.120% | 0.009738 | 0.942713 | 0.003949 | 0.900904 |
| 双模态残缺 | control | 98.770% | 0.008698 | 0.948653 | 0.003986 | 0.900257 |
| 双模态残缺 | v11f | 98.770% | 0.008453 | 0.949948 | 0.003944 | 0.900766 |
| 双模态残缺 | no_causal | 98.770% | 0.008479 | 0.949906 | 0.003955 | 0.900685 |
| 仅音频残缺 | control | 96.750% | 0.003525 | 0.974426 | 0.003997 | 0.900021 |
| 仅音频残缺 | v11f | 96.750% | 0.003463 | 0.975142 | 0.003997 | 0.900021 |
| 仅音频残缺 | no_causal | 96.750% | 0.003497 | 0.974773 | 0.003997 | 0.900021 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.790810 | 0.005269 | 0.017942 |
| 图像干净、音频残缺 | v11f | 20.790810 | 0.004960 | 0.017952 |
| 图像干净、音频残缺 | no_causal | 20.790810 | 0.005041 | 0.018017 |
| 双模态残缺 | control | 23.127621 | 0.005198 | 0.017747 |
| 双模态残缺 | v11f | 23.219348 | 0.004942 | 0.017741 |
| 双模态残缺 | no_causal | 23.240727 | 0.005008 | 0.017800 |
| 仅音频残缺 | control | 36.145908 | 0.005324 | 0.018041 |
| 仅音频残缺 | v11f | 35.857163 | 0.005324 | 0.018041 |
| 仅音频残缺 | no_causal | 35.518400 | 0.005324 | 0.018041 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.003733 | 0.015994 |
| 图像干净、音频残缺 | v11f | 0.003733 | 0.015994 |
| 图像干净、音频残缺 | no_causal | 0.003733 | 0.015994 |
| 双模态残缺 | control | 0.003747 | 0.016053 |
| 双模态残缺 | v11f | 0.003747 | 0.016053 |
| 双模态残缺 | no_causal | 0.003747 | 0.016053 |
| 仅音频残缺 | control | 0.003734 | 0.015998 |
| 仅音频残缺 | v11f | 0.003734 | 0.015998 |
| 仅音频残缺 | no_causal | 0.003734 | 0.015998 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.154400 | 0.152054 | 81.583% |
| 图像干净、音频残缺 | v11f | 0.154069 | 0.152054 | 81.600% |
| 图像干净、音频残缺 | no_causal | 0.154676 | 0.152054 | 81.592% |
| 双模态残缺 | control | 0.154617 | 0.152054 | 81.568% |
| 双模态残缺 | v11f | 0.154357 | 0.152054 | 81.586% |
| 双模态残缺 | no_causal | 0.154845 | 0.152054 | 81.571% |
| 仅音频残缺 | control | 0.154387 | 0.152054 | 81.547% |
| 仅音频残缺 | v11f | 0.154387 | 0.152054 | 81.547% |
| 仅音频残缺 | no_causal | 0.154387 | 0.152054 | 81.547% |

</details>

**训练统计**

仅统计每个完整 epoch 的平均训练 loss；不同 loss 定义不可横比，最低训练 loss 也不是测试最优 checkpoint。

| 实验 | 完成 epoch | 本轮轮数 | 首轮 loss | 末轮 loss | 最低 loss | 末轮 LR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| control | 冻结父模型 | 0 | N/A | N/A | N/A | N/A |
| v11f | 0–29 | 30 | 0.7827 | 0.8212 | 0.7632 | 0.000010 |
| no_causal | 0–29 | 30 | 0.7385 | 0.7789 | 0.7202 | 0.000010 |


<details>
<summary>逐 epoch loss / LR</summary>

| epoch | 实验 | 平均 loss | LR |
| --- | --- | ---: | ---: |
| 0 | v11f | 0.7827 | 0.000100 |
| 0 | no_causal | 0.7385 | 0.000100 |
| 1 | v11f | 0.8434 | 0.000100 |
| 1 | no_causal | 0.7984 | 0.000100 |
| 2 | v11f | 0.9060 | 0.000099 |
| 2 | no_causal | 0.8607 | 0.000099 |
| 3 | v11f | 0.8205 | 0.000098 |
| 3 | no_causal | 0.7762 | 0.000098 |
| 4 | v11f | 0.8441 | 0.000096 |
| 4 | no_causal | 0.8004 | 0.000096 |
| 5 | v11f | 0.8012 | 0.000094 |
| 5 | no_causal | 0.7583 | 0.000094 |
| 6 | v11f | 0.8487 | 0.000091 |
| 6 | no_causal | 0.8046 | 0.000091 |
| 7 | v11f | 0.8489 | 0.000088 |
| 7 | no_causal | 0.8064 | 0.000088 |
| 8 | v11f | 0.8207 | 0.000085 |
| 8 | no_causal | 0.7770 | 0.000085 |
| 9 | v11f | 0.8206 | 0.000081 |
| 9 | no_causal | 0.7787 | 0.000081 |
| 10 | v11f | 0.8459 | 0.000077 |
| 10 | no_causal | 0.8038 | 0.000077 |
| 11 | v11f | 0.8059 | 0.000073 |
| 11 | no_causal | 0.7616 | 0.000073 |
| 12 | v11f | 0.7915 | 0.000069 |
| 12 | no_causal | 0.7495 | 0.000069 |
| 13 | v11f | 0.8713 | 0.000064 |
| 13 | no_causal | 0.8282 | 0.000064 |
| 14 | v11f | 0.7739 | 0.000060 |
| 14 | no_causal | 0.7301 | 0.000060 |
| 15 | v11f | 0.8118 | 0.000055 |
| 15 | no_causal | 0.7687 | 0.000055 |
| 16 | v11f | 0.7713 | 0.000050 |
| 16 | no_causal | 0.7284 | 0.000050 |
| 17 | v11f | 0.7632 | 0.000046 |
| 17 | no_causal | 0.7202 | 0.000046 |
| 18 | v11f | 0.7760 | 0.000041 |
| 18 | no_causal | 0.7330 | 0.000041 |
| 19 | v11f | 0.7905 | 0.000037 |
| 19 | no_causal | 0.7486 | 0.000037 |
| 20 | v11f | 0.8049 | 0.000033 |
| 20 | no_causal | 0.7617 | 0.000033 |
| 21 | v11f | 0.7818 | 0.000029 |
| 21 | no_causal | 0.7390 | 0.000029 |
| 22 | v11f | 0.7895 | 0.000025 |
| 22 | no_causal | 0.7461 | 0.000025 |
| 23 | v11f | 0.8543 | 0.000022 |
| 23 | no_causal | 0.8116 | 0.000022 |
| 24 | v11f | 0.8331 | 0.000019 |
| 24 | no_causal | 0.7905 | 0.000019 |
| 25 | v11f | 0.8281 | 0.000016 |
| 25 | no_causal | 0.7850 | 0.000016 |
| 26 | v11f | 0.8383 | 0.000014 |
| 26 | no_causal | 0.7951 | 0.000014 |
| 27 | v11f | 0.7932 | 0.000012 |
| 27 | no_causal | 0.7501 | 0.000012 |
| 28 | v11f | 0.7644 | 0.000011 |
| 28 | no_causal | 0.7228 | 0.000011 |
| 29 | v11f | 0.8212 | 0.000010 |
| 29 | no_causal | 0.7789 | 0.000010 |

</details>

训练证据：
- `v11f`：[train.log](../v11f_outputs_with_ckpt/outputs/outputs_v11f/logs/train.log)
- `no_causal`：[train.log](../v11f_outputs_with_ckpt/outputs/outputs_v11f_no_causal/logs/train.log)

**Demo 小样本**

每份 demo 为 10 条样本，不是全测试集分数；fixed/random 分开。原始图片与逐样本预测留在对应产物目录，本节不宣称重新进行了图像质量评审。


<details>
<summary>fixed_mask demo (n=10)</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 70.000% | 0.023100 | 0.775000 | 0.009000 | 0.766000 |
| 仅音频残缺 | v11f | 70.000% | 0.022500 | 0.790000 | 0.009000 | 0.766000 |
| 仅音频残缺 | no_causal | 70.000% | 0.022300 | 0.796000 | 0.009000 | 0.766000 |
| 仅图像残缺 | control | 90.000% | 0.006400 | 0.957000 | 0.001500 | 0.867000 |
| 仅图像残缺 | v11f | 90.000% | 0.006400 | 0.957000 | 0.001500 | 0.868000 |
| 仅图像残缺 | no_causal | 90.000% | 0.006400 | 0.957000 | 0.001500 | 0.867000 |
| 双模态残缺 | control | 100.000% | 0.006400 | 0.958000 | 0.008500 | 0.779000 |
| 双模态残缺 | v11f | 100.000% | 0.006300 | 0.958000 | 0.008200 | 0.782000 |
| 双模态残缺 | no_causal | 100.000% | 0.006200 | 0.959000 | 0.008200 | 0.784000 |

| 输入模式 | 实验 | 图像缺失区 MSE | 音频缺失区 MSE |
| --- | --- | ---: | ---: |
| 仅音频残缺 | control | 0.023100 | 0.021000 |
| 仅音频残缺 | v11f | 0.022500 | 0.021000 |
| 仅音频残缺 | no_causal | 0.022300 | 0.021000 |
| 仅图像残缺 | control | 0.019500 | 0.001500 |
| 仅图像残缺 | v11f | 0.019500 | 0.001500 |
| 仅图像残缺 | no_causal | 0.019500 | 0.001500 |
| 双模态残缺 | control | 0.019400 | 0.019900 |
| 双模态残缺 | v11f | 0.018900 | 0.019100 |
| 双模态残缺 | no_causal | 0.018800 | 0.018800 |

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.040300 | 0.147100 | 1.000000 |
| 仅音频残缺 | v11f | 0.040300 | 0.147100 | 1.000000 |
| 仅音频残缺 | no_causal | 0.040300 | 0.147100 | 1.000000 |
| 仅图像残缺 | control | 0.016500 | 0.069800 | 0.934600 |
| 仅图像残缺 | v11f | 0.016900 | 0.070000 | 0.883300 |
| 仅图像残缺 | no_causal | 0.016600 | 0.070300 | 0.894000 |
| 双模态残缺 | control | 0.041100 | 0.148800 | 1.000000 |
| 双模态残缺 | v11f | 0.041400 | 0.147300 | 1.000000 |
| 双模态残缺 | no_causal | 0.042300 | 0.150600 | 1.000000 |

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.047400 | 0.168700 | 1.000000 | 79.400% |
| 仅音频残缺 | v11f | 0.047400 | 0.168700 | 1.000000 | 79.400% |
| 仅音频残缺 | no_causal | 0.047400 | 0.168700 | 1.000000 | 79.400% |
| 仅图像残缺 | control | 0.020700 | 0.084100 | 0.987900 | 82.900% |
| 仅图像残缺 | v11f | 0.020700 | 0.084100 | 0.987900 | 82.800% |
| 仅图像残缺 | no_causal | 0.020700 | 0.084100 | 0.987900 | 82.700% |
| 双模态残缺 | control | 0.047400 | 0.168700 | 1.000000 | 79.500% |
| 双模态残缺 | v11f | 0.047400 | 0.168700 | 1.000000 | 79.500% |
| 双模态残缺 | no_causal | 0.047400 | 0.168700 | 1.000000 | 79.400% |

</details>


<details>
<summary>legacy_random demo (n=10)</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 60.000% | 0.018900 | 0.814000 | 0.015900 | 0.647000 |
| 仅音频残缺 | v11f | 60.000% | 0.018700 | 0.825000 | 0.015900 | 0.647000 |
| 仅音频残缺 | no_causal | 60.000% | 0.018400 | 0.830000 | 0.015900 | 0.647000 |
| 仅图像残缺 | control | 80.000% | 0.012000 | 0.925000 | 0.002000 | 0.840000 |
| 仅图像残缺 | v11f | 80.000% | 0.012000 | 0.925000 | 0.002000 | 0.841000 |
| 仅图像残缺 | no_causal | 80.000% | 0.012000 | 0.925000 | 0.002100 | 0.839000 |
| 双模态残缺 | control | 90.000% | 0.009800 | 0.939000 | 0.012900 | 0.729000 |
| 双模态残缺 | v11f | 90.000% | 0.009600 | 0.940000 | 0.012500 | 0.733000 |
| 双模态残缺 | no_causal | 90.000% | 0.009700 | 0.940000 | 0.012400 | 0.738000 |

| 输入模式 | 实验 | 图像缺失区 MSE | 音频缺失区 MSE |
| --- | --- | ---: | ---: |
| 仅音频残缺 | control | 0.018900 | 0.033400 |
| 仅音频残缺 | v11f | 0.018700 | 0.033400 |
| 仅音频残缺 | no_causal | 0.018400 | 0.033400 |
| 仅图像残缺 | control | 0.077800 | 0.002000 |
| 仅图像残缺 | v11f | 0.077800 | 0.002000 |
| 仅图像残缺 | no_causal | 0.077800 | 0.002100 |
| 双模态残缺 | control | 0.063500 | 0.025900 |
| 双模态残缺 | v11f | 0.062200 | 0.025000 |
| 双模态残缺 | no_causal | 0.062700 | 0.024700 |

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.039100 | 0.143900 | 1.000000 |
| 仅音频残缺 | v11f | 0.039100 | 0.143900 | 1.000000 |
| 仅音频残缺 | no_causal | 0.039100 | 0.143900 | 1.000000 |
| 仅图像残缺 | control | 0.018300 | 0.072800 | 1.000000 |
| 仅图像残缺 | v11f | 0.018700 | 0.072300 | 0.988500 |
| 仅图像残缺 | no_causal | 0.018500 | 0.073500 | 1.000000 |
| 双模态残缺 | control | 0.038900 | 0.139200 | 1.000000 |
| 双模态残缺 | v11f | 0.039200 | 0.138400 | 1.000000 |
| 双模态残缺 | no_causal | 0.039700 | 0.140900 | 1.000000 |

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.047400 | 0.168700 | 1.000000 | 73.700% |
| 仅音频残缺 | v11f | 0.047400 | 0.168700 | 1.000000 | 73.700% |
| 仅音频残缺 | no_causal | 0.047400 | 0.168700 | 1.000000 | 73.700% |
| 仅图像残缺 | control | 0.020700 | 0.084100 | 0.987900 | 80.800% |
| 仅图像残缺 | v11f | 0.020700 | 0.084100 | 0.987900 | 80.800% |
| 仅图像残缺 | no_causal | 0.020700 | 0.084100 | 0.987900 | 80.700% |
| 双模态残缺 | control | 0.047400 | 0.168700 | 1.000000 | 76.100% |
| 双模态残缺 | v11f | 0.047400 | 0.168700 | 1.000000 | 76.300% |
| 双模态残缺 | no_causal | 0.047400 | 0.168700 | 1.000000 | 76.200% |

</details>

**各实验结论与比较限制**

- **仅音频残缺**：主实验相对 control，音频缺失区 MSE 变化 +0.00%，Index ACC 变化 +0.000 个百分点。
- **双模态残缺**：主实验相对 control，音频缺失区 MSE 变化 -3.33%，Index ACC 变化 +0.000 个百分点。
- **control**：冻结 v11e_control，额外训练 0 轮；是父模型参考，不是与主实验等预算重训。
- **v11f**：额外训练 30 轮的 Masked Cross-Key + causal；Index ACC 与 control 相同符合冻结设计，不能当作分类提升。恢复有小幅改善，但需同时看 wrong/same-class 和 win_both。
- **no_causal**：相同父权重、相同 30 轮预算，不加 causal；部分恢复指标更好，不能仅凭主实验的 causal loss 宣称因果项全面有效。
- 内容 ACC 是冻结原网络对恢复结果的内部再分类，不是独立外部识别；same_class 的样本集合少 4 条，应读配对 same_damage，不能直接相减两个不同 n 的绝对均值。
- 本节仅汇总已有结果，没有重跑训练或推理。单 seed、重复音频曝光和旧日志舍入限制仍在，不报告统计显著性。

[返回统一评估导航](#evaluation-format-20260912)

<a id="evaluation-v11g"></a>

### v11g

本节覆盖 **3 组实验、52 个主评估/干预字段、7 个音频分布诊断字段**，以及独立 family、训练统计和本地已有 demo。字段按实际产物统计，含明确标注的不适用项；不把其他版本的缺项补成零。

**实验说明**

| 实验 | 权重起点 | 本轮训练范围 | 实际轮数 |
| --- | --- | ---: | ---: |
| control | v11e_control | 冻结父模型；不训练 | 0 |
| v11g | v11e_control | 仅 Masked Cross-Key adapter + causal | 30（0–29） |
| no_causal | v11e_control | 仅 adapter；关闭 causal | 30（0–29） |

训练配置 seed=1234、batch_size=128；fixed_mask seed=1234、severity=0.4。轮数以上表和完成日志为准，不把父模型历史轮数计成本轮新增训练。历史分支配置不是当前分支的运行入口。

**恢复目标：仅图像为 sample/category，仅音频为 category/sample，双模态为 sample/sample。** category 使用训练集 medoid；“仅音频残缺”表示图像全缺失、音频部分残缺，反向同理。

| 实验 | fixed / random | Cross-Key 扫描 | demo fixed / random |
| --- | --- | ---: | ---: |
| control | 有 / 有 | fixed + random | 有 / 有 |
| v11g | 有 / 有 | fixed + random | 有 / 有 |
| no_causal | 有 / 有 | fixed + random | 有 / 有 |

**数据来源**：
- `control`：[本地产物](../v11g_outputs/outputs/outputs_v11g_control/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。
- `v11g`：[本地产物](../v11g_outputs/outputs/outputs_v11g/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。
- `no_causal`：[本地产物](../v11g_outputs/outputs/outputs_v11g_no_causal/)，主表优先使用 normal CSV；旧版 CSV 未保存的字段取对应 normal 日志。

MSE/L1 越低越好，ACC/SSIM/PSNR 越高越好；gate、res/V 和多样性仅是诊断。宏平均为五个 family 等权均值，不是把 normal、sweep、独立音频 family 重复叠加。N/A 表示未保存或在该场景不适用，详见字段覆盖说明。

**有效 n**：normal 有效指标每 family 通常 n=10000，不适用区域 n=0；same-class 干预通常 n=9996，其余有效干预 n=10000。五类是同一测试集的重复曝光，不是 50000 条独立音频；精确字段计数见附表。

**分类与恢复：fixed 五类等权宏平均**

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11g | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11g | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965833 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966043 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11g | 98.830% | 0.002053 | 0.985947 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002085 | 0.985614 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 98.934% | 0.009790 | 0.942597 | 0.006097 | 0.807590 |
| 图像干净、音频残缺 | v11g | 98.934% | 0.009790 | 0.942597 | 0.005922 | 0.809892 |
| 图像干净、音频残缺 | no_causal | 98.934% | 0.009790 | 0.942597 | 0.005991 | 0.810125 |
| 音频干净、图像残缺 | control | 99.144% | 0.007502 | 0.956305 | 0.003647 | 0.911207 |
| 音频干净、图像残缺 | v11g | 99.144% | 0.007321 | 0.957235 | 0.003647 | 0.911207 |
| 音频干净、图像残缺 | no_causal | 99.144% | 0.007390 | 0.956850 | 0.003647 | 0.911207 |
| 双模态残缺 | control | 97.536% | 0.007520 | 0.956189 | 0.006176 | 0.805881 |
| 双模态残缺 | v11g | 97.536% | 0.007371 | 0.957016 | 0.006041 | 0.807610 |
| 双模态残缺 | no_causal | 97.536% | 0.007431 | 0.956679 | 0.006095 | 0.807901 |
| 仅音频残缺 | control | 84.572% | 0.013212 | 0.877703 | 0.006290 | 0.804689 |
| 仅音频残缺 | v11g | 84.572% | 0.012756 | 0.885779 | 0.006290 | 0.804689 |
| 仅音频残缺 | no_causal | 84.572% | 0.012351 | 0.891893 | 0.006290 | 0.804689 |
| 仅图像残缺 | control | 92.028% | 0.007680 | 0.955151 | 0.000668 | 0.916781 |
| 仅图像残缺 | v11g | 92.028% | 0.007680 | 0.955151 | 0.000670 | 0.916390 |
| 仅图像残缺 | no_causal | 92.028% | 0.007680 | 0.955151 | 0.000674 | 0.917291 |


<details>
<summary>fixed 五类等权宏平均：区域、内容、多样性与音频分布完整指标</summary>

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11g | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11g | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11g | 37.192527 | 0.002053 | 0.002053 | 0.007442 |
| 仅干净音频 | no_causal | 36.885119 | 0.002085 | 0.002085 | 0.007491 |
| 图像干净、音频残缺 | control | 20.768335 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | 20.768335 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.768335 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 22.993817 | 0.054834 | 0.025918 | 0.062284 |
| 音频干净、图像残缺 | v11g | 23.092333 | 0.055859 | 0.025263 | 0.062754 |
| 音频干净、图像残缺 | no_causal | 23.067494 | 0.054915 | 0.025441 | 0.062001 |
| 双模态残缺 | control | 22.976000 | 0.054686 | 0.025912 | 0.062268 |
| 双模态残缺 | v11g | 23.057996 | 0.054989 | 0.025356 | 0.062814 |
| 双模态残缺 | no_causal | 23.028828 | 0.054236 | 0.025528 | 0.062522 |
| 仅音频残缺 | control | 30.894253 | 0.013212 | 0.013212 | 0.025060 |
| 仅音频残缺 | v11g | 30.739226 | 0.012756 | 0.012756 | 0.025332 |
| 仅音频残缺 | no_causal | 30.457587 | 0.012351 | 0.012351 | 0.025794 |
| 仅图像残缺 | control | 22.904681 | 0.056325 | 0.026662 | 0.063682 |
| 仅图像残缺 | v11g | 22.904681 | 0.056325 | 0.026662 | 0.063682 |
| 仅图像残缺 | no_causal | 22.904681 | 0.056325 | 0.026662 | 0.063682 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.030732 | 8.090e-09 | 0.000081 |
| 音频干净、图像残缺 | v11g | 0.030732 | 8.090e-09 | 0.000081 |
| 音频干净、图像残缺 | no_causal | 0.030732 | 8.090e-09 | 0.000081 |
| 双模态残缺 | control | 0.030551 | 8.089e-09 | 0.000081 |
| 双模态残缺 | v11g | 0.030551 | 8.089e-09 | 0.000081 |
| 双模态残缺 | no_causal | 0.030551 | 8.089e-09 | 0.000081 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.031234 | 8.090e-09 | 0.000081 |
| 仅图像残缺 | v11g | 0.031234 | 8.090e-09 | 0.000081 |
| 仅图像残缺 | no_causal | 0.031234 | 8.090e-09 | 0.000081 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11g | 0.000294 | 0.003616 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003565 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.010599 | 0.030586 | 0.003143 | 0.013437 |
| 图像干净、音频残缺 | v11g | 0.010128 | 0.030621 | 0.003143 | 0.013437 |
| 图像干净、音频残缺 | no_causal | 0.010311 | 0.030733 | 0.003143 | 0.013437 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.010775 | 0.030940 | 0.003137 | 0.013386 |
| 双模态残缺 | v11g | 0.010416 | 0.030932 | 0.003137 | 0.013386 |
| 双模态残缺 | no_causal | 0.010558 | 0.031016 | 0.003137 | 0.013386 |
| 仅音频残缺 | control | 0.011079 | 0.031012 | 0.003143 | 0.013438 |
| 仅音频残缺 | v11g | 0.011079 | 0.031012 | 0.003143 | 0.013438 |
| 仅音频残缺 | no_causal | 0.011079 | 0.031012 | 0.003143 | 0.013438 |
| 仅图像残缺 | control | 0.000668 | 0.005295 | N/A | N/A |
| 仅图像残缺 | v11g | 0.000670 | 0.005428 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000674 | 0.005330 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.338677 |
| 干净双模态 | v11g | 97.790% | 96.040% | 0.058851 | 0.338677 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.338677 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.340023 |
| 仅干净图像 | v11g | 96.820% | 94.200% | 0.059386 | 0.340023 |
| 仅干净图像 | no_causal | 96.820% | 94.480% | 0.059386 | 0.340023 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.287549 |
| 仅干净音频 | v11g | 98.090% | 96.000% | 0.045921 | 0.286325 |
| 仅干净音频 | no_causal | 97.710% | 96.000% | 0.046246 | 0.287506 |
| 图像干净、音频残缺 | control | 97.520% | 88.970% | 0.059321 | 0.339469 |
| 图像干净、音频残缺 | v11g | 97.520% | 88.936% | 0.059321 | 0.339469 |
| 图像干净、音频残缺 | no_causal | 97.520% | 89.218% | 0.059321 | 0.339469 |
| 音频干净、图像残缺 | control | 96.720% | 95.968% | 0.061073 | 0.345199 |
| 音频干净、图像残缺 | v11g | 96.750% | 95.968% | 0.060658 | 0.343979 |
| 音频干净、图像残缺 | no_causal | 96.754% | 95.968% | 0.060846 | 0.344517 |
| 双模态残缺 | control | 96.178% | 87.012% | 0.061083 | 0.345650 |
| 双模态残缺 | v11g | 96.266% | 87.022% | 0.060877 | 0.345074 |
| 双模态残缺 | no_causal | 96.258% | 87.150% | 0.060968 | 0.345329 |
| 仅音频残缺 | control | 82.882% | 80.600% | 0.040052 | 0.265096 |
| 仅音频残缺 | v11g | 83.066% | 80.600% | 0.039582 | 0.263287 |
| 仅音频残缺 | no_causal | 82.942% | 80.600% | 0.039488 | 0.262683 |
| 仅图像残缺 | control | 95.514% | 83.518% | 0.060839 | 0.344495 |
| 仅图像残缺 | v11g | 95.514% | 83.468% | 0.060839 | 0.344495 |
| 仅图像残缺 | no_causal | 95.514% | 84.086% | 0.060839 | 0.344495 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11g | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11g | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11g | 0.015600 | 0.068500 | 0.981700 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.996400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.040800 | 0.147600 | 1.000000 |
| 图像干净、音频残缺 | v11g | 0.041160 | 0.146560 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.041560 | 0.148560 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155680 | 1.000000 |
| 音频干净、图像残缺 | v11g | 0.043200 | 0.155680 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155680 | 1.000000 |
| 双模态残缺 | control | 0.040800 | 0.147500 | 1.000000 |
| 双模态残缺 | v11g | 0.041060 | 0.146780 | 1.000000 |
| 双模态残缺 | no_causal | 0.041400 | 0.148340 | 1.000000 |
| 仅音频残缺 | control | 0.040340 | 0.146620 | 1.000000 |
| 仅音频残缺 | v11g | 0.040340 | 0.146620 | 1.000000 |
| 仅音频残缺 | no_causal | 0.040340 | 0.146620 | 1.000000 |
| 仅图像残缺 | control | 0.015280 | 0.066280 | 0.999900 |
| 仅图像残缺 | v11g | 0.015700 | 0.066220 | 0.994760 |
| 仅图像残缺 | no_causal | 0.015440 | 0.066820 | 0.999380 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11g | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.100% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.340% |
| 图像干净、音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 78.420% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.420% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.240% |
| 双模态残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 77.940% |
| 仅音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 77.940% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 77.940% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 78.700% |
| 仅图像残缺 | v11g | 0.015600 | 0.070500 | 0.987900 | 78.640% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 78.580% |

</details>


<details>
<summary>fixed 五类等权宏平均：各字段有效样本数</summary>

| 实验 | 指标字段（同一计数口径） | 每 family 的 n |
| --- | --- | ---: |
| control | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| control | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| control | `pair_l2`、`pix_var`、`psnr` | 10000 |
| control | `ssim` | 10000 |
| control | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| control | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| control | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| control | `img_visible_mse` | 0, 10000 |
| control | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| control | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| control | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| control | `top15_recall` | 未逐指标保存 |
| v11g | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| v11g | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| v11g | `pair_l2`、`pix_var`、`psnr` | 10000 |
| v11g | `ssim` | 10000 |
| v11g | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| v11g | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| v11g | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| v11g | `img_visible_mse` | 0, 10000 |
| v11g | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| v11g | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| v11g | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| v11g | `top15_recall` | 未逐指标保存 |
| no_causal | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| no_causal | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| no_causal | `pair_l2`、`pix_var`、`psnr` | 10000 |
| no_causal | `ssim` | 10000 |
| no_causal | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| no_causal | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| no_causal | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| no_causal | `img_visible_mse` | 0, 10000 |
| no_causal | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| no_causal | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| no_causal | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| no_causal | `top15_recall` | 未逐指标保存 |

</details>

**分类与恢复：random 单次评估**

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11g | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11g | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965833 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966043 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11g | 98.830% | 0.002053 | 0.985947 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002085 | 0.985614 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 98.960% | 0.009783 | 0.942602 | 0.005872 | 0.816051 |
| 图像干净、音频残缺 | v11g | 98.960% | 0.009783 | 0.942602 | 0.005707 | 0.818191 |
| 图像干净、音频残缺 | no_causal | 98.960% | 0.009783 | 0.942602 | 0.005778 | 0.818126 |
| 音频干净、图像残缺 | control | 99.160% | 0.007417 | 0.956508 | 0.003647 | 0.911193 |
| 音频干净、图像残缺 | v11g | 99.160% | 0.007207 | 0.957610 | 0.003647 | 0.911193 |
| 音频干净、图像残缺 | no_causal | 99.160% | 0.007267 | 0.957267 | 0.003647 | 0.911193 |
| 双模态残缺 | control | 97.280% | 0.007719 | 0.954641 | 0.005927 | 0.820633 |
| 双模态残缺 | v11g | 97.280% | 0.007564 | 0.955441 | 0.005804 | 0.822269 |
| 双模态残缺 | no_causal | 97.280% | 0.007620 | 0.955153 | 0.005849 | 0.822521 |
| 仅音频残缺 | control | 85.790% | 0.012209 | 0.887254 | 0.006055 | 0.813113 |
| 仅音频残缺 | v11g | 85.790% | 0.011811 | 0.894453 | 0.006055 | 0.813113 |
| 仅音频残缺 | no_causal | 85.790% | 0.011455 | 0.899870 | 0.006055 | 0.813113 |
| 仅图像残缺 | control | 92.360% | 0.007614 | 0.955219 | 0.000627 | 0.921694 |
| 仅图像残缺 | v11g | 92.360% | 0.007614 | 0.955219 | 0.000629 | 0.921352 |
| 仅图像残缺 | no_causal | 92.360% | 0.007614 | 0.955219 | 0.000632 | 0.922205 |

random 不与 fixed 混算；表中整行 N/A 表示该实验未找到 random 结果，不表示实验得分为 0。


<details>
<summary>random 单次评估：区域、内容、多样性与音频分布完整指标</summary>

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11g | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11g | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11g | 37.192527 | 0.002053 | 0.002053 | 0.007442 |
| 仅干净音频 | no_causal | 36.885119 | 0.002085 | 0.002085 | 0.007491 |
| 图像干净、音频残缺 | control | 20.770485 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | 20.770485 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.770485 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 22.956013 | 0.058904 | 0.028262 | 0.067230 |
| 音频干净、图像残缺 | v11g | 23.071542 | 0.059632 | 0.027429 | 0.067548 |
| 音频干净、图像残缺 | no_causal | 23.047332 | 0.058533 | 0.027569 | 0.066669 |
| 双模态残缺 | control | 22.756852 | 0.054375 | 0.026184 | 0.062785 |
| 双模态残缺 | v11g | 22.833014 | 0.054636 | 0.025615 | 0.063274 |
| 双模态残缺 | no_causal | 22.808921 | 0.053835 | 0.025760 | 0.062704 |
| 仅音频残缺 | control | 31.275545 | 0.012209 | 0.012209 | 0.023570 |
| 仅音频残缺 | v11g | 31.118603 | 0.011811 | 0.011811 | 0.023849 |
| 仅音频残缺 | no_causal | 30.826284 | 0.011455 | 0.011455 | 0.024285 |
| 仅图像残缺 | control | 22.865423 | 0.060709 | 0.029233 | 0.068927 |
| 仅图像残缺 | v11g | 22.865423 | 0.060709 | 0.029233 | 0.068927 |
| 仅图像残缺 | no_causal | 22.865423 | 0.060709 | 0.029233 | 0.068927 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.028630 | 8.143e-09 | 0.000081 |
| 音频干净、图像残缺 | v11g | 0.028630 | 8.143e-09 | 0.000081 |
| 音频干净、图像残缺 | no_causal | 0.028630 | 8.143e-09 | 0.000081 |
| 双模态残缺 | control | 0.030697 | 8.093e-09 | 0.000081 |
| 双模态残缺 | v11g | 0.030697 | 8.093e-09 | 0.000081 |
| 双模态残缺 | no_causal | 0.030697 | 8.093e-09 | 0.000081 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.029101 | 8.143e-09 | 0.000081 |
| 仅图像残缺 | v11g | 0.029101 | 8.143e-09 | 0.000081 |
| 仅图像残缺 | no_causal | 0.029101 | 8.143e-09 | 0.000081 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11g | 0.000294 | 0.003616 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003565 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.009929 | 0.029169 | 0.003206 | 0.013700 |
| 图像干净、音频残缺 | v11g | 0.009486 | 0.029191 | 0.003206 | 0.013700 |
| 图像干净、音频残缺 | no_causal | 0.009675 | 0.029313 | 0.003206 | 0.013700 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.009910 | 0.028544 | 0.003315 | 0.014139 |
| 双模态残缺 | v11g | 0.009581 | 0.028517 | 0.003315 | 0.014139 |
| 双模态残缺 | no_causal | 0.009699 | 0.028584 | 0.003315 | 0.014139 |
| 仅音频残缺 | control | 0.010381 | 0.029567 | 0.003207 | 0.013704 |
| 仅音频残缺 | v11g | 0.010381 | 0.029567 | 0.003207 | 0.013704 |
| 仅音频残缺 | no_causal | 0.010381 | 0.029567 | 0.003207 | 0.013704 |
| 仅图像残缺 | control | 0.000627 | 0.005133 | N/A | N/A |
| 仅图像残缺 | v11g | 0.000629 | 0.005265 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000632 | 0.005169 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.339940 |
| 干净双模态 | v11g | 97.790% | 96.040% | 0.058851 | 0.339940 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.339940 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.339948 |
| 仅干净图像 | v11g | 96.820% | 94.200% | 0.059386 | 0.339948 |
| 仅干净图像 | no_causal | 96.820% | 94.480% | 0.059386 | 0.339948 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.285884 |
| 仅干净音频 | v11g | 98.090% | 96.000% | 0.045921 | 0.284642 |
| 仅干净音频 | no_causal | 97.710% | 96.000% | 0.046246 | 0.285814 |
| 图像干净、音频残缺 | control | 97.500% | 89.830% | 0.059285 | 0.341653 |
| 图像干净、音频残缺 | v11g | 97.500% | 89.740% | 0.059285 | 0.341653 |
| 图像干净、音频残缺 | no_causal | 97.500% | 90.040% | 0.059285 | 0.341653 |
| 音频干净、图像残缺 | control | 96.690% | 96.000% | 0.061193 | 0.346267 |
| 音频干净、图像残缺 | v11g | 96.730% | 96.000% | 0.060829 | 0.345189 |
| 音频干净、图像残缺 | no_causal | 96.720% | 96.000% | 0.061011 | 0.345697 |
| 双模态残缺 | control | 96.340% | 87.310% | 0.060926 | 0.345003 |
| 双模态残缺 | v11g | 96.330% | 87.460% | 0.060658 | 0.344233 |
| 双模态残缺 | no_causal | 96.310% | 87.620% | 0.060829 | 0.344707 |
| 仅音频残缺 | control | 84.090% | 81.770% | 0.041126 | 0.274684 |
| 仅音频残缺 | v11g | 84.250% | 81.770% | 0.040548 | 0.272443 |
| 仅音频残缺 | no_causal | 84.080% | 81.770% | 0.040358 | 0.271634 |
| 仅图像残缺 | control | 95.490% | 84.270% | 0.060948 | 0.345536 |
| 仅图像残缺 | v11g | 95.490% | 84.630% | 0.060948 | 0.345536 |
| 仅图像残缺 | no_causal | 95.490% | 84.780% | 0.060948 | 0.345536 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11g | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11g | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11g | 0.015600 | 0.068500 | 0.981700 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.996400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.041100 | 0.148400 | 1.000000 |
| 图像干净、音频残缺 | v11g | 0.041300 | 0.147400 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.041700 | 0.149300 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11g | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155700 | 1.000000 |
| 双模态残缺 | control | 0.040800 | 0.147900 | 1.000000 |
| 双模态残缺 | v11g | 0.041000 | 0.147300 | 1.000000 |
| 双模态残缺 | no_causal | 0.041300 | 0.148600 | 1.000000 |
| 仅音频残缺 | control | 0.040600 | 0.147500 | 1.000000 |
| 仅音频残缺 | v11g | 0.040600 | 0.147500 | 1.000000 |
| 仅音频残缺 | no_causal | 0.040600 | 0.147500 | 1.000000 |
| 仅图像残缺 | control | 0.015200 | 0.066200 | 1.000000 |
| 仅图像残缺 | v11g | 0.015600 | 0.066100 | 0.993700 |
| 仅图像残缺 | no_causal | 0.015400 | 0.066800 | 0.999300 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11g | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.100% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.600% |
| 图像干净、音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 78.700% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.700% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.500% |
| 双模态残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 78.600% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.600% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 仅音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.300% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 78.900% |
| 仅图像残缺 | v11g | 0.015600 | 0.070500 | 0.987900 | 78.800% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 78.800% |

</details>


<details>
<summary>random 单次评估：各字段有效样本数</summary>

| 实验 | 指标字段（同一计数口径） | n |
| --- | --- | ---: |
| control | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| control | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| control | `pair_l2`、`pix_var`、`psnr` | 10000 |
| control | `ssim` | 10000 |
| control | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| control | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| control | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| control | `img_visible_mse` | 0, 10000 |
| control | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| control | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| control | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| control | `top15_recall` | 未逐指标保存 |
| v11g | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| v11g | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| v11g | `pair_l2`、`pix_var`、`psnr` | 10000 |
| v11g | `ssim` | 10000 |
| v11g | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| v11g | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| v11g | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| v11g | `img_visible_mse` | 0, 10000 |
| v11g | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| v11g | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| v11g | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| v11g | `top15_recall` | 未逐指标保存 |
| no_causal | `acc`、`aud_mse`、`aud_ssim` | 10000 |
| no_causal | `content_aud_normal_acc`、`content_img_normal_acc`、`img_mse` | 10000 |
| no_causal | `pair_l2`、`pix_var`、`psnr` | 10000 |
| no_causal | `ssim` | 10000 |
| no_causal | `aud_masked_l1`、`aud_masked_mse`、`aud_visible_l1` | 0, 10000 |
| no_causal | `aud_visible_mse`、`img_coarse_masked_mse`、`img_coarse_visible_mse` | 0, 10000 |
| no_causal | `img_masked_l1`、`img_masked_mse`、`img_visible_l1` | 0, 10000 |
| no_causal | `img_visible_mse` | 0, 10000 |
| no_causal | `pair_a2i_r1`、`pair_i2a_r1` | 0 |
| no_causal | `rec_max`、`rec_mean`、`rec_std` | 未逐指标保存 |
| no_causal | `tgt_max`、`tgt_mean`、`tgt_std` | 未逐指标保存 |
| no_causal | `top15_recall` | 未逐指标保存 |

</details>

**Cross-Key / Cross-Detail 干预**

同一 cue/mask 内，gain=zero−normal，wrong damage=wrong−normal，same damage 为有效同类替换上的配对差；正 gain 表示改善。gate 非零本身不代表恢复有效。


<details>
<summary>fixed 五类等权宏平均：干预完整指标</summary>

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11g | 0.002053 | 0.002085 | 0.002050 | 0.002042 |
| 仅干净音频 | no_causal | 0.002085 | 0.002085 | 0.002096 | 0.002081 |
| 音频干净、图像残缺 | control | 0.025918 | 0.025918 | 0.025918 | 0.025916 |
| 音频干净、图像残缺 | v11g | 0.025263 | 0.025918 | 0.025307 | 0.025264 |
| 音频干净、图像残缺 | no_causal | 0.025441 | 0.025918 | 0.025544 | 0.025442 |
| 双模态残缺 | control | 0.025912 | 0.025912 | 0.025912 | 0.025907 |
| 双模态残缺 | v11g | 0.025356 | 0.025912 | 0.025382 | 0.025343 |
| 双模态残缺 | no_causal | 0.025528 | 0.025912 | 0.025601 | 0.025515 |
| 仅音频残缺 | control | 0.013212 | 0.013212 | 0.013212 | 0.013212 |
| 仅音频残缺 | v11g | 0.012756 | 0.013212 | 0.012740 | 0.012734 |
| 仅音频残缺 | no_causal | 0.012351 | 0.013212 | 0.012606 | 0.012577 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11g | 0.000032 | -0.000003 | -0.000012 |
| 仅干净音频 | no_causal | 1.001e-08 | 0.000011 | -0.000005 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.000655 | 0.000043 | 0.000002 |
| 音频干净、图像残缺 | no_causal | 0.000477 | 0.000103 | 0.000003 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000557 | 0.000026 | -0.000008 |
| 双模态残缺 | no_causal | 0.000384 | 0.000073 | -0.000009 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11g | 0.000455 | -0.000017 | -0.000022 |
| 仅音频残缺 | no_causal | 0.000861 | 0.000255 | 0.000226 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11g | 36.370% | 63.070% | 27.930% |
| 仅干净音频 | no_causal | 30.170% | 65.550% | 25.520% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11g | 68.372% | 52.320% | 38.102% |
| 音频干净、图像残缺 | no_causal | 61.138% | 52.908% | 36.356% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 67.536% | 49.806% | 36.534% |
| 双模态残缺 | no_causal | 60.966% | 50.236% | 35.632% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11g | 53.986% | 57.434% | 33.558% |
| 仅音频残缺 | no_causal | 44.568% | 61.516% | 32.076% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11g | 0.534002 | 0.436642 |
| 仅干净音频 | no_causal | 0.757989 | 0.179866 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.563705 | 0.231000 |
| 音频干净、图像残缺 | no_causal | 0.746061 | 0.091677 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.563012 | 0.185046 |
| 双模态残缺 | no_causal | 0.749693 | 0.087927 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11g | 0.535810 | 0.349206 |
| 仅音频残缺 | no_causal | 0.760346 | 0.176541 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11g | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000324 | 0.000301 |
| 图像干净、音频残缺 | control | 0.010599 | 0.010599 | 0.010599 | 0.010600 |
| 图像干净、音频残缺 | v11g | 0.010128 | 0.010599 | 0.010143 | 0.010127 |
| 图像干净、音频残缺 | no_causal | 0.010311 | 0.010599 | 0.010393 | 0.010310 |
| 双模态残缺 | control | 0.010775 | 0.010775 | 0.010775 | 0.010775 |
| 双模态残缺 | v11g | 0.010416 | 0.010775 | 0.010423 | 0.010417 |
| 双模态残缺 | no_causal | 0.010558 | 0.010775 | 0.010606 | 0.010558 |
| 仅图像残缺 | control | 0.000668 | 0.000668 | 0.000668 | 0.000668 |
| 仅图像残缺 | v11g | 0.000670 | 0.000668 | 0.000675 | 0.000672 |
| 仅图像残缺 | no_causal | 0.000674 | 0.000668 | 0.000687 | 0.000683 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11g | -0.000005 | 0.000010 | 7.588e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000028 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.000471 | 0.000015 | -0.000002 |
| 图像干净、音频残缺 | no_causal | 0.000288 | 0.000082 | -0.000002 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000359 | 0.000007 | 5.457e-07 |
| 双模态残缺 | no_causal | 0.000216 | 0.000047 | -3.955e-07 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11g | -0.000002 | 0.000005 | 0.000002 |
| 仅图像残缺 | no_causal | -0.000005 | 0.000013 | 0.000009 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11g | 35.190% | 69.260% | 26.800% |
| 仅干净图像 | no_causal | 28.510% | 71.180% | 23.370% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11g | 67.030% | 43.610% | 36.314% |
| 图像干净、音频残缺 | no_causal | 54.438% | 45.348% | 32.890% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 67.740% | 43.006% | 35.866% |
| 双模态残缺 | no_causal | 54.652% | 44.612% | 32.084% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11g | 32.878% | 66.106% | 23.774% |
| 仅图像残缺 | no_causal | 38.516% | 65.556% | 27.536% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11g | 0.947892 | 0.040982 |
| 仅干净图像 | no_causal | 0.961301 | 0.027724 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.947075 | 0.015465 |
| 图像干净、音频残缺 | no_causal | 0.959665 | 0.011053 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.944893 | 0.012324 |
| 双模态残缺 | no_causal | 0.957989 | 0.008755 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11g | 0.946077 | 0.032254 |
| 仅图像残缺 | no_causal | 0.960106 | 0.021593 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11g | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11g | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11g | 98.090% | 97.710% | 97.490% | 97.499% |
| 仅干净音频 | no_causal | 97.710% | 97.710% | 97.710% | 97.579% |
| 图像干净、音频残缺 | control | 97.520% | 97.520% | 97.520% | 97.519% |
| 图像干净、音频残缺 | v11g | 97.520% | 97.520% | 97.520% | 97.519% |
| 图像干净、音频残缺 | no_causal | 97.520% | 97.520% | 97.520% | 97.519% |
| 音频干净、图像残缺 | control | 96.720% | 96.720% | 96.720% | 96.719% |
| 音频干净、图像残缺 | v11g | 96.750% | 96.720% | 96.738% | 96.761% |
| 音频干净、图像残缺 | no_causal | 96.754% | 96.720% | 96.684% | 96.703% |
| 双模态残缺 | control | 96.178% | 96.178% | 96.178% | 96.176% |
| 双模态残缺 | v11g | 96.266% | 96.178% | 96.282% | 96.295% |
| 双模态残缺 | no_causal | 96.258% | 96.178% | 96.246% | 96.293% |
| 仅音频残缺 | control | 82.882% | 82.882% | 82.882% | 82.881% |
| 仅音频残缺 | v11g | 83.066% | 82.882% | 82.912% | 82.945% |
| 仅音频残缺 | no_causal | 82.942% | 82.882% | 82.896% | 82.885% |
| 仅图像残缺 | control | 95.514% | 95.514% | 95.514% | 95.512% |
| 仅图像残缺 | v11g | 95.514% | 95.514% | 95.514% | 95.512% |
| 仅图像残缺 | no_causal | 95.514% | 95.514% | 95.514% | 95.512% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11g | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11g | 94.200% | 94.320% | 94.080% | 94.088% |
| 仅干净图像 | no_causal | 94.480% | 94.320% | 94.690% | 94.538% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11g | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 88.970% | 88.970% | 88.970% | 88.968% |
| 图像干净、音频残缺 | v11g | 88.936% | 88.970% | 88.768% | 88.880% |
| 图像干净、音频残缺 | no_causal | 89.218% | 88.970% | 88.906% | 89.158% |
| 音频干净、图像残缺 | control | 95.968% | 95.968% | 95.968% | 95.966% |
| 音频干净、图像残缺 | v11g | 95.968% | 95.968% | 95.968% | 95.966% |
| 音频干净、图像残缺 | no_causal | 95.968% | 95.968% | 95.968% | 95.966% |
| 双模态残缺 | control | 87.012% | 87.012% | 87.012% | 87.009% |
| 双模态残缺 | v11g | 87.022% | 87.012% | 87.004% | 86.997% |
| 双模态残缺 | no_causal | 87.150% | 87.012% | 87.012% | 87.191% |
| 仅音频残缺 | control | 80.600% | 80.600% | 80.600% | 80.600% |
| 仅音频残缺 | v11g | 80.600% | 80.600% | 80.600% | 80.600% |
| 仅音频残缺 | no_causal | 80.600% | 80.600% | 80.600% | 80.600% |
| 仅图像残缺 | control | 83.518% | 83.518% | 83.518% | 83.513% |
| 仅图像残缺 | v11g | 83.468% | 83.518% | 83.444% | 83.451% |
| 仅图像残缺 | no_causal | 84.086% | 83.518% | 84.312% | 84.128% |

</details>


<details>
<summary>fixed 五类等权宏平均：干预各字段有效 n</summary>

| 实验 | 指标字段（同一计数口径） | 每 family 的 n |
| --- | --- | ---: |
| control | `aud2img_correct_gain`、`aud2img_normal_mse`、`aud2img_ratio` | 10000 |
| control | `aud2img_win_both`、`aud2img_win_wrong`、`aud2img_win_zero` | 10000 |
| control | `aud2img_wrong_damage`、`aud2img_wrong_mse`、`aud2img_zero_mse` | 10000 |
| control | `content_aud_normal_acc`、`content_aud_wrong_acc`、`content_aud_zero_acc` | 10000 |
| control | `content_img_normal_acc`、`content_img_wrong_acc`、`content_img_zero_acc` | 10000 |
| control | `img2aud_correct_gain`、`img2aud_normal_mse`、`img2aud_ratio` | 10000 |
| control | `img2aud_win_both`、`img2aud_win_wrong`、`img2aud_win_zero` | 10000 |
| control | `img2aud_wrong_damage`、`img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| control | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| control | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |
| v11g | `aud2img_correct_gain`、`aud2img_gate`、`aud2img_normal_mse` | 10000 |
| v11g | `aud2img_ratio`、`aud2img_win_both`、`aud2img_win_wrong` | 10000 |
| v11g | `aud2img_win_zero`、`aud2img_wrong_damage`、`aud2img_wrong_mse` | 10000 |
| v11g | `aud2img_zero_mse`、`content_aud_normal_acc`、`content_aud_wrong_acc` | 10000 |
| v11g | `content_aud_zero_acc`、`content_img_normal_acc`、`content_img_wrong_acc` | 10000 |
| v11g | `content_img_zero_acc`、`img2aud_correct_gain`、`img2aud_gate` | 10000 |
| v11g | `img2aud_normal_mse`、`img2aud_ratio`、`img2aud_win_both` | 10000 |
| v11g | `img2aud_win_wrong`、`img2aud_win_zero`、`img2aud_wrong_damage` | 10000 |
| v11g | `img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| v11g | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| v11g | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |
| no_causal | `aud2img_correct_gain`、`aud2img_gate`、`aud2img_normal_mse` | 10000 |
| no_causal | `aud2img_ratio`、`aud2img_win_both`、`aud2img_win_wrong` | 10000 |
| no_causal | `aud2img_win_zero`、`aud2img_wrong_damage`、`aud2img_wrong_mse` | 10000 |
| no_causal | `aud2img_zero_mse`、`content_aud_normal_acc`、`content_aud_wrong_acc` | 10000 |
| no_causal | `content_aud_zero_acc`、`content_img_normal_acc`、`content_img_wrong_acc` | 10000 |
| no_causal | `content_img_zero_acc`、`img2aud_correct_gain`、`img2aud_gate` | 10000 |
| no_causal | `img2aud_normal_mse`、`img2aud_ratio`、`img2aud_win_both` | 10000 |
| no_causal | `img2aud_win_wrong`、`img2aud_win_zero`、`img2aud_wrong_damage` | 10000 |
| no_causal | `img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| no_causal | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| no_causal | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |

</details>


<details>
<summary>random：干预完整指标</summary>

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11g | 0.002053 | 0.002085 | 0.002050 | 0.002042 |
| 仅干净音频 | no_causal | 0.002085 | 0.002085 | 0.002096 | 0.002081 |
| 音频干净、图像残缺 | control | 0.028262 | 0.028262 | 0.028262 | 0.028243 |
| 音频干净、图像残缺 | v11g | 0.027429 | 0.028262 | 0.027445 | 0.027408 |
| 音频干净、图像残缺 | no_causal | 0.027569 | 0.028262 | 0.027648 | 0.027550 |
| 双模态残缺 | control | 0.026184 | 0.026184 | 0.026184 | 0.026159 |
| 双模态残缺 | v11g | 0.025615 | 0.026184 | 0.025633 | 0.025581 |
| 双模态残缺 | no_causal | 0.025760 | 0.026184 | 0.025837 | 0.025731 |
| 仅音频残缺 | control | 0.012209 | 0.012209 | 0.012209 | 0.012191 |
| 仅音频残缺 | v11g | 0.011811 | 0.012209 | 0.011800 | 0.011778 |
| 仅音频残缺 | no_causal | 0.011455 | 0.012209 | 0.011678 | 0.011658 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11g | 0.000032 | -0.000003 | -0.000012 |
| 仅干净音频 | no_causal | 1.001e-08 | 0.000011 | -0.000005 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.000833 | 0.000015 | -0.000003 |
| 音频干净、图像残缺 | no_causal | 0.000693 | 0.000079 | -0.000002 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000569 | 0.000018 | -0.000011 |
| 双模态残缺 | no_causal | 0.000424 | 0.000078 | -0.000008 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11g | 0.000398 | -0.000011 | -0.000016 |
| 仅音频残缺 | no_causal | 0.000753 | 0.000223 | 0.000217 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11g | 36.370% | 63.070% | 27.930% |
| 仅干净音频 | no_causal | 30.170% | 65.550% | 25.520% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11g | 70.960% | 51.900% | 38.730% |
| 音频干净、图像残缺 | no_causal | 64.220% | 53.180% | 38.060% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 66.980% | 49.910% | 36.250% |
| 双模态残缺 | no_causal | 60.210% | 51.770% | 35.870% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11g | 52.860% | 57.040% | 33.200% |
| 仅音频残缺 | no_causal | 43.560% | 61.040% | 31.420% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11g | 0.534002 | 0.436642 |
| 仅干净音频 | no_causal | 0.757989 | 0.179866 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.562402 | 0.217568 |
| 音频干净、图像残缺 | no_causal | 0.740382 | 0.086494 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.563208 | 0.188709 |
| 双模态残缺 | no_causal | 0.748743 | 0.086983 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11g | 0.535583 | 0.352793 |
| 仅音频残缺 | no_causal | 0.760123 | 0.174541 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11g | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000324 | 0.000301 |
| 图像干净、音频残缺 | control | 0.009929 | 0.009929 | 0.009929 | 0.009922 |
| 图像干净、音频残缺 | v11g | 0.009486 | 0.009929 | 0.009495 | 0.009476 |
| 图像干净、音频残缺 | no_causal | 0.009675 | 0.009929 | 0.009740 | 0.009659 |
| 双模态残缺 | control | 0.009910 | 0.009910 | 0.009910 | 0.009903 |
| 双模态残缺 | v11g | 0.009581 | 0.009910 | 0.009584 | 0.009572 |
| 双模态残缺 | no_causal | 0.009699 | 0.009910 | 0.009735 | 0.009685 |
| 仅图像残缺 | control | 0.000627 | 0.000627 | 0.000627 | 0.000628 |
| 仅图像残缺 | v11g | 0.000629 | 0.000627 | 0.000635 | 0.000631 |
| 仅图像残缺 | no_causal | 0.000632 | 0.000627 | 0.000646 | 0.000641 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11g | -0.000005 | 0.000010 | 7.588e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000028 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.000443 | 0.000009 | -0.000002 |
| 图像干净、音频残缺 | no_causal | 0.000254 | 0.000065 | -0.000008 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000329 | 0.000003 | -0.000002 |
| 双模态残缺 | no_causal | 0.000212 | 0.000036 | -0.000006 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11g | -0.000002 | 0.000005 | 0.000002 |
| 仅图像残缺 | no_causal | -0.000004 | 0.000014 | 0.000009 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11g | 35.190% | 69.260% | 26.800% |
| 仅干净图像 | no_causal | 28.510% | 71.180% | 23.370% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11g | 69.310% | 43.690% | 36.780% |
| 图像干净、音频残缺 | no_causal | 55.910% | 44.740% | 32.110% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 66.350% | 41.870% | 35.300% |
| 双模态残缺 | no_causal | 53.300% | 42.590% | 30.470% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11g | 33.090% | 66.760% | 24.090% |
| 仅图像残缺 | no_causal | 37.990% | 66.120% | 28.010% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11g | 0.947892 | 0.040982 |
| 仅干净图像 | no_causal | 0.961301 | 0.027724 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.946372 | 0.017210 |
| 图像干净、音频残缺 | no_causal | 0.958783 | 0.012209 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.946352 | 0.012240 |
| 双模态残缺 | no_causal | 0.959606 | 0.008705 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11g | 0.946163 | 0.032303 |
| 仅图像残缺 | no_causal | 0.960165 | 0.021626 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11g | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11g | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11g | 98.090% | 97.710% | 97.490% | 97.499% |
| 仅干净音频 | no_causal | 97.710% | 97.710% | 97.710% | 97.579% |
| 图像干净、音频残缺 | control | 97.500% | 97.500% | 97.500% | 97.499% |
| 图像干净、音频残缺 | v11g | 97.500% | 97.500% | 97.500% | 97.499% |
| 图像干净、音频残缺 | no_causal | 97.500% | 97.500% | 97.500% | 97.499% |
| 音频干净、图像残缺 | control | 96.690% | 96.690% | 96.690% | 96.689% |
| 音频干净、图像残缺 | v11g | 96.730% | 96.690% | 96.780% | 96.779% |
| 音频干净、图像残缺 | no_causal | 96.720% | 96.690% | 96.810% | 96.809% |
| 双模态残缺 | control | 96.340% | 96.340% | 96.340% | 96.339% |
| 双模态残缺 | v11g | 96.330% | 96.340% | 96.410% | 96.289% |
| 双模态残缺 | no_causal | 96.310% | 96.340% | 96.230% | 96.399% |
| 仅音频残缺 | control | 84.090% | 84.090% | 84.090% | 84.114% |
| 仅音频残缺 | v11g | 84.250% | 84.090% | 84.100% | 84.194% |
| 仅音频残缺 | no_causal | 84.080% | 84.090% | 84.210% | 84.144% |
| 仅图像残缺 | control | 95.490% | 95.490% | 95.490% | 95.488% |
| 仅图像残缺 | v11g | 95.490% | 95.490% | 95.490% | 95.488% |
| 仅图像残缺 | no_causal | 95.490% | 95.490% | 95.490% | 95.488% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11g | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11g | 94.200% | 94.320% | 94.080% | 94.088% |
| 仅干净图像 | no_causal | 94.480% | 94.320% | 94.690% | 94.538% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11g | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 89.830% | 89.830% | 89.830% | 89.826% |
| 图像干净、音频残缺 | v11g | 89.740% | 89.830% | 89.430% | 89.546% |
| 图像干净、音频残缺 | no_causal | 90.040% | 89.830% | 89.680% | 89.966% |
| 音频干净、图像残缺 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 音频干净、图像残缺 | v11g | 96.000% | 96.000% | 96.000% | 95.998% |
| 音频干净、图像残缺 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 双模态残缺 | control | 87.310% | 87.310% | 87.310% | 87.305% |
| 双模态残缺 | v11g | 87.460% | 87.310% | 87.430% | 87.495% |
| 双模态残缺 | no_causal | 87.620% | 87.310% | 87.400% | 87.485% |
| 仅音频残缺 | control | 81.770% | 81.770% | 81.770% | 81.783% |
| 仅音频残缺 | v11g | 81.770% | 81.770% | 81.770% | 81.783% |
| 仅音频残缺 | no_causal | 81.770% | 81.770% | 81.770% | 81.783% |
| 仅图像残缺 | control | 84.270% | 84.270% | 84.270% | 84.264% |
| 仅图像残缺 | v11g | 84.630% | 84.270% | 84.560% | 84.574% |
| 仅图像残缺 | no_causal | 84.780% | 84.270% | 85.110% | 84.904% |

</details>


<details>
<summary>random：干预各字段有效 n</summary>

| 实验 | 指标字段（同一计数口径） | n |
| --- | --- | ---: |
| control | `aud2img_correct_gain`、`aud2img_normal_mse`、`aud2img_ratio` | 10000 |
| control | `aud2img_win_both`、`aud2img_win_wrong`、`aud2img_win_zero` | 10000 |
| control | `aud2img_wrong_damage`、`aud2img_wrong_mse`、`aud2img_zero_mse` | 10000 |
| control | `content_aud_normal_acc`、`content_aud_wrong_acc`、`content_aud_zero_acc` | 10000 |
| control | `content_img_normal_acc`、`content_img_wrong_acc`、`content_img_zero_acc` | 10000 |
| control | `img2aud_correct_gain`、`img2aud_normal_mse`、`img2aud_ratio` | 10000 |
| control | `img2aud_win_both`、`img2aud_win_wrong`、`img2aud_win_zero` | 10000 |
| control | `img2aud_wrong_damage`、`img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| control | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| control | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |
| v11g | `aud2img_correct_gain`、`aud2img_gate`、`aud2img_normal_mse` | 10000 |
| v11g | `aud2img_ratio`、`aud2img_win_both`、`aud2img_win_wrong` | 10000 |
| v11g | `aud2img_win_zero`、`aud2img_wrong_damage`、`aud2img_wrong_mse` | 10000 |
| v11g | `aud2img_zero_mse`、`content_aud_normal_acc`、`content_aud_wrong_acc` | 10000 |
| v11g | `content_aud_zero_acc`、`content_img_normal_acc`、`content_img_wrong_acc` | 10000 |
| v11g | `content_img_zero_acc`、`img2aud_correct_gain`、`img2aud_gate` | 10000 |
| v11g | `img2aud_normal_mse`、`img2aud_ratio`、`img2aud_win_both` | 10000 |
| v11g | `img2aud_win_wrong`、`img2aud_win_zero`、`img2aud_wrong_damage` | 10000 |
| v11g | `img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| v11g | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| v11g | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |
| no_causal | `aud2img_correct_gain`、`aud2img_gate`、`aud2img_normal_mse` | 10000 |
| no_causal | `aud2img_ratio`、`aud2img_win_both`、`aud2img_win_wrong` | 10000 |
| no_causal | `aud2img_win_zero`、`aud2img_wrong_damage`、`aud2img_wrong_mse` | 10000 |
| no_causal | `aud2img_zero_mse`、`content_aud_normal_acc`、`content_aud_wrong_acc` | 10000 |
| no_causal | `content_aud_zero_acc`、`content_img_normal_acc`、`content_img_wrong_acc` | 10000 |
| no_causal | `content_img_zero_acc`、`img2aud_correct_gain`、`img2aud_gate` | 10000 |
| no_causal | `img2aud_normal_mse`、`img2aud_ratio`、`img2aud_win_both` | 10000 |
| no_causal | `img2aud_win_wrong`、`img2aud_win_zero`、`img2aud_wrong_damage` | 10000 |
| no_causal | `img2aud_wrong_mse`、`img2aud_zero_mse` | 10000 |
| no_causal | `aud2img_same_class_mse`、`aud2img_same_damage`、`content_aud_same_class_acc` | 9996 |
| no_causal | `content_img_same_class_acc`、`img2aud_same_class_mse`、`img2aud_same_damage` | 9996 |

</details>

**逐 family 完整分项**

以下是与主表同源的五组 image/audio family pair，每组内仍按输入模式、实验排列。每组约 10000 个 MNIST 测试条目；音频会复用，旧版有效 n 未逐指标保存。


<details>
<summary>family 1：occlusion/time_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11g | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11g | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965833 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966043 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11g | 98.830% | 0.002053 | 0.985947 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002085 | 0.985614 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 99.140% | 0.009767 | 0.942667 | 0.006178 | 0.834596 |
| 图像干净、音频残缺 | v11g | 99.140% | 0.009767 | 0.942667 | 0.006012 | 0.837069 |
| 图像干净、音频残缺 | no_causal | 99.140% | 0.009767 | 0.942667 | 0.006037 | 0.838124 |
| 音频干净、图像残缺 | control | 99.130% | 0.008792 | 0.948068 | 0.003647 | 0.911187 |
| 音频干净、图像残缺 | v11g | 99.130% | 0.008541 | 0.949360 | 0.003647 | 0.911187 |
| 音频干净、图像残缺 | no_causal | 99.130% | 0.008544 | 0.949418 | 0.003647 | 0.911187 |
| 双模态残缺 | control | 97.850% | 0.008731 | 0.948098 | 0.006377 | 0.827549 |
| 双模态残缺 | v11g | 97.850% | 0.008501 | 0.949346 | 0.006236 | 0.829691 |
| 双模态残缺 | no_causal | 97.850% | 0.008526 | 0.949318 | 0.006254 | 0.830707 |
| 仅音频残缺 | control | 90.500% | 0.009049 | 0.925291 | 0.006279 | 0.831989 |
| 仅音频残缺 | v11g | 90.500% | 0.008833 | 0.928694 | 0.006279 | 0.831989 |
| 仅音频残缺 | no_causal | 90.500% | 0.008784 | 0.929215 | 0.006279 | 0.831989 |
| 仅图像残缺 | control | 89.790% | 0.009160 | 0.945520 | 0.000822 | 0.900165 |
| 仅图像残缺 | v11g | 89.790% | 0.009160 | 0.945520 | 0.000822 | 0.900144 |
| 仅图像残缺 | no_causal | 89.790% | 0.009160 | 0.945520 | 0.000830 | 0.900623 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11g | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11g | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11g | 37.192527 | 0.002053 | 0.002053 | 0.007442 |
| 仅干净音频 | no_causal | 36.885119 | 0.002085 | 0.002085 | 0.007491 |
| 图像干净、音频残缺 | control | 20.777420 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | 20.777420 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.777420 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 23.028585 | 0.098624 | 0.056964 | 0.127488 |
| 音频干净、图像残缺 | v11g | 23.110824 | 0.099808 | 0.055339 | 0.128548 |
| 音频干净、图像残缺 | no_causal | 23.150764 | 0.096593 | 0.055360 | 0.126359 |
| 双模态残缺 | control | 22.973645 | 0.098165 | 0.056573 | 0.126785 |
| 双模态残缺 | v11g | 23.051802 | 0.098690 | 0.055082 | 0.127753 |
| 双模态残缺 | no_causal | 23.073641 | 0.095734 | 0.055244 | 0.125877 |
| 仅音频残缺 | control | 32.892612 | 0.009049 | 0.009049 | 0.018964 |
| 仅音频残缺 | v11g | 32.673049 | 0.008833 | 0.008833 | 0.019255 |
| 仅音频残缺 | no_causal | 32.303378 | 0.008784 | 0.008784 | 0.019470 |
| 仅图像残缺 | control | 22.868519 | 0.102796 | 0.059348 | 0.131669 |
| 仅图像残缺 | v11g | 22.868519 | 0.102796 | 0.059348 | 0.131669 |
| 仅图像残缺 | no_causal | 22.868519 | 0.102796 | 0.059348 | 0.131669 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.015330 | 8.420e-09 | 0.000084 |
| 音频干净、图像残缺 | v11g | 0.015330 | 8.420e-09 | 0.000084 |
| 音频干净、图像残缺 | no_causal | 0.015330 | 8.420e-09 | 0.000084 |
| 双模态残缺 | control | 0.015193 | 8.415e-09 | 0.000084 |
| 双模态残缺 | v11g | 0.015193 | 8.415e-09 | 0.000084 |
| 双模态残缺 | no_causal | 0.015193 | 8.415e-09 | 0.000084 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.015612 | 8.420e-09 | 0.000084 |
| 仅图像残缺 | v11g | 0.015612 | 8.420e-09 | 0.000084 |
| 仅图像残缺 | no_causal | 0.015612 | 8.420e-09 | 0.000084 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11g | 0.000294 | 0.003616 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003565 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.009056 | 0.023760 | 0.004209 | 0.017673 |
| 图像干净、音频残缺 | v11g | 0.008648 | 0.023671 | 0.004209 | 0.017673 |
| 图像干净、音频残缺 | no_causal | 0.008708 | 0.023684 | 0.004209 | 0.017673 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.009616 | 0.025095 | 0.004161 | 0.017356 |
| 双模态残缺 | v11g | 0.009269 | 0.025017 | 0.004161 | 0.017356 |
| 双模态残缺 | no_causal | 0.009315 | 0.025023 | 0.004161 | 0.017356 |
| 仅音频残缺 | control | 0.009326 | 0.024048 | 0.004193 | 0.017643 |
| 仅音频残缺 | v11g | 0.009326 | 0.024048 | 0.004193 | 0.017643 |
| 仅音频残缺 | no_causal | 0.009326 | 0.024048 | 0.004193 | 0.017643 |
| 仅图像残缺 | control | 0.000822 | 0.005739 | N/A | N/A |
| 仅图像残缺 | v11g | 0.000822 | 0.005877 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000830 | 0.005795 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.339353 |
| 干净双模态 | v11g | 97.790% | 96.040% | 0.058851 | 0.339353 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.339353 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.338850 |
| 仅干净图像 | v11g | 96.820% | 94.200% | 0.059386 | 0.338850 |
| 仅干净图像 | no_causal | 96.820% | 94.480% | 0.059386 | 0.338850 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.287092 |
| 仅干净音频 | v11g | 98.090% | 96.000% | 0.045921 | 0.285914 |
| 仅干净音频 | no_causal | 97.710% | 96.000% | 0.046246 | 0.287088 |
| 图像干净、音频残缺 | control | 97.610% | 89.110% | 0.059190 | 0.339690 |
| 图像干净、音频残缺 | v11g | 97.610% | 89.140% | 0.059190 | 0.339690 |
| 图像干净、音频残缺 | no_causal | 97.610% | 89.410% | 0.059190 | 0.339690 |
| 音频干净、图像残缺 | control | 95.920% | 95.920% | 0.060249 | 0.343568 |
| 音频干净、图像残缺 | v11g | 96.350% | 95.920% | 0.059805 | 0.342219 |
| 音频干净、图像残缺 | no_causal | 96.490% | 95.920% | 0.060233 | 0.343407 |
| 双模态残缺 | control | 95.140% | 87.600% | 0.060214 | 0.341794 |
| 双模态残缺 | v11g | 95.480% | 87.650% | 0.059875 | 0.340793 |
| 双模态残缺 | no_causal | 95.660% | 87.830% | 0.060281 | 0.341993 |
| 仅音频残缺 | control | 89.060% | 86.090% | 0.042524 | 0.283295 |
| 仅音频残缺 | v11g | 89.390% | 86.090% | 0.042110 | 0.281868 |
| 仅音频残缺 | no_causal | 89.270% | 86.090% | 0.042191 | 0.282154 |
| 仅图像残缺 | control | 93.750% | 82.120% | 0.059824 | 0.342283 |
| 仅图像残缺 | v11g | 93.750% | 81.980% | 0.059824 | 0.342283 |
| 仅图像残缺 | no_causal | 93.750% | 82.490% | 0.059824 | 0.342283 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11g | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11g | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11g | 0.015600 | 0.068500 | 0.981700 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.996400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.040000 | 0.147100 | 1.000000 |
| 图像干净、音频残缺 | v11g | 0.040400 | 0.146600 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.040700 | 0.148000 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11g | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155700 | 1.000000 |
| 双模态残缺 | control | 0.039900 | 0.146800 | 1.000000 |
| 双模态残缺 | v11g | 0.040200 | 0.146300 | 1.000000 |
| 双模态残缺 | no_causal | 0.040500 | 0.147500 | 1.000000 |
| 仅音频残缺 | control | 0.039800 | 0.146600 | 1.000000 |
| 仅音频残缺 | v11g | 0.039800 | 0.146600 | 1.000000 |
| 仅音频残缺 | no_causal | 0.039800 | 0.146600 | 1.000000 |
| 仅图像残缺 | control | 0.015000 | 0.065400 | 1.000000 |
| 仅图像残缺 | v11g | 0.015500 | 0.065400 | 0.992000 |
| 仅图像残缺 | no_causal | 0.015200 | 0.066100 | 0.999500 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11g | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.100% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 78.100% |
| 图像干净、音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 78.200% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 78.200% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 77.800% |
| 双模态残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 77.900% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 77.800% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 77.900% |
| 仅音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 77.900% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 77.900% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 78.300% |
| 仅图像残缺 | v11g | 0.015600 | 0.070500 | 0.987900 | 78.300% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 78.200% |

**本 family 干预**

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11g | 0.002053 | 0.002085 | 0.002050 | 0.002042 |
| 仅干净音频 | no_causal | 0.002085 | 0.002085 | 0.002096 | 0.002081 |
| 音频干净、图像残缺 | control | 0.056964 | 0.056964 | 0.056964 | 0.056959 |
| 音频干净、图像残缺 | v11g | 0.055339 | 0.056964 | 0.055444 | 0.055343 |
| 音频干净、图像残缺 | no_causal | 0.055360 | 0.056964 | 0.055673 | 0.055375 |
| 双模态残缺 | control | 0.056573 | 0.056573 | 0.056573 | 0.056550 |
| 双模态残缺 | v11g | 0.055082 | 0.056573 | 0.055180 | 0.055040 |
| 双模态残缺 | no_causal | 0.055244 | 0.056573 | 0.055500 | 0.055207 |
| 仅音频残缺 | control | 0.009049 | 0.009049 | 0.009049 | 0.009050 |
| 仅音频残缺 | v11g | 0.008833 | 0.009049 | 0.008773 | 0.008757 |
| 仅音频残缺 | no_causal | 0.008784 | 0.009049 | 0.008888 | 0.008868 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11g | 0.000032 | -0.000003 | -0.000012 |
| 仅干净音频 | no_causal | 1.001e-08 | 0.000011 | -0.000005 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.001625 | 0.000105 | 0.000007 |
| 音频干净、图像残缺 | no_causal | 0.001604 | 0.000313 | 0.000017 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.001491 | 0.000098 | -0.000021 |
| 双模态残缺 | no_causal | 0.001329 | 0.000256 | -0.000018 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11g | 0.000216 | -0.000060 | -0.000077 |
| 仅音频残缺 | no_causal | 0.000265 | 0.000104 | 0.000083 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11g | 36.370% | 63.070% | 27.930% |
| 仅干净音频 | no_causal | 30.170% | 65.550% | 25.520% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11g | 64.910% | 51.470% | 36.130% |
| 音频干净、图像残缺 | no_causal | 61.690% | 53.810% | 37.810% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 65.530% | 52.260% | 37.170% |
| 双模态残缺 | no_causal | 62.440% | 52.910% | 38.410% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11g | 49.750% | 58.110% | 32.820% |
| 仅音频残缺 | no_causal | 39.660% | 63.810% | 30.730% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11g | 0.534002 | 0.436642 |
| 仅干净音频 | no_causal | 0.757989 | 0.179866 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.559727 | 0.104021 |
| 音频干净、图像残缺 | no_causal | 0.687750 | 0.041843 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.558517 | 0.085549 |
| 双模态残缺 | no_causal | 0.693397 | 0.036836 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11g | 0.534741 | 0.368449 |
| 仅音频残缺 | no_causal | 0.759678 | 0.166590 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11g | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000324 | 0.000301 |
| 图像干净、音频残缺 | control | 0.009056 | 0.009056 | 0.009056 | 0.009057 |
| 图像干净、音频残缺 | v11g | 0.008648 | 0.009056 | 0.008663 | 0.008644 |
| 图像干净、音频残缺 | no_causal | 0.008708 | 0.009056 | 0.008765 | 0.008705 |
| 双模态残缺 | control | 0.009616 | 0.009616 | 0.009616 | 0.009618 |
| 双模态残缺 | v11g | 0.009269 | 0.009616 | 0.009284 | 0.009272 |
| 双模态残缺 | no_causal | 0.009315 | 0.009616 | 0.009371 | 0.009318 |
| 仅图像残缺 | control | 0.000822 | 0.000822 | 0.000822 | 0.000821 |
| 仅图像残缺 | v11g | 0.000822 | 0.000822 | 0.000828 | 0.000824 |
| 仅图像残缺 | no_causal | 0.000830 | 0.000822 | 0.000847 | 0.000844 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11g | -0.000005 | 0.000010 | 7.588e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000028 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.000408 | 0.000015 | -0.000005 |
| 图像干净、音频残缺 | no_causal | 0.000348 | 0.000056 | -0.000004 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000347 | 0.000015 | 0.000002 |
| 双模态残缺 | no_causal | 0.000301 | 0.000056 | 0.000002 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11g | -1.299e-08 | 0.000006 | 0.000003 |
| 仅图像残缺 | no_causal | -0.000008 | 0.000018 | 0.000014 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11g | 35.190% | 69.260% | 26.800% |
| 仅干净图像 | no_causal | 28.510% | 71.180% | 23.370% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11g | 51.990% | 34.830% | 28.520% |
| 图像干净、音频残缺 | no_causal | 45.480% | 35.640% | 27.020% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 53.440% | 35.340% | 29.010% |
| 双模态残缺 | no_causal | 47.430% | 36.090% | 27.540% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11g | 35.730% | 64.900% | 25.010% |
| 仅图像残缺 | no_causal | 37.520% | 65.660% | 27.460% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11g | 0.947892 | 0.040982 |
| 仅干净图像 | no_causal | 0.961301 | 0.027724 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.957384 | 0.011939 |
| 图像干净、音频残缺 | no_causal | 0.971644 | 0.009225 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.956368 | 0.009723 |
| 双模态残缺 | no_causal | 0.970794 | 0.007494 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11g | 0.947147 | 0.033570 |
| 仅图像残缺 | no_causal | 0.960839 | 0.022473 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11g | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11g | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11g | 98.090% | 97.710% | 97.490% | 97.499% |
| 仅干净音频 | no_causal | 97.710% | 97.710% | 97.710% | 97.579% |
| 图像干净、音频残缺 | control | 97.610% | 97.610% | 97.610% | 97.609% |
| 图像干净、音频残缺 | v11g | 97.610% | 97.610% | 97.610% | 97.609% |
| 图像干净、音频残缺 | no_causal | 97.610% | 97.610% | 97.610% | 97.609% |
| 音频干净、图像残缺 | control | 95.920% | 95.920% | 95.920% | 95.918% |
| 音频干净、图像残缺 | v11g | 96.350% | 95.920% | 96.350% | 96.279% |
| 音频干净、图像残缺 | no_causal | 96.490% | 95.920% | 96.300% | 96.218% |
| 双模态残缺 | control | 95.140% | 95.140% | 95.140% | 95.138% |
| 双模态残缺 | v11g | 95.480% | 95.140% | 95.390% | 95.538% |
| 双模态残缺 | no_causal | 95.660% | 95.140% | 95.390% | 95.558% |
| 仅音频残缺 | control | 89.060% | 89.060% | 89.060% | 89.056% |
| 仅音频残缺 | v11g | 89.390% | 89.060% | 89.080% | 89.206% |
| 仅音频残缺 | no_causal | 89.270% | 89.060% | 89.170% | 89.206% |
| 仅图像残缺 | control | 93.750% | 93.750% | 93.750% | 93.747% |
| 仅图像残缺 | v11g | 93.750% | 93.750% | 93.750% | 93.747% |
| 仅图像残缺 | no_causal | 93.750% | 93.750% | 93.750% | 93.747% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11g | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11g | 94.200% | 94.320% | 94.080% | 94.088% |
| 仅干净图像 | no_causal | 94.480% | 94.320% | 94.690% | 94.538% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11g | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 89.110% | 89.110% | 89.110% | 89.116% |
| 图像干净、音频残缺 | v11g | 89.140% | 89.110% | 89.120% | 89.146% |
| 图像干净、音频残缺 | no_causal | 89.410% | 89.110% | 89.090% | 89.476% |
| 音频干净、图像残缺 | control | 95.920% | 95.920% | 95.920% | 95.918% |
| 音频干净、图像残缺 | v11g | 95.920% | 95.920% | 95.920% | 95.918% |
| 音频干净、图像残缺 | no_causal | 95.920% | 95.920% | 95.920% | 95.918% |
| 双模态残缺 | control | 87.600% | 87.600% | 87.600% | 87.595% |
| 双模态残缺 | v11g | 87.650% | 87.600% | 87.830% | 87.785% |
| 双模态残缺 | no_causal | 87.830% | 87.600% | 87.810% | 87.995% |
| 仅音频残缺 | control | 86.090% | 86.090% | 86.090% | 86.094% |
| 仅音频残缺 | v11g | 86.090% | 86.090% | 86.090% | 86.094% |
| 仅音频残缺 | no_causal | 86.090% | 86.090% | 86.090% | 86.094% |
| 仅图像残缺 | control | 82.120% | 82.120% | 82.120% | 82.123% |
| 仅图像残缺 | v11g | 81.980% | 82.120% | 82.020% | 82.223% |
| 仅图像残缺 | no_causal | 82.490% | 82.120% | 82.980% | 82.533% |

</details>


<details>
<summary>family 2：pixel_delete/freq_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11g | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11g | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965833 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966043 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11g | 98.830% | 0.002053 | 0.985947 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002085 | 0.985614 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 98.970% | 0.009764 | 0.942683 | 0.004813 | 0.860118 |
| 图像干净、音频残缺 | v11g | 98.970% | 0.009764 | 0.942683 | 0.004670 | 0.862249 |
| 图像干净、音频残缺 | no_causal | 98.970% | 0.009764 | 0.942683 | 0.004725 | 0.861815 |
| 音频干净、图像残缺 | control | 99.190% | 0.004433 | 0.974220 | 0.003645 | 0.911259 |
| 音频干净、图像残缺 | v11g | 99.190% | 0.004190 | 0.975614 | 0.003645 | 0.911259 |
| 音频干净、图像残缺 | no_causal | 99.190% | 0.004285 | 0.975056 | 0.003645 | 0.911259 |
| 双模态残缺 | control | 98.110% | 0.004434 | 0.974162 | 0.004848 | 0.860361 |
| 双模态残缺 | v11g | 98.110% | 0.004233 | 0.975326 | 0.004737 | 0.861979 |
| 双模态残缺 | no_causal | 98.110% | 0.004306 | 0.974907 | 0.004777 | 0.861683 |
| 仅音频残缺 | control | 92.480% | 0.007755 | 0.941128 | 0.004865 | 0.858251 |
| 仅音频残缺 | v11g | 92.480% | 0.007591 | 0.943055 | 0.004865 | 0.858251 |
| 仅音频残缺 | no_causal | 92.480% | 0.007655 | 0.942127 | 0.004865 | 0.858251 |
| 仅图像残缺 | control | 95.010% | 0.004457 | 0.974026 | 0.000483 | 0.940544 |
| 仅图像残缺 | v11g | 95.010% | 0.004457 | 0.974026 | 0.000486 | 0.939813 |
| 仅图像残缺 | no_causal | 95.010% | 0.004457 | 0.974026 | 0.000488 | 0.940820 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11g | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11g | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11g | 37.192527 | 0.002053 | 0.002053 | 0.007442 |
| 仅干净音频 | no_causal | 36.885119 | 0.002085 | 0.002085 | 0.007491 |
| 图像干净、音频残缺 | control | 20.778868 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | 20.778868 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.778868 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 24.098065 | 0.036370 | 0.011070 | 0.033332 |
| 音频干净、图像残缺 | v11g | 24.340218 | 0.035390 | 0.010464 | 0.032744 |
| 音频干净、图像残缺 | no_causal | 24.240744 | 0.035491 | 0.010702 | 0.032879 |
| 双模态残缺 | control | 24.090888 | 0.036313 | 0.011067 | 0.033377 |
| 双模态残缺 | v11g | 24.291382 | 0.034767 | 0.010565 | 0.032809 |
| 双模态残缺 | no_causal | 24.215470 | 0.034995 | 0.010746 | 0.032936 |
| 仅音频残缺 | control | 31.764547 | 0.007755 | 0.007755 | 0.017812 |
| 仅音频残缺 | v11g | 31.659415 | 0.007591 | 0.007591 | 0.018030 |
| 仅音频残缺 | no_causal | 31.336235 | 0.007655 | 0.007655 | 0.018162 |
| 仅图像残缺 | control | 24.077221 | 0.037197 | 0.011131 | 0.033512 |
| 仅图像残缺 | v11g | 24.077221 | 0.037197 | 0.011131 | 0.033512 |
| 仅图像残缺 | no_causal | 24.077221 | 0.037197 | 0.011131 | 0.033512 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.028126 | 8.145e-09 | 0.000081 |
| 音频干净、图像残缺 | v11g | 0.028126 | 8.145e-09 | 0.000081 |
| 音频干净、图像残缺 | no_causal | 0.028126 | 8.145e-09 | 0.000081 |
| 双模态残缺 | control | 0.028026 | 8.144e-09 | 0.000081 |
| 双模态残缺 | v11g | 0.028026 | 8.144e-09 | 0.000081 |
| 双模态残缺 | no_causal | 0.028026 | 8.144e-09 | 0.000081 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.028769 | 8.145e-09 | 0.000081 |
| 仅图像残缺 | v11g | 0.028769 | 8.145e-09 | 0.000081 |
| 仅图像残缺 | no_causal | 0.028769 | 8.145e-09 | 0.000081 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11g | 0.000294 | 0.003616 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003565 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.006317 | 0.021067 | 0.003784 | 0.016163 |
| 图像干净、音频残缺 | v11g | 0.005965 | 0.021143 | 0.003784 | 0.016163 |
| 图像干净、音频残缺 | no_causal | 0.006099 | 0.021260 | 0.003784 | 0.016163 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.006427 | 0.021316 | 0.003767 | 0.016127 |
| 双模态残缺 | v11g | 0.006155 | 0.021325 | 0.003767 | 0.016127 |
| 双模态残缺 | no_causal | 0.006252 | 0.021406 | 0.003767 | 0.016127 |
| 仅音频残缺 | control | 0.006428 | 0.021270 | 0.003795 | 0.016189 |
| 仅音频残缺 | v11g | 0.006428 | 0.021270 | 0.003795 | 0.016189 |
| 仅音频残缺 | no_causal | 0.006428 | 0.021270 | 0.003795 | 0.016189 |
| 仅图像残缺 | control | 0.000483 | 0.004556 | N/A | N/A |
| 仅图像残缺 | v11g | 0.000486 | 0.004690 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000488 | 0.004592 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.339185 |
| 干净双模态 | v11g | 97.790% | 96.040% | 0.058851 | 0.339185 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.339185 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.339899 |
| 仅干净图像 | v11g | 96.820% | 94.200% | 0.059386 | 0.339899 |
| 仅干净图像 | no_causal | 96.820% | 94.480% | 0.059386 | 0.339899 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.289794 |
| 仅干净音频 | v11g | 98.090% | 96.000% | 0.045921 | 0.288549 |
| 仅干净音频 | no_causal | 97.710% | 96.000% | 0.046246 | 0.289732 |
| 图像干净、音频残缺 | control | 97.510% | 89.560% | 0.059181 | 0.340259 |
| 图像干净、音频残缺 | v11g | 97.510% | 89.990% | 0.059181 | 0.340259 |
| 图像干净、音频残缺 | no_causal | 97.510% | 89.820% | 0.059181 | 0.340259 |
| 音频干净、图像残缺 | control | 97.380% | 96.010% | 0.064142 | 0.353424 |
| 音频干净、图像残缺 | v11g | 97.420% | 96.010% | 0.064136 | 0.353446 |
| 音频干净、图像残缺 | no_causal | 97.340% | 96.010% | 0.064040 | 0.353166 |
| 双模态残缺 | control | 97.330% | 88.800% | 0.064141 | 0.354006 |
| 双模态残缺 | v11g | 97.400% | 88.960% | 0.064318 | 0.354526 |
| 双模态残缺 | no_causal | 97.350% | 88.690% | 0.064187 | 0.354143 |
| 仅音频残缺 | control | 91.530% | 87.110% | 0.043328 | 0.281725 |
| 仅音频残缺 | v11g | 91.560% | 87.110% | 0.042964 | 0.280516 |
| 仅音频残缺 | no_causal | 91.620% | 87.110% | 0.043140 | 0.281167 |
| 仅图像残缺 | control | 97.110% | 87.490% | 0.064040 | 0.353135 |
| 仅图像残缺 | v11g | 97.110% | 87.730% | 0.064040 | 0.353135 |
| 仅图像残缺 | no_causal | 97.110% | 88.240% | 0.064040 | 0.353135 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11g | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11g | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11g | 0.015600 | 0.068500 | 0.981700 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.996400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.041700 | 0.151500 | 1.000000 |
| 图像干净、音频残缺 | v11g | 0.042100 | 0.150300 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.042500 | 0.152300 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11g | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155700 | 1.000000 |
| 双模态残缺 | control | 0.041900 | 0.151700 | 1.000000 |
| 双模态残缺 | v11g | 0.042100 | 0.151000 | 1.000000 |
| 双模态残缺 | no_causal | 0.042500 | 0.152500 | 1.000000 |
| 仅音频残缺 | control | 0.041700 | 0.151400 | 1.000000 |
| 仅音频残缺 | v11g | 0.041700 | 0.151400 | 1.000000 |
| 仅音频残缺 | no_causal | 0.041700 | 0.151400 | 1.000000 |
| 仅图像残缺 | control | 0.015500 | 0.067600 | 1.000000 |
| 仅图像残缺 | v11g | 0.015900 | 0.067500 | 0.996100 |
| 仅图像残缺 | no_causal | 0.015600 | 0.068100 | 0.999800 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11g | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.100% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 图像干净、音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 80.200% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 双模态残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 80.100% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 80.000% |
| 仅音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 80.000% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 80.000% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 79.700% |
| 仅图像残缺 | v11g | 0.015600 | 0.070500 | 0.987900 | 79.600% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 79.600% |

**本 family 干预**

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11g | 0.002053 | 0.002085 | 0.002050 | 0.002042 |
| 仅干净音频 | no_causal | 0.002085 | 0.002085 | 0.002096 | 0.002081 |
| 音频干净、图像残缺 | control | 0.011070 | 0.011070 | 0.011070 | 0.011070 |
| 音频干净、图像残缺 | v11g | 0.010464 | 0.011070 | 0.010475 | 0.010463 |
| 音频干净、图像残缺 | no_causal | 0.010702 | 0.011070 | 0.010727 | 0.010703 |
| 双模态残缺 | control | 0.011067 | 0.011067 | 0.011067 | 0.011066 |
| 双模态残缺 | v11g | 0.010565 | 0.011067 | 0.010575 | 0.010568 |
| 双模态残缺 | no_causal | 0.010746 | 0.011067 | 0.010763 | 0.010746 |
| 仅音频残缺 | control | 0.007755 | 0.007755 | 0.007755 | 0.007756 |
| 仅音频残缺 | v11g | 0.007591 | 0.007755 | 0.007580 | 0.007556 |
| 仅音频残缺 | no_causal | 0.007655 | 0.007755 | 0.007674 | 0.007648 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11g | 0.000032 | -0.000003 | -0.000012 |
| 仅干净音频 | no_causal | 1.001e-08 | 0.000011 | -0.000005 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.000606 | 0.000011 | -0.000001 |
| 音频干净、图像残缺 | no_causal | 0.000368 | 0.000024 | 5.612e-07 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000502 | 0.000010 | 0.000003 |
| 双模态残缺 | no_causal | 0.000320 | 0.000017 | -2.856e-07 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11g | 0.000164 | -0.000010 | -0.000036 |
| 仅音频残缺 | no_causal | 0.000099 | 0.000018 | -0.000009 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11g | 36.370% | 63.070% | 27.930% |
| 仅干净音频 | no_causal | 30.170% | 65.550% | 25.520% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11g | 85.010% | 52.110% | 45.720% |
| 音频干净、图像残缺 | no_causal | 77.880% | 53.700% | 44.590% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 84.390% | 51.470% | 44.580% |
| 双模态残缺 | no_causal | 78.660% | 52.140% | 43.510% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11g | 52.890% | 57.500% | 35.290% |
| 仅音频残缺 | no_causal | 40.900% | 57.750% | 30.410% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11g | 0.534002 | 0.436642 |
| 仅干净音频 | no_causal | 0.757989 | 0.179866 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.556954 | 0.275032 |
| 音频干净、图像残缺 | no_causal | 0.759856 | 0.109110 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.556409 | 0.221365 |
| 双模态残缺 | no_causal | 0.763156 | 0.091244 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11g | 0.534458 | 0.352831 |
| 仅音频残缺 | no_causal | 0.758746 | 0.150288 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11g | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000324 | 0.000301 |
| 图像干净、音频残缺 | control | 0.006317 | 0.006317 | 0.006317 | 0.006319 |
| 图像干净、音频残缺 | v11g | 0.005965 | 0.006317 | 0.005976 | 0.005965 |
| 图像干净、音频残缺 | no_causal | 0.006099 | 0.006317 | 0.006164 | 0.006101 |
| 双模态残缺 | control | 0.006427 | 0.006427 | 0.006427 | 0.006427 |
| 双模态残缺 | v11g | 0.006155 | 0.006427 | 0.006154 | 0.006155 |
| 双模态残缺 | no_causal | 0.006252 | 0.006427 | 0.006276 | 0.006251 |
| 仅图像残缺 | control | 0.000483 | 0.000483 | 0.000483 | 0.000483 |
| 仅图像残缺 | v11g | 0.000486 | 0.000483 | 0.000491 | 0.000487 |
| 仅图像残缺 | no_causal | 0.000488 | 0.000483 | 0.000501 | 0.000494 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11g | -0.000005 | 0.000010 | 7.588e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000028 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.000352 | 0.000011 | -9.733e-07 |
| 图像干净、音频残缺 | no_causal | 0.000218 | 0.000064 | 4.284e-07 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000272 | -4.083e-07 | 1.880e-07 |
| 双模态残缺 | no_causal | 0.000174 | 0.000024 | -0.000001 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11g | -0.000003 | 0.000005 | 0.000001 |
| 仅图像残缺 | no_causal | -0.000005 | 0.000013 | 0.000006 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11g | 35.190% | 69.260% | 26.800% |
| 仅干净图像 | no_causal | 28.510% | 71.180% | 23.370% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11g | 78.000% | 50.660% | 43.140% |
| 图像干净、音频残缺 | no_causal | 55.840% | 52.650% | 35.460% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 80.120% | 49.520% | 42.910% |
| 双模态残缺 | no_causal | 55.970% | 51.150% | 33.770% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11g | 27.060% | 66.510% | 19.920% |
| 仅图像残缺 | no_causal | 35.230% | 65.960% | 25.430% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11g | 0.947892 | 0.040982 |
| 仅干净图像 | no_causal | 0.961301 | 0.027724 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.944536 | 0.028347 |
| 图像干净、音频残缺 | no_causal | 0.957178 | 0.019610 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.942359 | 0.021542 |
| 双模态残缺 | no_causal | 0.955608 | 0.014878 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11g | 0.945812 | 0.030963 |
| 仅图像残缺 | no_causal | 0.959948 | 0.020966 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11g | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11g | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11g | 98.090% | 97.710% | 97.490% | 97.499% |
| 仅干净音频 | no_causal | 97.710% | 97.710% | 97.710% | 97.579% |
| 图像干净、音频残缺 | control | 97.510% | 97.510% | 97.510% | 97.509% |
| 图像干净、音频残缺 | v11g | 97.510% | 97.510% | 97.510% | 97.509% |
| 图像干净、音频残缺 | no_causal | 97.510% | 97.510% | 97.510% | 97.509% |
| 音频干净、图像残缺 | control | 97.380% | 97.380% | 97.380% | 97.379% |
| 音频干净、图像残缺 | v11g | 97.420% | 97.380% | 97.340% | 97.359% |
| 音频干净、图像残缺 | no_causal | 97.340% | 97.380% | 97.250% | 97.309% |
| 双模态残缺 | control | 97.330% | 97.330% | 97.330% | 97.329% |
| 双模态残缺 | v11g | 97.400% | 97.330% | 97.440% | 97.459% |
| 双模态残缺 | no_causal | 97.350% | 97.330% | 97.400% | 97.379% |
| 仅音频残缺 | control | 91.530% | 91.530% | 91.530% | 91.527% |
| 仅音频残缺 | v11g | 91.560% | 91.530% | 91.620% | 91.597% |
| 仅音频残缺 | no_causal | 91.620% | 91.530% | 91.530% | 91.607% |
| 仅图像残缺 | control | 97.110% | 97.110% | 97.110% | 97.109% |
| 仅图像残缺 | v11g | 97.110% | 97.110% | 97.110% | 97.109% |
| 仅图像残缺 | no_causal | 97.110% | 97.110% | 97.110% | 97.109% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11g | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11g | 94.200% | 94.320% | 94.080% | 94.088% |
| 仅干净图像 | no_causal | 94.480% | 94.320% | 94.690% | 94.538% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11g | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 89.560% | 89.560% | 89.560% | 89.556% |
| 图像干净、音频残缺 | v11g | 89.990% | 89.560% | 89.830% | 89.946% |
| 图像干净、音频残缺 | no_causal | 89.820% | 89.560% | 89.620% | 89.656% |
| 音频干净、图像残缺 | control | 96.010% | 96.010% | 96.010% | 96.008% |
| 音频干净、图像残缺 | v11g | 96.010% | 96.010% | 96.010% | 96.008% |
| 音频干净、图像残缺 | no_causal | 96.010% | 96.010% | 96.010% | 96.008% |
| 双模态残缺 | control | 88.800% | 88.800% | 88.800% | 88.796% |
| 双模态残缺 | v11g | 88.960% | 88.800% | 88.820% | 88.796% |
| 双模态残缺 | no_causal | 88.690% | 88.800% | 88.630% | 88.735% |
| 仅音频残缺 | control | 87.110% | 87.110% | 87.110% | 87.115% |
| 仅音频残缺 | v11g | 87.110% | 87.110% | 87.110% | 87.115% |
| 仅音频残缺 | no_causal | 87.110% | 87.110% | 87.110% | 87.115% |
| 仅图像残缺 | control | 87.490% | 87.490% | 87.490% | 87.485% |
| 仅图像残缺 | v11g | 87.730% | 87.490% | 87.740% | 87.625% |
| 仅图像残缺 | no_causal | 88.240% | 87.490% | 88.390% | 88.245% |

</details>


<details>
<summary>family 3：mask_vertical/feature_dropout</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11g | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11g | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965833 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966043 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11g | 98.830% | 0.002053 | 0.985947 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002085 | 0.985614 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 99.310% | 0.009733 | 0.942750 | 0.003736 | 0.904357 |
| 图像干净、音频残缺 | v11g | 99.310% | 0.009733 | 0.942750 | 0.003660 | 0.905488 |
| 图像干净、音频残缺 | no_causal | 99.310% | 0.009733 | 0.942750 | 0.003689 | 0.905003 |
| 音频干净、图像残缺 | control | 99.130% | 0.007666 | 0.957471 | 0.003646 | 0.911234 |
| 音频干净、图像残缺 | v11g | 99.130% | 0.007545 | 0.958088 | 0.003646 | 0.911234 |
| 音频干净、图像残缺 | no_causal | 99.130% | 0.007612 | 0.957711 | 0.003646 | 0.911234 |
| 双模态残缺 | control | 99.150% | 0.007630 | 0.957875 | 0.003737 | 0.904205 |
| 双模态残缺 | v11g | 99.150% | 0.007509 | 0.958511 | 0.003675 | 0.905141 |
| 双模态残缺 | no_causal | 99.150% | 0.007579 | 0.958138 | 0.003699 | 0.904723 |
| 仅音频残缺 | control | 97.230% | 0.002978 | 0.978846 | 0.003734 | 0.904240 |
| 仅音频残缺 | v11g | 97.230% | 0.002927 | 0.979464 | 0.003734 | 0.904240 |
| 仅音频残缺 | no_causal | 97.230% | 0.002967 | 0.978958 | 0.003734 | 0.904240 |
| 仅图像残缺 | control | 94.530% | 0.007853 | 0.956388 | 0.000504 | 0.939148 |
| 仅图像残缺 | v11g | 94.530% | 0.007853 | 0.956388 | 0.000507 | 0.938761 |
| 仅图像残缺 | no_causal | 94.530% | 0.007853 | 0.956388 | 0.000509 | 0.939429 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11g | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11g | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11g | 37.192527 | 0.002053 | 0.002053 | 0.007442 |
| 仅干净音频 | no_causal | 36.885119 | 0.002085 | 0.002085 | 0.007491 |
| 图像干净、音频残缺 | control | 20.793251 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | 20.793251 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.793251 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 24.345296 | 0.030801 | 0.019513 | 0.045871 |
| 音频干净、图像残缺 | v11g | 24.376821 | 0.031223 | 0.019205 | 0.046710 |
| 音频干净、图像残缺 | no_causal | 24.398199 | 0.030777 | 0.019375 | 0.045999 |
| 双模态残缺 | control | 24.379809 | 0.030731 | 0.019422 | 0.045644 |
| 双模态残缺 | v11g | 24.408639 | 0.031068 | 0.019114 | 0.046479 |
| 双模态残缺 | no_causal | 24.430554 | 0.030575 | 0.019293 | 0.045675 |
| 仅音频残缺 | control | 36.157700 | 0.002978 | 0.002978 | 0.009158 |
| 仅音频残缺 | v11g | 35.870819 | 0.002927 | 0.002927 | 0.009420 |
| 仅音频残缺 | no_causal | 35.478775 | 0.002967 | 0.002967 | 0.009540 |
| 仅图像残缺 | control | 24.244018 | 0.031509 | 0.019989 | 0.046766 |
| 仅图像残缺 | v11g | 24.244018 | 0.031509 | 0.019989 | 0.046766 |
| 仅图像残缺 | no_causal | 24.244018 | 0.031509 | 0.019989 | 0.046766 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.019340 | 7.760e-09 | 0.000078 |
| 音频干净、图像残缺 | v11g | 0.019340 | 7.760e-09 | 0.000078 |
| 音频干净、图像残缺 | no_causal | 0.019340 | 7.760e-09 | 0.000078 |
| 双模态残缺 | control | 0.019353 | 7.757e-09 | 0.000078 |
| 双模态残缺 | v11g | 0.019353 | 7.757e-09 | 0.000078 |
| 双模态残缺 | no_causal | 0.019353 | 7.757e-09 | 0.000078 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.019562 | 7.760e-09 | 0.000078 |
| 仅图像残缺 | v11g | 0.019562 | 7.760e-09 | 0.000078 |
| 仅图像残缺 | no_causal | 0.019562 | 7.760e-09 | 0.000078 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11g | 0.000294 | 0.003616 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003565 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.003793 | 0.015930 | 0.003698 | 0.015719 |
| 图像干净、音频残缺 | v11g | 0.003603 | 0.015720 | 0.003698 | 0.015719 |
| 图像干净、音频残缺 | no_causal | 0.003675 | 0.015797 | 0.003698 | 0.015719 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.003804 | 0.015958 | 0.003692 | 0.015713 |
| 双模态残缺 | v11g | 0.003651 | 0.015782 | 0.003692 | 0.015713 |
| 双模态残缺 | no_causal | 0.003710 | 0.015842 | 0.003692 | 0.015713 |
| 仅音频残缺 | control | 0.003791 | 0.015931 | 0.003695 | 0.015719 |
| 仅音频残缺 | v11g | 0.003791 | 0.015931 | 0.003695 | 0.015719 |
| 仅音频残缺 | no_causal | 0.003791 | 0.015931 | 0.003695 | 0.015719 |
| 仅图像残缺 | control | 0.000504 | 0.004496 | N/A | N/A |
| 仅图像残缺 | v11g | 0.000507 | 0.004624 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000509 | 0.004538 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.337367 |
| 干净双模态 | v11g | 97.790% | 96.040% | 0.058851 | 0.337367 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.337367 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.341075 |
| 仅干净图像 | v11g | 96.820% | 94.200% | 0.059386 | 0.341075 |
| 仅干净图像 | no_causal | 96.820% | 94.480% | 0.059386 | 0.341075 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.285967 |
| 仅干净音频 | v11g | 98.090% | 96.000% | 0.045921 | 0.284719 |
| 仅干净音频 | no_causal | 97.710% | 96.000% | 0.046246 | 0.285900 |
| 图像干净、音频残缺 | control | 97.610% | 94.530% | 0.058961 | 0.337150 |
| 图像干净、音频残缺 | v11g | 97.610% | 94.670% | 0.058961 | 0.337150 |
| 图像干净、音频残缺 | no_causal | 97.610% | 94.670% | 0.058961 | 0.337150 |
| 音频干净、图像残缺 | control | 96.740% | 96.000% | 0.060439 | 0.342839 |
| 音频干净、图像残缺 | v11g | 96.680% | 96.000% | 0.060129 | 0.341919 |
| 音频干净、图像残缺 | no_causal | 96.510% | 96.000% | 0.060329 | 0.342525 |
| 双模态残缺 | control | 96.600% | 94.680% | 0.060428 | 0.343442 |
| 双模态残缺 | v11g | 96.540% | 94.670% | 0.060150 | 0.342739 |
| 双模态残缺 | no_causal | 96.500% | 94.630% | 0.060384 | 0.343362 |
| 仅音频残缺 | control | 96.850% | 93.950% | 0.045526 | 0.283439 |
| 仅音频残缺 | v11g | 96.870% | 93.950% | 0.045201 | 0.282380 |
| 仅音频残缺 | no_causal | 96.860% | 93.950% | 0.045425 | 0.283231 |
| 仅图像残缺 | control | 95.430% | 88.100% | 0.060211 | 0.342225 |
| 仅图像残缺 | v11g | 95.430% | 87.980% | 0.060211 | 0.342225 |
| 仅图像残缺 | no_causal | 95.430% | 88.540% | 0.060211 | 0.342225 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11g | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11g | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11g | 0.015600 | 0.068500 | 0.981700 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.996400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.042100 | 0.153900 | 1.000000 |
| 图像干净、音频残缺 | v11g | 0.042100 | 0.153300 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.042200 | 0.153900 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155600 | 1.000000 |
| 音频干净、图像残缺 | v11g | 0.043200 | 0.155600 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155600 | 1.000000 |
| 双模态残缺 | control | 0.042100 | 0.153900 | 1.000000 |
| 双模态残缺 | v11g | 0.042100 | 0.153400 | 1.000000 |
| 双模态残缺 | no_causal | 0.042200 | 0.153900 | 1.000000 |
| 仅音频残缺 | control | 0.042100 | 0.153900 | 1.000000 |
| 仅音频残缺 | v11g | 0.042100 | 0.153900 | 1.000000 |
| 仅音频残缺 | no_causal | 0.042100 | 0.153900 | 1.000000 |
| 仅图像残缺 | control | 0.015300 | 0.067500 | 1.000000 |
| 仅图像残缺 | v11g | 0.015700 | 0.067400 | 0.992600 |
| 仅图像残缺 | no_causal | 0.015400 | 0.067900 | 0.999300 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11g | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.100% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 图像干净、音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 双模态残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 仅音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.700% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 79.800% |
| 仅图像残缺 | v11g | 0.015600 | 0.070500 | 0.987900 | 79.800% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 79.700% |

**本 family 干预**

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11g | 0.002053 | 0.002085 | 0.002050 | 0.002042 |
| 仅干净音频 | no_causal | 0.002085 | 0.002085 | 0.002096 | 0.002081 |
| 音频干净、图像残缺 | control | 0.019513 | 0.019513 | 0.019513 | 0.019510 |
| 音频干净、图像残缺 | v11g | 0.019205 | 0.019513 | 0.019246 | 0.019201 |
| 音频干净、图像残缺 | no_causal | 0.019375 | 0.019513 | 0.019441 | 0.019375 |
| 双模态残缺 | control | 0.019422 | 0.019422 | 0.019422 | 0.019419 |
| 双模态残缺 | v11g | 0.019114 | 0.019422 | 0.019147 | 0.019107 |
| 双模态残缺 | no_causal | 0.019293 | 0.019422 | 0.019347 | 0.019286 |
| 仅音频残缺 | control | 0.002978 | 0.002978 | 0.002978 | 0.002979 |
| 仅音频残缺 | v11g | 0.002927 | 0.002978 | 0.002919 | 0.002909 |
| 仅音频残缺 | no_causal | 0.002967 | 0.002978 | 0.002978 | 0.002962 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11g | 0.000032 | -0.000003 | -0.000012 |
| 仅干净音频 | no_causal | 1.001e-08 | 0.000011 | -0.000005 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.000307 | 0.000041 | -0.000001 |
| 音频干净、图像残缺 | no_causal | 0.000138 | 0.000066 | 0.000004 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000307 | 0.000033 | -0.000004 |
| 双模态残缺 | no_causal | 0.000129 | 0.000054 | -0.000004 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11g | 0.000051 | -0.000008 | -0.000019 |
| 仅音频残缺 | no_causal | 0.000011 | 0.000011 | -0.000006 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11g | 36.370% | 63.070% | 27.930% |
| 仅干净音频 | no_causal | 30.170% | 65.550% | 25.520% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11g | 57.750% | 52.460% | 33.680% |
| 音频干净、图像残缺 | no_causal | 53.430% | 51.500% | 33.540% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 58.160% | 51.190% | 33.040% |
| 双模态残缺 | no_causal | 53.810% | 50.470% | 33.730% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11g | 39.970% | 60.570% | 29.410% |
| 仅音频残缺 | no_causal | 29.090% | 60.000% | 23.550% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11g | 0.534002 | 0.436642 |
| 仅干净音频 | no_causal | 0.757989 | 0.179866 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.561817 | 0.240391 |
| 音频干净、图像残缺 | no_causal | 0.776032 | 0.095655 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.561371 | 0.219364 |
| 双模态残缺 | no_causal | 0.780191 | 0.087478 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11g | 0.533948 | 0.396622 |
| 仅音频残缺 | no_causal | 0.759839 | 0.164238 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11g | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000324 | 0.000301 |
| 图像干净、音频残缺 | control | 0.003793 | 0.003793 | 0.003793 | 0.003793 |
| 图像干净、音频残缺 | v11g | 0.003603 | 0.003793 | 0.003595 | 0.003604 |
| 图像干净、音频残缺 | no_causal | 0.003675 | 0.003793 | 0.003670 | 0.003675 |
| 双模态残缺 | control | 0.003804 | 0.003804 | 0.003804 | 0.003804 |
| 双模态残缺 | v11g | 0.003651 | 0.003804 | 0.003643 | 0.003651 |
| 双模态残缺 | no_causal | 0.003710 | 0.003804 | 0.003706 | 0.003710 |
| 仅图像残缺 | control | 0.000504 | 0.000504 | 0.000504 | 0.000504 |
| 仅图像残缺 | v11g | 0.000507 | 0.000504 | 0.000512 | 0.000508 |
| 仅图像残缺 | no_causal | 0.000509 | 0.000504 | 0.000524 | 0.000516 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11g | -0.000005 | 0.000010 | 7.588e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000028 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.000189 | -0.000009 | -2.101e-07 |
| 图像干净、音频残缺 | no_causal | 0.000118 | -0.000005 | 3.181e-08 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000153 | -0.000008 | 5.030e-08 |
| 双模态残缺 | no_causal | 0.000094 | -0.000004 | 3.631e-08 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11g | -0.000003 | 0.000006 | 0.000001 |
| 仅图像残缺 | no_causal | -0.000005 | 0.000015 | 0.000007 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11g | 35.190% | 69.260% | 26.800% |
| 仅干净图像 | no_causal | 28.510% | 71.180% | 23.370% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11g | 87.940% | 48.480% | 44.330% |
| 图像干净、音频残缺 | no_causal | 76.810% | 50.310% | 42.570% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 88.670% | 48.220% | 44.340% |
| 双模态残缺 | no_causal | 76.400% | 50.310% | 41.950% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11g | 33.500% | 67.590% | 24.540% |
| 仅图像残缺 | no_causal | 33.680% | 67.710% | 24.860% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11g | 0.947892 | 0.040982 |
| 仅干净图像 | no_causal | 0.961301 | 0.027724 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.941875 | 0.026806 |
| 图像干净、音频残缺 | no_causal | 0.952861 | 0.018557 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.939761 | 0.022501 |
| 双模态残缺 | no_causal | 0.951191 | 0.015470 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11g | 0.945804 | 0.034319 |
| 仅图像残缺 | no_causal | 0.959891 | 0.023071 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11g | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11g | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11g | 98.090% | 97.710% | 97.490% | 97.499% |
| 仅干净音频 | no_causal | 97.710% | 97.710% | 97.710% | 97.579% |
| 图像干净、音频残缺 | control | 97.610% | 97.610% | 97.610% | 97.609% |
| 图像干净、音频残缺 | v11g | 97.610% | 97.610% | 97.610% | 97.609% |
| 图像干净、音频残缺 | no_causal | 97.610% | 97.610% | 97.610% | 97.609% |
| 音频干净、图像残缺 | control | 96.740% | 96.740% | 96.740% | 96.739% |
| 音频干净、图像残缺 | v11g | 96.680% | 96.740% | 96.640% | 96.619% |
| 音频干净、图像残缺 | no_causal | 96.510% | 96.740% | 96.600% | 96.589% |
| 双模态残缺 | control | 96.600% | 96.600% | 96.600% | 96.599% |
| 双模态残缺 | v11g | 96.540% | 96.600% | 96.520% | 96.489% |
| 双模态残缺 | no_causal | 96.500% | 96.600% | 96.650% | 96.479% |
| 仅音频残缺 | control | 96.850% | 96.850% | 96.850% | 96.849% |
| 仅音频残缺 | v11g | 96.870% | 96.850% | 96.880% | 96.879% |
| 仅音频残缺 | no_causal | 96.860% | 96.850% | 96.830% | 96.839% |
| 仅图像残缺 | control | 95.430% | 95.430% | 95.430% | 95.428% |
| 仅图像残缺 | v11g | 95.430% | 95.430% | 95.430% | 95.428% |
| 仅图像残缺 | no_causal | 95.430% | 95.430% | 95.430% | 95.428% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11g | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11g | 94.200% | 94.320% | 94.080% | 94.088% |
| 仅干净图像 | no_causal | 94.480% | 94.320% | 94.690% | 94.538% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11g | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 94.530% | 94.530% | 94.530% | 94.528% |
| 图像干净、音频残缺 | v11g | 94.670% | 94.530% | 94.650% | 94.708% |
| 图像干净、音频残缺 | no_causal | 94.670% | 94.530% | 94.730% | 94.608% |
| 音频干净、图像残缺 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 音频干净、图像残缺 | v11g | 96.000% | 96.000% | 96.000% | 95.998% |
| 音频干净、图像残缺 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 双模态残缺 | control | 94.680% | 94.680% | 94.680% | 94.678% |
| 双模态残缺 | v11g | 94.670% | 94.680% | 94.770% | 94.618% |
| 双模态残缺 | no_causal | 94.630% | 94.680% | 94.640% | 94.638% |
| 仅音频残缺 | control | 93.950% | 93.950% | 93.950% | 93.948% |
| 仅音频残缺 | v11g | 93.950% | 93.950% | 93.950% | 93.948% |
| 仅音频残缺 | no_causal | 93.950% | 93.950% | 93.950% | 93.948% |
| 仅图像残缺 | control | 88.100% | 88.100% | 88.100% | 88.095% |
| 仅图像残缺 | v11g | 87.980% | 88.100% | 88.050% | 88.005% |
| 仅图像残缺 | no_causal | 88.540% | 88.100% | 88.850% | 88.645% |

</details>


<details>
<summary>family 4：mask_horizontal/partial_temporal</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11g | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11g | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965833 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966043 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11g | 98.830% | 0.002053 | 0.985947 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002085 | 0.985614 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 98.070% | 0.009949 | 0.942169 | 0.011784 | 0.538387 |
| 图像干净、音频残缺 | v11g | 98.070% | 0.009949 | 0.942169 | 0.011342 | 0.543528 |
| 图像干净、音频残缺 | no_causal | 98.070% | 0.009949 | 0.942169 | 0.011563 | 0.544686 |
| 音频干净、图像残缺 | control | 99.180% | 0.011375 | 0.933588 | 0.003646 | 0.911236 |
| 音频干净、图像残缺 | v11g | 99.180% | 0.011236 | 0.934170 | 0.003646 | 0.911236 |
| 音频干净、图像残缺 | no_causal | 99.180% | 0.011319 | 0.933721 | 0.003646 | 0.911236 |
| 双模态残缺 | control | 93.960% | 0.011546 | 0.932764 | 0.011946 | 0.537024 |
| 双模态残缺 | v11g | 93.960% | 0.011500 | 0.933092 | 0.011619 | 0.540546 |
| 双模态残缺 | no_causal | 93.960% | 0.011541 | 0.932779 | 0.011795 | 0.541813 |
| 仅音频残缺 | control | 45.880% | 0.042710 | 0.569261 | 0.012585 | 0.528776 |
| 仅音频残缺 | v11g | 45.880% | 0.040929 | 0.602984 | 0.012585 | 0.528776 |
| 仅音频残缺 | no_causal | 45.880% | 0.038806 | 0.634932 | 0.012585 | 0.528776 |
| 仅图像残缺 | control | 93.020% | 0.011626 | 0.932092 | 0.000617 | 0.921198 |
| 仅图像残缺 | v11g | 93.020% | 0.011626 | 0.932092 | 0.000617 | 0.921078 |
| 仅图像残缺 | no_causal | 93.020% | 0.011626 | 0.932092 | 0.000621 | 0.921899 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11g | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11g | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11g | 37.192527 | 0.002053 | 0.002053 | 0.007442 |
| 仅干净音频 | no_causal | 36.885119 | 0.002085 | 0.002085 | 0.007491 |
| 图像干净、音频残缺 | control | 20.701504 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | 20.701504 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.701504 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 20.224451 | 0.041941 | 0.028956 | 0.066252 |
| 音频干净、图像残缺 | v11g | 20.238247 | 0.043062 | 0.028601 | 0.067427 |
| 音频干净、图像残缺 | no_causal | 20.236644 | 0.042349 | 0.028813 | 0.066172 |
| 双模态残缺 | control | 20.167815 | 0.041696 | 0.029389 | 0.067064 |
| 双模态残缺 | v11g | 20.152729 | 0.041409 | 0.029272 | 0.068746 |
| 双模态残缺 | no_causal | 20.118608 | 0.041196 | 0.029376 | 0.069579 |
| 仅音频残缺 | control | 17.523188 | 0.042710 | 0.042710 | 0.069514 |
| 仅音频残缺 | v11g | 17.649859 | 0.040929 | 0.040929 | 0.069847 |
| 仅音频残缺 | no_causal | 17.674417 | 0.038806 | 0.038806 | 0.071605 |
| 仅图像残缺 | control | 20.117320 | 0.042756 | 0.029594 | 0.067630 |
| 仅图像残缺 | v11g | 20.117320 | 0.042756 | 0.029594 | 0.067630 |
| 仅图像残缺 | no_causal | 20.117320 | 0.042756 | 0.029594 | 0.067630 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.024606 | 7.982e-09 | 0.000080 |
| 音频干净、图像残缺 | v11g | 0.024606 | 7.982e-09 | 0.000080 |
| 音频干净、图像残缺 | no_causal | 0.024606 | 7.982e-09 | 0.000080 |
| 双模态残缺 | control | 0.023821 | 7.982e-09 | 0.000080 |
| 双模态残缺 | v11g | 0.023821 | 7.982e-09 | 0.000080 |
| 双模态残缺 | no_causal | 0.023821 | 7.982e-09 | 0.000080 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.025068 | 7.982e-09 | 0.000080 |
| 仅图像残缺 | v11g | 0.025068 | 7.982e-09 | 0.000080 |
| 仅图像残缺 | no_causal | 0.025068 | 7.982e-09 | 0.000080 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11g | 0.000294 | 0.003616 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003565 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.028559 | 0.074284 | 0.000306 | 0.001634 |
| 图像干净、音频残缺 | v11g | 0.027472 | 0.074671 | 0.000306 | 0.001634 |
| 图像干净、音频残缺 | no_causal | 0.028015 | 0.074921 | 0.000306 | 0.001634 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.028952 | 0.074811 | 0.000311 | 0.001645 |
| 双模态残缺 | v11g | 0.028147 | 0.075039 | 0.000311 | 0.001645 |
| 双模态残缺 | no_causal | 0.028579 | 0.075237 | 0.000311 | 0.001645 |
| 仅音频残缺 | control | 0.030529 | 0.075829 | 0.000308 | 0.001635 |
| 仅音频残缺 | v11g | 0.030529 | 0.075829 | 0.000308 | 0.001635 |
| 仅音频残缺 | no_causal | 0.030529 | 0.075829 | 0.000308 | 0.001635 |
| 仅图像残缺 | control | 0.000617 | 0.005049 | N/A | N/A |
| 仅图像残缺 | v11g | 0.000617 | 0.005170 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000621 | 0.005082 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.339376 |
| 干净双模态 | v11g | 97.790% | 96.040% | 0.058851 | 0.339376 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.339376 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.340113 |
| 仅干净图像 | v11g | 96.820% | 94.200% | 0.059386 | 0.340113 |
| 仅干净图像 | no_causal | 96.820% | 94.480% | 0.059386 | 0.340113 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.291384 |
| 仅干净音频 | v11g | 98.090% | 96.000% | 0.045921 | 0.290186 |
| 仅干净音频 | no_causal | 97.710% | 96.000% | 0.046246 | 0.291363 |
| 图像干净、音频残缺 | control | 97.040% | 76.880% | 0.060324 | 0.342538 |
| 图像干净、音频残缺 | v11g | 97.040% | 75.920% | 0.060324 | 0.342538 |
| 图像干净、音频残缺 | no_causal | 97.040% | 77.280% | 0.060324 | 0.342538 |
| 音频干净、图像残缺 | control | 96.280% | 95.960% | 0.057461 | 0.335875 |
| 音频干净、图像残缺 | v11g | 96.130% | 95.960% | 0.056720 | 0.333666 |
| 音频干净、图像残缺 | no_causal | 96.300% | 95.960% | 0.057066 | 0.334680 |
| 双模态残缺 | control | 94.510% | 69.490% | 0.057604 | 0.336153 |
| 双模态残缺 | v11g | 94.630% | 69.200% | 0.057441 | 0.335672 |
| 双模态残缺 | no_causal | 94.530% | 70.020% | 0.057356 | 0.335416 |
| 仅音频残缺 | control | 40.770% | 41.520% | 0.023301 | 0.188507 |
| 仅音频残缺 | v11g | 41.060% | 41.520% | 0.022417 | 0.184406 |
| 仅音频残缺 | no_causal | 40.750% | 41.520% | 0.021187 | 0.178633 |
| 仅图像残缺 | control | 94.140% | 84.400% | 0.057135 | 0.334808 |
| 仅图像残缺 | v11g | 94.140% | 84.350% | 0.057135 | 0.334808 |
| 仅图像残缺 | no_causal | 94.140% | 84.780% | 0.057135 | 0.334808 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11g | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11g | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11g | 0.015600 | 0.068500 | 0.981700 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.996400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.037500 | 0.131100 | 1.000000 |
| 图像干净、音频残缺 | v11g | 0.038400 | 0.128600 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.039400 | 0.133900 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11g | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155700 | 1.000000 |
| 双模态残缺 | control | 0.037300 | 0.130500 | 1.000000 |
| 双模态残缺 | v11g | 0.038000 | 0.128800 | 1.000000 |
| 双模态残缺 | no_causal | 0.038800 | 0.133000 | 1.000000 |
| 仅音频残缺 | control | 0.035400 | 0.126800 | 1.000000 |
| 仅音频残缺 | v11g | 0.035400 | 0.126800 | 1.000000 |
| 仅音频残缺 | no_causal | 0.035400 | 0.126800 | 1.000000 |
| 仅图像残缺 | control | 0.015400 | 0.067400 | 1.000000 |
| 仅图像残缺 | v11g | 0.015800 | 0.067300 | 0.996700 |
| 仅图像残缺 | no_causal | 0.015600 | 0.067900 | 0.999800 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11g | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.100% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 70.200% |
| 图像干净、音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 70.400% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 70.500% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 70.000% |
| 双模态残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 70.200% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 70.300% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 68.600% |
| 仅音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 68.600% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 68.600% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 79.000% |
| 仅图像残缺 | v11g | 0.015600 | 0.070500 | 0.987900 | 78.900% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 78.800% |

**本 family 干预**

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11g | 0.002053 | 0.002085 | 0.002050 | 0.002042 |
| 仅干净音频 | no_causal | 0.002085 | 0.002085 | 0.002096 | 0.002081 |
| 音频干净、图像残缺 | control | 0.028956 | 0.028956 | 0.028956 | 0.028954 |
| 音频干净、图像残缺 | v11g | 0.028601 | 0.028956 | 0.028652 | 0.028603 |
| 音频干净、图像残缺 | no_causal | 0.028813 | 0.028956 | 0.028915 | 0.028804 |
| 双模态残缺 | control | 0.029389 | 0.029389 | 0.029389 | 0.029388 |
| 双模态残缺 | v11g | 0.029272 | 0.029389 | 0.029254 | 0.029253 |
| 双模态残缺 | no_causal | 0.029376 | 0.029389 | 0.029402 | 0.029351 |
| 仅音频残缺 | control | 0.042710 | 0.042710 | 0.042710 | 0.042704 |
| 仅音频残缺 | v11g | 0.040929 | 0.042710 | 0.040934 | 0.040968 |
| 仅音频残缺 | no_causal | 0.038806 | 0.042710 | 0.039934 | 0.039868 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11g | 0.000032 | -0.000003 | -0.000012 |
| 仅干净音频 | no_causal | 1.001e-08 | 0.000011 | -0.000005 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.000355 | 0.000051 | 0.000004 |
| 音频干净、图像残缺 | no_causal | 0.000143 | 0.000102 | -0.000007 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000117 | -0.000017 | -0.000018 |
| 双模态残缺 | no_causal | 0.000013 | 0.000026 | -0.000024 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11g | 0.001780 | 0.000005 | 0.000044 |
| 仅音频残缺 | no_causal | 0.003904 | 0.001128 | 0.001067 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11g | 36.370% | 63.070% | 27.930% |
| 仅干净音频 | no_causal | 30.170% | 65.550% | 25.520% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11g | 58.150% | 53.750% | 33.450% |
| 音频干净、图像残缺 | no_causal | 52.010% | 53.430% | 31.240% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 52.330% | 42.930% | 26.290% |
| 双模态残缺 | no_causal | 48.730% | 44.250% | 27.040% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11g | 84.670% | 49.720% | 39.340% |
| 仅音频残缺 | no_causal | 81.470% | 63.180% | 49.800% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11g | 0.534002 | 0.436642 |
| 仅干净音频 | no_causal | 0.757989 | 0.179866 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.562186 | 0.234166 |
| 音频干净、图像残缺 | no_causal | 0.750847 | 0.094299 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.561076 | 0.113323 |
| 双模态残缺 | no_causal | 0.754130 | 0.111980 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11g | 0.541857 | 0.212622 |
| 仅音频残缺 | no_causal | 0.764824 | 0.229576 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11g | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000324 | 0.000301 |
| 图像干净、音频残缺 | control | 0.028559 | 0.028559 | 0.028559 | 0.028559 |
| 图像干净、音频残缺 | v11g | 0.027472 | 0.028559 | 0.027521 | 0.027467 |
| 图像干净、音频残缺 | no_causal | 0.028015 | 0.028559 | 0.028262 | 0.028005 |
| 双模态残缺 | control | 0.028952 | 0.028952 | 0.028952 | 0.028952 |
| 双模态残缺 | v11g | 0.028147 | 0.028952 | 0.028173 | 0.028146 |
| 双模态残缺 | no_causal | 0.028579 | 0.028952 | 0.028725 | 0.028574 |
| 仅图像残缺 | control | 0.000617 | 0.000617 | 0.000617 | 0.000617 |
| 仅图像残缺 | v11g | 0.000617 | 0.000617 | 0.000622 | 0.000619 |
| 仅图像残缺 | no_causal | 0.000621 | 0.000617 | 0.000636 | 0.000630 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11g | -0.000005 | 0.000010 | 7.588e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000028 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.001087 | 0.000049 | -0.000005 |
| 图像干净、音频残缺 | no_causal | 0.000544 | 0.000247 | -0.000009 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000805 | 0.000026 | -0.000001 |
| 双模态残缺 | no_causal | 0.000372 | 0.000146 | -0.000004 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11g | -5.867e-07 | 0.000004 | 0.000001 |
| 仅图像残缺 | no_causal | -0.000005 | 0.000015 | 0.000008 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11g | 35.190% | 69.260% | 26.800% |
| 仅干净图像 | no_causal | 28.510% | 71.180% | 23.370% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11g | 75.220% | 53.330% | 42.530% |
| 图像干净、音频残缺 | no_causal | 60.140% | 56.460% | 38.780% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 74.400% | 53.070% | 41.320% |
| 双模态残缺 | no_causal | 59.490% | 56.480% | 38.180% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11g | 37.180% | 67.840% | 28.050% |
| 仅图像残缺 | no_causal | 39.940% | 67.490% | 29.360% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11g | 0.947892 | 0.040982 |
| 仅干净图像 | no_causal | 0.961301 | 0.027724 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.936137 | 0.002587 |
| 图像干净、音频残缺 | no_causal | 0.947921 | 0.001891 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.932481 | 0.001998 |
| 双模态残缺 | no_causal | 0.944833 | 0.001457 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11g | 0.945818 | 0.032351 |
| 仅图像残缺 | no_causal | 0.959918 | 0.021507 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11g | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11g | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11g | 98.090% | 97.710% | 97.490% | 97.499% |
| 仅干净音频 | no_causal | 97.710% | 97.710% | 97.710% | 97.579% |
| 图像干净、音频残缺 | control | 97.040% | 97.040% | 97.040% | 97.039% |
| 图像干净、音频残缺 | v11g | 97.040% | 97.040% | 97.040% | 97.039% |
| 图像干净、音频残缺 | no_causal | 97.040% | 97.040% | 97.040% | 97.039% |
| 音频干净、图像残缺 | control | 96.280% | 96.280% | 96.280% | 96.279% |
| 音频干净、图像残缺 | v11g | 96.130% | 96.280% | 96.110% | 96.248% |
| 音频干净、图像残缺 | no_causal | 96.300% | 96.280% | 96.050% | 96.108% |
| 双模态残缺 | control | 94.510% | 94.510% | 94.510% | 94.508% |
| 双模态残缺 | v11g | 94.630% | 94.510% | 94.720% | 94.668% |
| 双模态残缺 | no_causal | 94.530% | 94.510% | 94.540% | 94.768% |
| 仅音频残缺 | control | 40.770% | 40.770% | 40.770% | 40.776% |
| 仅音频残缺 | v11g | 41.060% | 40.770% | 40.710% | 40.806% |
| 仅音频残缺 | no_causal | 40.750% | 40.770% | 40.610% | 40.506% |
| 仅图像残缺 | control | 94.140% | 94.140% | 94.140% | 94.138% |
| 仅图像残缺 | v11g | 94.140% | 94.140% | 94.140% | 94.138% |
| 仅图像残缺 | no_causal | 94.140% | 94.140% | 94.140% | 94.138% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11g | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11g | 94.200% | 94.320% | 94.080% | 94.088% |
| 仅干净图像 | no_causal | 94.480% | 94.320% | 94.690% | 94.538% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11g | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 76.880% | 76.880% | 76.880% | 76.871% |
| 图像干净、音频残缺 | v11g | 75.920% | 76.880% | 75.340% | 75.620% |
| 图像干净、音频残缺 | no_causal | 77.280% | 76.880% | 76.320% | 77.181% |
| 音频干净、图像残缺 | control | 95.960% | 95.960% | 95.960% | 95.958% |
| 音频干净、图像残缺 | v11g | 95.960% | 95.960% | 95.960% | 95.958% |
| 音频干净、图像残缺 | no_causal | 95.960% | 95.960% | 95.960% | 95.958% |
| 双模态残缺 | control | 69.490% | 69.490% | 69.490% | 69.488% |
| 双模态残缺 | v11g | 69.200% | 69.490% | 69.040% | 69.218% |
| 双模态残缺 | no_causal | 70.020% | 69.490% | 69.530% | 70.048% |
| 仅音频残缺 | control | 41.520% | 41.520% | 41.520% | 41.517% |
| 仅音频残缺 | v11g | 41.520% | 41.520% | 41.520% | 41.517% |
| 仅音频残缺 | no_causal | 41.520% | 41.520% | 41.520% | 41.517% |
| 仅图像残缺 | control | 84.400% | 84.400% | 84.400% | 84.394% |
| 仅图像残缺 | v11g | 84.350% | 84.400% | 84.230% | 84.344% |
| 仅图像残缺 | no_causal | 84.780% | 84.400% | 85.110% | 84.904% |

</details>


<details>
<summary>family 5：salt_mask/time_freq_block</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | v11g | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 干净双模态 | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| 仅干净图像 | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| 仅干净图像 | v11g | 97.700% | 0.009777 | 0.942646 | 0.000294 | 0.965833 |
| 仅干净图像 | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000296 | 0.966043 |
| 仅干净音频 | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| 仅干净音频 | v11g | 98.830% | 0.002053 | 0.985947 | 0.003648 | 0.911157 |
| 仅干净音频 | no_causal | 98.830% | 0.002085 | 0.985614 | 0.003648 | 0.911157 |
| 图像干净、音频残缺 | control | 99.180% | 0.009738 | 0.942714 | 0.003975 | 0.900494 |
| 图像干净、音频残缺 | v11g | 99.180% | 0.009738 | 0.942714 | 0.003923 | 0.901128 |
| 图像干净、音频残缺 | no_causal | 99.180% | 0.009738 | 0.942714 | 0.003940 | 0.900996 |
| 音频干净、图像残缺 | control | 99.090% | 0.005243 | 0.968180 | 0.003650 | 0.911121 |
| 音频干净、图像残缺 | v11g | 99.090% | 0.005091 | 0.968942 | 0.003650 | 0.911121 |
| 音频干净、图像残缺 | no_causal | 99.090% | 0.005190 | 0.968346 | 0.003650 | 0.911121 |
| 双模态残缺 | control | 98.610% | 0.005257 | 0.968044 | 0.003972 | 0.900263 |
| 双模态残缺 | v11g | 98.610% | 0.005111 | 0.968804 | 0.003937 | 0.900693 |
| 双模态残缺 | no_causal | 98.610% | 0.005205 | 0.968252 | 0.003949 | 0.900582 |
| 仅音频残缺 | control | 96.770% | 0.003567 | 0.973989 | 0.003986 | 0.900188 |
| 仅音频残缺 | v11g | 96.770% | 0.003502 | 0.974696 | 0.003986 | 0.900188 |
| 仅音频残缺 | no_causal | 96.770% | 0.003543 | 0.974232 | 0.003986 | 0.900188 |
| 仅图像残缺 | control | 87.790% | 0.005307 | 0.967732 | 0.000917 | 0.882853 |
| 仅图像残缺 | v11g | 87.790% | 0.005307 | 0.967732 | 0.000919 | 0.882156 |
| 仅图像残缺 | no_causal | 87.790% | 0.005307 | 0.967732 | 0.000921 | 0.883685 |

**图像缺失区域**

| 输入模式 | 实验 | 图像 PSNR (dB) | 图像 coarse 缺失区 MSE | 图像缺失区 MSE | 图像缺失区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | v11g | 20.794109 | N/A | N/A | N/A |
| 干净双模态 | no_causal | 20.794109 | N/A | N/A | N/A |
| 仅干净图像 | control | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | v11g | 20.771370 | N/A | N/A | N/A |
| 仅干净图像 | no_causal | 20.771370 | N/A | N/A | N/A |
| 仅干净音频 | control | 37.624735 | 0.002085 | 0.002085 | 0.007154 |
| 仅干净音频 | v11g | 37.192527 | 0.002053 | 0.002053 | 0.007442 |
| 仅干净音频 | no_causal | 36.885119 | 0.002085 | 0.002085 | 0.007491 |
| 图像干净、音频残缺 | control | 20.790630 | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | 20.790630 | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | 20.790630 | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 23.272686 | 0.066434 | 0.013088 | 0.038475 |
| 音频干净、图像残缺 | v11g | 23.395554 | 0.069814 | 0.012708 | 0.038341 |
| 音频干净、图像残缺 | no_causal | 23.311120 | 0.069365 | 0.012956 | 0.038598 |
| 双模态残缺 | control | 23.267845 | 0.066524 | 0.013111 | 0.038469 |
| 双模态残缺 | v11g | 23.385427 | 0.069013 | 0.012746 | 0.038282 |
| 双模态残缺 | no_causal | 23.305865 | 0.068679 | 0.012981 | 0.038543 |
| 仅音频残缺 | control | 36.133221 | 0.003567 | 0.003567 | 0.009855 |
| 仅音频残缺 | v11g | 35.842990 | 0.003502 | 0.003502 | 0.010109 |
| 仅音频残缺 | no_causal | 35.495129 | 0.003543 | 0.003543 | 0.010191 |
| 仅图像残缺 | control | 23.216327 | 0.067368 | 0.013248 | 0.038833 |
| 仅图像残缺 | v11g | 23.216327 | 0.067368 | 0.013248 | 0.038833 |
| 仅图像残缺 | no_causal | 23.216327 | 0.067368 | 0.013248 | 0.038833 |

**图像可见区域**

| 输入模式 | 实验 | 图像 coarse 可见区 MSE | 图像可见区 MSE | 图像可见区 L1 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A |
| 仅干净图像 | control | N/A | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A | N/A |
| 音频干净、图像残缺 | control | 0.066257 | 8.146e-09 | 0.000081 |
| 音频干净、图像残缺 | v11g | 0.066257 | 8.146e-09 | 0.000081 |
| 音频干净、图像残缺 | no_causal | 0.066257 | 8.146e-09 | 0.000081 |
| 双模态残缺 | control | 0.066359 | 8.145e-09 | 0.000081 |
| 双模态残缺 | v11g | 0.066359 | 8.145e-09 | 0.000081 |
| 双模态残缺 | no_causal | 0.066359 | 8.145e-09 | 0.000081 |
| 仅音频残缺 | control | N/A | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A | N/A |
| 仅图像残缺 | control | 0.067160 | 8.146e-09 | 0.000081 |
| 仅图像残缺 | v11g | 0.067160 | 8.146e-09 | 0.000081 |
| 仅图像残缺 | no_causal | 0.067160 | 8.146e-09 | 0.000081 |

**音频区域误差**

| 输入模式 | 实验 | 音频缺失区 MSE | 音频缺失区 L1 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | N/A | N/A | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A | N/A | N/A |
| 仅干净图像 | control | 0.000289 | 0.003470 | N/A | N/A |
| 仅干净图像 | v11g | 0.000294 | 0.003616 | N/A | N/A |
| 仅干净图像 | no_causal | 0.000296 | 0.003565 | N/A | N/A |
| 仅干净音频 | control | N/A | N/A | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A | N/A | N/A |
| 图像干净、音频残缺 | control | 0.005272 | 0.017887 | 0.003719 | 0.015995 |
| 图像干净、音频残缺 | v11g | 0.004953 | 0.017902 | 0.003719 | 0.015995 |
| 图像干净、音频残缺 | no_causal | 0.005060 | 0.018003 | 0.003719 | 0.015995 |
| 音频干净、图像残缺 | control | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A | N/A | N/A |
| 双模态残缺 | control | 0.005077 | 0.017518 | 0.003754 | 0.016087 |
| 双模态残缺 | v11g | 0.004860 | 0.017498 | 0.003754 | 0.016087 |
| 双模态残缺 | no_causal | 0.004936 | 0.017570 | 0.003754 | 0.016087 |
| 仅音频残缺 | control | 0.005321 | 0.017980 | 0.003722 | 0.016004 |
| 仅音频残缺 | v11g | 0.005321 | 0.017980 | 0.003722 | 0.016004 |
| 仅音频残缺 | no_causal | 0.005321 | 0.017980 | 0.003722 | 0.016004 |
| 仅图像残缺 | control | 0.000917 | 0.006634 | N/A | N/A |
| 仅图像残缺 | v11g | 0.000919 | 0.006777 | N/A | N/A |
| 仅图像残缺 | no_causal | 0.000921 | 0.006645 | N/A | N/A |

**内容分类与多样性**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 音频内容 ACC (normal) | 像素方差 | 样本间 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 96.040% | 0.058851 | 0.338103 |
| 干净双模态 | v11g | 97.790% | 96.040% | 0.058851 | 0.338103 |
| 干净双模态 | no_causal | 97.790% | 96.040% | 0.058851 | 0.338103 |
| 仅干净图像 | control | 96.820% | 94.320% | 0.059386 | 0.340177 |
| 仅干净图像 | v11g | 96.820% | 94.200% | 0.059386 | 0.340177 |
| 仅干净图像 | no_causal | 96.820% | 94.480% | 0.059386 | 0.340177 |
| 仅干净音频 | control | 97.710% | 96.000% | 0.046292 | 0.283510 |
| 仅干净音频 | v11g | 98.090% | 96.000% | 0.045921 | 0.282255 |
| 仅干净音频 | no_causal | 97.710% | 96.000% | 0.046246 | 0.283448 |
| 图像干净、音频残缺 | control | 97.830% | 94.770% | 0.058946 | 0.337707 |
| 图像干净、音频残缺 | v11g | 97.830% | 94.960% | 0.058946 | 0.337707 |
| 图像干净、音频残缺 | no_causal | 97.830% | 94.910% | 0.058946 | 0.337707 |
| 音频干净、图像残缺 | control | 97.280% | 95.950% | 0.063072 | 0.350291 |
| 音频干净、图像残缺 | v11g | 97.170% | 95.950% | 0.062500 | 0.348644 |
| 音频干净、图像残缺 | no_causal | 97.130% | 95.950% | 0.062562 | 0.348808 |
| 双模态残缺 | control | 97.310% | 94.490% | 0.063027 | 0.352854 |
| 双模态残缺 | v11g | 97.280% | 94.630% | 0.062601 | 0.351639 |
| 双模态残缺 | no_causal | 97.250% | 94.580% | 0.062634 | 0.351731 |
| 仅音频残缺 | control | 96.200% | 94.330% | 0.045583 | 0.288516 |
| 仅音频残缺 | v11g | 96.450% | 94.330% | 0.045215 | 0.287266 |
| 仅音频残缺 | no_causal | 96.210% | 94.330% | 0.045499 | 0.288228 |
| 仅图像残缺 | control | 97.140% | 75.480% | 0.062985 | 0.350024 |
| 仅图像残缺 | v11g | 97.140% | 75.300% | 0.062985 | 0.350024 |
| 仅图像残缺 | no_causal | 97.140% | 76.380% | 0.062985 | 0.350024 |

**配对诊断**

| 输入模式 | 实验 | 配对 R@1 (图→音) | 配对 R@1 (音→图) |
| --- | --- | ---: | ---: |
| 干净双模态 | control | N/A | N/A |
| 干净双模态 | v11g | N/A | N/A |
| 干净双模态 | no_causal | N/A | N/A |
| 仅干净图像 | control | N/A | N/A |
| 仅干净图像 | v11g | N/A | N/A |
| 仅干净图像 | no_causal | N/A | N/A |
| 仅干净音频 | control | N/A | N/A |
| 仅干净音频 | v11g | N/A | N/A |
| 仅干净音频 | no_causal | N/A | N/A |
| 图像干净、音频残缺 | control | N/A | N/A |
| 图像干净、音频残缺 | v11g | N/A | N/A |
| 图像干净、音频残缺 | no_causal | N/A | N/A |
| 音频干净、图像残缺 | control | N/A | N/A |
| 音频干净、图像残缺 | v11g | N/A | N/A |
| 音频干净、图像残缺 | no_causal | N/A | N/A |
| 双模态残缺 | control | N/A | N/A |
| 双模态残缺 | v11g | N/A | N/A |
| 双模态残缺 | no_causal | N/A | N/A |
| 仅音频残缺 | control | N/A | N/A |
| 仅音频残缺 | v11g | N/A | N/A |
| 仅音频残缺 | no_causal | N/A | N/A |
| 仅图像残缺 | control | N/A | N/A |
| 仅图像残缺 | v11g | N/A | N/A |
| 仅图像残缺 | no_causal | N/A | N/A |

**音频恢复分布**

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 干净双模态 | control | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 干净双模态 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 仅干净图像 | control | 0.015100 | 0.068600 | 1.000000 |
| 仅干净图像 | v11g | 0.015600 | 0.068500 | 0.981700 |
| 仅干净图像 | no_causal | 0.015300 | 0.069200 | 0.996400 |
| 仅干净音频 | control | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | v11g | 0.043100 | 0.155600 | 1.000000 |
| 仅干净音频 | no_causal | 0.043100 | 0.155600 | 1.000000 |
| 图像干净、音频残缺 | control | 0.042700 | 0.154400 | 1.000000 |
| 图像干净、音频残缺 | v11g | 0.042800 | 0.154000 | 1.000000 |
| 图像干净、音频残缺 | no_causal | 0.043000 | 0.154700 | 1.000000 |
| 音频干净、图像残缺 | control | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | v11g | 0.043200 | 0.155700 | 1.000000 |
| 音频干净、图像残缺 | no_causal | 0.043200 | 0.155700 | 1.000000 |
| 双模态残缺 | control | 0.042800 | 0.154600 | 1.000000 |
| 双模态残缺 | v11g | 0.042900 | 0.154400 | 1.000000 |
| 双模态残缺 | no_causal | 0.043000 | 0.154800 | 1.000000 |
| 仅音频残缺 | control | 0.042700 | 0.154400 | 1.000000 |
| 仅音频残缺 | v11g | 0.042700 | 0.154400 | 1.000000 |
| 仅音频残缺 | no_causal | 0.042700 | 0.154400 | 1.000000 |
| 仅图像残缺 | control | 0.015200 | 0.063500 | 0.999500 |
| 仅图像残缺 | v11g | 0.015600 | 0.063500 | 0.996400 |
| 仅图像残缺 | no_causal | 0.015400 | 0.064100 | 0.998500 |

**音频目标分布**

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 干净双模态 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.200% |
| 仅干净图像 | control | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | v11g | 0.015600 | 0.070500 | 0.987900 | 81.300% |
| 仅干净图像 | no_causal | 0.015600 | 0.070500 | 0.987900 | 81.100% |
| 仅干净音频 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 仅干净音频 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 图像干净、音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 图像干净、音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 图像干净、音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 音频干净、图像残缺 | control | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 音频干净、图像残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 82.100% |
| 双模态残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 双模态残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 双模态残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.600% |
| 仅音频残缺 | control | 0.040600 | 0.152100 | 1.000000 | 81.500% |
| 仅音频残缺 | v11g | 0.040600 | 0.152100 | 1.000000 | 81.500% |
| 仅音频残缺 | no_causal | 0.040600 | 0.152100 | 1.000000 | 81.500% |
| 仅图像残缺 | control | 0.015600 | 0.070500 | 0.987900 | 76.700% |
| 仅图像残缺 | v11g | 0.015600 | 0.070500 | 0.987900 | 76.600% |
| 仅图像残缺 | no_causal | 0.015600 | 0.070500 | 0.987900 | 76.600% |

**本 family 干预**

**音频 Key → 图像：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净音频 | control | 0.002085 | 0.002085 | 0.002085 | 0.002086 |
| 仅干净音频 | v11g | 0.002053 | 0.002085 | 0.002050 | 0.002042 |
| 仅干净音频 | no_causal | 0.002085 | 0.002085 | 0.002096 | 0.002081 |
| 音频干净、图像残缺 | control | 0.013088 | 0.013088 | 0.013088 | 0.013088 |
| 音频干净、图像残缺 | v11g | 0.012708 | 0.013088 | 0.012717 | 0.012708 |
| 音频干净、图像残缺 | no_causal | 0.012956 | 0.013088 | 0.012964 | 0.012954 |
| 双模态残缺 | control | 0.013111 | 0.013111 | 0.013111 | 0.013112 |
| 双模态残缺 | v11g | 0.012746 | 0.013111 | 0.012751 | 0.012748 |
| 双模态残缺 | no_causal | 0.012981 | 0.013111 | 0.012992 | 0.012983 |
| 仅音频残缺 | control | 0.003567 | 0.003567 | 0.003567 | 0.003569 |
| 仅音频残缺 | v11g | 0.003502 | 0.003567 | 0.003493 | 0.003480 |
| 仅音频残缺 | no_causal | 0.003543 | 0.003567 | 0.003557 | 0.003538 |

**音频 Key → 图像：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净音频 | v11g | 0.000032 | -0.000003 | -0.000012 |
| 仅干净音频 | no_causal | 1.001e-08 | 0.000011 | -0.000005 |
| 音频干净、图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.000381 | 0.000010 | 5.853e-07 |
| 音频干净、图像残缺 | no_causal | 0.000132 | 0.000008 | -8.805e-07 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000365 | 0.000005 | 6.599e-07 |
| 双模态残缺 | no_causal | 0.000131 | 0.000011 | 0.000002 |
| 仅音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅音频残缺 | v11g | 0.000065 | -0.000009 | -0.000023 |
| 仅音频残缺 | no_causal | 0.000024 | 0.000014 | -0.000006 |

**音频 Key → 图像：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净音频 | control | 0.000% | 0.000% | 0.000% |
| 仅干净音频 | v11g | 36.370% | 63.070% | 27.930% |
| 仅干净音频 | no_causal | 30.170% | 65.550% | 25.520% |
| 音频干净、图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 音频干净、图像残缺 | v11g | 76.040% | 51.810% | 41.530% |
| 音频干净、图像残缺 | no_causal | 60.680% | 52.100% | 34.600% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 77.270% | 51.180% | 41.590% |
| 双模态残缺 | no_causal | 61.190% | 51.410% | 35.470% |
| 仅音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅音频残缺 | v11g | 42.650% | 61.270% | 30.930% |
| 仅音频残缺 | no_causal | 31.720% | 62.840% | 25.890% |

**音频 Key → 图像：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净音频 | control | N/A | 0.000000 |
| 仅干净音频 | v11g | 0.534002 | 0.436642 |
| 仅干净音频 | no_causal | 0.757989 | 0.179866 |
| 音频干净、图像残缺 | control | N/A | 0.000000 |
| 音频干净、图像残缺 | v11g | 0.577842 | 0.301388 |
| 音频干净、图像残缺 | no_causal | 0.755822 | 0.117478 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.577687 | 0.285628 |
| 双模态残缺 | no_causal | 0.757593 | 0.112098 |
| 仅音频残缺 | control | N/A | 0.000000 |
| 仅音频残缺 | v11g | 0.534045 | 0.415506 |
| 仅音频残缺 | no_causal | 0.758642 | 0.172015 |

**图像 Key → 音频：绝对误差**

| 输入模式 | 实验 | normal MSE | zero MSE | wrong MSE | same-class MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000289 | 0.000289 | 0.000289 | 0.000289 |
| 仅干净图像 | v11g | 0.000294 | 0.000289 | 0.000304 | 0.000295 |
| 仅干净图像 | no_causal | 0.000296 | 0.000289 | 0.000324 | 0.000301 |
| 图像干净、音频残缺 | control | 0.005272 | 0.005272 | 0.005272 | 0.005274 |
| 图像干净、音频残缺 | v11g | 0.004953 | 0.005272 | 0.004962 | 0.004955 |
| 图像干净、音频残缺 | no_causal | 0.005060 | 0.005272 | 0.005105 | 0.005066 |
| 双模态残缺 | control | 0.005077 | 0.005077 | 0.005077 | 0.005075 |
| 双模态残缺 | v11g | 0.004860 | 0.005077 | 0.004861 | 0.004860 |
| 双模态残缺 | no_causal | 0.004936 | 0.005077 | 0.004949 | 0.004936 |
| 仅图像残缺 | control | 0.000917 | 0.000917 | 0.000917 | 0.000917 |
| 仅图像残缺 | v11g | 0.000919 | 0.000917 | 0.000922 | 0.000920 |
| 仅图像残缺 | no_causal | 0.000921 | 0.000917 | 0.000927 | 0.000930 |

**图像 Key → 音频：配对增益与损害**

| 输入模式 | 实验 | zero−normal | wrong−normal | same−normal（配对） |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅干净图像 | v11g | -0.000005 | 0.000010 | 7.588e-07 |
| 仅干净图像 | no_causal | -0.000007 | 0.000028 | 0.000004 |
| 图像干净、音频残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.000319 | 0.000010 | 4.144e-07 |
| 图像干净、音频残缺 | no_causal | 0.000212 | 0.000046 | 0.000004 |
| 双模态残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 双模态残缺 | v11g | 0.000216 | 9.051e-07 | 0.000002 |
| 双模态残缺 | no_causal | 0.000141 | 0.000013 | 0.000002 |
| 仅图像残缺 | control | 0.000000 | 0.000000 | 0.000000 |
| 仅图像残缺 | v11g | -0.000002 | 0.000003 | 9.614e-07 |
| 仅图像残缺 | no_causal | -0.000005 | 0.000006 | 0.000008 |

**图像 Key → 音频：逐样本改善比例**

| 输入模式 | 实验 | win_zero | win_wrong | win_both |
| --- | --- | ---: | ---: | ---: |
| 仅干净图像 | control | 0.000% | 0.000% | 0.000% |
| 仅干净图像 | v11g | 35.190% | 69.260% | 26.800% |
| 仅干净图像 | no_causal | 28.510% | 71.180% | 23.370% |
| 图像干净、音频残缺 | control | 0.000% | 0.000% | 0.000% |
| 图像干净、音频残缺 | v11g | 42.000% | 30.750% | 23.050% |
| 图像干净、音频残缺 | no_causal | 33.920% | 31.680% | 20.620% |
| 双模态残缺 | control | 0.000% | 0.000% | 0.000% |
| 双模态残缺 | v11g | 42.070% | 28.880% | 21.750% |
| 双模态残缺 | no_causal | 33.970% | 29.030% | 18.980% |
| 仅图像残缺 | control | 0.000% | 0.000% | 0.000% |
| 仅图像残缺 | v11g | 30.920% | 63.690% | 21.350% |
| 仅图像残缺 | no_causal | 46.210% | 60.960% | 30.570% |

**图像 Key → 音频：通路幅度**

| 输入模式 | 实验 | gate | res/V |
| --- | --- | ---: | ---: |
| 仅干净图像 | control | N/A | 0.000000 |
| 仅干净图像 | v11g | 0.947892 | 0.040982 |
| 仅干净图像 | no_causal | 0.961301 | 0.027724 |
| 图像干净、音频残缺 | control | N/A | 0.000000 |
| 图像干净、音频残缺 | v11g | 0.955443 | 0.007644 |
| 图像干净、音频残缺 | no_causal | 0.968722 | 0.005985 |
| 双模态残缺 | control | N/A | 0.000000 |
| 双模态残缺 | v11g | 0.953498 | 0.005858 |
| 双模态残缺 | no_causal | 0.967517 | 0.004475 |
| 仅图像残缺 | control | N/A | 0.000000 |
| 仅图像残缺 | v11g | 0.945801 | 0.030067 |
| 仅图像残缺 | no_causal | 0.959936 | 0.019950 |

**干预后的图像内容分类**

| 输入模式 | 实验 | 图像内容 ACC (normal) | 图像内容 ACC (zero) | 图像内容 ACC (wrong) | 图像内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | v11g | 97.790% | 97.790% | 97.790% | 97.789% |
| 干净双模态 | no_causal | 97.790% | 97.790% | 97.790% | 97.789% |
| 仅干净图像 | control | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | v11g | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净图像 | no_causal | 96.820% | 96.820% | 96.820% | 96.819% |
| 仅干净音频 | control | 97.710% | 97.710% | 97.710% | 97.709% |
| 仅干净音频 | v11g | 98.090% | 97.710% | 97.490% | 97.499% |
| 仅干净音频 | no_causal | 97.710% | 97.710% | 97.710% | 97.579% |
| 图像干净、音频残缺 | control | 97.830% | 97.830% | 97.830% | 97.829% |
| 图像干净、音频残缺 | v11g | 97.830% | 97.830% | 97.830% | 97.829% |
| 图像干净、音频残缺 | no_causal | 97.830% | 97.830% | 97.830% | 97.829% |
| 音频干净、图像残缺 | control | 97.280% | 97.280% | 97.280% | 97.279% |
| 音频干净、图像残缺 | v11g | 97.170% | 97.280% | 97.250% | 97.299% |
| 音频干净、图像残缺 | no_causal | 97.130% | 97.280% | 97.220% | 97.289% |
| 双模态残缺 | control | 97.310% | 97.310% | 97.310% | 97.309% |
| 双模态残缺 | v11g | 97.280% | 97.310% | 97.340% | 97.319% |
| 双模态残缺 | no_causal | 97.250% | 97.310% | 97.250% | 97.279% |
| 仅音频残缺 | control | 96.200% | 96.200% | 96.200% | 96.198% |
| 仅音频残缺 | v11g | 96.450% | 96.200% | 96.270% | 96.238% |
| 仅音频残缺 | no_causal | 96.210% | 96.200% | 96.340% | 96.269% |
| 仅图像残缺 | control | 97.140% | 97.140% | 97.140% | 97.139% |
| 仅图像残缺 | v11g | 97.140% | 97.140% | 97.140% | 97.139% |
| 仅图像残缺 | no_causal | 97.140% | 97.140% | 97.140% | 97.139% |

**干预后的音频内容分类**

| 输入模式 | 实验 | 音频内容 ACC (normal) | 音频内容 ACC (zero) | 音频内容 ACC (wrong) | 音频内容 ACC (same_class) |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净双模态 | control | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | v11g | 96.040% | 96.040% | 96.040% | 96.038% |
| 干净双模态 | no_causal | 96.040% | 96.040% | 96.040% | 96.038% |
| 仅干净图像 | control | 94.320% | 94.320% | 94.320% | 94.318% |
| 仅干净图像 | v11g | 94.200% | 94.320% | 94.080% | 94.088% |
| 仅干净图像 | no_causal | 94.480% | 94.320% | 94.690% | 94.538% |
| 仅干净音频 | control | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | v11g | 96.000% | 96.000% | 96.000% | 95.998% |
| 仅干净音频 | no_causal | 96.000% | 96.000% | 96.000% | 95.998% |
| 图像干净、音频残缺 | control | 94.770% | 94.770% | 94.770% | 94.768% |
| 图像干净、音频残缺 | v11g | 94.960% | 94.770% | 94.900% | 94.978% |
| 图像干净、音频残缺 | no_causal | 94.910% | 94.770% | 94.770% | 94.868% |
| 音频干净、图像残缺 | control | 95.950% | 95.950% | 95.950% | 95.948% |
| 音频干净、图像残缺 | v11g | 95.950% | 95.950% | 95.950% | 95.948% |
| 音频干净、图像残缺 | no_causal | 95.950% | 95.950% | 95.950% | 95.948% |
| 双模态残缺 | control | 94.490% | 94.490% | 94.490% | 94.488% |
| 双模态残缺 | v11g | 94.630% | 94.490% | 94.560% | 94.568% |
| 双模态残缺 | no_causal | 94.580% | 94.490% | 94.450% | 94.538% |
| 仅音频残缺 | control | 94.330% | 94.330% | 94.330% | 94.328% |
| 仅音频残缺 | v11g | 94.330% | 94.330% | 94.330% | 94.328% |
| 仅音频残缺 | no_causal | 94.330% | 94.330% | 94.330% | 94.328% |
| 仅图像残缺 | control | 75.480% | 75.480% | 75.480% | 75.470% |
| 仅图像残缺 | v11g | 75.300% | 75.480% | 75.180% | 75.060% |
| 仅图像残缺 | no_causal | 76.380% | 75.480% | 76.230% | 76.311% |

</details>

**独立音频 family breakdown**

此处来自各实验的 `tables/audio_family_breakdown_fixed.csv`，单独改变音频 family；不能用它替代上面的双 family-pair 主表或再次计入宏平均。该 CSV 不含逐指标 n。


<details>
<summary>time_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.030% | 0.009772 | 0.942626 | 0.006218 | 0.833523 |
| 图像干净、音频残缺 | v11g | 99.030% | 0.009772 | 0.942626 | 0.006048 | 0.836068 |
| 图像干净、音频残缺 | no_causal | 99.030% | 0.009772 | 0.942626 | 0.006073 | 0.837172 |
| 双模态残缺 | control | 97.710% | 0.008706 | 0.948581 | 0.006304 | 0.830981 |
| 双模态残缺 | v11g | 97.710% | 0.008476 | 0.949814 | 0.006162 | 0.833107 |
| 双模态残缺 | no_causal | 97.710% | 0.008486 | 0.949871 | 0.006181 | 0.834065 |
| 仅音频残缺 | control | 90.390% | 0.009307 | 0.923568 | 0.006326 | 0.830664 |
| 仅音频残缺 | v11g | 90.390% | 0.009092 | 0.926963 | 0.006326 | 0.830664 |
| 仅音频残缺 | no_causal | 90.390% | 0.009042 | 0.927479 | 0.006326 | 0.830664 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.776195 | 0.009109 | 0.023938 |
| 图像干净、音频残缺 | v11g | 20.776195 | 0.008691 | 0.023848 |
| 图像干净、音频残缺 | no_causal | 20.776195 | 0.008752 | 0.023857 |
| 双模态残缺 | control | 23.123210 | 0.009356 | 0.024483 |
| 双模态残缺 | v11g | 23.197685 | 0.009008 | 0.024396 |
| 双模态残缺 | no_causal | 23.233468 | 0.009054 | 0.024403 |
| 仅音频残缺 | control | 32.712509 | 0.009393 | 0.024228 |
| 仅音频残缺 | v11g | 32.494704 | 0.009393 | 0.024228 |
| 仅音频残缺 | no_causal | 32.140332 | 0.009393 | 0.024228 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.004240 | 0.017718 |
| 图像干净、音频残缺 | v11g | 0.004240 | 0.017718 |
| 图像干净、音频残缺 | no_causal | 0.004240 | 0.017718 |
| 双模态残缺 | control | 0.004215 | 0.017613 |
| 双模态残缺 | v11g | 0.004215 | 0.017613 |
| 双模态残缺 | no_causal | 0.004215 | 0.017613 |
| 仅音频残缺 | control | 0.004228 | 0.017696 |
| 仅音频残缺 | v11g | 0.004228 | 0.017696 |
| 仅音频残缺 | no_causal | 0.004228 | 0.017696 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.147264 | 0.152054 | 78.086% |
| 图像干净、音频残缺 | v11g | 0.146717 | 0.152054 | 78.208% |
| 图像干净、音频残缺 | no_causal | 0.148177 | 0.152054 | 78.161% |
| 双模态残缺 | control | 0.147181 | 0.152054 | 77.929% |
| 双模态残缺 | v11g | 0.146750 | 0.152054 | 78.033% |
| 双模态残缺 | no_causal | 0.147923 | 0.152054 | 77.990% |
| 仅音频残缺 | control | 0.146734 | 0.152054 | 77.866% |
| 仅音频残缺 | v11g | 0.146734 | 0.152054 | 77.866% |
| 仅音频残缺 | no_causal | 0.146734 | 0.152054 | 77.866% |

</details>


<details>
<summary>freq_mask</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 98.840% | 0.009763 | 0.942693 | 0.004856 | 0.860159 |
| 图像干净、音频残缺 | v11g | 98.840% | 0.009763 | 0.942693 | 0.004711 | 0.862336 |
| 图像干净、音频残缺 | no_causal | 98.840% | 0.009763 | 0.942693 | 0.004762 | 0.862068 |
| 双模态残缺 | control | 97.910% | 0.008750 | 0.948192 | 0.004888 | 0.859557 |
| 双模态残缺 | v11g | 97.910% | 0.008523 | 0.949420 | 0.004767 | 0.861309 |
| 双模态残缺 | no_causal | 97.910% | 0.008546 | 0.949415 | 0.004813 | 0.860953 |
| 仅音频残缺 | control | 92.220% | 0.007925 | 0.938894 | 0.004905 | 0.858229 |
| 仅音频残缺 | v11g | 92.220% | 0.007764 | 0.940868 | 0.004905 | 0.858229 |
| 仅音频残缺 | no_causal | 92.220% | 0.007821 | 0.939995 | 0.004905 | 0.858229 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.781139 | 0.006477 | 0.021425 |
| 图像干净、音频残缺 | v11g | 20.781139 | 0.006118 | 0.021512 |
| 图像干净、音频残缺 | no_causal | 20.781139 | 0.006246 | 0.021616 |
| 双模态残缺 | control | 23.020364 | 0.006543 | 0.021547 |
| 双模态残缺 | v11g | 23.094092 | 0.006244 | 0.021591 |
| 双模态残缺 | no_causal | 23.122557 | 0.006357 | 0.021696 |
| 仅音频残缺 | control | 31.814109 | 0.006583 | 0.021626 |
| 仅音频残缺 | v11g | 31.708459 | 0.006583 | 0.021626 |
| 仅音频残缺 | no_causal | 31.376858 | 0.006583 | 0.021626 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.003747 | 0.016059 |
| 图像干净、音频残缺 | v11g | 0.003747 | 0.016059 |
| 图像干净、音频残缺 | no_causal | 0.003747 | 0.016059 |
| 双模态残缺 | control | 0.003756 | 0.016091 |
| 双模态残缺 | v11g | 0.003756 | 0.016091 |
| 双模态残缺 | no_causal | 0.003756 | 0.016091 |
| 仅音频残缺 | control | 0.003757 | 0.016081 |
| 仅音频残缺 | v11g | 0.003757 | 0.016081 |
| 仅音频残缺 | no_causal | 0.003757 | 0.016081 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.151326 | 0.152054 | 80.131% |
| 图像干净、音频残缺 | v11g | 0.150158 | 0.152054 | 80.160% |
| 图像干净、音频残缺 | no_causal | 0.152195 | 0.152054 | 80.138% |
| 双模态残缺 | control | 0.151691 | 0.152054 | 80.076% |
| 双模态残缺 | v11g | 0.150793 | 0.152054 | 80.109% |
| 双模态残缺 | no_causal | 0.152405 | 0.152054 | 80.092% |
| 仅音频残缺 | control | 0.151239 | 0.152054 | 80.019% |
| 仅音频残缺 | v11g | 0.151239 | 0.152054 | 80.019% |
| 仅音频残缺 | no_causal | 0.151239 | 0.152054 | 80.019% |

</details>


<details>
<summary>feature_dropout</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.320% | 0.009733 | 0.942740 | 0.003738 | 0.904314 |
| 图像干净、音频残缺 | v11g | 99.320% | 0.009733 | 0.942740 | 0.003662 | 0.905451 |
| 图像干净、音频残缺 | no_causal | 99.320% | 0.009733 | 0.942740 | 0.003690 | 0.904974 |
| 双模态残缺 | control | 99.030% | 0.008696 | 0.948433 | 0.003731 | 0.904324 |
| 双模态残缺 | v11g | 99.030% | 0.008451 | 0.949710 | 0.003669 | 0.905277 |
| 双模态残缺 | no_causal | 99.030% | 0.008487 | 0.949644 | 0.003692 | 0.904873 |
| 仅音频残缺 | control | 97.100% | 0.003009 | 0.978650 | 0.003738 | 0.904178 |
| 仅音频残缺 | v11g | 97.100% | 0.002965 | 0.979222 | 0.003738 | 0.904178 |
| 仅音频残缺 | no_causal | 97.100% | 0.003000 | 0.978763 | 0.003738 | 0.904178 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.793747 | 0.003796 | 0.015939 |
| 图像干净、音频残缺 | v11g | 20.793747 | 0.003607 | 0.015729 |
| 图像干净、音频残缺 | no_causal | 20.793747 | 0.003677 | 0.015804 |
| 双模态残缺 | control | 23.151188 | 0.003790 | 0.015938 |
| 双模态残缺 | v11g | 23.236763 | 0.003634 | 0.015761 |
| 双模态残缺 | no_causal | 23.265952 | 0.003692 | 0.015822 |
| 仅音频残缺 | control | 36.140352 | 0.003797 | 0.015945 |
| 仅音频残缺 | v11g | 35.848476 | 0.003797 | 0.015945 |
| 仅音频残缺 | no_causal | 35.462194 | 0.003797 | 0.015945 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.003698 | 0.015718 |
| 图像干净、音频残缺 | v11g | 0.003698 | 0.015718 |
| 图像干净、音频残缺 | no_causal | 0.003698 | 0.015718 |
| 双模态残缺 | control | 0.003691 | 0.015713 |
| 双模态残缺 | v11g | 0.003691 | 0.015713 |
| 双模态残缺 | no_causal | 0.003691 | 0.015713 |
| 仅音频残缺 | control | 0.003699 | 0.015725 |
| 仅音频残缺 | v11g | 0.003699 | 0.015725 |
| 仅音频残缺 | no_causal | 0.003699 | 0.015725 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.153838 | 0.152054 | 81.663% |
| 图像干净、音频残缺 | v11g | 0.153236 | 0.152054 | 81.694% |
| 图像干净、音频残缺 | no_causal | 0.153839 | 0.152054 | 81.681% |
| 双模态残缺 | control | 0.153931 | 0.152054 | 81.662% |
| 双模态残缺 | v11g | 0.153454 | 0.152054 | 81.685% |
| 双模态残缺 | no_causal | 0.153931 | 0.152054 | 81.678% |
| 仅音频残缺 | control | 0.153804 | 0.152054 | 81.634% |
| 仅音频残缺 | v11g | 0.153804 | 0.152054 | 81.634% |
| 仅音频残缺 | no_causal | 0.153804 | 0.152054 | 81.634% |

</details>


<details>
<summary>partial_temporal</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 98.070% | 0.009949 | 0.942169 | 0.011784 | 0.538387 |
| 图像干净、音频残缺 | v11g | 98.070% | 0.009949 | 0.942169 | 0.011342 | 0.543528 |
| 图像干净、音频残缺 | no_causal | 98.070% | 0.009949 | 0.942169 | 0.011563 | 0.544686 |
| 双模态残缺 | control | 92.030% | 0.008929 | 0.947314 | 0.011935 | 0.536746 |
| 双模态残缺 | v11g | 92.030% | 0.008744 | 0.948459 | 0.011577 | 0.540671 |
| 双模态残缺 | no_causal | 92.030% | 0.008706 | 0.948636 | 0.011760 | 0.541756 |
| 仅音频残缺 | control | 45.880% | 0.042710 | 0.569261 | 0.012585 | 0.528776 |
| 仅音频残缺 | v11g | 45.880% | 0.040929 | 0.602984 | 0.012585 | 0.528776 |
| 仅音频残缺 | no_causal | 45.880% | 0.038806 | 0.634932 | 0.012585 | 0.528776 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.701504 | 0.028559 | 0.074284 |
| 图像干净、音频残缺 | v11g | 20.701504 | 0.027472 | 0.074671 |
| 图像干净、音频残缺 | no_causal | 20.701504 | 0.028015 | 0.074921 |
| 双模态残缺 | control | 22.906776 | 0.028924 | 0.074755 |
| 双模态残缺 | v11g | 22.956473 | 0.028043 | 0.075043 |
| 双模态残缺 | no_causal | 22.919407 | 0.028493 | 0.075261 |
| 仅音频残缺 | control | 17.523188 | 0.030529 | 0.075829 |
| 仅音频残缺 | v11g | 17.649859 | 0.030529 | 0.075829 |
| 仅音频残缺 | no_causal | 17.674417 | 0.030529 | 0.075829 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.000306 | 0.001634 |
| 图像干净、音频残缺 | v11g | 0.000306 | 0.001634 |
| 图像干净、音频残缺 | no_causal | 0.000306 | 0.001634 |
| 双模态残缺 | control | 0.000311 | 0.001646 |
| 双模态残缺 | v11g | 0.000311 | 0.001646 |
| 双模态残缺 | no_causal | 0.000311 | 0.001646 |
| 仅音频残缺 | control | 0.000308 | 0.001635 |
| 仅音频残缺 | v11g | 0.000308 | 0.001635 |
| 仅音频残缺 | no_causal | 0.000308 | 0.001635 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.131051 | 0.152054 | 70.184% |
| 图像干净、音频残缺 | v11g | 0.128595 | 0.152054 | 70.409% |
| 图像干净、音频残缺 | no_causal | 0.133901 | 0.152054 | 70.491% |
| 双模态残缺 | control | 0.130508 | 0.152054 | 69.956% |
| 双模态残缺 | v11g | 0.128590 | 0.152054 | 70.132% |
| 双模态残缺 | no_causal | 0.132946 | 0.152054 | 70.200% |
| 仅音频残缺 | control | 0.126849 | 0.152054 | 68.606% |
| 仅音频残缺 | v11g | 0.126849 | 0.152054 | 68.606% |
| 仅音频残缺 | no_causal | 0.126849 | 0.152054 | 68.606% |

</details>


<details>
<summary>time_freq_block</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 99.120% | 0.009738 | 0.942713 | 0.003986 | 0.900341 |
| 图像干净、音频残缺 | v11g | 99.120% | 0.009738 | 0.942713 | 0.003935 | 0.900968 |
| 图像干净、音频残缺 | no_causal | 99.120% | 0.009738 | 0.942713 | 0.003953 | 0.900842 |
| 双模态残缺 | control | 98.770% | 0.008698 | 0.948653 | 0.003986 | 0.900257 |
| 双模态残缺 | v11g | 98.770% | 0.008455 | 0.949928 | 0.003944 | 0.900763 |
| 双模态残缺 | no_causal | 98.770% | 0.008470 | 0.949959 | 0.003959 | 0.900634 |
| 仅音频残缺 | control | 96.750% | 0.003525 | 0.974426 | 0.003997 | 0.900021 |
| 仅音频残缺 | v11g | 96.750% | 0.003464 | 0.975121 | 0.003997 | 0.900021 |
| 仅音频残缺 | no_causal | 96.750% | 0.003506 | 0.974639 | 0.003997 | 0.900021 |

| 输入模式 | 实验 | 图像 PSNR (dB) | 音频缺失区 MSE | 音频缺失区 L1 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 20.790810 | 0.005269 | 0.017942 |
| 图像干净、音频残缺 | v11g | 20.790810 | 0.004961 | 0.017957 |
| 图像干净、音频残缺 | no_causal | 20.790810 | 0.005068 | 0.018062 |
| 双模态残缺 | control | 23.127621 | 0.005198 | 0.017747 |
| 双模态残缺 | v11g | 23.216220 | 0.004942 | 0.017744 |
| 双模态残缺 | no_causal | 23.250161 | 0.005031 | 0.017833 |
| 仅音频残缺 | control | 36.145908 | 0.005324 | 0.018041 |
| 仅音频残缺 | v11g | 35.845417 | 0.005324 | 0.018041 |
| 仅音频残缺 | no_causal | 35.493708 | 0.005324 | 0.018041 |

| 输入模式 | 实验 | 音频可见区 MSE | 音频可见区 L1 |
| --- | --- | ---: | ---: |
| 图像干净、音频残缺 | control | 0.003733 | 0.015994 |
| 图像干净、音频残缺 | v11g | 0.003733 | 0.015994 |
| 图像干净、音频残缺 | no_causal | 0.003733 | 0.015994 |
| 双模态残缺 | control | 0.003747 | 0.016053 |
| 双模态残缺 | v11g | 0.003747 | 0.016053 |
| 双模态残缺 | no_causal | 0.003747 | 0.016053 |
| 仅音频残缺 | control | 0.003734 | 0.015998 |
| 仅音频残缺 | v11g | 0.003734 | 0.015998 |
| 仅音频残缺 | no_causal | 0.003734 | 0.015998 |

| 输入模式 | 实验 | 恢复标准差 | 目标标准差 | top15% 召回 |
| --- | --- | ---: | ---: | ---: |
| 图像干净、音频残缺 | control | 0.154400 | 0.152054 | 81.583% |
| 图像干净、音频残缺 | v11g | 0.154069 | 0.152054 | 81.599% |
| 图像干净、音频残缺 | no_causal | 0.154681 | 0.152054 | 81.592% |
| 双模态残缺 | control | 0.154617 | 0.152054 | 81.568% |
| 双模态残缺 | v11g | 0.154356 | 0.152054 | 81.584% |
| 双模态残缺 | no_causal | 0.154854 | 0.152054 | 81.570% |
| 仅音频残缺 | control | 0.154387 | 0.152054 | 81.547% |
| 仅音频残缺 | v11g | 0.154387 | 0.152054 | 81.547% |
| 仅音频残缺 | no_causal | 0.154387 | 0.152054 | 81.547% |

</details>

**训练统计**

仅统计每个完整 epoch 的平均训练 loss；不同 loss 定义不可横比，最低训练 loss 也不是测试最优 checkpoint。

| 实验 | 完成 epoch | 本轮轮数 | 首轮 loss | 末轮 loss | 最低 loss | 末轮 LR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| control | 冻结父模型 | 0 | N/A | N/A | N/A | N/A |
| v11g | 0–29 | 30 | 0.7827 | 0.8213 | 0.7632 | 0.000010 |
| no_causal | 0–29 | 30 | 0.7385 | 0.7790 | 0.7203 | 0.000010 |


<details>
<summary>逐 epoch loss / LR</summary>

| epoch | 实验 | 平均 loss | LR |
| --- | --- | ---: | ---: |
| 0 | v11g | 0.7827 | 0.000100 |
| 0 | no_causal | 0.7385 | 0.000100 |
| 1 | v11g | 0.8434 | 0.000100 |
| 1 | no_causal | 0.7984 | 0.000100 |
| 2 | v11g | 0.9059 | 0.000099 |
| 2 | no_causal | 0.8607 | 0.000099 |
| 3 | v11g | 0.8205 | 0.000098 |
| 3 | no_causal | 0.7762 | 0.000098 |
| 4 | v11g | 0.8441 | 0.000096 |
| 4 | no_causal | 0.8005 | 0.000096 |
| 5 | v11g | 0.8013 | 0.000094 |
| 5 | no_causal | 0.7583 | 0.000094 |
| 6 | v11g | 0.8487 | 0.000091 |
| 6 | no_causal | 0.8046 | 0.000091 |
| 7 | v11g | 0.8489 | 0.000088 |
| 7 | no_causal | 0.8064 | 0.000088 |
| 8 | v11g | 0.8207 | 0.000085 |
| 8 | no_causal | 0.7770 | 0.000085 |
| 9 | v11g | 0.8206 | 0.000081 |
| 9 | no_causal | 0.7787 | 0.000081 |
| 10 | v11g | 0.8460 | 0.000077 |
| 10 | no_causal | 0.8039 | 0.000077 |
| 11 | v11g | 0.8060 | 0.000073 |
| 11 | no_causal | 0.7616 | 0.000073 |
| 12 | v11g | 0.7915 | 0.000069 |
| 12 | no_causal | 0.7495 | 0.000069 |
| 13 | v11g | 0.8713 | 0.000064 |
| 13 | no_causal | 0.8282 | 0.000064 |
| 14 | v11g | 0.7739 | 0.000060 |
| 14 | no_causal | 0.7301 | 0.000060 |
| 15 | v11g | 0.8119 | 0.000055 |
| 15 | no_causal | 0.7687 | 0.000055 |
| 16 | v11g | 0.7713 | 0.000050 |
| 16 | no_causal | 0.7285 | 0.000050 |
| 17 | v11g | 0.7632 | 0.000046 |
| 17 | no_causal | 0.7203 | 0.000046 |
| 18 | v11g | 0.7760 | 0.000041 |
| 18 | no_causal | 0.7331 | 0.000041 |
| 19 | v11g | 0.7905 | 0.000037 |
| 19 | no_causal | 0.7487 | 0.000037 |
| 20 | v11g | 0.8049 | 0.000033 |
| 20 | no_causal | 0.7618 | 0.000033 |
| 21 | v11g | 0.7819 | 0.000029 |
| 21 | no_causal | 0.7391 | 0.000029 |
| 22 | v11g | 0.7896 | 0.000025 |
| 22 | no_causal | 0.7461 | 0.000025 |
| 23 | v11g | 0.8543 | 0.000022 |
| 23 | no_causal | 0.8117 | 0.000022 |
| 24 | v11g | 0.8332 | 0.000019 |
| 24 | no_causal | 0.7906 | 0.000019 |
| 25 | v11g | 0.8282 | 0.000016 |
| 25 | no_causal | 0.7850 | 0.000016 |
| 26 | v11g | 0.8383 | 0.000014 |
| 26 | no_causal | 0.7952 | 0.000014 |
| 27 | v11g | 0.7933 | 0.000012 |
| 27 | no_causal | 0.7502 | 0.000012 |
| 28 | v11g | 0.7645 | 0.000011 |
| 28 | no_causal | 0.7228 | 0.000011 |
| 29 | v11g | 0.8213 | 0.000010 |
| 29 | no_causal | 0.7790 | 0.000010 |

</details>

训练证据：
- `v11g`：[train.log](../v11g_outputs/outputs/outputs_v11g/logs/train.log)
- `no_causal`：[train.log](../v11g_outputs/outputs/outputs_v11g_no_causal/logs/train.log)

**Demo 小样本**

每份 demo 为 10 条样本，不是全测试集分数；fixed/random 分开。原始图片与逐样本预测留在对应产物目录，本节不宣称重新进行了图像质量评审。


<details>
<summary>fixed_mask demo (n=10)</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 70.000% | 0.023100 | 0.775000 | 0.009000 | 0.766000 |
| 仅音频残缺 | v11g | 70.000% | 0.022500 | 0.790000 | 0.009000 | 0.766000 |
| 仅音频残缺 | no_causal | 70.000% | 0.022400 | 0.794000 | 0.009000 | 0.766000 |
| 仅图像残缺 | control | 90.000% | 0.006400 | 0.957000 | 0.001500 | 0.867000 |
| 仅图像残缺 | v11g | 90.000% | 0.006400 | 0.957000 | 0.001500 | 0.868000 |
| 仅图像残缺 | no_causal | 90.000% | 0.006400 | 0.957000 | 0.001500 | 0.866000 |
| 双模态残缺 | control | 100.000% | 0.006400 | 0.958000 | 0.008500 | 0.779000 |
| 双模态残缺 | v11g | 100.000% | 0.006300 | 0.958000 | 0.008200 | 0.782000 |
| 双模态残缺 | no_causal | 100.000% | 0.006200 | 0.959000 | 0.008200 | 0.785000 |

| 输入模式 | 实验 | 图像缺失区 MSE | 音频缺失区 MSE |
| --- | --- | ---: | ---: |
| 仅音频残缺 | control | 0.023100 | 0.021000 |
| 仅音频残缺 | v11g | 0.022500 | 0.021000 |
| 仅音频残缺 | no_causal | 0.022400 | 0.021000 |
| 仅图像残缺 | control | 0.019500 | 0.001500 |
| 仅图像残缺 | v11g | 0.019500 | 0.001500 |
| 仅图像残缺 | no_causal | 0.019500 | 0.001500 |
| 双模态残缺 | control | 0.019400 | 0.019900 |
| 双模态残缺 | v11g | 0.018900 | 0.019100 |
| 双模态残缺 | no_causal | 0.018700 | 0.018800 |

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.040300 | 0.147100 | 1.000000 |
| 仅音频残缺 | v11g | 0.040300 | 0.147100 | 1.000000 |
| 仅音频残缺 | no_causal | 0.040300 | 0.147100 | 1.000000 |
| 仅图像残缺 | control | 0.016500 | 0.069800 | 0.934600 |
| 仅图像残缺 | v11g | 0.016900 | 0.070000 | 0.880400 |
| 仅图像残缺 | no_causal | 0.016600 | 0.070300 | 0.886100 |
| 双模态残缺 | control | 0.041100 | 0.148800 | 1.000000 |
| 双模态残缺 | v11g | 0.041400 | 0.147300 | 1.000000 |
| 双模态残缺 | no_causal | 0.042300 | 0.150600 | 1.000000 |

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.047400 | 0.168700 | 1.000000 | 79.400% |
| 仅音频残缺 | v11g | 0.047400 | 0.168700 | 1.000000 | 79.400% |
| 仅音频残缺 | no_causal | 0.047400 | 0.168700 | 1.000000 | 79.400% |
| 仅图像残缺 | control | 0.020700 | 0.084100 | 0.987900 | 82.900% |
| 仅图像残缺 | v11g | 0.020700 | 0.084100 | 0.987900 | 82.800% |
| 仅图像残缺 | no_causal | 0.020700 | 0.084100 | 0.987900 | 82.700% |
| 双模态残缺 | control | 0.047400 | 0.168700 | 1.000000 | 79.500% |
| 双模态残缺 | v11g | 0.047400 | 0.168700 | 1.000000 | 79.500% |
| 双模态残缺 | no_causal | 0.047400 | 0.168700 | 1.000000 | 79.300% |

</details>


<details>
<summary>legacy_random demo (n=10)</summary>

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 60.000% | 0.018900 | 0.814000 | 0.015900 | 0.647000 |
| 仅音频残缺 | v11g | 60.000% | 0.018700 | 0.825000 | 0.015900 | 0.647000 |
| 仅音频残缺 | no_causal | 60.000% | 0.018400 | 0.829000 | 0.015900 | 0.647000 |
| 仅图像残缺 | control | 80.000% | 0.012000 | 0.925000 | 0.002000 | 0.840000 |
| 仅图像残缺 | v11g | 80.000% | 0.012000 | 0.925000 | 0.002000 | 0.841000 |
| 仅图像残缺 | no_causal | 80.000% | 0.012000 | 0.925000 | 0.002100 | 0.839000 |
| 双模态残缺 | control | 90.000% | 0.009800 | 0.939000 | 0.012900 | 0.729000 |
| 双模态残缺 | v11g | 90.000% | 0.009600 | 0.940000 | 0.012500 | 0.733000 |
| 双模态残缺 | no_causal | 90.000% | 0.009600 | 0.940000 | 0.012400 | 0.737000 |

| 输入模式 | 实验 | 图像缺失区 MSE | 音频缺失区 MSE |
| --- | --- | ---: | ---: |
| 仅音频残缺 | control | 0.018900 | 0.033400 |
| 仅音频残缺 | v11g | 0.018700 | 0.033400 |
| 仅音频残缺 | no_causal | 0.018400 | 0.033400 |
| 仅图像残缺 | control | 0.077800 | 0.002000 |
| 仅图像残缺 | v11g | 0.077800 | 0.002000 |
| 仅图像残缺 | no_causal | 0.077800 | 0.002100 |
| 双模态残缺 | control | 0.063500 | 0.025900 |
| 双模态残缺 | v11g | 0.062200 | 0.025000 |
| 双模态残缺 | no_causal | 0.062400 | 0.024800 |

| 输入模式 | 实验 | 恢复均值 | 恢复标准差 | 恢复最大值 |
| --- | --- | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.039100 | 0.143900 | 1.000000 |
| 仅音频残缺 | v11g | 0.039100 | 0.143900 | 1.000000 |
| 仅音频残缺 | no_causal | 0.039100 | 0.143900 | 1.000000 |
| 仅图像残缺 | control | 0.018300 | 0.072800 | 1.000000 |
| 仅图像残缺 | v11g | 0.018700 | 0.072300 | 0.988200 |
| 仅图像残缺 | no_causal | 0.018500 | 0.073600 | 1.000000 |
| 双模态残缺 | control | 0.038900 | 0.139200 | 1.000000 |
| 双模态残缺 | v11g | 0.039200 | 0.138300 | 1.000000 |
| 双模态残缺 | no_causal | 0.039600 | 0.140400 | 1.000000 |

| 输入模式 | 实验 | 目标均值 | 目标标准差 | 目标最大值 | top15% 召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| 仅音频残缺 | control | 0.047400 | 0.168700 | 1.000000 | 73.700% |
| 仅音频残缺 | v11g | 0.047400 | 0.168700 | 1.000000 | 73.700% |
| 仅音频残缺 | no_causal | 0.047400 | 0.168700 | 1.000000 | 73.700% |
| 仅图像残缺 | control | 0.020700 | 0.084100 | 0.987900 | 80.800% |
| 仅图像残缺 | v11g | 0.020700 | 0.084100 | 0.987900 | 80.800% |
| 仅图像残缺 | no_causal | 0.020700 | 0.084100 | 0.987900 | 80.700% |
| 双模态残缺 | control | 0.047400 | 0.168700 | 1.000000 | 76.100% |
| 双模态残缺 | v11g | 0.047400 | 0.168700 | 1.000000 | 76.300% |
| 双模态残缺 | no_causal | 0.047400 | 0.168700 | 1.000000 | 76.200% |

</details>

**各实验结论与比较限制**

- **仅音频残缺**：主实验相对 control，音频缺失区 MSE 变化 +0.00%，Index ACC 变化 +0.000 个百分点。
- **双模态残缺**：主实验相对 control，音频缺失区 MSE 变化 -3.33%，Index ACC 变化 +0.000 个百分点。
- **control**：冻结 v11e_control，额外训练 0 轮；是父模型参考，不是与主实验等预算重训。
- **v11g**：额外训练 30 轮的 Masked Cross-Key + causal；Index ACC 与 control 相同符合冻结设计，不能当作分类提升。恢复有小幅改善，但需同时看 wrong/same-class 和 win_both。
- **no_causal**：相同父权重、相同 30 轮预算，不加 causal；部分恢复指标更好，不能仅凭主实验的 causal loss 宣称因果项全面有效。
- 内容 ACC 是冻结原网络对恢复结果的内部再分类，不是独立外部识别；same_class 的样本集合少 4 条，应读配对 same_damage，不能直接相减两个不同 n 的绝对均值。
- **勘误**：旧 v11g 摘要把 partial_temporal 音频 SSIM 写成约 0.60；独立音频 family 表实际为 **0.528776**，不是图像 SSIM。该表不与主 family-pair 评估混算。
- v11g 是 v11f 方案的版本入口迁移与复现，不是新架构消融；单次运行的小幅数值差异不构成结构性提升证据。
- 本节仅汇总已有结果，没有重跑训练或推理。单 seed、重复音频曝光和旧日志舍入限制仍在，不报告统计显著性。

[返回统一评估导航](#evaluation-format-20260912)

**本次版式整理验收**

- 覆盖 v11c–v11g 共 12 组实验；152 行已有 fixed/random 主结果的 760 个数值，
  已用独立提取方式逐项对照源 CSV/日志；缺失协议行不计入已有结果行数。
- normal、干预和独立音频 family 的实际指标字段均有对应表列；fixed/random、
  family-pair、音频 breakdown 与 n=10 demo 分开记录，没有重复合并。
- 已检查 Markdown 表格列数、91 处折叠区的标签配对、版本锚点、UTF-8 编码及
  历史日志保留情况；`git diff --check` 通过。
- 仅修改三份核心文档：`dev_log.md`、`implementation.md`、`user_requirements.md`。
  未改模型、配置、原始 CSV/日志/图片、checkpoint，未重跑模型、未提交或推送。

#### 历史运行说明（v11g，已归档）

本段是 v11g 历史运行说明，已归档；必须在项目根目录、激活对应 GPU 环境后执行。旧版本命令仍在各历史分支，不要使用当前 v12a 代码重跑旧权重来冒充原实验。

```bash
# 主实验、control 和 no_causal 的顺序训练与评估
nohup python -u scripts/run_v11g_suite.py --with_ablations > v11g_suite.log 2>&1 < /dev/null &
tail -f v11g_suite.log

# 仅评估现有 v11g 三组 checkpoint
python -u scripts/run_v11g_suite.py --eval_only --with_ablations
```

完整命令见 [实现指南](implementation.md#6-运行命令)。本段不属于当前 v12a 运行入口。

## 2026-09-13 v12a 实现记录

**改动类型：** 从 v11g checkpoint 启动的音频恢复修正；已完成代码、配置和 CPU 回归，尚未训练或评估。

**实现内容：**

- 音频 decoder 输出在 `audio_refiner.visible_paste_back=true` 时执行
  `M * prediction + (1-M) * cue`，因此可见区域直接保持输入值；缺失区域仍由 decoder
  和 Masked Cross-Key 预测。
- Audio Encoder 暴露第二层卷积脉冲率 `[T,B,64,16,16]`，转为 rate 后经零初始化
  `AudioLocalCueProjector` 投影到 `[B,16,64,64]`，加到 audio decoder feature。新分支
  初始权重为零，加载 v11g 权重时不会改变原 decoder 输出。
- v12a 只开放 `audio_decoder.*`、`audio_local_cue_projector.*`、
  `img_cross_adapter.*` 和 `aud_cross_adapter.*`；Encoder、Key、Index、Value、
  Classifier 及图像路径保持冻结。音频缺失区 weighted MSE 和时频梯度损失分别启用
  `1.0` 与 `0.5`。
- control 关闭局部 cue 并直接使用 v11g checkpoint；no-causal 与主实验共享 v12a
  结构和训练范围，仅关闭 causal loss。

**验证：**

- `python -m py_compile models/encoders.py models/decoders.py models/network.py scripts/train.py scripts/evaluate.py scripts/demo_inference.py scripts/mkdir_outputs.py scripts/run_v12a_suite.py scripts/smoke_test_v12a.py`：通过。
- `python scripts/smoke_test_v12a.py`：通过，验证输出 shape、音频可见区精确回填、局部 cue shape 和 trainable prefixes。
- 使用本地 v11g checkpoint 做 `strict=False` 兼容检查：仅缺少预期的四个
  `audio_local_cue_projector.*` 参数，无 unexpected keys；尚未执行 GPU 训练。

**运行说明：**

```bash
# 主实验、control 和 no-causal 顺序执行；去掉 --with_ablations 则不运行 no-causal
nohup python -u scripts/run_v12a_suite.py --with_ablations > v12a_suite.log 2>&1 < /dev/null &
tail -f v12a_suite.log

# 已有 checkpoint 时只评估三组
python -u scripts/run_v12a_suite.py --eval_only --with_ablations
```

本次没有上传 `_data/`、`outputs/`、日志、临时目录或 v11g 旧配置；训练和评估结果待
后续直接追加到本日志的 v12a 条目中。

<a id="evaluation-v12a"></a>

## 2026-09-13 v12a 评估归档

**评估性质：** 审阅本地已完成的服务器产物，未在当前 CPU 工作站重复跑完整套件。
三组实验均有 checkpoint、fixed/random normal、Cross-Key sweep、audio family breakdown
和 demo；每个 cue 模式有效 `n=10000`。训练预算为 control 0 轮、v12a 30 轮、
no-causal 30 轮，seed=`1234`，severity=`0.4`，`batch_size=128`。结果目标仍按
MNIST/FSDD 类别级绑定区分 sample/category，不能解释为精确录音配对恢复。

### fixed normal：8 个 cue 模式等权宏平均

MSE 越低越好，SSIM/ACC 越高越好；百分数已换算为 `%`。`content-aud` 是冻结原模型
对恢复音频的内部内容一致性，不是独立外部识别器。

| 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM | content-aud |
|---|---:|---:|---:|---:|---:|---:|
| control | 95.9905% | 0.00841228 | 0.944859 | 0.00380736 | 0.879315 | 90.3035% |
| v12a | 95.9905% | 0.00823786 | 0.947081 | 0.00276268 | 0.905477 | 92.9930% |
| no_causal | 95.9905% | 0.00822339 | 0.947435 | 0.00280953 | 0.904980 | 93.0490% |

相对 control，v12a 音频 MSE 下降 `0.00104468`（相对 `27.43%`），音频 SSIM 提升
`0.026162`，图像 MSE 下降 `0.00017442`；Index ACC 无变化，符合冻结 Index 的设计。
no-causal 的音频 MSE 比 v12a 高 `0.00004685`，SSIM 低 `0.000497`，但 content-aud
高 `0.056` 个百分点；这个差异只能作为同预算消融观察，不能单独证明 causal 项全面
优于无 causal。

### fixed normal：逐 cue 主指标

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
|---|---|---:|---:|---:|---:|---:|
| corrupt_img_only | control | 87.790% | 0.005307 | 0.967732 | 0.000917 | 0.882853 |
| corrupt_img_only | v12a | 87.790% | 0.005307 | 0.967732 | 0.000799 | 0.901195 |
| corrupt_img_only | no_causal | 87.790% | 0.005307 | 0.967732 | 0.000818 | 0.901830 |
| corrupt_aud_only | control | 96.770% | 0.003567 | 0.973989 | 0.003986 | 0.900188 |
| corrupt_aud_only | v12a | 96.770% | 0.003473 | 0.974960 | 0.000853 | 0.979303 |
| corrupt_aud_only | no_causal | 96.770% | 0.003512 | 0.974537 | 0.000863 | 0.979095 |
| corrupt_both | control | 98.610% | 0.005257 | 0.968044 | 0.003972 | 0.900263 |
| corrupt_both | v12a | 98.610% | 0.004948 | 0.969866 | 0.000788 | 0.980112 |
| corrupt_both | no_causal | 98.610% | 0.004990 | 0.969639 | 0.000813 | 0.979637 |
| clean_img_corrupt_aud | control | 99.180% | 0.009738 | 0.942714 | 0.003975 | 0.900494 |
| clean_img_corrupt_aud | v12a | 99.180% | 0.009737 | 0.942714 | 0.000805 | 0.980240 |
| clean_img_corrupt_aud | no_causal | 99.180% | 0.009737 | 0.942714 | 0.000832 | 0.979730 |
| corrupt_img_clean_aud | control | 99.090% | 0.005243 | 0.968180 | 0.003650 | 0.911121 |
| corrupt_img_clean_aud | v12a | 99.090% | 0.004921 | 0.970062 | 0.003502 | 0.911499 |
| corrupt_img_clean_aud | no_causal | 99.090% | 0.004962 | 0.969804 | 0.003539 | 0.910949 |
| clean_img_only | control | 97.700% | 0.009777 | 0.942646 | 0.000289 | 0.965964 |
| clean_img_only | v12a | 97.700% | 0.009777 | 0.942646 | 0.000240 | 0.973681 |
| clean_img_only | no_causal | 97.700% | 0.009777 | 0.942646 | 0.000246 | 0.973081 |
| clean_aud_only | control | 98.830% | 0.002085 | 0.985595 | 0.003648 | 0.911157 |
| clean_aud_only | v12a | 98.830% | 0.002039 | 0.986057 | 0.003497 | 0.911541 |
| clean_aud_only | no_causal | 98.830% | 0.002072 | 0.985700 | 0.003535 | 0.911013 |
| clean_both | control | 99.180% | 0.009732 | 0.942685 | 0.003643 | 0.911253 |
| clean_both | v12a | 99.180% | 0.009732 | 0.942685 | 0.003503 | 0.911523 |
| clean_both | no_causal | 99.180% | 0.009732 | 0.942685 | 0.003539 | 0.910961 |

### random normal：8 个 cue 模式等权宏平均

| 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM | content-aud |
|---|---:|---:|---:|---:|---:|---:|
| control | 96.1575% | 0.00829207 | 0.945894 | 0.00371376 | 0.883882 | 90.6925% |
| v12a | 96.1575% | 0.00812653 | 0.947936 | 0.00267065 | 0.910193 | 93.3300% |
| no_causal | 96.1575% | 0.00811582 | 0.948236 | 0.00271537 | 0.909680 | 93.4113% |

### 区域、Cross-Key、family 与 demo

- v12a 的 fixed normal `aud_visible_mse` 在存在音频 cue 的模式为 `0.0`，说明可见区
  回填生效；音频 `aud_masked_mse` 仍是缺失区恢复质量的主要判断指标，不能用全图 MSE
  代替。逐模式的缺失区/可见区、PSNR、content classification、pair L2/Recall 和
  `pix_var` 已保存在三组对应的 normal CSV 中。
- v12a Cross-Key sweep 的 `img->aud` 与 `aud->img` 记录了 normal/zero/wrong/
  same-class 绝对 MSE、gain/damage、gate、ratio、win_zero/win_wrong/win_both 和 n；
  从 fixed 日志看 v12a 的 `img->aud` gain 约 `0.0000~0.0002`，说明当前 Cross-Key
  仍是弱增益，不能仅凭 gate 非零宣布跨模态恢复成功。
- 三组均有五类音频 family breakdown 和 fixed/random demo 图；其中
  `partial_temporal` 是音频恢复最弱的 family，不能被其它四类均值掩盖。

证据目录：`v12a_outputs/outputs/outputs_v12a/`、
`v12a_outputs/outputs/outputs_v12a_control/`、
`v12a_outputs/outputs/outputs_v12a_no_causal/`；本次没有修改原始 CSV、日志或图片。

## 历史运行说明（v12a，2026-09-13）

本历史条目记录 v12a 当时的运行方式。训练和评估必须在项目根目录、已激活环境和可用 GPU 中执行；
本机本次仅审阅已有服务器产物，没有继续使用 CPU 重跑完整套件。

```bash
# 训练主实验；加 --with_ablations 才顺序训练 no-causal
nohup python -u scripts/run_v12a_suite.py --with_ablations > v12a_suite.log 2>&1 < /dev/null &
tail -f v12a_suite.log

# 已有三组 checkpoint 时只评估
python -u scripts/run_v12a_suite.py --eval_only --with_ablations
```

当前输出目录为 `outputs/outputs_v12a/`、`outputs/outputs_v12a_control/` 和
`outputs/outputs_v12a_no_causal/`；本地已审阅产物位于 `v12a_outputs/outputs/`。

## 2026-09-14 v12b 实现记录

**改动类型**：在 v12a 音频恢复路径上增加多尺度局部 cue 和中间尺度 Cross-Key；不改变
MNIST/FSDD 类别级绑定，不把恢复梯度经 Value 传回 Index。

**实现内容**：

- Audio Encoder 暴露 `conv1`、`conv2` 局部脉冲序列，经过 rate 后由带 mask 的零初始化
  projector 注入 Audio Decoder 的 `16/32/64` 三个特征尺度；旧的单尺度 projector 保留为
  v12a 兼容 fallback。
- 在音频 decoder 的 `32x32` 中间特征增加零初始化 `aud_cross_adapter_mid`，与最终
  Cross-Key 一样只调制音频缺失区，不写回 Value 或 Index。
- 时频梯度损失的相邻差分 mask 改为两端并集，补足缺失区域边界的对称监督。
- 新增 `configs/v12b.yaml`、`v12b_control.yaml`、`v12b_no_causal.yaml`、
  `scripts/run_v12b_suite.py` 和 `scripts/smoke_test_v12b.py`。

**训练协议**：从 `outputs/checkpoints/cross_modal_snn_v12a.pt` 启动，父权重 SHA256 为
`0a49aaf709ed257d7a6361ec5c09dcdd7b3e01ce90548070a3ff178a5402290e`；主实验和
`no_causal` 先额外训练 30 轮，control 为同一 v12a 权重的冻结评估；保持
`batch_size=128`、五类 audio family 均衡采样和 `severity=0.4`。

**验证**：已通过 Python 语法检查、v12b CPU shape/pasteback/local-cue/Cross-Key smoke
检查和 `git diff --check`。本次未上传数据、outputs 或 checkpoint；训练与全量评估待在
GPU 实例执行，结果按统一 fixed/random、逐 family、Cross-Key、训练统计和 demo 指标
追加到本文件，不另建独立结果文件。

## 2026-09-14 v12b 初始运行记录

当前活动版本为 `v12b`。必须在项目根目录、已激活环境和可用 GPU 中运行；父 checkpoint
需先存在于 `outputs/checkpoints/cross_modal_snn_v12a.pt`。

```bash
# 主实验、control，并可选顺序训练 no-causal
nohup python -u scripts/run_v12b_suite.py --with_ablations > v12b_suite.log 2>&1 < /dev/null &
tail -f v12b_suite.log

# 三组已有 checkpoint 时只评估
python -u scripts/run_v12b_suite.py --eval_only --with_ablations
```

输出目录为 `outputs/outputs_v12b/`、`outputs/outputs_v12b_control/` 和
`outputs/outputs_v12b_no_causal/`；本段为首次实现时的运行记录，后续完整结果见
[v12b 评估归档](#evaluation-v12b)。

### 2026-09-14 05:56 control 评估兼容修复

**触发与诊断**：用户服务器在 `v12b_control.yaml` 的 fixed normal 评估中，完成
`corrupt_img_only` 后进入 `corrupt_aud_only`，于 `models/network.py` 报
`AttributeError: 'dict' object has no attribute 'detach'`。本地复现相同异常。
原因是 v12b encoder 返回 conv1/conv2 字典，而 control 的单尺度 projector 仍按
tensor 调用 detach；此前只检查主模型前向和 control 参数加载，遗漏 control 实际前向。

**修改与文档同步**：先更新 `implementation.md` 的 cue 接口和续跑说明，再修改：

- `models/network.py`：单尺度路径从字典取 conv2 后 detach，兼容原 tensor 输入。
  checkpoint 的参数名称、shape 及主实验多尺度计算保持原样；本修复无需重训主模型。
- `scripts/run_v12b_suite.py`：新增 `--start_from main/control/no_causal`，从指定实验
  开始构建和检查任务。默认流程保持不变，选择 no_causal 必须启用 `--with_ablations`。
- `scripts/smoke_test_v12b.py`：增加三组配置各八种 cue 前向、control 新旧输入等价性
  和 suite 任务选择回归；control 使用非零 projector 权重，避免零输出掩盖问题。

**验证**：三组共 24 个 cue 场景通过；control 字典路径与旧 tensor 路径逐元素一致；
本地真实 v12a checkpoint 严格加载后，八种 cue 及新旧路径等价性检查均通过。
Python 语法检查、suite dry-run 和 diff 检查通过。本次为代码回归检查，未在本地重跑
全测试集评估，不能将这些测试记作模型指标。用户截图确认 control 评估中断；若按默认
顺序运行，主实验训练/评估已先执行，后续 no-causal 尚未启动。

#### 运行说明

服务器拉取 v12b 修复后，在项目根目录继续 control 评估，然后训练/评估 no-causal：

```bash
git pull --ff-only origin v12b
nohup python -u scripts/run_v12b_suite.py --start_from control --with_ablations > v12b_suite_continue.log 2>&1 < /dev/null &
tail -f v12b_suite_continue.log
```

该命令不会重跑 main，control 的失败评估日志会重新生成；使用新的总日志保存续跑输出。
如果 no-causal checkpoint 也已训练完成，则添加 `--eval_only`；部分训练则使用
`--resume`。两者都要求所选的训练实验已有对应 checkpoint。

## 2026-09-14 评估归档

<a id="evaluation-v12b"></a>

### v12b

本条审阅并重新汇总本地服务器产物 `v12b_outputs/outputs/`，校验 checkpoint 和配置；
没有在当前 CPU 工作站重跑全量推理。主实验、control、no-causal 的 fixed/random normal、
Cross-Key sweep、五类音频专项和 fixed/random demo 均已完成；未修改原始产物。

#### 实验与验收结论

v12b 提高了干净音频重建与局部残缺恢复，但最初优先验收的 `partial_temporal` 长时间缺失未获实质改善。
三组 Index ACC 逐项相同，且 checkpoint 冻结参数逐张量一致；这验证了冻结约束，没有证明 Index 的困难样本识别得到改善。

| 实验 | 父权重 | 额外训练 | seed / batch size | 变化与结论 |
|---|---|---|---|---|
| control | v12a | 0 轮，固定父模型 | 1234 / 128 | 复现原 v12a；不是等预算重训；父模型保留已有 Cross-Key |
| v12b | 同一 v12a | 30 轮，epoch 0-29 | 1234 / 128 | 多尺度局部 cue、中间 Cross-Key、对称边界损失；因果权重 0.5；整体恢复改善、长缺失目标未达成 |
| no_causal | 同一 v12a | 30 轮，epoch 0-29 | 1234 / 128 | 相同新增结构，因果项关闭；音频误差略高，部分内容识别/图像指标略优；长缺失存在退化 |

两组训练的有效配置除输出路径、版本名和因果项开关/权重外一致。真实可训练范围为 audio decoder、
local cue projector 和双向 Cross-Key adapters；Encoder、Key、Index、Value、Classifier 未变化。
旧单尺度 audio_local_cue_projector 在多尺度启用时不走前向，权重也未改变。

共同 severity=0.4，数值 random seed=4321，固定评估 seed=1234。每个 cue/family 的主指标 n=10000；
fixed 在五组 family 上等权宏平均，再按需要对八个 cue 等权平均。重复 family 曝光数 50000 不是独立样本数；
MNIST 测试图像与 FSDD 测试录音按类别配对，音频被复用，未保存独立录音计数或多种子结果，不生成显著性结论。
random 数值评估为一次可复现的混合 family 抽样；音频专项使用独立 seed 和固定图像 occlusion，不与主 fixed 表混算。

图像-only 的 target 为 sample/category，音频-only 为 category/sample，双模态为 sample/sample；
category 为训练集 medoid。八 cue 总均值仅为兼容历史的汇总，同时单列三种残缺音频模式，避免目标混合掩盖困难任务。

MSE/L1 越低越好，ACC/SSIM/PSNR 越高越好。content ACC 是冻结原模型的内部再分类，未使用独立外部识别器。
gate、ratio、多样性量不具有统一优劣方向；win 指样本占比；N/A 不计为 0。

#### 分类与恢复：协议汇总

| 协议 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM | 图内容 ACC | 音内容 ACC |
|---|---|---|---|---|---|---|---|---|
| fixed_mask | control | 95.9905% | 0.008237862572 | 0.9470808497 | 0.002762682049 | 0.9054769648 | 95.1715% | 92.9930% |
| fixed_mask | v12b | 95.9905% | 0.008177996649 | 0.9477534371 | 0.001770412991 | 0.9301376986 | 95.2083% | 93.4813% |
| fixed_mask | no_causal | 95.9905% | 0.008167375479 | 0.9480038999 | 0.001803554336 | 0.9299717102 | 95.1053% | 93.4590% |
| legacy_random | control | 96.1575% | 0.008126525406 | 0.9479357083 | 0.002670646783 | 0.9101925147 | 95.3062% | 93.3300% |
| legacy_random | v12b | 96.1575% | 0.008070809757 | 0.9485515076 | 0.001670994745 | 0.935057421 | 95.3563% | 93.7900% |
| legacy_random | no_causal | 96.1575% | 0.008063437424 | 0.9487602548 | 0.001701495033 | 0.934885805 | 95.2550% | 93.7500% |

| 协议 | 残缺音频三模式均值 | 音频 MSE | 缺失区 MSE | 音频 SSIM |
|---|---|---|---|---|
| fixed_mask | control | 0.003591249455 | 0.009437425678 | 0.8683462321 |
| fixed_mask | v12b | 0.003438546596 | 0.009023196225 | 0.8726330751 |
| fixed_mask | no_causal | 0.003519287197 | 0.009232421211 | 0.8722765453 |
| legacy_random | control | 0.003358097716 | 0.008836402046 | 0.8794631724 |
| legacy_random | v12b | 0.003185777469 | 0.008376554488 | 0.8843348426 |
| legacy_random | no_causal | 0.003261292418 | 0.008572075456 | 0.8839244832 |

v12b 八 cue 音频 MSE 相对 control：fixed 下降 35.917%，random 下降 37.431%。
fixed 绝对下降中 93.665% 来自三种音频干净模式，单个模式下降约 70.8%。
音频干净时 aud_mask=None，现有回填逻辑不触发，因而这些改善代表 decoder 重建干净 cue 的能力。
三种残缺音频模式的 fixed 全图 MSE 仅下降约 4.25%，缺失区 MSE 下降约 4.39%；不能把 35.9% 宣称为补洞改善。
主模型与 no-causal 的 fixed 八 cue 音频 MSE 差为约 0.00003314135，SSIM 差约 0.000165989；
主模型音频内容 ACC 高 0.02225 个百分点，但干净图像+残缺音频及双残缺时 no-causal 内容 ACC 略优。

#### 分类与恢复：fixed 五类等权宏平均

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
|---|---|---|---|---|---|---|
| 干净双模态 | control | 99.1800% | 0.009732332389 | 0.9426845028 | 0.003503285427 | 0.9115226718 |
| 干净双模态 | v12b | 99.1800% | 0.009732332389 | 0.9426845028 | 0.001023170835 | 0.9706698141 |
| 干净双模态 | no_causal | 99.1800% | 0.009732332389 | 0.9426845028 | 0.001022163968 | 0.9707627995 |
| 仅干净图像 | control | 97.7000% | 0.009777385685 | 0.9426458778 | 0.0002402440196 | 0.9736808665 |
| 仅干净图像 | v12b | 97.7000% | 0.009777385685 | 0.9426458778 | 0.0002218772086 | 0.9763999848 |
| 仅干净图像 | no_causal | 97.7000% | 0.009777385685 | 0.9426458778 | 0.0002282006215 | 0.976126532 |
| 仅干净音频 | control | 98.8300% | 0.002039422297 | 0.9860573347 | 0.003496576117 | 0.9115410681 |
| 仅干净音频 | v12b | 98.8300% | 0.002025680222 | 0.9861481328 | 0.001021110903 | 0.9707879167 |
| 仅干净音频 | no_causal | 98.8300% | 0.002050657357 | 0.9859331862 | 0.001020037457 | 0.9709046022 |
| 仅残缺图像 | control | 92.0280% | 0.007680466978 | 0.955151473 | 0.0005867628023 | 0.9304794941 |
| 仅残缺图像 | v12b | 92.0280% | 0.007680466978 | 0.955151473 | 0.00056038177 | 0.9346278995 |
| 仅残缺图像 | no_causal | 92.0280% | 0.007680466978 | 0.955151473 | 0.0005802758713 | 0.9343264542 |
| 仅残缺音频 | control | 84.5720% | 0.01238851457 | 0.892031235 | 0.003788456834 | 0.8625069361 |
| 仅残缺音频 | v12b | 84.5720% | 0.0121429631 | 0.8959478379 | 0.003619976872 | 0.8670313654 |
| 仅残缺音频 | no_causal | 84.5720% | 0.01196811502 | 0.8984157582 | 0.003657144785 | 0.8666763434 |
| 干净图像+残缺音频 | control | 98.9340% | 0.009790254229 | 0.9425965783 | 0.003439189032 | 0.8728393103 |
| 干净图像+残缺音频 | v12b | 98.9340% | 0.009790254229 | 0.9425965783 | 0.003302744176 | 0.8768458413 |
| 干净图像+残缺音频 | no_causal | 98.9340% | 0.009790254229 | 0.9425965783 | 0.003409866151 | 0.8764936821 |
| 残缺图像+干净音频 | control | 99.1440% | 0.007218561702 | 0.9579087732 | 0.003500839663 | 0.9115529218 |
| 残缺图像+干净音频 | v12b | 99.1440% | 0.007103987851 | 0.9586285542 | 0.001021123422 | 0.9707167486 |
| 残缺图像+干净音频 | no_causal | 99.1440% | 0.007149694622 | 0.95843637 | 0.001019895179 | 0.9708236575 |
| 双残缺 | control | 97.5360% | 0.007275962728 | 0.9575710225 | 0.0035461025 | 0.8696924499 |
| 双残缺 | v12b | 97.5360% | 0.007170902732 | 0.9582245398 | 0.003392918741 | 0.8740220185 |
| 双残缺 | no_causal | 97.5360% | 0.007190097554 | 0.9581674529 | 0.003490850654 | 0.8736596105 |

#### 分类与恢复：random 单次评估

| 输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM |
|---|---|---|---|---|---|---|
| 干净双模态 | control | 99.1800% | 0.009732332389 | 0.9426845028 | 0.003503285427 | 0.9115226718 |
| 干净双模态 | v12b | 99.1800% | 0.009732332389 | 0.9426845028 | 0.001023170835 | 0.9706698141 |
| 干净双模态 | no_causal | 99.1800% | 0.009732332389 | 0.9426845028 | 0.001022163968 | 0.9707627995 |
| 仅干净图像 | control | 97.7000% | 0.009777385685 | 0.9426458778 | 0.0002402440196 | 0.9736808665 |
| 仅干净图像 | v12b | 97.7000% | 0.009777385685 | 0.9426458778 | 0.0002218772086 | 0.9763999848 |
| 仅干净图像 | no_causal | 97.7000% | 0.009777385685 | 0.9426458778 | 0.0002282006215 | 0.976126532 |
| 仅干净音频 | control | 98.8300% | 0.002039422297 | 0.9860573347 | 0.003496576117 | 0.9115410681 |
| 仅干净音频 | v12b | 98.8300% | 0.002025680222 | 0.9861481328 | 0.001021110903 | 0.9707879167 |
| 仅干净音频 | no_causal | 98.8300% | 0.002050657357 | 0.9859331862 | 0.001020037457 | 0.9709046022 |
| 仅残缺图像 | control | 92.3600% | 0.007614292219 | 0.9552188177 | 0.0005491464123 | 0.934874578 |
| 仅残缺图像 | v12b | 92.3600% | 0.007614292219 | 0.9552188177 | 0.0005232986961 | 0.938891894 |
| 仅残缺图像 | no_causal | 92.3600% | 0.007614292219 | 0.9552188177 | 0.0005377318716 | 0.9387075771 |
| 仅残缺音频 | control | 85.7900% | 0.01148949463 | 0.9000236948 | 0.003578324331 | 0.8705906674 |
| 仅残缺音频 | v12b | 85.7900% | 0.01127184049 | 0.9035166462 | 0.003382371552 | 0.8760217186 |
| 仅残缺音频 | no_causal | 85.7900% | 0.01112184997 | 0.9056596614 | 0.003418815638 | 0.8755758115 |
| 干净图像+残缺音频 | control | 98.9600% | 0.009783406097 | 0.9426022731 | 0.003252138307 | 0.8807206806 |
| 干净图像+残缺音频 | v12b | 98.9600% | 0.009783406097 | 0.9426022731 | 0.003089098994 | 0.8855609869 |
| 干净图像+残缺音频 | no_causal | 98.9600% | 0.009783406097 | 0.9426022731 | 0.00319140949 | 0.8850794596 |
| 残缺图像+干净音频 | control | 99.1600% | 0.007102384464 | 0.958277329 | 0.003501629138 | 0.9115314157 |
| 残缺图像+干净音频 | v12b | 99.1600% | 0.006989629062 | 0.9589780867 | 0.001021167909 | 0.9707052304 |
| 残缺图像+干净音频 | no_causal | 99.1600% | 0.007024656107 | 0.958841262 | 0.001019949091 | 0.9708114794 |
| 双残缺 | control | 97.2800% | 0.007473485468 | 0.9559758368 | 0.003243830509 | 0.8870781691 |
| 双残缺 | v12b | 97.2800% | 0.007371911891 | 0.9566177238 | 0.003085861862 | 0.8914218223 |
| 双残缺 | no_causal | 97.2800% | 0.007402919574 | 0.956496457 | 0.003173652124 | 0.8911181785 |

#### 部分残缺与 Cross-Key 判断

fixed 的 img->aud 正确 Key 相对 zero：干净图像+残缺音频增益 0.0001603091（约 1.812%），
双残缺增益 0.0001138269（约 1.264%）。二者小于父模型对应 0.0002762809、0.0002022408。
v12b 的绝对恢复更好，但相对自身 zero 的 Cross-Key 边际收益没有超过父模型。
同时正确 Key 优于 wrong 的样本率仅 44.474% / 43.800%，同时优于 zero 与 wrong 为 34.930% / 33.322%；
正确/错误 Key 的平均差仅约 6.48e-6 / 2.91e-6，类别选择性仍弱。
no-causal 在双残缺的 fixed img->aud gain=-1.368e-5，random=-2.151e-6；因果项有助于维持正向边际收益，
但缺少无新结构等预算续训及独立中间 adapter 消融，不能独立归因新增尺度。
音频全缺失时只检索类别原型：clean_img_only 的主模型正确 Key 相对 zero 改善约 10.97%，
win_wrong=86.06%、win_both=82.56%，好于部分音频缺失的选择性；这是类别原型任务，不是特定录音恢复。
aud->img 在干净音频+残缺图像的 gain 由 0.0009556563 增至 0.0012966759，图像补洞有小幅改善。

Cross-Key sweep 同时干预 decoder 条件，不改变 Index/Value。same-class 绝对 MSE 的有效集合可能比 normal 少，
same_damage 为同集合配对差，不能总用两个总体均值直接相减替代。CSV gate/ratio 未单独输出中间尺度字段。

<details>
<summary>fixed_mask：关键 Cross-Key 配对对照</summary>

| 输入模式 | 方向 | 实验 | normal MSE | zero MSE | wrong MSE | same MSE | gain | win_zero | win_wrong | win_both |
|---|---|---|---|---|---|---|---|---|---|---|
| 干净图像+残缺音频 | img2aud | control | 0.009055382171 | 0.009331663075 | 0.009053783996 | 0.009055011668 | 0.0002762808874 | 68.0400% | 43.5740% | 36.0320% |
| 干净图像+残缺音频 | img2aud | v12b | 0.008685033526 | 0.008845342677 | 0.008691518495 | 0.008684787714 | 0.0001603091368 | 63.9340% | 44.4740% | 34.9300% |
| 干净图像+残缺音频 | img2aud | no_causal | 0.008962544561 | 0.008966815091 | 0.008972579324 | 0.008961976089 | 4.270494836e-06 | 56.0680% | 44.0560% | 31.2380% |
| 双残缺 | img2aud | control | 0.009306500793 | 0.009508741612 | 0.009304240097 | 0.009307047037 | 0.0002022408083 | 67.0560% | 42.8340% | 34.7220% |
| 双残缺 | img2aud | v12b | 0.008892291977 | 0.009006118852 | 0.008895199425 | 0.008892722659 | 0.000113826913 | 62.7020% | 43.8000% | 33.3220% |
| 双残缺 | img2aud | no_causal | 0.009145992404 | 0.009132311989 | 0.00915015201 | 0.009146301944 | -1.368040765e-05 | 55.3260% | 43.3120% | 29.6720% |
| 仅干净图像 | img2aud | control | 0.0002402440197 | 0.0002614530888 | 0.0002473042549 | 0.0002405052923 | 2.120906993e-05 | 95.0400% | 82.8300% | 79.8300% |
| 仅干净图像 | img2aud | v12b | 0.00022187721 | 0.0002492204076 | 0.0002297595468 | 0.0002228766707 | 2.734319982e-05 | 94.6400% | 86.0600% | 82.5600% |
| 仅干净图像 | img2aud | no_causal | 0.0002282006204 | 0.0002572875584 | 0.0002388485523 | 0.0002300969015 | 2.90869362e-05 | 95.0200% | 88.3200% | 85.4500% |
| 残缺图像+干净音频 | aud2img | control | 0.02496237731 | 0.02591803356 | 0.02501689519 | 0.02496240177 | 0.0009556563124 | 73.8880% | 53.0020% | 41.4380% |
| 残缺图像+干净音频 | aud2img | v12b | 0.02462135759 | 0.02591803356 | 0.02468080577 | 0.02462020766 | 0.001296675922 | 77.3480% | 53.0060% | 43.1480% |
| 残缺图像+干净音频 | aud2img | no_causal | 0.02476057522 | 0.02591803356 | 0.02484446874 | 0.0247615172 | 0.001157458343 | 73.3260% | 53.3260% | 41.7300% |
| 双残缺 | aud2img | control | 0.0250583435 | 0.02591244846 | 0.02509570633 | 0.0250443522 | 0.0008541049949 | 71.7880% | 50.2320% | 38.9580% |
| 双残缺 | aud2img | v12b | 0.02474838701 | 0.02591244846 | 0.02479509829 | 0.02473611551 | 0.001164061437 | 74.6980% | 50.7020% | 40.6140% |
| 双残缺 | aud2img | no_causal | 0.02480292383 | 0.02591244846 | 0.02487365013 | 0.02479095154 | 0.001109524558 | 72.7120% | 51.0080% | 40.2520% |

</details>

<details>
<summary>legacy_random：关键 Cross-Key 配对对照</summary>

| 输入模式 | 方向 | 实验 | normal MSE | zero MSE | wrong MSE | same MSE | gain | win_zero | win_wrong | win_both |
|---|---|---|---|---|---|---|---|---|---|---|
| 干净图像+残缺音频 | img2aud | control | 0.008538770679 | 0.008829892039 | 0.008537004831 | 0.008532503409 | 0.0002911213193 | 70.0600% | 43.5900% | 36.1900% |
| 干净图像+残缺音频 | img2aud | v12b | 0.008107095659 | 0.008288860309 | 0.008114551979 | 0.008101494878 | 0.0001817647181 | 66.2600% | 45.5300% | 36.1000% |
| 干净图像+残缺音频 | img2aud | no_causal | 0.008371060044 | 0.008403754005 | 0.00838248594 | 0.008364864397 | 3.269380778e-05 | 58.7100% | 44.9500% | 32.3800% |
| 双残缺 | img2aud | control | 0.008598824614 | 0.008793782392 | 0.008598744443 | 0.008591988844 | 0.0001949578249 | 66.0000% | 42.6500% | 35.0900% |
| 双残缺 | img2aud | v12b | 0.00817188829 | 0.008286556166 | 0.008176063168 | 0.008166542198 | 0.0001146678842 | 61.6800% | 43.2300% | 33.3200% |
| 双残缺 | img2aud | no_causal | 0.008401221268 | 0.008399070302 | 0.008407805932 | 0.008394838782 | -2.151020546e-06 | 55.0400% | 42.9600% | 30.1400% |
| 仅干净图像 | img2aud | control | 0.0002402440197 | 0.0002614530888 | 0.0002473042549 | 0.0002405052923 | 2.120906993e-05 | 95.0400% | 82.8300% | 79.8300% |
| 仅干净图像 | img2aud | v12b | 0.00022187721 | 0.0002492204076 | 0.0002297595468 | 0.0002228766707 | 2.734319982e-05 | 94.6400% | 86.0600% | 82.5600% |
| 仅干净图像 | img2aud | no_causal | 0.0002282006204 | 0.0002572875584 | 0.0002388485523 | 0.0002300969015 | 2.90869362e-05 | 95.0200% | 88.3200% | 85.4500% |
| 残缺图像+干净音频 | aud2img | control | 0.02710606877 | 0.02826212105 | 0.02712971728 | 0.02708293691 | 0.001156052208 | 75.9700% | 52.2100% | 42.1300% |
| 残缺图像+干净音频 | aud2img | v12b | 0.02675469481 | 0.02826212105 | 0.02678845391 | 0.02673262496 | 0.001507425934 | 78.8300% | 52.3400% | 43.5500% |
| 残缺图像+干净音频 | aud2img | no_causal | 0.02686068637 | 0.02826212105 | 0.02691241012 | 0.02684275328 | 0.001401434613 | 74.6300% | 52.5800% | 41.9400% |
| 双残缺 | aud2img | control | 0.02532390426 | 0.02618421544 | 0.02535640543 | 0.02528559596 | 0.0008603106864 | 70.6500% | 50.2800% | 38.5600% |
| 双残缺 | aud2img | v12b | 0.0250211262 | 0.02618421544 | 0.02506388687 | 0.02498310283 | 0.001163088947 | 74.4200% | 50.6700% | 40.5500% |
| 双残缺 | aud2img | no_causal | 0.02510097159 | 0.02618421544 | 0.02516194481 | 0.02506784399 | 0.001083243625 | 70.9300% | 50.9600% | 39.1600% |

</details>

#### 音频 family 专项与长缺失

以下专项覆盖三种音频残缺 cue、五种 family；每次完整遍历测试集，另用独立 mask seed。
partial_temporal 遮挡有效语音尾段，不能与只遮到静音的 time_mask 等同。
主模型 partial_temporal 缺失区 MSE：音频-only 下降约 0.284%，干净图像+残缺音频上升约 0.462%，
双残缺上升约 0.169%；后两者 SSIM 也略低，优先验收目标未通过。
no-causal 同项相对 control 分别恶化约 0.733%、3.942%、3.350%。因果项抑制了部分退化，但并未解决长缺失。
feature_dropout 的主模型缺失区 MSE 改善约 36%-40%，time_mask/freq_mask 多为约 3%-5%，
支持当前模型更擅长利用局部保留信息补小范围空缺的解释，不能直接证明具体模块的因果贡献。

<details>
<summary>音频专项：time_mask</summary>

| 输入模式 | 实验 | Index ACC | 音频 MSE | 音频 SSIM | 缺失区 MSE | 缺失区 L1 | top15% 召回 |
|---|---|---|---|---|---|---|---|
| 仅残缺音频 | control | 90.3900% | 0.003279069786 | 0.9098891888 | 0.00807155631 | 0.02359827327 | 89.0814% |
| 仅残缺音频 | v12b | 90.3900% | 0.003157877918 | 0.9131757013 | 0.007773238017 | 0.02309024938 | 89.0921% |
| 仅残缺音频 | no_causal | 90.3900% | 0.003187179682 | 0.9134275942 | 0.007845365428 | 0.02301557826 | 89.2550% |
| 干净图像+残缺音频 | control | 99.0300% | 0.003023781309 | 0.9156962244 | 0.007443153918 | 0.02267431073 | 89.2857% |
| 干净图像+残缺音频 | v12b | 99.0300% | 0.002930759226 | 0.9188755859 | 0.007214176594 | 0.02211288065 | 89.2537% |
| 干净图像+残缺音频 | no_causal | 99.0300% | 0.003016473691 | 0.9189032138 | 0.007425165942 | 0.02196166251 | 89.6610% |
| 双残缺 | control | 97.7100% | 0.003124714765 | 0.9137972696 | 0.007691605515 | 0.02323797907 | 89.0935% |
| 双残缺 | v12b | 97.7100% | 0.003026747897 | 0.9172232037 | 0.007450456326 | 0.02265567213 | 89.0651% |
| 双残缺 | no_causal | 97.7100% | 0.003105988445 | 0.9173071476 | 0.007645509886 | 0.02251297696 | 89.4223% |

</details>

<details>
<summary>音频专项：freq_mask</summary>

| 输入模式 | 实验 | Index ACC | 音频 MSE | 音频 SSIM | 缺失区 MSE | 缺失区 L1 | top15% 召回 |
|---|---|---|---|---|---|---|---|
| 仅残缺音频 | control | 92.2200% | 0.002509089853 | 0.9234900279 | 0.006176221218 | 0.02116418124 | 88.5553% |
| 仅残缺音频 | v12b | 92.2200% | 0.002391924905 | 0.9269546106 | 0.005887815131 | 0.0205923305 | 88.7232% |
| 仅残缺音频 | no_causal | 92.2200% | 0.002420955343 | 0.9261392849 | 0.005959274631 | 0.02065475956 | 88.8311% |
| 干净图像+残缺音频 | control | 98.8400% | 0.002372239015 | 0.9275494032 | 0.005839357587 | 0.02074631183 | 88.5968% |
| 干净图像+残缺音频 | v12b | 98.8400% | 0.002293973531 | 0.9304506823 | 0.005646704089 | 0.02014290541 | 88.6947% |
| 干净图像+残缺音频 | no_causal | 98.8400% | 0.002359590735 | 0.928885383 | 0.005808223369 | 0.02013378111 | 89.1860% |
| 双残缺 | control | 97.9100% | 0.002420372422 | 0.9252812969 | 0.00595783983 | 0.02091210476 | 88.5190% |
| 双残缺 | v12b | 97.9100% | 0.002333033354 | 0.9285822127 | 0.005742851369 | 0.02029994751 | 88.6610% |
| 双残缺 | no_causal | 97.9100% | 0.002396090774 | 0.9269600659 | 0.005898069561 | 0.02031819614 | 89.0700% |

</details>

<details>
<summary>音频专项：feature_dropout</summary>

| 输入模式 | 实验 | Index ACC | 音频 MSE | 音频 SSIM | 缺失区 MSE | 缺失区 L1 | top15% 召回 |
|---|---|---|---|---|---|---|---|
| 仅残缺音频 | control | 97.1000% | 0.001481112938 | 0.9601202813 | 0.003700818744 | 0.01584962049 | 90.7819% |
| 仅残缺音频 | v12b | 97.1000% | 0.0009486985296 | 0.9736976358 | 0.002370283177 | 0.01280620803 | 91.7119% |
| 仅残缺音频 | no_causal | 97.1000% | 0.0009572662797 | 0.9734175484 | 0.00239165116 | 0.01287264701 | 91.7030% |
| 干净图像+残缺音频 | control | 99.3200% | 0.001314147486 | 0.9637205889 | 0.00328360173 | 0.0150761658 | 90.8990% |
| 干净图像+残缺音频 | v12b | 99.3200% | 0.0007899013032 | 0.9774551545 | 0.001973526457 | 0.01180962684 | 91.9442% |
| 干净图像+残缺音频 | no_causal | 99.3200% | 0.0008179364141 | 0.9766598041 | 0.002043541364 | 0.01194976779 | 91.9087% |
| 双残缺 | control | 99.0300% | 0.001340888193 | 0.963340916 | 0.003351513963 | 0.01519876381 | 90.8771% |
| 双残缺 | v12b | 99.0300% | 0.0008129757855 | 0.9770864037 | 0.002031798462 | 0.01194314387 | 91.9232% |
| 双残缺 | no_causal | 99.0300% | 0.0008377781662 | 0.9763924413 | 0.002093800471 | 0.01207448888 | 91.8781% |

</details>

<details>
<summary>音频专项：partial_temporal</summary>

| 输入模式 | 实验 | Index ACC | 音频 MSE | 音频 SSIM | 缺失区 MSE | 缺失区 L1 | top15% 召回 |
|---|---|---|---|---|---|---|---|
| 仅残缺音频 | control | 45.8800% | 0.0108897884 | 0.5384149365 | 0.0268056334 | 0.07526028515 | 71.1527% |
| 仅残缺音频 | v12b | 45.8800% | 0.01085890736 | 0.5397922348 | 0.02672961808 | 0.07409888403 | 71.1769% |
| 仅残缺音频 | no_causal | 45.8800% | 0.01096961021 | 0.5390261048 | 0.02700211719 | 0.0741106366 | 71.2771% |
| 干净图像+残缺音频 | control | 98.0700% | 0.009733709633 | 0.5766152874 | 0.02395990073 | 0.07094715257 | 73.3127% |
| 干净图像+残缺音频 | v12b | 98.0700% | 0.009778726122 | 0.5759790228 | 0.02407071021 | 0.07001101228 | 73.5867% |
| 干净图像+残缺音频 | no_causal | 98.0700% | 0.01011742623 | 0.576913965 | 0.02490443355 | 0.06928070993 | 73.7006% |
| 双残缺 | control | 92.0300% | 0.009967471337 | 0.5694172061 | 0.0245353141 | 0.07215869381 | 72.9936% |
| 双残缺 | v12b | 92.0300% | 0.009984299806 | 0.5693533241 | 0.02457673797 | 0.07111899968 | 73.2348% |
| 双残缺 | no_causal | 92.0300% | 0.01030136759 | 0.5701290045 | 0.02535721245 | 0.07049502829 | 73.3401% |

</details>

<details>
<summary>音频专项：time_freq_block</summary>

| 输入模式 | 实验 | Index ACC | 音频 MSE | 音频 SSIM | 缺失区 MSE | 缺失区 L1 | top15% 召回 |
|---|---|---|---|---|---|---|---|
| 仅残缺音频 | control | 96.7500% | 0.0008515375959 | 0.979371387 | 0.00515961235 | 0.01795495559 | 95.7505% |
| 仅残缺音频 | v12b | 96.7500% | 0.0007983488309 | 0.9806154703 | 0.004837332493 | 0.01736980207 | 95.8191% |
| 仅残缺音频 | no_causal | 96.7500% | 0.0008048859115 | 0.9804754872 | 0.004876941901 | 0.01741033518 | 95.8570% |
| 干净图像+残缺音频 | control | 99.1200% | 0.0008051505313 | 0.9804057076 | 0.004878545237 | 0.01761080545 | 95.7135% |
| 干净图像+残缺音频 | v12b | 99.1200% | 0.000763986958 | 0.9814773455 | 0.004629128063 | 0.01697668422 | 95.7470% |
| 干净图像+残缺音频 | no_causal | 99.1200% | 0.0007823672045 | 0.9811295378 | 0.00474049716 | 0.0169661741 | 95.9784% |
| 双残缺 | control | 98.7700% | 0.0008025279009 | 0.9806737814 | 0.004862654246 | 0.01747624641 | 95.7146% |
| 双残缺 | v12b | 98.7700% | 0.0007578328617 | 0.9818267635 | 0.004591839346 | 0.01681593883 | 95.7739% |
| 双残缺 | no_causal | 98.7700% | 0.0007754509369 | 0.9814042282 | 0.004698590288 | 0.01683450817 | 95.9462% |

</details>

#### 全指标：按 cue 汇总

已扫描 normal/sweep CSV，实际共有 52 类指标。下面保留所有已计算字段；
fixed 的数值是有效 family 等权均值，n_sum 只表示评估曝光数；原始逐 family 有效 n 见随后附表。
pair Recall 在类别绑定下未启用；无该模态缺失区/可见区时相应 n=0，显示 N/A；源文件没有的字段不臆造。
未记录 Cross-Detail 干预，因为 cross_detail=False；detail_normal 文件后缀不意味着做过 Cross-Detail 消融。

<details>
<summary>fixed_mask / 干净双模态：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 99.1800% | 99.1800% | 99.1800% | 50000 | 50000 | 50000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003503285427 | 0.001023170835 | 0.001022163968 | 50000 | 50000 | 50000 |
| aud_ssim | 0.9115226718 | 0.9706698141 | 0.9707627995 | 50000 | 50000 | 50000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.9600% | 97.0700% | 96.8400% | 50000 | 50000 | 50000 |
| content_aud_same_class_acc | 96.9588% | 97.0688% | 96.8387% | 49980 | 49980 | 49980 |
| content_aud_wrong_acc | 96.9600% | 97.0700% | 96.8400% | 50000 | 50000 | 50000 |
| content_aud_zero_acc | 96.9600% | 97.0700% | 96.8400% | 50000 | 50000 | 50000 |
| content_img_normal_acc | 97.7900% | 97.7900% | 97.7900% | 50000 | 50000 | 50000 |
| content_img_same_class_acc | 97.7891% | 97.7891% | 97.7891% | 49980 | 49980 | 49980 |
| content_img_wrong_acc | 97.7900% | 97.7900% | 97.7900% | 50000 | 50000 | 50000 |
| content_img_zero_acc | 97.7900% | 97.7900% | 97.7900% | 50000 | 50000 | 50000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009732332389 | 0.009732332389 | 0.009732332389 | 50000 | 50000 | 50000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3386767626 | 0.3386767626 | 0.3386767626 | 50000 | 50000 | 50000 |
| pix_var | 0.05885057896 | 0.05885057896 | 0.05885057896 | 50000 | 50000 | 50000 |
| psnr | 20.79410919 | 20.79410919 | 20.79410919 | 50000 | 50000 | 50000 |
| ssim | 0.9426845028 | 0.9426845028 | 0.9426845028 | 50000 | 50000 | 50000 |

</details>

<details>
<summary>fixed_mask / 仅干净图像：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 97.7000% | 97.7000% | 97.7000% | 50000 | 50000 | 50000 |
| aud_masked_l1 | 0.002905382487 | 0.002580899412 | 0.0025012506 | 50000 | 50000 | 50000 |
| aud_masked_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 50000 | 50000 | 50000 |
| aud_mse | 0.0002402440196 | 0.0002218772086 | 0.0002282006215 | 50000 | 50000 | 50000 |
| aud_ssim | 0.9736808665 | 0.9763999848 | 0.976126532 | 50000 | 50000 | 50000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.1300% | 96.5000% | 96.4000% | 50000 | 50000 | 50000 |
| content_aud_same_class_acc | 96.1485% | 96.5186% | 96.4786% | 49980 | 49980 | 49980 |
| content_aud_wrong_acc | 96.1000% | 96.4500% | 96.4300% | 50000 | 50000 | 50000 |
| content_aud_zero_acc | 96.1500% | 96.6200% | 96.4500% | 50000 | 50000 | 50000 |
| content_img_normal_acc | 96.8200% | 96.8200% | 96.8200% | 50000 | 50000 | 50000 |
| content_img_same_class_acc | 96.8187% | 96.8187% | 96.8187% | 49980 | 49980 | 49980 |
| content_img_wrong_acc | 96.8200% | 96.8200% | 96.8200% | 50000 | 50000 | 50000 |
| content_img_zero_acc | 96.8200% | 96.8200% | 96.8200% | 50000 | 50000 | 50000 |
| img2aud_correct_gain | 2.120906993e-05 | 2.734319982e-05 | 2.90869362e-05 | 50000 | 50000 | 50000 |
| img2aud_gate | 0.9636753441 | 0.9270737396 | 0.9410639594 | 50000 | 50000 | 50000 |
| img2aud_normal_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 50000 | 50000 | 50000 |
| img2aud_ratio | 0.05812706452 | 0.06269425778 | 0.05521903026 | 50000 | 50000 | 50000 |
| img2aud_same_class_mse | 0.0002405052923 | 0.0002228766707 | 0.0002300969015 | 49980 | 49980 | 49980 |
| img2aud_same_damage | 2.011856675e-07 | 9.414399495e-07 | 1.835515222e-06 | 49980 | 49980 | 49980 |
| img2aud_win_both | 79.8300% | 82.5600% | 85.4500% | 50000 | 50000 | 50000 |
| img2aud_win_wrong | 82.8300% | 86.0600% | 88.3200% | 50000 | 50000 | 50000 |
| img2aud_win_zero | 95.0400% | 94.6400% | 95.0200% | 50000 | 50000 | 50000 |
| img2aud_wrong_damage | 7.060236394e-06 | 7.88233878e-06 | 1.064793261e-05 | 50000 | 50000 | 50000 |
| img2aud_wrong_mse | 0.0002473042549 | 0.0002297595468 | 0.0002388485523 | 50000 | 50000 | 50000 |
| img2aud_zero_mse | 0.0002614530888 | 0.0002492204076 | 0.0002572875584 | 50000 | 50000 | 50000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009777385685 | 0.009777385685 | 0.009777385685 | 50000 | 50000 | 50000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.340022999 | 0.340022999 | 0.340022999 | 50000 | 50000 | 50000 |
| pix_var | 0.05938630551 | 0.05938630551 | 0.05938630551 | 50000 | 50000 | 50000 |
| psnr | 20.77137023 | 20.77137023 | 20.77137023 | 50000 | 50000 | 50000 |
| ssim | 0.9426458778 | 0.9426458778 | 0.9426458778 | 50000 | 50000 | 50000 |

</details>

<details>
<summary>fixed_mask / 仅干净音频：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 98.8300% | 98.8300% | 98.8300% | 50000 | 50000 | 50000 |
| aud2img_correct_gain | 4.550894817e-05 | 5.925099576e-05 | 3.427388306e-05 | 50000 | 50000 | 50000 |
| aud2img_gate | 0.5948837502 | 0.6400160534 | 0.6886220585 | 50000 | 50000 | 50000 |
| aud2img_normal_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 50000 | 50000 | 50000 |
| aud2img_ratio | 0.6138913798 | 0.8292583166 | 0.855119685 | 50000 | 50000 | 50000 |
| aud2img_same_class_mse | 0.002024612459 | 0.00200570006 | 0.002030547302 | 49980 | 49980 | 49980 |
| aud2img_same_damage | -1.557539837e-05 | -2.074199916e-05 | -2.087765998e-05 | 49980 | 49980 | 49980 |
| aud2img_win_both | 40.8600% | 51.8800% | 43.2100% | 50000 | 50000 | 50000 |
| aud2img_win_wrong | 71.3800% | 79.1800% | 76.4900% | 50000 | 50000 | 50000 |
| aud2img_win_zero | 51.6500% | 62.1900% | 52.4400% | 50000 | 50000 | 50000 |
| aud2img_wrong_damage | 3.748856293e-06 | 1.19613159e-05 | 1.177280845e-05 | 50000 | 50000 | 50000 |
| aud2img_wrong_mse | 0.002043171231 | 0.00203764165 | 0.002062430253 | 50000 | 50000 | 50000 |
| aud2img_zero_mse | 0.002084931313 | 0.002084931313 | 0.002084931313 | 50000 | 50000 | 50000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003496576117 | 0.001021110903 | 0.001020037457 | 50000 | 50000 | 50000 |
| aud_ssim | 0.9115410681 | 0.9707879167 | 0.9709046022 | 50000 | 50000 | 50000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 95.5300% | 96.5900% | 96.5600% | 50000 | 50000 | 50000 |
| content_aud_same_class_acc | 95.5282% | 96.5886% | 96.5586% | 49980 | 49980 | 49980 |
| content_aud_wrong_acc | 95.5300% | 96.5900% | 96.5600% | 50000 | 50000 | 50000 |
| content_aud_zero_acc | 95.5300% | 96.5900% | 96.5600% | 50000 | 50000 | 50000 |
| content_img_normal_acc | 97.7300% | 98.0900% | 97.3500% | 50000 | 50000 | 50000 |
| content_img_same_class_acc | 97.4690% | 97.7491% | 97.6491% | 49980 | 49980 | 49980 |
| content_img_wrong_acc | 97.5200% | 97.7800% | 97.6700% | 50000 | 50000 | 50000 |
| content_img_zero_acc | 97.7100% | 97.7100% | 97.7100% | 50000 | 50000 | 50000 |
| img_coarse_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 50000 | 50000 | 50000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.007310281691 | 0.007187631063 | 0.007240436512 | 50000 | 50000 | 50000 |
| img_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 50000 | 50000 | 50000 |
| img_mse | 0.002039422297 | 0.002025680222 | 0.002050657357 | 50000 | 50000 | 50000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2869477093 | 0.2871338844 | 0.2874145865 | 50000 | 50000 | 50000 |
| pix_var | 0.0461221002 | 0.04620051384 | 0.04626523703 | 50000 | 50000 | 50000 |
| psnr | 37.44974285 | 37.68731136 | 37.47306895 | 50000 | 50000 | 50000 |
| ssim | 0.9860573347 | 0.9861481328 | 0.9859331862 | 50000 | 50000 | 50000 |

</details>

<details>
<summary>fixed_mask / 仅残缺图像：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 92.0280% | 92.0280% | 92.0280% | 50000 | 50000 | 50000 |
| aud_masked_l1 | 0.004555799426 | 0.004169377031 | 0.004057133362 | 50000 | 50000 | 50000 |
| aud_masked_mse | 0.0005867628014 | 0.0005603817686 | 0.0005802758706 | 50000 | 50000 | 50000 |
| aud_mse | 0.0005867628023 | 0.00056038177 | 0.0005802758713 | 50000 | 50000 | 50000 |
| aud_ssim | 0.9304794941 | 0.9346278995 | 0.9343264542 | 50000 | 50000 | 50000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 87.3260% | 88.2720% | 88.3580% | 50000 | 50000 | 50000 |
| content_aud_same_class_acc | 87.3569% | 88.3914% | 88.4234% | 49980 | 49980 | 49980 |
| content_aud_wrong_acc | 87.2360% | 88.2880% | 88.3260% | 50000 | 50000 | 50000 |
| content_aud_zero_acc | 87.4360% | 88.5140% | 88.4180% | 50000 | 50000 | 50000 |
| content_img_normal_acc | 95.5140% | 95.5140% | 95.5140% | 50000 | 50000 | 50000 |
| content_img_same_class_acc | 95.5122% | 95.5122% | 95.5122% | 49980 | 49980 | 49980 |
| content_img_wrong_acc | 95.5140% | 95.5140% | 95.5140% | 50000 | 50000 | 50000 |
| content_img_zero_acc | 95.5140% | 95.5140% | 95.5140% | 50000 | 50000 | 50000 |
| img2aud_correct_gain | 1.471573329e-05 | 1.418887044e-05 | 8.873982471e-06 | 50000 | 50000 | 50000 |
| img2aud_gate | 0.9625790353 | 0.9266087442 | 0.9402460162 | 50000 | 50000 | 50000 |
| img2aud_normal_mse | 0.0005867628014 | 0.0005603817686 | 0.0005802758706 | 50000 | 50000 | 50000 |
| img2aud_ratio | 0.0458131744 | 0.04932446477 | 0.04337704837 | 50000 | 50000 | 50000 |
| img2aud_same_class_mse | 0.0005867439695 | 0.0005609063867 | 0.0005819839584 | 49980 | 49980 | 49980 |
| img2aud_same_damage | -1.752735322e-08 | 5.332726119e-07 | 1.690780805e-06 | 49980 | 49980 | 49980 |
| img2aud_win_both | 65.2080% | 69.7140% | 72.6440% | 50000 | 50000 | 50000 |
| img2aud_win_wrong | 74.0340% | 77.8800% | 79.3180% | 50000 | 50000 | 50000 |
| img2aud_win_zero | 84.4040% | 85.2740% | 87.0500% | 50000 | 50000 | 50000 |
| img2aud_wrong_damage | 3.131124957e-06 | 3.842635163e-06 | 4.962827731e-06 | 50000 | 50000 | 50000 |
| img2aud_wrong_mse | 0.0005898939247 | 0.0005642244014 | 0.0005852386984 | 50000 | 50000 | 50000 |
| img2aud_zero_mse | 0.0006014785353 | 0.0005745706387 | 0.0005891498545 | 50000 | 50000 | 50000 |
| img_coarse_masked_mse | 0.0563251378 | 0.0563251378 | 0.0563251378 | 50000 | 50000 | 50000 |
| img_coarse_visible_mse | 0.03123442326 | 0.03123442326 | 0.03123442326 | 50000 | 50000 | 50000 |
| img_masked_l1 | 0.06368204526 | 0.06368204526 | 0.06368204526 | 50000 | 50000 | 50000 |
| img_masked_mse | 0.02666187984 | 0.02666187984 | 0.02666187984 | 50000 | 50000 | 50000 |
| img_mse | 0.007680466978 | 0.007680466978 | 0.007680466978 | 50000 | 50000 | 50000 |
| img_visible_l1 | 8.090668826e-05 | 8.090668826e-05 | 8.090668826e-05 | 50000 | 50000 | 50000 |
| img_visible_mse | 8.090485169e-09 | 8.090485169e-09 | 8.090485169e-09 | 50000 | 50000 | 50000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3444949508 | 0.3444949508 | 0.3444949508 | 50000 | 50000 | 50000 |
| pix_var | 0.06083904132 | 0.06083904132 | 0.06083904132 | 50000 | 50000 | 50000 |
| psnr | 22.90468118 | 22.90468118 | 22.90468118 | 50000 | 50000 | 50000 |
| ssim | 0.955151473 | 0.955151473 | 0.955151473 | 50000 | 50000 | 50000 |

</details>

<details>
<summary>fixed_mask / 仅残缺音频：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 84.5720% | 84.5720% | 84.5720% | 50000 | 50000 | 50000 |
| aud2img_correct_gain | 0.000823115665 | 0.001068666988 | 0.001243515147 | 50000 | 50000 | 50000 |
| aud2img_gate | 0.5949703392 | 0.6386924635 | 0.6875510692 | 50000 | 50000 | 50000 |
| aud2img_normal_mse | 0.01238851484 | 0.01214296354 | 0.01196811529 | 50000 | 50000 | 50000 |
| aud2img_ratio | 0.4970438705 | 0.6691037681 | 0.6941749318 | 50000 | 50000 | 50000 |
| aud2img_same_class_mse | 0.01245668784 | 0.01230433057 | 0.01218867983 | 49980 | 49980 | 49980 |
| aud2img_same_damage | 6.813985479e-05 | 0.0001613083565 | 0.0002204437419 | 49980 | 49980 | 49980 |
| aud2img_win_both | 43.1060% | 51.1320% | 43.5380% | 50000 | 50000 | 50000 |
| aud2img_win_wrong | 63.8240% | 68.2760% | 65.3100% | 50000 | 50000 | 50000 |
| aud2img_win_zero | 63.6980% | 73.4160% | 62.6900% | 50000 | 50000 | 50000 |
| aud2img_wrong_damage | 8.229305412e-05 | 0.0001962980319 | 0.0002614112083 | 50000 | 50000 | 50000 |
| aud2img_wrong_mse | 0.01247080785 | 0.01233926152 | 0.01222952651 | 50000 | 50000 | 50000 |
| aud2img_zero_mse | 0.01321163048 | 0.01321163048 | 0.01321163048 | 50000 | 50000 | 50000 |
| aud_masked_l1 | 0.03065372642 | 0.02949716002 | 0.02951811614 | 50000 | 50000 | 50000 |
| aud_masked_mse | 0.00995039407 | 0.009492263173 | 0.009588726669 | 50000 | 50000 | 50000 |
| aud_mse | 0.003788456834 | 0.003619976872 | 0.003657144785 | 50000 | 50000 | 50000 |
| aud_ssim | 0.8625069361 | 0.8670313654 | 0.8666763434 | 50000 | 50000 | 50000 |
| aud_visible_l1 | 0 | 0 | 0 | 50000 | 50000 | 50000 |
| aud_visible_mse | 0 | 0 | 0 | 50000 | 50000 | 50000 |
| content_aud_normal_acc | 84.6120% | 84.7960% | 84.8940% | 50000 | 50000 | 50000 |
| content_aud_same_class_acc | 84.6138% | 84.7979% | 84.8960% | 49980 | 49980 | 49980 |
| content_aud_wrong_acc | 84.6120% | 84.7960% | 84.8940% | 50000 | 50000 | 50000 |
| content_aud_zero_acc | 84.6120% | 84.7960% | 84.8940% | 50000 | 50000 | 50000 |
| content_img_normal_acc | 82.8660% | 82.8940% | 82.6640% | 50000 | 50000 | 50000 |
| content_img_same_class_acc | 82.9052% | 83.0192% | 82.9092% | 49980 | 49980 | 49980 |
| content_img_wrong_acc | 82.8580% | 82.9100% | 82.6920% | 50000 | 50000 | 50000 |
| content_img_zero_acc | 82.8820% | 82.8820% | 82.8820% | 50000 | 50000 | 50000 |
| img_coarse_masked_mse | 0.01238851484 | 0.01214296354 | 0.01196811529 | 50000 | 50000 | 50000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.02538469176 | 0.02538130294 | 0.02583650662 | 50000 | 50000 | 50000 |
| img_masked_mse | 0.01238851484 | 0.01214296354 | 0.01196811529 | 50000 | 50000 | 50000 |
| img_mse | 0.01238851457 | 0.0121429631 | 0.01196811502 | 50000 | 50000 | 50000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2626154751 | 0.2620010257 | 0.2617212683 | 50000 | 50000 | 50000 |
| pix_var | 0.03948016092 | 0.03936574236 | 0.0392814029 | 50000 | 50000 | 50000 |
| psnr | 30.90624976 | 31.12207014 | 30.90838183 | 50000 | 50000 | 50000 |
| ssim | 0.892031235 | 0.8959478379 | 0.8984157582 | 50000 | 50000 | 50000 |

</details>

<details>
<summary>fixed_mask / 干净图像+残缺音频：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 98.9340% | 98.9340% | 98.9340% | 50000 | 50000 | 50000 |
| aud_masked_l1 | 0.02930830253 | 0.02812448006 | 0.02797100497 | 50000 | 50000 | 50000 |
| aud_masked_mse | 0.009055382171 | 0.008685033526 | 0.008962544561 | 50000 | 50000 | 50000 |
| aud_mse | 0.003439189032 | 0.003302744176 | 0.003409866151 | 50000 | 50000 | 50000 |
| aud_ssim | 0.8728393103 | 0.8768458413 | 0.8764936821 | 50000 | 50000 | 50000 |
| aud_visible_l1 | 0 | 0 | 0 | 50000 | 50000 | 50000 |
| aud_visible_mse | 0 | 0 | 0 | 50000 | 50000 | 50000 |
| content_aud_normal_acc | 94.2780% | 94.6280% | 94.7540% | 50000 | 50000 | 50000 |
| content_aud_same_class_acc | 94.2597% | 94.6098% | 94.7859% | 49980 | 49980 | 49980 |
| content_aud_wrong_acc | 94.1500% | 94.6300% | 94.7340% | 50000 | 50000 | 50000 |
| content_aud_zero_acc | 94.1200% | 94.5340% | 94.6420% | 50000 | 50000 | 50000 |
| content_img_normal_acc | 97.5200% | 97.5200% | 97.5200% | 50000 | 50000 | 50000 |
| content_img_same_class_acc | 97.5190% | 97.5190% | 97.5190% | 49980 | 49980 | 49980 |
| content_img_wrong_acc | 97.5200% | 97.5200% | 97.5200% | 50000 | 50000 | 50000 |
| content_img_zero_acc | 97.5200% | 97.5200% | 97.5200% | 50000 | 50000 | 50000 |
| img2aud_correct_gain | 0.0002762808874 | 0.0001603091368 | 4.270494836e-06 | 50000 | 50000 | 50000 |
| img2aud_gate | 0.9557682544 | 0.9202015691 | 0.9336279799 | 50000 | 50000 | 50000 |
| img2aud_normal_mse | 0.009055382171 | 0.008685033526 | 0.008962544561 | 50000 | 50000 | 50000 |
| img2aud_ratio | 0.02158005093 | 0.02311903475 | 0.02059960777 | 50000 | 50000 | 50000 |
| img2aud_same_class_mse | 0.009055011668 | 0.008684787714 | 0.008961976089 | 49980 | 49980 | 49980 |
| img2aud_same_damage | -1.140159209e-06 | -8.850975082e-07 | -1.014432983e-06 | 49980 | 49980 | 49980 |
| img2aud_win_both | 36.0320% | 34.9300% | 31.2380% | 50000 | 50000 | 50000 |
| img2aud_win_wrong | 43.5740% | 44.4740% | 44.0560% | 50000 | 50000 | 50000 |
| img2aud_win_zero | 68.0400% | 63.9340% | 56.0680% | 50000 | 50000 | 50000 |
| img2aud_wrong_damage | -1.598179023e-06 | 6.484932923e-06 | 1.003476575e-05 | 50000 | 50000 | 50000 |
| img2aud_wrong_mse | 0.009053783996 | 0.008691518495 | 0.008972579324 | 50000 | 50000 | 50000 |
| img2aud_zero_mse | 0.009331663075 | 0.008845342677 | 0.008966815091 | 50000 | 50000 | 50000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009790254229 | 0.009790254229 | 0.009790254229 | 50000 | 50000 | 50000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3394689441 | 0.3394689441 | 0.3394689441 | 50000 | 50000 | 50000 |
| pix_var | 0.05932056382 | 0.05932056382 | 0.05932056382 | 50000 | 50000 | 50000 |
| psnr | 20.76833471 | 20.76833471 | 20.76833471 | 50000 | 50000 | 50000 |
| ssim | 0.9425965783 | 0.9425965783 | 0.9425965783 | 50000 | 50000 | 50000 |

</details>

<details>
<summary>fixed_mask / 残缺图像+干净音频：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 99.1440% | 99.1440% | 99.1440% | 50000 | 50000 | 50000 |
| aud2img_correct_gain | 0.0009556563124 | 0.001296675922 | 0.001157458343 | 50000 | 50000 | 50000 |
| aud2img_gate | 0.6189587738 | 0.6616415961 | 0.6977425102 | 50000 | 50000 | 50000 |
| aud2img_normal_mse | 0.02496237731 | 0.02462135759 | 0.02476057522 | 50000 | 50000 | 50000 |
| aud2img_ratio | 0.3679328426 | 0.5709908211 | 0.5741675758 | 50000 | 50000 | 50000 |
| aud2img_same_class_mse | 0.02496240177 | 0.02462020766 | 0.0247615172 | 49980 | 49980 | 49980 |
| aud2img_same_damage | 1.563935201e-06 | 3.819621403e-07 | 2.263696923e-06 | 49980 | 49980 | 49980 |
| aud2img_win_both | 41.4380% | 43.1480% | 41.7300% | 50000 | 50000 | 50000 |
| aud2img_win_wrong | 53.0020% | 53.0060% | 53.3260% | 50000 | 50000 | 50000 |
| aud2img_win_zero | 73.8880% | 77.3480% | 73.3260% | 50000 | 50000 | 50000 |
| aud2img_wrong_damage | 5.451788243e-05 | 5.944807109e-05 | 8.389348639e-05 | 50000 | 50000 | 50000 |
| aud2img_wrong_mse | 0.02501689519 | 0.02468080577 | 0.02484446874 | 50000 | 50000 | 50000 |
| aud2img_zero_mse | 0.02591803356 | 0.02591803356 | 0.02591803356 | 50000 | 50000 | 50000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003500839663 | 0.001021123422 | 0.001019895179 | 50000 | 50000 | 50000 |
| aud_ssim | 0.9115529218 | 0.9707167486 | 0.9708236575 | 50000 | 50000 | 50000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.6180% | 96.9940% | 96.7000% | 50000 | 50000 | 50000 |
| content_aud_same_class_acc | 96.6166% | 96.9928% | 96.6987% | 49980 | 49980 | 49980 |
| content_aud_wrong_acc | 96.6180% | 96.9940% | 96.7000% | 50000 | 50000 | 50000 |
| content_aud_zero_acc | 96.6180% | 96.9940% | 96.7000% | 50000 | 50000 | 50000 |
| content_img_normal_acc | 96.7820% | 96.7120% | 96.7880% | 50000 | 50000 | 50000 |
| content_img_same_class_acc | 96.7467% | 96.7387% | 96.7607% | 49980 | 49980 | 49980 |
| content_img_wrong_acc | 96.7420% | 96.7360% | 96.7360% | 50000 | 50000 | 50000 |
| content_img_zero_acc | 96.7200% | 96.7200% | 96.7200% | 50000 | 50000 | 50000 |
| img_coarse_masked_mse | 0.05402381544 | 0.05201543581 | 0.05126693592 | 50000 | 50000 | 50000 |
| img_coarse_visible_mse | 0.03073154717 | 0.03073154717 | 0.03073154717 | 50000 | 50000 | 50000 |
| img_masked_l1 | 0.06211222736 | 0.06132812118 | 0.06060463013 | 50000 | 50000 | 50000 |
| img_masked_mse | 0.02496237731 | 0.02462135759 | 0.02476057522 | 50000 | 50000 | 50000 |
| img_mse | 0.007218561702 | 0.007103987851 | 0.007149694622 | 50000 | 50000 | 50000 |
| img_visible_l1 | 8.090668826e-05 | 8.090668826e-05 | 8.090668826e-05 | 50000 | 50000 | 50000 |
| img_visible_mse | 8.090485169e-09 | 8.090485169e-09 | 8.090485169e-09 | 50000 | 50000 | 50000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3445616007 | 0.3449865341 | 0.3457467437 | 50000 | 50000 | 50000 |
| pix_var | 0.06085577756 | 0.0610014841 | 0.06126609221 | 50000 | 50000 | 50000 |
| psnr | 23.19099652 | 23.30264554 | 23.30334092 | 50000 | 50000 | 50000 |
| ssim | 0.9579087732 | 0.9586285542 | 0.95843637 | 50000 | 50000 | 50000 |

</details>

<details>
<summary>fixed_mask / 双残缺：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 97.5360% | 97.5360% | 97.5360% | 50000 | 50000 | 50000 |
| aud2img_correct_gain | 0.0008541049949 | 0.001164061437 | 0.001109524558 | 50000 | 50000 | 50000 |
| aud2img_gate | 0.6169865298 | 0.6580509715 | 0.694807246 | 50000 | 50000 | 50000 |
| aud2img_normal_mse | 0.0250583435 | 0.02474838701 | 0.02480292383 | 50000 | 50000 | 50000 |
| aud2img_ratio | 0.3037250458 | 0.4810166701 | 0.4807973651 | 50000 | 50000 | 50000 |
| aud2img_same_class_mse | 0.0250443522 | 0.02473611551 | 0.02479095154 | 49980 | 49980 | 49980 |
| aud2img_same_damage | -8.779802408e-06 | -7.045806848e-06 | -7.088383272e-06 | 49980 | 49980 | 49980 |
| aud2img_win_both | 38.9580% | 40.6140% | 40.2520% | 50000 | 50000 | 50000 |
| aud2img_win_wrong | 50.2320% | 50.7020% | 51.0080% | 50000 | 50000 | 50000 |
| aud2img_win_zero | 71.7880% | 74.6980% | 72.7120% | 50000 | 50000 | 50000 |
| aud2img_wrong_damage | 3.736286544e-05 | 4.671115425e-05 | 7.072625957e-05 | 50000 | 50000 | 50000 |
| aud2img_wrong_mse | 0.02509570633 | 0.02479509829 | 0.02487365013 | 50000 | 50000 | 50000 |
| aud2img_zero_mse | 0.02591244846 | 0.02591244846 | 0.02591244846 | 50000 | 50000 | 50000 |
| aud_masked_l1 | 0.02984895339 | 0.02857941883 | 0.02846586969 | 50000 | 50000 | 50000 |
| aud_masked_mse | 0.009306500793 | 0.008892291977 | 0.009145992404 | 50000 | 50000 | 50000 |
| aud_mse | 0.0035461025 | 0.003392918741 | 0.003490850654 | 50000 | 50000 | 50000 |
| aud_ssim | 0.8696924499 | 0.8740220185 | 0.8736596105 | 50000 | 50000 | 50000 |
| aud_visible_l1 | 0 | 0 | 0 | 50000 | 50000 | 50000 |
| aud_visible_mse | 0 | 0 | 0 | 50000 | 50000 | 50000 |
| content_aud_normal_acc | 92.4900% | 93.0000% | 93.1660% | 50000 | 50000 | 50000 |
| content_aud_same_class_acc | 92.4810% | 92.9772% | 93.1713% | 49980 | 49980 | 49980 |
| content_aud_wrong_acc | 92.4640% | 92.9060% | 93.1320% | 50000 | 50000 | 50000 |
| content_aud_zero_acc | 92.4620% | 92.9040% | 92.9980% | 50000 | 50000 | 50000 |
| content_img_normal_acc | 96.3500% | 96.3260% | 96.3960% | 50000 | 50000 | 50000 |
| content_img_same_class_acc | 96.3205% | 96.3165% | 96.3605% | 49980 | 49980 | 49980 |
| content_img_wrong_acc | 96.3240% | 96.3280% | 96.2920% | 50000 | 50000 | 50000 |
| content_img_zero_acc | 96.1780% | 96.1780% | 96.1780% | 50000 | 50000 | 50000 |
| img2aud_correct_gain | 0.0002022408083 | 0.000113826913 | -1.368040765e-05 | 50000 | 50000 | 50000 |
| img2aud_gate | 0.9548942977 | 0.9200227778 | 0.9331558047 | 50000 | 50000 | 50000 |
| img2aud_normal_mse | 0.009306500793 | 0.008892291977 | 0.009145992404 | 50000 | 50000 | 50000 |
| img2aud_ratio | 0.01718734891 | 0.01835457455 | 0.01631499477 | 50000 | 50000 | 50000 |
| img2aud_same_class_mse | 0.009307047037 | 0.008892722659 | 0.009146301944 | 49980 | 49980 | 49980 |
| img2aud_same_damage | 3.193298668e-07 | 2.212590411e-07 | 1.836232026e-07 | 49980 | 49980 | 49980 |
| img2aud_win_both | 34.7220% | 33.3220% | 29.6720% | 50000 | 50000 | 50000 |
| img2aud_win_wrong | 42.8340% | 43.8000% | 43.3120% | 50000 | 50000 | 50000 |
| img2aud_win_zero | 67.0560% | 62.7020% | 55.3260% | 50000 | 50000 | 50000 |
| img2aud_wrong_damage | -2.260745511e-06 | 2.90745709e-06 | 4.159590178e-06 | 50000 | 50000 | 50000 |
| img2aud_wrong_mse | 0.009304240097 | 0.008895199425 | 0.00915015201 | 50000 | 50000 | 50000 |
| img2aud_zero_mse | 0.009508741612 | 0.009006118852 | 0.009132311989 | 50000 | 50000 | 50000 |
| img_coarse_masked_mse | 0.05374747917 | 0.05236045036 | 0.05149886524 | 50000 | 50000 | 50000 |
| img_coarse_visible_mse | 0.0305505526 | 0.0305505526 | 0.0305505526 | 50000 | 50000 | 50000 |
| img_masked_l1 | 0.06250876536 | 0.06192210415 | 0.06142812622 | 50000 | 50000 | 50000 |
| img_masked_mse | 0.0250583435 | 0.02474838701 | 0.02480292383 | 50000 | 50000 | 50000 |
| img_mse | 0.007275962728 | 0.007170902732 | 0.007190097554 | 50000 | 50000 | 50000 |
| img_visible_l1 | 8.088703689e-05 | 8.088703689e-05 | 8.088703689e-05 | 50000 | 50000 | 50000 |
| img_visible_mse | 8.088519822e-09 | 8.088519822e-09 | 8.088519822e-09 | 50000 | 50000 | 50000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3452373803 | 0.3454288125 | 0.3459680259 | 50000 | 50000 | 50000 |
| pix_var | 0.06093841419 | 0.0610062629 | 0.06119816527 | 50000 | 50000 | 50000 |
| psnr | 23.1389515 | 23.24099221 | 23.24639548 | 50000 | 50000 | 50000 |
| ssim | 0.9575710225 | 0.9582245398 | 0.9581674529 | 50000 | 50000 | 50000 |

</details>

<details>
<summary>legacy_random / 干净双模态：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 99.1800% | 99.1800% | 99.1800% | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003503285427 | 0.001023170835 | 0.001022163968 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115226718 | 0.9706698141 | 0.9707627995 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.9588% | 97.0688% | 96.8387% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.7891% | 97.7891% | 97.7891% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009732332389 | 0.009732332389 | 0.009732332389 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3399402201 | 0.3399402201 | 0.3399402201 | 10000 | 10000 | 10000 |
| pix_var | 0.05885057896 | 0.05885057896 | 0.05885057896 | 10000 | 10000 | 10000 |
| psnr | 20.79410919 | 20.79410919 | 20.79410919 | 10000 | 10000 | 10000 |
| ssim | 0.9426845028 | 0.9426845028 | 0.9426845028 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>legacy_random / 仅干净图像：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 97.7000% | 97.7000% | 97.7000% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.002905382487 | 0.002580899412 | 0.0025012506 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| aud_mse | 0.0002402440196 | 0.0002218772086 | 0.0002282006215 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9736808665 | 0.9763999848 | 0.976126532 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.1300% | 96.5000% | 96.4000% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.1485% | 96.5186% | 96.4786% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.1000% | 96.4500% | 96.4300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.1500% | 96.6200% | 96.4500% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.8187% | 96.8187% | 96.8187% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 2.120906993e-05 | 2.734319982e-05 | 2.90869362e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9636753441 | 0.9270737396 | 0.9410639594 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.05812706452 | 0.06269425778 | 0.05521903026 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0002405052923 | 0.0002228766707 | 0.0002300969015 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 2.011856675e-07 | 9.414399495e-07 | 1.835515222e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 79.8300% | 82.5600% | 85.4500% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 82.8300% | 86.0600% | 88.3200% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 95.0400% | 94.6400% | 95.0200% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 7.060236394e-06 | 7.88233878e-06 | 1.064793261e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0002473042549 | 0.0002297595468 | 0.0002388485523 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0002614530888 | 0.0002492204076 | 0.0002572875584 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009777385685 | 0.009777385685 | 0.009777385685 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3399478495 | 0.3399478495 | 0.3399478495 | 10000 | 10000 | 10000 |
| pix_var | 0.05938630551 | 0.05938630551 | 0.05938630551 | 10000 | 10000 | 10000 |
| psnr | 20.77137023 | 20.77137023 | 20.77137023 | 10000 | 10000 | 10000 |
| ssim | 0.9426458778 | 0.9426458778 | 0.9426458778 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>legacy_random / 仅干净音频：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 98.8300% | 98.8300% | 98.8300% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 4.550894817e-05 | 5.925099576e-05 | 3.427388306e-05 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5948837502 | 0.6400160534 | 0.6886220585 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.6138913798 | 0.8292583166 | 0.855119685 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.002024612459 | 0.00200570006 | 0.002030547302 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -1.557539837e-05 | -2.074199916e-05 | -2.087765998e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 40.8600% | 51.8800% | 43.2100% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 71.3800% | 79.1800% | 76.4900% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 51.6500% | 62.1900% | 52.4400% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 3.748856293e-06 | 1.19613159e-05 | 1.177280845e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.002043171231 | 0.00203764165 | 0.002062430253 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.002084931313 | 0.002084931313 | 0.002084931313 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003496576117 | 0.001021110903 | 0.001020037457 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115410681 | 0.9707879167 | 0.9709046022 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 95.5282% | 96.5886% | 96.5586% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7300% | 98.0900% | 97.3500% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.4690% | 97.7491% | 97.6491% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.5200% | 97.7800% | 97.6700% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7100% | 97.7100% | 97.7100% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.007310281691 | 0.007187631063 | 0.007240436512 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_mse | 0.002039422297 | 0.002025680222 | 0.002050657357 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2852356732 | 0.2854161859 | 0.285684526 | 10000 | 10000 | 10000 |
| pix_var | 0.0461221002 | 0.04620051384 | 0.04626523703 | 10000 | 10000 | 10000 |
| psnr | 37.44974285 | 37.68731136 | 37.47306895 | 10000 | 10000 | 10000 |
| ssim | 0.9860573347 | 0.9861481328 | 0.9859331862 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>legacy_random / 仅残缺图像：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 92.3600% | 92.3600% | 92.3600% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.004401024371 | 0.004016935972 | 0.003898758639 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.000549146415 | 0.0005232986979 | 0.0005377318759 | 10000 | 10000 | 10000 |
| aud_mse | 0.0005491464123 | 0.0005232986961 | 0.0005377318716 | 10000 | 10000 | 10000 |
| aud_ssim | 0.934874578 | 0.938891894 | 0.9387075771 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 88.0700% | 89.1900% | 89.1900% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 88.1553% | 89.2957% | 89.2457% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 88.2000% | 89.1000% | 89.2700% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 88.1200% | 89.3800% | 89.2400% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 95.4900% | 95.4900% | 95.4900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 95.4882% | 95.4882% | 95.4882% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 95.4900% | 95.4900% | 95.4900% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 95.4900% | 95.4900% | 95.4900% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 1.433363229e-05 | 1.575728901e-05 | 1.225653508e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9626510489 | 0.926657808 | 0.9403141308 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.000549146415 | 0.0005232986979 | 0.0005377318759 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.04585074922 | 0.04928698386 | 0.04333371771 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0005494240526 | 0.000524346426 | 0.0005401250394 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 1.846825965e-07 | 9.470778609e-07 | 2.252867799e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 66.2200% | 70.6300% | 74.1100% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 75.2400% | 78.7100% | 80.6300% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 84.7700% | 85.7500% | 87.6000% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 3.536865775e-06 | 4.173188881e-06 | 5.532323274e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0005526832814 | 0.0005274718828 | 0.0005432641973 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.000563480047 | 0.000539055987 | 0.0005499884168 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.06070868487 | 0.06070868487 | 0.06070868487 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.02910082169 | 0.02910082169 | 0.02910082169 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.06892660542 | 0.06892660542 | 0.06892660542 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.0292326478 | 0.0292326478 | 0.0292326478 | 10000 | 10000 | 10000 |
| img_mse | 0.007614292219 | 0.007614292219 | 0.007614292219 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.143082536e-05 | 8.143082536e-05 | 8.143082536e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.142905017e-09 | 8.142905017e-09 | 8.142905017e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3455364406 | 0.3455364406 | 0.3455364406 | 10000 | 10000 | 10000 |
| pix_var | 0.06094776839 | 0.06094776839 | 0.06094776839 | 10000 | 10000 | 10000 |
| psnr | 22.86542305 | 22.86542305 | 22.86542305 | 10000 | 10000 | 10000 |
| ssim | 0.9552188177 | 0.9552188177 | 0.9552188177 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>legacy_random / 仅残缺音频：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 85.7900% | 85.7900% | 85.7900% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0007191967652 | 0.0009368507476 | 0.0010868413 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5947398814 | 0.638415197 | 0.6873055408 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.01148949487 | 0.01127184076 | 0.01112185018 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.5017223105 | 0.675679085 | 0.7000467607 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.0115440222 | 0.01141726737 | 0.01132947641 | 9996 | 9996 | 9996 |
| aud2img_same_damage | 7.026617956e-05 | 0.0001606745704 | 0.0002221993437 | 9996 | 9996 | 9996 |
| aud2img_win_both | 43.0900% | 51.2700% | 42.8900% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 63.6300% | 68.1800% | 64.7600% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 62.7800% | 73.2500% | 62.0100% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 7.682295728e-05 | 0.0001769999542 | 0.000235098807 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.01156631779 | 0.01144884081 | 0.011356949 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.01220869151 | 0.01220869151 | 0.01220869151 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02927086294 | 0.02797595997 | 0.0280077034 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.009371610844 | 0.008850679514 | 0.008943945056 | 10000 | 10000 | 10000 |
| aud_mse | 0.003578324331 | 0.003382371552 | 0.003418815638 | 10000 | 10000 | 10000 |
| aud_ssim | 0.8705906674 | 0.8760217186 | 0.8755758115 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 85.9300% | 86.0700% | 85.9900% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 85.9544% | 86.0944% | 86.0144% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 85.9300% | 86.0700% | 85.9900% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 85.9300% | 86.0700% | 85.9900% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 84.0700% | 84.0800% | 83.8600% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 84.0636% | 84.2037% | 83.9336% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 83.9500% | 84.1800% | 83.9500% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 84.0900% | 84.0900% | 84.0900% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.01148949487 | 0.01127184076 | 0.01112185018 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.02389682578 | 0.0238848722 | 0.02430988313 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.01148949487 | 0.01127184076 | 0.01112185018 | 10000 | 10000 | 10000 |
| img_mse | 0.01148949463 | 0.01127184049 | 0.01112184997 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.271656692 | 0.2710115016 | 0.2706104815 | 10000 | 10000 | 10000 |
| pix_var | 0.04038360715 | 0.04023037851 | 0.04010954499 | 10000 | 10000 | 10000 |
| psnr | 31.28457344 | 31.50659519 | 31.2869743 | 10000 | 10000 | 10000 |
| ssim | 0.9000236948 | 0.9035166462 | 0.9056596614 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>legacy_random / 干净图像+残缺音频：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 98.9600% | 98.9600% | 98.9600% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02803460948 | 0.02671373892 | 0.02658687366 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.008538770679 | 0.008107095659 | 0.008371060044 | 10000 | 10000 | 10000 |
| aud_mse | 0.003252138307 | 0.003089098994 | 0.00319140949 | 10000 | 10000 | 10000 |
| aud_ssim | 0.8807206806 | 0.8855609869 | 0.8850794596 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 94.7600% | 94.9800% | 95.1900% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 94.8379% | 95.0280% | 95.1681% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 94.7200% | 94.9400% | 94.9800% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 94.8500% | 94.9400% | 95.0200% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.5000% | 97.5000% | 97.5000% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.4990% | 97.4990% | 97.4990% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.5000% | 97.5000% | 97.5000% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.5000% | 97.5000% | 97.5000% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0002911213193 | 0.0001817647181 | 3.269380778e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9554412912 | 0.9198254534 | 0.9332742281 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.008538770679 | 0.008107095659 | 0.008371060044 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.02401790369 | 0.02578279207 | 0.02290292897 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.008532503409 | 0.008101494878 | 0.008364864397 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -2.43109985e-07 | 8.201565527e-07 | 1.079840022e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 36.1900% | 36.1000% | 32.3800% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 43.5900% | 45.5300% | 44.9500% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 70.0600% | 66.2600% | 58.7100% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | -1.765788326e-06 | 7.456294744e-06 | 1.142577223e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.008537004831 | 0.008114551979 | 0.00838248594 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.008829892039 | 0.008288860309 | 0.008403754005 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009783406097 | 0.009783406097 | 0.009783406097 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3416532576 | 0.3416532576 | 0.3416532576 | 10000 | 10000 | 10000 |
| pix_var | 0.05928457528 | 0.05928457528 | 0.05928457528 | 10000 | 10000 | 10000 |
| psnr | 20.77048488 | 20.77048488 | 20.77048488 | 10000 | 10000 | 10000 |
| ssim | 0.9426022731 | 0.9426022731 | 0.9426022731 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>legacy_random / 残缺图像+干净音频：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 99.1600% | 99.1600% | 99.1600% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.001156052208 | 0.001507425934 | 0.001401434613 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6170308393 | 0.6594413919 | 0.6949520991 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.02710606877 | 0.02675469481 | 0.02686068637 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.3379561121 | 0.5088711754 | 0.516051271 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.02708293691 | 0.02673262496 | 0.02684275328 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -5.602101909e-06 | -4.799576746e-06 | -1.569811177e-06 | 9996 | 9996 | 9996 |
| aud2img_win_both | 42.1300% | 43.5500% | 41.9400% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 52.2100% | 52.3400% | 52.5800% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 75.9700% | 78.8300% | 74.6300% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 2.364825222e-05 | 3.375893191e-05 | 5.172351762e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.02712971728 | 0.02678845391 | 0.02691241012 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.02826212105 | 0.02826212105 | 0.02826212105 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003501629138 | 0.001021167909 | 0.001019949091 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115314157 | 0.9707052304 | 0.9708114794 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.6600% | 96.9700% | 96.6000% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.6587% | 96.9688% | 96.5986% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.6600% | 96.9700% | 96.6000% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.6600% | 96.9700% | 96.6000% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.7700% | 96.7100% | 96.7400% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.7387% | 96.8788% | 96.7387% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.7900% | 96.9000% | 96.7900% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.6900% | 96.6900% | 96.6900% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.05777017832 | 0.05563699696 | 0.05477552149 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.02862981399 | 0.02862981399 | 0.02862981399 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.06686613178 | 0.06606935723 | 0.06523520889 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.02710606877 | 0.02675469481 | 0.02686068637 | 10000 | 10000 | 10000 |
| img_mse | 0.007102384464 | 0.006989629062 | 0.007024656107 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.143082536e-05 | 8.143082536e-05 | 8.143082536e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.142905017e-09 | 8.142905017e-09 | 8.142905017e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3456449509 | 0.3460197151 | 0.3467237353 | 10000 | 10000 | 10000 |
| pix_var | 0.06099751964 | 0.06113048643 | 0.06139310449 | 10000 | 10000 | 10000 |
| psnr | 23.17452033 | 23.28899767 | 23.29306716 | 10000 | 10000 | 10000 |
| ssim | 0.958277329 | 0.9589780867 | 0.958841262 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>legacy_random / 双残缺：全部指标</summary>

| 指标 | control | v12b | no_causal | control n_sum | v12b n_sum | no_causal n_sum |
|---|---|---|---|---|---|---|
| acc | 97.2800% | 97.2800% | 97.2800% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0008603106864 | 0.001163088947 | 0.001083243625 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6170564057 | 0.658111935 | 0.6945510743 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.02532390426 | 0.0250211262 | 0.02510097159 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.3038183321 | 0.4699307745 | 0.4721096097 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.02528559596 | 0.02498310283 | 0.02506784399 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -1.589017392e-05 | -1.610946085e-05 | -1.241369187e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 38.5600% | 40.5500% | 39.1600% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 50.2800% | 50.6700% | 50.9600% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 70.6500% | 74.4200% | 70.9300% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 3.250067465e-05 | 4.27605171e-05 | 6.097319155e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.02535640543 | 0.02506388687 | 0.02516194481 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.02618421544 | 0.02618421544 | 0.02618421544 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02759425491 | 0.02635841867 | 0.026269459 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.008598824614 | 0.00817188829 | 0.008401221268 | 10000 | 10000 | 10000 |
| aud_mse | 0.003243830509 | 0.003085861862 | 0.003173652124 | 10000 | 10000 | 10000 |
| aud_ssim | 0.8870781691 | 0.8914218223 | 0.8911181785 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 92.6000% | 92.9500% | 93.2300% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 92.5370% | 93.0472% | 93.1773% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 92.4600% | 92.8800% | 93.0900% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 92.5500% | 93.0500% | 92.8100% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.2800% | 96.3700% | 96.4900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.4486% | 96.3886% | 96.3886% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.3900% | 96.4500% | 96.3600% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.3400% | 96.3400% | 96.3400% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0001949578249 | 0.0001146678842 | -2.151020546e-06 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9559663939 | 0.9210142807 | 0.9340382945 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.008598824614 | 0.00817188829 | 0.008401221268 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.01707282281 | 0.01809543395 | 0.01611432302 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.008591988844 | 0.008166542198 | 0.008394838782 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -5.621254821e-07 | 1.257507768e-06 | 1.007103719e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 35.0900% | 33.3200% | 30.1400% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 42.6500% | 43.2300% | 42.9600% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 66.0000% | 61.6800% | 55.0400% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | -8.016939973e-08 | 4.174842969e-06 | 6.584647554e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.008598744443 | 0.008176063168 | 0.008407805932 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.008793782392 | 0.008286556166 | 0.008399070302 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.05335954244 | 0.05211471896 | 0.05109498558 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.03069736354 | 0.03069736354 | 0.03069736354 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.06291228101 | 0.06236043167 | 0.06176770277 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.02532390426 | 0.0250211262 | 0.02510097159 | 10000 | 10000 | 10000 |
| img_mse | 0.007473485468 | 0.007371911891 | 0.007402919574 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.09325792e-05 | 8.09325792e-05 | 8.09325792e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.093074925e-09 | 8.093074925e-09 | 8.093074925e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3445170522 | 0.3447151184 | 0.3455003798 | 10000 | 10000 | 10000 |
| pix_var | 0.06075927243 | 0.06083749235 | 0.06110813096 | 10000 | 10000 | 10000 |
| psnr | 22.90560941 | 22.99644513 | 22.99737832 | 10000 | 10000 | 10000 |
| ssim | 0.9559758368 | 0.9566177238 | 0.956496457 | 10000 | 10000 | 10000 |

</details>

#### 全指标：逐 family 原始值与有效 n

本节对源 sweep 每条 metric/value/n 给出三组对照；normal 共有项经逐项验证一致，不重复计算。
random 原始表只有一个混合组，上节的 random 数值即其原始值；仍在下表记录，便于覆盖审计。

<details>
<summary>fixed_mask / mask_horizontal / partial_temporal：全部原始指标</summary>

**干净双模态 (`clean_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1800% | 99.1800% | 99.1800% | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003503285427 | 0.001023170835 | 0.001022163968 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115226718 | 0.9706698141 | 0.9707627995 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.9588% | 97.0688% | 96.8387% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.7891% | 97.7891% | 97.7891% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009732332389 | 0.009732332389 | 0.009732332389 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3393760026 | 0.3393760026 | 0.3393760026 | 10000 | 10000 | 10000 |
| pix_var | 0.05885057896 | 0.05885057896 | 0.05885057896 | 10000 | 10000 | 10000 |
| psnr | 20.79410919 | 20.79410919 | 20.79410919 | 10000 | 10000 | 10000 |
| ssim | 0.9426845028 | 0.9426845028 | 0.9426845028 | 10000 | 10000 | 10000 |

**仅干净图像 (`clean_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 97.7000% | 97.7000% | 97.7000% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.002905382487 | 0.002580899412 | 0.0025012506 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| aud_mse | 0.0002402440196 | 0.0002218772086 | 0.0002282006215 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9736808665 | 0.9763999848 | 0.976126532 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.1300% | 96.5000% | 96.4000% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.1485% | 96.5186% | 96.4786% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.1000% | 96.4500% | 96.4300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.1500% | 96.6200% | 96.4500% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.8187% | 96.8187% | 96.8187% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 2.120906993e-05 | 2.734319982e-05 | 2.90869362e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9636753441 | 0.9270737396 | 0.9410639594 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.05812706452 | 0.06269425778 | 0.05521903026 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0002405052923 | 0.0002228766707 | 0.0002300969015 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 2.011856675e-07 | 9.414399495e-07 | 1.835515222e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 79.8300% | 82.5600% | 85.4500% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 82.8300% | 86.0600% | 88.3200% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 95.0400% | 94.6400% | 95.0200% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 7.060236394e-06 | 7.88233878e-06 | 1.064793261e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0002473042549 | 0.0002297595468 | 0.0002388485523 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0002614530888 | 0.0002492204076 | 0.0002572875584 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009777385685 | 0.009777385685 | 0.009777385685 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3401130736 | 0.3401130736 | 0.3401130736 | 10000 | 10000 | 10000 |
| pix_var | 0.05938630551 | 0.05938630551 | 0.05938630551 | 10000 | 10000 | 10000 |
| psnr | 20.77137023 | 20.77137023 | 20.77137023 | 10000 | 10000 | 10000 |
| ssim | 0.9426458778 | 0.9426458778 | 0.9426458778 | 10000 | 10000 | 10000 |

**仅干净音频 (`clean_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 98.8300% | 98.8300% | 98.8300% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 4.550894817e-05 | 5.925099576e-05 | 3.427388306e-05 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5948837502 | 0.6400160534 | 0.6886220585 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.6138913798 | 0.8292583166 | 0.855119685 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.002024612459 | 0.00200570006 | 0.002030547302 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -1.557539837e-05 | -2.074199916e-05 | -2.087765998e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 40.8600% | 51.8800% | 43.2100% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 71.3800% | 79.1800% | 76.4900% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 51.6500% | 62.1900% | 52.4400% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 3.748856293e-06 | 1.19613159e-05 | 1.177280845e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.002043171231 | 0.00203764165 | 0.002062430253 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.002084931313 | 0.002084931313 | 0.002084931313 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003496576117 | 0.001021110903 | 0.001020037457 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115410681 | 0.9707879167 | 0.9709046022 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 95.5282% | 96.5886% | 96.5586% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7300% | 98.0900% | 97.3500% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.4690% | 97.7491% | 97.6491% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.5200% | 97.7800% | 97.6700% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7100% | 97.7100% | 97.7100% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.007310281691 | 0.007187631063 | 0.007240436512 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_mse | 0.002039422297 | 0.002025680222 | 0.002050657357 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2908204198 | 0.291008085 | 0.2912720442 | 10000 | 10000 | 10000 |
| pix_var | 0.0461221002 | 0.04620051384 | 0.04626523703 | 10000 | 10000 | 10000 |
| psnr | 37.44974285 | 37.68731136 | 37.47306895 | 10000 | 10000 | 10000 |
| ssim | 0.9860573347 | 0.9861481328 | 0.9859331862 | 10000 | 10000 | 10000 |

**仅残缺图像 (`corrupt_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 93.0200% | 93.0200% | 93.0200% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.004308295818 | 0.003930952943 | 0.003834146488 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0005384710513 | 0.000514847415 | 0.0005376132005 | 10000 | 10000 | 10000 |
| aud_mse | 0.000538471055 | 0.0005148474159 | 0.0005376131995 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9358811138 | 0.9399658245 | 0.9393372795 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 87.9700% | 89.0900% | 89.0600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 87.8451% | 88.9656% | 89.0956% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 87.9500% | 89.0300% | 89.0500% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 88.0300% | 89.3100% | 89.1300% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 94.1400% | 94.1400% | 94.1400% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 94.1377% | 94.1377% | 94.1377% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 94.1400% | 94.1400% | 94.1400% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 94.1400% | 94.1400% | 94.1400% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 1.499454912e-05 | 1.299456475e-05 | 6.611951333e-06 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9625293028 | 0.9266102537 | 0.9402363409 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.0005384710513 | 0.000514847415 | 0.0005376132005 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.04583670205 | 0.04907570965 | 0.0431821288 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0005386167781 | 0.0005152297887 | 0.0005389565024 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -2.083913027e-08 | 2.15035633e-07 | 1.166060847e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 67.0700% | 71.6200% | 74.3100% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 74.3000% | 78.1900% | 79.7900% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 87.0700% | 87.4000% | 88.4200% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 2.301058148e-06 | 3.042614913e-06 | 3.452348804e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0005407721132 | 0.0005178900277 | 0.000541065549 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0005534656043 | 0.0005278419744 | 0.0005442251532 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.04275578155 | 0.04275578155 | 0.04275578155 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.02506823976 | 0.02506823976 | 0.02506823976 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.0676304951 | 0.0676304951 | 0.0676304951 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.02959376612 | 0.02959376612 | 0.02959376612 | 10000 | 10000 | 10000 |
| img_mse | 0.01162612444 | 0.01162612444 | 0.01162612444 | 10000 | 10000 | 10000 |
| img_visible_l1 | 7.982248354e-05 | 7.982248354e-05 | 7.982248354e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 7.982054639e-09 | 7.982054639e-09 | 7.982054639e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3348076046 | 0.3348076046 | 0.3348076046 | 10000 | 10000 | 10000 |
| pix_var | 0.05713506788 | 0.05713506788 | 0.05713506788 | 10000 | 10000 | 10000 |
| psnr | 20.11732049 | 20.11732049 | 20.11732049 | 10000 | 10000 | 10000 |
| ssim | 0.932092064 | 0.932092064 | 0.932092064 | 10000 | 10000 | 10000 |

**仅残缺音频 (`corrupt_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 45.8800% | 45.8800% | 45.8800% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.003367625887 | 0.004416854337 | 0.005389572685 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.600040829 | 0.6441019821 | 0.6919345767 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.03934193233 | 0.03829270391 | 0.0373199852 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.316932072 | 0.4181916834 | 0.4636037961 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.03977006787 | 0.03913834654 | 0.03842388616 | 9996 | 9996 | 9996 |
| aud2img_same_damage | 0.0004327289243 | 0.0008500783174 | 0.001107990297 | 9996 | 9996 | 9996 |
| aud2img_win_both | 50.6700% | 54.5000% | 50.0300% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 59.4300% | 60.9700% | 59.6100% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 87.4500% | 90.8900% | 86.2200% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 0.0004255994499 | 0.0009209498582 | 0.001203935171 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.03976753154 | 0.03921365359 | 0.03852392037 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.04270955797 | 0.04270955797 | 0.04270955797 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.07526028515 | 0.07409888403 | 0.0741106366 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0268056334 | 0.02672961808 | 0.02700211719 | 10000 | 10000 | 10000 |
| aud_mse | 0.0108897884 | 0.01085890736 | 0.01096961021 | 10000 | 10000 | 10000 |
| aud_ssim | 0.5384149365 | 0.5397922348 | 0.5390261048 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 46.5600% | 46.9100% | 46.9800% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 46.5686% | 46.9188% | 46.9888% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 46.5600% | 46.9100% | 46.9800% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 46.5600% | 46.9100% | 46.9800% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 40.1900% | 40.5000% | 39.3100% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 40.4762% | 40.7163% | 40.4562% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 40.5100% | 40.6000% | 39.7300% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 40.7700% | 40.7700% | 40.7700% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.03934193233 | 0.03829270391 | 0.0373199852 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.07041732078 | 0.07098338275 | 0.07273483686 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.03934193233 | 0.03829270391 | 0.0373199852 | 10000 | 10000 | 10000 |
| img_mse | 0.03934193166 | 0.03829270253 | 0.03731998438 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.1800100356 | 0.1771179438 | 0.1747694761 | 10000 | 10000 | 10000 |
| pix_var | 0.02153018862 | 0.02092609182 | 0.0202894453 | 10000 | 10000 | 10000 |
| psnr | 17.78878238 | 17.92227048 | 17.88179701 | 10000 | 10000 | 10000 |
| ssim | 0.6312104433 | 0.6495516552 | 0.6625311481 | 10000 | 10000 | 10000 |

**干净图像+残缺音频 (`clean_img_corrupt_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 98.0700% | 98.0700% | 98.0700% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.07094715257 | 0.07001101228 | 0.06928070993 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.02395990073 | 0.02407071021 | 0.02490443355 | 10000 | 10000 | 10000 |
| aud_mse | 0.009733709633 | 0.009778726122 | 0.01011742623 | 10000 | 10000 | 10000 |
| aud_ssim | 0.5766152874 | 0.5759790228 | 0.576913965 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 88.6300% | 88.9700% | 89.4900% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 88.4854% | 88.8555% | 89.4258% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 88.2900% | 88.9400% | 89.3500% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 88.2800% | 88.8400% | 89.2200% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.0400% | 97.0400% | 97.0400% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.0388% | 97.0388% | 97.0388% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.0400% | 97.0400% | 97.0400% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.0400% | 97.0400% | 97.0400% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0003408268514 | 4.721179411e-05 | -0.0003892705949 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9452833536 | 0.9098243631 | 0.9245236612 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.02395990073 | 0.02407071021 | 0.02490443355 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.003020593762 | 0.003571298747 | 0.003341565092 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.02395719769 | 0.0240687161 | 0.02490206588 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -2.8497935e-06 | -1.961198982e-06 | -1.708196373e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 36.4500% | 32.5000% | 26.9200% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 50.1600% | 50.6600% | 50.2100% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 72.4400% | 60.1500% | 43.9400% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | -1.152974996e-06 | 6.469710357e-06 | 8.316619811e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.02395874771 | 0.02407718008 | 0.02491275021 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0243007276 | 0.02411792193 | 0.02451516308 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009948950207 | 0.009948950207 | 0.009948950207 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.342538029 | 0.342538029 | 0.342538029 | 10000 | 10000 | 10000 |
| pix_var | 0.06032439694 | 0.06032439694 | 0.06032439694 | 10000 | 10000 | 10000 |
| psnr | 20.70150414 | 20.70150414 | 20.70150414 | 10000 | 10000 | 10000 |
| ssim | 0.9421690227 | 0.9421690227 | 0.9421690227 | 10000 | 10000 | 10000 |

**残缺图像+干净音频 (`corrupt_img_clean_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1800% | 99.1800% | 99.1800% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0004825082848 | 0.0006273822494 | 0.0004151761797 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6175967471 | 0.6609685463 | 0.6985430444 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.02847314928 | 0.02832827531 | 0.02854048129 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.3366720876 | 0.4467068217 | 0.4700327413 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.02847759685 | 0.02833193207 | 0.02854397241 | 9996 | 9996 | 9996 |
| aud2img_same_damage | 5.85602491e-06 | 5.105417348e-06 | 4.510465642e-06 | 9996 | 9996 | 9996 |
| aud2img_win_both | 37.1800% | 40.1000% | 36.3800% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 54.6600% | 54.7900% | 54.0700% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 63.8300% | 69.0200% | 62.2800% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 6.644569186e-05 | 7.192524979e-05 | 7.340105171e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.02853959492 | 0.02840020045 | 0.02861388248 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.0289556576 | 0.0289556576 | 0.0289556576 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003501447439 | 0.001021526249 | 0.00102023107 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115525736 | 0.9707092572 | 0.9708219679 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.4800% | 97.0500% | 96.7300% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.4786% | 97.0488% | 96.7287% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.4800% | 97.0500% | 96.7300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.4800% | 97.0500% | 96.7300% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.1800% | 96.0400% | 96.1800% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.2485% | 96.1084% | 96.0684% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.0100% | 96.1800% | 96.0600% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.2800% | 96.2800% | 96.2800% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.04212021937 | 0.04159533229 | 0.04072402384 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.02460565225 | 0.02460565225 | 0.02460565225 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.0670605459 | 0.06654748895 | 0.06530352464 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.02847314928 | 0.02832827531 | 0.02854048129 | 10000 | 10000 | 10000 |
| img_mse | 0.01118588228 | 0.01112896744 | 0.01121233414 | 10000 | 10000 | 10000 |
| img_visible_l1 | 7.982248354e-05 | 7.982248354e-05 | 7.982248354e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 7.982054639e-09 | 7.982054639e-09 | 7.982054639e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3345243335 | 0.3348597288 | 0.3363880217 | 10000 | 10000 | 10000 |
| pix_var | 0.05700721964 | 0.05711578205 | 0.05762873963 | 10000 | 10000 | 10000 |
| psnr | 20.27368851 | 20.30541603 | 20.31479204 | 10000 | 10000 | 10000 |
| ssim | 0.9346308561 | 0.9349952307 | 0.9346669648 | 10000 | 10000 | 10000 |

**双残缺 (`corrupt_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 93.9600% | 93.9600% | 93.9600% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0001959937173 | 0.0003263510766 | 0.0002861304407 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6149195808 | 0.6563676734 | 0.6946499985 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.02919278086 | 0.02906242355 | 0.02910264412 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.1716297723 | 0.2227913448 | 0.2446037555 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.02917218927 | 0.02905050597 | 0.02909154819 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -1.922587916e-05 | -1.056770116e-05 | -9.890147984e-06 | 9996 | 9996 | 9996 |
| aud2img_win_both | 27.5500% | 29.6100% | 30.5600% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 43.7200% | 44.4500% | 44.8800% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 54.3200% | 58.0800% | 57.6900% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | -6.204077112e-06 | 8.674171148e-06 | 2.462085185e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.0291865769 | 0.02907109779 | 0.02912726498 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.02938877462 | 0.02938877462 | 0.02938877462 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.07223833251 | 0.0710341591 | 0.07040539296 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.02462949918 | 0.02464272563 | 0.02539641225 | 10000 | 10000 | 10000 |
| aud_mse | 0.0100057341 | 0.01001110737 | 0.01031729248 | 10000 | 10000 | 10000 |
| aud_ssim | 0.568627318 | 0.5686294236 | 0.5692397774 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 81.8000% | 82.8100% | 83.3600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 81.8427% | 82.9132% | 83.5434% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 81.8000% | 82.5600% | 83.4000% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 81.9600% | 82.8000% | 83.2100% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 94.7600% | 94.6600% | 94.8200% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 94.6679% | 94.7479% | 94.7179% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 94.6300% | 94.6900% | 94.5300% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 94.5100% | 94.5100% | 94.5100% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0002108905587 | -3.271036143e-05 | -0.0003774330216 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9440092066 | 0.9096574726 | 0.9237992979 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.02462949918 | 0.02464272563 | 0.02539641225 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.002301689109 | 0.002687014375 | 0.002514761809 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0246264849 | 0.02464145167 | 0.025395143 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -2.941510186e-06 | -1.172726107e-06 | -7.022833542e-07 | 9996 | 9996 | 9996 |
| img2aud_win_both | 34.6000% | 29.0200% | 24.1400% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 50.0300% | 50.7900% | 50.1700% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 68.4100% | 54.9000% | 42.1000% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | -3.888677689e-07 | 2.827502857e-06 | 3.486141551e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0246291104 | 0.02464555314 | 0.0253998984 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.02484038967 | 0.02461001513 | 0.02501897921 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.04140156307 | 0.04135269203 | 0.0410759243 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.02382140907 | 0.02382140907 | 0.02382140907 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.06924787614 | 0.06899124789 | 0.06872203923 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.02919278086 | 0.02906242355 | 0.02910264412 | 10000 | 10000 | 10000 |
| img_mse | 0.0114685946 | 0.01141738282 | 0.01143318379 | 10000 | 10000 | 10000 |
| img_visible_l1 | 7.98185515e-05 | 7.98185515e-05 | 7.98185515e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 7.981661896e-09 | 7.981661896e-09 | 7.981661896e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3352610469 | 0.3350937963 | 0.3353635371 | 10000 | 10000 | 10000 |
| pix_var | 0.05731028318 | 0.0572594516 | 0.05735629797 | 10000 | 10000 | 10000 |
| psnr | 20.15269634 | 20.17537556 | 20.17959635 | 10000 | 10000 | 10000 |
| ssim | 0.9332381775 | 0.9335279154 | 0.933493727 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>fixed_mask / mask_vertical / feature_dropout：全部原始指标</summary>

**干净双模态 (`clean_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1800% | 99.1800% | 99.1800% | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003503285427 | 0.001023170835 | 0.001022163968 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115226718 | 0.9706698141 | 0.9707627995 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.9588% | 97.0688% | 96.8387% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.7891% | 97.7891% | 97.7891% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009732332389 | 0.009732332389 | 0.009732332389 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3373669684 | 0.3373669684 | 0.3373669684 | 10000 | 10000 | 10000 |
| pix_var | 0.05885057896 | 0.05885057896 | 0.05885057896 | 10000 | 10000 | 10000 |
| psnr | 20.79410919 | 20.79410919 | 20.79410919 | 10000 | 10000 | 10000 |
| ssim | 0.9426845028 | 0.9426845028 | 0.9426845028 | 10000 | 10000 | 10000 |

**仅干净图像 (`clean_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 97.7000% | 97.7000% | 97.7000% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.002905382487 | 0.002580899412 | 0.0025012506 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| aud_mse | 0.0002402440196 | 0.0002218772086 | 0.0002282006215 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9736808665 | 0.9763999848 | 0.976126532 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.1300% | 96.5000% | 96.4000% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.1485% | 96.5186% | 96.4786% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.1000% | 96.4500% | 96.4300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.1500% | 96.6200% | 96.4500% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.8187% | 96.8187% | 96.8187% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 2.120906993e-05 | 2.734319982e-05 | 2.90869362e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9636753441 | 0.9270737396 | 0.9410639594 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.05812706452 | 0.06269425778 | 0.05521903026 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0002405052923 | 0.0002228766707 | 0.0002300969015 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 2.011856675e-07 | 9.414399495e-07 | 1.835515222e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 79.8300% | 82.5600% | 85.4500% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 82.8300% | 86.0600% | 88.3200% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 95.0400% | 94.6400% | 95.0200% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 7.060236394e-06 | 7.88233878e-06 | 1.064793261e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0002473042549 | 0.0002297595468 | 0.0002388485523 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0002614530888 | 0.0002492204076 | 0.0002572875584 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009777385685 | 0.009777385685 | 0.009777385685 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3410749137 | 0.3410749137 | 0.3410749137 | 10000 | 10000 | 10000 |
| pix_var | 0.05938630551 | 0.05938630551 | 0.05938630551 | 10000 | 10000 | 10000 |
| psnr | 20.77137023 | 20.77137023 | 20.77137023 | 10000 | 10000 | 10000 |
| ssim | 0.9426458778 | 0.9426458778 | 0.9426458778 | 10000 | 10000 | 10000 |

**仅干净音频 (`clean_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 98.8300% | 98.8300% | 98.8300% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 4.550894817e-05 | 5.925099576e-05 | 3.427388306e-05 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5948837502 | 0.6400160534 | 0.6886220585 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.6138913798 | 0.8292583166 | 0.855119685 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.002024612459 | 0.00200570006 | 0.002030547302 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -1.557539837e-05 | -2.074199916e-05 | -2.087765998e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 40.8600% | 51.8800% | 43.2100% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 71.3800% | 79.1800% | 76.4900% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 51.6500% | 62.1900% | 52.4400% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 3.748856293e-06 | 1.19613159e-05 | 1.177280845e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.002043171231 | 0.00203764165 | 0.002062430253 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.002084931313 | 0.002084931313 | 0.002084931313 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003496576117 | 0.001021110903 | 0.001020037457 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115410681 | 0.9707879167 | 0.9709046022 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 95.5282% | 96.5886% | 96.5586% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7300% | 98.0900% | 97.3500% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.4690% | 97.7491% | 97.6491% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.5200% | 97.7800% | 97.6700% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7100% | 97.7100% | 97.7100% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.007310281691 | 0.007187631063 | 0.007240436512 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_mse | 0.002039422297 | 0.002025680222 | 0.002050657357 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2853073776 | 0.2854863703 | 0.2857626975 | 10000 | 10000 | 10000 |
| pix_var | 0.0461221002 | 0.04620051384 | 0.04626523703 | 10000 | 10000 | 10000 |
| psnr | 37.44974285 | 37.68731136 | 37.47306895 | 10000 | 10000 | 10000 |
| ssim | 0.9860573347 | 0.9861481328 | 0.9859331862 | 10000 | 10000 | 10000 |

**仅残缺图像 (`corrupt_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 94.5300% | 94.5300% | 94.5300% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.003811176671 | 0.003450530321 | 0.003361942034 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0004383070039 | 0.00041632913 | 0.0004301401492 | 10000 | 10000 | 10000 |
| aud_mse | 0.0004383070055 | 0.0004163291255 | 0.0004301401479 | 10000 | 10000 | 10000 |
| aud_ssim | 0.94991475 | 0.9532160601 | 0.9528345764 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 91.2100% | 91.7500% | 91.9100% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 91.2765% | 91.9968% | 91.8968% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 91.0800% | 91.8800% | 91.8500% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 91.2800% | 92.0000% | 91.8000% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 95.4300% | 95.4300% | 95.4300% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 95.4282% | 95.4282% | 95.4282% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 95.4300% | 95.4300% | 95.4300% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 95.4300% | 95.4300% | 95.4300% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 1.551439727e-05 | 1.728220216e-05 | 1.532430831e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9626154993 | 0.9266807708 | 0.9403615976 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.0004383070039 | 0.00041632913 | 0.0004301401492 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.04868245244 | 0.05286279396 | 0.0464397868 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0004383655127 | 0.0004170708804 | 0.0004320377565 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -7.636871116e-08 | 6.080002547e-07 | 1.757195862e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 72.4200% | 76.6700% | 79.1800% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 78.6400% | 82.3000% | 84.0000% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 89.5600% | 90.0900% | 90.9800% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 3.923928417e-06 | 4.595617019e-06 | 6.435089925e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0004422309332 | 0.0004209247418 | 0.0004365752364 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0004538214013 | 0.0004336113281 | 0.000445464455 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.03150895892 | 0.03150895892 | 0.03150895892 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.0195624702 | 0.0195624702 | 0.0195624702 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.04676557893 | 0.04676557893 | 0.04676557893 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.01998888024 | 0.01998888024 | 0.01998888024 | 10000 | 10000 | 10000 |
| img_mse | 0.007852777302 | 0.007852777302 | 0.007852777302 | 10000 | 10000 | 10000 |
| img_visible_l1 | 7.75976398e-05 | 7.75976398e-05 | 7.75976398e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 7.759548365e-09 | 7.759548365e-09 | 7.759548365e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3422250748 | 0.3422250748 | 0.3422250748 | 10000 | 10000 | 10000 |
| pix_var | 0.06021087617 | 0.06021087617 | 0.06021087617 | 10000 | 10000 | 10000 |
| psnr | 24.24401786 | 24.24401786 | 24.24401786 | 10000 | 10000 | 10000 |
| ssim | 0.9563879644 | 0.9563879644 | 0.9563879644 | 10000 | 10000 | 10000 |

**仅残缺音频 (`corrupt_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 97.2300% | 97.2300% | 97.2300% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 7.701876803e-05 | 9.978611251e-05 | 7.264250115e-05 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5932294149 | 0.6365840066 | 0.6860371544 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.002900950903 | 0.002878183558 | 0.00290532718 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.5608093807 | 0.7576825909 | 0.7774535178 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.002881880465 | 0.002854349496 | 0.002886606147 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -2.015601017e-05 | -2.491397256e-05 | -1.980940936e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 38.3400% | 47.4600% | 38.7300% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 66.1200% | 71.8800% | 67.4400% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 52.0900% | 63.7400% | 51.5300% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | -3.870412477e-06 | -1.099342946e-07 | 3.959868355e-06 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.002897080472 | 0.002878073616 | 0.002909287025 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.002977969647 | 0.002977969647 | 0.002977969647 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.01583932369 | 0.01280380886 | 0.01287571787 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.003699577341 | 0.002370328187 | 0.002394690873 | 10000 | 10000 | 10000 |
| aud_mse | 0.001479476427 | 0.0009480306027 | 0.0009577870566 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9602550959 | 0.9737540712 | 0.9734532598 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 97.0300% | 97.4600% | 97.4300% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 97.0288% | 97.4590% | 97.4290% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 97.0300% | 97.4600% | 97.4300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 97.0300% | 97.4600% | 97.4300% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.8600% | 96.8900% | 96.8800% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.8988% | 96.8888% | 96.9188% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.8800% | 96.8500% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.8500% | 96.8500% | 96.8500% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.002900950903 | 0.002878183558 | 0.00290532718 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.009330742572 | 0.009184590533 | 0.009270244902 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.002900950903 | 0.002878183558 | 0.00290532718 | 10000 | 10000 | 10000 |
| img_mse | 0.002900950804 | 0.002878183479 | 0.002905327102 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2827616036 | 0.2827438414 | 0.2831624746 | 10000 | 10000 | 10000 |
| pix_var | 0.0453363955 | 0.04535669088 | 0.04545391351 | 10000 | 10000 | 10000 |
| psnr | 36.04240825 | 36.2904348 | 36.0240836 | 10000 | 10000 | 10000 |
| ssim | 0.9797209461 | 0.9798339647 | 0.979629397 | 10000 | 10000 | 10000 |

**干净图像+残缺音频 (`clean_img_corrupt_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.3100% | 99.3100% | 99.3100% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.01507214698 | 0.01180888142 | 0.01195586268 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.003284703045 | 0.001973935867 | 0.002046996861 | 10000 | 10000 | 10000 |
| aud_mse | 0.001313599583 | 0.0007895174518 | 0.000818754537 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9638010433 | 0.9774813473 | 0.976667665 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 97.4400% | 97.8400% | 97.6800% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 97.5090% | 97.7591% | 97.6791% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 97.4200% | 97.8300% | 97.7600% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 97.3200% | 97.6500% | 97.6100% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.6100% | 97.6100% | 97.6100% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.6090% | 97.6090% | 97.6090% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.6100% | 97.6100% | 97.6100% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.6100% | 97.6100% | 97.6100% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0004139234483 | 0.0003965792266 | 0.0003474383007 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.954477431 | 0.9178040148 | 0.9312438955 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.003284703045 | 0.001973935867 | 0.002046996861 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.03734643748 | 0.04005886077 | 0.03516830056 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.003285316373 | 0.001974423163 | 0.002047409486 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 2.368099519e-07 | 3.328550374e-07 | 2.647335785e-07 | 9996 | 9996 | 9996 |
| img2aud_win_both | 47.9100% | 51.0800% | 50.3200% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 50.1000% | 52.9400% | 52.8500% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 93.8500% | 95.1700% | 93.4100% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | -1.237589887e-05 | 8.732379792e-08 | 8.253225111e-07 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.003272327141 | 0.001974023212 | 0.002047822194 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.003698626472 | 0.002370515106 | 0.002394435181 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009733007652 | 0.009733007652 | 0.009733007652 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3371500075 | 0.3371500075 | 0.3371500075 | 10000 | 10000 | 10000 |
| pix_var | 0.05896090344 | 0.05896090344 | 0.05896090344 | 10000 | 10000 | 10000 |
| psnr | 20.79325094 | 20.79325094 | 20.79325094 | 10000 | 10000 | 10000 |
| ssim | 0.942749668 | 0.942749668 | 0.942749668 | 10000 | 10000 | 10000 |

**残缺图像+干净音频 (`corrupt_img_clean_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1300% | 99.1300% | 99.1300% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0004071670754 | 0.0005435705693 | 0.0003814672264 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6217618739 | 0.6668134611 | 0.7099768436 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.01910534524 | 0.01896894197 | 0.0191310451 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.3477137192 | 0.4641274428 | 0.488587981 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.01910394293 | 0.01896840687 | 0.0191344855 | 9996 | 9996 | 9996 |
| aud2img_same_damage | 1.289046408e-06 | 2.129242776e-06 | 6.206566631e-06 | 9996 | 9996 | 9996 |
| aud2img_win_both | 35.7400% | 37.2200% | 35.5000% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 52.5500% | 52.4200% | 52.2400% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 61.0200% | 64.5900% | 60.2900% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 5.271982566e-05 | 5.723728233e-05 | 7.364103876e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.01915806506 | 0.01902617907 | 0.01920468612 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.0195125124 | 0.0195125124 | 0.0195125124 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003501021862 | 0.001021534997 | 0.001020356122 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115664515 | 0.9706995981 | 0.9707976408 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.7800% | 96.9900% | 96.7300% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.7787% | 96.9888% | 96.7287% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.7800% | 96.9900% | 96.7300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.7800% | 96.9900% | 96.7300% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.6900% | 96.5400% | 96.6700% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.5786% | 96.6887% | 96.6687% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.6600% | 96.6100% | 96.5600% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.7400% | 96.7400% | 96.7400% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.03070584329 | 0.03022716106 | 0.02980108343 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.01933980342 | 0.01933980342 | 0.01933980342 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.0466015941 | 0.04634610553 | 0.04562942518 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.01910534524 | 0.01896894197 | 0.0191310451 | 10000 | 10000 | 10000 |
| img_mse | 0.00750567438 | 0.007452087301 | 0.007515770732 | 10000 | 10000 | 10000 |
| img_visible_l1 | 7.75976398e-05 | 7.75976398e-05 | 7.75976398e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 7.759548365e-09 | 7.759548365e-09 | 7.759548365e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3422724605 | 0.342348367 | 0.343066901 | 10000 | 10000 | 10000 |
| pix_var | 0.06024320796 | 0.06026275828 | 0.06052107364 | 10000 | 10000 | 10000 |
| psnr | 24.41530656 | 24.48032501 | 24.4933081 | 10000 | 10000 | 10000 |
| ssim | 0.958357748 | 0.9586821317 | 0.958381815 | 10000 | 10000 | 10000 |

**双残缺 (`corrupt_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1500% | 99.1500% | 99.1500% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0004220186479 | 0.0005460827984 | 0.0004290122056 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6198184256 | 0.6630303761 | 0.7072435339 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.01899967515 | 0.01887561094 | 0.01899268165 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.3204191099 | 0.4291943167 | 0.4482163449 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.01899604329 | 0.01887283085 | 0.01899580909 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -9.978982564e-07 | -1.446761544e-07 | 5.866948333e-06 | 9996 | 9996 | 9996 |
| aud2img_win_both | 34.9600% | 36.3500% | 35.3100% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 51.3800% | 51.5300% | 51.4500% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 60.9100% | 63.7700% | 60.6600% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 4.14312678e-05 | 4.243989458e-05 | 5.570218859e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.01904110646 | 0.01891805119 | 0.01904838379 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.01942169369 | 0.01942169369 | 0.01942169369 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.01521085984 | 0.01194527182 | 0.01208894789 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.003360194002 | 0.002033841617 | 0.002100092085 | 10000 | 10000 | 10000 |
| aud_mse | 0.001343944363 | 0.000813520997 | 0.0008400163684 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9630590771 | 0.9768650753 | 0.9761089613 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 97.3100% | 97.6400% | 97.6600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 97.3289% | 97.5790% | 97.5690% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 97.3100% | 97.6200% | 97.6000% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 97.2300% | 97.5500% | 97.4100% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.6400% | 96.5600% | 96.6900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.5786% | 96.6086% | 96.6186% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.6300% | 96.6500% | 96.6000% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.6000% | 96.6000% | 96.6000% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.000339619421 | 0.0003375786696 | 0.0002926248804 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9538035017 | 0.9179037086 | 0.9310949402 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.003360194002 | 0.002033841617 | 0.002100092085 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.03131898697 | 0.03378072627 | 0.02960470315 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.003361157544 | 0.002034160938 | 0.00210033068 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 4.938347822e-07 | 9.741753919e-08 | 2.161461217e-08 | 9996 | 9996 | 9996 |
| img2aud_win_both | 46.9300% | 49.3100% | 48.5300% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 49.3600% | 51.1600% | 51.3100% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 93.5900% | 95.3200% | 93.1100% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | -1.215883166e-05 | -2.816382227e-06 | -1.796247676e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.003348035178 | 0.002031025214 | 0.002098295847 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.003699813426 | 0.002371420284 | 0.002392716985 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.03092079527 | 0.03051720037 | 0.03001220628 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.01935330717 | 0.01935330717 | 0.01935330717 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.04652890924 | 0.04640482248 | 0.04575806869 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.01899967515 | 0.01887561094 | 0.01899268165 | 10000 | 10000 | 10000 |
| img_mse | 0.007464161095 | 0.007415421622 | 0.007461413616 | 10000 | 10000 | 10000 |
| img_visible_l1 | 7.75696178e-05 | 7.75696178e-05 | 7.75696178e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 7.756745671e-09 | 7.756745671e-09 | 7.756745671e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3427125812 | 0.3429273665 | 0.3435639441 | 10000 | 10000 | 10000 |
| pix_var | 0.06012714654 | 0.06019291282 | 0.06042471528 | 10000 | 10000 | 10000 |
| psnr | 24.44094405 | 24.50774389 | 24.52356365 | 10000 | 10000 | 10000 |
| ssim | 0.958757481 | 0.9590626469 | 0.9588653486 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>fixed_mask / occlusion / time_mask：全部原始指标</summary>

**干净双模态 (`clean_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1800% | 99.1800% | 99.1800% | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003503285427 | 0.001023170835 | 0.001022163968 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115226718 | 0.9706698141 | 0.9707627995 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.9588% | 97.0688% | 96.8387% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.7891% | 97.7891% | 97.7891% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009732332389 | 0.009732332389 | 0.009732332389 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3393525779 | 0.3393525779 | 0.3393525779 | 10000 | 10000 | 10000 |
| pix_var | 0.05885057896 | 0.05885057896 | 0.05885057896 | 10000 | 10000 | 10000 |
| psnr | 20.79410919 | 20.79410919 | 20.79410919 | 10000 | 10000 | 10000 |
| ssim | 0.9426845028 | 0.9426845028 | 0.9426845028 | 10000 | 10000 | 10000 |

**仅干净图像 (`clean_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 97.7000% | 97.7000% | 97.7000% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.002905382487 | 0.002580899412 | 0.0025012506 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| aud_mse | 0.0002402440196 | 0.0002218772086 | 0.0002282006215 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9736808665 | 0.9763999848 | 0.976126532 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.1300% | 96.5000% | 96.4000% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.1485% | 96.5186% | 96.4786% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.1000% | 96.4500% | 96.4300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.1500% | 96.6200% | 96.4500% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.8187% | 96.8187% | 96.8187% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 2.120906993e-05 | 2.734319982e-05 | 2.90869362e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9636753441 | 0.9270737396 | 0.9410639594 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.05812706452 | 0.06269425778 | 0.05521903026 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0002405052923 | 0.0002228766707 | 0.0002300969015 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 2.011856675e-07 | 9.414399495e-07 | 1.835515222e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 79.8300% | 82.5600% | 85.4500% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 82.8300% | 86.0600% | 88.3200% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 95.0400% | 94.6400% | 95.0200% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 7.060236394e-06 | 7.88233878e-06 | 1.064793261e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0002473042549 | 0.0002297595468 | 0.0002388485523 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0002614530888 | 0.0002492204076 | 0.0002572875584 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009777385685 | 0.009777385685 | 0.009777385685 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3388502598 | 0.3388502598 | 0.3388502598 | 10000 | 10000 | 10000 |
| pix_var | 0.05938630551 | 0.05938630551 | 0.05938630551 | 10000 | 10000 | 10000 |
| psnr | 20.77137023 | 20.77137023 | 20.77137023 | 10000 | 10000 | 10000 |
| ssim | 0.9426458778 | 0.9426458778 | 0.9426458778 | 10000 | 10000 | 10000 |

**仅干净音频 (`clean_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 98.8300% | 98.8300% | 98.8300% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 4.550894817e-05 | 5.925099576e-05 | 3.427388306e-05 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5948837502 | 0.6400160534 | 0.6886220585 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.6138913798 | 0.8292583166 | 0.855119685 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.002024612459 | 0.00200570006 | 0.002030547302 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -1.557539837e-05 | -2.074199916e-05 | -2.087765998e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 40.8600% | 51.8800% | 43.2100% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 71.3800% | 79.1800% | 76.4900% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 51.6500% | 62.1900% | 52.4400% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 3.748856293e-06 | 1.19613159e-05 | 1.177280845e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.002043171231 | 0.00203764165 | 0.002062430253 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.002084931313 | 0.002084931313 | 0.002084931313 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003496576117 | 0.001021110903 | 0.001020037457 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115410681 | 0.9707879167 | 0.9709046022 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 95.5282% | 96.5886% | 96.5586% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7300% | 98.0900% | 97.3500% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.4690% | 97.7491% | 97.6491% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.5200% | 97.7800% | 97.6700% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7100% | 97.7100% | 97.7100% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.007310281691 | 0.007187631063 | 0.007240436512 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_mse | 0.002039422297 | 0.002025680222 | 0.002050657357 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2865543067 | 0.2867441773 | 0.2870205343 | 10000 | 10000 | 10000 |
| pix_var | 0.0461221002 | 0.04620051384 | 0.04626523703 | 10000 | 10000 | 10000 |
| psnr | 37.44974285 | 37.68731136 | 37.47306895 | 10000 | 10000 | 10000 |
| ssim | 0.9860573347 | 0.9861481328 | 0.9859331862 | 10000 | 10000 | 10000 |

**仅残缺图像 (`corrupt_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 89.7900% | 89.7900% | 89.7900% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.005045679452 | 0.00467246092 | 0.004587584142 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0007448691032 | 0.0007160181351 | 0.0007427234123 | 10000 | 10000 | 10000 |
| aud_mse | 0.000744869101 | 0.0007160181413 | 0.0007427234105 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9135811641 | 0.9179264864 | 0.916845718 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 85.3500% | 86.3000% | 86.3800% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 85.4842% | 86.5046% | 86.3545% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 85.3700% | 86.4200% | 86.3500% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 85.5900% | 86.5600% | 86.3300% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 93.7500% | 93.7500% | 93.7500% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 93.7475% | 93.7475% | 93.7475% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 93.7500% | 93.7500% | 93.7500% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 93.7500% | 93.7500% | 93.7500% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 1.884364179e-05 | 1.713768141e-05 | 9.501509683e-06 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9630876428 | 0.9268135476 | 0.940565112 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.0007448691032 | 0.0007160181351 | 0.0007427234123 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.04764706746 | 0.05119685192 | 0.04497674026 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0007452205916 | 0.0007174459805 | 0.0007465518927 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 8.639172673e-07 | 1.992022128e-06 | 4.351196872e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 65.7900% | 67.5100% | 70.3000% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 74.8500% | 76.9500% | 78.2600% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 84.1400% | 84.0700% | 85.2100% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 4.404396722e-06 | 5.218653093e-06 | 7.648912751e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0007492734954 | 0.0007212367894 | 0.0007503723292 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0007637127385 | 0.0007331558191 | 0.0007522249186 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.1027959726 | 0.1027959726 | 0.1027959726 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.01561249283 | 0.01561249283 | 0.01561249283 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.1316692386 | 0.1316692386 | 0.1316692386 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.05934831666 | 0.05934831666 | 0.05934831666 | 10000 | 10000 | 10000 |
| img_mse | 0.009159630261 | 0.009159630261 | 0.009159630261 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.420416259e-05 | 8.420416259e-05 | 8.420416259e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.420264396e-09 | 8.420264396e-09 | 8.420264396e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3422829211 | 0.3422829211 | 0.3422829211 | 10000 | 10000 | 10000 |
| pix_var | 0.05982412025 | 0.05982412025 | 0.05982412025 | 10000 | 10000 | 10000 |
| psnr | 22.86851887 | 22.86851887 | 22.86851887 | 10000 | 10000 | 10000 |
| ssim | 0.9455200526 | 0.9455200526 | 0.9455200526 | 10000 | 10000 | 10000 |

**仅残缺音频 (`corrupt_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 90.5000% | 90.5000% | 90.5000% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0003436827168 | 0.0004258857787 | 0.0004344832831 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5938334194 | 0.6372512712 | 0.6863718651 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.008705206656 | 0.00862300362 | 0.008614406067 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.5220929265 | 0.7044291528 | 0.7250470215 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.008684310348 | 0.008658704611 | 0.008679643238 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -2.150764351e-05 | 3.511616935e-05 | 6.469111774e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 41.7600% | 49.5000% | 42.7200% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 64.3800% | 68.4300% | 67.1600% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 59.3300% | 68.7400% | 57.7300% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | -8.406286867e-06 | 4.842653425e-05 | 8.511574579e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.008696800396 | 0.008671430084 | 0.008699521914 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.009048889524 | 0.009048889524 | 0.009048889524 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02342858999 | 0.02297728202 | 0.02289740709 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.008044499707 | 0.007772397104 | 0.007845177482 | 10000 | 10000 | 10000 |
| aud_mse | 0.003268077976 | 0.003157536332 | 0.003187103355 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9108133539 | 0.9137452469 | 0.9140872436 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 89.3200% | 89.4700% | 89.8300% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 89.3157% | 89.4658% | 89.8259% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 89.3200% | 89.4700% | 89.8300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 89.3200% | 89.4700% | 89.8300% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 89.1400% | 89.1700% | 89.1400% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 89.1557% | 89.2557% | 89.2257% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 89.1300% | 89.0800% | 89.0700% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 89.0600% | 89.0600% | 89.0600% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.008705206656 | 0.00862300362 | 0.008614406067 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.01923546729 | 0.01913065741 | 0.01932394504 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.008705206656 | 0.00862300362 | 0.008614406067 | 10000 | 10000 | 10000 |
| img_mse | 0.0087052064 | 0.00862300328 | 0.008614405876 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2818480432 | 0.2816305757 | 0.2816714346 | 10000 | 10000 | 10000 |
| pix_var | 0.04212484509 | 0.04208213091 | 0.04208541289 | 10000 | 10000 | 10000 |
| psnr | 32.82428661 | 33.03493653 | 32.79241913 | 10000 | 10000 | 10000 |
| ssim | 0.9305089819 | 0.931302472 | 0.9315012836 | 10000 | 10000 | 10000 |

**干净图像+残缺音频 (`clean_img_corrupt_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1400% | 99.1400% | 99.1400% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02254827385 | 0.02204040097 | 0.02188411876 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.007449128635 | 0.007242267686 | 0.007450463033 | 10000 | 10000 | 10000 |
| aud_mse | 0.003026208534 | 0.002942171279 | 0.003026750606 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9160588418 | 0.918869541 | 0.9190117455 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 93.0600% | 93.5700% | 93.7800% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 92.9372% | 93.5774% | 93.9176% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 92.8000% | 93.6100% | 93.8100% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 92.8500% | 93.3300% | 93.5500% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.6100% | 97.6100% | 97.6100% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.6090% | 97.6090% | 97.6090% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.6100% | 97.6100% | 97.6100% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.6100% | 97.6100% | 97.6100% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0002130302425 | 0.0001110767513 | -2.319096126e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9625726442 | 0.9258362919 | 0.9387225227 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.007449128635 | 0.007242267686 | 0.007450463033 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.01713567408 | 0.01761723807 | 0.0161018443 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.007448195574 | 0.007241860928 | 0.007448650962 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -1.470493411e-06 | -8.363047669e-07 | -1.894268176e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 29.7900% | 28.3200% | 24.9500% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 35.7500% | 36.0500% | 35.7800% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 53.6300% | 51.0300% | 44.8100% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 7.345579463e-06 | 1.210953464e-05 | 1.85519969e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.007456474249 | 0.007254377259 | 0.007469014949 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.007662158912 | 0.007353344484 | 0.007427272083 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009767318833 | 0.009767318833 | 0.009767318833 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.339690268 | 0.339690268 | 0.339690268 | 10000 | 10000 | 10000 |
| pix_var | 0.05919039622 | 0.05919039622 | 0.05919039622 | 10000 | 10000 | 10000 |
| psnr | 20.77741999 | 20.77741999 | 20.77741999 | 10000 | 10000 | 10000 |
| ssim | 0.942667157 | 0.942667157 | 0.942667157 | 10000 | 10000 | 10000 |

**残缺图像+干净音频 (`corrupt_img_clean_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1300% | 99.1300% | 99.1300% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.001998503633 | 0.002437640028 | 0.002247507094 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6073615396 | 0.6468766301 | 0.6721185633 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.0549652799 | 0.05452614286 | 0.05471627646 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.1517502374 | 0.2026534835 | 0.2149826603 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.05495980068 | 0.05451023369 | 0.05470517478 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -2.259624099e-06 | -1.288031717e-05 | -8.819622546e-06 | 9996 | 9996 | 9996 |
| aud2img_win_both | 39.1000% | 40.5500% | 39.4500% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 52.4700% | 51.8900% | 52.9300% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 69.1800% | 73.5000% | 69.0600% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 0.00010803477 | 0.0001016694849 | 0.0001798142719 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.05507331485 | 0.0546278132 | 0.05489609061 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.05696378303 | 0.05696378303 | 0.05696378303 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003500739961 | 0.001020981195 | 0.001019750237 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115270317 | 0.9707132039 | 0.970818454 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.7400% | 97.0100% | 96.6400% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.7387% | 97.0088% | 96.6387% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.7400% | 97.0100% | 96.6400% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.7400% | 97.0100% | 96.6400% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.2400% | 96.2200% | 96.2700% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.2485% | 96.2185% | 96.2285% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.3500% | 96.2100% | 96.3500% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 95.9200% | 95.9200% | 95.9200% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.09786147797 | 0.09585471704 | 0.09357460346 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.015329687 | 0.015329687 | 0.015329687 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.1276205694 | 0.1265517003 | 0.1247284459 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.0549652799 | 0.05452614286 | 0.05471627646 | 10000 | 10000 | 10000 |
| img_mse | 0.008483166875 | 0.008415391946 | 0.008444736467 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.420416259e-05 | 8.420416259e-05 | 8.420416259e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.420264396e-09 | 8.420264396e-09 | 8.420264396e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3425923288 | 0.3428884745 | 0.3440980315 | 10000 | 10000 | 10000 |
| pix_var | 0.05992761254 | 0.06002291292 | 0.06044617295 | 10000 | 10000 | 10000 |
| psnr | 23.16550557 | 23.22456714 | 23.25470703 | 10000 | 10000 | 10000 |
| ssim | 0.9497622129 | 0.9502182844 | 0.9501723248 | 10000 | 10000 | 10000 |

**双残缺 (`corrupt_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 97.8500% | 97.8500% | 97.8500% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.001981065226 | 0.002359811446 | 0.002314563196 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6048811511 | 0.6425506818 | 0.6682702808 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.05459239092 | 0.05421364453 | 0.05425889239 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.127211702 | 0.1712031429 | 0.180392571 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.05454255207 | 0.05416103134 | 0.05419925391 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -2.801401331e-05 | -3.078040507e-05 | -3.946129705e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 38.7600% | 40.3900% | 39.1000% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 52.6800% | 52.7300% | 52.9100% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 68.0800% | 71.2200% | 68.4600% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 0.0001208175815 | 0.0001286350959 | 0.0001997547672 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.05471320827 | 0.05434227965 | 0.05445864735 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.05657345603 | 0.05657345603 | 0.05657345603 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02385285862 | 0.02323473512 | 0.02309007092 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.00790282775 | 0.007634090176 | 0.007831425075 | 10000 | 10000 | 10000 |
| aud_mse | 0.003210523818 | 0.003101349132 | 0.003181516488 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9094707897 | 0.913147192 | 0.913352055 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 91.6300% | 92.3500% | 92.6200% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 91.5766% | 92.2869% | 92.5670% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 91.5500% | 92.0900% | 92.5200% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 91.5100% | 92.2300% | 92.1800% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 95.6100% | 95.6700% | 95.7500% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 95.5782% | 95.3982% | 95.6883% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 95.6200% | 95.5700% | 95.5800% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 95.1400% | 95.1400% | 95.1400% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0001760049677 | 9.143412766e-05 | -2.799767614e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.961832164 | 0.9252690526 | 0.9380507574 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.00790282775 | 0.007634090176 | 0.007831425075 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.01393142235 | 0.01422720544 | 0.01299872201 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.007906899148 | 0.0076369377 | 0.007835082269 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 2.072505723e-06 | 8.998009877e-07 | 1.644023045e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 29.2700% | 29.0600% | 25.2900% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 35.5800% | 36.5300% | 35.6800% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 54.0100% | 51.7900% | 45.8800% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 6.892992757e-06 | 9.330021165e-06 | 1.320877729e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.007909720877 | 0.007643420197 | 0.007844633933 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.00807883275 | 0.007725524321 | 0.007803427455 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.09790662336 | 0.09638799579 | 0.09412681849 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.01519316517 | 0.01519316517 | 0.01519316517 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.1273387166 | 0.126686763 | 0.1250932738 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.05459239092 | 0.05421364453 | 0.05425889239 | 10000 | 10000 | 10000 |
| img_mse | 0.008425616361 | 0.008367161894 | 0.008374145375 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.414989689e-05 | 8.414989689e-05 | 8.414989689e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.414837291e-09 | 8.414837291e-09 | 8.414837291e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3406689465 | 0.3408475816 | 0.3418743312 | 10000 | 10000 | 10000 |
| pix_var | 0.05983707681 | 0.059897203 | 0.0602539815 | 10000 | 10000 | 10000 |
| psnr | 23.09987961 | 23.14882939 | 23.17875549 | 10000 | 10000 | 10000 |
| ssim | 0.949746816 | 0.9501711159 | 0.9502367862 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>fixed_mask / pixel_delete / freq_mask：全部原始指标</summary>

**干净双模态 (`clean_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1800% | 99.1800% | 99.1800% | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003503285427 | 0.001023170835 | 0.001022163968 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115226718 | 0.9706698141 | 0.9707627995 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.9588% | 97.0688% | 96.8387% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.7891% | 97.7891% | 97.7891% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009732332389 | 0.009732332389 | 0.009732332389 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3391851187 | 0.3391851187 | 0.3391851187 | 10000 | 10000 | 10000 |
| pix_var | 0.05885057896 | 0.05885057896 | 0.05885057896 | 10000 | 10000 | 10000 |
| psnr | 20.79410919 | 20.79410919 | 20.79410919 | 10000 | 10000 | 10000 |
| ssim | 0.9426845028 | 0.9426845028 | 0.9426845028 | 10000 | 10000 | 10000 |

**仅干净图像 (`clean_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 97.7000% | 97.7000% | 97.7000% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.002905382487 | 0.002580899412 | 0.0025012506 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| aud_mse | 0.0002402440196 | 0.0002218772086 | 0.0002282006215 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9736808665 | 0.9763999848 | 0.976126532 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.1300% | 96.5000% | 96.4000% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.1485% | 96.5186% | 96.4786% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.1000% | 96.4500% | 96.4300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.1500% | 96.6200% | 96.4500% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.8187% | 96.8187% | 96.8187% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 2.120906993e-05 | 2.734319982e-05 | 2.90869362e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9636753441 | 0.9270737396 | 0.9410639594 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.05812706452 | 0.06269425778 | 0.05521903026 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0002405052923 | 0.0002228766707 | 0.0002300969015 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 2.011856675e-07 | 9.414399495e-07 | 1.835515222e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 79.8300% | 82.5600% | 85.4500% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 82.8300% | 86.0600% | 88.3200% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 95.0400% | 94.6400% | 95.0200% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 7.060236394e-06 | 7.88233878e-06 | 1.064793261e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0002473042549 | 0.0002297595468 | 0.0002388485523 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0002614530888 | 0.0002492204076 | 0.0002572875584 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009777385685 | 0.009777385685 | 0.009777385685 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3398992717 | 0.3398992717 | 0.3398992717 | 10000 | 10000 | 10000 |
| pix_var | 0.05938630551 | 0.05938630551 | 0.05938630551 | 10000 | 10000 | 10000 |
| psnr | 20.77137023 | 20.77137023 | 20.77137023 | 10000 | 10000 | 10000 |
| ssim | 0.9426458778 | 0.9426458778 | 0.9426458778 | 10000 | 10000 | 10000 |

**仅干净音频 (`clean_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 98.8300% | 98.8300% | 98.8300% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 4.550894817e-05 | 5.925099576e-05 | 3.427388306e-05 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5948837502 | 0.6400160534 | 0.6886220585 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.6138913798 | 0.8292583166 | 0.855119685 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.002024612459 | 0.00200570006 | 0.002030547302 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -1.557539837e-05 | -2.074199916e-05 | -2.087765998e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 40.8600% | 51.8800% | 43.2100% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 71.3800% | 79.1800% | 76.4900% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 51.6500% | 62.1900% | 52.4400% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 3.748856293e-06 | 1.19613159e-05 | 1.177280845e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.002043171231 | 0.00203764165 | 0.002062430253 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.002084931313 | 0.002084931313 | 0.002084931313 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003496576117 | 0.001021110903 | 0.001020037457 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115410681 | 0.9707879167 | 0.9709046022 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 95.5282% | 96.5886% | 96.5586% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7300% | 98.0900% | 97.3500% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.4690% | 97.7491% | 97.6491% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.5200% | 97.7800% | 97.6700% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7100% | 97.7100% | 97.7100% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.007310281691 | 0.007187631063 | 0.007240436512 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_mse | 0.002039422297 | 0.002025680222 | 0.002050657357 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2891766727 | 0.2893635035 | 0.2896432877 | 10000 | 10000 | 10000 |
| pix_var | 0.0461221002 | 0.04620051384 | 0.04626523703 | 10000 | 10000 | 10000 |
| psnr | 37.44974285 | 37.68731136 | 37.47306895 | 10000 | 10000 | 10000 |
| ssim | 0.9860573347 | 0.9861481328 | 0.9859331862 | 10000 | 10000 | 10000 |

**仅残缺图像 (`corrupt_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 95.0100% | 95.0100% | 95.0100% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.003835059356 | 0.003447259386 | 0.003330046894 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.000413524797 | 0.0003884864116 | 0.0004005439484 | 10000 | 10000 | 10000 |
| aud_mse | 0.0004135247926 | 0.0003884864097 | 0.0004005439551 | 10000 | 10000 | 10000 |
| aud_ssim | 0.951825332 | 0.9554765461 | 0.9554608649 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 91.3000% | 92.0100% | 92.0900% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 91.3365% | 92.1469% | 92.2069% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 91.2200% | 92.0800% | 92.0900% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 91.3500% | 92.2900% | 92.2300% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.1100% | 97.1100% | 97.1100% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.1088% | 97.1088% | 97.1088% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.1100% | 97.1100% | 97.1100% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.1100% | 97.1100% | 97.1100% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 1.364526912e-05 | 1.493586396e-05 | 1.268125092e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9624624066 | 0.9265758574 | 0.9401886028 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.000413524797 | 0.0003884864116 | 0.0004005439484 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.04415509217 | 0.04767656994 | 0.04192593294 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0004137357385 | 0.0003893213555 | 0.0004024959208 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 2.202153989e-07 | 8.337853645e-07 | 1.89719927e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 67.4200% | 73.4100% | 76.7800% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 74.3500% | 80.0500% | 81.9000% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 87.5000% | 88.0100% | 90.2000% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 2.828159746e-06 | 3.852359479e-06 | 5.041249078e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0004163529567 | 0.0003923387686 | 0.0004055851962 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.000427170065 | 0.0004034222752 | 0.0004132252026 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.03719740973 | 0.03719740973 | 0.03719740973 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.02876905217 | 0.02876905217 | 0.02876905217 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.03351222687 | 0.03351222687 | 0.03351222687 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.01113058853 | 0.01113058853 | 0.01113058853 | 10000 | 10000 | 10000 |
| img_mse | 0.004457060372 | 0.004457060372 | 0.004457060372 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.145148684e-05 | 8.145148684e-05 | 8.145148684e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.144970207e-09 | 8.144970207e-09 | 8.144970207e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3531352878 | 0.3531352878 | 0.3531352878 | 10000 | 10000 | 10000 |
| pix_var | 0.06403968483 | 0.06403968483 | 0.06403968483 | 10000 | 10000 | 10000 |
| psnr | 24.07722128 | 24.07722128 | 24.07722128 | 10000 | 10000 | 10000 |
| ssim | 0.9740256632 | 0.9740256632 | 0.9740256632 | 10000 | 10000 | 10000 |

**仅残缺音频 (`corrupt_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 92.4800% | 92.4800% | 92.4800% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0002327938814 | 0.0002808066848 | 0.0002339824773 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5934983014 | 0.6368534152 | 0.6858307825 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.00752174806 | 0.007473735253 | 0.007520559478 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.4996181069 | 0.6740063148 | 0.6904659146 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.00749466281 | 0.007446876098 | 0.007493975772 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -2.881582172e-05 | -2.860000985e-05 | -2.831134508e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 42.9300% | 51.3000% | 42.6300% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 61.0600% | 65.5400% | 61.4300% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 63.3200% | 75.2700% | 61.8100% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | -6.55491167e-07 | 4.297043651e-06 | 4.978459951e-06 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.007521092536 | 0.007478032315 | 0.007525537874 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.007754541882 | 0.007754541882 | 0.007754541882 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02081972019 | 0.02027070971 | 0.02033335173 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.006036753169 | 0.005753536693 | 0.005824915885 | 10000 | 10000 | 10000 |
| aud_mse | 0.002452430975 | 0.002337374286 | 0.002366372054 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9237487059 | 0.9273950845 | 0.9265277929 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 93.1400% | 93.1300% | 93.1600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 93.1473% | 93.1373% | 93.1673% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 93.1400% | 93.1300% | 93.1600% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 93.1400% | 93.1300% | 93.1600% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 91.7200% | 91.7200% | 91.7500% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 91.6967% | 91.7467% | 91.6867% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 91.5900% | 91.6700% | 91.5300% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 91.5300% | 91.5300% | 91.5300% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.00752174806 | 0.007473735253 | 0.007520559478 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.01795260176 | 0.01776934626 | 0.01792726211 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.00752174806 | 0.007473735253 | 0.007520559478 | 10000 | 10000 | 10000 |
| img_mse | 0.007521747839 | 0.007473735026 | 0.007520559242 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2807147205 | 0.2806611657 | 0.280885905 | 10000 | 10000 | 10000 |
| pix_var | 0.04303168133 | 0.04303285107 | 0.043079108 | 10000 | 10000 | 10000 |
| psnr | 31.80816242 | 32.04960331 | 31.78091229 | 10000 | 10000 | 10000 |
| ssim | 0.9437553378 | 0.9439590086 | 0.9435978314 | 10000 | 10000 | 10000 |

**干净图像+残缺音频 (`clean_img_corrupt_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 98.9700% | 98.9700% | 98.9700% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02041057873 | 0.01982809407 | 0.01981546949 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.005703678265 | 0.005514252874 | 0.005672238063 | 10000 | 10000 | 10000 |
| aud_mse | 0.002317119278 | 0.002240165234 | 0.002304346723 | 10000 | 10000 | 10000 |
| aud_ssim | 0.927481733 | 0.9306347778 | 0.9290298914 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 94.9600% | 95.3700% | 95.3200% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 95.0480% | 95.4182% | 95.4082% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 94.9700% | 95.3800% | 95.3300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 94.7100% | 95.4600% | 95.3300% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.5100% | 97.5100% | 97.5100% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.5090% | 97.5090% | 97.5090% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.5100% | 97.5100% | 97.5100% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.5100% | 97.5100% | 97.5100% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0001944988099 | 0.0001029553605 | 1.69591725e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9550562046 | 0.9203363917 | 0.9338783075 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.005703678265 | 0.005514252874 | 0.005672238063 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.03936887819 | 0.04290874408 | 0.03791066148 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.00570415654 | 0.005514377989 | 0.005671883267 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -5.337151626e-07 | -8.382608377e-07 | -1.293874487e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 40.3900% | 37.6900% | 31.1700% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 50.2100% | 50.9600% | 50.1000% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 74.4600% | 67.7400% | 55.8500% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 6.704864645e-07 | 9.573371075e-06 | 1.467425329e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.005704348786 | 0.005523826242 | 0.005686912339 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.005898177144 | 0.005617208251 | 0.005689197243 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009764321695 | 0.009764321695 | 0.009764321695 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3402593434 | 0.3402593434 | 0.3402593434 | 10000 | 10000 | 10000 |
| pix_var | 0.05918137729 | 0.05918137729 | 0.05918137729 | 10000 | 10000 | 10000 |
| psnr | 20.7788683 | 20.7788683 | 20.7788683 | 10000 | 10000 | 10000 |
| ssim | 0.9426828535 | 0.9426828535 | 0.9426828535 | 10000 | 10000 | 10000 |

**残缺图像+干净音频 (`corrupt_img_clean_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1900% | 99.1900% | 99.1900% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.001086140746 | 0.001618183674 | 0.001563034368 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6153630577 | 0.6590167403 | 0.7017963562 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.00998387365 | 0.009451830594 | 0.00950697993 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.3955825868 | 0.5481617214 | 0.5586788889 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.009983919675 | 0.009454678454 | 0.009510504655 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -1.514288953e-08 | 2.802225502e-06 | 3.48976249e-06 | 9996 | 9996 | 9996 |
| aud2img_win_both | 48.2000% | 49.1700% | 48.9000% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 52.9500% | 53.1700% | 53.9700% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 89.1700% | 90.3300% | 88.2900% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 2.172960307e-05 | 3.541099811e-05 | 5.100276731e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.01000560321 | 0.00948724162 | 0.009557982787 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.0110700144 | 0.0110700144 | 0.0110700144 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003499352797 | 0.001020723127 | 0.001019466943 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9116188906 | 0.9707444804 | 0.9708549847 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.8500% | 96.9900% | 96.7400% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.8487% | 96.9888% | 96.7387% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.8500% | 96.9900% | 96.7400% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.8500% | 96.9900% | 96.7400% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.5000% | 97.4700% | 97.4900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.3689% | 97.4690% | 97.5490% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.4100% | 97.4100% | 97.4400% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.3800% | 97.3800% | 97.3800% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.03292696757 | 0.02937004498 | 0.02933466944 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.02812562053 | 0.02812562053 | 0.02812562053 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.03199914673 | 0.03108205955 | 0.03110212153 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.00998387365 | 0.009451830594 | 0.00950697993 | 10000 | 10000 | 10000 |
| img_mse | 0.003997021772 | 0.003783659536 | 0.003805669677 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.145148684e-05 | 8.145148684e-05 | 8.145148684e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.144970207e-09 | 8.144970207e-09 | 8.144970207e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3536060154 | 0.3540245891 | 0.3541216254 | 10000 | 10000 | 10000 |
| pix_var | 0.06418840587 | 0.06433641911 | 0.06437528133 | 10000 | 10000 | 10000 |
| psnr | 24.54670638 | 24.77698449 | 24.75443303 | 10000 | 10000 | 10000 |
| ssim | 0.9767314587 | 0.9780051371 | 0.9778848225 | 10000 | 10000 | 10000 |

**双残缺 (`corrupt_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 98.1100% | 98.1100% | 98.1100% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0008991165135 | 0.001364910049 | 0.001360247549 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6132846718 | 0.6550963888 | 0.6986741354 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.01016770366 | 0.009701910199 | 0.00970657272 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.3210676871 | 0.4445041903 | 0.4491385244 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.01017060693 | 0.009705269774 | 0.009709593918 | 9996 | 9996 | 9996 |
| aud2img_same_damage | 3.592304881e-06 | 4.062868039e-06 | 3.743885119e-06 | 9996 | 9996 | 9996 |
| aud2img_win_both | 47.0200% | 48.7300% | 48.1700% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 51.7400% | 52.3500% | 52.7700% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 89.1600% | 91.5300% | 89.5900% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 1.843625954e-05 | 2.872897608e-05 | 3.944773069e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.0101861399 | 0.009730639279 | 0.009746020442 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.01106682024 | 0.01106682024 | 0.01106682024 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02065796443 | 0.0200497023 | 0.02007767462 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.005863321823 | 0.005652054295 | 0.005798602038 | 10000 | 10000 | 10000 |
| aud_mse | 0.002381974522 | 0.002296147027 | 0.002355682068 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9271931437 | 0.9301777066 | 0.9286475119 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 94.3700% | 94.7900% | 94.7500% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 94.3878% | 94.7679% | 94.7579% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 94.3600% | 94.8600% | 94.7500% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 94.2700% | 94.5800% | 94.7300% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.3600% | 97.4100% | 97.4400% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.4090% | 97.5190% | 97.4090% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.3800% | 97.4400% | 97.4400% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.3300% | 97.3300% | 97.3300% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.000136315612 | 6.898090453e-05 | -1.604890119e-06 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9541105313 | 0.9201177547 | 0.9333787857 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.005863321823 | 0.005652054295 | 0.005798602038 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.03001767832 | 0.0325523833 | 0.02866192758 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.005864735386 | 0.005652569622 | 0.005798111174 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 1.012966248e-06 | 1.658057333e-07 | -8.392320434e-07 | 9996 | 9996 | 9996 |
| img2aud_win_both | 39.0700% | 35.6400% | 29.1300% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 49.0700% | 49.9300% | 49.4400% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 73.7500% | 66.0500% | 53.9900% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | -2.258712746e-06 | 3.583409466e-06 | 6.200671263e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.005861063091 | 0.00565563767 | 0.005804802684 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.00599963747 | 0.005721035155 | 0.005796997076 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.03268492988 | 0.03040343979 | 0.02983011493 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.02802559313 | 0.02802559313 | 0.02802559313 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.03217802968 | 0.03140366296 | 0.03133666456 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.01016770366 | 0.009701910199 | 0.00970657272 | 10000 | 10000 | 10000 |
| img_mse | 0.004073623262 | 0.003886929117 | 0.00388868753 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.144256312e-05 | 8.144256312e-05 | 8.144256312e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.144077533e-09 | 8.144077533e-09 | 8.144077533e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3548003435 | 0.3548741043 | 0.3551630676 | 10000 | 10000 | 10000 |
| pix_var | 0.06441956013 | 0.06444672495 | 0.06455276161 | 10000 | 10000 | 10000 |
| psnr | 24.4629626 | 24.66373293 | 24.66385347 | 10000 | 10000 | 10000 |
| ssim | 0.9762466478 | 0.9773551582 | 0.9773538333 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>fixed_mask / salt_mask / time_freq_block：全部原始指标</summary>

**干净双模态 (`clean_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1800% | 99.1800% | 99.1800% | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003503285427 | 0.001023170835 | 0.001022163968 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115226718 | 0.9706698141 | 0.9707627995 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.9588% | 97.0688% | 96.8387% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.7891% | 97.7891% | 97.7891% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009732332389 | 0.009732332389 | 0.009732332389 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3381031454 | 0.3381031454 | 0.3381031454 | 10000 | 10000 | 10000 |
| pix_var | 0.05885057896 | 0.05885057896 | 0.05885057896 | 10000 | 10000 | 10000 |
| psnr | 20.79410919 | 20.79410919 | 20.79410919 | 10000 | 10000 | 10000 |
| ssim | 0.9426845028 | 0.9426845028 | 0.9426845028 | 10000 | 10000 | 10000 |

**仅干净图像 (`clean_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 97.7000% | 97.7000% | 97.7000% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.002905382487 | 0.002580899412 | 0.0025012506 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| aud_mse | 0.0002402440196 | 0.0002218772086 | 0.0002282006215 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9736808665 | 0.9763999848 | 0.976126532 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.1300% | 96.5000% | 96.4000% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.1485% | 96.5186% | 96.4786% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.1000% | 96.4500% | 96.4300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.1500% | 96.6200% | 96.4500% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.8187% | 96.8187% | 96.8187% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 2.120906993e-05 | 2.734319982e-05 | 2.90869362e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9636753441 | 0.9270737396 | 0.9410639594 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.05812706452 | 0.06269425778 | 0.05521903026 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0002405052923 | 0.0002228766707 | 0.0002300969015 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 2.011856675e-07 | 9.414399495e-07 | 1.835515222e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 79.8300% | 82.5600% | 85.4500% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 82.8300% | 86.0600% | 88.3200% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 95.0400% | 94.6400% | 95.0200% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 7.060236394e-06 | 7.88233878e-06 | 1.064793261e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0002473042549 | 0.0002297595468 | 0.0002388485523 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0002614530888 | 0.0002492204076 | 0.0002572875584 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009777385685 | 0.009777385685 | 0.009777385685 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3401774764 | 0.3401774764 | 0.3401774764 | 10000 | 10000 | 10000 |
| pix_var | 0.05938630551 | 0.05938630551 | 0.05938630551 | 10000 | 10000 | 10000 |
| psnr | 20.77137023 | 20.77137023 | 20.77137023 | 10000 | 10000 | 10000 |
| ssim | 0.9426458778 | 0.9426458778 | 0.9426458778 | 10000 | 10000 | 10000 |

**仅干净音频 (`clean_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 98.8300% | 98.8300% | 98.8300% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 4.550894817e-05 | 5.925099576e-05 | 3.427388306e-05 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5948837502 | 0.6400160534 | 0.6886220585 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.6138913798 | 0.8292583166 | 0.855119685 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.002024612459 | 0.00200570006 | 0.002030547302 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -1.557539837e-05 | -2.074199916e-05 | -2.087765998e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 40.8600% | 51.8800% | 43.2100% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 71.3800% | 79.1800% | 76.4900% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 51.6500% | 62.1900% | 52.4400% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 3.748856293e-06 | 1.19613159e-05 | 1.177280845e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.002043171231 | 0.00203764165 | 0.002062430253 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.002084931313 | 0.002084931313 | 0.002084931313 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003496576117 | 0.001021110903 | 0.001020037457 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115410681 | 0.9707879167 | 0.9709046022 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 95.5282% | 96.5886% | 96.5586% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7300% | 98.0900% | 97.3500% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.4690% | 97.7491% | 97.6491% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.5200% | 97.7800% | 97.6700% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7100% | 97.7100% | 97.7100% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.007310281691 | 0.007187631063 | 0.007240436512 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_mse | 0.002039422297 | 0.002025680222 | 0.002050657357 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2828797698 | 0.283067286 | 0.2833743691 | 10000 | 10000 | 10000 |
| pix_var | 0.0461221002 | 0.04620051384 | 0.04626523703 | 10000 | 10000 | 10000 |
| psnr | 37.44974285 | 37.68731136 | 37.47306895 | 10000 | 10000 | 10000 |
| ssim | 0.9860573347 | 0.9861481328 | 0.9859331862 | 10000 | 10000 | 10000 |

**仅残缺图像 (`corrupt_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 87.7900% | 87.7900% | 87.7900% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.005778785831 | 0.005345681586 | 0.005171947253 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0007986420516 | 0.0007662277512 | 0.0007903586426 | 10000 | 10000 | 10000 |
| aud_mse | 0.0007986420572 | 0.0007662277577 | 0.0007903586436 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9011951105 | 0.9065545802 | 0.9071538321 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 80.8000% | 82.2100% | 82.3500% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 80.8423% | 82.3429% | 82.5630% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 80.5600% | 82.0300% | 82.2900% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 80.9300% | 82.4100% | 82.6000% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.1400% | 97.1400% | 97.1400% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.1389% | 97.1389% | 97.1389% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.1400% | 97.1400% | 97.1400% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.1400% | 97.1400% | 97.1400% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 1.058080915e-05 | 8.594039938e-06 | 2.508921054e-07 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9622003248 | 0.9263632915 | 0.9398784277 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.0007986420516 | 0.0007662277512 | 0.0007903586426 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.04274455789 | 0.04581039838 | 0.04036065302 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0007977812267 | 0.0007654639284 | 0.0007898777196 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -1.074561591e-06 | -9.824803206e-07 | -7.177488246e-07 | 9996 | 9996 | 9996 |
| img2aud_win_both | 53.3400% | 59.3600% | 62.6500% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 68.0300% | 71.9100% | 72.6400% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 73.7500% | 76.8000% | 80.4400% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 2.19808175e-06 | 2.503931308e-06 | 2.236538098e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.000800840125 | 0.0007687316794 | 0.0007925951811 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0008092228674 | 0.0007748217965 | 0.0007906095428 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.0673675662 | 0.0673675662 | 0.0673675662 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.06715986135 | 0.06715986135 | 0.06715986135 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.03883268682 | 0.03883268682 | 0.03883268682 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.01324784766 | 0.01324784766 | 0.01324784766 | 10000 | 10000 | 10000 |
| img_mse | 0.005306742514 | 0.005306742514 | 0.005306742514 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.145766852e-05 | 8.145766852e-05 | 8.145766852e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.145588238e-09 | 8.145588238e-09 | 8.145588238e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3500238657 | 0.3500238657 | 0.3500238657 | 10000 | 10000 | 10000 |
| pix_var | 0.06298545748 | 0.06298545748 | 0.06298545748 | 10000 | 10000 | 10000 |
| psnr | 23.21632739 | 23.21632739 | 23.21632739 | 10000 | 10000 | 10000 |
| ssim | 0.967731621 | 0.967731621 | 0.967731621 | 10000 | 10000 | 10000 |

**仅残缺音频 (`corrupt_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 96.7700% | 96.7700% | 96.7700% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 9.445707166e-05 | 0.0001200020259 | 8.689478616e-05 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5942497313 | 0.6386716425 | 0.6875809671 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.003472736272 | 0.003447191342 | 0.003480298529 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.5857668663 | 0.7912090986 | 0.8143044091 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.00345251771 | 0.003423376111 | 0.003459287835 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -2.155017496e-05 | -2.513872188e-05 | -2.23419506e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 41.8300% | 52.9000% | 43.5800% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 68.1300% | 74.5600% | 70.9100% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 56.3000% | 68.4400% | 56.1600% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | -1.201988754e-06 | 7.926657418e-06 | 9.066795793e-06 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.003471534305 | 0.003455117979 | 0.00348936537 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.003567193356 | 0.003567193356 | 0.003567193356 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.01792071308 | 0.01733511547 | 0.0173734674 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.005165506733 | 0.004835435798 | 0.004876731916 | 10000 | 10000 | 10000 |
| aud_mse | 0.0008525103914 | 0.0007980357829 | 0.0008048512543 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9793025884 | 0.9804701895 | 0.9802873158 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 97.0100% | 97.0100% | 97.0700% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 97.0088% | 97.0088% | 97.0688% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 97.0100% | 97.0100% | 97.0700% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 97.0100% | 97.0100% | 97.0700% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.4200% | 96.1900% | 96.2400% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.2985% | 96.4886% | 96.2585% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.1800% | 96.3500% | 96.3100% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.2000% | 96.2000% | 96.2000% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.003472736272 | 0.003447191342 | 0.003480298529 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.009987326416 | 0.00983853772 | 0.009926244199 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.003472736272 | 0.003447191342 | 0.003480298529 | 10000 | 10000 | 10000 |
| img_mse | 0.003472736135 | 0.003447191207 | 0.003480298504 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2877429724 | 0.2878516018 | 0.2881170511 | 10000 | 10000 | 10000 |
| pix_var | 0.04537769407 | 0.0454309471 | 0.04549913481 | 10000 | 10000 | 10000 |
| psnr | 36.06760914 | 36.31310556 | 36.06269712 | 10000 | 10000 | 10000 |
| ssim | 0.9749604658 | 0.9750920891 | 0.9748191306 | 10000 | 10000 | 10000 |

**干净图像+残缺音频 (`clean_img_corrupt_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1800% | 99.1800% | 99.1800% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.01756336051 | 0.01693401155 | 0.01691886399 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.00487950018 | 0.004624000993 | 0.004738591301 | 10000 | 10000 | 10000 |
| aud_mse | 0.000805308133 | 0.0007631407963 | 0.000782052662 | 10000 | 10000 | 10000 |
| aud_ssim | 0.980239646 | 0.9812645176 | 0.9808451436 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 97.3000% | 97.3900% | 97.5000% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 97.3189% | 97.4390% | 97.4990% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 97.2700% | 97.3900% | 97.4200% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 97.4400% | 97.3900% | 97.5000% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.8300% | 97.8300% | 97.8300% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.8291% | 97.8291% | 97.8291% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.8300% | 97.8300% | 97.8300% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.8300% | 97.8300% | 97.8300% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0002191250848 | 0.0001437225515 | 6.941655717e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9614516386 | 0.927206784 | 0.9397715122 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.00487950018 | 0.004624000993 | 0.004738591301 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.01102867115 | 0.01143903206 | 0.01047566741 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.00488019217 | 0.004624560389 | 0.004739870848 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -1.083603922e-06 | -1.122577992e-06 | -4.405594563e-07 | 9996 | 9996 | 9996 |
| img2aud_win_both | 25.6200% | 25.0600% | 22.8300% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 31.6500% | 31.7600% | 31.3400% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 45.8200% | 45.5800% | 42.3300% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | -2.478087178e-06 | 4.184724743e-06 | 7.805636222e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.004877022097 | 0.004628185688 | 0.004746396923 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.005098625248 | 0.004767723616 | 0.00480800787 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009737672757 | 0.009737672757 | 0.009737672757 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3377070725 | 0.3377070725 | 0.3377070725 | 10000 | 10000 | 10000 |
| pix_var | 0.05894574523 | 0.05894574523 | 0.05894574523 | 10000 | 10000 | 10000 |
| psnr | 20.7906302 | 20.7906302 | 20.7906302 | 10000 | 10000 | 10000 |
| ssim | 0.9427141905 | 0.9427141905 | 0.9427141905 | 10000 | 10000 | 10000 |

**残缺图像+干净音频 (`corrupt_img_clean_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.0900% | 99.0900% | 99.0900% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0008039618233 | 0.001256603088 | 0.001180106846 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6327106505 | 0.6745326028 | 0.7062777435 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.01228423848 | 0.0118315972 | 0.01190809334 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.6079455818 | 1.193304636 | 1.138555608 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.01228674869 | 0.01183578722 | 0.01191344867 | 9996 | 9996 | 9996 |
| aud2img_same_damage | 2.949371674e-06 | 4.753242249e-06 | 5.931312397e-06 | 9996 | 9996 | 9996 |
| aud2img_win_both | 46.9700% | 48.7000% | 48.4200% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 52.3800% | 52.7600% | 53.4200% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 86.2400% | 89.3000% | 86.7100% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 2.365952162e-05 | 3.099734031e-05 | 4.160830225e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.01230789792 | 0.01186259451 | 0.01194970172 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.0130882004 | 0.0130882004 | 0.0130882004 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003501636255 | 0.001020851541 | 0.001019671523 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9114996616 | 0.9707172032 | 0.9708252401 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.2400% | 96.9300% | 96.6600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.2385% | 96.9288% | 96.6587% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.2400% | 96.9300% | 96.6600% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.2400% | 96.9300% | 96.6600% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.3000% | 97.2900% | 97.3300% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.2889% | 97.2089% | 97.2889% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.2800% | 97.2700% | 97.2700% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.2800% | 97.2800% | 97.2800% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.06650456897 | 0.06302992365 | 0.06290029943 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.06625697267 | 0.06625697267 | 0.06625697267 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.03727928067 | 0.03611325157 | 0.0362596334 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.01228423848 | 0.0118315972 | 0.01190809334 | 10000 | 10000 | 10000 |
| img_mse | 0.004921063209 | 0.004739833036 | 0.004769962095 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.145766852e-05 | 8.145766852e-05 | 8.145766852e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.145588238e-09 | 8.145588238e-09 | 8.145588238e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3498128653 | 0.3508115113 | 0.3510591388 | 10000 | 10000 | 10000 |
| pix_var | 0.06291244179 | 0.06326954812 | 0.0633591935 | 10000 | 10000 | 10000 |
| psnr | 23.55377555 | 23.72593504 | 23.69946437 | 10000 | 10000 | 10000 |
| ssim | 0.9700615904 | 0.9712419872 | 0.9710759232 | 10000 | 10000 | 10000 |

**双残缺 (`corrupt_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 98.6100% | 98.6100% | 98.6100% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0007723308697 | 0.001223151816 | 0.001157669397 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6320288195 | 0.6732097376 | 0.7051982813 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.01233916689 | 0.01188834585 | 0.01195382828 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.5782969576 | 1.137390355 | 1.08163563 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.01234036943 | 0.01189093963 | 0.01195855259 | 9996 | 9996 | 9996 |
| aud2img_same_damage | 7.464738056e-07 | 2.200880107e-06 | 4.298695224e-06 | 9996 | 9996 | 9996 |
| aud2img_win_both | 46.5000% | 47.9900% | 48.1200% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 51.6400% | 52.4500% | 53.0300% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 86.4700% | 88.8900% | 87.1600% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 1.233329544e-05 | 2.507763351e-05 | 3.410575952e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.01235150011 | 0.01191342354 | 0.01198793411 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.01311149772 | 0.01311149772 | 0.01311149772 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.01728475155 | 0.01663322582 | 0.01666726209 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.004776661211 | 0.004498748161 | 0.004603430567 | 10000 | 10000 | 10000 |
| aud_mse | 0.0007883356947 | 0.0007424691813 | 0.0007597458694 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9801119209 | 0.981290695 | 0.9809497472 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 97.3400% | 97.4100% | 97.4400% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 97.2689% | 97.3389% | 97.4190% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 97.3000% | 97.4000% | 97.3900% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 97.3400% | 97.3600% | 97.4600% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.3800% | 97.3300% | 97.2800% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.3689% | 97.3089% | 97.3689% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.3600% | 97.2900% | 97.3100% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.3100% | 97.3100% | 97.3100% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0001483734821 | 0.0001038512247 | 4.600866914e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9607160848 | 0.9271659006 | 0.9394552425 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.004776661211 | 0.004498748161 | 0.004603430567 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.008366967776 | 0.008525543368 | 0.00779485928 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.004775958208 | 0.004498493366 | 0.004602842598 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 9.588527674e-07 | 1.115997053e-06 | 7.939937537e-07 | 9996 | 9996 | 9996 |
| img2aud_win_both | 23.7400% | 23.5800% | 21.2700% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 30.1300% | 30.5900% | 29.9600% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 45.5200% | 45.4500% | 41.5500% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | -3.390308136e-06 | 1.612734189e-06 | -3.013915441e-07 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.004773270936 | 0.0045003609 | 0.004603129181 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.004925034738 | 0.004602599367 | 0.004649439217 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.06582348427 | 0.06314092381 | 0.0624492622 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.06635928849 | 0.06635928849 | 0.06635928849 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.03725029513 | 0.03612402446 | 0.03623058478 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.01233916689 | 0.01188834585 | 0.01195382828 | 10000 | 10000 | 10000 |
| img_mse | 0.004947818325 | 0.004767618213 | 0.004793057459 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.145455515e-05 | 8.145455515e-05 | 8.145455515e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.145276718e-09 | 8.145276718e-09 | 8.145276718e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3527439833 | 0.3534012139 | 0.3538752496 | 10000 | 10000 | 10000 |
| pix_var | 0.06299800426 | 0.06323502213 | 0.06340306997 | 10000 | 10000 | 10000 |
| psnr | 23.53827489 | 23.70927925 | 23.68620844 | 10000 | 10000 | 10000 |
| ssim | 0.9698659904 | 0.9710058624 | 0.9708875696 | 10000 | 10000 | 10000 |

</details>

<details>
<summary>legacy_random / random_mix / random_mix：全部原始指标</summary>

**干净双模态 (`clean_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1800% | 99.1800% | 99.1800% | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003503285427 | 0.001023170835 | 0.001022163968 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115226718 | 0.9706698141 | 0.9707627995 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.9588% | 97.0688% | 96.8387% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.9600% | 97.0700% | 96.8400% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.7891% | 97.7891% | 97.7891% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7900% | 97.7900% | 97.7900% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009732332389 | 0.009732332389 | 0.009732332389 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3399402201 | 0.3399402201 | 0.3399402201 | 10000 | 10000 | 10000 |
| pix_var | 0.05885057896 | 0.05885057896 | 0.05885057896 | 10000 | 10000 | 10000 |
| psnr | 20.79410919 | 20.79410919 | 20.79410919 | 10000 | 10000 | 10000 |
| ssim | 0.9426845028 | 0.9426845028 | 0.9426845028 | 10000 | 10000 | 10000 |

**仅干净图像 (`clean_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 97.7000% | 97.7000% | 97.7000% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.002905382487 | 0.002580899412 | 0.0025012506 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| aud_mse | 0.0002402440196 | 0.0002218772086 | 0.0002282006215 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9736808665 | 0.9763999848 | 0.976126532 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.1300% | 96.5000% | 96.4000% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.1485% | 96.5186% | 96.4786% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.1000% | 96.4500% | 96.4300% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.1500% | 96.6200% | 96.4500% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.8187% | 96.8187% | 96.8187% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.8200% | 96.8200% | 96.8200% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 2.120906993e-05 | 2.734319982e-05 | 2.90869362e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9636753441 | 0.9270737396 | 0.9410639594 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.0002402440197 | 0.00022187721 | 0.0002282006204 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.05812706452 | 0.06269425778 | 0.05521903026 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0002405052923 | 0.0002228766707 | 0.0002300969015 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 2.011856675e-07 | 9.414399495e-07 | 1.835515222e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 79.8300% | 82.5600% | 85.4500% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 82.8300% | 86.0600% | 88.3200% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 95.0400% | 94.6400% | 95.0200% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 7.060236394e-06 | 7.88233878e-06 | 1.064793261e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0002473042549 | 0.0002297595468 | 0.0002388485523 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.0002614530888 | 0.0002492204076 | 0.0002572875584 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009777385685 | 0.009777385685 | 0.009777385685 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3399478495 | 0.3399478495 | 0.3399478495 | 10000 | 10000 | 10000 |
| pix_var | 0.05938630551 | 0.05938630551 | 0.05938630551 | 10000 | 10000 | 10000 |
| psnr | 20.77137023 | 20.77137023 | 20.77137023 | 10000 | 10000 | 10000 |
| ssim | 0.9426458778 | 0.9426458778 | 0.9426458778 | 10000 | 10000 | 10000 |

**仅干净音频 (`clean_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 98.8300% | 98.8300% | 98.8300% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 4.550894817e-05 | 5.925099576e-05 | 3.427388306e-05 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5948837502 | 0.6400160534 | 0.6886220585 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.6138913798 | 0.8292583166 | 0.855119685 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.002024612459 | 0.00200570006 | 0.002030547302 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -1.557539837e-05 | -2.074199916e-05 | -2.087765998e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 40.8600% | 51.8800% | 43.2100% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 71.3800% | 79.1800% | 76.4900% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 51.6500% | 62.1900% | 52.4400% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 3.748856293e-06 | 1.19613159e-05 | 1.177280845e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.002043171231 | 0.00203764165 | 0.002062430253 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.002084931313 | 0.002084931313 | 0.002084931313 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003496576117 | 0.001021110903 | 0.001020037457 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115410681 | 0.9707879167 | 0.9709046022 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 95.5282% | 96.5886% | 96.5586% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 95.5300% | 96.5900% | 96.5600% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.7300% | 98.0900% | 97.3500% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.4690% | 97.7491% | 97.6491% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.5200% | 97.7800% | 97.6700% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.7100% | 97.7100% | 97.7100% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.007310281691 | 0.007187631063 | 0.007240436512 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.002039422361 | 0.002025680301 | 0.002050657432 | 10000 | 10000 | 10000 |
| img_mse | 0.002039422297 | 0.002025680222 | 0.002050657357 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.2852356732 | 0.2854161859 | 0.285684526 | 10000 | 10000 | 10000 |
| pix_var | 0.0461221002 | 0.04620051384 | 0.04626523703 | 10000 | 10000 | 10000 |
| psnr | 37.44974285 | 37.68731136 | 37.47306895 | 10000 | 10000 | 10000 |
| ssim | 0.9860573347 | 0.9861481328 | 0.9859331862 | 10000 | 10000 | 10000 |

**仅残缺图像 (`corrupt_img_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 92.3600% | 92.3600% | 92.3600% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.004401024371 | 0.004016935972 | 0.003898758639 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.000549146415 | 0.0005232986979 | 0.0005377318759 | 10000 | 10000 | 10000 |
| aud_mse | 0.0005491464123 | 0.0005232986961 | 0.0005377318716 | 10000 | 10000 | 10000 |
| aud_ssim | 0.934874578 | 0.938891894 | 0.9387075771 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 88.0700% | 89.1900% | 89.1900% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 88.1553% | 89.2957% | 89.2457% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 88.2000% | 89.1000% | 89.2700% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 88.1200% | 89.3800% | 89.2400% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 95.4900% | 95.4900% | 95.4900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 95.4882% | 95.4882% | 95.4882% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 95.4900% | 95.4900% | 95.4900% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 95.4900% | 95.4900% | 95.4900% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 1.433363229e-05 | 1.575728901e-05 | 1.225653508e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9626510489 | 0.926657808 | 0.9403141308 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.000549146415 | 0.0005232986979 | 0.0005377318759 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.04585074922 | 0.04928698386 | 0.04333371771 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.0005494240526 | 0.000524346426 | 0.0005401250394 | 9996 | 9996 | 9996 |
| img2aud_same_damage | 1.846825965e-07 | 9.470778609e-07 | 2.252867799e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 66.2200% | 70.6300% | 74.1100% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 75.2400% | 78.7100% | 80.6300% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 84.7700% | 85.7500% | 87.6000% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | 3.536865775e-06 | 4.173188881e-06 | 5.532323274e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.0005526832814 | 0.0005274718828 | 0.0005432641973 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.000563480047 | 0.000539055987 | 0.0005499884168 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.06070868487 | 0.06070868487 | 0.06070868487 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.02910082169 | 0.02910082169 | 0.02910082169 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.06892660542 | 0.06892660542 | 0.06892660542 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.0292326478 | 0.0292326478 | 0.0292326478 | 10000 | 10000 | 10000 |
| img_mse | 0.007614292219 | 0.007614292219 | 0.007614292219 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.143082536e-05 | 8.143082536e-05 | 8.143082536e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.142905017e-09 | 8.142905017e-09 | 8.142905017e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3455364406 | 0.3455364406 | 0.3455364406 | 10000 | 10000 | 10000 |
| pix_var | 0.06094776839 | 0.06094776839 | 0.06094776839 | 10000 | 10000 | 10000 |
| psnr | 22.86542305 | 22.86542305 | 22.86542305 | 10000 | 10000 | 10000 |
| ssim | 0.9552188177 | 0.9552188177 | 0.9552188177 | 10000 | 10000 | 10000 |

**仅残缺音频 (`corrupt_aud_only`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 85.7900% | 85.7900% | 85.7900% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0007191967652 | 0.0009368507476 | 0.0010868413 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.5947398814 | 0.638415197 | 0.6873055408 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.01148949487 | 0.01127184076 | 0.01112185018 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.5017223105 | 0.675679085 | 0.7000467607 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.0115440222 | 0.01141726737 | 0.01132947641 | 9996 | 9996 | 9996 |
| aud2img_same_damage | 7.026617956e-05 | 0.0001606745704 | 0.0002221993437 | 9996 | 9996 | 9996 |
| aud2img_win_both | 43.0900% | 51.2700% | 42.8900% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 63.6300% | 68.1800% | 64.7600% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 62.7800% | 73.2500% | 62.0100% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 7.682295728e-05 | 0.0001769999542 | 0.000235098807 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.01156631779 | 0.01144884081 | 0.011356949 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.01220869151 | 0.01220869151 | 0.01220869151 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02927086294 | 0.02797595997 | 0.0280077034 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.009371610844 | 0.008850679514 | 0.008943945056 | 10000 | 10000 | 10000 |
| aud_mse | 0.003578324331 | 0.003382371552 | 0.003418815638 | 10000 | 10000 | 10000 |
| aud_ssim | 0.8705906674 | 0.8760217186 | 0.8755758115 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 85.9300% | 86.0700% | 85.9900% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 85.9544% | 86.0944% | 86.0144% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 85.9300% | 86.0700% | 85.9900% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 85.9300% | 86.0700% | 85.9900% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 84.0700% | 84.0800% | 83.8600% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 84.0636% | 84.2037% | 83.9336% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 83.9500% | 84.1800% | 83.9500% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 84.0900% | 84.0900% | 84.0900% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.01148949487 | 0.01127184076 | 0.01112185018 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | 0.02389682578 | 0.0238848722 | 0.02430988313 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.01148949487 | 0.01127184076 | 0.01112185018 | 10000 | 10000 | 10000 |
| img_mse | 0.01148949463 | 0.01127184049 | 0.01112184997 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.271656692 | 0.2710115016 | 0.2706104815 | 10000 | 10000 | 10000 |
| pix_var | 0.04038360715 | 0.04023037851 | 0.04010954499 | 10000 | 10000 | 10000 |
| psnr | 31.28457344 | 31.50659519 | 31.2869743 | 10000 | 10000 | 10000 |
| ssim | 0.9000236948 | 0.9035166462 | 0.9056596614 | 10000 | 10000 | 10000 |

**干净图像+残缺音频 (`clean_img_corrupt_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 98.9600% | 98.9600% | 98.9600% | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02803460948 | 0.02671373892 | 0.02658687366 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.008538770679 | 0.008107095659 | 0.008371060044 | 10000 | 10000 | 10000 |
| aud_mse | 0.003252138307 | 0.003089098994 | 0.00319140949 | 10000 | 10000 | 10000 |
| aud_ssim | 0.8807206806 | 0.8855609869 | 0.8850794596 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 94.7600% | 94.9800% | 95.1900% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 94.8379% | 95.0280% | 95.1681% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 94.7200% | 94.9400% | 94.9800% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 94.8500% | 94.9400% | 95.0200% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 97.5000% | 97.5000% | 97.5000% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 97.4990% | 97.4990% | 97.4990% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 97.5000% | 97.5000% | 97.5000% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 97.5000% | 97.5000% | 97.5000% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0002911213193 | 0.0001817647181 | 3.269380778e-05 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9554412912 | 0.9198254534 | 0.9332742281 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.008538770679 | 0.008107095659 | 0.008371060044 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.02401790369 | 0.02578279207 | 0.02290292897 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.008532503409 | 0.008101494878 | 0.008364864397 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -2.43109985e-07 | 8.201565527e-07 | 1.079840022e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 36.1900% | 36.1000% | 32.3800% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 43.5900% | 45.5300% | 44.9500% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 70.0600% | 66.2600% | 58.7100% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | -1.765788326e-06 | 7.456294744e-06 | 1.142577223e-05 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.008537004831 | 0.008114551979 | 0.00838248594 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.008829892039 | 0.008288860309 | 0.008403754005 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_coarse_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| img_mse | 0.009783406097 | 0.009783406097 | 0.009783406097 | 10000 | 10000 | 10000 |
| img_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| img_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3416532576 | 0.3416532576 | 0.3416532576 | 10000 | 10000 | 10000 |
| pix_var | 0.05928457528 | 0.05928457528 | 0.05928457528 | 10000 | 10000 | 10000 |
| psnr | 20.77048488 | 20.77048488 | 20.77048488 | 10000 | 10000 | 10000 |
| ssim | 0.9426022731 | 0.9426022731 | 0.9426022731 | 10000 | 10000 | 10000 |

**残缺图像+干净音频 (`corrupt_img_clean_aud`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 99.1600% | 99.1600% | 99.1600% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.001156052208 | 0.001507425934 | 0.001401434613 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6170308393 | 0.6594413919 | 0.6949520991 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.02710606877 | 0.02675469481 | 0.02686068637 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.3379561121 | 0.5088711754 | 0.516051271 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.02708293691 | 0.02673262496 | 0.02684275328 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -5.602101909e-06 | -4.799576746e-06 | -1.569811177e-06 | 9996 | 9996 | 9996 |
| aud2img_win_both | 42.1300% | 43.5500% | 41.9400% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 52.2100% | 52.3400% | 52.5800% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 75.9700% | 78.8300% | 74.6300% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 2.364825222e-05 | 3.375893191e-05 | 5.172351762e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.02712971728 | 0.02678845391 | 0.02691241012 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.02826212105 | 0.02826212105 | 0.02826212105 | 10000 | 10000 | 10000 |
| aud_masked_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_masked_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_mse | 0.003501629138 | 0.001021167909 | 0.001019949091 | 10000 | 10000 | 10000 |
| aud_ssim | 0.9115314157 | 0.9707052304 | 0.9708114794 | 10000 | 10000 | 10000 |
| aud_visible_l1 | N/A | N/A | N/A | 0 | 0 | 0 |
| aud_visible_mse | N/A | N/A | N/A | 0 | 0 | 0 |
| content_aud_normal_acc | 96.6600% | 96.9700% | 96.6000% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 96.6587% | 96.9688% | 96.5986% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 96.6600% | 96.9700% | 96.6000% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 96.6600% | 96.9700% | 96.6000% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.7700% | 96.7100% | 96.7400% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.7387% | 96.8788% | 96.7387% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.7900% | 96.9000% | 96.7900% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.6900% | 96.6900% | 96.6900% | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.05777017832 | 0.05563699696 | 0.05477552149 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.02862981399 | 0.02862981399 | 0.02862981399 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.06686613178 | 0.06606935723 | 0.06523520889 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.02710606877 | 0.02675469481 | 0.02686068637 | 10000 | 10000 | 10000 |
| img_mse | 0.007102384464 | 0.006989629062 | 0.007024656107 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.143082536e-05 | 8.143082536e-05 | 8.143082536e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.142905017e-09 | 8.142905017e-09 | 8.142905017e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3456449509 | 0.3460197151 | 0.3467237353 | 10000 | 10000 | 10000 |
| pix_var | 0.06099751964 | 0.06113048643 | 0.06139310449 | 10000 | 10000 | 10000 |
| psnr | 23.17452033 | 23.28899767 | 23.29306716 | 10000 | 10000 | 10000 |
| ssim | 0.958277329 | 0.9589780867 | 0.958841262 | 10000 | 10000 | 10000 |

**双残缺 (`corrupt_both`)**

| 指标 | control | v12b | no_causal | control n | v12b n | no_causal n |
|---|---|---|---|---|---|---|
| acc | 97.2800% | 97.2800% | 97.2800% | 10000 | 10000 | 10000 |
| aud2img_correct_gain | 0.0008603106864 | 0.001163088947 | 0.001083243625 | 10000 | 10000 | 10000 |
| aud2img_gate | 0.6170564057 | 0.658111935 | 0.6945510743 | 10000 | 10000 | 10000 |
| aud2img_normal_mse | 0.02532390426 | 0.0250211262 | 0.02510097159 | 10000 | 10000 | 10000 |
| aud2img_ratio | 0.3038183321 | 0.4699307745 | 0.4721096097 | 10000 | 10000 | 10000 |
| aud2img_same_class_mse | 0.02528559596 | 0.02498310283 | 0.02506784399 | 9996 | 9996 | 9996 |
| aud2img_same_damage | -1.589017392e-05 | -1.610946085e-05 | -1.241369187e-05 | 9996 | 9996 | 9996 |
| aud2img_win_both | 38.5600% | 40.5500% | 39.1600% | 10000 | 10000 | 10000 |
| aud2img_win_wrong | 50.2800% | 50.6700% | 50.9600% | 10000 | 10000 | 10000 |
| aud2img_win_zero | 70.6500% | 74.4200% | 70.9300% | 10000 | 10000 | 10000 |
| aud2img_wrong_damage | 3.250067465e-05 | 4.27605171e-05 | 6.097319155e-05 | 10000 | 10000 | 10000 |
| aud2img_wrong_mse | 0.02535640543 | 0.02506388687 | 0.02516194481 | 10000 | 10000 | 10000 |
| aud2img_zero_mse | 0.02618421544 | 0.02618421544 | 0.02618421544 | 10000 | 10000 | 10000 |
| aud_masked_l1 | 0.02759425491 | 0.02635841867 | 0.026269459 | 10000 | 10000 | 10000 |
| aud_masked_mse | 0.008598824614 | 0.00817188829 | 0.008401221268 | 10000 | 10000 | 10000 |
| aud_mse | 0.003243830509 | 0.003085861862 | 0.003173652124 | 10000 | 10000 | 10000 |
| aud_ssim | 0.8870781691 | 0.8914218223 | 0.8911181785 | 10000 | 10000 | 10000 |
| aud_visible_l1 | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| aud_visible_mse | 0 | 0 | 0 | 10000 | 10000 | 10000 |
| content_aud_normal_acc | 92.6000% | 92.9500% | 93.2300% | 10000 | 10000 | 10000 |
| content_aud_same_class_acc | 92.5370% | 93.0472% | 93.1773% | 9996 | 9996 | 9996 |
| content_aud_wrong_acc | 92.4600% | 92.8800% | 93.0900% | 10000 | 10000 | 10000 |
| content_aud_zero_acc | 92.5500% | 93.0500% | 92.8100% | 10000 | 10000 | 10000 |
| content_img_normal_acc | 96.2800% | 96.3700% | 96.4900% | 10000 | 10000 | 10000 |
| content_img_same_class_acc | 96.4486% | 96.3886% | 96.3886% | 9996 | 9996 | 9996 |
| content_img_wrong_acc | 96.3900% | 96.4500% | 96.3600% | 10000 | 10000 | 10000 |
| content_img_zero_acc | 96.3400% | 96.3400% | 96.3400% | 10000 | 10000 | 10000 |
| img2aud_correct_gain | 0.0001949578249 | 0.0001146678842 | -2.151020546e-06 | 10000 | 10000 | 10000 |
| img2aud_gate | 0.9559663939 | 0.9210142807 | 0.9340382945 | 10000 | 10000 | 10000 |
| img2aud_normal_mse | 0.008598824614 | 0.00817188829 | 0.008401221268 | 10000 | 10000 | 10000 |
| img2aud_ratio | 0.01707282281 | 0.01809543395 | 0.01611432302 | 10000 | 10000 | 10000 |
| img2aud_same_class_mse | 0.008591988844 | 0.008166542198 | 0.008394838782 | 9996 | 9996 | 9996 |
| img2aud_same_damage | -5.621254821e-07 | 1.257507768e-06 | 1.007103719e-06 | 9996 | 9996 | 9996 |
| img2aud_win_both | 35.0900% | 33.3200% | 30.1400% | 10000 | 10000 | 10000 |
| img2aud_win_wrong | 42.6500% | 43.2300% | 42.9600% | 10000 | 10000 | 10000 |
| img2aud_win_zero | 66.0000% | 61.6800% | 55.0400% | 10000 | 10000 | 10000 |
| img2aud_wrong_damage | -8.016939973e-08 | 4.174842969e-06 | 6.584647554e-06 | 10000 | 10000 | 10000 |
| img2aud_wrong_mse | 0.008598744443 | 0.008176063168 | 0.008407805932 | 10000 | 10000 | 10000 |
| img2aud_zero_mse | 0.008793782392 | 0.008286556166 | 0.008399070302 | 10000 | 10000 | 10000 |
| img_coarse_masked_mse | 0.05335954244 | 0.05211471896 | 0.05109498558 | 10000 | 10000 | 10000 |
| img_coarse_visible_mse | 0.03069736354 | 0.03069736354 | 0.03069736354 | 10000 | 10000 | 10000 |
| img_masked_l1 | 0.06291228101 | 0.06236043167 | 0.06176770277 | 10000 | 10000 | 10000 |
| img_masked_mse | 0.02532390426 | 0.0250211262 | 0.02510097159 | 10000 | 10000 | 10000 |
| img_mse | 0.007473485468 | 0.007371911891 | 0.007402919574 | 10000 | 10000 | 10000 |
| img_visible_l1 | 8.09325792e-05 | 8.09325792e-05 | 8.09325792e-05 | 10000 | 10000 | 10000 |
| img_visible_mse | 8.093074925e-09 | 8.093074925e-09 | 8.093074925e-09 | 10000 | 10000 | 10000 |
| pair_a2i_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_i2a_r1 | N/A | N/A | N/A | 0 | 0 | 0 |
| pair_l2 | 0.3445170522 | 0.3447151184 | 0.3455003798 | 10000 | 10000 | 10000 |
| pix_var | 0.06075927243 | 0.06083749235 | 0.06110813096 | 10000 | 10000 | 10000 |
| psnr | 22.90560941 | 22.99644513 | 22.99737832 | 10000 | 10000 | 10000 |
| ssim | 0.9559758368 | 0.9566177238 | 0.956496457 | 10000 | 10000 | 10000 |

</details>

#### 音频专项完整字段

源专项 CSV 未输出 n 列；日志显示完整 79/79 batches，主评估 n=10000，不能把 absent n 字段填为 0。

<details>
<summary>仅残缺音频 / time_mask：音频专项全部字段</summary>

family_group：`paper_aligned_time_gap`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 90.3900% | 90.3900% | 90.3900% |
| img_mse | 0.008965390432 | 0.008881425519 | 0.008872701649 |
| psnr | 32.64531893 | 32.85885403 | 32.61591738 |
| img_ssim | 0.9287584259 | 0.9295776613 | 0.9297687733 |
| aud_mse | 0.003279069786 | 0.003157877918 | 0.003187179682 |
| aud_ssim | 0.9098891888 | 0.9131757013 | 0.9134275942 |
| aud_masked_mse | 0.00807155631 | 0.007773238017 | 0.007845365428 |
| aud_masked_l1 | 0.02359827327 | 0.02309024938 | 0.02301557826 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1474183841 | 0.1476193828 | 0.1481330888 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 89.0814% | 89.0921% | 89.2550% |

</details>

<details>
<summary>仅残缺音频 / freq_mask：音频专项全部字段</summary>

family_group：`spectrogram_occlusion`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 92.2200% | 92.2200% | 92.2200% |
| img_mse | 0.007694944333 | 0.007649067219 | 0.007690512003 |
| psnr | 31.85095721 | 32.09247219 | 31.82741021 |
| img_ssim | 0.9416014874 | 0.9417770714 | 0.9414784641 |
| aud_mse | 0.002509089853 | 0.002391924905 | 0.002420955343 |
| aud_ssim | 0.9234900279 | 0.9269546106 | 0.9261392849 |
| aud_masked_mse | 0.006176221218 | 0.005887815131 | 0.005959274631 |
| aud_masked_l1 | 0.02116418124 | 0.0205923305 | 0.02065475956 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1501113447 | 0.1500455263 | 0.1503676044 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 88.5553% | 88.7232% | 88.8311% |

</details>

<details>
<summary>仅残缺音频 / feature_dropout：音频专项全部字段</summary>

family_group：`spectrogram_occlusion`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 97.1000% | 97.1000% | 97.1000% |
| img_mse | 0.002939252363 | 0.002914416999 | 0.002940109077 |
| psnr | 36.0209568 | 36.27273156 | 36.00376942 |
| img_ssim | 0.9794806812 | 0.9796060573 | 0.9794160336 |
| aud_mse | 0.001481112938 | 0.0009486985296 | 0.0009572662797 |
| aud_ssim | 0.9601202813 | 0.9736976358 | 0.9734175484 |
| aud_masked_mse | 0.003700818744 | 0.002370283177 | 0.00239165116 |
| aud_masked_l1 | 0.01584962049 | 0.01280620803 | 0.01287264701 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1514549498 | 0.1516230294 | 0.1515145656 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 90.7819% | 91.7119% | 91.7030% |

</details>

<details>
<summary>仅残缺音频 / partial_temporal：音频专项全部字段</summary>

family_group：`paper_aligned_time_gap`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 45.8800% | 45.8800% | 45.8800% |
| img_mse | 0.03934193166 | 0.03829270253 | 0.03731998438 |
| psnr | 17.78878238 | 17.92227048 | 17.88179701 |
| img_ssim | 0.6312104433 | 0.6495516552 | 0.6625311481 |
| aud_mse | 0.0108897884 | 0.01085890736 | 0.01096961021 |
| aud_ssim | 0.5384149365 | 0.5397922348 | 0.5390261048 |
| aud_masked_mse | 0.0268056334 | 0.02672961808 | 0.02700211719 |
| aud_masked_l1 | 0.07526028515 | 0.07409888403 | 0.0741106366 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1323257123 | 0.1314531773 | 0.1324566024 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 71.1527% | 71.1769% | 71.2771% |

</details>

<details>
<summary>仅残缺音频 / time_freq_block：音频专项全部字段</summary>

family_group：`spectrogram_occlusion`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 96.7500% | 96.7500% | 96.7500% |
| img_mse | 0.003434860813 | 0.003409354978 | 0.003440159147 |
| psnr | 36.07656951 | 36.32461522 | 36.07391398 |
| img_ssim | 0.9753842435 | 0.9755061541 | 0.9752642349 |
| aud_mse | 0.0008515375959 | 0.0007983488309 | 0.0008048859115 |
| aud_ssim | 0.979371387 | 0.9806154703 | 0.9804754872 |
| aud_masked_mse | 0.00515961235 | 0.004837332493 | 0.004876941901 |
| aud_masked_l1 | 0.01795495559 | 0.01736980207 | 0.01741033518 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1515181381 | 0.1514877222 | 0.1515485561 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 95.7505% | 95.8191% | 95.8570% |

</details>

<details>
<summary>干净图像+残缺音频 / time_mask：音频专项全部字段</summary>

family_group：`paper_aligned_time_gap`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 99.0300% | 99.0300% | 99.0300% |
| img_mse | 0.009772308242 | 0.009772308242 | 0.009772308242 |
| psnr | 20.77619474 | 20.77619474 | 20.77619474 |
| img_ssim | 0.9426264809 | 0.9426264809 | 0.9426264809 |
| aud_mse | 0.003023781309 | 0.002930759226 | 0.003016473691 |
| aud_ssim | 0.9156962244 | 0.9188755859 | 0.9189032138 |
| aud_masked_mse | 0.007443153918 | 0.007214176594 | 0.007425165942 |
| aud_masked_l1 | 0.02267431073 | 0.02211288065 | 0.02196166251 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1471098462 | 0.1476235157 | 0.1490076341 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 89.2857% | 89.2537% | 89.6610% |

</details>

<details>
<summary>干净图像+残缺音频 / freq_mask：音频专项全部字段</summary>

family_group：`spectrogram_occlusion`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 98.8400% | 98.8400% | 98.8400% |
| img_mse | 0.009762657514 | 0.009762657514 | 0.009762657514 |
| psnr | 20.78113918 | 20.78113918 | 20.78113918 |
| img_ssim | 0.9426926541 | 0.9426926541 | 0.9426926541 |
| aud_mse | 0.002372239015 | 0.002293973531 | 0.002359590735 |
| aud_ssim | 0.9275494032 | 0.9304506823 | 0.928885383 |
| aud_masked_mse | 0.005839357587 | 0.005646704089 | 0.005808223369 |
| aud_masked_l1 | 0.02074631183 | 0.02014290541 | 0.02013378111 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1493716673 | 0.1495797211 | 0.1508239555 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 88.5968% | 88.6947% | 89.1860% |

</details>

<details>
<summary>干净图像+残缺音频 / feature_dropout：音频专项全部字段</summary>

family_group：`spectrogram_occlusion`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 99.3200% | 99.3200% | 99.3200% |
| img_mse | 0.00973329923 | 0.00973329923 | 0.00973329923 |
| psnr | 20.79374686 | 20.79374686 | 20.79374686 |
| img_ssim | 0.942739988 | 0.942739988 | 0.942739988 |
| aud_mse | 0.001314147486 | 0.0007899013032 | 0.0008179364141 |
| aud_ssim | 0.9637205889 | 0.9774551545 | 0.9766598041 |
| aud_masked_mse | 0.00328360173 | 0.001973526457 | 0.002043541364 |
| aud_masked_l1 | 0.0150761658 | 0.01180962684 | 0.01194976779 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1515817175 | 0.1513788533 | 0.1517061435 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 90.8990% | 91.9442% | 91.9087% |

</details>

<details>
<summary>干净图像+残缺音频 / partial_temporal：音频专项全部字段</summary>

family_group：`paper_aligned_time_gap`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 98.0700% | 98.0700% | 98.0700% |
| img_mse | 0.009948950207 | 0.009948950207 | 0.009948950207 |
| psnr | 20.70150414 | 20.70150414 | 20.70150414 |
| img_ssim | 0.9421690227 | 0.9421690227 | 0.9421690227 |
| aud_mse | 0.009733709633 | 0.009778726122 | 0.01011742623 |
| aud_ssim | 0.5766152874 | 0.5759790228 | 0.576913965 |
| aud_masked_mse | 0.02395990073 | 0.02407071021 | 0.02490443355 |
| aud_masked_l1 | 0.07094715257 | 0.07001101228 | 0.06928070993 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1324527766 | 0.1325855958 | 0.1364162573 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 73.3127% | 73.5867% | 73.7006% |

</details>

<details>
<summary>干净图像+残缺音频 / time_freq_block：音频专项全部字段</summary>

family_group：`spectrogram_occlusion`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 99.1200% | 99.1200% | 99.1200% |
| img_mse | 0.009738052943 | 0.009738052943 | 0.009738052943 |
| psnr | 20.79080988 | 20.79080988 | 20.79080988 |
| img_ssim | 0.9427129206 | 0.9427129206 | 0.9427129206 |
| aud_mse | 0.0008051505313 | 0.000763986958 | 0.0007823672045 |
| aud_ssim | 0.9804057076 | 0.9814773455 | 0.9811295378 |
| aud_masked_mse | 0.004878545237 | 0.004629128063 | 0.00474049716 |
| aud_masked_l1 | 0.01761080545 | 0.01697668422 | 0.0169661741 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1512744673 | 0.1512877773 | 0.151631094 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 95.7135% | 95.7470% | 95.9784% |

</details>

<details>
<summary>双残缺 / time_mask：音频专项全部字段</summary>

family_group：`paper_aligned_time_gap`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 97.7100% | 97.7100% | 97.7100% |
| img_mse | 0.008400182439 | 0.008342494671 | 0.008338178958 |
| psnr | 23.24847541 | 23.29802882 | 23.33288668 |
| img_ssim | 0.950198068 | 0.9506006111 | 0.9507450939 |
| aud_mse | 0.003124714765 | 0.003026747897 | 0.003105988445 |
| aud_ssim | 0.9137972696 | 0.9172232037 | 0.9173071476 |
| aud_masked_mse | 0.007691605515 | 0.007450456326 | 0.007645509886 |
| aud_masked_l1 | 0.02323797907 | 0.02265567213 | 0.02251297696 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1472824425 | 0.1477179326 | 0.1489718895 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 89.0935% | 89.0651% | 89.4223% |

</details>

<details>
<summary>双残缺 / freq_mask：音频专项全部字段</summary>

family_group：`spectrogram_occlusion`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 97.9100% | 97.9100% | 97.9100% |
| img_mse | 0.008445940813 | 0.008385779788 | 0.008384603432 |
| psnr | 23.14202959 | 23.19274402 | 23.22740846 |
| img_ssim | 0.9498323706 | 0.9502467397 | 0.950376258 |
| aud_mse | 0.002420372422 | 0.002333033354 | 0.002396090774 |
| aud_ssim | 0.9252812969 | 0.9285822127 | 0.9269600659 |
| aud_masked_mse | 0.00595783983 | 0.005742851369 | 0.005898069561 |
| aud_masked_l1 | 0.02091210476 | 0.02029994751 | 0.02031819614 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1499808499 | 0.1500678698 | 0.151139963 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 88.5190% | 88.6610% | 89.0700% |

</details>

<details>
<summary>双残缺 / feature_dropout：音频专项全部字段</summary>

family_group：`spectrogram_occlusion`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 99.0300% | 99.0300% | 99.0300% |
| img_mse | 0.008364160635 | 0.008297055145 | 0.008300649044 |
| psnr | 23.2923449 | 23.35052498 | 23.3863947 |
| img_ssim | 0.950173947 | 0.9506594228 | 0.9507638533 |
| aud_mse | 0.001340888193 | 0.0008129757855 | 0.0008377781662 |
| aud_ssim | 0.963340916 | 0.9770864037 | 0.9763924413 |
| aud_masked_mse | 0.003351513963 | 0.002031798462 | 0.002093800471 |
| aud_masked_l1 | 0.01519876381 | 0.01194314387 | 0.01207448888 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1517175861 | 0.1516220759 | 0.1518705346 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 90.8771% | 91.9232% | 91.8781% |

</details>

<details>
<summary>双残缺 / partial_temporal：音频专项全部字段</summary>

family_group：`paper_aligned_time_gap`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 92.0300% | 92.0300% | 92.0300% |
| img_mse | 0.008659692102 | 0.008614761883 | 0.008598770045 |
| psnr | 22.98078659 | 23.00590612 | 22.99758459 |
| img_ssim | 0.9489094667 | 0.9491715032 | 0.9493107038 |
| aud_mse | 0.009967471337 | 0.009984299806 | 0.01030136759 |
| aud_ssim | 0.5694172061 | 0.5693533241 | 0.5701290045 |
| aud_masked_mse | 0.0245353141 | 0.02457673797 | 0.02535721245 |
| aud_masked_l1 | 0.07215869381 | 0.07111899968 | 0.07049502829 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1329129082 | 0.1326738854 | 0.1362676412 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 72.9936% | 73.2348% | 73.3401% |

</details>

<details>
<summary>双残缺 / time_freq_block：音频专项全部字段</summary>

family_group：`spectrogram_occlusion`。

| 指标 | control | v12b | no_causal |
|---|---|---|---|
| acc | 98.7700% | 98.7700% | 98.7700% |
| img_mse | 0.008381125343 | 0.008313903844 | 0.008328898995 |
| psnr | 23.26843915 | 23.32697118 | 23.35583488 |
| img_ssim | 0.9503676345 | 0.9508573534 | 0.9508950438 |
| aud_mse | 0.0008025279009 | 0.0007578328617 | 0.0007754509369 |
| aud_ssim | 0.9806737814 | 0.9818267635 | 0.9814042282 |
| aud_masked_mse | 0.004862654246 | 0.004591839346 | 0.004698590288 |
| aud_masked_l1 | 0.01747624641 | 0.01681593883 | 0.01683450817 |
| aud_visible_mse | 0 | 0 | 0 |
| aud_visible_l1 | 0 | 0 | 0 |
| rec_std | 0.1514070904 | 0.1513905429 | 0.1516913418 |
| tgt_std | 0.1520535966 | 0.1520535966 | 0.1520535966 |
| top15_recall | 95.7146% | 95.7739% | 95.9462% |

</details>

#### 音频能量诊断

下表来自 normal 日志的全测试集诊断，保持日志四位小数和召回一位百分数精度；不能用其舍入值替代 CSV 精确指标。
rec/tgt std 大体接近，未见整体零输出塌缩；长缺失的时频结构偏差依然明显。

<details>
<summary>fixed_mask：各 family/cue 的能量诊断</summary>

| family | 输入模式 | 实验 | rec mean | rec std | rec max | tgt mean | tgt std | tgt max | top15% 召回 |
|---|---|---|---|---|---|---|---|---|---|
| mask_horizontal/partial_temporal | clean_both | control | 0.0427 | 0.1550 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.9% |
| mask_horizontal/partial_temporal | clean_both | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| mask_horizontal/partial_temporal | clean_both | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| mask_horizontal/partial_temporal | clean_img_only | control | 0.0152 | 0.0695 | 0.9998 | 0.0156 | 0.0705 | 0.9879 | 83.3% |
| mask_horizontal/partial_temporal | clean_img_only | v12b | 0.0153 | 0.0698 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.0% |
| mask_horizontal/partial_temporal | clean_img_only | no_causal | 0.0153 | 0.0699 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.6% |
| mask_horizontal/partial_temporal | clean_aud_only | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| mask_horizontal/partial_temporal | clean_aud_only | v12b | 0.0411 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| mask_horizontal/partial_temporal | clean_aud_only | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.8% |
| mask_horizontal/partial_temporal | corrupt_img_only | control | 0.0156 | 0.0685 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 80.9% |
| mask_horizontal/partial_temporal | corrupt_img_only | v12b | 0.0154 | 0.0681 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 82.4% |
| mask_horizontal/partial_temporal | corrupt_img_only | no_causal | 0.0155 | 0.0690 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 82.9% |
| mask_horizontal/partial_temporal | corrupt_aud_only | control | 0.0403 | 0.1323 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 71.2% |
| mask_horizontal/partial_temporal | corrupt_aud_only | v12b | 0.0394 | 0.1315 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 71.2% |
| mask_horizontal/partial_temporal | corrupt_aud_only | no_causal | 0.0394 | 0.1325 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 71.3% |
| mask_horizontal/partial_temporal | clean_img_corrupt_aud | control | 0.0406 | 0.1325 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 73.3% |
| mask_horizontal/partial_temporal | clean_img_corrupt_aud | v12b | 0.0398 | 0.1326 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 73.6% |
| mask_horizontal/partial_temporal | clean_img_corrupt_aud | no_causal | 0.0393 | 0.1364 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 73.7% |
| mask_horizontal/partial_temporal | corrupt_img_clean_aud | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| mask_horizontal/partial_temporal | corrupt_img_clean_aud | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| mask_horizontal/partial_temporal | corrupt_img_clean_aud | no_causal | 0.0412 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| mask_horizontal/partial_temporal | corrupt_both | control | 0.0410 | 0.1335 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 73.1% |
| mask_horizontal/partial_temporal | corrupt_both | v12b | 0.0400 | 0.1329 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 73.3% |
| mask_horizontal/partial_temporal | corrupt_both | no_causal | 0.0396 | 0.1363 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 73.4% |
| mask_vertical/feature_dropout | clean_both | control | 0.0427 | 0.1550 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.9% |
| mask_vertical/feature_dropout | clean_both | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| mask_vertical/feature_dropout | clean_both | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| mask_vertical/feature_dropout | clean_img_only | control | 0.0152 | 0.0695 | 0.9998 | 0.0156 | 0.0705 | 0.9879 | 83.3% |
| mask_vertical/feature_dropout | clean_img_only | v12b | 0.0153 | 0.0698 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.0% |
| mask_vertical/feature_dropout | clean_img_only | no_causal | 0.0153 | 0.0699 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.6% |
| mask_vertical/feature_dropout | clean_aud_only | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| mask_vertical/feature_dropout | clean_aud_only | v12b | 0.0411 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| mask_vertical/feature_dropout | clean_aud_only | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.8% |
| mask_vertical/feature_dropout | corrupt_img_only | control | 0.0154 | 0.0686 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 81.9% |
| mask_vertical/feature_dropout | corrupt_img_only | v12b | 0.0152 | 0.0683 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 83.5% |
| mask_vertical/feature_dropout | corrupt_img_only | no_causal | 0.0153 | 0.0688 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 84.0% |
| mask_vertical/feature_dropout | corrupt_aud_only | control | 0.0408 | 0.1515 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 90.8% |
| mask_vertical/feature_dropout | corrupt_aud_only | v12b | 0.0408 | 0.1517 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 91.7% |
| mask_vertical/feature_dropout | corrupt_aud_only | no_causal | 0.0408 | 0.1516 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 91.7% |
| mask_vertical/feature_dropout | clean_img_corrupt_aud | control | 0.0409 | 0.1516 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 90.9% |
| mask_vertical/feature_dropout | clean_img_corrupt_aud | v12b | 0.0408 | 0.1514 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 92.0% |
| mask_vertical/feature_dropout | clean_img_corrupt_aud | no_causal | 0.0408 | 0.1518 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 91.9% |
| mask_vertical/feature_dropout | corrupt_img_clean_aud | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| mask_vertical/feature_dropout | corrupt_img_clean_aud | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| mask_vertical/feature_dropout | corrupt_img_clean_aud | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| mask_vertical/feature_dropout | corrupt_both | control | 0.0410 | 0.1518 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 90.9% |
| mask_vertical/feature_dropout | corrupt_both | v12b | 0.0409 | 0.1519 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 91.9% |
| mask_vertical/feature_dropout | corrupt_both | no_causal | 0.0409 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 91.9% |
| occlusion/time_mask | clean_both | control | 0.0427 | 0.1550 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.9% |
| occlusion/time_mask | clean_both | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| occlusion/time_mask | clean_both | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| occlusion/time_mask | clean_img_only | control | 0.0152 | 0.0695 | 0.9998 | 0.0156 | 0.0705 | 0.9879 | 83.3% |
| occlusion/time_mask | clean_img_only | v12b | 0.0153 | 0.0698 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.0% |
| occlusion/time_mask | clean_img_only | no_causal | 0.0153 | 0.0699 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.6% |
| occlusion/time_mask | clean_aud_only | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| occlusion/time_mask | clean_aud_only | v12b | 0.0411 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| occlusion/time_mask | clean_aud_only | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.8% |
| occlusion/time_mask | corrupt_img_only | control | 0.0152 | 0.0668 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 80.3% |
| occlusion/time_mask | corrupt_img_only | v12b | 0.0152 | 0.0667 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 81.7% |
| occlusion/time_mask | corrupt_img_only | no_causal | 0.0152 | 0.0675 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 82.2% |
| occlusion/time_mask | corrupt_aud_only | control | 0.0403 | 0.1473 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 89.1% |
| occlusion/time_mask | corrupt_aud_only | v12b | 0.0404 | 0.1475 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 89.1% |
| occlusion/time_mask | corrupt_aud_only | no_causal | 0.0405 | 0.1480 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 89.3% |
| occlusion/time_mask | clean_img_corrupt_aud | control | 0.0403 | 0.1470 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 89.3% |
| occlusion/time_mask | clean_img_corrupt_aud | v12b | 0.0404 | 0.1475 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 89.3% |
| occlusion/time_mask | clean_img_corrupt_aud | no_causal | 0.0404 | 0.1489 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 89.7% |
| occlusion/time_mask | corrupt_img_clean_aud | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| occlusion/time_mask | corrupt_img_clean_aud | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| occlusion/time_mask | corrupt_img_clean_aud | no_causal | 0.0412 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| occlusion/time_mask | corrupt_both | control | 0.0403 | 0.1469 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.7% |
| occlusion/time_mask | corrupt_both | v12b | 0.0404 | 0.1474 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.7% |
| occlusion/time_mask | corrupt_both | no_causal | 0.0405 | 0.1487 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 89.1% |
| pixel_delete/freq_mask | clean_both | control | 0.0427 | 0.1550 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.9% |
| pixel_delete/freq_mask | clean_both | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| pixel_delete/freq_mask | clean_both | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| pixel_delete/freq_mask | clean_img_only | control | 0.0152 | 0.0695 | 0.9998 | 0.0156 | 0.0705 | 0.9879 | 83.3% |
| pixel_delete/freq_mask | clean_img_only | v12b | 0.0153 | 0.0698 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.0% |
| pixel_delete/freq_mask | clean_img_only | no_causal | 0.0153 | 0.0699 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.6% |
| pixel_delete/freq_mask | clean_aud_only | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| pixel_delete/freq_mask | clean_aud_only | v12b | 0.0411 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| pixel_delete/freq_mask | clean_aud_only | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.8% |
| pixel_delete/freq_mask | corrupt_img_only | control | 0.0156 | 0.0689 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 81.9% |
| pixel_delete/freq_mask | corrupt_img_only | v12b | 0.0154 | 0.0683 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 83.4% |
| pixel_delete/freq_mask | corrupt_img_only | no_causal | 0.0154 | 0.0689 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 84.0% |
| pixel_delete/freq_mask | corrupt_aud_only | control | 0.0410 | 0.1502 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.5% |
| pixel_delete/freq_mask | corrupt_aud_only | v12b | 0.0409 | 0.1502 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.7% |
| pixel_delete/freq_mask | corrupt_aud_only | no_causal | 0.0410 | 0.1505 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.8% |
| pixel_delete/freq_mask | clean_img_corrupt_aud | control | 0.0410 | 0.1495 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.6% |
| pixel_delete/freq_mask | clean_img_corrupt_aud | v12b | 0.0408 | 0.1497 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.7% |
| pixel_delete/freq_mask | clean_img_corrupt_aud | no_causal | 0.0409 | 0.1509 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 89.2% |
| pixel_delete/freq_mask | corrupt_img_clean_aud | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| pixel_delete/freq_mask | corrupt_img_clean_aud | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| pixel_delete/freq_mask | corrupt_img_clean_aud | no_causal | 0.0412 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| pixel_delete/freq_mask | corrupt_both | control | 0.0411 | 0.1502 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.6% |
| pixel_delete/freq_mask | corrupt_both | v12b | 0.0409 | 0.1501 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.8% |
| pixel_delete/freq_mask | corrupt_both | no_causal | 0.0409 | 0.1511 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 89.1% |
| salt_mask/time_freq_block | clean_both | control | 0.0427 | 0.1550 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.9% |
| salt_mask/time_freq_block | clean_both | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| salt_mask/time_freq_block | clean_both | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| salt_mask/time_freq_block | clean_img_only | control | 0.0152 | 0.0695 | 0.9998 | 0.0156 | 0.0705 | 0.9879 | 83.3% |
| salt_mask/time_freq_block | clean_img_only | v12b | 0.0153 | 0.0698 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.0% |
| salt_mask/time_freq_block | clean_img_only | no_causal | 0.0153 | 0.0699 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.6% |
| salt_mask/time_freq_block | clean_aud_only | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| salt_mask/time_freq_block | clean_aud_only | v12b | 0.0411 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| salt_mask/time_freq_block | clean_aud_only | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.8% |
| salt_mask/time_freq_block | corrupt_img_only | control | 0.0158 | 0.0661 | 0.9986 | 0.0156 | 0.0705 | 0.9879 | 78.9% |
| salt_mask/time_freq_block | corrupt_img_only | v12b | 0.0156 | 0.0660 | 0.9984 | 0.0156 | 0.0705 | 0.9879 | 80.2% |
| salt_mask/time_freq_block | corrupt_img_only | no_causal | 0.0157 | 0.0673 | 0.9998 | 0.0156 | 0.0705 | 0.9879 | 80.8% |
| salt_mask/time_freq_block | corrupt_aud_only | control | 0.0408 | 0.1515 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 95.7% |
| salt_mask/time_freq_block | corrupt_aud_only | v12b | 0.0408 | 0.1514 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 95.8% |
| salt_mask/time_freq_block | corrupt_aud_only | no_causal | 0.0408 | 0.1515 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 95.8% |
| salt_mask/time_freq_block | clean_img_corrupt_aud | control | 0.0408 | 0.1512 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 95.7% |
| salt_mask/time_freq_block | clean_img_corrupt_aud | v12b | 0.0407 | 0.1512 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 95.7% |
| salt_mask/time_freq_block | clean_img_corrupt_aud | no_causal | 0.0407 | 0.1516 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 95.9% |
| salt_mask/time_freq_block | corrupt_img_clean_aud | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| salt_mask/time_freq_block | corrupt_img_clean_aud | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| salt_mask/time_freq_block | corrupt_img_clean_aud | no_causal | 0.0412 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| salt_mask/time_freq_block | corrupt_both | control | 0.0408 | 0.1515 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 95.7% |
| salt_mask/time_freq_block | corrupt_both | v12b | 0.0407 | 0.1514 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 95.8% |
| salt_mask/time_freq_block | corrupt_both | no_causal | 0.0408 | 0.1517 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 95.9% |

</details>

<details>
<summary>legacy_random：各 family/cue 的能量诊断</summary>

| family | 输入模式 | 实验 | rec mean | rec std | rec max | tgt mean | tgt std | tgt max | top15% 召回 |
|---|---|---|---|---|---|---|---|---|---|
| occlusion/time_mask | clean_both | control | 0.0427 | 0.1550 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.9% |
| occlusion/time_mask | clean_both | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| occlusion/time_mask | clean_both | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| occlusion/time_mask | clean_img_only | control | 0.0152 | 0.0695 | 0.9998 | 0.0156 | 0.0705 | 0.9879 | 83.3% |
| occlusion/time_mask | clean_img_only | v12b | 0.0153 | 0.0698 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.0% |
| occlusion/time_mask | clean_img_only | no_causal | 0.0153 | 0.0699 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 85.6% |
| occlusion/time_mask | clean_aud_only | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| occlusion/time_mask | clean_aud_only | v12b | 0.0411 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| occlusion/time_mask | clean_aud_only | no_causal | 0.0411 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.8% |
| occlusion/time_mask | corrupt_img_only | control | 0.0154 | 0.0677 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 81.0% |
| occlusion/time_mask | corrupt_img_only | v12b | 0.0153 | 0.0674 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 82.5% |
| occlusion/time_mask | corrupt_img_only | no_causal | 0.0153 | 0.0681 | 1.0000 | 0.0156 | 0.0705 | 0.9879 | 83.0% |
| occlusion/time_mask | corrupt_aud_only | control | 0.0408 | 0.1473 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 87.3% |
| occlusion/time_mask | corrupt_aud_only | v12b | 0.0406 | 0.1472 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 87.6% |
| occlusion/time_mask | corrupt_aud_only | no_causal | 0.0407 | 0.1475 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 87.7% |
| occlusion/time_mask | clean_img_corrupt_aud | control | 0.0408 | 0.1471 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 87.8% |
| occlusion/time_mask | clean_img_corrupt_aud | v12b | 0.0407 | 0.1472 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.1% |
| occlusion/time_mask | clean_img_corrupt_aud | no_causal | 0.0406 | 0.1485 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.4% |
| occlusion/time_mask | corrupt_img_clean_aud | control | 0.0427 | 0.1549 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 81.8% |
| occlusion/time_mask | corrupt_img_clean_aud | v12b | 0.0412 | 0.1523 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.7% |
| occlusion/time_mask | corrupt_img_clean_aud | no_causal | 0.0412 | 0.1521 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 85.9% |
| occlusion/time_mask | corrupt_both | control | 0.0407 | 0.1472 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.3% |
| occlusion/time_mask | corrupt_both | v12b | 0.0405 | 0.1472 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.6% |
| occlusion/time_mask | corrupt_both | no_causal | 0.0405 | 0.1483 | 1.0000 | 0.0406 | 0.1521 | 1.0000 | 88.8% |

</details>

#### 训练统计与权重验证

control 额外训练 0 轮；两训练组都有连续 epoch 0-29。主模型 loss 包含因果项，不能与 no-causal 总 loss 直接比较优劣。
各组日志均存在 libgomp 的 OMP_NUM_THREADS 值警告，后续训练/评估正常完成；当前产物中未见训练 NaN、OOM 或 traceback。

| 实验 | 首轮 loss | 末轮 loss | 最低 epoch loss | 最低 loss epoch | 首轮 LR | 末轮 LR | 日志采样 batch 数 |
|---|---|---|---|---|---|---|---|
| v12b | 1.161 | 1.146 | 1.0785 | 28 | 0.0001 | 1e-05 | 330 |
| no_causal | 1.1178 | 1.1019 | 1.0353 | 28 | 0.0001 | 1e-05 | 330 |

<details>
<summary>全部 epoch loss 与 LR</summary>

| epoch（0-based） | v12b loss | no_causal loss | v12b LR | no_causal LR |
|---|---|---|---|---|
| 0 | 1.161 | 1.1178 | 0.0001 | 0.0001 |
| 1 | 1.2122 | 1.167 | 0.0001 | 0.0001 |
| 2 | 1.2779 | 1.2296 | 9.9e-05 | 9.9e-05 |
| 3 | 1.1595 | 1.1144 | 9.8e-05 | 9.8e-05 |
| 4 | 1.2155 | 1.1711 | 9.6e-05 | 9.6e-05 |
| 5 | 1.1475 | 1.101 | 9.4e-05 | 9.4e-05 |
| 6 | 1.2305 | 1.1848 | 9.1e-05 | 9.1e-05 |
| 7 | 1.2283 | 1.1843 | 8.8e-05 | 8.8e-05 |
| 8 | 1.1506 | 1.1046 | 8.5e-05 | 8.5e-05 |
| 9 | 1.1813 | 1.1379 | 8.1e-05 | 8.1e-05 |
| 10 | 1.196 | 1.1524 | 7.7e-05 | 7.7e-05 |
| 11 | 1.122 | 1.0754 | 7.3e-05 | 7.3e-05 |
| 12 | 1.1404 | 1.0968 | 6.9e-05 | 6.9e-05 |
| 13 | 1.2247 | 1.1804 | 6.4e-05 | 6.4e-05 |
| 14 | 1.1053 | 1.0585 | 6e-05 | 6e-05 |
| 15 | 1.1436 | 1.0983 | 5.5e-05 | 5.5e-05 |
| 16 | 1.0889 | 1.0452 | 5e-05 | 5e-05 |
| 17 | 1.0858 | 1.0412 | 4.6e-05 | 4.6e-05 |
| 18 | 1.0843 | 1.0398 | 4.1e-05 | 4.1e-05 |
| 19 | 1.104 | 1.0599 | 3.7e-05 | 3.7e-05 |
| 20 | 1.1053 | 1.0605 | 3.3e-05 | 3.3e-05 |
| 21 | 1.1017 | 1.0571 | 2.9e-05 | 2.9e-05 |
| 22 | 1.1122 | 1.0672 | 2.5e-05 | 2.5e-05 |
| 23 | 1.1732 | 1.1287 | 2.2e-05 | 2.2e-05 |
| 24 | 1.1571 | 1.1126 | 1.9e-05 | 1.9e-05 |
| 25 | 1.1278 | 1.0829 | 1.6e-05 | 1.6e-05 |
| 26 | 1.1533 | 1.1082 | 1.4e-05 | 1.4e-05 |
| 27 | 1.103 | 1.0583 | 1.2e-05 | 1.2e-05 |
| 28 | 1.0785 | 1.0353 | 1.1e-05 | 1.1e-05 |
| 29 | 1.146 | 1.1019 | 1e-05 | 1e-05 |

</details>

<details>
<summary>训练日志已打印字段的统计</summary>

每个字段仅统计日志中有该字段的 batch，给出采样均值、min/max、记录次数；不是所有 14040 步的精确 epoch 均值。

| 日志字段 | 实验 | 采样均值 | min | max | 出现次数 |
|---|---|---|---|---|---|
| adapter_step | v12b | 1 | 1 | 1 | 330 |
| adapter_step | no_causal | 1 | 1 | 1 | 330 |
| aud(cat) | v12b | 0.02892926829 | 0.0148 | 0.0447 | 41 |
| aud(cat) | no_causal | 0.02785609756 | 0.0144 | 0.0398 | 41 |
| aud(sam) | v12b | 0.06744290657 | 0.0318 | 0.1349 | 289 |
| aud(sam) | no_causal | 0.06662629758 | 0.0308 | 0.1359 | 289 |
| aud2img_correct | v12b | 0.03090462963 | 0.0001 | 0.0638 | 216 |
| aud2img_correct | no_causal | N/A | N/A | N/A | 0 |
| aud2img_correct_rel | v12b | 0.663174537 | 0.0116 | 0.987 | 216 |
| aud2img_correct_rel | no_causal | N/A | N/A | N/A | 0 |
| aud2img_gain | v12b | 0.001281944444 | -0.0001 | 0.0033 | 216 |
| aud2img_gain | no_causal | N/A | N/A | N/A | 0 |
| aud2img_n | v12b | 128 | 128 | 128 | 216 |
| aud2img_n | no_causal | N/A | N/A | N/A | 0 |
| aud2img_win_zero | v12b | 0.6231194444 | 0.0547 | 0.7891 | 216 |
| aud2img_win_zero | no_causal | N/A | N/A | N/A | 0 |
| aud2img_wrong | v12b | 0.03097731481 | 0.0001 | 0.0638 | 216 |
| aud2img_wrong | no_causal | N/A | N/A | N/A | 0 |
| aud2img_wrong_n | v12b | 128 | 128 | 128 | 216 |
| aud2img_wrong_n | no_causal | N/A | N/A | N/A | 0 |
| aud2img_wrong_rel | v12b | 0.6655041667 | 0.0141 | 0.9895 | 216 |
| aud2img_wrong_rel | no_causal | N/A | N/A | N/A | 0 |
| aud2img_zero | v12b | 0.03218148148 | 0.0001 | 0.0665 | 216 |
| aud2img_zero | no_causal | N/A | N/A | N/A | 0 |
| aud_act | v12b | 0.002824913495 | 0 | 0.0083 | 289 |
| aud_act | no_causal | 0.002553633218 | 0 | 0.0084 | 289 |
| aud_mask_l1 | v12b | 0.01611454545 | 0.0081 | 0.0244 | 165 |
| aud_mask_l1 | no_causal | 0.01587878788 | 0.0082 | 0.024 | 165 |
| aud_mask_mse | v12b | 0.004071515152 | 0.0009 | 0.0071 | 165 |
| aud_mask_mse | no_causal | 0.004103636364 | 0.0009 | 0.0072 | 165 |
| aud_mask_wmse | v12b | 0.008503636364 | 0.0021 | 0.0144 | 165 |
| aud_mask_wmse | no_causal | 0.008418787879 | 0.0022 | 0.0151 | 165 |
| aud_visible_l1 | v12b | 0 | 0 | 0 | 165 |
| aud_visible_l1 | no_causal | 0 | 0 | 0 | 165 |
| aud_visible_mse | v12b | 0 | 0 | 0 | 165 |
| aud_visible_mse | no_causal | 0 | 0 | 0 | 165 |
| aux_aud | v12b | 0.0617733564 | 0 | 0.2375 | 289 |
| aux_aud | no_causal | 0.0617733564 | 0 | 0.2375 | 289 |
| bind_aud | v12b | 0.001607878788 | 0.0004 | 0.0023 | 330 |
| bind_aud | no_causal | 0.001607878788 | 0.0004 | 0.0023 | 330 |
| bind_img | v12b | 0.0006648484848 | 0.0002 | 0.0009 | 330 |
| bind_img | no_causal | 0.0006648484848 | 0.0002 | 0.0009 | 330 |
| cls | v12b | 0.05725818182 | 0 | 0.6309 | 330 |
| cls | no_causal | 0.05725818182 | 0 | 0.6309 | 330 |
| cross_aud2img | v12b | 0.09028981481 | 0.0737 | 0.1102 | 216 |
| cross_aud2img | no_causal | N/A | N/A | N/A | 0 |
| cross_img2aud | v12b | 0.0942083871 | 0.0803 | 0.115 | 155 |
| cross_img2aud | no_causal | N/A | N/A | N/A | 0 |
| cross_pair | v12b | 0.09212475248 | 0.0737 | 0.115 | 303 |
| cross_pair | no_causal | N/A | N/A | N/A | 0 |
| idx_cons | v12b | 0.003051785714 | 0.0009 | 0.0068 | 224 |
| idx_cons | no_causal | 0.003051785714 | 0.0009 | 0.0068 | 224 |
| img(cat) | v12b | 0.07958202247 | 0.0636 | 0.1094 | 89 |
| img(cat) | no_causal | 0.07952921348 | 0.0636 | 0.109 | 89 |
| img(sam) | v12b | 0.2398896266 | 0.1366 | 0.3388 | 241 |
| img(sam) | no_causal | 0.2393979253 | 0.1366 | 0.3388 | 241 |
| img2aud_correct | v12b | 0.003080645161 | 0.0001 | 0.006 | 155 |
| img2aud_correct | no_causal | N/A | N/A | N/A | 0 |
| img2aud_correct_rel | v12b | 0.6047845161 | 0.0265 | 1.0019 | 155 |
| img2aud_correct_rel | no_causal | N/A | N/A | N/A | 0 |
| img2aud_gain | v12b | 5.935483871e-05 | -0.0001 | 0.0002 | 155 |
| img2aud_gain | no_causal | N/A | N/A | N/A | 0 |
| img2aud_n | v12b | 128 | 128 | 128 | 155 |
| img2aud_n | no_causal | N/A | N/A | N/A | 0 |
| img2aud_win_zero | v12b | 0.5701064516 | 0.1562 | 0.9922 | 155 |
| img2aud_win_zero | no_causal | N/A | N/A | N/A | 0 |
| img2aud_wrong | v12b | 0.003090967742 | 0.0001 | 0.006 | 155 |
| img2aud_wrong | no_causal | N/A | N/A | N/A | 0 |
| img2aud_wrong_n | v12b | 128 | 128 | 128 | 155 |
| img2aud_wrong_n | no_causal | N/A | N/A | N/A | 0 |
| img2aud_wrong_rel | v12b | 0.6070619355 | 0.0287 | 1.0034 | 155 |
| img2aud_wrong_rel | no_causal | N/A | N/A | N/A | 0 |
| img2aud_zero | v12b | 0.00314 | 0.0002 | 0.0061 | 155 |
| img2aud_zero | no_causal | N/A | N/A | N/A | 0 |
| img_mask_l1 | v12b | 0.1198528302 | 0.0365 | 0.1443 | 159 |
| img_mask_l1 | no_causal | 0.1185213836 | 0.0359 | 0.1443 | 159 |
| img_mask_mse | v12b | 0.05175283019 | 0.0142 | 0.0664 | 159 |
| img_mask_mse | no_causal | 0.05191446541 | 0.0143 | 0.0664 | 159 |
| img_visible_l1 | v12b | 0.0001 | 0.0001 | 0.0001 | 159 |
| img_visible_l1 | no_causal | 0.0001 | 0.0001 | 0.0001 | 159 |
| img_visible_mse | v12b | 0 | 0 | 0 | 159 |
| img_visible_mse | no_causal | 0 | 0 | 0 | 159 |
| key_align | v12b | 2.9488575 | 2.6094 | 3.3397 | 200 |
| key_align | no_causal | 2.9488575 | 2.6094 | 3.3397 | 200 |
| loss | v12b | 1.02927697 | 0.2444 | 2.1017 | 330 |
| loss | no_causal | 0.982970303 | 0.1912 | 2.0482 | 330 |
| sev | v12b | 0.4 | 0.4 | 0.4 | 330 |
| sev | no_causal | 0.4 | 0.4 | 0.4 | 330 |
| soft_cls | v12b | 0.1231741071 | 0 | 1.0417 | 224 |
| soft_cls | no_causal | 0.1231741071 | 0 | 1.0417 | 224 |

</details>

| checkpoint | epoch | state 参数元素数 | SHA256 |
|---|---|---|---|
| cross_modal_snn_v12a.pt | 29 | 22831770 | 0a49aaf709ed257d7a6361ec5c09dcdd7b3e01ce90548070a3ff178a5402290e |
| cross_modal_snn_v12b.pt | 29 | 22975434 | afa97ee547233dedcac3a1c8ea7af795cfaaa1830dccfc87741a660af92cde1a |
| cross_modal_snn_v12b_no_causal.pt | 29 | 22975434 | 8962f252f77391967a67610ad7687e41fc38c33fb5fc1b288a79870fbd3fef90 |

主实验/no-causal 相对父模型均新增 20 个 tensor、修改 34 个既有 tensor；其余既有 tensor 与父模型逐元素相同。
所有变化均落入 trainable_prefixes 白名单，未发现冻结主干参数漂移。state 元素数包含构建但可能旁路的模块，不代表活跃计算量。
optimizer 的 step 按参数分组依次为 4922、7059、8379、11903、14040，条件通路仅在相应 cue 时更新，
14040=30×468，与完整训练日志一致；参数 step 不同不能解释为有部分 epoch 丢失。

#### Demo 小样本

三组 fixed/random 均使用 sample_seed=4321，从整个测试集随机抽取同一组 10 个索引：
`[6061, 555, 3548, 353, 6496, 7994, 1605, 9428, 7739, 2226]`。
fixed demo 每个 family 展示两个样本；random demo 使用默认 occlusion/time_mask 的随机位置，
其 family 覆盖不等于 random 全量数值评估，且只作可视化，不能替代 n=10000 的主指标。
对照主模型/父模型 fixed 双残缺图：随机点缺失的局部谐波更接近 target，长时间缺失的数字 3/5 仍出现过度平滑、
错位或能量缺失。图中 audio=类别 来自 Index 分类标签，不能当作独立的恢复语音识别正确证明。

| 协议 | 实验 | 模式 | Index ACC | 图 SSIM | 图 MSE | 音 SSIM | 音 MSE | 图 maskMSE | 音 maskMSE | 图 target | 音 target |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_mask | control | audio-only | 90.0% | 0.959 | 0.0059 | 0.862 | 0.0026 | 0.0059 | 0.0068 | category | sample |
| fixed_mask | control | image-only | 90.0% | 0.966 | 0.0065 | 0.921 | 0.0004 | 0.0184 | 0.0004 | sample | category |
| fixed_mask | control | image+audio | 100.0% | 0.966 | 0.0063 | 0.881 | 0.0023 | 0.0183 | 0.0060 | sample | sample |
| fixed_mask | v12b | audio-only | 90.0% | 0.959 | 0.0058 | 0.875 | 0.0024 | 0.0058 | 0.0064 | category | sample |
| fixed_mask | v12b | image-only | 90.0% | 0.966 | 0.0065 | 0.927 | 0.0003 | 0.0184 | 0.0003 | sample | category |
| fixed_mask | v12b | image+audio | 100.0% | 0.967 | 0.0062 | 0.886 | 0.0022 | 0.0180 | 0.0059 | sample | sample |
| fixed_mask | no_causal | audio-only | 90.0% | 0.959 | 0.0058 | 0.877 | 0.0024 | 0.0058 | 0.0065 | category | sample |
| fixed_mask | no_causal | image-only | 90.0% | 0.966 | 0.0065 | 0.918 | 0.0004 | 0.0184 | 0.0004 | sample | category |
| fixed_mask | no_causal | image+audio | 100.0% | 0.967 | 0.0061 | 0.884 | 0.0023 | 0.0179 | 0.0062 | sample | sample |
| legacy_random | control | audio-only | 100.0% | 0.966 | 0.0050 | 0.886 | 0.0022 | 0.0050 | 0.0054 | category | sample |
| legacy_random | control | image-only | 100.0% | 0.931 | 0.0136 | 0.950 | 0.0003 | 0.0881 | 0.0003 | sample | category |
| legacy_random | control | image+audio | 100.0% | 0.945 | 0.0099 | 0.827 | 0.0036 | 0.0639 | 0.0088 | sample | sample |
| legacy_random | v12b | audio-only | 100.0% | 0.966 | 0.0049 | 0.899 | 0.0020 | 0.0049 | 0.0050 | category | sample |
| legacy_random | v12b | image-only | 100.0% | 0.931 | 0.0136 | 0.955 | 0.0002 | 0.0881 | 0.0002 | sample | category |
| legacy_random | v12b | image+audio | 100.0% | 0.946 | 0.0098 | 0.847 | 0.0031 | 0.0635 | 0.0075 | sample | sample |
| legacy_random | no_causal | audio-only | 100.0% | 0.966 | 0.0050 | 0.907 | 0.0020 | 0.0050 | 0.0050 | category | sample |
| legacy_random | no_causal | image-only | 100.0% | 0.931 | 0.0136 | 0.956 | 0.0002 | 0.0881 | 0.0002 | sample | category |
| legacy_random | no_causal | image+audio | 100.0% | 0.946 | 0.0097 | 0.848 | 0.0032 | 0.0627 | 0.0079 | sample | sample |

<details>
<summary>Demo 能量诊断与逐样本分类</summary>

| 协议 | 实验 | 模式 | rec mean | rec std | rec max | tgt mean | tgt std | tgt max | top15% 召回 |
|---|---|---|---|---|---|---|---|---|---|
| fixed_mask | control | audio-only | 0.0342 | 0.1326 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 83.2% |
| fixed_mask | control | image-only | 0.0141 | 0.0657 | 1.0000 | 0.0131 | 0.0644 | 0.9879 | 77.2% |
| fixed_mask | control | image+audio | 0.0333 | 0.1310 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 83.6% |
| fixed_mask | v12b | audio-only | 0.0340 | 0.1329 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 83.3% |
| fixed_mask | v12b | image-only | 0.0139 | 0.0650 | 1.0000 | 0.0131 | 0.0644 | 0.9879 | 78.5% |
| fixed_mask | v12b | image+audio | 0.0330 | 0.1320 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 83.8% |
| fixed_mask | no_causal | audio-only | 0.0341 | 0.1341 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 83.2% |
| fixed_mask | no_causal | image-only | 0.0140 | 0.0658 | 1.0000 | 0.0131 | 0.0644 | 0.9879 | 79.1% |
| fixed_mask | no_causal | image+audio | 0.0332 | 0.1343 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 84.3% |
| legacy_random | control | audio-only | 0.0350 | 0.1325 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 82.8% |
| legacy_random | control | image-only | 0.0130 | 0.0611 | 0.9955 | 0.0131 | 0.0644 | 0.9879 | 76.5% |
| legacy_random | control | image+audio | 0.0326 | 0.1231 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 83.9% |
| legacy_random | v12b | audio-only | 0.0343 | 0.1319 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 83.7% |
| legacy_random | v12b | image-only | 0.0130 | 0.0611 | 0.9931 | 0.0131 | 0.0644 | 0.9879 | 77.9% |
| legacy_random | v12b | image+audio | 0.0333 | 0.1243 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 84.0% |
| legacy_random | no_causal | audio-only | 0.0341 | 0.1322 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 83.4% |
| legacy_random | no_causal | image-only | 0.0129 | 0.0614 | 1.0000 | 0.0131 | 0.0644 | 0.9879 | 78.5% |
| legacy_random | no_causal | image+audio | 0.0334 | 0.1269 | 1.0000 | 0.0318 | 0.1314 | 1.0000 | 83.9% |

fixed_mask 三组分类逐样本完全相同：

| 序号 | 真值 | audio-only | image-only | image+audio |
|---|---|---|---|---|
| 0 | 3 | 9✗ | 9✗ | 3✓ |
| 1 | 4 | 4✓ | 4✓ | 4✓ |
| 2 | 3 | 3✓ | 3✓ | 3✓ |
| 3 | 6 | 6✓ | 6✓ | 6✓ |
| 4 | 9 | 9✓ | 9✓ | 9✓ |
| 5 | 2 | 2✓ | 2✓ | 2✓ |
| 6 | 3 | 3✓ | 3✓ | 3✓ |
| 7 | 5 | 5✓ | 5✓ | 5✓ |
| 8 | 2 | 2✓ | 2✓ | 2✓ |
| 9 | 0 | 0✓ | 0✓ | 0✓ |

legacy_random 三组分类逐样本完全相同：

| 序号 | 真值 | audio-only | image-only | image+audio |
|---|---|---|---|---|
| 0 | 3 | 3✓ | 3✓ | 3✓ |
| 1 | 4 | 4✓ | 4✓ | 4✓ |
| 2 | 3 | 3✓ | 3✓ | 3✓ |
| 3 | 6 | 6✓ | 6✓ | 6✓ |
| 4 | 9 | 9✓ | 9✓ | 9✓ |
| 5 | 2 | 2✓ | 2✓ | 2✓ |
| 6 | 3 | 3✓ | 3✓ | 3✓ |
| 7 | 5 | 5✓ | 5✓ | 5✓ |
| 8 | 2 | 2✓ | 2✓ | 2✓ |
| 9 | 0 | 0✓ | 0✓ | 0✓ |

</details>

#### 独立结论与比较限制

- control：父 v12a checkpoint 的 SHA256 与设计记录一致，fixed/random 全部共有指标与旧 v12a 相同；修复后的 control 可比。
- v12b：音频干净重建明显改善，稀疏缺失恢复改善较大，主干稳定；长时间缺失优先验收未达成，不能宣称全面解决音频恢复。
- no_causal：同预算条件下音频 MSE 总体略高，双残缺 Cross-Key 边际收益可为负；部分内容识别与图像 MSE 略优，说明因果项有取舍。
- 主模型相对 control 的结构变化与额外 30 轮训练捆绑；不能将全部增益归于多尺度或中间 Cross-Key。
- 新中间 adapter 与最终 adapter 的干预联合发生，尚无独立尺度消融，且 CSV 未保存 mid gate/ratio。
- 同类错误 Key 不应被当作实例绑定反例；MNIST/FSDD 仅同类别关联，无法恢复任意指定录音的全部实例细节。
- Index ACC 恒定与冻结主干一致，partial_temporal 音频-only 分类 45.88% 仍未解决；外部识别、多种子置信区间未计算。
- standalone family CSV、主 fixed、random 与 demo 的 mask 口径分开；缺项和 N/A 均已列明。当前未形成新版本修改方案。

#### 完成检查与来源

12 个 normal/sweep CSV 共 8928 行；共有字段逐项相同，没有把 normal 与 sweep 重复计入均值。
主指标 n=10000；sweep 每个有效同类置换的 n 原样列在原始指标表；无效区域和未启用 Recall 的 n=0 显示 N/A。

源 sweep 的 5760 条 value/n 已全部进入本条完整对照表，覆盖全部 52 类已记录指标。

<details>
<summary>证据文件及校验和</summary>

| 文件 | 字节 | SHA256 |
|---|---|---|
| v12b_outputs/outputs/outputs_v12b_control/figures/demo_aud_only.png | 241913 | a9d4acc062b1c846fdd7799d668bdb9376b9ab3b89fc6bc3a9c1c334a771a25e |
| v12b_outputs/outputs/outputs_v12b_control/figures/demo_aud_only_random.png | 192879 | 3892013ab7bf07c0262da852507bcba6d941a80fee7e1637af1e22a3f6135d69 |
| v12b_outputs/outputs/outputs_v12b_control/figures/demo_both.png | 303036 | 5f75bc209f73edee56a18de39a739ad8a97d6a4feabd6f476f778707736be84a |
| v12b_outputs/outputs/outputs_v12b_control/figures/demo_both_random.png | 216152 | 3b41957f11fa88bca150cb0b6cf47663dc0e27563d547c56f3acf491e1f73a07 |
| v12b_outputs/outputs/outputs_v12b_control/figures/demo_img_only.png | 208956 | c08e6b6dce8dad4b164f7d989abaa912f00c424bfd9720d6f25fafccb80fa8c1 |
| v12b_outputs/outputs/outputs_v12b_control/figures/demo_img_only_random.png | 171485 | 0a9fd1203ead0f5ff11a6d01673ec09190d7e2eba24a3bb6c008203039538a53 |
| v12b_outputs/outputs/outputs_v12b_control/logs/demo_fixed_mask.log | 3093 | 89f2f78afee5a35a91b4b8232b687a087d53aec9c7cbb027e5f9b27a8622130d |
| v12b_outputs/outputs/outputs_v12b_control/logs/demo_legacy_random.log | 2762 | 5741f4d8660cf4713fe3c9ef3793e66c595dd45eaecc73316ac78569ea8ca7ed |
| v12b_outputs/outputs/outputs_v12b_control/logs/eval_fixed_mask_normal.log | 366690 | 598abe17202c7ccc482488eaf5e6632646d8bef1a394960cd4a61bbde2c173dc |
| v12b_outputs/outputs/outputs_v12b_control/logs/eval_fixed_mask_sweep.log | 291920 | 3289b67b1394067e76ada60556808cd5f2b3cce79d9220ac176a6886785bdbc6 |
| v12b_outputs/outputs/outputs_v12b_control/logs/eval_legacy_random_normal.log | 55535 | f8efdede35672979590b47271cba686c803fed6fd17bfeb86b677ecd8af7a822 |
| v12b_outputs/outputs/outputs_v12b_control/logs/eval_legacy_random_sweep.log | 61181 | bc59dd5b1da3cf14a18622a47d36e5a3fe02ab219cf0c744dc48425e9213e3cd |
| v12b_outputs/outputs/outputs_v12b_control/tables/audio_family_breakdown_fixed.csv | 4181 | 08004dd14f797eb0d0b31b11fc045666d119f628a84ea5e5dd60cc1633eec34f |
| v12b_outputs/outputs/outputs_v12b_control/tables/demo_eval_table.txt | 1262 | bb573ce9b4b2b61610fcafb8d76061dc2352d7b2434c6bbcdc8a037e3859e81f |
| v12b_outputs/outputs/outputs_v12b_control/tables/demo_eval_table_random.txt | 1262 | 5abc9f62999e9b3eef162cff863765e97be08f26c9e413184d8f2309a11016e7 |
| v12b_outputs/outputs/outputs_v12b_control/tables/eval_fixed_mask_sev0.4_key_normal_detail_normal.csv | 92056 | 53484fe0b0930b28fded9611914123a373d975cefae14db136c2c5d686ddccda |
| v12b_outputs/outputs/outputs_v12b_control/tables/eval_fixed_mask_sev0.4_key_sweep_detail_normal.csv | 174988 | d2a1f375cb203b49804036995ffb719426aa10fcbb1489d7b6015193eda95bf4 |
| v12b_outputs/outputs/outputs_v12b_control/tables/eval_legacy_random_sev0.4_key_normal_detail_normal.csv | 18234 | 4201c6f624028fc71aa262a2b5ffca9963de054b6ccb20871b8e91dee406bbc6 |
| v12b_outputs/outputs/outputs_v12b_control/tables/eval_legacy_random_sev0.4_key_sweep_detail_normal.csv | 34614 | 0544c34f7f9204206308f0e4790921b1b97cb7b83b70a2a89199784f00c8a166 |
| v12b_outputs/outputs/outputs_v12b/figures/demo_aud_only.png | 241833 | f0b7d81b4cb2dd0e17ef931ec4f676eb7094e0ea7f5ff478bbb8dfe28c7f0caf |
| v12b_outputs/outputs/outputs_v12b/figures/demo_aud_only_random.png | 192931 | cfc1b6961599c7035ad3de5402343d65a0015e6be6a859235538133b69bb51fa |
| v12b_outputs/outputs/outputs_v12b/figures/demo_both.png | 302346 | 7ba518c7f9834de25b2abe700cd291073dee4fa406eb591a082ac8073afd9977 |
| v12b_outputs/outputs/outputs_v12b/figures/demo_both_random.png | 217956 | ed3115925bffcaf39859aba8cdd230a37e3c0b2ca416a7cece2c95631b700568 |
| v12b_outputs/outputs/outputs_v12b/figures/demo_img_only.png | 208956 | c08e6b6dce8dad4b164f7d989abaa912f00c424bfd9720d6f25fafccb80fa8c1 |
| v12b_outputs/outputs/outputs_v12b/figures/demo_img_only_random.png | 171485 | 0a9fd1203ead0f5ff11a6d01673ec09190d7e2eba24a3bb6c008203039538a53 |
| v12b_outputs/outputs/outputs_v12b/logs/demo_fixed_mask.log | 3061 | 0d2562728e142117a7a41680b6d076eb7fab4957e48f22f2253ef9f9c2f07cb7 |
| v12b_outputs/outputs/outputs_v12b/logs/demo_legacy_random.log | 2730 | 97614a933825730ebd66435edd630d83ffe5b8323b8127f59ffd1b0d1de2cf3c |
| v12b_outputs/outputs/outputs_v12b/logs/eval_fixed_mask_normal.log | 365208 | a73c6a0873ee103e52e8c9115bf37f2ecdd4f7fb78e2b69807c293fddc688311 |
| v12b_outputs/outputs/outputs_v12b/logs/eval_fixed_mask_sweep.log | 291912 | a50064a1d6e464d6a378bd025fbd07a743ee8ce2506d53dbb1e23c34c066bd64 |
| v12b_outputs/outputs/outputs_v12b/logs/eval_legacy_random_normal.log | 55284 | b77254c8ba06a9271ca87e3ec555a03392a4ebac8ca8cae893cecbe7cec916a8 |
| v12b_outputs/outputs/outputs_v12b/logs/eval_legacy_random_sweep.log | 61173 | b4a0494b36bdb0c81e1784f88ee92e85a02123aae2e93b39063ab3da862be9bc |
| v12b_outputs/outputs/outputs_v12b/logs/train.log | 198878 | 5cd25496b0f9e903085723c9a50c05d3758db8307f8f63cdf47655324f94309f |
| v12b_outputs/outputs/outputs_v12b/tables/audio_family_breakdown_fixed.csv | 4174 | 855889e4d74331841a1cbea08ca7f94d14e885d5ec8b9016ad836e3d424e78a7 |
| v12b_outputs/outputs/outputs_v12b/tables/demo_eval_table.txt | 1262 | 388a87d92cf63c3fc55e9f30d8fca3ba7a9cff0d2369bd044eb883f0aa4ef70d |
| v12b_outputs/outputs/outputs_v12b/tables/demo_eval_table_random.txt | 1262 | aa4074d5cd2a9b707e50ebbd3b9bc4a62353cdc56ad5ae7ea135ad5084464fe9 |
| v12b_outputs/outputs/outputs_v12b/tables/eval_fixed_mask_sev0.4_key_normal_detail_normal.csv | 92059 | 22ab4b371ad6054c6d88ec17726d8fe2ab44a0af5707d70afa0574c5d5b3055c |
| v12b_outputs/outputs/outputs_v12b/tables/eval_fixed_mask_sev0.4_key_sweep_detail_normal.csv | 174954 | 70acd651b2148d3f054eccef47b127d15e43511883df81df42cd9e5958921bc6 |
| v12b_outputs/outputs/outputs_v12b/tables/eval_legacy_random_sev0.4_key_normal_detail_normal.csv | 18233 | 0445f49f9f47a4da42d8779405017b48fa9e0feaa51c95cd8d5e236487734824 |
| v12b_outputs/outputs/outputs_v12b/tables/eval_legacy_random_sev0.4_key_sweep_detail_normal.csv | 34601 | 785c6bb5c39aee1283eb359217994dc72e3a49d33f92e8cb4922455277368ed7 |
| v12b_outputs/outputs/outputs_v12b_no_causal/figures/demo_aud_only.png | 241629 | 5b69632ed0ed8280962d9acb17763b291d198f1e93c5491ea940e9e06ded9840 |
| v12b_outputs/outputs/outputs_v12b_no_causal/figures/demo_aud_only_random.png | 192709 | d3c56e768dbb1fb548d498efe70aa59201641084496067b7b55221faea02c40f |
| v12b_outputs/outputs/outputs_v12b_no_causal/figures/demo_both.png | 301020 | e0ba3a96c660f87a608538f37db27160c0e6093daa334c79097aa628839f2f2d |
| v12b_outputs/outputs/outputs_v12b_no_causal/figures/demo_both_random.png | 216321 | 76063ad21189ffae4f3beed552dc97622994f6d990f2a6d5e4f8566f2f811d84 |
| v12b_outputs/outputs/outputs_v12b_no_causal/figures/demo_img_only.png | 208956 | c08e6b6dce8dad4b164f7d989abaa912f00c424bfd9720d6f25fafccb80fa8c1 |
| v12b_outputs/outputs/outputs_v12b_no_causal/figures/demo_img_only_random.png | 171485 | 0a9fd1203ead0f5ff11a6d01673ec09190d7e2eba24a3bb6c008203039538a53 |
| v12b_outputs/outputs/outputs_v12b_no_causal/logs/demo_fixed_mask.log | 3111 | 7376b335866195c52c068d316ddb49e6a026f0bc1e714853d6dd225f34f0063d |
| v12b_outputs/outputs/outputs_v12b_no_causal/logs/demo_legacy_random.log | 2780 | 6e9decb867bcc7c8efa3a4ca36e181c07be702771ef22896ba5e1b42397fa52a |
| v12b_outputs/outputs/outputs_v12b_no_causal/logs/eval_fixed_mask_normal.log | 367185 | f2543088020977d963c8904e44632cdee1d3e0858dd5878d0fc440aa75bc1332 |
| v12b_outputs/outputs/outputs_v12b_no_causal/logs/eval_fixed_mask_sweep.log | 291932 | 7e633c281fb2edd00b012ecd8faca5f23f340338100c1de5969dabd6c1ec93f2 |
| v12b_outputs/outputs/outputs_v12b_no_causal/logs/eval_legacy_random_normal.log | 56023 | e73268eb939eb9d162369368422a920737d58fe5a5278b8d78aeb407a1e00eab |
| v12b_outputs/outputs/outputs_v12b_no_causal/logs/eval_legacy_random_sweep.log | 61193 | ea62be777a287f1178505cb734af942cd6443dead645c01acd63b1f718b8f1d4 |
| v12b_outputs/outputs/outputs_v12b_no_causal/logs/train.log | 110225 | 555fb266ecbb2ea5227a0a5f6dfc8242bee6a1656e861a9a15f41acce53379e7 |
| v12b_outputs/outputs/outputs_v12b_no_causal/tables/audio_family_breakdown_fixed.csv | 4174 | a9891a08047d9131d6d04d98cd8fd0cc04aea0575bb0ada7fa58a1b07b32895a |
| v12b_outputs/outputs/outputs_v12b_no_causal/tables/demo_eval_table.txt | 1262 | a92591f5e32b166a4aff15bf01f44a2878d837e8dc02efcdad75b8282da1a8da |
| v12b_outputs/outputs/outputs_v12b_no_causal/tables/demo_eval_table_random.txt | 1262 | c6b78c12882e9e06180b98e92f235c40e427742c35350c492be8d4ad27e95676 |
| v12b_outputs/outputs/outputs_v12b_no_causal/tables/eval_fixed_mask_sev0.4_key_normal_detail_normal.csv | 92047 | d49abe2974ed5fac201d1fc330a3a8b61faefc1c6a9fdf0cec372ce7d26b3035 |
| v12b_outputs/outputs/outputs_v12b_no_causal/tables/eval_fixed_mask_sev0.4_key_sweep_detail_normal.csv | 174961 | 4d0ecb8a702ce4d2f0661c5092e49885c5da8074858978c08b2df8cb1bee17f6 |
| v12b_outputs/outputs/outputs_v12b_no_causal/tables/eval_legacy_random_sev0.4_key_normal_detail_normal.csv | 18231 | 470bbd6d6a6c12f370f19951ce9f58dc574ef8bebd9ef3c4db8fdb6816477271 |
| v12b_outputs/outputs/outputs_v12b_no_causal/tables/eval_legacy_random_sev0.4_key_sweep_detail_normal.csv | 34608 | 2f7e605c9d799fdcea502190f09569a0ee7564d6ad90daa0042b2df90b5f64e3 |

</details>

## 2026-09-15

<a id="implementation-v13pro"></a>

### v13pro：论文证据补充套件

授权：先完成 v12b 本地评估，再补充面向 CCF C 的实验工程；用户随后指定分支
`v13pro` 并要求上传。此处记录实现和验证，不重复填写上方 v12b 的全部评估表。

实现范围：独立 train/val/test、train-only norm/medoid、同父同额外预算 control/main/
no-causal/no-cross、独立内容识别器、简单 CNN 和类别原型基线、逐样本评估、配对
聚类统计、Index 撤去 cue/扰动轨迹、可重启顺序 suite。未修改 v12b 的模型主体：
仍保留类别绑定、Value+own cue、batch 128、同时电流、恢复 Value 到 Index 的 detach。

| 检查 | 实际结果 | 解释边界 |
|---|---|---|
| CPU 协议回归 | 15 项通过 | 不是 CUDA 收敛或性能实验 |
| 干净提交目录验证 | 仅导出待提交文件后，15 项测试再次通过；组合开关 dry-run 和 compileall 通过 | 不依赖本地旧 YAML/专用脚本 |
| 原 v12b 回归 | 三配置共 24 个 cue 前向通过；control 与旧 tensor 路径一致 | 在迁移前验证，旧专用脚本不随新分支上传 |
| 父/子训练链路 | 小模型 synthetic fixture 跑完 parent、control、main、no-cross 各 1 轮并加载 best | 不作真实识别率/恢复率结果 |
| 冻结范围 | 测试中 Encoder、memory、Classifier 权重与父 checkpoint 逐项相同 | 仅恢复分支冻结；父模型从头训练 |
| 断点恢复 | 已完成的小模型 main 再次 --resume 不重训、不改变 best SHA；恢复时核对配置 | CUDA 端仍待实际运行验证 |
| 全量口径防护 | smoke 单独输出且 full_test=false，正式路径核对 n | 10 个 demo 不能代替全量 |
| 撤去输入 | 连投影 bias 一起关闭；默认 Index 前向与 trace 前向一致 | 活动静默不等于吸引子成功 |
| 缺失区域与回填 | 八个 cue 的形状、反传、可见区复制与 N/A 检查通过 | clean mask=None 仍按现有模型重建策略 |
| 本地真实数据划分 | MNIST train=54001、val=5999、test=10000，train/val 交集=0 | 按类别四舍五入留出，非精确 6000 |
| 本地 FSDD 清单 | train=2400、val=300、test=300；6 speakers | 配对曝光不是新增录音 |
| 环境 | PyTorch 1.13.1+cpu，CUDA unavailable | 未执行服务器 GPU 全训练 |

默认计划：3 training seeds（1234/2345/3456），每 seed parent 从头 100 轮，四个
恢复分支各 30 轮，四个 CNN/分类任务各 30 轮。共 27 个训练任务、1020 model-epochs；
有 21 个模型评估任务，其中 classifier 产生 hard/soft/oracle 三种恢复基线。默认 fixed
五组 family、random、severity=0.4、mask_seed=5678，均跑全测试集。实际耗时和 GPU
峰值显存尚未测得，不能宣称已验证 3080 完整父模型训练的显存适配。

主要终点预设为三个音频部分缺失 cue 的 missing MSE，partial_temporal 单列。
恢复模型按全验证集固定 family 轮转的等权分组指标选 best，而不是看 test。内部
Index 分类和外部 CNN 内容分类分开。global_ssim 仍是原简化全局定义，明确不能当
标准滑窗 SSIM。Cross-Key 报告 correct/zero/wrong/same-class 和训练 no-cross 对照。

统计区间为每个固定训练模型条件下的等录音 cluster 配对差值区间；训练 seed 均值/
标准差另列。录音 cluster 不消除 speaker 内相关性，speaker bootstrap 是敏感性分析，
只有六个 speaker 时仍不能夸大稳健性。没有多重比较校正的显著性主张。

状态：本次仅工程实现和 CPU 验证完成，所有正式 v13pro 实验指标均为**待训练/待评估**，
没有以 0 或模拟结果填充。第二数据集、参数/预算匹配 ANN、已有文献关联记忆方法、
FLOPs/SOP/硬件能效及论文新颖性仍待补足。默认 CNN 的 30 轮与 SNN 的 100+30 轮
不构成总预算匹配；可提高 `--baseline_epochs`，不能因为新增简单 CNN 就称强基线齐全。

提交范围：仅当前代码、测试、`configs/v13pro.yaml` 与四份核心文档和 README。
旧 v12a/v12b YAML 及专用 suite/smoke 从分支跟踪清单移除，但保留本地文件；用户
已有 `configs/v12a_control.yaml` 修改不覆盖。数据、权重、结果包、PDF、临时文件不上传。

#### 运行说明

从服务器项目根目录执行，不需要 v11g/v12a/v12b 权重：

```bash
python scripts/run_v13pro_suite.py --dry_run
nohup python -u scripts/run_v13pro_suite.py --run > v13pro_suite.log 2>&1 < /dev/null &
tail -f v13pro_suite.log
```

训练先于全量评估/统计。SSH 断开先检查进程是否仍在，确实退出后重启同一命令；
完成标记校验计划、代码、checkpoint 和评估产物，未完成训练恢复 last。更改计划
须另设 `--output`；不同时运行两份。数据/输出路径均来自当前项目，不使用旧服务器
绝对路径。详细可选预算、speaker/OOD/机制开关见 `implementation.md` 第 6 节。
