# 开发日志：Cross-Modal Attractor SNN

> 创建时间：2026-07-06 18:43 | 最后更新：2026-07-09
> 关联实现指南：`docs/implementation.md`
> 当前阶段：ResearchPilot F 阶段补档与迭代
> 本文件原则上只追加，不删除。每次代码修改都必须追加新的日志条目。

## 项目概览


| 项目              | 内容                                            |
| --------------- | --------------------------------------------- |
| 研究方向            | 跨模态 attractor SNN 联想记忆                        |
| 当前阶段            | F：代码迭代                                        |
| 当前配置            | `configs/v10a.yaml`                           |
| 代码结构            | 根目录下的 `data/`、`models/`、`scripts/`、`configs/` |
| 主要任务            | MNIST 图像 + FSDD 音频 cue -> digit 分类 + 图像/音频恢复  |
| 框架              | PyTorch                                       |
| 主 checkpoint 目标 | `outputs/checkpoints/cross_modal_snn_v10a.pt` |
| 版本化输出目标         | `outputs/outputs_v10a/`                       |
| 硬性工作流           | 先改文档，再改代码；每次改代码后追加本日志                         |


## F 阶段规则

1. 每次代码修改前，先判断改动范围：只改配置、模型结构、数据管线、训练逻辑、评估逻辑、demo，还是实验设计。
2. 若改动会影响行为、tensor shape、函数接口、命令或输出格式，必须先更新 `docs/implementation.md`。
3. 若改动涉及 Method 或实验设计，必须先更新 `docs/idea_report.md`。如果该文件尚不存在，则先创建或补充相关设计说明。
4. 每次代码修改后，都必须在本文件追加日志条目。
5. 本文件末尾固定保留 `运行说明` 章节；命令、参数、输出文件或输出路径变化时必须同步更新。
6. `configs/` **版本卫生（每次切到新版本分支必做）**：`configs/` 只保留**当前活动版本**的主配置 `v10X.yaml` 及其消融 `v10X_ab_*.yaml`；删除其他版本遗留 yaml（如 v10f 分支不应保留 v10d/v10e）。同步把 `paths.py`、`common.py`、`scripts/`* 默认 `--config` 改为当前版本；在本文档追加清理记录。历史配置留在对应 Git 分支或本地 `outputs/`，不堆在活跃分支里。



## 项目架构

```text
clean/corrupt image cue -> ImageSNNEncoder -> Key_img \
                                                   -> recurrent Index A -> Value_img -> ImageDecoder
clean/corrupt audio cue -> AudioSNNEncoder -> Key_aud /                    -> Value_aud -> AudioDecoder

Index state -> ClassifierHead
cue detail states -> optional gated concat into decoders
```



## 实现进度

状态说明：`已有` 表示代码已存在但本次文档补档未重新完整验证；`文档完成` 表示本轮已生成或中文化；`待运行` 表示当前 v10a 产物尚未生成。


| 模块         | 文件                                                            | 状态   | 时间               | 备注                                     |
| ---------- | ------------------------------------------------------------- | ---- | ---------------- | -------------------------------------- |
| 用户需求记录     | `docs/user_requirements.md`                                   | 文档完成 | 2026-07-06 18:58 | 已中文化，记录 D-F 硬规则                        |
| 实现指南       | `docs/implementation.md`                                      | 文档完成 | 2026-07-06 18:58 | 已中文化，基于当前代码整理                          |
| 开发日志       | `docs/dev_log.md`                                             | 文档完成 | 2026-07-06 18:58 | 当前文件                                   |
| 配置         | `configs/v10a.yaml`                                           | 已有   | 补档前              | v10a 当前活动配置                            |
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
| 当前 v10a 输出 | `outputs/outputs_v10a/`                                       | 待运行  | 2026-07-06 18:43 | 创建日志时目录不存在                             |




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

## 已知问题

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

## 运行说明

本章固定放在文件末尾。凡代码修改影响命令、参数、输出文件或输出格式时，必须在同一轮迭代中更新本章。

### 环境准备

```bash
pip install -r requirements.txt
```

- 安装当前项目依赖。
- 当前 `requirements.txt` 包含 PyTorch 系依赖，因为当前 README 说明这些依赖是必需的。
- 若使用 CUDA，建议先按 PyTorch 官网命令安装匹配 CUDA 版本的 PyTorch，再安装其余依赖。



### 创建输出目录

```bash
python scripts/mkdir_outputs.py --config configs/v10a.yaml
```

- 读取 config 中的 `train.output_version`。
- 创建：
  - `outputs/checkpoints/`
  - `outputs/outputs_v10a/figures/`
  - `outputs/outputs_v10a/logs/`
  - `outputs/outputs_v10a/tables/`



### 主训练

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



### 后台训练并写日志

```bash
nohup env PYTHONUNBUFFERED=1 python -u scripts/train.py --config configs/v10a.yaml > outputs/outputs_v10a/logs/train_v10a_50ep.log 2>&1 < /dev/null &
```

- 运行前需要先创建输出目录。
- stdout/stderr 写入 `outputs/outputs_v10a/logs/train_v10a_50ep.log`。



### 主训练加消融 suite

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



### 评估

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



### Demo 图

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



### 渲染评估表

```bash
python scripts/plot_eval_summary.py outputs/outputs_v10a/tables/demo_eval_table.txt
```

- 解析 demo table 文本。
- 在版本化输出目录附近生成 PNG 和 CSV 汇总表。

解析完整评估日志：

```bash
python scripts/plot_eval_summary.py outputs/outputs_v10a/logs/eval_v10a_fixed_mask.log
```



### Smoke Test

```bash
python -u scripts/smoke_test.py
```

- 只使用随机张量。
- 检查 6 种 cue mode 的 binding/readout 前向和反向。
- 检查 corruption 函数。
- 检查 audio-only 和 image-only 推理。
- 不需要下载 MNIST 或 FSDD。



### v10b 运行命令

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



### v10c 运行命令

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



### v10d 运行命令

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

### v10e 运行命令

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



### v10f 运行命令

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



### v11a 运行命令

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


### v11c 运行命令

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

### v11d / v11e 远端分支评价命令索引

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

## 2026-09-09 v11f：冻结父模型的缺失区域 Cross-Key

### 需求与回溯

用户同意四项修改并最终指定 `v11f` 分支。不是 `v11`，也不覆盖 v11e。
按 F 阶段先更新需求、implementation 与实验设计，再实现和测试。
诊断是 v11e 的部分残缺收益有限且联合训练存在分类代价；本轮通过冻结基线和
局部调制检验改进假设，不预先认定之前差异的因果来源已经完全确定。

### 逐文件修改

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

### 验证结果与限制

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

## 运行说明

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
