# 开发日志：Cross-Modal Attractor SNN
> 创建时间：2026-07-06 18:43 | 最后更新：2026-07-06 18:58
> 关联实现指南：`docs/implementation.md`
> 当前阶段：ResearchPilot F 阶段补档与迭代
> 本文件原则上只追加，不删除。每次代码修改都必须追加新的日志条目。

## 项目概览

| 项目 | 内容 |
|------|------|
| 研究方向 | 跨模态 attractor SNN 联想记忆 |
| 当前阶段 | F：代码迭代 |
| 当前配置 | `configs/v10a.yaml` |
| 代码结构 | 根目录下的 `data/`、`models/`、`scripts/`、`configs/` |
| 主要任务 | MNIST 图像 + FSDD 音频 cue -> digit 分类 + 图像/音频恢复 |
| 框架 | PyTorch |
| 主 checkpoint 目标 | `outputs/checkpoints/cross_modal_snn_v10a.pt` |
| 版本化输出目标 | `outputs/outputs_v10a/` |
| 硬性工作流 | 先改文档，再改代码；每次改代码后追加本日志 |

## F 阶段规则

1. 每次代码修改前，先判断改动范围：只改配置、模型结构、数据管线、训练逻辑、评估逻辑、demo，还是实验设计。
2. 若改动会影响行为、tensor shape、函数接口、命令或输出格式，必须先更新 `docs/implementation.md`。
3. 若改动涉及 Method 或实验设计，必须先更新 `docs/idea_report.md`。如果该文件尚不存在，则先创建或补充相关设计说明。
4. 每次代码修改后，都必须在本文件追加日志条目。
5. 本文件末尾固定保留 `运行说明` 章节；命令、参数、输出文件或输出路径变化时必须同步更新。

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

| 模块 | 文件 | 状态 | 时间 | 备注 |
|------|------|------|------|------|
| 用户需求记录 | `docs/user_requirements.md` | 文档完成 | 2026-07-06 18:58 | 已中文化，记录 D-F 硬规则 |
| 实现指南 | `docs/implementation.md` | 文档完成 | 2026-07-06 18:58 | 已中文化，基于当前代码整理 |
| 开发日志 | `docs/dev_log.md` | 文档完成 | 2026-07-06 18:58 | 当前文件 |
| 配置 | `configs/v10a.yaml` | 已有 | 补档前 | v10a 当前活动配置 |
| 数据管线 | `data/*.py` | 已有 | 补档前 | MNIST + FSDD log-mel + medoids |
| 残缺 cue 管线 | `data/corruption.py`, `common.py` | 已有 | 补档前 | 6 种 cue mode 和 corruption family |
| SNN 基础模块 | `models/lif.py` | 已有 | 补档前 | LIF + surrogate gradient |
| 编码器 | `models/encoders.py` | 已有 | 补档前 | image/audio SNN encoders |
| 记忆模块 | `models/memory.py` | 已有 | 补档前 | Key/Index/Value + binding/readout |
| 解码器 | `models/decoders.py` | 已有 | 补档前 | classifier、image decoder、audio decoder |
| 顶层模型 | `models/network.py` | 已有 | 补档前 | `CrossModalSNN` |
| 训练脚本 | `scripts/train.py` | 已有 | 补档前 | decoder pretrain + full training |
| 评估脚本 | `scripts/evaluate.py` | 已有 | 补档前 | fixed/random protocols |
| Demo 脚本 | `scripts/demo_inference.py` | 已有 | 补档前 | demo figures 和 table |
| 消融 suite | `scripts/make_v10a_ablations.py`, `scripts/run_v10a_suite.py` | 已有 | 补档前 | 三个 v10a 消融变体 |
| 当前 v10a 输出 | `outputs/outputs_v10a/` | 待运行 | 2026-07-06 18:43 | 创建日志时目录不存在 |

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

## 已知问题

- [x] `docs/idea_report.md` 曾缺失；已在 2026-07-07 v10c 迭代中创建 F 阶段实验设计补充记录。
- [ ] `outputs/outputs_v10a/` 尚不存在。重定向 v10a 日志前，应先运行 `python scripts/mkdir_outputs.py --config configs/v10a.yaml`。
- [ ] 创建日志时未在 `outputs/checkpoints/` 下检测到 v10a checkpoint。
- [ ] 当前 `requirements.txt` 包含 `torch`、`torchvision`、`torchaudio`；后续需决定保留当前项目实用约定，还是改为只在 README 中说明 PyTorch 安装。
- [ ] 已尝试用 Codex 自带 Python runtime 运行 smoke test，但该 runtime 缺少 `torch`。
- [ ] v10b 尚未在带 PyTorch 的项目环境中完成 smoke test、训练或评估。

---

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
python -u scripts/evaluate.py --config configs/v10c.yaml --protocol fixed_mask --family_breakdown
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
