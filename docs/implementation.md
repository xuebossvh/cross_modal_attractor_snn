# 实现指南：Cross-Modal Attractor SNN
> 生成时间：2026-07-06 18:43 | 生成策略：基于当前代码反向整理 | 状态：ACTIVE_F_STAGE
> 关联配置：`configs/v10a.yaml`、`configs/v10b.yaml`、`configs/v10c.yaml`
> 省略说明：当前仓库尚无 `docs/idea_report.md`，因此本文档暂不绑定 Part 2/Part 3。
> 扩展说明：本项目已按 ResearchPilot F 阶段管理，后续所有代码修改都走 D-F 迭代。

---

## 0 F 阶段迭代约定

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
| 改模型结构 | 更新 `docs/implementation.md`；若 `docs/idea_report.md` 已存在，也更新 Method | 追加 `docs/dev_log.md`，写清预期效果与验证结果 |
| 改实验设计 | 更新 `docs/implementation.md`；若 `docs/idea_report.md` 已存在，也更新 Part 3 | 追加 `docs/dev_log.md`，同步结果格式和输出路径 |
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

---

## 1 项目结构

### 1.1 当前真实目录树

```text
cross_modal_attractor_snn/
├── configs/
│   ├── v10a.yaml
│   ├── v10b.yaml
│   └── v10c.yaml
├── data/
│   ├── audio_features.py
│   ├── corruption.py
│   ├── dataset.py
│   └── fsdd.py
├── docs/
│   ├── GPT_HANDOFF.md
│   ├── implementation.md
│   ├── dev_log.md
│   ├── user_requirements.md
│   └── figures/
├── models/
│   ├── decoders.py
│   ├── encoders.py
│   ├── lif.py
│   ├── memory.py
│   ├── network.py
│   └── __init__.py
├── scripts/
│   ├── bootstrap.py
│   ├── demo_inference.py
│   ├── evaluate.py
│   ├── make_v10a_ablations.py
│   ├── mkdir_outputs.py
│   ├── plot_eval_summary.py
│   ├── run_v10a_suite.py
│   ├── smoke_test.py
│   └── train.py
├── _data/
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
| `scripts/evaluate.py` | 6 种 cue 模式评估，支持 fixed/random 协议 | checkpoint、test loader | 终端表格、family CSV | CLI |
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

`common.CUE_MODES` 定义 6 种训练和评估模式：

| cue mode | 图像 cue | 音频 cue |
|----------|----------|----------|
| `corrupt_img_only` | 残缺图像 | 缺失 |
| `corrupt_aud_only` | 缺失 | 残缺音频 |
| `corrupt_both` | 残缺图像 | 残缺音频 |
| `clean_img_only` | 干净图像 | 缺失 |
| `clean_aud_only` | 缺失 | 干净音频 |
| `clean_both` | 干净图像 | 干净音频 |

`common.sample_cue_mode(cfg)` 按 `configs/v10a.yaml::cue_modes` 中的概率采样。若 `ablation.use_modality_dropout=false`，则只采样双模态 cue。

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

**`AudioDecoder(nn.Module)`**

- 构造函数：`AudioDecoder(n_value_aud, n_mels, n_frames, base_ch=128, start_hw=4, refine_blocks=0)`。
- `forward(value_state) -> Tensor`
- 输入：`[B,n_value_aud + optional_detail_dim]`。
- 输出：`[B,n_mels,n_frames]`，范围 `[0,1]`。
- 实现逻辑：FC 到方形 feature map，多层 ConvTranspose2d 上采样，可选 refinement Conv2d，最后 softplus 并 clamp。

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

**`forward(x_img_cue=None, x_aud_cue=None, x_img_target=None, x_aud_target=None, training_mode=False, phase="readout") -> dict`**

- 至少需要一种 cue 模态。
- 推理时强制 `phase="readout"`，并清空 target，防止答案泄漏。
- 流程：
  1. 对存在的 image/audio cue 编码。
  2. binding 阶段对 clean targets 编码。
  3. 调用 memory。
  4. 从 `index_state` 分类。
  5. 若启用 audio aux classifier，从 `key_aud` rate 辅助分类。
  6. 将 A-driven Value 和 cue detail 融合。
  7. 解码 recovered image 和 recovered audio。
- 返回字典包含：`index_spikes`、`index_state`、cue spikes、key spikes、`v_*_from_A`、`v_*_target`、detail states、`logits`、`aux_aud_logits`、`recovered_img`、`recovered_aud`。

**`infer(x_img_cue=None, x_aud_cue=None) -> dict`**

- eval 模式下执行 readout 推理，不允许 target 输入。

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

**`build_loaders(cfg) -> tuple[DataLoader,DataLoader]`**

- real audio 启用时先保证 audio norm stats。
- 创建 train/test dataset。
- 从 train set 构建 medoids，并共享给 test set。
- train loader shuffle 且 drop_last，test loader 不 shuffle。

### 4.5 `common.py`

| 函数 | 职责 |
|------|------|
| `fix_console_encoding()` | Windows 终端 UTF-8 输出 |
| `log(msg)` | flush print |
| `load_config(path="configs/v10a.yaml")` | 读取 YAML |
| `set_seed(seed)` | 设置 random、numpy、torch seed |
| `sample_cue_mode(cfg)` | 按概率采样 cue mode |
| `sample_train_severity(cfg, epoch)` | fixed/random/staged severity |
| `resolve_train_corrupt_modes(cfg, epoch, step=None)` | 训练时采样 image/audio corruption family |
| `build_cue(clean_img, clean_aud, mode, cfg, severity=None, img_mode=None, aud_mode=None, return_masks=False)` | 构造 clean/corrupt/single-modal cue；`return_masks=True` 时返回 `(img_cue,aud_cue,{"img":None,"aud":aud_mask})` |
| `is_aud_only_mode(mode)` | 判断 audio-only cue |
| `cue_modalities(mode)` | 返回 cue 是否含 image/audio |
| `select_targets(cue_mode, clean_img, clean_aud, proto_img, proto_aud, labels)` | 按恢复粒度选择 target |
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
12. sample-level audio target 时加入 audio active loss。
13. 加 spike activity regularization。

### 5.3 checkpoint 格式

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

主要函数：

| 函数 | 职责 |
|------|------|
| `_reseed(seed)` | 固定 corruption mask |
| `_fixed_eval_families(cfg)` | 论文主对照的固定 image/audio corruption family |
| `_audio_masked_metrics(rec, target, mask)` | 计算 audio masked/visible MSE 和 L1，mask 不适用时返回 NaN |
| `_log_audio_diag(diag_rows)` | 打印音频塌缩诊断 |
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
| `pix_var` | 图像重建多样性诊断 | 诊断项 |
| `pair_l2` | 随机样本对重建距离 | 诊断项 |
| `rec_mean/std/max`、`topk_recall` | 音频塌缩诊断 | 诊断项 |

### 6.2 `scripts/demo_inference.py`

常用命令：

```bash
python -u scripts/demo_inference.py --config configs/v10a.yaml --num 10 --severity 0.5
python -u scripts/demo_inference.py --config configs/v10a.yaml --num 10 --severity 0.5 --protocol legacy_random
```

默认输出：

| 输出 | 含义 |
|------|------|
| `outputs/outputs_v10a/figures/demo_aud_only.png` | audio-only cue -> category image + sample audio |
| `outputs/outputs_v10a/figures/demo_img_only.png` | image-only cue -> sample image + category audio |
| `outputs/outputs_v10a/figures/demo_both.png` | bimodal cue -> sample image + sample audio |
| `outputs/outputs_v10a/tables/demo_eval_table.txt` | demo 汇总表与逐样本分类表 |

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
| `cue_modes` | 6 种 cue mode 采样概率 |
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

---

## 8 数据下载与准备

### 8.1 数据集

| 数据集 | 类型 | 来源 | 路径 |
|--------|------|------|------|
| MNIST | 图像 digit 数据集 | `torchvision.datasets.MNIST` | `_data/MNIST` |
| FSDD | spoken digit wav | Jakobovski free-spoken-digit-dataset | `_data/fsdd/recordings` |

### 8.2 准备逻辑

1. `data.dataset.build_loaders(cfg)` 在 real audio 启用时调用 `ensure_audio_norm_stats(cfg)`。
2. `ensure_audio_norm_stats` 需要 FSDD train wav。
3. 若 FSDD wav 不存在且 `audio.auto_download=true`，`data.fsdd.ensure_fsdd` 先尝试 git clone，再尝试 zip 下载。
4. MNIST 由 torchvision 按需下载。
5. 音频归一化统计保存到 `_data/audio_norm_stats.pt`。

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

### 9.3 日志

| 日志 | 建议路径 | 内容 |
|------|----------|------|
| 主训练日志 | `outputs/outputs_v10a/logs/train_v10a_50ep.log` | epoch/step loss 与 checkpoint 保存记录 |
| fixed eval 日志 | `outputs/outputs_v10a/logs/eval_v10a_fixed_mask.log` | 6 模式指标与音频诊断 |
| random eval 日志 | `outputs/outputs_v10a/logs/eval_v10a_legacy_random.log` | 随机 family 鲁棒性检查 |
| suite 日志 | `outputs/outputs_v10a/logs/suite_v10a_with_ablations.log` | 长实验编排输出 |
| v10b fixed eval 日志 | `outputs/outputs_v10b/logs/eval_v10b_fixed_mask.log` | 含 masked audio 指标的 time-mask 对齐评估 |
| v10c fixed eval 日志 | `outputs/outputs_v10c/logs/eval_v10c_fixed_mask.log` | fixed occlusion + time-mask 下的 v10c 主评估 |

### 9.4 表格与图像

| 产物 | 路径 | 格式 |
|------|------|------|
| demo figures | `outputs/outputs_v10a/figures/demo_*.png` | PNG |
| demo table | `outputs/outputs_v10a/tables/demo_eval_table*.txt` | 对齐文本 |
| audio family breakdown | `outputs/outputs_v10a/tables/audio_family_breakdown_fixed.csv` | CSV |
| v10b audio family breakdown | `outputs/outputs_v10b/tables/audio_family_breakdown_fixed.csv` | CSV，包含 `family_group` 和 masked audio 指标 |
| v10c audio family breakdown | `outputs/outputs_v10c/tables/audio_family_breakdown_fixed.csv` | CSV，包含 `family_group` 和 masked audio 指标 |
| 渲染后的评估表 | `outputs/outputs_v10a/tables/*.png` 和 `.csv` | PNG + CSV |

---

## 10 后续 F 阶段修改顺序

未来任何代码修改都按以下顺序执行：

1. 读取 `docs/user_requirements.md`。
2. 读取本文档。
3. 读取 `docs/dev_log.md`。
4. 判断改动类型：配置、模型结构、数据管线、训练、评估、demo、实验设计或文档。
5. 若行为、API、tensor shape、输出格式或命令变化，先更新本文档。
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
| 6 种 cue mode | 通过 | 已覆盖 `common.py`、train/evaluate/demo |
| 跨模态恢复 target 语义 | 通过 | 已在 2.4 节说明 |
| binding/readout 两阶段 | 通过 | 已在模型和训练章节说明 |
| detail-conditioned decoders | 通过 | 已在架构和 `network.py` 章节说明 |
| v10a 消融 | 通过 | 已在 6.3 节说明 |
| 评估协议 | 通过 | fixed_mask 和 legacy_random 均已说明 |
| 音频塌缩诊断 | 通过 | 已在指标和结果文件中说明 |

### 11.2 逻辑一致性

| 检查项 | 状态 | 说明 |
|--------|------|------|
| tensor shape 链路 | 通过 | image/audio encoder -> Key -> Index -> Value -> decoder 与 `v10a.yaml` 一致 |
| 防止 target 泄漏 | 通过 | inference 强制清空 target，decoder 主输入为 A-driven Value |
| 输出路径 | 通过，有警告 | v10a 路径已记录，但生成本文档时 `outputs/outputs_v10a/` 尚不存在 |
| requirements 规则 | 警告 | 当前 `requirements.txt` 包含 torch/torchvision/torchaudio，与 ResearchPilot 默认“不写 torch 系”规则不同；本文档保留当前项目现实状态 |

### 11.3 完整性

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 源码文件覆盖 | 通过 | 当前 root/data/models/scripts/config 文件均已覆盖 |
| 函数级说明 | 通过，部分压缩 | 核心类和函数含签名、职责与逻辑；绘图辅助函数按角色汇总 |
| D-F 迭代约束 | 通过 | 第 0 节和第 10 节已规定 |
| 缺失 A-C 文档 | 警告 | `docs/idea_report.md` 尚不存在；后续涉及 Method 或实验设计变化时应创建或补充 |
