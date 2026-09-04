# 跨模态循环吸引子 SNN 记忆网络

可运行的 PyTorch 原型，实现**跨模态脉冲联想记忆**网络。图像与音频模态通过
共享的**循环吸引子 Index 层**绑定。统一任务接口：

> 给定**残缺 / 干净**的 cue（图像、音频或双模态），统一输出
> **分类 label + 完整图像 + 完整音频特征**。

即任意一种（可能残缺的）模态线索都能从吸引子记忆中**补全 / 联想**出
另一模态与类别。LIF 神经元与 surrogate gradient 均为手写实现，无需外部 SNN 库。

> **v11e 真实配对策略（重要）**：使用 GRID 同一次 utterance 的视频帧和音频，
> 图像输入是该帧序列的 28×28 rank-pooled mouth-motion dynamic image。manifest
> 强制 `source_id == image_source_id == audio_source_id`，并按 speaker 划分
> train/val/test。v11e 不允许退回 MNIST/FSDD 同类别伪配对。

### 目标策略（target selection by cue mode）

| cue 模式 | recovered image | recovered audio | classification |
|----------|-----------------|-----------------|----------------|
| **audio-only** | 同一真实 `source_id` 的 clean dynamic image | 同一真实 `source_id` 的 clean log-mel | digit label |
| **image-only** | 同一真实 `source_id` 的 clean dynamic image | 同一真实 `source_id` 的 clean log-mel | digit label |
| **image+audio** | 同一真实 `source_id` 的 clean dynamic image | 同一真实 `source_id` 的 clean log-mel | digit label |

- v11e 的一个 manifest row 就是一个真实 utterance pair；增广/重复曝光不得产生
  新 `pair_id`。评估表中的目标统一标记为 `paired-sample`，不再出现 `category`。
- Cross-Key 继续传递类别语义；新增 Cross-Detail 使用 Key 前一层的高维脉冲率，
  通过独立 Project + Vector Gate 向对侧 Decoder 注入实例细节。
- 同类不同 `pair_id` 同时作为 Cross-Detail 因果困难负样本与 pair-embedding
  InfoNCE 困难负样本，避免网络只学会数字类别。

旧版类别代表原型仍为 **class medoid**（真实样本，非均值图）：
`prototype[c] = argmin_i || x_i − mean(x_c) ||₂`，由 `data/dataset.py` 在
**训练集**上构建，test/demo 复用同一份；选择逻辑集中在 `common.py :: select_targets`。
binding 与 readout 两阶段共用同一套 target；不对完整 `index_state` 做强对齐
（仅分类 loss 约束类别语义），以保留样本细节。

---

## 1. 安装依赖

```bash
cd cross_modal_attractor_snn
pip install -r requirements.txt
```

已在 Python 3.10、`torch 1.13`（CPU 即可）下测试通过。`torchaudio` 为**必需**依赖（FSDD log-mel）；缺失或 FSDD 未下载会直接报错。

## 2. 项目结构

```
cross_modal_attractor_snn/
├── configs/           # 超参 yaml
├── data/              # 数据集、损坏、FSDD、log-mel
├── models/            # SNN 编码器 / 记忆层 / decoder
├── scripts/           # 可执行入口（train / evaluate / demo / smoke_test）
├── common.py          # cue 采样、target 选择、指标
├── paths.py           # 项目根目录与 outputs 路径
├── outputs/           # 运行产物（不入 git，见 .gitignore）
│   ├── checkpoints/   # 各版本 *.pt 权重（共用）
│   └── outputs_v11e/  # 当前主实验产物（主实验/control 各有独立目录）
│       ├── logs/
│       ├── figures/
│       └── tables/
├── docs/              # 文档（如 GPT_HANDOFF.md）
├── _data/             # GRID 预处理数据；旧版 MNIST/FSDD 数据
├── requirements.txt
└── README.md
```

## 3. 训练

当前实验配置族为 **v11e**。由于输入域从 MNIST/FSDD 改成 GRID 真实视听事件，
主实验和 control 均从头端到端训练 100 epoch，不继承 v11c/v11d 权重。默认 GRID
约 33 个可用 speaker 时按 speaker 分为约 27k train / 3k val / 3k test；每 epoch
从 unique train pairs 有放回采样 60,000 次，用相同 optimizer exposure 对标 MNIST，
但报告的 unique pair 数不变。

```bash
pip install -r requirements.txt
python -u scripts/prepare_grid_v11e.py --audio_root /path/to/GRID/audio --frames_root /path/to/GRID/jpg --output_root _data/grid_v11e
python -u scripts/smoke_test_v11e.py
python scripts/mkdir_outputs.py --config configs/v11e_control.yaml
python -u scripts/train.py --config configs/v11e_control.yaml
python scripts/mkdir_outputs.py --config configs/v11e.yaml
python -u scripts/train.py --config configs/v11e.yaml
```

每个 batch 采样一种 cue 模式，并分两阶段计算损失：

- **binding 阶段**（可关）：cue 驱动 Index A 收敛；干净 target 经 Encoder 写入
  target Value（可延迟若干步）。`bind loss` 让 **A 驱动的 Value** 对齐
  **target Value（stop-grad）**，从而学习 `A→V` 绑定。此阶段不走 decoder。
- **readout 阶段**：关闭 target。v11e 先把对侧 cue 的 Key rate 投影为 Value 空间
  residual，与 `v_*_from_A` 相加；再与对应 cue 的同模态 detail state 拼接后送入
  Decoder，并把对侧 Key 前实例特征经 Project + Vector Gate 加到 detail channel。
  随后计算分类、图像/音频恢复、Cross-Key 因果损失、同类错配 Cross-Detail
  因果损失、同类内实例 InfoNCE 和脉冲正则。AudioDecoder 仍只输出一个
  `recovered_aud`，v11e 不构造旧 AudioRefiner。

每个配置使用独立 checkpoint/output_version。每轮在 speaker-disjoint validation
split 上计算恢复、分类和 pair retrieval 组合分数，并额外保存 `*_best.pt`；正式
评估默认读取 best checkpoint，final checkpoint 仍用于断点续训。

> 注意：分支初始化默认 strict。evaluate 遇到缺失或结构不匹配的
> checkpoint 会直接报错，不会用随机权重生成伪评估结果。

快速冒烟：运行 `python -u scripts/smoke_test_v11e.py`；正式训练前确认
`_data/grid_v11e/pairs.csv` 存在且严格校验通过。

## 4. 评估与 Demo

```bash
python -u scripts/evaluate.py --config configs/v11e_control.yaml --protocol fixed_mask --severity 0.4 --family_breakdown 2>&1 | tee outputs/outputs_v11e_control/logs/eval_v11e_control_fixed_mask_sev04.log
python -u scripts/evaluate.py --config configs/v11e.yaml --protocol fixed_mask --severity 0.4 --family_breakdown 2>&1 | tee outputs/outputs_v11e/logs/eval_v11e_fixed_mask_sev04.log
python -u scripts/evaluate.py --config configs/v11e.yaml --protocol fixed_mask --severity 0.4 --cross_key sweep 2>&1 | tee outputs/outputs_v11e/logs/eval_v11e_cross_key_sweep_sev04.log
python -u scripts/evaluate.py --config configs/v11e.yaml --protocol fixed_mask --severity 0.4 --cross_detail sweep 2>&1 | tee outputs/outputs_v11e/logs/eval_v11e_cross_detail_sweep_sev04.log
python -u scripts/demo_inference.py --config configs/v11e_control.yaml --num 10 --severity 0.4
python -u scripts/demo_inference.py --config configs/v11e.yaml --num 10 --severity 0.4
```

- `evaluate.py`：8 种 cue 模式下的 acc / 图像 MSE·PSNR·SSIM / **log-mel MSE** 等；
  指标按样本数加权，完全缺失的模态按 100% missing mask 评估。Cross-Key sweep
  比较 correct/zero/wrong-class/same-class Key；Cross-Detail sweep 比较
  correct/zero/same-class wrong-pair Detail。v11e 所有模式的 target 均为
  `paired-sample/paired-sample`，并报告同类候选中的 image→audio / audio→image
  exact-pair Recall@1。快速试跑：`python -u scripts/evaluate.py --config configs/v11e.yaml --max_batches 1`。
- `demo_inference.py` 输出三张图，标题明确区分恢复粒度，每格标注
  cue type / target type / true label / pred label / confidence：
  - `outputs/outputs_v11e/figures/demo_aud_only.png`：audio-only cue → real paired image + audio
  - `outputs/outputs_v11e/figures/demo_img_only.png`：image-only cue → real paired image + audio
  - `outputs/outputs_v11e/figures/demo_both.png`：双模态 cue → real paired image + audio
  - random 可视化会输出 `demo_aud_only_random.png` / `demo_img_only_random.png` / `demo_both_random.png`
  - 评估表：`outputs/outputs_v11e/tables/demo_eval_table.txt`（无 category 列/目标）
  - 全量 eval 表格图（按 family 子目录）：`tables/family01_occlusion_time_mask/full_eval.png` 等；
    生成：`python scripts/plot_eval_summary.py outputs/outputs_v11e/logs/eval_v11e_fixed_mask_sev04.log`

---

## 5. 模块与架构图对应关系

| 代码 | 架构图模块 |
|------|-----------|
| `models/encoders.py :: ImageSNNEncoder` | **Image SNN Encoder**（首脉冲 + trace → LIF） |
| `models/encoders.py :: AudioSNNEncoder` | **Audio SNN Encoder**（重复电流 / Poisson → LIF） |
| `models/memory.py :: KeyLayer`（×2：`K_img`, `K_aud`） | **Key 层**（记忆层内部，独立权重） |
| `models/memory.py :: RecurrentIndexLayer` | **Index 层 A**（唯一循环吸引子核心） |
| `models/memory.py :: ValueLayer`（×2：`V_img`, `V_aud`） | **Value 层**（A 路 + target 路分离） |
| `models/memory.py :: CrossModalAttractorMemory` | 中央记忆模块整体 |
| `models/decoders.py :: ClassifierHead / ImageDecoder / AudioDecoder` | 分类 MLP + 图像/音频 **CNN decoder** |
| `models/network.py :: CrossModalSNN` | 完整网络 + cue/phase 路由 |
| `models/lif.py` | LIF 神经元 + surrogate gradient |
| `data/corruption.py` | 图像 / 音频 cue 损坏函数 |

**Key 投射到 Index**（独立电流求和，不拼接）：

```
I_A = alpha_img * W_img_to_A(K_img) + alpha_aud * W_aud_to_A(K_aud)
      + [use_recurrent] W_rec(prev_spikes) - [use_kwta] 竞争抑制
```

**对侧 Key 条件化 Decoder**（不修改 Memory、Decoder 和 Refiner 内部结构）：

```
image Decoder: (V_img_from_A + gate(K_aud) * P_aud_to_img(K_aud))
               concat same-modal image detail -> Image Decoder
audio Decoder: (V_aud_from_A + gate(K_img) * P_img_to_aud(K_img))
               concat same-modal audio detail -> Audio Decoder
```

`P_*` 零初始化，因此新模块初始行为与 control 一致。对侧 Key 默认 detach，重建梯度
只训练投影器、门控和下游重建模块，不通过这条捷径改写 Key/Index。`--cross_key sweep`
在同一批样本上配对比较 correct / zero / wrong-class Key，并检查共享 Index 不变。
control 仍构造相同 projector/gate 以保持参数规模和后续模块初始化一致，但前向严格旁路。

**v11d 对侧实例 Detail 条件化**（从 Key 前一层提取，使用逐维门控）：

```
image detail channel += VectorGate(V_img, P_a2i(D_aud)) * P_a2i(D_aud)
audio detail channel += VectorGate(V_aud, P_i2a(D_img)) * P_i2a(D_img)
```

Cross-Detail residual 与原本同模态 detail 相加，因此不改变 v11c Decoder 的输入尺寸，
可以安全继承父 checkpoint。`--cross_detail sweep` 使用同类别、不同 `pair_id` 的
困难负样本，验证恢复结果是否真正依赖正确配对，而不是只依赖数字类别。

### 关键设计：杜绝答案泄漏
- decoder 的 **Value 主输入永远来自 `v_*_from_A`**（Index 驱动的 Value），
  从不读「A + Encoder 混合」Value。v10a 可额外融合 cue detail，但该 detail
  只来自当前 cue，缺失模态用 0；重建 loss 默认不通过 `V_from_A` 反向拖动 Index/Value。
- target（干净输入）**只在训练 binding 阶段**经 `W_enc_to_V` 进入 *target Value*，
  仅用于 `bind loss` 的 teacher，且与 `v_*_from_A` 在 `ValueLayer` 中走**两条独立路径**。
- `network.forward` 中 `training_mode=False` 时**强制** `phase="readout"` 并丢弃 target。

### k-WTA 竞争
- 方案 A（默认 `wta_mode: kwta`）：每步保留膜电位 top-k 的 Index 神经元。
- 方案 B（`wta_mode: inhibition_pool`）：全局抑制池均匀反馈抑制。

---

## 6. 数据集

- **图像**：MNIST（离线回退合成斑点图），值域 [0,1]。
- **音频（默认）**：**FSDD** 真实 wav → **log-mel spectrogram**
  （`torchaudio`，默认 **64 mel × 64 帧** = `aud_in: 4096`），decoder target 使用训练集全局 p1-p99 归一化，encoder cue 使用 hybrid norm。
  - **Audio Encoder 输入**、**Audio Decoder 输出**、**audio recovery loss** 均为 log-mel `[n_mels, n_frames]`。
  - v11d 固定增广一一配对：MNIST 数字 y 只配 FSDD spoken digit y；每张图像使用
    唯一 `pair_id` 和增广种子生成稳定音频实例，虚拟音频样本数与图像样本数相同。
  - 每个清单记录基础 FSDD 文件、基础池索引和增广种子，可完整复现。
  - 数据目录：`_data/fsdd/recordings`；`auto_download: true` 时自动 zip 下载。
- **冒烟**：`audio.use_real_audio: false` 仅用随机伪音频（`scripts/smoke_test.py` 不拉数据）。

## 7. cue 模式与损坏

8 种 cue 模式（`common.py :: CUE_MODES`）：原 6 种模式加
`clean_img_corrupt_aud` / `corrupt_img_clean_aud`，采样概率见
`configs/v11d.yaml :: cue_modes`。

损坏函数（`data/corruption.py`，`severity∈[0,1]`）：
- v11d 图像训练/主评估 family：`occlusion` / `pixel_delete` /
  `mask_vertical` / `mask_horizontal` / `salt_mask`。
- v11d 音频训练/主评估 family：`time_mask` / `freq_mask` /
  `feature_dropout` / `partial_temporal` / `time_freq_block`。
- `gaussian` 与方向化 `mask_left|right|top|bottom` 仍可由损坏函数单独调用，
  但不进入 v11c 的五 family 主结论。

## 8. 消融开关（`configs/v11d.yaml :: ablation`）

| 开关 | 作用 |
|------|------|
| `use_recurrent` | Index 是否启用 E↔E 循环 `W_rec` |
| `use_kwta` | Index 是否启用 k-WTA 竞争 |
| `use_binding_phase` | 是否启用 binding 阶段（`bind loss`） |
| `use_delayed_value_target` | binding 阶段 target Value 是否延迟 |
| `use_modality_dropout` | 是否启用单模态 cue（关闭则只用双模态） |

改对应字段为 `false` 即可一键消融，无需改代码。

## 9. 当前简化假设

1. `audio.use_real_audio: false` 时仅为冒烟随机伪音频（正常训练必须 FSDD）。
2. `A→V` 绑定用 surrogate-gradient 反向传播 + `bind loss`，**非** STDP/Hebbian。
3. Value/Index 状态读取为时间窗内**平均发放率**；decoder 为 rate-based MLP。
4. 默认重复电流编码（`encoding: current`），可切 `poisson`。
5. k-WTA 方案 A 为硬 top-k（前向硬选择，梯度经 surrogate 近似）。

## 10. 最小验收目标

在 MNIST + FSDD 音频上验证：残缺/干净 cue → 图像恢复、音频恢复、分类，
依次运行 `scripts/train.py` → `scripts/evaluate.py` → `scripts/demo_inference.py`。
