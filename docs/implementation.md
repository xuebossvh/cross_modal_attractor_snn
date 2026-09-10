# 实现指南：Cross-Modal Attractor SNN

> 当前实现分支：`v11g`
> 当前目标：冻结 `v11e_control`，只训练缺失区域的 Masked Feature Cross-Key adapter。
> 文档关系：初始设计见 `docs/idea_report.md`，过程与结果见 `docs/dev_log.md`，用户约束见 `docs/user_requirements.md`。

## 0. 当前版本边界

v11g 保留 MNIST + FSDD 类别级绑定、`Key -> simultaneous recurrent Index -> Value`、
`Value + same-modal cue detail` decoder 输入、`detach_value_for_recon=true`、
`batch_size=128` 和五类均衡缺失采样。它只增加一个受 mask 限制的 decoder 特征调制层，
不增加 GRID、固定伪配对、Cross-Detail、pair alignment 或新的 residual decoder。

v11g 的四个研究动作是：

1. 冻结 `v11e_control`，隔离恢复训练对 Index 和分类状态的影响。
2. 将 Cross-Key 作用位置从 Value 改为 decoder 的空间/时频特征，并只作用于缺失区。
3. 用 zero/wrong reference 的 margin 训练正确的对侧 Key，而不让 reference 反向传播。
4. 同时报告 Index 分类、恢复内容内部分类、区域误差和 Cross-Key 干预结果。

历史版本的设计动机和结果仍在 `idea_report.md`、`dev_log.md` 中；本文不重复展开已经
废止的版本配置。旧配置只能在对应 Git 分支中使用。

## 1. 真实目录与职责

```text
cross_modal_attractor_snn/
├── configs/
│   ├── v11g.yaml
│   ├── v11g_control.yaml
│   └── v11g_no_causal.yaml
├── data/                 # MNIST/FSDD、log-mel 和 corruption
├── models/               # LIF、encoder、memory、decoder、顶层网络
├── scripts/              # train/evaluate/demo/suite/smoke
├── docs/                 # 核心设计、实现、日志和用户约束
├── _data/                # 本地数据，不进代码仓库
└── outputs/              # 本地运行产物，不进代码仓库
```

| 路径 | 职责 |
|---|---|
| `common.py` | 配置合并、seed、cue、target、公共指标 |
| `data/dataset.py` | MNIST/FSDD 类别级 many-to-many 数据集与 train medoid |
| `data/audio_features.py` | WAV 到 64x64 log-mel 及训练集归一化统计 |
| `data/corruption.py` | 五类图像和五类音频缺失，返回 `1=missing` mask |
| `models/encoders.py` | 图像/音频 SNN encoder，输出时间脉冲序列 |
| `models/memory.py` | Key、循环 Index、Value 和 binding/readout |
| `models/decoders.py` | decoder、refiner 和 `MaskedCrossKeyAdapter` |
| `models/network.py` | 前向接线、冻结父模型、门控融合、Cross-Key 干预 |
| `models/frozen_base.py` | 父 checkpoint SHA256、配置和冻结 state 校验 |
| `scripts/train.py` | decoder/主训练、重建损失、causal margin |
| `scripts/evaluate.py` | fixed/random、family、Cross-Key sweep 和 CSV |
| `scripts/demo_inference.py` | fixed/random 小样本可视化 |
| `scripts/run_v11g_suite.py` | 顺序执行 train、eval、demo，可选 no-causal |
| `scripts/smoke_test_v11g.py` | v11g 的 shape、梯度、严格加载和 CLI 回归 |

## 2. 数据、target 与张量

MNIST 图像为 `[B,1,28,28]`，FSDD 音频先转成 `[B,64,64]` 的 log-mel。两种模态只共享
digit label，因此同类别内随机组合；没有人工 instance pair。缺失模态的类别 target
是只由训练集构建的 class medoid，不能使用测试集样本构建原型。

| cue | 图像 target | 音频 target |
|---|---|---|
| `image-only` | 当前图像 sample | 训练集音频 category medoid |
| `audio-only` | 训练集图像 category medoid | 当前音频 sample |
| `image+audio` | 当前图像 sample | 当前音频 sample |

脉冲序列使用 `[T,B,D]`，rate/state 使用 `[B,D]`，当前 `T=20`。

| 信号 | shape | 说明 |
|---|---:|---|
| image/audio encoder output | `[20,B,128]` | 模态编码脉冲 |
| `K_img`, `K_aud` | `[20,B,128]` | Key LIF 输出脉冲 |
| `Index A` | `[20,B,512]` | 两个 Key 同时驱动的循环层 |
| `index_state` | `[B,512]` | Index rate，送分类器 |
| `V_img_from_A` | `[B,384]` | 图像 Value state |
| `V_aud_from_A` | `[B,768]` | 音频 Value state |
| image/audio own detail | `[B,128]` / `[B,256]` | cue rate 经 projector |
| image/audio decoder state | `[B,512]` / `[B,1024]` | Value 与 gated detail 拼接 |
| decoder feature | `[B,32,28,28]` / `[B,16,64,64]` | Cross-Key 调制前特征 |
| final output | `[B,1,28,28]` / `[B,64,64]` | 图像概率图 / log-mel |

## 3. 前向计算

```text
image cue -> image encoder -> K_img -+
                                      +-> simultaneous recurrent Index -> V_img/V_aud
audio cue -> audio encoder -> K_aud -+                         |
                                                                +-> classifier

V_img + gated image detail -> ImageDecoder feature F_img
K_aud -> masked feature adapter -> F_img' -> image head/refiner

V_aud + gated audio detail -> AudioDecoder feature F_aud
K_img -> masked feature adapter -> F_aud' -> audio head
```

`_fuse_decoder_state` 先根据 `detach_value_for_recon` detach `V_from_A`，再把本模态
cue rate 投影到 detail channel，并用 `sigmoid(Linear([Value,detail]))` 做逐维 detail
门控。门控只改变送入 decoder 的 detail，不改变 Key、Index 或原始 Value。

当 `cross_key_conditioning.mode=masked_feature` 时，decoder 先得到基础特征 `F`。
`MaskedCrossKeyAdapter` 将对侧 Key rate 投影为 `key_channels=16` 的条件，并计算：

```text
F' = F + M * gate(F, cue, M, source_quality) * delta(F, K_other)
```

其中 `M=1` 是目标缺失区。adapter 的输出层零初始化，所以未训练时与父模型一致；
zero Key、缺少对侧 cue 或目标没有缺失区时不产生修正。head/refiner 后再次用 mask
保护可见位置。v11g 的 Cross-Key 不写回 Value，也不改变分类 logits。

## 4. 训练与三组实验

| 配置 | 父权重 | 额外训练 | 可训练参数 | causal |
|---|---|---:|---|---|
| `v11g_control` | `v11e_control` | 0 轮，仅评估 | 无 | 关 |
| `v11g` | `v11e_control` | 30 轮 | `img_cross_adapter.*`, `aud_cross_adapter.*` | 开，0.5 |
| `v11g_no_causal` | `v11e_control` | 30 轮 | 同上 | 关 |

三组固定 `seed=1234`、`batch_size=128`、`severity=0.4`、五类图像/音频 family 均衡
采样。control 是冻结父模型参考，不是等预算重训。causal loss 只在有效缺失区比较
正确 Key、zero Key 和异类 wrong Key；zero/wrong 前向在 `no_grad` 中执行，same-class
Key 只作诊断，不能作为负样本。

父权重必须通过配置中的 SHA256 和关键 forward 配置检查。冻结参数和 buffers 每次保存
前都计算摘要；缺失父权重、错误摘要或不兼容 checkpoint 直接失败。

## 5. 评估口径

`fixed_mask` 为主协议：按 `seed`、family 和 batch 确定 mask，同一位置可跨模型比较。
`legacy_random` 使用 `eval.random_seed=4321` 每个 batch 抽取 family 和 mask；相同 seed
可复现，但它不是单一固定 mask。两者不能混成一个平均值。

每个实验都必须逐 cue、逐 family 记录以下字段中实际存在的值：

- Index ACC；图像/音频 MSE、SSIM、PSNR；缺失区和可见区 MSE/L1。
- 恢复图像/音频经冻结原模型再分类的内部内容 ACC，并与 Index ACC 分开。
- Cross-Key `normal/zero/wrong/same-class` 绝对误差、relative gain、
  `win_zero/win_wrong/win_both`、gate、ratio 和有效 `n`。
- 音频 rec/tgt mean、std、max、top15% energy recall、family breakdown。
- 训练 loss、学习率、轮数和 demo 小样本统计，单独标记其非全测试集性质。

数值缺失写 `N/A` 并说明原因；不能用 0 替代。主实验、control 和每个消融必须有独立
结论，训练轮数、seed、mask 或 target 不一致时不能作强因果比较。完整结果直接追加到
`docs/dev_log.md`，原始 CSV/log/PNG 只作为 `outputs/` 证据。

## 6. 运行命令

必须在项目根目录、已激活环境和可用 GPU 中运行。suite 会按顺序运行，并把每个阶段的
stdout 同时写入对应日志；任何阶段失败都会停止后续任务。

```bash
# 主实验 + control；加 --with_ablations 才运行 no_causal
nohup python -u scripts/run_v11g_suite.py --with_ablations > v11g_suite.log 2>&1 < /dev/null &
tail -f v11g_suite.log

# 只评估已有三组 checkpoint
python -u scripts/run_v11g_suite.py --eval_only --with_ablations

# 单独评估；--family_breakdown 额外输出音频 family 表
python -u scripts/evaluate.py --config configs/v11g.yaml --protocol fixed_mask --severity 0.4 --cross_key sweep --family_breakdown
python -u scripts/evaluate.py --config configs/v11g.yaml --protocol legacy_random --severity 0.4 --cross_key sweep
python -u scripts/demo_inference.py --config configs/v11g.yaml --protocol fixed_mask --severity 0.4
python -u scripts/demo_inference.py --config configs/v11g.yaml --protocol legacy_random --severity 0.4
```

去掉 `--with_ablations` 时只运行 main 和冻结 control。`--max_batches` 只能限制评估，
不能缩短训练；`--resume` 只在目标训练 checkpoint 已存在时使用。

输出目录为 `outputs/outputs_v11g/`、`outputs/outputs_v11g_control/` 和
`outputs/outputs_v11g_no_causal/`；checkpoint 为 `outputs/checkpoints/` 下对应文件。
`outputs/`、`_data/` 和 checkpoint 默认不进入代码仓库。

## 7. 历史版本索引

| 版本 | 当前文档中的唯一入口 | 状态 |
|---|---|---|
| v9/v9a/v9b/v9c | `docs/idea_report.md` 与 `docs/dev_log.md` | 历史实验 |
| v10a-v10f | `docs/idea_report.md` 与 `docs/dev_log.md` | 历史实验 |
| v11a-v11e | `docs/idea_report.md` 与 `docs/dev_log.md` | 历史实验/已废止配置 |
| v11g | 本文第 0-6 节及 `docs/dev_log.md` 当前条目 | 当前实现 |

历史条目中的旧命令、旧输出路径和旧配置名称只用于追溯，不能复制到 v11g 运行。
当前分支 `configs/` 只保留 v11g 三个 YAML，避免把旧版本配置带入新分支。

## 8. 实现检查表

- [x] 当前目录树、模块职责和三份 v11g 配置与仓库一致。
- [x] MNIST/FSDD 64x64 log-mel、category target 和 tensor shape 已明确。
- [x] `Value + own detail`、`detach_value_for_recon` 和 masked Cross-Key 已明确。
- [x] main/control/no-causal 的父权重、预算、可训练范围和 causal 语义已明确。
- [x] fixed/random、逐实验指标和结果归档位置已明确。
- [ ] v11g 初始假设在 `idea_report.md`，完整实测结果待运行后追加到 `dev_log.md`。
