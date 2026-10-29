# 跨模态循环脉冲记忆：v15pro 局部污染检测与恢复

v15pro 从 v13pro（`5983308`）派生，新增图像/音频局部污染检测器，使用预测掩码进行选择性恢复。检测器由 MIT 许可的 [CBDNet-pytorch FCN](https://github.com/IDKiro/CBDNet-pytorch) 适配，原噪声水平回归改为污染支持二元分类；没有附带预训练权重。来源与许可见 [third_party/cbdnet](third_party/cbdnet/)。

支持随机局部高斯污染和局部脉冲污染。两阶段分别训练检测器与恢复网络；真实污染 mask 只用于监督、区域评分以及明确标记的 oracle 诊断。默认关闭 Cross-Key/Cross-Detail 条件，保留原共享 Index。当前交付是可运行代码，尚未完成新版本的三种子正式实验。

```bash
# 只显示计划；添加 --run 开始检测器预训练、恢复训练与评估。
python scripts/run_v15pro_suite.py
python scripts/run_v15pro_suite.py --run --device cuda
```

输出为 `outputs/v15pro/seed_*/`，分别包含检测器、恢复 checkpoint，以及 predicted/oracle/all 三种掩码条件的 JSON 和逐样本 CSV。已有数据可通过 `--data-root`、`--fsdd-root` 指定绝对路径；缺少 CUDA 时须显式选择 `--device cpu`。完整设置、初始化与断点续训说明见 [v15pro 文档](docs/v15pro.md)。

功能验证：

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

## 保留的 v13pro 基线

v13pro 在 v12b 模型结构上补充面向论文的实验基础设施，不宣称已经达到 CCF C 录用标准。
保留 MNIST/FSDD 类别级绑定、Value + gated own cue、simultaneous、batch 128，以及
恢复 loss 经 Value 到 Index 的梯度隔离。默认不需要 v11g/v12a/v12b 的旧权重。

## 实验内容

| 实验 | 每个 seed 的训练预算 | 含义 |
|---|---:|---|
| parent | 从头 100 轮 | 新划分、单尺度结构，验证集选 best |
| control | 同一 parent 后 30 轮 | 单尺度等额外预算对照，不再是零轮冻结参考 |
| main | 同一 parent 后 30 轮 | 多尺度局部 cue + 中间 Cross-Key |
| no_causal | 同一 parent 后 30 轮 | 去掉排名正则，保留 Cross-Key |
| no_cross | 同一 parent 后 30 轮 | 去掉 decoder Cross-Key，保留多模态 Index |
| classifier | 从头 30 轮 | predicted/soft medoid 基线；oracle medoid 只作诊断 |
| cue_cnn / conditioned_cnn | 各从头 30 轮 | 输入公平的轻量 mask-aware / 类别条件 CNN |
| matched_cnn | 从头 130 轮 | 按 SNN 参数量自动匹配宽度的 ANN 对照 |
| recognizer | 独立从头 30 轮 | 只看 clean 训练集，不参与恢复 loss |

默认 3 seeds：1234、2345、3456，共 30 个训练任务、累计 1410 model-epochs。
不同结构每轮耗时不同，不能把 model-epochs 当 GPU 小时。`matched_cnn` 是参数规模
公平对照；其它 CNN 仍是轻量筛查基线。matched CNN 的 130 轮预算在运行前固定，
不使用测试集挑预算。

## 数据与统计

MNIST 官方训练集分层留出约 6000 张作验证，其余约 54000 张训练，官方 10000 张测试。
FSDD index 10-49 训练、5-9 验证、0-4 测试；完整数据对应 2400/300/300 条录音。
归一化和类别 medoid 只来自新训练集。验证集负责选权重，不使用测试集选择 epoch。

mask seed 与训练 seed 分离，mask 按测试样本身份生成，不受 batch 大小影响。
全测试集逐样本指标包括分类、图像/音频误差、缺失/可见区、能量、独立内容识别和
Cross-Key 干预。全局简化 SSIM 明确命名 `global_ssim`，不是标准滑窗 SSIM。
三种音频部分残缺 cue 的 masked MSE 是预先指定主要终点，partial_temporal 单列。

bootstrap 按录音或 speaker 聚类，不能将 10000 次音频配对当作 10000 条独立录音。
训练 seed 标准差与固定模型的测试集区间分开报告。

## 一条命令

在 GPU 服务器的项目根目录运行；无需套用旧服务器的绝对路径：

```bash
nohup python -u scripts/run_v13pro_suite.py --run > v13pro_suite.log 2>&1 < /dev/null &
```

先完成训练，再全量评估和统计。SSH 断开后先检查进程；任务确实退出时重敲同一命令，
会校验完成记录并跳过已完成任务，未完成训练从 last checkpoint 续训。不要同时启动两份。

```bash
tail -f v13pro_suite.log
python scripts/run_v13pro_suite.py --dry_run
python scripts/smoke_test.py
```

输出在 `outputs/v13pro/`，逐实验 YAML 自动生成，不向 `configs/` 添加旧版本副本。
best.pt 用于评估，last.pt 用于恢复。不要直接训练 `configs/v13pro.yaml`，它只是套件模板。
更改代码、配置或协议时用新的 `--output`，不覆盖已有实验清单。

## 可选扩展

- `--mechanism_ablations`：无循环/无 kWTA 的父模型也从头训练，各 100+30 轮。
- `--severities 0.2 0.4 0.6 --mask_seeds 5678 6789 7890`：多强度、多 mask 重复。
- `--all_family_pairs`：五类图像 × 五类音频的 25 组合，显著增加评估量。
- `--speaker_test jackson --speaker_val nicolas --output outputs/v13pro_speaker`：指定说话人留出。
- `--holdout_audio_family partial_temporal --output outputs/v13pro_ood`：训练不见该 family，测试仍覆盖。
- `--eval_only --run`：只评估已完成权重；须保持原来的 seeds、预算及协议参数。
- `python scripts/paper_profile.py --root outputs/v13pro`：汇总每个 checkpoint 的参数、
  延迟、操作估计、脉冲率和 CUDA 峰值显存。

机制探针包含撤去完整外部电流、膜电位扰动、活动轨迹与保类统计；它们是操作性证据，
不是吸引子存在定理或能耗优势证明。默认记录完整前向的耗时与参数量，不宣称硬件能效。

完整设计见 [idea_report](docs/idea_report.md)，实现与参数见
[implementation](docs/implementation.md)，工程检查及真实实验结果只归档到
[dev_log](docs/dev_log.md)。CCF-C 级真实多 seed 训练仍需执行；CCF-B 的第二数据集和
正式文献基线仍需具备真实数据/实现后运行。工程测试通过不等于这些研究证据已经具备。
