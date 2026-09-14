# 跨模态循环吸引子 SNN：v13pro

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
| cue_cnn / conditioned_cnn | 各从头 30 轮 | 本模态 mask-aware / 预测类别条件 CNN |
| recognizer | 独立从头 30 轮 | 只看 clean 训练集，不参与恢复 loss |

默认 3 seeds：1234、2345、3456，共 27 个训练任务、累计 1020 model-epochs。
不同结构每轮耗时不同，不能把 model-epochs 当 GPU 小时。CNN 是简单筛查基线，
不是参数匹配 ANN；其 30 轮也不等于 SNN 的 100+30 轮总预算。
可用 `--baseline_epochs 130` 补做更充分的 CNN 预算对照，不用测试集挑预算。

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

机制探针包含撤去完整外部电流、膜电位扰动、活动轨迹与保类统计；它们是操作性证据，
不是吸引子存在定理或能耗优势证明。默认记录完整前向的耗时与参数量，不宣称硬件能效。

完整设计见 [idea_report](docs/idea_report.md)，实现与参数见
[implementation](docs/implementation.md)，工程检查及真实实验结果只归档到
[dev_log](docs/dev_log.md)。第二数据集、参数/预算匹配 ANN、正式文献基线和实际多 seed
训练结果仍需完成；工程测试通过不等于这些研究证据已经具备。
