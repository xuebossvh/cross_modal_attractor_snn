# 实现指南：Cross-Modal Attractor SNN

> 当前实现分支：`v14pro`。先看本文，再看 `docs/dev_log.md` 的真实结果。
> 研究假设见 `docs/idea_report.md`；用户约束见 `docs/user_requirements.md`。

## 0. 当前边界

v14pro 保留 v13pro/v12b 模型主线，新增 CCF-B 双轨实验协议，而不是新的恢复结构。
MNIST/FSDD 仍是类别级 many-to-many；保留 simultaneous、batch 128、
Value + gated own cue、`detach_value_for_recon=true`、多尺度局部音频 cue 和 Cross-Key。
正式套件从新划分重新训练父模型，不加载曾看过新验证子集的旧版本权重。第二轨道使用
`paired_manifest`，没有真实 manifest 时严格失败，不回退到 MNIST/FSDD 伪配对。

实现可运行不代表论文实验已完成，更不代表已经达到 CCF C 或 CCF B 录用要求。
v14pro 补充第二真实数据集的 manifest 校验、speaker/OOD 泛化入口和外部文献基线审计
接口；实际第二数据集与外部基线仍需真实数据/代码后运行，不能以占位脚本冒充完成。

## 1. 文件职责

| 路径 | 职责 |
|---|---|
| `configs/v14pro.yaml` | 唯一活动模板，禁止直接用于 train.py |
| `scripts/run_v14pro_suite.py` | 生成逐实验 YAML、锁定计划、顺序训练/评估/统计、重启 |
| `scripts/job_runner.py` | 记录子进程日志，失败停止，清理子进程 |
| `data/splits.py` | train/val/test 划分、音频内容指纹、划分审计 |
| `data/dataset.py` | 类别组合、训练集 medoid、稳定测试身份 |
| `data/audio_features.py` | 64x64 log-mel；仅新训练录音的归一化 |
| `models/network.py` / `memory.py` | 原 SNN 前向、可选 Index 动态探针 |
| `models/paper_baselines.py` | Clean CNN、mask-aware CNN、类别条件 CNN、参数匹配 ANN |
| `scripts/train.py` | SNN 父模型/恢复分支训练和 best/last checkpoint |
| `scripts/paper_baseline.py` | 分类、独立识别器与 CNN 恢复训练 |
| `scripts/paper_validation.py` | 新验证集上的音频部分缺失主要终点 |
| `scripts/paper_evaluate.py` | 全测试集、逐样本指标、Cross-Key、机制和耗时 |
| `scripts/paper_statistics.py` | 跨 seed 均值/标准差、配对聚类 bootstrap |
| `scripts/paper_profile.py` | 参数、MAC 上界、激活操作估计、脉冲率、延迟和 CUDA 显存 |
| `scripts/validate_paired_manifest.py` | 严格检查第二真实视听数据集 manifest 与 asset split |
| `scripts/audit_external_baseline.py` | 检查正式文献基线的来源、预算、数据和 checkpoint 哈希 |
| `scripts/test_paper_protocol.py` / `smoke_test.py` | CPU 离线回归与完整链路测试 |

逐实验配置和产物只在 `outputs/v14pro*/`，不进代码仓库。旧版本 YAML/专用 suite/smoke
保留在旧分支；迁移时本地旧文件可保留但不跟踪，不能再作为当前命令入口。

## 2. 数据与目标

默认 `data.paper_split.enabled=true` 由 suite 注入，旧默认数据协议不被暗中修改。

| 数据 | train | validation | test |
|---|---|---|---|
| MNIST | 官方训练集按类别留出后约 54000 | 约 6000，固定 split seed=20260915 | 官方 10000 |
| FSDD official | index 10-49，完整数据 2400 条 | index 5-9，300 条 | index 0-4，300 条 |
| FSDD speaker | 除留出 speaker 外全部录音 | 明确 val speaker | 明确 test speaker |

第二数据集使用 `data.dataset=paired_manifest` 和 CSV manifest；必需字段为
`pair_id, source_id, image_source_id, audio_source_id, speaker_id, split, label,
image_path, audio_path`。每行必须是唯一真实 source event；图像与音频 source_id 必须相同，
speaker/source 不得跨 split，路径必须存在，且每个 split 覆盖全部类别。音频必须提供
`64x64` log-mel 的 `.pt/.npy`，或明确使用 WAV 和 train-only normalization。manifest
模式的缺失模态 target 仍只从 train 构建 medoid。

音频归一化的缓存指纹包含训练录音内容 SHA256、划分和特征参数；不匹配则重算。
medoid 只由新 train 子集构建。MNIST 加载失败不允许静默切换为合成图像。
`split_val.json`、`split_test.json` 保存图像索引和录音列表，阻止同一计划中途换数据。

| cue | image target | audio target |
|---|---|---|
| image-only | 当前 sample | train category medoid |
| audio-only | train category medoid | 当前 sample |
| image+audio | 当前 sample | 当前 sample |

没有真实实例绑定，不启用 exact-pair InfoNCE/Cross-Detail。测试配对是稳定的同类组合，
`evaluation_identity` 保存 image_id、audio_id、speaker；复用录音不算独立新音频样本。

## 3. 模型与梯度

| 信号 | 维度 |
|---|---|
| image / audio | [B,1,28,28] / [B,64,64] |
| encoder / Key spikes | [20,B,128]，各模态独立 |
| Index spikes / rate | [20,B,512] / [B,512] |
| image / audio Value state | [B,384] / [B,768] |
| own detail | [B,128] / [B,256] |
| decoder input | [B,512] / [B,1024] |
| audio local rates | conv1 [B,32,32,32]；conv2 [B,64,16,16] |

```text
cue -> SNN Encoder -> Key -> simultaneous recurrent Index -> Value -> Decoder
  |                                         |                  ^
  +-> own detail / local rates              +-> classifier     |
opposite Key -> masked Cross-Key feature modulation ------------+
```

父模型从头训练全部可训练参数；恢复分支仅放开 audio decoder、local projectors、
image/audio Cross-Key adapters。图像基础 decoder/refiner 冻结，但图像 Cross-Key adapter
仍可训练，不能称整个图像路径冻结。恢复 loss 经 Value 回到 Index 的路径始终 detach；
从头训练阶段 own cue 等其它梯度路径仍存在，不能说整个基础模型从一开始都冻结。

control 使用单尺度局部 cue、末层 Cross-Key；main 使用多尺度和中间 Cross-Key。
no-cross 仅禁用 decoder Cross-Key，Index 仍接受双模态，不能叫“完全无跨模态联系”。

## 4. 预算与模型选择

| 实验 | 初始化 | 默认轮数 | 比较边界 |
|---|---|---:|---|
| parent | 从头 | 100 | 新划分重新学习，不是原 v12a checkpoint |
| control | 同 seed parent best | 30 | 与 main 相同额外优化预算 |
| main | 同 seed parent best | 30 | v12b 多尺度结构 |
| no_causal | 同 seed parent best | 30 | Cross-Key 排名正则关闭 |
| no_cross | 同 seed parent best | 30 | decoder 跨 Key 条件关闭 |
| classifier | 从头、与 external 不同种子偏移 | 30 | 预测类别/soft medoid |
| recognizer | clean-only 独立 CNN | 30 | 不参与恢复 loss 的外部内容识别 |
| cue_cnn / conditioned_cnn | 从头 | 各 30 | 输入公平的轻量 ANN 基线 |
| matched_cnn | 从头 | 130 | 按 SNN 参数量选宽度的 ANN 公平基线 |

默认 seeds=1234/2345/3456，30 个训练任务，累计 1410 model-epochs；其中
`matched_cnn` 使用 130 轮，避免把低预算 CNN 当作公平强基线。
`--baseline_epochs 130` 可提高四个 CNN/分类任务预算；不能将其默认 30 轮与 SNN
100+30 轮称为总预算匹配。`--mechanism_ablations` 另加每 seed 两组 100+30 轮：
no-recurrence 和 no-kWTA 的父模型也重新训练，默认合计 1800 model-epochs。

所有恢复模型只用 val 的三个音频部分缺失 cue 的 masked MSE 选择 best：
corrupt_aud_only、clean_img_corrupt_aud、corrupt_both。全验证集参与；family 按 batch
均衡轮转、分组等权宏平均，不是每条验证样本都遍历五个 family。留出 family 不参与
训练或验证选择。分类/外部识别器按 clean val 错误率选择。

`last.pt` 恢复优化器、scheduler 和 epoch；`best.pt` 只用于评估。checkpoint 原子替换，
先保存 best 再保存 last。新 suite 的训练 RNG 按 epoch/step 固定，避免模型额外随机
计算改变下一步 cue；不能承诺所有 CUDA 算子跨硬件逐位一致。实际轮数、seed 和预算
必须随最终结果报告；100/30 是预算，不是收敛保证。

## 5. 全量评估与统计

默认在全部训练完成后，遍历全部 10000 个测试组合、8 cue、fixed 五组 family 和
random。fixed 默认为五组配对 family，不是完整 25 组合；`--all_family_pairs` 展开 25。
`--severities`、`--mask_seeds` 可扩展强度和重复，mask 由样本身份生成，独立于模型 seed。

- 主指标：Index ACC（SNN）、classifier ACC（CNN）、image/audio MSE、global SSIM、
  PSNR、missing/visible MSE/L1、missing fraction、独立恢复内容 ACC。
- 音频辅助：foreground missing MSE、top15 recall、rec/target mean/std/max。
- Cross-Key：normal/zero/wrong/same-class 绝对 missing MSE、gain/damage、win、gate/ratio。
  source 不存在或没有有效区域/匹配时为 NaN，不用 0 替代。
- 外部识别器另报 clean test ACC；较差识别器不能作为可信恢复质量证据。oracle medoid
  使用真值标签，仅为诊断，不能列为可部署方法。
- `global_ssim` 明确是原全局简化定义，不是标准滑窗 SSIM。`paper_profile.py` 记录
  批次前向 latency、参数量、模块 MAC 上界、按非零输入比例加权的操作估计、各级脉冲率
  和 CUDA 峰值显存；操作估计不是精确 FLOPs/SOP，显存/延迟需在目标 GPU 实测，不能据此
  宣称芯片能耗或能效优势。
- 每个 SNN 额外全测试集运行 Index 探针：撤去 1/4、1/2、完整 T 后的外部输入电流
  （包含 bias），继续 10 步并施加 std=0.2 膜电位扰动。报告 5 步窗口 rate 的分类、
  一致率、膜电位/活动距离。未扰动变静默也可能“一致”，不能据此宣称吸引子已证实。

`per_item.csv.gz` 保存身份、family、seed 和所有已计算指标；`summary.json` 保存各场景
全量均值和有效 n；`index_probes.json` 保存动态曲线；`complete.json` 保存配置、权重、
外部识别器、测试清单及产物校验和。`--smoke_batches` 强制写入 smoke_evaluation，
明确 full_test=false，不能进入正式汇总。

`statistics.json` 分开报告训练种子的均值/样本标准差、每个固定模型的配对聚类区间。
主要终点是 fixed severity=0.4 下三种音频部分残缺 cue 的 missing MSE，partial_temporal
单列。比较 main/control、no-causal/control、no-cross/control、main/no-causal、main/no-cross。
bootstrap 对 cluster 内配对曝光先取均值，再等权重采 cluster；不是曝光加权估计。
录音 cluster 不处理同 speaker 内相关性，`--cluster speaker` 可补敏感性分析；六个
speaker 的区间也很不稳定。多指标探索不自动产生显著性结论。

## 6. 运行命令

在服务器当前项目根目录，先激活已有 CUDA Python 环境。默认无旧权重依赖：

```bash
python scripts/run_v14pro_suite.py --dry_run
nohup python -u scripts/run_v14pro_suite.py --run > v14pro_suite.log 2>&1 < /dev/null &
tail -f v14pro_suite.log
```

确实退出后重跑同一命令；先检查旧进程，禁止同时运行两份。已完成任务检查配置/代码/
权重及评估产物 SHA 后跳过，未完成训练恢复 last。修改计划要用新的 --output。
同样参数加 `--eval_only` 可只评估。任务异常中止，不把日志存在当完成。

```bash
python scripts/run_v14pro_suite.py --run --speaker_test jackson --speaker_val nicolas --output outputs/v14pro_speaker
python scripts/run_v14pro_suite.py --run --holdout_audio_family partial_temporal --output outputs/v14pro_ood
python scripts/validate_paired_manifest.py --manifest /path/to/paired_manifest.csv
python scripts/run_v14pro_suite.py --run --dataset paired_manifest --manifest /path/to/paired_manifest.csv --output outputs/v14pro_paired
python scripts/paper_statistics.py --root outputs/v14pro --cluster speaker
python scripts/paper_profile.py --root outputs/v14pro
python scripts/smoke_test.py
```

预算、强度和 seeds 请在训练前确定。例如 `--severities 0.2 0.4 0.6 --mask_seeds 5678 6789 7890`
会显著增加评估量。`--speaker_test/--speaker_val`、`--holdout_audio_family` 和
`--dataset paired_manifest` 是 CCF-B 扩展。没有第二数据集 manifest 时，paired 命令
必须失败；正式文献基线需要外部实现/结果和审计信息，不能用内部 CNN 代替。

## 7. 文档归档

历史版本的完整结果按日期留在 `dev_log.md`，不要把旧运行说明作为当前入口。
v14pro 的代码验证与状态追加在 2026-09-15 下；CUDA 正式实验、逐实验各指标、异常及
限制待实际运行后再归档。不要另建版本专用结果 Markdown，也不要伪造缺失指标。
