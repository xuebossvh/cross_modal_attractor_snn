# 跨模态循环吸引子 SNN：v11g

当前分支为 **v11g**。在已训练的 v11e_control 上冻结基础网络，只训练
**缺失区域 Cross-Key 特征调制**。研究目标是让正确的对侧 Key 改善部分残缺和
全缺失恢复，同时避免基础分类能力因恢复训练而漂移。

详细设计、实验协议与结果归档统一记录在
[implementation](docs/implementation.md) 和 [dev_log](docs/dev_log.md)；
历史版本配置请切换对应分支查看。

文档闭环：v11g 的初始方案与研究假设在
[idea_report](docs/idea_report.md)，最终实现与真实配置在
[implementation](docs/implementation.md)，逐实验完整评估和结论在
[dev_log](docs/dev_log.md)。不再创建版本专用协议或结果 Markdown。

## 1. 四项修改

1. **局部条件化**：不再全局修改 Value。Value + gated own cue detail 经原
   decoder 得到特征，在末层卷积前用对侧 Key 调制缺失区域的空间/时频特征。
   仍是同一个 decoder，不增加独立的图像/音频输出 residual 分支。
2. **冻结基础模型**：Encoder、Key、Index、Value、Classifier、原 Decoder/Refiner
   和 own-detail fusion 的参数与运行状态均固定，只训练两个新 adapter。
   重建不能通过 Value 影响 Index，`detach_value_for_recon=true` 保持不变。
3. **正向因果目标**：正确 Key 的绝对恢复 loss，加上相对 zero/wrong-class 的
   margin loss；参考输出不反传，归一化有 batch floor，避免除以单样本近零误差。
   同类别其它实例不是负样本，不能通过降低 zero 基线质量制造收益。
4. **分开评价**：Index ACC、恢复内容类别一致性、masked MSE、normal/zero/wrong/
   same-class 对照及改善比例。门控非零不等于成功，必须看 normal 是否优于冻结 zero。

```text
MNIST -> Image Encoder -> Image Key --+
                                     +-> recurrent Index -> Value states
FSDD  -> Audio Encoder -> Audio Key --+        |
                                              +-> Classifier (frozen)

Value_img + gated own image detail -> Image Decoder features F_img
  F_img + masked Cross-Key modulation(K_aud) -> same decoder head -> image

Value_aud + gated own audio detail -> Audio Decoder features F_aud
  F_aud + masked Cross-Key modulation(K_img) -> same decoder head -> log-mel
```

adapter 使用目标 cue、缺失 mask、基础特征和对侧已知损坏比例作为上下文。
对侧 Key 投影后乘性调制局部特征；输出层零初始化。对侧不存在则该方向没有
Cross-Key；目标完全干净则不修改恢复。部分残缺时通路参与训练，是否真的帮助
由 paired 指标判断，不能把“允许模型忽略它”当作实验结论。

输出端再次按 mask 选择，保证可见位置和关闭通路时的输出与冻结基线一致。
这里的保护是保持**基线输出**，不意味着音频可见区一定等于原始 cue。

## 2. 数据与维度

MNIST 和 FSDD 仍采用同数字类别内的 many-to-many 随机组合。没有人工固定
实例对，不启用 GRID、Cross-Detail 或 exact-pair alignment。

| cue | 图像 target | 音频 target |
|---|---|---|
| audio-only | 训练集类别 medoid | 当前 clean sample |
| image-only | 当前 clean sample | 训练集类别 medoid |
| image+audio | 当前 clean sample | 当前 clean sample |

| 模块 | 图像 | 音频 |
|---|---:|---:|
| 输入 | 1x28x28 | 64x64 log-mel |
| cue encoder rate / Key rate | 128 / 128 | 128 / 128 |
| Value state | 384 | 768 |
| own detail 投影 | 128 | 256 |
| decoder 输入 | 512 | 1024 |
| adapter 所在特征图 | 32x28x28 | 16x64x64 |

Index=512、T=20、simultaneous、batch_size=128 保持不变。
五种图像 family：occlusion、pixel_delete、mask_vertical、mask_horizontal、salt_mask。
五种音频 family：time_mask、freq_mask、feature_dropout、partial_temporal、time_freq_block。
训练均衡轮转 family，severity 固定 0.4，不重复父模型低强度暖启动。

## 3. 前置条件

在**项目根目录**运行，使用已验证可 GPU 训练的环境。训练入口不再静默回退 CPU。
本分支不要求更换已工作的 PyTorch/CUDA；依赖见 `requirements.txt`。

- 数据：`_data/MNIST/`、`_data/fsdd/recordings/`、父模型使用的音频归一化统计。
- 父权重：`outputs/checkpoints/cross_modal_snn_v11e_control.pt`。
- 父权重 SHA256：
  `5a792f10e57a95947c8e51bd915b09897baee01475103010826f25275b71501b`。
- checkpoint 来自独立 checkpoint 仓库或已有 v11e 输出，不随代码仓库上传。
- 父模型缺失、摘要或关键配置不匹配直接失败；不要删掉这些检查来从随机模型起跑。

## 4. 一条命令顺序执行

```bash
nohup python -u scripts/run_v11g_suite.py --with_ablations > v11g_suite.log 2>&1 < /dev/null &
```

执行顺序：main 训练及评估/可视化 -> 冻结 control 评估/可视化 ->
no_causal 训练及评估/可视化。去掉 `--with_ablations` 不运行 no_causal。
主实验和 no_causal 各额外训练 30 轮；control 是父模型的固定零通路参考，
**0 轮额外优化，不是等预算重训的 control**。

每个配置运行 fixed normal+family breakdown、fixed sweep、random normal、
random sweep，以及 fixed/random 的三张 demo。日志同时输出终端并写入各自目录，
失败停止后续步骤；最后会输出 `[suite] ALL STAGES COMPLETED`。

```bash
tail -f v11g_suite.log
python -u scripts/run_v11g_suite.py --eval_only --with_ablations
python -u scripts/run_v11g_suite.py --resume
python scripts/run_v11g_suite.py --with_ablations --dry_run
```

`--resume` 要求对应训练 checkpoint 已存在。若主实验已完成但 no_causal 尚未
开始，不要给整个套件加 `--resume --with_ablations`；单独启动 no_causal。

## 5. 单独运行

```bash
python -u scripts/train.py --config configs/v11g.yaml
python -u scripts/train.py --config configs/v11g.yaml --resume
python -u scripts/train.py --config configs/v11g_no_causal.yaml

python -u scripts/evaluate.py --config configs/v11g_control.yaml --protocol fixed_mask --severity 0.4
python -u scripts/evaluate.py --config configs/v11g.yaml --protocol fixed_mask --severity 0.4 --cross_key sweep
python -u scripts/evaluate.py --config configs/v11g.yaml --protocol legacy_random --severity 0.4 --cross_key sweep
python -u scripts/demo_inference.py --config configs/v11g.yaml --protocol fixed_mask --severity 0.4
python -u scripts/demo_inference.py --config configs/v11g.yaml --protocol legacy_random --severity 0.4
```

fixed_mask 对每种 family 用确定性 mask；random 在每 batch 随机抽取 family 和
mask，使用独立 `eval.random_seed=4321` 以便重复比较。
`evaluate.py --random_seed 5678` 可换一组抽样；random demo 是小样本展示，
不等于全量 random 评估。十张示例不能替代全测试集统计。

## 6. 输出与验收

产物在 `outputs/outputs_v11g{,_control,_no_causal}/`：
`logs/`、`tables/`、`figures/`。随机 demo 带 `_random.png` 后缀。

`tables/eval_<protocol>_sev0.4_key_<normal|sweep>_detail_normal.csv` 为长表，
包含 family、cue、target 粒度、metric、value、有效样本数。
sweep 重点检查 `*_normal_mse`、`*_zero_mse`、`*_wrong_mse`、
`*_correct_gain`、`*_win_zero` 和 `*_win_both`。正 gain 才代表相对基线改善。

`content_img_*_acc` / `content_aud_*_acc` 是恢复内容经冻结原模型单模态再分类
的准确率，**只是内部一致性代理，不是独立识别器**；它与 Index ACC 不同。
原 demo 的 pred 注释仍来自 Index，不能据此声称恢复内容类别正确。

```bash
python -u scripts/smoke_test_v11g.py --cli
python -u scripts/smoke_test_v11g.py --parent outputs/checkpoints/cross_modal_snn_v11e_control.pt
```

上述为离线 CPU 回归，不下载数据，也不改现有训练 checkpoint。
完整 3080 GPU 收敛、各 family 的真实收益仍需正式训练与评估验证。
