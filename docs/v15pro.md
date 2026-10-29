# v15pro：局部污染检测与选择性恢复

v15pro 从 v13pro 的 `5983308` 代码基线派生。仓库已有 v14 相关工作，因此本次使用 v15pro，保留 [v13pro 配置](../configs/v13pro.yaml)和原有入口。这里新增的是可运行的训练与评估流程，尚无完整实验结果，不声称提高了恢复精度。

## 任务范围

本版本只处理图像或 log-mel 频谱中的**局部污染**。编码器、Key、Index、Value、解码器的层次与维度延续 v13pro；图像 Refiner 保留，音频外置 Refiner 继续旁路。检测器是新增模块。Cross-Key 条件、Cross-Detail 和 pair alignment 默认关闭，未开展新的跨模态条件机制训练。共享 Index 仍然接收两种可用模态。

两种污染 family：

| Family | 注入方式 | 区域参数 | 幅度参数 |
| --- | --- | --- | --- |
| `local_gaussian` | 随机矩形内叠加高斯噪声并裁剪到 [0,1] | `train_severity` 是近似面积比例 | `noise_std` 是标准差 |
| `local_impulse` | 随机矩形内各位置替换为随机 0 或 1 | 同上 | 正值不改变替换幅度；0 关闭污染 |

默认覆盖率 0.25，Gaussian 标准差 0.2。图像 28×28 对应 14×14 区域；频谱 64×64 对应 32×32 区域。整数尺寸会量化其他覆盖率。`severity=0` 或 `noise_std=0` 时返回原值与全 0 support。矩形以外数值精确保留。

真实 mask 标记**注入区域**，包括裁剪或替换后数值恰好没变的位置。它不是通过含噪值与干净目标比较生成的掩码。音频污染发生在归一化 log-mel 域，本实验不能直接解释为波形域背景噪声或给定 SNR 的语音去噪。

## GitHub 检测模块来源

两个模态分别使用一个五层 3×3 卷积检测器，来源为 MIT 许可的 [IDKiro/CBDNet-pytorch](https://github.com/IDKiro/CBDNet-pytorch)，固定上游提交 `09a2e55b2098039ee99ada8c634a06fc28c6d8a1` 中 [model/cbdnet.py 的 FCN](https://github.com/IDKiro/CBDNet-pytorch/blob/09a2e55b2098039ee99ada8c634a06fc28c6d8a1/model/cbdnet.py)。许可与来源记录保存在 [third_party/cbdnet](../third_party/cbdnet/)。

原模块估计 RGB 图像噪声水平，本地 [noise_detector.py](../models/noise_detector.py) 将它适配为单通道输入、单通道污染 logits，隐藏通道数为 32。输出不保留原估计器的非负激活，以便使用稳定的 logits BCE。没有下载或使用上游预训练权重；需要在本项目的局部污染任务上训练。这项适配不等同于原论文已经验证了本任务。

## 两阶段训练与信息边界

1. **检测器预训练**：仅将实际 cue 输入检测器，用合成污染的真实 support 监督二元分类；干净 cue 的标签为全 0。默认 10 轮、学习率 0.001、正类权重 3。根据独立验证集选择检测器最佳权重。
2. **恢复训练**：加载并冻结预训练检测器。检测概率以固定阈值 0.5 转为硬 mask，恢复模型据此选择性预测和回填。默认训练 100 轮、初始学习率 0.0002；除冻结检测器及原本不参与优化的参数外，恢复网络允许训练，不延续 v13pro main 只训练音频分支的限制。

阈值比较不可导，所以 detector 使用单独的 BCE 监督。恢复阶段中真实污染 mask 只用于监督与评估，不进入预测 mask 的前向计算。预测 mask=1 的位置使用恢复预测，mask=0 的位置保留 cue；检测漏报会留下噪声，误报可能损伤干净区域。图像接口内部使用 logits，概率转换会产生数值裁剪误差。

恢复阶段同时包含粗解码辅助监督（权重 0.5），降低预测 mask 较少时训练信号不足的问题。需要预训练 detector 的配置会拒绝缺失检测器权重；功能验证生成的 smoke checkpoint 默认不能用于正式恢复训练。

输入模式比例为 clean/clean 0.1、污染/污染 0.4、clean/污染 0.25、污染/clean 0.25。两种污染 family 均衡采样。两种 cue 都存在，恢复目标是当前实例的干净样本。标准数据划分保持 `official`，划分种子 20260915、验证比例 0.1；测试集仅用于最后评估。

完整设置见 [v15pro.yaml](../configs/v15pro.yaml)。所有默认轮数、阈值、噪声强度只是初始实验设置，不能视为已完成调参或已达到收敛。

## 一键计划与运行

在项目根目录执行。未添加 `--run` 时只打印任务计划，不创建输出目录：

```bash
python scripts/run_v15pro_suite.py
```

正式 GPU 运行示例（路径替换成机器上的实际绝对路径）：

```bash
python scripts/run_v15pro_suite.py --run --device cuda --data-root /data/mnist-fsdd --fsdd-root /data/fsdd/recordings --output outputs/v15pro
```

Windows 路径同样支持，例如 `--data-root "D:/datasets" --fsdd-root "D:/datasets/fsdd/recordings"`。默认三个随机种子为 1234、2345、3456。`--seeds`、`--detector-epochs`、`--epochs` 可调整预算；与既有计划不同的设置必须使用新的 `--output`。

默认恢复网络从头训练。可用 `--base-checkpoint` 指定完整路径，初始化匹配当前结构的 v13pro checkpoint。只允许缺失新增 detector 的参数，其他结构差异会报错；这意味着缺少多尺度音频局部模块的旧父模型不能冒充完整匹配的基础 checkpoint。基础 checkpoint 的 SHA256 会写入计划，且不会载入旧优化器状态。

每个种子的执行顺序：

```text
train_noise_detector.py → train.py → evaluate_v15pro.py
                                  ├─ predicted：主要结果
                                  ├─ oracle：真实 mask 诊断
                                  └─ all：全部位置预测诊断
```

`oracle` 是使用额外真实污染位置的对照，不能作为盲检测的结果；`all` 是同一恢复 checkpoint 的全域预测诊断，不能代替独立训练的全域去噪基线。三种评估固定测试条件，以便观察 mask 来源的影响。

`--skip-eval` 可暂时只训练。随后以相同参数去掉该选项重新执行，会核验并跳过已完成训练，补跑评估。评估报告应同时检查全域误差、真实污染区误差、真实干净区误差以及检测 precision/recall/F1/IoU。阈值只能用验证集选取，不能利用测试结果调整后重新汇报为未见测试集结果。

## 输出与继续运行

```text
outputs/v15pro/
  plan.json
  logs/
  seed_1234/
    config.yaml
    detector/best.pt
    detector/last.pt
    recovery/best.pt
    recovery/last.pt
    evaluation/predicted.json
    evaluation/oracle.json
    evaluation/all.json
  seed_2345/...
  seed_3456/...
```

套件记录解析后的配置与代码指纹。同一输出目录的不同计划会被拒绝，避免误覆盖。重新执行完全相同的命令时，未完成训练若存在 `last.pt` 会使用 `--resume`；已完成任务会核对完成标记及 best/last checkpoint 的 SHA256，产物被替换或损坏则报错。不要手动修改生成的配置后继续使用旧计划目录。

`--device cpu` 支持功能验证，本地 CPU 检查仅能证明数据流、梯度、保存/加载和指标路径工作正常。完整三种子实验需要在具有数据集与足够资源的机器上执行，建议 GPU；本次代码交付不包含这轮完整训练结果。
