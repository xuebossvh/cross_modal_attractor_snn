# v15 有效内容覆盖补充评估

本工具复用已训练的三个 v15 模型，增加有效内容覆盖测试、图像音频联合图和分区域指标。原训练代码、原随机位置测试及正式输出保留为基线。

## 协议

- `uniform` 逐样本调用原确定性损坏函数，保留原测试条件。
- `foreground` 保持同样的矩形面积和噪声强度，从覆盖有效内容的候选位置中随机选择。与 uniform 使用相同的全图噪声场，仅改变位置。
- 默认图像有效内容为干净目标值严格大于 0.1 的像素。音频为干净 log-mel 值严格大于 `max(0.02, 0.1 × 本样本最大值)` 的位置。它是本协议的有效频谱定义，不是波形能量或语音活动检测真值。
- 默认要求污染块覆盖至少 35% 的有效内容像素；污染块面积仍为整幅图的25%。两个比例的分母不同。
- 无可行窗口时选择有效内容覆盖最多的位置，并记录约束未满足；无有效内容时记录不可评价。不会无限重采样或伪造成功。
- 规则在补测前固定。前256个验证样本仅用于可行性检查，没有按测试恢复效果调整阈值。
- 干净目标和真实污染掩码仅用于生成损坏、区域评价和图示。预测模式模型只接收两路受损cue。

## 输入资产

在原 v15 环境中运行：

```bash
python tools/package_v15_eval_assets.py \
  --root outputs/v15pro \
  --output outputs/v15pro_eval_assets \
  --seeds 1234 2345 3456 --threads 1
```

资产保留精确的测试图像、去重音频特征及配对索引、标签、源身份、原配置和训练归一化统计。三个种子的输入只有在完整内容指纹一致时才共享。打包的模型保留原模型张量，省略优化器和调度器，只用于推理，不能用于续训。

## 推理

在具有相同原运行时代码的独立目录运行：

```bash
python -u tools/evaluate_v15_foreground.py \
  --assets outputs/v15pro_eval_assets \
  --output outputs/foreground_test_full \
  --seeds 1234 2345 3456 \
  --split test --device cuda --batch-size 32 --threads 2 \
  --examples 8 --cuda-memory-fraction 0.20
```

完整测试包含每种子10000个配对：uniform七种条件，foreground六种受污染条件。干净条件共用，不重复计入两种位置策略的总体比较。三个种子共39个条件、390000条条件记录；不是390000个独立音频样本。

每个条件完成后写入带指纹的 `complete.json`，完整重复运行会核验后跳过已完成条件。变更代码、设置或样本数应使用新输出目录。`--max-samples` 或验证集前256条资产只构成部分检查，报告会明确标注。

## 联合图与报告

绘图可在不安装Torch的独立结果环境执行：

```bash
python tools/report_v15_foreground.py \
  --root outputs/foreground_test_full \
  --original-results outputs/server_results \
  --audit-scripts /path/to/nature-figure/scripts
```

`--audit-scripts` 指向实际安装的科研绘图审计脚本目录，需含面板对齐、PDF文字和PDF碰撞审计模块；不是数据目录。缺少审计依赖会明确报错。

每个固定前8样本生成一张两行六列联合图，并保留PNG、PDF、SVG、原始数值NPZ、元数据、面板对齐和PDF审计记录。第一行图像、第二行音频，六列是目标、输入cue、真实污染掩码、预测掩码、实际修复掩码、恢复结果。图头标明split、样本ID、类别和Index预测。

`review.html` 是主阅读入口。`aggregate.json` 保留按相同family/cue条件计算的三个训练种子均值及样本标准差、配对位置策略差异和旧uniform结果复现差异。

## 指标口径

分别统计全图、污染区、未污染区、有效内容、污染区内有效内容、污染背景、干净有效内容和干净背景。先累计SSE和真实区域像素数，再计算MSE；空区域为null。掩码检测指标仍按真实污染support评价。

报告的准确率来自共享Index分类器，不是恢复后内容识别率。Foreground是对现有模型的困难条件测试，不是新模型训练或性能改进；不能将uniform含干净条件的七条件平均与foreground六条件平均直接比较。

长期状态记录见[共享实验计划](../../04_实验计划/v15_v16实验改进计划.html)。新结果只在本项目验证；迁移至v16及新增恢复训练属于后续计划。
