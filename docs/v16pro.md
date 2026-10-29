# v16pro：共享 Index 的补全、去噪与机制实验

本版本从 v15pro 分支建立。所有父训练、恢复适配与测试关闭解码端 Cross-Key、Cross-Detail；保留共享 Index。正式恢复配置只有 `main`、`no_value`、`no_recurrence`，没有单尺度消融或 CNN 恢复对比。

## 输入与目标

每次前向只有图像、音频两路 cue。C 为完整干净、P 为已知位置的部分缺失、N 为位置未知的局部污染、A 为整模态缺席。

| 输入组 | 图像/音频状态 | 训练抽样概率 |
|---|---|---|
| 旧八种 | P/A、A/P、P/P、C/P、P/C、C/A、A/C、C/C | 每种 0.0625，共 50% |
| 新五种 | N/A、A/N、N/N、C/N、N/C | 每种 0.1，共 50% |

P 与 N 跨模态混合、同模态同时缺失与污染，不属于本次主协议。每个 batch 采样一种布局；位置与噪声按样本生成，13 种模式不意味着训练轮数乘以 13。

输入存在时，目标是当前干净实例；输入完全缺席时，目标是对应类别的训练集 medoid。训练标签可以选择监督目标；推理前向不能接收真实类别或真实污染位置。

已知缺失掩码与真实污染标签分别存储。所有存在的 cue 都经过噪声检测器，包括干净 cue；最终修复区域为已知缺口与检测器在其余观测位置预测的污染区域的并集。检测损失排除已知缺口；缺席模态不作为干净样本送入检测器。正式结果使用预测污染掩码。

干净 cue 未被误检时最终输出会复制输入。因此同时保存回填前预测指标，不能将干净最终 MSE 接近零解释为解码器重建能力。

## 训练计划

| 每个 seed 的作业 | 轮数 | 初始化与更新范围 |
|---|---:|---|
| detector | 10 | 两路噪声检测器同一作业；后续冻结共用 |
| parent | 100 | 从头训练有循环核心与基础恢复，局部注入使用 64×64 |
| parent_no_recurrence | 100 | 从头训练无循环反馈父模型，其余协议相同 |
| main | 30 | 共同 parent 最佳权重；适配两路解码器、图像细化与多尺度局部投影 |
| no_value | 30 | 同一 parent；Value 在门控与解码拼接之前统一清零 |
| no_recurrence | 30 | 对应无循环 parent；多尺度恢复 |
| recognizer | 30 | 两路独立内容识别器，仅用干净训练集 |

每 seed 330 个作业轮，3 seeds（1234/2345/3456）共 21 个训练作业、990 个作业轮。不同作业每轮成本不同。没有另训模板分类器：hard/soft medoid 使用同一模型 Index 预测的类别/概率，模板仅由训练集构造。

恢复适配冻结编码器、Key、Index、Value、分类器、全局细节投影和门控。main/no_value 在同一核心上检验适配期共享读出路径，二者 Index 分类输出应相同。无循环分支从父训练开始移除循环反馈，LIF 自身积分和时间窗口仍保留。三种恢复配置都使用 16/32/64 多尺度局部路径。

检测器、父模型、恢复模型与识别器都按验证集选择 best，不用测试集调阈值、挑 epoch 或挑样例。父训练与适配分别采用 0.0002/0.0001 初始学习率；检测器与识别器为 0.001。

恢复验证预先固定为 13 种输入模式等权、每种模式内 family 等权：C 采用回填前全局 MSE，P 采用真实缺失区 MSE，N 采用真实污染区 MSE，A 采用类别目标全局 MSE；图像/音频权重为 1/4。此分数只用于选权重，各任务的正式结果仍分别报告。

## 启动

先只打印计划（不写结果、不启动训练）：

```bash
python scripts/run_v16pro_suite.py --device cuda
```

训练并在成功后自动评估、画图、汇总三个种子：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nohup python -u scripts/run_v16pro_suite.py \
  --run --device cuda --output outputs/v16pro \
  > v16pro_train_eval.log 2>&1 < /dev/null &
```

```bash
tail -f v16pro_train_eval.log
```

同一命令再次执行会核验已完成作业并从未完成作业的 last checkpoint 恢复。代码、配置或依赖权重改变时需新的输出目录；不覆盖另一份实验。不要同时启动多个进程写入同一输出目录。

`--skip-eval` 仅训练；之后单独导出：

```bash
python -u tools/export_v16pro.py \
  --root outputs/v16pro --seeds 1234 2345 3456 --device cuda --examples 8
```

套件默认在 predicted 主结果之后，自动对主模型额外运行 oracle（真实污染位置）和 all（整幅预测）诊断；不为消融重复诊断或强度扫描。分别写入 `inference_oracle` / `inference_all` 及 `report_oracle` / `report_all`，不混入预测掩码主结果。使用 `--skip-mask-diagnostics` 可跳过这两个推理作业。手工补做可运行：

```bash
python tools/export_v16pro.py --root outputs/v16pro --device cuda --models main --skip-scans --mask-source oracle
python tools/export_v16pro.py --root outputs/v16pro --device cuda --models main --skip-scans --mask-source all
```

oracle 是位置已知的诊断条件，不保证性能最好；all 是同一权重的整幅预测诊断，不是独立训练的全域去噪基线。

`--examples` 只控制图片数量，不限制正式数值测试样本。只跑一个 seed 可以显式使用 `--seeds 1234`。CLI 的小轮数、受限批次和受限样本选项只用于流程检查，不可作为完整研究结果。

## 实验展示与结果

1. 类别读出与实例补全：Index ACC、真实缺失区/污染区误差、独立恢复内容识别率，回填前预测作为诊断。
2. 模态可见性与关联读出：按 C/P/N/A 和实例/类别目标分别记录，完全缺席模态按类别代表评价。
3. 类别模板参照：同一 Index 判断的 hard/soft medoid 与当前实例恢复比较，统一可见区处理。
4. 缺失类型与强度：所有恢复分支均评价 fixed 家族和 random 家族；主模型额外扫描缺失强度 0.2/0.4/0.6/0.8、局部污染面积 0.1/0.25/0.4、Gaussian 标准差 0.1/0.2/0.3。采用三个双模态布局的单因素扫描，缺失强度扫描固定 occlusion/time_mask 家族，避免穷举所有因素组合。
5. 机制验证：main/no_value/no_recurrence；去 Value 主要解读本模态存在的实例恢复，完全缺席时不把失去唯一共享输入后的失败单独当作机制优势。
6. 实际恢复样例：固定前 N 个测试样本，不按分类正确或恢复成功筛选；显示目标、cue、已知缺口、预测污染、恢复与误差，并记录样本身份。

不同任务、区域与目标不合并成一个含糊的总 MSE。污染区域和干净区域依据真实支持划分；空区域的指标为 null，不伪造为零。缺失强度在不同家族中控制长度或概率，不一定等于面积比例；同时记录真实区域像素数。

每个恢复模型的结果位于：

```text
outputs/v16pro/seed_1234/main/inference/
  figures/
  logs/
  tables/
```

其他 seed 和消融使用对应独立目录。跨 seed 的绝对结果、模板参照、强度扫描及消融汇总位于 `outputs/v16pro/report/`，同样分 figures/logs/tables。误差条表示训练 seed 间样本标准差；一个 seed 不伪造标准差。10,000 个图像/语音组合不等于 10,000 条独立测试录音。

逐样本结果为流式写入的 `per_sample.csv.gz`，可直接用 pandas `read_csv` 读取；条件与主任务汇总表为普通 CSV。默认主模型 87 个报告条件（含重复基准点的单独扫描标签），每个消融 48 个条件。验证使用固定的 18 个家族/布局条件，覆盖全部 13 种布局。

默认运行是正式数据完整流程。单元测试和小批次 smoke 输出仅用于验证软件，不能据此宣称模型性能改善或消融有效。
