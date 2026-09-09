# v11f：部分残缺的 Cross-Key 正向贡献

## 诊断与目标

v11e 的全局 Value 调制对不同残缺 family 收益不一致；联合优化还伴随分类变化。
这不证明因果原因已确定。v11f 固定同一个已训练 control 父模型，把可训练变化限制
在 decoder 内的 Cross-Key 特征调制，以检验局部条件是否比全局条件更适合补全。

研究目标是按 family 统计 `E_normal < E_zero` 且 `E_normal < E_wrong`，特别关注
双模态部分残缺。不能用 gate 非零或 wrong 变差代替 normal 相对固定基线的改善；
不要求每个含糊、无信息或本来正确的样本都满足严格不等式。

## Method

- 不变：MNIST 28x28、FSDD 64x64、同类 many-to-many、训练 medoid、目标可辨识性、
  Value + gated own detail、双 Key 同时输入、重建到 Value 的 stop-gradient。
- 冻结父模型：v11e_control，SHA256
  `5a792f10e57a95947c8e51bd915b09897baee01475103010826f25275b71501b`。
  不从 v11e main 初始化，避免把已有全局 Cross-Key 效果混入新方法。
- 新增方向独立 adapter：图像 32 通道特征、音频 16 通道特征，Key 128 维投影为
  16 个条件通道，隐藏通道 32；目标 cue/mask 提供局部上下文，对侧 mask 均值提供
  可观测损坏比例。该比例不是自动估计的可信度，也不代表语义正确概率。
- adapter 使用 Key 条件乘性特征调制，gate 逐位置/通道调整作用量，输出零初始化。
  mask 为 1 表示被损坏；全缺失用全 1；无损用全 0。源模态缺失则不调用该方向。
  对输出再作 mask 选择，避免卷积和 refiner 的邻域扩散改动原本可见位置。
- 只训练 adapter；基线子模块 eval，禁止恢复 loss 更新 Index 或基础 own-detail。
  分类稳定针对同一输入严格成立，不代表恢复内容类别自动正确。

## Objective

保留绝对图像/音频恢复项；增加正确 Key 相对冻结 zero、detached wrong 的 margin。
按方向使用 `scale=max(mean(E_zero), floor)`；图像 floor=0.01、音频 floor=0.005
作为初始训练超参，后续只能在训练/验证集调节。margin 为 0.05*scale，loss weight=0.5。
没有不同类别可换的 batch 仍训练 normal 对 zero，不丢弃可用监督。
same-class 是语义等价对照，不强制 correct-instance 优于同类其它实例。
clean-only batch 无可训练路径时跳过 optimizer step，不伪造梯度。

## Experiment Design

| 配置 | 来源 | 额外训练 | 用途 |
|---|---|---:|---|
| v11f | v11e_control | 30 epochs | 局部 Cross-Key + 因果项 |
| v11f_no_causal | 同一父模型、同一 seed | 30 epochs | 去因果项，保留局部结构 |
| v11f_control | 同一父模型 | 0 | 冻结零通路基线，仅评估 |

batch=128，均衡轮转五种腐蚀 family，8 种 cue 模式保持原分布。adapter 训练固定
severity=0.4，不重复基础模型的低强度暖启动。控制配置不训练，不冒充等预算实验。
除正式 control 比较，还对同一训练模型做 normal/zero/wrong/same-class 干预。

## Evaluation

- fixed_mask 与 legacy_random 保留；主比较必须使用一致 seed、样本、mask。
- v11f random 使用独立 `eval.random_seed=4321` 对每 batch 随机抽取五 family 中
  的一种并生成 mask，可用 `evaluate.py --random_seed` 改变抽样；不再误把配置中的
  固定 `occlusion/time_mask` 称为随机 family。CSV 标记 `random_mix`。
- 对五种 family 分别 sweep：有效数、normal/zero/wrong/same-class masked MSE、
  对 zero/wrong 的误差差值及样本改善比例；全缺失使用全区域误差。
- Index ACC 继续报告；新增恢复图像/音频经冻结原模型单模态再分类的 ACC。
  这是内部类别一致性，不是独立识别准确率，不声称能验证具体说话人或笔迹实例。
- 固定基线校验：所有 parent state（包括 buffers）摘要不变；zero 输出与父模型一致；
  normal/zero/wrong 的 Index logits 一致；未损坏区域输出不变。
- 成功标准：normal 相对固定 zero/control 真正改善，不能仅通过 wrong 变差获得较大 gap。
  本轮只实现并做 smoke/regression，不预先宣布训练收敛或结果提升。

## 运行与验证约束

父权重摘要、关键前向配置和训练集音频归一化统计必须匹配；不允许改换归一化后
仍宣称使用原冻结基线。套件的每个阶段独立日志，子进程失败即停止，最终输出
`ALL STAGES COMPLETED`。`--eval_only` 不训练；`--resume` 需要对应 checkpoint。
resume 恢复权重、优化器、调度器和 epoch；不承诺跨中断精确重放 DataLoader 顺序。

验证包含合成数据的 8 种 cue 前后向、父模型严格一致性、真实父 checkpoint 加载、
mask 不越界、错误 provenance 拒绝、same-class-only 因果项、fixed/random paired
指标以及 CLI 的 train/resume/eval/demo 流程。CPU 冒烟不替代 3080 完整 GPU 训练。
