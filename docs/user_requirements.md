# 用户需求记录
> 创建时间：2026-07-06 18:43 | 来源：Codex 对话中的用户要求

当前活动版本：`v12a`。本文的规则适用于当前版本及后续版本；底部带日期的版本段落是
需求变更历史，不是当前实现入口。当前实现入口见 `docs/implementation.md`。

## 项目阶段

本项目按 ResearchPilot **F 阶段：代码迭代** 处理。

- 当前项目不是从 A 阶段重新开始，而是在已有代码基础上进行 D-F 迭代。
- 首版 `docs/implementation.md` 以当前真实代码为依据反向生成。
- 后续凡涉及模型结构、实验设置、训练逻辑、评估逻辑或运行命令的代码修改，都必须先同步设计文档，再修改代码。

## 硬性规则

1. 每次改代码之前，必须先更新相关设计文档。
2. 实现层面的修改，必须先更新 `docs/implementation.md`。
3. 模型结构或实验设计层面的修改，若 `docs/idea_report.md` 已存在，必须先更新其中对应的 Method 或 Experiment Design；若该文件尚不存在，则先创建或补充相应设计说明。
4. 每次改完代码之后，必须在 `docs/dev_log.md` 追加新的日志条目。
5. `docs/dev_log.md` 只追加，不删除、不覆盖已有记录。
6. 每次代码修改都视为一次 D-F 迭代：先诊断或确认改动范围，再更新文档，再改代码，再验证，最后追加日志。
7. 每个消融实验完成评估后，必须立即在 `docs/dev_log.md` 追加对应结论；不得只记录配置、命令或原始指标。多个消融可以归入同一日志条目，但每个配置必须有独立结论，至少写清：消融对象、对照实验、训练轮数/seed、评估协议、关键指标变化、因果解释、异常与比较限制。
8. 未写入 `docs/dev_log.md` 的消融实验视为尚未完成归档，不得直接进入下一版本设计或论文结论。若主实验与消融的训练预算、seed 或评估 mask 不一致，必须在结论中显式标注，禁止作强因果表述。
9. 每次 commit/push 前必须检查 `git status --short` 与
   `git diff --cached --name-status`，只暂存当前版本运行、验证和说明必需的文件；
   禁止直接使用 `git add .` 把未跟踪文件、临时文件或无关历史产物一并上传。
10. 版本分支中的 `configs/` 默认只保留当前活动版本及其必要的 control/ablation
    配置。旧版本 YAML 应留在旧版本分支，不随新版本重复上传；若当前配置依赖公共
    配置，应使用明确命名的稳定 base，并在提交说明中解释依赖，不得通过
    `extends: v11c.yaml` 等方式迫使新分支携带整套旧版本配置。
11. `_data/`、`outputs/`、checkpoint、日志、压缩包、缓存、临时备份和本地研究资料
    默认不得进入代码仓库；只有用户明确要求上传的 checkpoint 才进入独立 checkpoint
    仓库。
12. 每次评估必须逐实验报告各项指标，覆盖主实验、control 和所有可用消融；
    不得只给定性结论、少数提升百分比或仅主实验的数据。评估结果必须直接归档
    到 `docs/dev_log.md` 的对应版本条目中，不得另建或依赖
    `Vxx_EVALUATION.md`、`Vxx_METRICS_SUMMARY.md` 或版本专用协议 Markdown 这类
    单独文档。具体交付和验收要求见下方“评估报告强制规范”，适用于当前及所有
    后续版本。原始 CSV、日志和图片可以继续保留在 `outputs/`，它们是运行证据，
    不是结果说明文档。
13. 每个版本必须形成完整文档闭环：最初方案、研究问题、预期验收条件先写入
    `docs/idea_report.md`；实际结构、配置、tensor shape、命令和验证方法写入
    `docs/implementation.md`；训练/评估完成后，逐实验完整指标、独立结论、异常、
    限制和证据路径追加到 `docs/dev_log.md`。即使不创建版本专用 Markdown，也不得
    省略方案或评估内容；清理重复文件时只能合并入口，不能删除记录。

## 评估报告强制规范

> 2026-09-10 用户明确要求：之后的评估都要告知各个实验的各个指标，不得遗漏。

### 实验与协议清单

- 先列出本次主实验、control、全部消融的配置/权重来源、实际训练轮数、seed、
  batch size、severity、冻结/可训练范围及评估产物路径。未评估的实验单列状态，
  不当作数值为 0，也不默认为已完成。
- fixed、random、独立 family breakdown 和 demo 小样本分别报告；逐 cue 模式
  列数值，逐 family 展示细分。不得把不同 seed/mask 的表混算为同一配对比较。
- 区分训练轮数与从父模型额外训练轮数；固定父模型 control 不能称为等预算重训。
- 区分 present-modality sample target 与 missing-modality category target，
  不跨不同目标直接比较误差高低；注明类别原型只能来自训练集。

### 指标清单与呈现

- 用户回复必须包含逐实验的主要指标对照表，而不只是文字结论或完整表链接。
  完整逐项数值也必须直接追加到本次 `docs/dev_log.md` 评估归档条目中；不得用
  单独的结果 Markdown 文件替代。表格过长时可在同一日志条目中分节，不能只留
  一个外部文件路径。
- 先扫描原始产物实际包含的指标字段，再汇总所有已计算指标。不能仅靠固定清单
  或沿用上一版的字段数；新增指标也必须进入完整表。
- 主要指标至少覆盖可用的 Index ACC、图像/音频 MSE 与 SSIM、图像 PSNR、
  缺失区/可见区 MSE 与 L1，以及恢复内容类别一致性。粗恢复和多样性诊断若存在，
  也必须列出。
- Cross-Key 或其它干预实验必须报告 normal/zero/wrong/same-class 的绝对误差、
  配对增益/损害、win_zero/win_wrong/win_both、gate、ratio 及各自有效样本数。
  不允许只给 gate 或 wrong 退化量，就宣布跨模态恢复成功。
- 音频诊断的 rec/tgt mean、std、max、top-k 能量召回等，只要已有记录就应纳入；
  训练 loss/LR/轮次与 demo 的小样本统计另列，不混入全测试集指标。
- Demo 默认 `n=10` 仅用于可视化，必须使用固定 seed 从完整测试集随机抽样并记录实际
  indices；禁止直接取测试集第一个 batch。fixed/random demo 应沿用同一抽样索引，只有
  corruption protocol 不同；demo 指标不能混入全量测试集评估。
- 数值缺失、不适用、未计算分别标明原因，可用 N/A 表示，但不得填 0、猜值或
  临时杜撰指标。主表、附表都要注明单位、优劣方向和必要精度。
- 同时给绝对值与有意义的对照变化，区分百分点和相对百分比；基线为 0 时不算
  相对变化。列明 family/样本汇总权重，均值不得重复计入 normal 与 sweep。

### 解释边界与完成检查

- Index 分类、恢复内容内部再分类、外部独立识别三个概念分开；如果识别器参与
  训练监督，不把其结果作为独立外部验证。
- 列明有效 n 与重复曝光，不将复用音频或多 mask 计数当作独立样本数。缺少
  逐样本数据或多 seed 结果时，不编造置信区间或统计显著性。
- 说明是审阅已有结果、重算表格，还是实际重跑模型。发现数值评估和可视化协议
  不一致时明确报告，不把缺项或错误口径藏在总体均值里。
- 每个消融仍须在 `docs/dev_log.md` 追加独立结论，并把本次完整指标直接写入
  同一日志条目。检查“全部实验 × 全部协议 × 全部已计算指标”后才宣布评估
  完成；有缺项时清楚说明已完成范围及未完成部分。不能用未归档或不可比结果
  直接下新版结论。

### 统一评估版式（2026-09-12）

- v11c 至 v11g 及后续版本均采用用户截图版式：先说明实验含义、父权重、实际训练
  预算和协议，再展示“分类与恢复”主表。主表固定列为：
  `输入模式 | 实验 | Index ACC | 图像 MSE | 图像 SSIM | 音频 MSE | 音频 SSIM`。
- 表格先按输入模式分组，再按 `control → 当前版本主实验 → 各消融` 排行；不能将
  多个实验的值用斜线挤在一格，也不能用一串 `metric=value` 代替分列指标。
  只有两组实验的版本只列两组，不凭空补出 no_causal。
- fixed 五个 family 的等权宏平均与 random 分表。区域误差、内容分类、Cross-Key/
  Cross-Detail、逐 family、训练统计及 demo 分节；长附表可在同一文档内折叠，
  但完整数值必须在文档里，不能只留下 CSV 路径。
- 统一版式不等于统一实验口径。必须标明类别/样本 target、缺失协议、日志舍入精度
  和有效 n；未保存的指标不补算成 0，不把一个 family 当五类均值。
- 结果继续归入 `dev_log.md`，通过版本锚点提供唯一的当前汇总入口。历史日志保留，
  发现旧条目错误时在新汇总中明确勘误；不新建版本专用评估文件，不改原始产物。
- 所有核心文档的当前入口必须放在前部并按版本顺序组织；旧方案、旧命令和旧结果
  只能放在明确标注的历史区，不在当前目录中重复占位。当前版本只出现一个主动入口，
  历史内容若因审计需要保留，应折叠或改为普通说明，不得与当前入口并列造成歧义。

## 文档偏好

- 语言：中文正文，英文技术标识保留原样。
- `implementation.md`：必须反映当前仓库的真实结构，不强行改写成 ResearchPilot 默认的 `code/src` 布局。
- `dev_log.md`：末尾固定保留 `运行说明` 章节；凡命令、参数、输出文件或路径发生变化，都要同步更新该章节。
- 文档中优先写清楚 tensor shape、文件职责、入口命令、输出路径和验证方式。

## 已知上下文

- 项目主题：基于 MNIST + FSDD 的跨模态 attractor SNN 联想记忆。
- 当前配置族：`configs/v12a.yaml` 及同版本 control/no_causal；旧配置留在旧分支。
- 当前代码结构：根目录下的 `data/`、`models/`、`scripts/`、`configs/`、`outputs/`。
- 历史输出显示项目已经迭代到 v9/v9c；创建本文档时尚未检测到 `outputs/outputs_v10a/`。

## 需求变更历史

以下条目按发生时间排列，仅记录规则如何形成；当前有效规则以上方章节为准。

### 2026-09-08 v11e 类别级绑定修订

- v11e 继续以 MNIST 图像和 FSDD `64x64` log-mel 为实验数据，不再把 GRID
  mouth-motion dynamic image 作为当前 v11e 输入。
- MNIST/FSDD 采用同数字类别内的 many-to-many 随机组合，不建立人工固定
  instance pair，也不把配对曝光数表述为新增独立音频样本。
- 单模态 cue 的恢复目标必须遵循可辨识性：image-only 为
  `sample/category`，audio-only 为 `category/sample`；仅双模态 cue 为
  `sample/sample`。
- 缺失模态的 category target 使用仅由训练集构建的 class medoid；禁止使用测试集
  构建原型。
- 不再对 MNIST/FSDD 使用 exact-pair InfoNCE、pair Recall@1 或要求
  same-class correct pair 优于 same-class wrong pair 的因果目标。

### 2026-09-09 v11f 修改授权

- 用户最终指定分支名 `v11f`，实现并推送代码仓库，不创建 `v11`。
- 保留 MNIST/FSDD 类别绑定、Value + own cue detail 解码、simultaneous、
  `detach_value_for_recon=true` 和 `batch_size=128`。
- Cross-Key 必须以改善部分残缺及全缺失恢复为目标；不能把忽略该通路当作成功。
- 使用缺失区域特征调制、冻结父模型、稳定正向因果目标、恢复内容类别评估四项修改。
- 冻结父模型控制分类代价；normal/zero/wrong/same-class 的对照共享 cue 和 mask。
- 统计收益需要真实实验验证，代码结构和非零 gate 本身不等于获得收益。

### 2026-09-11 v11g 版本迁移要求

v11g 复现 v11f 的已确定方案，不新增未经设计和记录的结构。版本入口、配置、
脚本和文档中的当前版本标识必须统一为 v11g；v11f 的 YAML、suite 和 smoke 脚本
留在 v11f 分支，不随 v11g 重复上传。v11g 的方案先写入 `idea_report.md`，真实
实现写入 `implementation.md`，训练和评估结果只能追加到 `dev_log.md`。

### 2026-09-13 v12a 实现要求

v12a 在 v11g 基础上优先修复音频恢复：保留类别级 MNIST/FSDD 绑定、Value 加本模态
detail、Masked Cross-Key 和 `batch_size=128`；音频可见区域必须原样回填，Audio Encoder
的局部卷积率经零初始化 projector 接入 Audio Decoder。恢复训练只允许显式列出的音频
decoder、局部 cue projector 和 Cross-Key adapter 更新，Encoder、Key、Index、Value、
Classifier 保持冻结。当前分支只上传 v12a 配置、脚本、实现和核心文档；v11g 配置与
版本入口留在历史分支。v12a 训练和评估完成后，完整逐实验指标仍只能追加到
`docs/dev_log.md`，不得创建版本专用评估 Markdown。
