# 用户需求记录
> 创建时间：2026-07-06 18:43 | 来源：Codex 对话中的用户要求

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
10. 版本分支中的 `configs/` 默认只保留当前版本及其必要的 control/ablation
    配置。旧版本 YAML 应留在旧版本分支，不随新版本重复上传；若当前配置依赖公共
    配置，应使用明确命名的稳定 base，并在提交说明中解释依赖，不得通过
    `extends: v11c.yaml` 等方式迫使新分支携带整套旧版本配置。
11. `_data/`、`outputs/`、checkpoint、日志、压缩包、缓存、临时备份和本地研究资料
    默认不得进入代码仓库；只有用户明确要求上传的 checkpoint 才进入独立 checkpoint
    仓库。

## 文档偏好

- 语言：中文正文，英文技术标识保留原样。
- `implementation.md`：必须反映当前仓库的真实结构，不强行改写成 ResearchPilot 默认的 `code/src` 布局。
- `dev_log.md`：末尾固定保留 `运行说明` 章节；凡命令、参数、输出文件或路径发生变化，都要同步更新该章节。
- 文档中优先写清楚 tensor shape、文件职责、入口命令、输出路径和验证方式。

## 已知上下文

- 项目主题：基于 MNIST + FSDD 的跨模态 attractor SNN 联想记忆。
- 当前配置族：`configs/v11f.yaml` 及同版本 control/no_causal；v11e 配置留在旧分支。
- 当前代码结构：根目录下的 `data/`、`models/`、`scripts/`、`configs/`、`outputs/`。
- 历史输出显示项目已经迭代到 v9/v9c；创建本文档时尚未检测到 `outputs/outputs_v10a/`。

## 2026-09-08 v11e 类别级绑定修订

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

## 2026-09-09 v11f 修改授权

- 用户最终指定分支名 `v11f`，实现并推送代码仓库，不创建 `v11`。
- 保留 MNIST/FSDD 类别绑定、Value + own cue detail 解码、simultaneous、
  `detach_value_for_recon=true` 和 `batch_size=128`。
- Cross-Key 必须以改善部分残缺及全缺失恢复为目标；不能把忽略该通路当作成功。
- 使用缺失区域特征调制、冻结父模型、稳定正向因果目标、恢复内容类别评估四项修改。
- 冻结父模型控制分类代价；normal/zero/wrong/same-class 的对照共享 cue 和 mask。
- 统计收益需要真实实验验证，代码结构和非零 gate 本身不等于获得收益。
