# Cross-Modal Attractor SNN — GPT Handoff Package

> 项目路径：`cross_modal_attractor_snn/`  
> 用途：MNIST + FSDD 跨模态联想记忆 SNN，残缺/干净 cue → 分类 + 图像恢复 + 音频恢复

## 项目目录（重组后）

```
configs/          超参
data/ models/     库代码
common.py paths.py
scripts/          train.py evaluate.py demo_inference.py smoke_test.py
outputs/          checkpoints/ logs/ figures/ tables/  （运行产物）
docs/             本文档
_data/            MNIST + FSDD
```

---

## 1. 任务与数据

| 项 | 内容 |
|---|---|
| 图像 | MNIST 28×28 |
| 音频 | FSDD 真实 wav → log-mel **32×32**（`torchaudio` 必需） |
| 配对 | 同 label 的 MNIST 图像 + FSDD 语音 |
| 类别原型 | class **medoid** |
| 训练规模 | 60000 样本，batch=128，468 step/epoch，**75 epoch** |

---

## 4. 最近一次训练

| 项 | 值 |
|---|---|
| 命令 | `python -u scripts/train.py --config configs/v4.yaml` |
| 日志 | `outputs/logs/train_75ep.log` |
| checkpoint | `outputs/checkpoints/cross_modal_snn.pt`（epoch=74） |
| loss 曲线 | `outputs/tables/loss_curve_75ep.csv` |

---

## 6. Demo 输出

- 图：`outputs/figures/demo_*.png`
- 表：`outputs/tables/demo_eval_table.txt`

---

## 7. 源码文件清单

| 文件 | 说明 |
|---|---|
| `configs/v4.yaml` | 超参 |
| `scripts/train.py` | 训练 |
| `scripts/evaluate.py` | 测试集评估 |
| `scripts/demo_inference.py` | 可视化 |
| `models/*.py` `data/*.py` `common.py` | 核心库 |

---

## 9. Checkpoint 配置摘要

```yaml
train:
  ckpt_path: outputs/checkpoints/cross_modal_snn.pt
```
