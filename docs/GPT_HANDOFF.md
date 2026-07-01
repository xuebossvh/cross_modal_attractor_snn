# Cross-Modal Attractor SNN — GPT Handoff Package

> 项目路径：`cross_modal_attractor_snn/`  
> 当前版本：`v9`
> 用途：MNIST + FSDD 跨模态联想记忆 SNN，残缺/干净 cue -> 分类 + 图像恢复 + 音频恢复

## 1. 任务与数据

| 项 | 内容 |
|---|---|
| 图像 | MNIST 28x28 |
| 音频 | FSDD wav -> log-mel 64x64 |
| 配对 | 同 label 的 MNIST 图像 + FSDD 语音 |
| 类别原型 | class medoid |
| 训练规模 | 60000 样本，batch=128，约 468 step/epoch，50 epoch |

## 2. 当前 v9 结构

- Key/Index/Value 主干保持不变。
- Index 负责稳定类别 basin，并通过 `A -> V` 激活图像/音频 Value。
- v9 decoder 输入改为拼接条件：
  - `ImageDecoder([V_img_from_A, image cue detail])`
  - `AudioDecoder([V_aud_from_A, audio cue detail])`
- `detail_conditioning.detach: true`，恢复损失默认不通过 detail path 反向拉动 Encoder/Key/Index。
- 缺失 cue 的 detail 用零向量填充，避免跨模态偷看。

## 3. 训练命令

```bash
cd /root/projects/cross_modal_attractor_snn_v9
source ~/snn-env/bin/activate
python scripts/mkdir_outputs.py --config configs/v9.yaml
nohup python -u scripts/train.py --config configs/v9.yaml > outputs/outputs_v9/logs/train_v9_50ep.log 2>&1 &
tail -f outputs/outputs_v9/logs/train_v9_50ep.log
```

## 4. 输出位置

- checkpoint: `outputs/checkpoints/cross_modal_snn_v9.pt`
- 日志: `outputs/outputs_v9/logs/`
- 图像: `outputs/outputs_v9/figures/`
- 表格: `outputs/outputs_v9/tables/`

## 5. 源码文件清单

| 文件 | 说明 |
|---|---|
| `configs/v9.yaml` | 当前版本唯一配置 |
| `models/network.py` | Key/Index/Value 主干和 detail-conditioned decoder 接线 |
| `models/decoders.py` | 分类头、图像 decoder、音频 decoder |
| `scripts/train.py` | 训练 |
| `scripts/evaluate.py` | 测试集评估 |
| `scripts/demo_inference.py` | 可视化 |
| `common.py` `paths.py` `data/*.py` | 公共逻辑与数据管线 |
