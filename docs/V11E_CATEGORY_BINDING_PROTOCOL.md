# v11e MNIST/FSDD 类别级绑定协议

## 数据与配对

- 图像：MNIST `[1, 28, 28]`。
- 音频：FSDD log-mel `[64, 64]`。
- 同类 many-to-many：MNIST digit `c` 只从 FSDD digit `c` 的池中抽取音频。
- 不固定一一配对，不返回 `pair_id`，重复组合不计为新增独立音频样本。

## Target 规则

| cue | image target | audio target |
|---|---|---|
| image-only | current clean sample | train class medoid |
| audio-only | train class medoid | current clean sample |
| image+audio | current clean sample | current clean sample |

class medoid 仅从训练 split 构建，test/evaluate/demo 复用同一原型。

## 模型边界

- 保留 Index attractor、双 Value、同模态 cue-detail gated concat 与 Cross-Key。
- `detail_conditioning.detach_value_for_recon=true`，恢复 loss 不经 `V_from_A`
  反向改写 Index。
- Cross-Detail 和 exact-pair alignment 关闭，因为 MNIST/FSDD 不提供真实实例对应。
- `v11e_control` 仅关闭 Cross-Key，作为同数据、同预算对照。

## 评估

主报告 ACC、image/audio full MSE、masked MSE、PSNR/SSIM、五类 corruption
breakdown 与 Cross-Key correct/zero/wrong/same sweep。不报告 exact-pair Recall@1。

GRID manifest 实验保留在历史提交中，不随当前 v11e 分支继续携带版本专属准备脚本。
