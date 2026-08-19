# DARG：非对称旋转抓取模型

DARG 全称为 **DETRIS-based Asymmetric Rotated Grasping**。它保留 DETRIS/DROG
的 DINOv2 视觉主干、CLIP 文本编码器和跨模态解码器，但不再直接回归一个左右
对称的抓取宽度，也不再单独学习与框几何可能不一致的中心 offset。

## 模型输出

语言条件分割分支先预测目标掩码，soft mask 用来门控抓取几何特征。抓取头预测：

- CenterNet 风格的抓取质量热图；
- FCOS centerness；
- 局部旋转坐标系中的 `d_left, d_right, d_top, d_bottom`；
- `sin(2θ), cos(2θ)`。

每个真值旋转框的整个内部区域都有几何监督。对于框内任意候选点，四边距离可以
恢复同一个完整旋转框；重叠框区域采用 centerness 更高的真值。

令长边方向为 `uθ=(cosθ,sinθ)`，短边方向为
`vθ=(-sinθ,cosθ)`，则：

```text
width      = d_left + d_right
short_side = d_top + d_bottom
Δlong      = (d_right - d_left) / 2
Δshort     = (d_bottom - d_top) / 2
c_new      = c_base + Δlong*uθ + Δshort*vθ
```

因此，左右长度不必相等；预测点即使不落在真实中心，仍可由同一组四边距离精确
修正中心。推理时 DARG 把上述结果转换为 ToolRGS 已有的 width、short-side 和
offset 输出，所以 GUI、机器人部署和多 IoU/多角度阈值评估器无需另写一套。

## 损失

总损失由以下部分组成：

- 语言目标分割 BCE；
- 抓取质量 BCE 和 centerness BCE；
- 框内加权的四边距离 Smooth-L1；
- 双角正弦/余弦 Smooth-L1 及单位圆约束；
- GWD 与对称 KLD 旋转框分布损失。

默认 `geometry_loss: hybrid` 同时启用 GWD 和 KLD。GWD 对中心及尺度变化较平滑，
KLD 对方向和长宽比差异更敏感；二者共同约束四边距离与角度形成一个一致的旋转框。

## 训练

单卡：

```bash
python train.py --config config/grasp_tools/darg.yaml
```

双卡：

```bash
torchrun --nproc_per_node=2 train.py --config config/grasp_tools/darg.yaml
```

配置默认沿用 DROG-OFF v2 original-scale 的 448 输入、每卡 batch size 8、36
epochs 和 300 px 抓取尺寸归一化。额外的关键消融项是
`mask_guidance_strength`、`geometry_loss`、`ltrb_loss_weight`、
`gwd_loss_weight` 和 `kld_loss_weight`。
