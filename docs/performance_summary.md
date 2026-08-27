# ToolRGS consolidated performance summary

This document collects reproduced results from ToolRGS, CROG-GPU and CROG-NPU in one
traceable index. It follows the ToolRGS table layout while keeping results from
different repositories, accelerators and evaluation protocols in separate
tables. A dash means that the source did not record that field.

## Recording rules

- Metrics are percentages in the range `0` to `100`.
- Do not overwrite a row obtained with a different split, checkpoint, decoder
  or protocol; add a new row instead.
- ToolRGS and CROG-NPU rows are provenance records, not evidence that the two
  implementations are numerically interchangeable.
- VCoT uses top-1 `GraspSR`, so it is reported as `GraspSR` rather than being
  relabelled as OCID-VLG `J@1`.

## OCID-VLG

### ToolRGS records

Source: ToolRGS `558efb937ca5cf343e1aab2b75713caf59bd564e`.

| Model | Config | Split | Checkpoint | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| CROG | `config/ocid_vlg/crog.yaml` | test | - | 77.20 | 87.70 | - | Source summary does not record the checkpoint or protocol. |
| DROG-OFF | `config/ocid_vlg/drogoff.yaml` | test | - | 85.95 | 91.51 | - | Source summary does not record the checkpoint or protocol. |

### CROG-NPU reproduced results

Dataset: 17,749 test expressions. Protocol: `crog_legacy`. Hardware: Ascend
910B3. Full logs and decoder compatibility notes are in
[`experiments/ocid_vlg_20260810`](experiments/ocid_vlg_20260810/README.md).

| Model | Config | Split | Checkpoint | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| DROG-OFF | `config/OCID-VLG/drogoff.yaml` | test | `best_epoch_030_J1_91.24_J5_94.12.pth` | **89.30** | 92.94 | `91d3c51` | Sigmoid width decoder, matched to training. |
| DROG | `config/OCID-VLG/drog.yaml` | test | `best_epoch_036_J1_85.78_J5_93.57.pth` | 89.10 | **93.58** | `91d3c51` | Clamp width decoder, matched to training. |
| CROG | `config/OCID-VLG/crog_multiple_r50.yaml` | test | `best_epoch_017_J1_85.80_J5_91.33.pth` | 88.17 | 91.36 | `20c9a6c` | Clamp width decoder, matched re-evaluation. |
| LGD | `config/OCID-VLG/lgd.yaml` | test | `best_epoch_035_J1_84.28_J5_88.71.pth` | 84.94 | 87.27 | `b09d9bf` | Training-era code required; diffusion evaluation is stochastic. |
| GRConvNet-CLIP | `config/OCID-VLG/grconvnetclip.yaml` | test | `best_epoch_036_J1_86.27_J5_90.94.pth` | 88.14 | 91.21 | `3fac083` | No segmentation head; its reported segmentation IoU is not meaningful. |
| GGCNN-CLIP | `config/OCID-VLG/ggcnnclip.yaml` | test | `best_epoch_036_J1_20.26_J5_24.32.pth` | 15.32 | 17.13 | `b09d9bf` | Training-era code required; no segmentation head. |
| ETRG | `config/OCID-VLG/etrg.yaml` | test | `best_epoch_034_J1_73.55_J5_78.77.pth` | 73.73 | 76.85 | `3fac083` | RGB evaluation profile. |

The corresponding segmentation and precision values are retained in
[`results.json`](experiments/ocid_vlg_20260810/results.json), rather than
expanding the ToolRGS summary table with a second metric schema.

## VCoT-GraspSet

### Evaluation protocol and provenance

All values in this section are top-1 grasp success rates (`GraspSR`, percent).
A prediction succeeds when its rotated rectangle has IoU greater than `0.25`
with at least one ground-truth grasp and its 180-degree-periodic orientation
error is at most `30` degrees. ToolRGS uses `IoU >= 0.25`; exact-boundary hits
are not expected to change the rounded scores. `Avg.` is the harmonic mean of
Seen and Unseen rather than their arithmetic mean:

```text
Avg. = 2 * Seen * Unseen / (Seen + Unseen)
```

The published rows below come from Tables III and IV of the
[VCoT-Grasp paper](https://arxiv.org/html/2510.05827v1#S4.SS1). The DROG-OFF
rows are reproduced by ToolRGS/CROG-NPU at evaluation commit `3d9afef` on the
same public Seen (3,000 samples) and Unseen (1,487 samples) splits. Detailed
logs, decoder ablations and the paper's real-robot results are retained in
[`experiments/vcot_20260810`](experiments/vcot_20260810/README.md).

### Unified published and reproduced comparison

The table is ordered by Seen/Unseen harmonic mean. Paper averages are retained
as published; the two DROG-OFF averages are computed from the displayed rounded
split scores using the same formula.

| Rank | Origin | Method | Variant / profile | Seen | Unseen | Avg. (harmonic mean) |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | VCoT-Grasp paper | VCoT-Grasp | LM head, pretrained position tokens | **83.60** | **58.98** | **69.16** |
| 2 | VCoT-Grasp paper | VCoT-Grasp | LM head, new position tokens | 82.89 | 58.80 | 68.80 |
| 3 | ToolRGS reproduced | **DROG-OFF** | Calibrated decoding | 80.77 | 58.71 | 68.00 |
| 4 | ToolRGS reproduced | **DROG-OFF** | Matched baseline decoding | 80.97 | 57.30 | 67.11 |
| 5 | VCoT-Grasp paper | VCoT-Grasp | MLP head | 73.37 | 52.25 | 61.03 |
| 6 | VCoT-Grasp paper | VCoT without visual CoT | MLP head | 67.60 | 49.36 | 57.06 |
| 7 | VCoT-Grasp paper | RT-Grasp | PaliGemma baseline | 58.93 | 44.79 | 50.80 |
| 8 | VCoT-Grasp paper | VCoT-Grasp | Diffusion head | 57.50 | 41.29 | 48.07 |
| 9 | VCoT-Grasp paper | GR-ConvNet + CLIP | Language-conditioned baseline | 70.80 | 33.29 | 45.29 |
| 10 | VCoT-Grasp paper | GG-CNN + CLIP | Language-conditioned baseline | 56.33 | 17.89 | 27.16 |
| 11 | VCoT-Grasp paper | CLIP-Fusion | Det-Seg proposal backbone | 52.40 | 13.51 | 21.48 |
| 12 | VCoT-Grasp paper | LGD | Published baseline | 38.67 | 13.42 | 19.93 |

The calibrated DROG-OFF profile is `6.97` points above the paper's VCoT MLP
head in harmonic mean and exceeds all published non-LM-head baselines. It is
`1.16` points below the best published LM-head result (`68.00` versus `69.16`).
This is a score-level comparison under the same public split and success
criterion, not an equal-training-budget claim: the paper trains its models for
three epochs at 224-pixel resolution, whereas this DROG-OFF checkpoint was
selected at epoch 20 and uses 448-pixel inputs.

### DROG-OFF reproduction details

| Profile | Config | Checkpoint | Seen Seg. IoU | Seen GraspSR | Unseen Seg. IoU | Unseen GraspSR | Avg. | Eval commit |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Matched baseline | `config/vcot/drogoff.yaml` | epoch-20 best | 88.98 | 80.97 | 60.96 | 57.30 | 67.11 | `3d9afef` |
| Calibrated decoding | `config/vcot/drogoff.yaml` | epoch-20 best | 88.98 | 80.77 | 60.96 | **58.71** | **68.00** | `3d9afef` |

Both profiles use sigmoid grasp-size decoding with factor `300`, offsets
enabled and segmentation-centre filtering disabled. The calibrated profile
only disables offset-geometry resampling and inverse image-scale restoration;
it is recorded separately because this is checkpoint-specific inference
calibration, not a retrained model.

## Grasp-Tools V3

### Multi-threshold mean Success Rate (mSR)

Grasp-Tools V3 reports one top-1 grasp success surface instead of relying on a
single permissive threshold. For validation sample `i`, let `p_i` be the
highest-quality predicted rotated grasp rectangle and let `G_i` be all valid
ground-truth grasps for the referred tool. The parallel-jaw angle error is
180-degree periodic:

```text
delta_theta(p, g) = abs(((theta_p - theta_g + 90) mod 180) - 90)
```

For an IoU threshold `t` and an angle threshold `a`, the sample is a binary
success when at least one ground-truth grasp satisfies both tests:

```text
s_i(t, a) = 1 if any g in G_i has rotated_IoU(p_i, g) > t
                              and delta_theta(p_i, g) <= a
              0 otherwise

SR(t, a) = 100 / N * sum_i s_i(t, a)
```

The V3 surface uses three IoU thresholds and four angle thresholds:

```text
T_iou   = {0.25, 0.50, 0.75}
T_angle = {5, 10, 20, 30} degrees
```

Its headline metric is the unweighted mean of all 12 success-rate cells:

```text
mSR = 1 / 12 * sum over t in T_iou and a in T_angle of SR(t, a)
```

Every validation sample and every threshold pair therefore has equal weight.
`SR(0.25, 30 deg)` remains the familiar permissive grasp criterion, while mSR
also measures precise overlap and orientation. This prevents a model with good
coarse localization but inaccurate grasp geometry from looking artificially
strong. Segmentation IoU is reported separately and is not part of mSR.

The decoder must match the training contract before the rectangles are scored.
In particular, DROG-OFF V1 uses sigmoid grasp-size decoding and CROG uses clamp
decoding. The evaluator compares continuous rotated rectangles and uses the
top-1 prediction only.

### Current Grasp-Tools V3 results

The table contains the same model families as the reproduced OCID-VLG table
above. Values are percentages from the common Grasp-Tools V3 validation set and
the best available mSR checkpoint. A dash means that no formal V3 result is
recorded yet; training-only or incompatible legacy metrics are not inserted as
substitutes.

| Model | Config | Checkpoint | Size decoder | Seg. IoU | mSR | SR(0.25, 30 deg) | SR(0.25, 10 deg) | SR(0.50, 10 deg) | SR(0.75, 30 deg) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **DROG-OFF V1** | `config/grasp_tools/v3_drogoff_v1_grasp_tools_15k_original_scale.yaml` | `best_msr_model.pth` (epoch 24) | sigmoid | **83.78** | **89.14** | **99.51** | **97.45** | **96.52** | **75.67** |
| DROG | — | — | — | — | — | — | — | — | — |
| **CROG** | `config/grasp_tools/v3_crog_grasp_tools_15k_original_scale.yaml` | `best_msr_model.pth` (epoch 36) | clamp | 81.72 | 75.28 | 99.15 | 95.75 | 90.81 | 43.77 |
| LGD | — | — | — | — | — | — | — | — | — |
| GRConvNet-CLIP | — | — | — | — | — | — | — | — | — |
| GGCNN-CLIP | — | — | — | — | — | — | — | — | — |
| ETRG | — | — | — | — | — | — | — | — | — |

DROG-OFF V1 leads CROG by `13.86` mSR points. Their permissive
`SR(0.25, 30 deg)` scores are nearly tied (`99.51` versus `99.15`), but the gap
widens at high overlap: `SR(0.75, 30 deg)` is `75.67` for DROG-OFF V1 and
`43.77` for CROG. This is the main reason mSR is more informative than quoting
only the traditional loose success rate.

The CROG checkpoint previously produced `42.05` mSR when it was decoded with
the wrong sigmoid size activation. Re-evaluating the same trained model with
its matching clamp decoder gives the `75.28` headline value above; the
mismatched score is retained only as a decoder-compatibility diagnostic.
