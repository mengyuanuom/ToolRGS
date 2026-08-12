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

## Grasp-Tools

### Current Grasp-Tools V2 strict-IoU results

All rows below are formal evaluations on the complete Grasp-Tools V2 `test`
split (8,000 language-conditioned samples). A grasp is successful when its
rotated rectangle has IoU at least `0.50` with a ground-truth rectangle. The
headline `J@1`/`J@5` columns use the conventional angle-error limit of `30`
degrees; the stricter `15`-degree results are shown separately where available.
Image-mask IoU and `Pr@50`--`Pr@90` measure segmentation rather than grasp
rectangle success.

| Model | Config | Checkpoint | Seg. IoU | J@1 (30 deg) | J@5 (30 deg) | J@1 (15 deg) | J@5 (15 deg) | Eval commit |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **DROG-OFF** | `config/grasp_tools/drogoff.yaml` | `drogoff_grasp_tools_v2_best_j1.pth` | **83.75** | **86.14** | **89.29** | - | - | `499fcba` |
| DROG | `config/grasp_tools/drog.yaml` | `best_epoch_036_J1_81.12_J5_84.47.pth` | 83.10 | 79.60 | 84.82 | 79.50 | 84.70 | `499fcba` + dual-angle evaluator |
| CROG | `config/grasp_tools/crog.yaml` | `best_epoch_024_J1_80.67_J5_83.95.pth` | 81.47 | 80.20 | 83.75 | 79.84 | 83.39 | `499fcba` + dual-angle evaluator |

The DROG-OFF V2 strict Test log explicitly loads the original checkpoint
`best_epoch_011_J1_29.45_J5_34.67.pth` and reports IoU `83.75`, J@1
`86.14`, and J@5 `89.29`. The `29.45/34.67` values embedded in that
original filename are historical training-time validation values from the older
evaluation path; they are not the strict Test scores. The same bytes are
published under the unambiguous name
[drogoff_grasp_tools_v2_best_j1.pth](https://github.com/mengyuanuom/ToolRGS/releases/download/grasp-tools-v2-weights/drogoff_grasp_tools_v2_best_j1.pth)
with SHA-256
`7fcef57dd968a381d61bab7ef35e5e3906149bcec9f4fdbb6658da23659e73d5`.
DROG-OFF has not yet been re-evaluated with the additional `15`-degree
counter, so those two cells remain intentionally blank.

| Model | Pr@50 | Pr@60 | Pr@70 | Pr@80 | Pr@90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **DROG-OFF** | **99.70** | **98.62** | **92.20** | **72.95** | **25.08** |
| DROG | 99.60 | 97.95 | 91.15 | 69.88 | 24.24 |
| CROG | 99.00 | 96.14 | 88.09 | 63.32 | 21.07 |

Under the common 30-degree protocol, DROG-OFF leads CROG by `5.94` J@1 and
`5.54` J@5 points. DROG is `0.60` points below CROG at J@1 but `1.07` points
above it at J@5. Tightening the angle threshold from 30 to 15 degrees changes
DROG by only `-0.10` J@1 / `-0.12` J@5 and CROG by `-0.36` / `-0.36`.

### Imported ToolRGS legacy records

The following values are preserved from ToolRGS `558efb9`. That source did not
record the checkpoint, evaluation commit or exact grasp-IoU protocol, so they
are retained for provenance but are not the current headline V2 results.

| Profile | Model | Config | Split | Checkpoint | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| Standard | CROG | `config/grasp_tools/crog.yaml` | test | - | 58.63 | 59.35 | - | Historical ToolRGS record; protocol unknown. |
| Standard | DROG-OFF | `config/grasp_tools/drogoff.yaml` | test | - | 62.10 | 62.34 | - | Historical ToolRGS record; protocol unknown. |
| Hard | CROG | `config/grasp_tools/crog.yaml` | test-hard | - | 25.23 | 25.76 | - | Historical ToolRGS record; protocol unknown. |
| Hard | DROG-OFF | `config/grasp_tools/drogoff.yaml` | test-hard | - | 27.35 | 27.73 | - | Historical ToolRGS record; protocol unknown. |
