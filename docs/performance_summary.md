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

Source: CROG-NPU evaluation commit `3d9afef`. Protocol: top-1 rotated IoU at
least `0.25` and 180-degree-periodic angle error at most `30` degrees. Full
details are in [`experiments/vcot_20260810`](experiments/vcot_20260810/README.md).

| Model | Config | Split | Checkpoint | Seg. IoU | GraspSR | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| DROG-OFF | `config/vcot/drogoff.yaml` | seen | epoch-20 best | 88.98 | 80.97 | `3d9afef` | Matched baseline decoding. |
| DROG-OFF | `config/vcot/drogoff.yaml` | unseen | epoch-20 best | 60.96 | 57.30 | `3d9afef` | Matched baseline decoding. |
| DROG-OFF | `config/vcot/drogoff.yaml` | seen | epoch-20 best | 88.98 | 80.77 | `3d9afef` | Calibrated decoding. |
| DROG-OFF | `config/vcot/drogoff.yaml` | unseen | epoch-20 best | 60.96 | **58.71** | `3d9afef` | Calibrated decoding; selected unseen result. |

The matched Seen/Unseen harmonic mean is `67.11`; calibrated decoding gives
`68.00`.

## Grasp-Tools

### Current Grasp-Tools V2 strict-IoU results

Both rows use rotated grasp IoU `0.50` and angle error at most `30` degrees.
The DROG-OFF row is the completed V2 test evaluation. The CROG row is the best
training-time validation checkpoint at epoch 24; a formal CROG test evaluation
has not yet been recorded, so the two rows must not be treated as a direct
test-set ranking.

| Model | Config | Split | Checkpoint | Seg. IoU | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| DROG-OFF | `config/grasp_tools/drogoff.yaml` | V2 test | - | **83.75** | **86.14** | **89.29** | - | Completed 3090 strict-IoU re-evaluation; exact checkpoint/commit still to be recovered from the server. |
| CROG | `config/grasp_tools/crog.yaml` | V2 val | `best_epoch_024_J1_80.67_J5_83.95.pth` | 80.59 | 80.67 | 83.95 | `499fcba` | Best validation checkpoint at epoch 24; not yet a formal test result. |

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
