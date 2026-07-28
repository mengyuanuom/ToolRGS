# ToolRGS Performance Summary

This document records results produced by the unified ToolRGS evaluator. Metric
cells use `-` until a checkpoint has been evaluated. Keep paper
reported numbers in notes or citations rather than mixing them with reproduced
ToolRGS results.

## Recording rules

- Record `J@1` and `J@5` as percentages from `0` to `100`.
- Use the exact dataset split shown in the row. Add a row instead of overwriting
  a result obtained with a different split, seed, checkpoint, or protocol.
- Store the checkpoint path and the Git commit used for evaluation so every
  number remains traceable.
- Keep `-` when a result has not been filled or a metric is not implemented.
- ETRG-A is listed only for OCID-VLG because it requires aligned RGB-D input.

## OCID-VLG

| Model | Config | Split | Checkpoint | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| CROG | `config/ocid_vlg/crog.yaml` | test |  | 77.2 |87.7  |  | - |
| DROG-OFF | `config/ocid_vlg/drogoff.yaml` | test | - | 85.95 | 91.51 | - | - | - |
| ETRG-A R50 | `config/ocid_vlg/etrg.yaml` | test | - | - | - | - | - |
| ETRG-A R101 | `config/ocid_vlg/etrg_r101.yaml` | test | - | - | - | - | - |
| MapleGrasp | `config/ocid_vlg/maplegrasp.yaml` | test | - | - | - | - | - |
| GGCNN-CLIP | `config/ocid_vlg/ggcnnclip.yaml` | test | - | - | - | - | - |
| GRConvNet-CLIP | `config/ocid_vlg/grconvnetclip.yaml` | test | - | - | - | - | - |
| GraspMamba | `config/ocid_vlg/graspmamba.yaml` | test | - | - | - | - | - |
| LGD | `config/ocid_vlg/lgd.yaml` | test | - | - | - | - | - |

## Grasp-Tools

| Model | Config | Split | Checkpoint | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| CROG | `config/grasp_tools/crog.yaml` | test | - | 58.63 | 59.35 | - | - |
| DROG-OFF | `config/grasp_tools/drogoff.yaml` | test |  | 62.10 | 62.34 | - | - |
| DROG-OFF v2 | `config/grasp_tools/drogoff_v2.yaml` | test | - | - | - | - | - |
| MapleGrasp | `config/grasp_tools/maplegrasp.yaml` | test | - | - | - | - | - |
| GGCNN-CLIP | `config/grasp_tools/ggcnnclip.yaml` | test | - | - | - | - | - |
| GRConvNet-CLIP | `config/grasp_tools/grconvnetclip.yaml` | test | - | - | - | - | - |
| GraspMamba | `config/grasp_tools/graspmamba.yaml` | test | - | - | - | - | - |
| LGD | `config/grasp_tools/lgd.yaml` | test | - | - | - | - | - |

## Grasp-Tools-hard

| Model | Config | Split | Checkpoint | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| CROG | `config/grasp_tools/crog.yaml` | test | - | 25.23 |25.76 | - | - |
| DROG-OFF | `config/grasp_tools/drogoff.yaml` | test |  | 27.35 | 27.73 | - | - |
| DROG-OFF v2 | `config/grasp_tools/drogoff_v2.yaml` | test | - | - | - | - | - |
| MapleGrasp | `config/grasp_tools/maplegrasp.yaml` | test | - | - | - | - | - |
| GGCNN-CLIP | `config/grasp_tools/ggcnnclip.yaml` | test | - | - | - | - | - |
| GRConvNet-CLIP | `config/grasp_tools/grconvnetclip.yaml` | test | - | - | - | - | - |
| GraspMamba | `config/grasp_tools/graspmamba.yaml` | test | - | - | - | - | - |
| LGD | `config/grasp_tools/lgd.yaml` | test | - | - | - | - | - |


## Evaluation command template

```bash
python evaluate.py \
  --config CONFIG_PATH \
  --checkpoint CHECKPOINT_PATH
```

Copy the final evaluation log values into the matching row and record
`git rev-parse --short HEAD` in `Eval commit`.
