# ToolRGS Performance Summary

This document records results produced by the unified ToolRGS evaluator. Metric
cells are intentionally blank until a checkpoint has been evaluated. Keep paper
reported numbers in notes or citations rather than mixing them with reproduced
ToolRGS results.

## Recording rules

- Record `IoU`, `Pr@50` through `Pr@90`, `J@1`, and `J@5` as percentages from
  `0` to `100`.
- Use the exact dataset split shown in the row. Add a row instead of overwriting
  a result obtained with a different split, seed, checkpoint, or protocol.
- Store the checkpoint path and the Git commit used for evaluation so every
  number remains traceable.
- Leave a metric cell empty when that metric is not implemented for a model.
- ETRG-A is listed only for OCID-VLG because it requires aligned RGB-D input.

## OCID-VLG

| Model | Config | Split | Checkpoint | IoU | Pr@50 | Pr@60 | Pr@70 | Pr@80 | Pr@90 | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CROG | `config/ocid_vlg/crog.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| DROG | `config/ocid_vlg/drog.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| DROG-OFF | `config/ocid_vlg/drogoff.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| ETRG-A R50 | `config/ocid_vlg/etrg.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| ETRG-A R101 | `config/ocid_vlg/etrg_r101.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| MapleGrasp | `config/ocid_vlg/maplegrasp.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| GGCNN-CLIP | `config/ocid_vlg/ggcnnclip.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| GRConvNet-CLIP | `config/ocid_vlg/grconvnetclip.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| GraspMamba | `config/ocid_vlg/graspmamba.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| LGD | `config/ocid_vlg/lgd.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |

## VCoT / Grasp-Anything

| Model | Config | Split | Checkpoint | IoU | Pr@50 | Pr@60 | Pr@70 | Pr@80 | Pr@90 | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CROG | `config/vcot/crog.yaml` | seen |  |  |  |  |  |  |  |  |  |  |  |
| CROG | `config/vcot/crog.yaml` | unseen |  |  |  |  |  |  |  |  |  |  |  |
| DROG | `config/vcot/drog.yaml` | seen |  |  |  |  |  |  |  |  |  |  |  |
| DROG | `config/vcot/drog.yaml` | unseen |  |  |  |  |  |  |  |  |  |  |  |
| DROG-OFF | `config/vcot/drogoff.yaml` | seen |  |  |  |  |  |  |  |  |  |  |  |
| DROG-OFF | `config/vcot/drogoff.yaml` | unseen |  |  |  |  |  |  |  |  |  |  |  |
| MapleGrasp | `config/vcot/maplegrasp.yaml` | seen |  |  |  |  |  |  |  |  |  |  |  |
| MapleGrasp | `config/vcot/maplegrasp.yaml` | unseen |  |  |  |  |  |  |  |  |  |  |  |
| GGCNN-CLIP | `config/vcot/ggcnnclip.yaml` | seen |  |  |  |  |  |  |  |  |  |  |  |
| GGCNN-CLIP | `config/vcot/ggcnnclip.yaml` | unseen |  |  |  |  |  |  |  |  |  |  |  |
| GRConvNet-CLIP | `config/vcot/grconvnetclip.yaml` | seen |  |  |  |  |  |  |  |  |  |  |  |
| GRConvNet-CLIP | `config/vcot/grconvnetclip.yaml` | unseen |  |  |  |  |  |  |  |  |  |  |  |
| GraspMamba | `config/vcot/graspmamba.yaml` | seen |  |  |  |  |  |  |  |  |  |  |  |
| GraspMamba | `config/vcot/graspmamba.yaml` | unseen |  |  |  |  |  |  |  |  |  |  |  |
| LGD | `config/vcot/lgd.yaml` | seen |  |  |  |  |  |  |  |  |  |  |  |
| LGD | `config/vcot/lgd.yaml` | unseen |  |  |  |  |  |  |  |  |  |  |  |

## Grasp-Tools

| Model | Config | Split | Checkpoint | IoU | Pr@50 | Pr@60 | Pr@70 | Pr@80 | Pr@90 | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CROG | `config/grasp_tools/crog.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| DROG | `config/grasp_tools/drog.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| DROG-OFF | `config/grasp_tools/drogoff.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| DROG-OFF v2 | `config/grasp_tools/drogoff_v2.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| MapleGrasp | `config/grasp_tools/maplegrasp.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| GGCNN-CLIP | `config/grasp_tools/ggcnnclip.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| GRConvNet-CLIP | `config/grasp_tools/grconvnetclip.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| GraspMamba | `config/grasp_tools/graspmamba.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |
| LGD | `config/grasp_tools/lgd.yaml` | test |  |  |  |  |  |  |  |  |  |  |  |

## Evaluation command template

```bash
python evaluate.py \
  --config CONFIG_PATH \
  --checkpoint CHECKPOINT_PATH
```

Copy the final evaluation log values into the matching row and record
`git rev-parse --short HEAD` in `Eval commit`.