# ToolRGS performance-summary snapshot - 2026-08-11

This directory preserves the populated experiment rows from ToolRGS commit
`558efb937ca5cf343e1aab2b75713caf59bd564e`. The values below are copied from
`docs/performance_summary.md`; empty provenance fields in the source remain
explicitly unknown rather than being reconstructed from model names.

All values are percentages. These rows must not be compared with the
CROG-NPU results without also accounting for implementation, checkpoint and
evaluation-protocol differences.

## OCID-VLG

| Model | Config | Split | Checkpoint | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| CROG | `config/ocid_vlg/crog.yaml` | test | - | 77.20 | 87.70 | - | Copied from ToolRGS summary. |
| DROG-OFF | `config/ocid_vlg/drogoff.yaml` | test | - | 85.95 | 91.51 | - | Copied from ToolRGS summary. |
| ETRG-A R50 | `config/ocid_vlg/etrg.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| ETRG-A R101 | `config/ocid_vlg/etrg_r101.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| MapleGrasp | `config/ocid_vlg/maplegrasp.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| GGCNN-CLIP | `config/ocid_vlg/ggcnnclip.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| GRConvNet-CLIP | `config/ocid_vlg/grconvnetclip.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| GraspMamba | `config/ocid_vlg/graspmamba.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| LGD | `config/ocid_vlg/lgd.yaml` | test | - | - | - | - | Not evaluated in the source summary. |

## Grasp-Tools

| Model | Config | Split | Checkpoint | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| CROG | `config/grasp_tools/crog.yaml` | test | - | 58.63 | 59.35 | - | 3090 result recorded by ToolRGS; checkpoint and protocol were not recorded. |
| DROG-OFF | `config/grasp_tools/drogoff.yaml` | test | - | 62.10 | 62.34 | - | 3090 result recorded by ToolRGS; checkpoint and protocol were not recorded. |
| DROG-OFF v2 | `config/grasp_tools/drogoff_v2.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| MapleGrasp | `config/grasp_tools/maplegrasp.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| GGCNN-CLIP | `config/grasp_tools/ggcnnclip.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| GRConvNet-CLIP | `config/grasp_tools/grconvnetclip.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| GraspMamba | `config/grasp_tools/graspmamba.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| LGD | `config/grasp_tools/lgd.yaml` | test | - | - | - | - | Not evaluated in the source summary. |

## Grasp-Tools-hard

| Model | Config | Split | Checkpoint | J@1 | J@5 | Eval commit | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| CROG | `config/grasp_tools/crog.yaml` | test | - | 25.23 | 25.76 | - | 3090 result recorded by ToolRGS; checkpoint and protocol were not recorded. |
| DROG-OFF | `config/grasp_tools/drogoff.yaml` | test | - | 27.35 | 27.73 | - | 3090 result recorded by ToolRGS; checkpoint and protocol were not recorded. |
| DROG-OFF v2 | `config/grasp_tools/drogoff_v2.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| MapleGrasp | `config/grasp_tools/maplegrasp.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| GGCNN-CLIP | `config/grasp_tools/ggcnnclip.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| GRConvNet-CLIP | `config/grasp_tools/grconvnetclip.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| GraspMamba | `config/grasp_tools/graspmamba.yaml` | test | - | - | - | - | Not evaluated in the source summary. |
| LGD | `config/grasp_tools/lgd.yaml` | test | - | - | - | - | Not evaluated in the source summary. |

