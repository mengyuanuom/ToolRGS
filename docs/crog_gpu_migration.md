# CROG-GPU to ToolRGS migration

This migration is functional rather than a byte-for-byte second code tree.
CROG-GPU behavior is mapped into ToolRGS registries, named grasp structures,
runner hooks and deployment components so that there remains one implementation
of each model and dataset.

## Migrated behavior

- CROG, CROG-OFF, DROG and DROG-OFF CUDA behavior and weight activation
  metadata.
- VCoT CROG/DROG-OFF long-side plus short-side prediction.
- The official two-stage MapleGrasp port, including the optional VCoT
  short-side head and Stage-1-to-Stage-2 initialization path.
- OCID-VLG, VCoT/Grasp-Anything and Grasp-Tools data adapters.
- Official and legacy VCoT positive-grasp directory names.
- Latest, periodic, best-IoU, best-J@1 and best-J@5 checkpoints, scheduled
  validation and resumable optimizer/scheduler state.
- Verified on-demand CLIP, DINOv2, MambaVision and ResNet downloads.
- The complete OCID-VLG, VCoT and Grasp-Tools experiment reports and raw logs
  under `docs/experiments/`.
- GPU environment checks and visible-GPU-aware launch scripts under `tools/`.
- GUI inference for fixed-height, predicted-short-side, offset and combined
  short-side-plus-offset heads.

## Canonical ToolRGS mappings

| CROG-GPU path or behavior | ToolRGS destination |
| --- | --- |
| `engine/crog_engine.py` | `toolrgs/engine/loops.py` and `val_loop.py` |
| `train_crog.py` checkpoint policy | `toolrgs/engine/runner.py` + hooks |
| tuple output parsing | `toolrgs/structures/grasp.py` |
| `utils/grasp_tool_dataset.py` | `utils/grasp_tool_dataset.py` registry adapter |
| VCoT geometry and two-size maps | `utils/vcot_geometry.py`, `vcot_dataset.py` |
| `model/toolrgs/*` and root model variants | one registered module in `model/` |
| experiment records | `docs/experiments/` and `performance_summary.md` |

The legacy SSG/OCID-Grasp segmentation-only path is not exposed as a grasp GUI
profile because it does not implement ToolRGS's language-driven dense grasp
contract. DETRIS remains a backbone/segmentation component for the same reason.
ETRG remains trainable on OCID-VLG, but is excluded from the RGB-only GUI until
the camera layer supplies aligned depth.

## Validation

Static parsing covers every Python source file. CPU-safe contract tests cover
the registries, configuration inheritance, checkpoint/profile selection,
five/six/seven-map result adapters, predicted short-side decoding and offset
geometry. Full model forward and CUDA distributed tests require a Linux CUDA
environment with PyTorch and the selected datasets.

```bash
pip install -r requirement-gpu.txt
python tools/check_cuda_env.py
CUDA_VISIBLE_DEVICES=0,1 bash tools/train_gpu.sh config/vcot/drogoff.yaml
```
