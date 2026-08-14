#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${TOOLRGS_GRASP_TOOLS_ROOT:-$ROOT/datasets/grasp-tools/aug_graspall_v2}"
WORK_DIR="${TOOLRGS_DETECTOR_WORK_DIR:-$ROOT/work_dirs/faster_rcnn_r50_fpn_grasp_tools_v2_24e}"
CONFIG="configs/detection/faster_rcnn_r50_fpn_grasp_tools_v2_24e.py"

if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/val" || ! -d "$DATA_ROOT/test" ]]; then
  printf 'Grasp-Tools V2 dataset is incomplete: %s\n' "$DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"

python tools/dataset_converters/grasp_tools/to_coco_detection.py \
  --dataset-root "$DATA_ROOT"

export TOOLRGS_GRASP_TOOLS_ROOT="$DATA_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

resume_args=()
if [[ "${TOOLRGS_DETECTOR_RESUME:-0}" == "1" ]]; then
  resume_args=(--resume auto)
fi

exec torchrun --standalone --nproc_per_node=2 \
  tools/train_detector.py "$CONFIG" \
  --launcher pytorch \
  --work-dir "$WORK_DIR" \
  "${resume_args[@]}"
