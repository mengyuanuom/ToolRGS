#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_ID="${MAPLEGRASP_GPU_ID:-1}"
PYTHON_BIN="${MAPLEGRASP_PYTHON:-python}"
STAGE1_CFG="config/grasp_tools/maplegrasp_v3_stage1.yaml"
STAGE2_CFG="config/grasp_tools/maplegrasp_v3_stage2.yaml"
STAGE1_DIR="exp/grasp_tools/maplegrasp_stage1_grasp_tools_v3_15k_original300"
STAGE2_DIR="exp/grasp_tools/maplegrasp_stage2_grasp_tools_v3_15k_original300"

run_stage() {
  local name="$1"
  local config="$2"
  local output_dir="$3"
  local console_log="$output_dir/console.log"
  local last_model="$output_dir/last_model.pth"
  local -a resume_args=()

  mkdir -p "$output_dir"
  if grep -q "Training time:" "$console_log" 2>/dev/null; then
    echo "$name is already complete; skipping."
    return 0
  fi
  if [[ -s "$last_model" ]]; then
    echo "$name will resume from $last_model"
    resume_args=(--opts TRAIN.resume "$last_model")
  else
    echo "$name will start from Epoch 1"
  fi

  env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u train.py \
    --config "$config" --gpu 0 "${resume_args[@]}" \
    >>"$console_log" 2>&1

  if ! grep -q "Training time:" "$console_log"; then
    echo "$name exited without a completion marker." >&2
    return 1
  fi
}

run_stage "MapleGrasp Stage 1" "$STAGE1_CFG" "$STAGE1_DIR"

STAGE1_BEST="$STAGE1_DIR/best_iou_model.pth"
if [[ ! -s "$STAGE1_BEST" ]]; then
  echo "Stage 1 completed without $STAGE1_BEST; Stage 2 will not start." >&2
  exit 1
fi

run_stage "MapleGrasp Stage 2" "$STAGE2_CFG" "$STAGE2_DIR"
echo "MapleGrasp V3 two-stage training completed."
