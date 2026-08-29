#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_ID="${1:?usage: train_grasp_tools_v3_unified_queue.sh <gpu-id> <config>...}"
shift
PYTHON_BIN="${TOOLRGS_PYTHON:-python}"
POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

wait_for_gpu() {
  local gpu_uuid
  gpu_uuid="$(nvidia-smi \
    --query-gpu=index,uuid --format=csv,noheader,nounits \
    | awk -F ', ' -v wanted="$GPU_ID" '$1 == wanted {print $2}')"
  if [[ -z "$gpu_uuid" ]]; then
    echo "[error] physical GPU $GPU_ID was not found" >&2
    return 1
  fi
  while nvidia-smi \
    --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Fxq "$gpu_uuid"; do
    echo "[wait] physical GPU $GPU_ID ($gpu_uuid) is occupied"
    sleep "$POLL_SECONDS"
  done
  echo "[ready] physical GPU $GPU_ID ($gpu_uuid) is free"
}

run_experiment() {
  local config="$1"
  local exp_name
  local output_dir
  local console_log
  local last_model
  local -a resume_args=()

  exp_name="$($PYTHON_BIN - "$config" <<'PY'
import sys
from utils.config import load_cfg_from_cfg_file
print(load_cfg_from_cfg_file(sys.argv[1]).exp_name)
PY
)"
  output_dir="exp/grasp_tools/$exp_name"
  console_log="$output_dir/console.log"
  last_model="$output_dir/last_model.pth"
  mkdir -p "$output_dir"

  if grep -q "Training time:" "$console_log" 2>/dev/null; then
    echo "[skip] $exp_name is already complete"
    return 0
  fi
  if [[ -s "$last_model" ]]; then
    echo "[resume] $exp_name from $last_model"
    resume_args=(--opts TRAIN.resume "$last_model")
  else
    echo "[start] $exp_name from Epoch 1"
  fi

  wait_for_gpu
  env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u train.py \
    --config "$config" --gpu 0 "${resume_args[@]}" \
    >>"$console_log" 2>&1

  if ! grep -q "Training time:" "$console_log"; then
    echo "[error] $exp_name exited without a completion marker" >&2
    return 1
  fi
  echo "[complete] $exp_name"
}

for config in "$@"; do
  run_experiment "$config"
done
