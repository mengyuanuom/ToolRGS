#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_ID="${1:?usage: evaluate_grasp_tools_msr_grid.sh <gpu-id> <config> <checkpoint> <result-dir> [split]}"
CONFIG="${2:?missing config}"
CHECKPOINT="${3:?missing checkpoint}"
RESULT_DIR="${4:?missing result directory}"
SPLIT="${5:-val}"
PYTHON_BIN="${TOOLRGS_PYTHON:-python}"
POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
EVAL_WORKERS="${EVAL_WORKERS:-}"

gpu_uuid="$(nvidia-smi \
  --query-gpu=index,uuid --format=csv,noheader,nounits \
  | awk -F ', ' -v wanted="$GPU_ID" '$1 == wanted {print $2}')"
if [[ -z "$gpu_uuid" ]]; then
  echo "[error] physical GPU $GPU_ID was not found" >&2
  exit 1
fi
if [[ "$ALLOW_SHARED_GPU" != "1" ]]; then
  while nvidia-smi \
    --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Fxq "$gpu_uuid"; do
    echo "[wait] physical GPU $GPU_ID ($gpu_uuid) is occupied"
    sleep "$POLL_SECONDS"
  done
fi

extra_opts=()
if [[ -n "$EVAL_BATCH_SIZE" ]]; then
  extra_opts+=(TRAIN.batch_size_val "$EVAL_BATCH_SIZE")
fi
if [[ -n "$EVAL_WORKERS" ]]; then
  extra_opts+=(TRAIN.workers_val "$EVAL_WORKERS")
fi

mkdir -p "$RESULT_DIR"
summary="$RESULT_DIR/summary.tsv"
log="$RESULT_DIR/evaluate.log"
cache="$RESULT_DIR/predictions.npz"
if grep -q $'^mSR\tmean\t' "$summary" 2>/dev/null; then
  echo "[skip] completed $summary"
  tail -n 1 "$summary"
  exit 0
fi

if [[ -f "$cache" ]]; then
  echo "[score] reuse decoded predictions without model inference"
  "$PYTHON_BIN" -u evaluate.py \
    --config "$CONFIG" \
    --score-cache "$cache" \
    --split "$SPLIT" \
    --msr-output "$summary" \
    --opts "${extra_opts[@]}" \
    >"$log" 2>&1
else
  echo "[evaluate] one forward pass and save decoded predictions"
  env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" \
    --prediction-cache "$cache" \
    --msr-output "$summary" \
    --opts "${extra_opts[@]}" \
    >"$log" 2>&1
fi
tail -n 1 "$summary"
