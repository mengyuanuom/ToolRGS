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
for iou in 0.25 0.50 0.75; do
  for angle in 5.0 10.0 20.0 30.0; do
    log="$RESULT_DIR/iou_${iou}_angle_${angle}.log"
    if grep -q "Final IoU=" "$log" 2>/dev/null; then
      echo "[skip] completed IoU=$iou angle=$angle"
      continue
    fi
    echo "[evaluate] IoU=$iou angle=$angle"
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u evaluate.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --split "$SPLIT" \
      --opts \
      TEST.grasp_iou_threshold "$iou" \
      TEST.grasp_angle_threshold "$angle" \
      "${extra_opts[@]}" \
      >"$log" 2>&1
  done
done

"$PYTHON_BIN" - "$RESULT_DIR" <<'PY'
import glob
import os
import re
import sys

result_dir = sys.argv[1]
pattern = re.compile(
    r"Final IoU=.*J=\[([0-9eE+.-]+),\s*([0-9eE+.-]+)\]"
)
name_pattern = re.compile(r"iou_([0-9.]+)_angle_([0-9.]+)\.log$")
rows = []
for path in sorted(glob.glob(os.path.join(result_dir, "iou_*_angle_*.log"))):
    name_match = name_pattern.search(path)
    if not name_match:
        continue
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError(f"Missing final J metrics in {path}")
    j1, j5 = map(float, matches[-1])
    rows.append((float(name_match.group(1)), float(name_match.group(2)), j1, j5))
if len(rows) != 12:
    raise RuntimeError(f"Expected 12 completed grid cells, found {len(rows)}")
msr1 = sum(row[2] for row in rows) / len(rows)
msr5 = sum(row[3] for row in rows) / len(rows)
summary = os.path.join(result_dir, "summary.tsv")
with open(summary, "w", encoding="utf-8") as handle:
    handle.write("iou\tangle\tJ@1\tJ@5\n")
    for iou, angle, j1, j5 in rows:
        handle.write(f"{iou:.2f}\t{angle:.1f}\t{100*j1:.4f}\t{100*j5:.4f}\n")
    handle.write(f"mSR\tmean\t{100*msr1:.4f}\t{100*msr5:.4f}\n")
print(f"mSR@1={100*msr1:.4f} mSR@5={100*msr5:.4f}")
print(summary)
PY
