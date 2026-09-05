#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${TOOLRGS_PYTHON:-}" ]]; then
  PYTHON_BIN="$TOOLRGS_PYTHON"
elif [[ -x "/home/aorus/anaconda3/envs/graspmamba/bin/python" ]]; then
  PYTHON_BIN="/home/aorus/anaconda3/envs/graspmamba/bin/python"
else
  PYTHON_BIN="python3"
fi
POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
STATE_DIR="${V4_FOLLOWUP_STATE_DIR:-$ROOT/exp/grasp_tools/.v4_maple_crog_followup}"
STAGE1_STARTED="$STATE_DIR/maple_stage1_started"
CROG_STARTED="$STATE_DIR/crog_started"

STAGE1_CFG="config/grasp_tools/maplegrasp_v4_dense_stage1_e12.yaml"
STAGE2_CFG="config/grasp_tools/maplegrasp_v4_dense_stage2_sigmoid_e12.yaml"
CROG_CFG="config/grasp_tools/crog_v4_dense_15k_sigmoid_e12.yaml"

STAGE1_DIR="exp/grasp_tools/maplegrasp_stage1_grasp_tools_v4_dense_15k_e12_bs8"
STAGE2_DIR="exp/grasp_tools/maplegrasp_stage2_grasp_tools_v4_dense_15k_sigmoid_e12_bs8"
CROG_DIR="exp/grasp_tools/crog_grasp_tools_v4_dense_15k_sigmoid_e12_bs32"

gpu_uuid() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
    | awk -F ', ' -v wanted="$1" '$1 == wanted {print $2}'
}

wait_for_gpu() {
  local gpu_id="$1"
  local uuid
  uuid="$(gpu_uuid "$gpu_id")"
  if [[ -z "$uuid" ]]; then
    echo "[error] physical GPU $gpu_id was not found" >&2
    return 1
  fi
  while nvidia-smi \
    --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Fx "$uuid" >/dev/null; do
    echo "[wait] physical GPU $gpu_id ($uuid) is occupied"
    sleep "$POLL_SECONDS"
  done
  echo "[ready] physical GPU $gpu_id ($uuid) is free"
}

run_experiment() {
  local name="$1"
  local gpu_id="$2"
  local config="$3"
  local output_dir="$4"
  local started_marker="${5:-}"
  local console_log="$output_dir/console.log"
  local last_model="$output_dir/last_model.pth"
  local -a resume_args=()

  mkdir -p "$output_dir"
  if grep -q "Training time:" "$console_log" 2>/dev/null; then
    echo "[skip] $name is already complete"
    [[ -z "$started_marker" ]] || touch "$started_marker"
    return 0
  fi
  if [[ -s "$last_model" ]]; then
    echo "[resume] $name from $last_model"
    resume_args=(--opts TRAIN.resume "$last_model")
  else
    echo "[start] $name from Epoch 1"
  fi

  wait_for_gpu "$gpu_id"
  [[ -z "$started_marker" ]] || touch "$started_marker"
  env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u train.py \
    --config "$config" --gpu 0 "${resume_args[@]}" \
    >>"$console_log" 2>&1

  if ! grep -q "Training time:" "$console_log"; then
    echo "[error] $name exited without a completion marker" >&2
    return 1
  fi
  echo "[complete] $name"
}

validate_contract() {
  "$PYTHON_BIN" - "$STAGE1_CFG" "$STAGE2_CFG" "$CROG_CFG" <<'PY'
import sys
from pathlib import Path

from utils.config import load_cfg_from_cfg_file

stage1, stage2, crog = [load_cfg_from_cfg_file(path) for path in sys.argv[1:]]
expected_root = Path("datasets/grasp-tools/aug_graspall_v4_dense_15k").resolve()
for cfg in (stage1, stage2, crog):
    assert Path(cfg.root_path).resolve() == expected_root
    assert cfg.epochs == 12
    assert list(cfg.milestones) == [10]
    assert cfg.val_start_epoch == 5
assert stage1.architecture == "maplegrasp" and stage1.stage1 and not stage1.stage2
assert stage1.weight is None and stage1.resume is None
assert stage2.architecture == "maplegrasp" and not stage2.stage1 and stage2.stage2
assert stage2.grasp_quality_loss_activation == "sigmoid"
assert stage2.grasp_width_loss_activation == "sigmoid"
assert stage2.grasp_quality_activation == "sigmoid"
assert stage2.grasp_size_activation == "sigmoid"
assert stage2.weight.endswith("maplegrasp_stage1_grasp_tools_v4_dense_15k_e12_bs8/best_iou_model.pth")
assert stage2.resume is None
assert crog.architecture == "crog"
assert crog.grasp_quality_loss_activation == "sigmoid"
assert crog.grasp_width_loss_activation == "sigmoid"
assert crog.grasp_quality_activation == "sigmoid"
assert crog.grasp_size_activation == "sigmoid"
print("[contract] V4 MapleGrasp Stage 1/2 and CROG configurations are valid")
PY
}

run_maple() {
  mkdir -p "$STATE_DIR"
  run_experiment "MapleGrasp V4 Stage 1" 1 "$STAGE1_CFG" "$STAGE1_DIR" "$STAGE1_STARTED"

  if [[ ! -s "$STAGE1_DIR/best_iou_model.pth" ]]; then
    echo "[error] Stage 1 completed without best_iou_model.pth; Stage 2 is blocked" >&2
    return 1
  fi
  while [[ ! -f "$CROG_STARTED" ]]; do
    echo "[wait] Stage 2 is waiting for CROG V4 to start"
    sleep "$POLL_SECONDS"
  done
  run_experiment "MapleGrasp V4 Stage 2" 1 "$STAGE2_CFG" "$STAGE2_DIR"
}

run_crog() {
  mkdir -p "$STATE_DIR"
  while [[ ! -f "$STAGE1_STARTED" ]]; do
    echo "[wait] CROG V4 is waiting for MapleGrasp Stage 1 to start"
    sleep "$POLL_SECONDS"
  done
  run_experiment "CROG V4" 0 "$CROG_CFG" "$CROG_DIR" "$CROG_STARTED"
}

schedule() {
  validate_contract
  mkdir -p "$STATE_DIR"
  if ! tmux has-session -t maplegrasp_v4_gpu1 2>/dev/null; then
    tmux new-session -d -s maplegrasp_v4_gpu1 \
      "cd '$ROOT' && env TOOLRGS_PYTHON='$PYTHON_BIN' GPU_POLL_SECONDS='$POLL_SECONDS' bash tools/train_grasp_tools_v4_maple_crog_followup.sh maple"
    echo "[scheduled] MapleGrasp V4 Stage 1 -> Stage 2 on physical GPU 1"
  fi
  if ! tmux has-session -t crog_v4_gpu0 2>/dev/null; then
    tmux new-session -d -s crog_v4_gpu0 \
      "cd '$ROOT' && env TOOLRGS_PYTHON='$PYTHON_BIN' GPU_POLL_SECONDS='$POLL_SECONDS' bash tools/train_grasp_tools_v4_maple_crog_followup.sh crog"
    echo "[scheduled] CROG V4 resume on physical GPU 0 after MapleGrasp Stage 1 starts"
  fi
}

case "${1:-schedule}" in
  schedule) schedule ;;
  maple) run_maple ;;
  crog) run_crog ;;
  *) echo "usage: $0 [schedule|maple|crog]" >&2; exit 2 ;;
esac
