#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${TOOLRGS_PYTHON:-python}"
DATASET_ROOT="${V4_DATASET_ROOT:-$ROOT/datasets/grasp-tools/aug_graspall_v4_dense_15k}"
GENERATION_SESSION="${V4_GENERATION_SESSION:-grasp_tools_v4_generate}"
POLL_SECONDS="${V4_POLL_SECONDS:-60}"
DROGOFF_CONFIG="config/grasp_tools/drogoff_v1_v4_dense_15k_sigmoid_e12.yaml"
CROG_CONFIG="config/grasp_tools/crog_v4_dense_15k_sigmoid_e12.yaml"
DROGOFF_SESSION="drogoff_v1_v4_gpu0"
CROG_SESSION="crog_v4_gpu1"

scene_count() {
  find "$DATASET_ROOT" -type f -name '*.json' 2>/dev/null | wc -l
}

while tmux has-session -t "$GENERATION_SESSION" 2>/dev/null; do
  echo "[wait] V4 generation is active: $(scene_count) JSON files"
  sleep "$POLL_SECONDS"
done

echo "[validate] checking the completed V4 dataset"
"$PYTHON_BIN" tools/dataset_converters/grasp_tools/validate_v4_dense.py \
  "$DATASET_ROOT" \
  --objects-min 10 \
  --objects-max 12 \
  --scale-min 0.3 \
  --scale-max 0.6 \
  --angle-bins 24

"$PYTHON_BIN" - "$DROGOFF_CONFIG" "$CROG_CONFIG" "$DATASET_ROOT" <<'PY'
import sys
from pathlib import Path

from utils.config import load_cfg_from_cfg_file

expected = (
    (sys.argv[1], "drogoff", 8),
    (sys.argv[2], "crog", 32),
)
dataset_root = Path(sys.argv[3]).resolve()
for config_path, architecture, batch_size in expected:
    cfg = load_cfg_from_cfg_file(config_path)
    assert cfg.architecture == architecture
    assert Path(cfg.root_path).resolve() == dataset_root
    assert cfg.epochs == 12
    assert list(cfg.milestones) == [10]
    assert cfg.batch_size == batch_size
    assert cfg.grasp_quality_loss_activation == "sigmoid"
    assert cfg.grasp_width_loss_activation == "sigmoid"
    assert cfg.grasp_quality_activation == "sigmoid"
    assert cfg.grasp_size_activation == "sigmoid"
    assert cfg.weight is None and cfg.resume is None
print("[contract] V4 training configurations are valid")
PY

launch() {
  local session="$1"
  local gpu="$2"
  local config="$3"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip] tmux session already exists: $session"
    return
  fi
  echo "[launch] $session on physical GPU $gpu"
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && env TOOLRGS_PYTHON='$PYTHON_BIN' GPU_POLL_SECONDS='$POLL_SECONDS' bash tools/train_grasp_tools_v3_unified_queue.sh '$gpu' '$config'"
}

launch "$DROGOFF_SESSION" 0 "$DROGOFF_CONFIG"
launch "$CROG_SESSION" 1 "$CROG_CONFIG"

echo "[scheduled] V4 DrogOff V1 and CROG jobs launched"

