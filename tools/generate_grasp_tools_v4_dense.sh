#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/datasets/grasp-tools/aug_graspall_v4_dense_15k}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" tools/dataset_converters/grasp_tools/augment.py \
  --src-dir assets/grasp_tools/graspall \
  --background-dir assets/grasp_tools/backgrounds \
  --out-dir "${OUT_DIR}" \
  --train-scenes "${TRAIN_SCENES:-12000}" \
  --val-scenes "${VAL_SCENES:-1000}" \
  --test-scenes "${TEST_SCENES:-2000}" \
  --objects-min 8 \
  --objects-max 10 \
  --queries-min 8 \
  --queries-max 10 \
  --max-query-difficulty 1 \
  --language-templates shared \
  --category-vocabulary canonical \
  --scales 0.4,0.5,0.6,0.7,0.8 \
  --adaptive-min-scale 0.4 \
  --adaptive-scale-step 0.1 \
  --angle-bins 24 \
  --same-category-probability 0 \
  --hard-negative-probability 0 \
  --placement-attempts 300 \
  --scene-attempts 80 \
  --candidate-replacements 44 \
  --require-exact-object-count \
  --query-every-object \
  --unique-categories \
  --balance-on-success \
  --seed "${SEED:-2026}" \
  --image-ext jpg \
  --jpeg-quality 95 \
  --preview-count "${PREVIEW_COUNT:-30}" \
  "$@"

"${PYTHON_BIN}" tools/dataset_converters/grasp_tools/validate_v4_dense.py \
  "${OUT_DIR}" \
  --objects-min 8 \
  --objects-max 10 \
  --scale-min 0.4 \
  --scale-max 0.8 \
  --angle-bins 24
