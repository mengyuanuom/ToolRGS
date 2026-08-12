#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  echo "Usage: bash tools/train_npu.sh <config.yaml>" >&2
  echo "Example: bash tools/train_npu.sh config/grasp_tools/drogoff_v2.yaml" >&2
}

if [[ "$#" -ne 1 ]]; then
  usage
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
CONFIG="$1"
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 2; }

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a NPU_DEVICE_LIST <<< "${ASCEND_RT_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#NPU_DEVICE_LIST[@]}"
fi

if grep -Eq '^[[:space:]]*dataset[[:space:]]*:[[:space:]]*vcot([[:space:]]|$)' "${CONFIG}"; then
  DATASET_NAME="VCoT/Grasp-Anything"
  DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/graspanything-vcot}"
  SPLIT_ROOT="${SPLIT_ROOT:-${DATA_ROOT}/split/vcot}"
elif grep -Eqi '^[[:space:]]*dataset[[:space:]]*:[[:space:]]*grasp-?tools?([[:space:]]|$)' "${CONFIG}"; then
  DATASET_NAME="Grasp-Tools"
  DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/grasp-tools/aug_graspall_v2}"
  SPLIT_ROOT=""
else
  DATASET_NAME="OCID-VLG"
  DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/OCID-VLG}"
  SPLIT_ROOT=""
fi

[[ -d "${DATA_ROOT}" ]] || { echo "${DATASET_NAME} not found: ${DATA_ROOT}" >&2; exit 2; }
TRAIN_OPTS=(DATA.root_path "${DATA_ROOT}")
[[ -z "${SPLIT_ROOT}" ]] || TRAIN_OPTS+=(DATA.split_root "${SPLIT_ROOT}")

if grep -Eq '^[[:space:]]*dino_pretrain[[:space:]]*:' "${CONFIG}"; then
  CLIP_WEIGHT="${CLIP_WEIGHT:-${REPO_ROOT}/pretrain/ViT-B-16.pt}"
  DINO_WEIGHT="${DINO_WEIGHT:-${REPO_ROOT}/pretrain/dinov2_vitb14_reg4_pretrain.pth}"
  python3 tools/download_pretrained.py clip-vit-b16 --output "${CLIP_WEIGHT}"
  python3 tools/download_pretrained.py dinov2-vitb14-reg4 --output "${DINO_WEIGHT}"
  TRAIN_OPTS+=(TRAIN.clip_pretrain "${CLIP_WEIGHT}" TRAIN.dino_pretrain "${DINO_WEIGHT}")
else
  CLIP_WEIGHT="${CLIP_WEIGHT:-${REPO_ROOT}/pretrain/RN50.pt}"
  python3 tools/download_pretrained.py clip-rn50 --output "${CLIP_WEIGHT}"
  TRAIN_OPTS+=(TRAIN.clip_pretrain "${CLIP_WEIGHT}")
fi

if grep -Eq '^[[:space:]]*architecture[[:space:]]*:[[:space:]]*etrg([[:space:]]|$)' "${CONFIG}"; then
  RESNET_WEIGHT="${RESNET_WEIGHT:-${REPO_ROOT}/pretrain/resnet18-f37072fd.pth}"
  python3 tools/download_pretrained.py resnet18 --output "${RESNET_WEIGHT}"
  TRAIN_OPTS+=(TRAIN.depth_pretrain "${RESNET_WEIGHT}")
fi

echo "[launch] config: ${CONFIG}"
echo "[launch] dataset: ${DATASET_NAME} (${DATA_ROOT})"
echo "[launch] visible NPUs: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "[launch] processes: ${NPROC_PER_NODE}; AMP is read only from YAML"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train.py --config "${CONFIG}" --opts "${TRAIN_OPTS[@]}"
