#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-drogoff}"
DATASET="${2:-ocid_vlg}"
exec bash "${SCRIPT_DIR}/train_npu.sh" "config/${DATASET}/${MODEL}.yaml"
