#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONFIG="${TOOLRGS_GUI_CONFIG:-config/deployment/lab.yaml}"
ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$ACTION" in
  check)
    exec python tools/check_deployment.py \
      --config "$CONFIG" "$@"
    ;;
  demo)
    exec python deploy_gui_realsense.py --config "$CONFIG" "$@"
    ;;
  gi-preview)
    exec python deploy_gui_gi.py --config "$CONFIG" "$@"
    ;;
  robot)
    exec python deploy_gui_gi.py \
      --config "$CONFIG" --allow-robot "$@"
    ;;
  *)
    printf '%s\n' \
      'ToolRGS GUI quick launcher' \
      '' \
      'Usage:' \
      '  bash tools/gui_quickstart.sh check       # check and download configured weights' \
      '  bash tools/gui_quickstart.sh demo        # direct RealSense demo; never sends robot data' \
      '  bash tools/gui_quickstart.sh gi-preview  # GI shared video; robot dry-run' \
      '  bash tools/gui_quickstart.sh robot       # GI shared video + real robot TCP output' \
      '' \
      'Extra GUI arguments may follow the action, for example:' \
      '  bash tools/gui_quickstart.sh demo --prompt "the screwdriver"' \
      '' \
      'Override the deployment YAML when needed:' \
      '  TOOLRGS_GUI_CONFIG=config/deployment/lab.yaml bash tools/gui_quickstart.sh check'
    [[ "$ACTION" == "help" ]] || exit 2
    ;;
esac
