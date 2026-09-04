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
    INSTALL_DEPS=0
    INSTALL_DETECTOR=0
    CHECK_ARGS=()
    for argument in "$@"; do
      case "$argument" in
        --install-deps)
          INSTALL_DEPS=1
          ;;
        --with-detector)
          INSTALL_DETECTOR=1
          ;;
        *)
          CHECK_ARGS+=("$argument")
          ;;
      esac
    done
    if [[ "$INSTALL_DEPS" -eq 1 ]]; then
      INSTALL_ARGS=()
      if [[ "$INSTALL_DETECTOR" -eq 1 ]]; then
        INSTALL_ARGS+=(--with-detector)
      fi
      bash tools/install_gui_dependencies.sh "${INSTALL_ARGS[@]}"
    elif [[ "$INSTALL_DETECTOR" -eq 1 ]]; then
      printf '%s\n' '--with-detector requires --install-deps' >&2
      exit 2
    fi
    exec python tools/check_deployment.py \
      --config "$CONFIG" "${CHECK_ARGS[@]}"
    ;;
  install)
    exec bash tools/install_gui_dependencies.sh "$@"
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
      '  bash tools/gui_quickstart.sh check --install-deps  # install GUI/Mamba deps, then check' \
      '  bash tools/gui_quickstart.sh check --install-deps --with-detector  # include MMDetection' \
      '  bash tools/gui_quickstart.sh install     # install dependencies without running preflight' \
      '  bash tools/gui_quickstart.sh demo        # direct RealSense demo; never sends robot data' \
      '  bash tools/gui_quickstart.sh gi-preview  # GI shared video; robot dry-run' \
      '  bash tools/gui_quickstart.sh robot       # GI shared video + real robot TCP output' \
      '' \
      'Extra GUI arguments may follow the action, for example:' \
      '  bash tools/gui_quickstart.sh demo --prompt "Grasp the screwdriver"' \
      '' \
      'Override the deployment YAML when needed:' \
      '  TOOLRGS_GUI_CONFIG=config/deployment/lab.yaml bash tools/gui_quickstart.sh check'
    [[ "$ACTION" == "help" ]] || exit 2
    ;;
esac
