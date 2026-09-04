#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

WITH_DETECTOR=0
for argument in "$@"; do
  case "$argument" in
    --with-detector)
      WITH_DETECTOR=1
      ;;
    *)
      printf 'Unknown installer argument: %s\n' "$argument" >&2
      exit 2
      ;;
  esac
done

PYTHON="$(command -v python)"

"$PYTHON" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "Install the CUDA-matched PyTorch build before ToolRGS dependencies.\n"
        "ToolRGS training was validated with Python 3.9 and PyTorch 2.0.1; "
        "deployment also supports newer PyTorch releases.\n"
        f"Original import error: {exc}"
    )

print(f"[install] Python {sys.version.split()[0]}")
print(f"[install] PyTorch {torch.__version__}; torch CUDA={torch.version.cuda}")
if torch.version.cuda is None:
    raise SystemExit(
        "The active environment has a CPU-only PyTorch build. Install a "
        "CUDA-matched PyTorch wheel before compiling mamba-ssm."
    )
PY

"$PYTHON" -m pip install -r requirement.txt
"$PYTHON" -m pip install -r requirement-deploy.txt

# mamba-ssm must compile against the already installed CUDA/PyTorch build.
"$PYTHON" -m pip install "mamba-ssm==2.2.4" --no-build-isolation
"$PYTHON" -m pip install -r requirement-mamba.txt

if [[ "$WITH_DETECTOR" -eq 1 ]]; then
  "$PYTHON" -m pip install -U openmim
  "$PYTHON" -m mim install "mmengine==0.10.7" "mmcv==2.1.0"
  "$PYTHON" -m pip install -r requirement-detector.txt
fi

printf '%s\n' \
  '[install] ToolRGS GUI dependencies are ready.' \
  '[install] The following preflight will download and verify configured weights.'
