"""Check the Ascend runtime without constructing a ToolRGS model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from toolrgs.runtime import device_name, get_torch_npu, set_device


def main():
    adapter = get_torch_npu()
    device = set_device(0)
    print(f"torch={torch.__version__}")
    print(f"torch_npu={getattr(adapter, '__version__', 'unknown')}")
    print(f"device={device} name={device_name(0)}")
    print(f"HCCL available={torch.distributed.is_available()}")
    left = torch.ones((2, 3), device=device)
    right = torch.ones((3, 2), device=device)
    result = left @ right
    adapter.npu.synchronize()
    print(f"NPU matmul OK: shape={tuple(result.shape)} dtype={result.dtype}")


if __name__ == "__main__":
    main()
