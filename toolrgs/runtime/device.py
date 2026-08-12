"""Explicit PyTorch Ascend integration without CUDA API monkey-patching."""

from contextlib import nullcontext
from typing import Any, Optional

import torch

try:
    import torch_npu  # type: ignore
except Exception as exc:
    torch_npu = None
    _TORCH_NPU_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None


def get_torch_npu():
    if torch_npu is None:
        raise RuntimeError(
            "torch_npu is unavailable. Install a torch/torch_npu pair matching "
            "the server CANN version and source the CANN set_env.sh first."
        ) from _TORCH_NPU_IMPORT_ERROR
    return torch_npu


def is_npu_available() -> bool:
    if torch_npu is None:
        return False
    try:
        return bool(torch_npu.npu.is_available())
    except Exception:
        return False


def require_npu() -> None:
    adapter = get_torch_npu()
    if not adapter.npu.is_available():
        raise RuntimeError(
            "torch_npu imported, but no Ascend NPU is available. Check "
            "npu-smi info, ASCEND_RT_VISIBLE_DEVICES and the CANN environment."
        )


def set_device(index: int = 0) -> torch.device:
    require_npu()
    device = torch.device(f"npu:{int(index)}")
    torch_npu.npu.set_device(device)
    return device


def current_device(index: Optional[int] = None) -> torch.device:
    require_npu()
    if index is None:
        index = int(torch_npu.npu.current_device())
    return torch.device(f"npu:{int(index)}")


def device_name(index: int = 0) -> str:
    require_npu()
    return str(torch_npu.npu.get_device_name(int(index)))


def move_to_device(value: Any, device: torch.device, non_blocking: bool = True):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {
            key: move_to_device(item, device, non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking) for item in value]
    return value


def autocast(enabled: bool = False):
    if not enabled:
        return nullcontext()
    return get_torch_npu().npu.amp.autocast(enabled=True)


class NoOpGradScaler:
    """FP32 optimizer adapter with the GradScaler interface."""

    enabled = False

    @staticmethod
    def scale(loss):
        return loss

    @staticmethod
    def unscale_(optimizer):
        return None

    @staticmethod
    def step(optimizer):
        return optimizer.step()

    @staticmethod
    def update():
        return None


def build_grad_scaler(enabled: bool = False):
    if not enabled:
        return NoOpGradScaler()
    return get_torch_npu().npu.amp.GradScaler(enabled=True)


def build_optimizer(parameters, cfg):
    name = str(getattr(cfg, "optimizer", "adam")).lower()
    kwargs = {
        "lr": float(cfg.base_lr),
        "weight_decay": float(cfg.weight_decay),
    }
    if name in {"npu_fused_adam", "fused_adam"}:
        fused_adam = getattr(getattr(get_torch_npu(), "optim", None), "NpuFusedAdam", None)
        if fused_adam is None:
            raise RuntimeError(
                "This torch_npu build has no NpuFusedAdam; use TRAIN.optimizer=adam."
            )
        return fused_adam(parameters, **kwargs)
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    raise ValueError("TRAIN.optimizer must be 'adam' or 'npu_fused_adam'")


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch_npu is not None:
        torch_npu.npu.manual_seed_all(seed)
