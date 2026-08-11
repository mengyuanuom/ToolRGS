"""Base class for new structured-output ToolRGS models."""

from abc import ABC, abstractmethod
import inspect

import torch.nn as nn

from toolrgs.structures import GraspModelResult


class BaseGraspModel(nn.Module, ABC):
    """New models should return :class:`GraspModelResult` from ``forward``."""

    supports_offset = False
    requires_depth = False

    @abstractmethod
    def forward(self, *args, **kwargs) -> GraspModelResult:
        raise NotImplementedError


def model_requires_depth(model) -> bool:
    """Read the input contract through DataParallel/DDP wrappers."""
    module = getattr(model, "module", model)
    return bool(getattr(module, "requires_depth", False))


def dense_grasp_target_kwargs(model, **targets):
    """Map named runner targets onto historical model forward signatures."""
    module = getattr(model, "module", model)
    parameters = inspect.signature(module.forward).parameters
    kwargs = {}
    instance = targets.pop("instance", None)
    if "mask" in parameters:
        kwargs["mask"] = instance
    elif "ins_mask" in parameters:
        kwargs["ins_mask"] = instance
    for name, value in targets.items():
        if name in parameters:
            kwargs[name] = value
    return kwargs
