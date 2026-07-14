"""Runner loops and hooks for ToolRGS, with lazy PyTorch loop imports."""

from .hooks import Hook, HookList, LoopState, NoOpHook

__all__ = [
    "BaseLoop",
    "GraspTrainLoop",
    "Hook",
    "HookList",
    "LoopState",
    "NoOpHook",
]


def __getattr__(name):
    if name in {"BaseLoop", "GraspTrainLoop"}:
        from .loops import BaseLoop, GraspTrainLoop

        return {"BaseLoop": BaseLoop, "GraspTrainLoop": GraspTrainLoop}[name]
    raise AttributeError(name)
