"""Configuration-driven model registry for ToolRGS."""

from loguru import logger

from .crog import CROG
from .crogoff import CROGOFF
from .drog import DROG
from .drogoff import DROGOFF
from .segmenter import DETRIS


MODEL_REGISTRY = {
    "crog": CROG,
    "crogoff": CROGOFF,
    "detris": DETRIS,
    "drog": DROG,
    "drogoff": DROGOFF,
}


def build_model(cfg):
    name = str(getattr(cfg, "architecture", "drog")).lower()
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}; available: {available}") from exc

    model = model_cls(cfg)
    backbone, head, frozen = [], [], []
    for param_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen.append(parameter)
        elif (
            param_name.startswith("backbone")
            or param_name.startswith("txt_backbone")
            or param_name.startswith("dinov2")
        ):
            backbone.append(parameter)
        else:
            head.append(parameter)

    parameter_groups = [
        {"params": backbone, "initial_lr": cfg.lr_multi * cfg.base_lr},
        {"params": head, "initial_lr": cfg.base_lr},
    ]
    logger.info(
        "Build {}: backbone={}, head={}, frozen={}",
        name,
        sum(p.numel() for p in backbone),
        sum(p.numel() for p in head),
        sum(p.numel() for p in frozen),
    )
    return model, parameter_groups


# Compatibility aliases for imported DETRIS scripts.
def build_segmenter(cfg):
    cfg.architecture = "detris"
    return build_model(cfg)


def build_drog(cfg):
    cfg.architecture = "drog"
    return build_model(cfg)


def build_drogoff(cfg):
    cfg.architecture = "drogoff"
    return build_model(cfg)
