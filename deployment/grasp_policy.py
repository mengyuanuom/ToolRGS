"""Robot-command policies kept separate from raw model predictions."""

import math
from typing import Mapping, Optional, Sequence

import numpy as np


def mask_span_width(
    mask: np.ndarray,
    center_xy: Sequence[float],
    theta_degrees: float,
    step: float = 0.5,
    safety_margin: float = 30.0,
    maximum: Optional[float] = None,
) -> float:
    """Measure the object span through a grasp centre along the opening axis."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"mask_span_width expects a 2-D mask, got {binary.shape}")
    if step <= 0:
        raise ValueError("mask_span_width step must be positive")
    cx, cy = float(center_xy[0]), float(center_xy[1])
    height, width = binary.shape
    if not (0 <= cx < width and 0 <= cy < height):
        raise ValueError(f"grasp centre {(cx, cy)} is outside mask {binary.shape}")

    angle = math.radians(float(theta_degrees))
    ux, uy = math.cos(angle), math.sin(angle)

    def march(sign: float) -> float:
        distance = 0.0
        last_inside = 0.0
        limit = math.hypot(height, width)
        while distance <= limit:
            x = cx + sign * distance * ux
            y = cy + sign * distance * uy
            if x < 0 or x >= width or y < 0 or y >= height:
                break
            ix, iy = int(round(x)), int(round(y))
            if ix < 0 or ix >= width or iy < 0 or iy >= height:
                break
            if not binary[iy, ix]:
                break
            last_inside = distance
            distance += step
        return last_inside

    opening = march(1.0) + march(-1.0) + float(safety_margin)
    if maximum is not None:
        opening = min(opening, float(maximum))
    return max(float(step), float(opening))


def command_width(
    model_width: float,
    mask: np.ndarray,
    center_xy: Sequence[float],
    theta_degrees: float,
    prompt: str,
    cfg: Mapping[str, object],
) -> float:
    """Resolve the configured robot width without mutating model output."""
    policy = str(cfg.get("type", "model")).strip().lower().replace("-", "_")
    if policy == "model":
        return float(model_width)
    if policy != "mask_span":
        raise ValueError("robot.width_policy.type must be model or mask_span")
    from .robot import find_tool_classes

    excluded = {str(item).casefold() for item in cfg.get("exclude", [])}
    if any(name.casefold() in excluded for name in find_tool_classes(prompt)):
        return float(model_width)
    return mask_span_width(
        mask,
        center_xy,
        theta_degrees,
        step=float(cfg.get("step", 0.5)),
        safety_margin=float(cfg.get("safety_margin", 30.0)),
        maximum=cfg.get("maximum"),
    )


def command_theta(theta_degrees: float, cfg: Mapping[str, object]) -> float:
    """Apply the receiver-specific angle offset and normalization convention."""
    theta = float(theta_degrees) * float(cfg.get("sign", 1.0))
    theta += float(cfg.get("offset_degrees", 0.0))
    convention = str(cfg.get("normalization", "signed_90")).strip().lower()
    if convention == "signed_90":
        return (theta + 90.0) % 180.0 - 90.0
    if convention == "zero_180":
        return theta % 180.0
    if convention == "zero_360":
        return theta % 360.0
    if convention == "none":
        return theta
    raise ValueError(
        "robot.theta_policy.normalization must be signed_90, zero_180, "
        "zero_360, or none"
    )
