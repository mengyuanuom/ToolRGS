"""Display-only grasp geometry and A-palette drawing (OpenCV BGR colors)."""

import math

import cv2
import numpy as np


JAW_COLOR = (180, 0, 128)  # Deep purple #8000B4 (OpenCV BGR)
OPENING_COLOR = (0, 255, 255)  # #FFFF00


def validate_display_padding(value):
    padding = float(value)
    if not math.isfinite(padding) or not 0.0 <= padding <= 300.0:
        raise ValueError("grasp_display_padding_px must be between 0 and 300")
    return padding


def display_grasp_corners(grasp, padding_per_side=20.0):
    """Extend the opening axis in source pixels without changing the grasp."""
    x, y, width, height, theta = map(float, grasp[:5])
    padding = validate_display_padding(padding_per_side)
    return cv2.boxPoints(((x, y), (width + 2.0 * padding, height), theta))


def draw_grasp_overlay(image, grasp, padding_per_side=20.0):
    """Draw yellow opening edges then thicker deep-purple jaws on an image."""
    points = np.rint(display_grasp_corners(grasp, padding_per_side)).astype(np.int32)
    # boxPoints edges 0->1 and 2->3 span height (jaws), even for rotated boxes.
    for start in (1, 3):
        cv2.line(image, tuple(points[start]), tuple(points[(start + 1) % 4]),
                 OPENING_COLOR, 2, cv2.LINE_AA)
    for start in (0, 2):
        cv2.line(image, tuple(points[start]), tuple(points[(start + 1) % 4]),
                 JAW_COLOR, 4, cv2.LINE_AA)
