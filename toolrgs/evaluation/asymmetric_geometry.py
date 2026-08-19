"""Dense asymmetric rotated-box targets for language-guided grasping."""

import cv2
import numpy as np

from .geometry import corners_to_five, five_to_corners, rect_to_five


ASYMMETRIC_TARGET_KEYS = (
    "ltrb",
    "centerness",
    "geometry_weight",
    "geometry_sin",
    "geometry_cos",
)


def transform_grasps_to_input(rectangles, inverse_matrix):
    """Map original-image grasp rectangles into the network input canvas."""
    inverse = np.asarray(inverse_matrix, dtype=np.float32).reshape(2, 3)
    forward = cv2.invertAffineTransform(inverse)
    transformed = []
    for rectangle in rectangles:
        corners = five_to_corners(rect_to_five(rectangle))
        homogeneous = np.concatenate(
            [corners, np.ones((4, 1), dtype=np.float32)], axis=1
        )
        transformed.append(corners_to_five(homogeneous @ forward.T))
    return np.asarray(transformed, dtype=np.float32).reshape(-1, 5)


def generate_asymmetric_grasp_targets(
    rectangles,
    image_hw,
    size_factor=100.0,
):
    """Rasterize full-box FCOS targets for rotated grasp rectangles.

    Each positive pixel regresses distances to the left, right, top and bottom
    sides in the grasp-local coordinate frame.  Overlapping candidates are
    assigned to the rectangle with the highest centerness at that pixel.
    """
    height, width = (int(value) for value in image_hw)
    size_factor = float(size_factor)
    if height <= 0 or width <= 0:
        raise ValueError(f"image_hw must be positive, got {image_hw}")
    if size_factor <= 0:
        raise ValueError("size_factor must be positive")

    ltrb = np.zeros((4, height, width), dtype=np.float32)
    centerness = np.zeros((1, height, width), dtype=np.float32)
    geometry_weight = np.zeros((1, height, width), dtype=np.float32)
    sine = np.zeros((1, height, width), dtype=np.float32)
    cosine = np.zeros((1, height, width), dtype=np.float32)

    for rectangle in rectangles:
        center_x, center_y, box_width, box_height, theta_deg = rect_to_five(
            rectangle
        )
        box_width = float(box_width)
        box_height = float(box_height)
        if not np.isfinite(
            [center_x, center_y, box_width, box_height, theta_deg]
        ).all() or box_width <= 0 or box_height <= 0:
            continue

        theta = np.deg2rad(float(theta_deg))
        axis_x = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        axis_y = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
        corners = five_to_corners(
            [center_x, center_y, box_width, box_height, theta_deg]
        )
        minimum_x = max(0, int(np.floor(corners[:, 0].min())))
        maximum_x = min(width - 1, int(np.ceil(corners[:, 0].max())))
        minimum_y = max(0, int(np.floor(corners[:, 1].min())))
        maximum_y = min(height - 1, int(np.ceil(corners[:, 1].max())))
        if minimum_x > maximum_x or minimum_y > maximum_y:
            continue

        columns, rows = np.meshgrid(
            np.arange(minimum_x, maximum_x + 1, dtype=np.float32),
            np.arange(minimum_y, maximum_y + 1, dtype=np.float32),
        )
        delta_x = columns - float(center_x)
        delta_y = rows - float(center_y)
        local_x = delta_x * axis_x[0] + delta_y * axis_x[1]
        local_y = delta_x * axis_y[0] + delta_y * axis_y[1]
        half_width = box_width / 2.0
        half_height = box_height / 2.0
        inside = (
            (np.abs(local_x) <= half_width + 1e-4)
            & (np.abs(local_y) <= half_height + 1e-4)
        )
        if not inside.any():
            continue

        distances = np.stack(
            [
                local_x + half_width,
                half_width - local_x,
                local_y + half_height,
                half_height - local_y,
            ],
            axis=0,
        )
        left, right, top, bottom = np.maximum(distances, 0.0)
        horizontal = np.minimum(left, right) / np.maximum(
            np.maximum(left, right), 1e-6
        )
        vertical = np.minimum(top, bottom) / np.maximum(
            np.maximum(top, bottom), 1e-6
        )
        candidate_centerness = np.sqrt(
            np.clip(horizontal * vertical, 0.0, 1.0)
        ).astype(np.float32)

        row_slice = slice(minimum_y, maximum_y + 1)
        column_slice = slice(minimum_x, maximum_x + 1)
        current_center = centerness[0, row_slice, column_slice]
        current_weight = geometry_weight[0, row_slice, column_slice]
        replace = inside & (
            (current_weight == 0.0) | (candidate_centerness > current_center)
        )
        if not replace.any():
            continue

        target_ltrb = ltrb[:, row_slice, column_slice]
        normalized = distances / size_factor
        for channel in range(4):
            target_ltrb[channel][replace] = normalized[channel][replace]
        current_center[replace] = candidate_centerness[replace]
        current_weight[replace] = 1.0
        double_angle = 2.0 * theta
        sine[0, row_slice, column_slice][replace] = np.sin(double_angle)
        cosine[0, row_slice, column_slice][replace] = np.cos(double_angle)

    return {
        "ltrb": ltrb,
        "centerness": centerness,
        "geometry_weight": geometry_weight,
        "geometry_sin": sine,
        "geometry_cos": cosine,
    }
