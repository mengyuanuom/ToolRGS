import numpy as np

from tools.analyze_grasptools_center_oracle import analyze_cache


def test_center_replacement_changes_only_localization_and_keeps_missing_failures():
    cache = {
        "segmentation_iou": np.zeros(2, dtype=np.float32),
        "rectangle_offsets": np.asarray([0, 1, 1], dtype=np.int64),
        "rectangles": np.asarray([[0, 10, 10, 20, 0]], dtype=np.float32),
        "target_offsets": np.asarray([0, 1, 2], dtype=np.int64),
        "targets": np.asarray(
            [[8, 10, 10, 20, 0, 0], [8, 10, 10, 20, 0, 0]],
            dtype=np.float32,
        ),
        "target_width_cap": np.full(2, 300, dtype=np.float32),
        "target_height": np.full(2, 20, dtype=np.float32),
    }
    metadata = {
        "split": "test",
        "size_coordinate": "original",
        "size_factor": 300,
    }

    result = analyze_cache(cache, metadata, image_shape=(40, 40))

    assert result["baseline"]["J@1"] == 0.0
    assert result["gt_center_corrected"]["J@1"] == 0.5
    assert result["gt_center_corrected"]["mSR@1"] == 0.5
    assert result["center_error_fraction_gt_width"]["median"] == 0.8
    assert result["missing_predictions"] == 1
