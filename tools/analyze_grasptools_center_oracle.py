"""Analyze CROG center localization from a cached GraspTools test pass.

The controlled GT-center variant replaces only the top-1 prediction center by
the nearest annotated grasp center.  Predicted angle, long side, and short side
remain unchanged.  Both variants are scored with corrected image-bounded
rotated IoU and the Fine-mSR-90 protocol; no model inference is performed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.grasp_raster import GEOMETRY_VERSION, grasp_matches


IOU_THRESHOLDS = tuple(float(value) for value in np.arange(0.25, 0.951, 0.05))
ANGLE_THRESHOLDS = tuple(float(value) for value in np.arange(5.0, 30.1, 5.0))


def _load_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(cache.pop("metadata_json").item()))
    return cache, metadata


def _validate(cache, metadata):
    sample_count = len(cache["segmentation_iou"])
    if metadata.get("split") != "test":
        raise ValueError(f"expected test split, got {metadata.get('split')!r}")
    if metadata.get("size_coordinate") != "original":
        raise ValueError("expected original-coordinate grasp sizes")
    if float(metadata.get("size_factor", -1.0)) != 300.0:
        raise ValueError("expected the audited 300-pixel width factor")
    if not np.all(cache["target_width_cap"] == 300.0):
        raise ValueError("target width cap is not uniformly 300 pixels")
    if not np.all(cache["target_height"] == 20.0):
        raise ValueError("target short side is not uniformly 20 pixels")
    for name in ("rectangle_offsets", "target_offsets"):
        if cache[name].shape != (sample_count + 1,):
            raise ValueError(f"invalid {name} shape")
    for name in ("rectangles", "targets"):
        if not np.isfinite(cache[name]).all():
            raise ValueError(f"non-finite values in {name}")
    return sample_count


def _score_match_blocks(match_blocks, sample_count):
    counts = np.zeros((len(IOU_THRESHOLDS), len(ANGLE_THRESHOLDS)), dtype=np.int64)
    conventional = 0
    iou_thresholds = np.asarray(IOU_THRESHOLDS, dtype=np.float64)
    for matches in match_blocks:
        if not len(matches):
            continue
        conventional += int(
            np.any((matches[:, 1] > 0.25) & (matches[:, 2] <= 30.0))
        )
        for angle_index, angle_threshold in enumerate(ANGLE_THRESHOLDS):
            valid = matches[:, 2] <= angle_threshold
            if np.any(valid):
                counts[:, angle_index] += np.max(matches[valid, 1]) > iou_thresholds
    rates = counts.astype(np.float64) / max(sample_count, 1)
    return {
        "J@1": conventional / max(sample_count, 1),
        "mSR@1": float(rates.mean()),
        "grid": [
            {
                "iou": iou_threshold,
                "angle": angle_threshold,
                "SR@1": float(rates[iou_index, angle_index]),
            }
            for iou_index, iou_threshold in enumerate(IOU_THRESHOLDS)
            for angle_index, angle_threshold in enumerate(ANGLE_THRESHOLDS)
        ],
    }


def analyze_cache(cache, metadata, image_shape=(720, 1280)):
    sample_count = _validate(cache, metadata)
    baseline_matches = []
    corrected_matches = []
    normalized_center_errors = []
    missing_predictions = 0

    for sample_index in range(sample_count):
        prediction_start = int(cache["rectangle_offsets"][sample_index])
        prediction_end = int(cache["rectangle_offsets"][sample_index + 1])
        target_start = int(cache["target_offsets"][sample_index])
        target_end = int(cache["target_offsets"][sample_index + 1])
        predictions = cache["rectangles"][prediction_start:prediction_end]
        targets = cache["targets"][target_start:target_end]

        if not len(predictions) or not len(targets):
            missing_predictions += int(not len(predictions))
            baseline_matches.append(np.empty((0, 3), dtype=np.float32))
            corrected_matches.append(np.empty((0, 3), dtype=np.float32))
            continue

        top1 = np.asarray(predictions[:1], dtype=np.float32)
        distances = np.linalg.norm(targets[:, :2] - top1[0, :2], axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_target = targets[nearest_index]
        normalized_center_errors.append(
            float(distances[nearest_index]) / max(float(nearest_target[2]), 1e-6)
        )

        corrected = top1.copy()
        corrected[0, :2] = nearest_target[:2]
        baseline_matches.append(
            np.asarray(
                grasp_matches(top1, targets, 300.0, 20.0, image_shape),
                dtype=np.float32,
            ).reshape(-1, 3)
        )
        corrected_matches.append(
            np.asarray(
                grasp_matches(corrected, targets, 300.0, 20.0, image_shape),
                dtype=np.float32,
            ).reshape(-1, 3)
        )

    baseline = _score_match_blocks(baseline_matches, sample_count)
    corrected = _score_match_blocks(corrected_matches, sample_count)
    errors = np.asarray(normalized_center_errors, dtype=np.float64)
    return {
        "protocol": "GraspTools Fine-mSR-90 center-localization analysis",
        "geometry_version": GEOMETRY_VERSION,
        "num_samples": sample_count,
        "num_predictions": sample_count - missing_predictions,
        "missing_predictions": missing_predictions,
        "image_hw": list(image_shape),
        "target_width_cap": 300.0,
        "target_height": 20.0,
        "iou_rule": ">",
        "angle_rule": "<=",
        "iou_thresholds": IOU_THRESHOLDS,
        "angle_thresholds": ANGLE_THRESHOLDS,
        "baseline": baseline,
        "gt_center_corrected": corrected,
        "gain_J@1_pp": 100.0 * (corrected["J@1"] - baseline["J@1"]),
        "gain_mSR@1_pp": 100.0 * (corrected["mSR@1"] - baseline["mSR@1"]),
        "center_error_fraction_gt_width": {
            "median": float(np.median(errors)) if len(errors) else None,
            "q1": float(np.quantile(errors, 0.25)) if len(errors) else None,
            "q3": float(np.quantile(errors, 0.75)) if len(errors) else None,
        },
        "metadata": metadata,
    }


def _write_summary(path, result):
    center_error = result["center_error_fraction_gt_width"]
    lines = [
        "variant\tJ@1\tFine-mSR-90@1",
        (
            "CROG baseline\t"
            f"{100.0 * result['baseline']['J@1']:.4f}\t"
            f"{100.0 * result['baseline']['mSR@1']:.4f}"
        ),
        (
            "GT-center corrected\t"
            f"{100.0 * result['gt_center_corrected']['J@1']:.4f}\t"
            f"{100.0 * result['gt_center_corrected']['mSR@1']:.4f}"
        ),
        f"gain_pp\t{result['gain_J@1_pp']:.4f}\t{result['gain_mSR@1_pp']:.4f}",
        (
            "median_center_error_pct_gt_width\t"
            f"{100.0 * center_error['median']:.4f}\t"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--image-shape",
        nargs=2,
        type=int,
        default=(720, 1280),
        metavar=("HEIGHT", "WIDTH"),
    )
    args = parser.parse_args()

    cache, metadata = _load_cache(args.cache)
    result = analyze_cache(cache, metadata, tuple(args.image_shape))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_summary(output / "summary.tsv", result)
    print((output / "summary.tsv").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
