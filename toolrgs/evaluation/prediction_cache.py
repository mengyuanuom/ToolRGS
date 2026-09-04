"""Portable decoded-prediction caches for inference-free grasp rescoring."""

import json
import os
import tempfile

import numpy as np

from toolrgs.evaluation.metrics import GraspThresholdGridMetric
from utils.grasp_eval import calculate_jacquard_from_matches


PREDICTION_CACHE_VERSION = 1


def _flatten_records(records, key, width, dtype):
    offsets = [0]
    values = []
    for record in records:
        value = np.asarray(record.get(key, []), dtype=dtype)
        if value.size:
            value = value.reshape(-1, width)
            values.append(value)
            offsets.append(offsets[-1] + value.shape[0])
        else:
            offsets.append(offsets[-1])
    flattened = (
        np.concatenate(values, axis=0)
        if values
        else np.empty((0, width), dtype=dtype)
    )
    return np.asarray(offsets, dtype=np.int64), flattened


def save_prediction_cache(path, records, metadata=None):
    """Save threshold-independent decoded predictions without pickle data."""

    records = list(records)
    metadata = dict(metadata or {})
    metadata["format_version"] = PREDICTION_CACHE_VERSION
    metadata["num_samples"] = len(records)
    rectangle_offsets, rectangles = _flatten_records(
        records, "rectangles", 5, np.float32
    )
    target_offsets, targets = _flatten_records(records, "targets", 6, np.float32)
    match_offsets, matches = _flatten_records(records, "matches", 3, np.float32)
    arrays = {
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, ensure_ascii=False)
        ),
        "segmentation_iou": np.asarray(
            [record["segmentation_iou"] for record in records], dtype=np.float32
        ),
        "rectangle_offsets": rectangle_offsets,
        "rectangles": rectangles,
        "target_offsets": target_offsets,
        "targets": targets,
        "match_offsets": match_offsets,
        "matches": matches,
        "target_width_cap": np.asarray(
            [record["target_width_cap"] for record in records], dtype=np.float32
        ),
        "target_height": np.asarray(
            [record["target_height"] for record in records], dtype=np.float32
        ),
    }
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=".prediction-cache-", suffix=".npz", dir=os.path.dirname(path), delete=False
    )
    temporary_path = temporary.name
    temporary.close()
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return path


def load_prediction_cache(path):
    path = os.path.abspath(path)
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(cache.pop("metadata_json").item()))
    if int(metadata.get("format_version", -1)) != PREDICTION_CACHE_VERSION:
        raise ValueError(
            f"Unsupported prediction cache version: {metadata.get('format_version')}"
        )
    sample_count = int(cache["segmentation_iou"].shape[0])
    for name in ("rectangle_offsets", "target_offsets", "match_offsets"):
        if cache[name].shape != (sample_count + 1,):
            raise ValueError(f"Invalid {name} shape in prediction cache: {cache[name].shape}")
    for name in (
        "segmentation_iou",
        "rectangles",
        "targets",
        "matches",
        "target_width_cap",
        "target_height",
    ):
        if not np.isfinite(cache[name]).all():
            raise ValueError(f"Prediction cache contains non-finite values in {name}")
    cache["metadata"] = metadata
    cache["path"] = path
    return cache


def score_prediction_cache(
    cache,
    *,
    topk=(1, 5),
    grasp_iou_threshold=0.25,
    grasp_angle_threshold=30.0,
    grasp_iou_thresholds=(0.25, 0.50, 0.75),
    grasp_angle_thresholds=(5.0, 10.0, 20.0, 30.0),
    segmentation_iou_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
):
    """Score a decoded prediction cache without loading a model or CUDA."""

    topk = tuple(int(value) for value in topk)
    cached_max_topk = int(cache["metadata"].get("max_topk", 0))
    if topk and max(topk) > cached_max_topk:
        raise ValueError(
            f"Cache contains top-{cached_max_topk} predictions, requested top-{max(topk)}; "
            "rerun inference with a larger TEST.grasp_topk"
        )
    segmentation_ious = np.asarray(cache["segmentation_iou"], dtype=np.float64)
    count = int(segmentation_ious.size)
    mean_iou = float(segmentation_ious.mean()) if count else 0.0
    precision = {
        f"Pr@{int(round(float(threshold) * 100))}": (
            float((segmentation_ious > float(threshold)).mean()) if count else 0.0
        )
        for threshold in segmentation_iou_thresholds
    }
    base_correct = {value: 0.0 for value in topk}
    grid = GraspThresholdGridMetric(
        iou_thresholds=grasp_iou_thresholds,
        angle_thresholds=grasp_angle_thresholds,
        topk=topk,
    )
    offsets = cache["match_offsets"]
    all_matches = cache["matches"]
    for sample_index in range(count):
        start, end = int(offsets[sample_index]), int(offsets[sample_index + 1])
        matches = all_matches[start:end]
        for value in topk:
            base_correct[value] += calculate_jacquard_from_matches(
                matches,
                value,
                iou_threshold=grasp_iou_threshold,
                angle_threshold=grasp_angle_threshold,
            )
        for iou_threshold, angle_threshold in grid.threshold_pairs:
            for value in topk:
                grid.update(
                    iou_threshold,
                    angle_threshold,
                    value,
                    calculate_jacquard_from_matches(
                        matches,
                        value,
                        iou_threshold=iou_threshold,
                        angle_threshold=angle_threshold,
                    ),
                )
    grid_results = grid.compute()
    return {
        "num_samples": count,
        "iou": mean_iou,
        "precision": precision,
        "j_index": [base_correct[value] / max(1, count) for value in topk],
        "topk": topk,
        "grasp_threshold_grid": grid_results["rows"],
        "msr": grid_results["msr"],
        "metadata": cache["metadata"],
    }


def write_score_summary(path, scores):
    """Write the threshold grid and mSR row in a stable TSV format."""

    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    topk = tuple(scores["topk"])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("iou\tangle\t" + "\t".join(f"J@{k}" for k in topk) + "\n")
        for row in scores["grasp_threshold_grid"]:
            values = "\t".join(
                f"{100.0 * row['values'][k]:.4f}" for k in topk
            )
            handle.write(f"{row['iou']:.2f}\t{row['angle']:.1f}\t{values}\n")
        handle.write(
            "mSR\tmean\t"
            + "\t".join(f"{100.0 * scores['msr'][k]:.4f}" for k in topk)
            + "\n"
        )
    return path
