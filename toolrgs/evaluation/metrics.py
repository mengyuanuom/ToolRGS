"""Stateful evaluation metrics independent of models and datasets."""

from typing import Iterable

import numpy as np

from toolrgs.registry import METRICS


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@METRICS.register_module(name="binary_segmentation", aliases=("segmentation_iou",))
class BinarySegmentationMetric:
    """Mean per-sample IoU and precision at configurable IoU thresholds."""

    def __init__(
        self,
        mask_threshold: float = 0.35,
        iou_thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
        from_logits: bool = False,
    ):
        self.mask_threshold = float(mask_threshold)
        self.iou_thresholds = tuple(float(value) for value in iou_thresholds)
        self.from_logits = bool(from_logits)
        self.reset()

    def reset(self):
        self.ious = []

    def update(self, prediction, target):
        prediction = _numpy(prediction).astype(np.float32)
        target = _numpy(target)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Segmentation prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
            )
        if self.from_logits:
            prediction = 1.0 / (1.0 + np.exp(-prediction))
        if prediction.ndim == 2:
            prediction = prediction[None]
            target = target[None]
        prediction = prediction.reshape(prediction.shape[0], -1) > self.mask_threshold
        target = target.reshape(target.shape[0], -1).astype(bool)
        intersection = np.logical_and(prediction, target).sum(axis=1)
        union = np.logical_or(prediction, target).sum(axis=1)
        self.ious.extend((intersection / (union + 1e-6)).tolist())

    def compute(self):
        values = np.asarray(self.ious, dtype=np.float64)
        mean_iou = float(values.mean()) if values.size else 0.0
        precision = {
            f"Pr@{int(round(threshold * 100))}": (
                float((values > threshold).mean()) if values.size else 0.0
            )
            for threshold in self.iou_thresholds
        }
        return {"iou": mean_iou, "precision": precision, "num_samples": int(values.size)}


@METRICS.register_module(name="grasp_success", aliases=("j_index",))
class GraspSuccessMetric:
    """Aggregate binary Jacquard successes for one or more top-k settings."""

    def __init__(self, topk=(1, 5)):
        self.topk = tuple(int(value) for value in topk)
        self.reset()

    def reset(self):
        self.correct = {value: 0.0 for value in self.topk}
        self.total = {value: 0 for value in self.topk}

    def update(self, topk: int, success):
        topk = int(topk)
        if topk not in self.correct:
            raise KeyError(f"top-k {topk} was not configured; available: {self.topk}")
        self.correct[topk] += float(success)
        self.total[topk] += 1

    def compute(self):
        return {
            f"J@{value}": self.correct[value] / max(1, self.total[value])
            for value in self.topk
        }


@METRICS.register_module(name="grasp_threshold_grid")
class GraspThresholdGridMetric:
    """Aggregate top-k grasp success over an IoU/angle threshold grid."""

    def __init__(self, iou_thresholds, angle_thresholds, topk=(1, 5)):
        self.iou_thresholds = tuple(float(value) for value in iou_thresholds)
        self.angle_thresholds = tuple(float(value) for value in angle_thresholds)
        self.topk = tuple(int(value) for value in topk)
        if not self.iou_thresholds or not self.angle_thresholds:
            raise ValueError("The grasp threshold grid cannot be empty")
        self.threshold_pairs = tuple(
            (iou, angle)
            for iou in self.iou_thresholds
            for angle in self.angle_thresholds
        )
        self.reset()

    def reset(self):
        keys = (
            (iou, angle, topk)
            for iou, angle in self.threshold_pairs
            for topk in self.topk
        )
        self.correct = {key: 0.0 for key in keys}
        self.total = {key: 0 for key in self.correct}

    def update(self, iou_threshold, angle_threshold, topk, success):
        key = (float(iou_threshold), float(angle_threshold), int(topk))
        if key not in self.correct:
            raise KeyError(f"Threshold-grid key was not configured: {key}")
        self.correct[key] += float(success)
        self.total[key] += 1

    def compute(self):
        rows = []
        for iou, angle in self.threshold_pairs:
            values = {
                topk: self.correct[(iou, angle, topk)]
                / max(1, self.total[(iou, angle, topk)])
                for topk in self.topk
            }
            rows.append({"iou": iou, "angle": angle, "values": values})
        msr = {
            topk: sum(row["values"][topk] for row in rows) / len(rows)
            for topk in self.topk
        }
        return {"rows": rows, "msr": msr}
