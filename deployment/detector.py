"""Optional MMDetection adapter used by the server demo's detection tab."""

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from .config import resolve_repo_path
from .weights import ensure_deployment_checkpoint
from toolrgs.registry import DETECTORS


def _trusted_mmengine_checkpoint_context(enabled: bool):
    """Allowlist MMEngine metadata without disabling weights-only loading."""
    if not enabled:
        return nullcontext()
    try:
        import torch
        from mmengine.logging.history_buffer import HistoryBuffer
    except ImportError:
        return nullcontext()
    safe_globals = getattr(
        getattr(torch, "serialization", None), "safe_globals", None
    )
    if safe_globals is None:
        return nullcontext()
    return safe_globals([HistoryBuffer])


class MMDetectionAdapter:
    def __init__(self, cfg: Dict[str, Any], repo_root: str):
        try:
            from mmdet.apis import inference_detector, init_detector
        except ImportError as exc:
            raise RuntimeError(
                "Object detection requires a compatible MMDetection/MMCV installation"
            ) from exc
        config_path = resolve_repo_path(cfg.get("config"), repo_root)
        checkpoint_path = resolve_repo_path(cfg.get("checkpoint"), repo_root)
        if config_path is None or not config_path.is_file():
            raise FileNotFoundError(f"Detector config does not exist: {config_path}")
        if checkpoint_path is None:
            raise FileNotFoundError("Detector checkpoint path is empty")
        checkpoint_path = ensure_deployment_checkpoint(
            checkpoint_path,
            cfg.get("checkpoint_url", ""),
            cfg.get("checkpoint_sha256", ""),
        )
        self.inference_detector = inference_detector
        trusted_checkpoint = bool(cfg.get("trusted_checkpoint", False))
        try:
            with _trusted_mmengine_checkpoint_context(trusted_checkpoint):
                self.model = init_detector(
                    str(config_path),
                    str(checkpoint_path),
                    device=str(cfg.get("device", "cuda:0")),
                )
        except Exception as exc:
            if "Weights only load failed" in str(exc) and not trusted_checkpoint:
                raise RuntimeError(
                    "PyTorch blocked legacy MMEngine metadata in the detector "
                    "checkpoint. If this checkpoint comes from your trusted "
                    "training/server source, set detector.trusted_checkpoint: "
                    "true. Do not enable it for downloaded or unverified files."
                ) from exc
            raise
        self.threshold = float(cfg.get("score_threshold", 0.7))
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("detector.score_threshold must be between 0 and 1")
        self.max_detections = int(cfg.get("max_detections", 100))
        if self.max_detections < 1:
            raise ValueError("detector.max_detections must be at least 1")
        self.box_thickness = max(1, int(cfg.get("box_thickness", 2)))
        self.text_scale = float(cfg.get("text_scale", 0.55))
        dataset_meta = getattr(self.model, "dataset_meta", {}) or {}
        self.classes = list(cfg.get("classes") or dataset_meta.get("classes", []))
        self.palette = list(cfg.get("palette") or dataset_meta.get("palette", []))
        if not self.classes:
            raise ValueError(
                "Detector classes are empty; keep the 13 training classes in deployment YAML"
            )
        bbox_head = getattr(getattr(self.model, "roi_head", None), "bbox_head", None)
        num_classes = getattr(bbox_head, "num_classes", None)
        if num_classes is not None and int(num_classes) != len(self.classes):
            raise ValueError(
                f"Detector checkpoint/config expects {num_classes} classes, "
                f"but deployment YAML defines {len(self.classes)}"
            )
        self.model.dataset_meta = {
            **dataset_meta,
            "classes": tuple(self.classes),
            "palette": self.palette,
        }

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        result = self.inference_detector(self.model, frame_bgr)
        instances = result.pred_instances.cpu()
        scores = instances.scores.numpy()
        boxes = instances.bboxes.numpy()
        labels = instances.labels.numpy()
        output = frame_bgr.copy()
        kept = 0
        for score, box, label in zip(scores, boxes, labels):
            if float(score) < self.threshold:
                continue
            label_index = int(label)
            if label_index < 0 or label_index >= len(self.classes):
                raise ValueError(
                    f"Detector returned class index {label_index}, but only "
                    f"{len(self.classes)} classes are configured"
                )
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            if label_index < len(self.palette):
                color = tuple(int(value) for value in self.palette[label_index])
            else:
                color = (30, 220, 30)
            cv2.rectangle(
                output, (x1, y1), (x2, y2), color, self.box_thickness
            )
            name = self.classes[label_index]
            cv2.putText(
                output,
                f"{name} {float(score):.2f}",
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                color,
                self.box_thickness,
                cv2.LINE_AA,
            )
            kept += 1
            if kept >= self.max_detections:
                break
        return output


DETECTORS.register_module(
    MMDetectionAdapter,
    name="mmdetection",
    aliases=("mmdet", "faster_rcnn"),
)
DETECTOR_REGISTRY = DETECTORS.module_dict


def build_detector(cfg: Dict[str, Any], repo_root: str):
    component_type = cfg.get("type", "mmdetection")
    try:
        detector_class = DETECTORS.require(component_type)
    except KeyError as exc:
        available = ", ".join(sorted(DETECTORS.keys()))
        raise ValueError(
            f"Unknown detector {component_type!r}; available: {available}"
        ) from exc
    return detector_class(cfg, repo_root)
