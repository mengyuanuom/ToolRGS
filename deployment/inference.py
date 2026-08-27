"""Configuration-driven inference for every ToolRGS dense grasp model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import build_model
from utils.config import load_cfg_from_cfg_file
from utils.config import (
    resolve_grasp_quality_activation,
    resolve_grasp_size_activation,
)
from utils.pretrained import ensure_pretrained
from utils.dataset import CLIP_MEAN, CLIP_STD, tokenize
from toolrgs.structures import GraspModelResult
from toolrgs.evaluation import DenseGraspPostProcessor  # registers evaluation components
from toolrgs.registry import POSTPROCESSORS

from .config import resolve_repo_path
from .weights import ensure_deployment_checkpoint


@dataclass
class GraspPrediction:
    prompt: str
    annotated_bgr: np.ndarray
    segmentation: np.ndarray
    quality: np.ndarray
    angle: np.ndarray
    width: np.ndarray
    short_side: Optional[np.ndarray]
    grasps: List[List[float]]
    model_grasps: List[List[float]]
    scores: List[float]


def _load_state(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    cleaned = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=True)


def _input_size(value: Any) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"input_size must have two values: {value}")
        return int(value[0]), int(value[1])
    size = int(value)
    return size, size


def _affine_matrices(
    original_hw: Tuple[int, int], input_hw: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, float]:
    ori_h, ori_w = original_hw
    inp_h, inp_w = input_hw
    scale = min(inp_h / ori_h, inp_w / ori_w)
    new_h, new_w = ori_h * scale, ori_w * scale
    bias_x, bias_y = (inp_w - new_w) / 2.0, (inp_h - new_h) / 2.0
    src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], dtype=np.float32)
    dst = np.array(
        [[bias_x, bias_y], [new_w + bias_x, bias_y], [bias_x, new_h + bias_y]],
        dtype=np.float32,
    )
    return cv2.getAffineTransform(src, dst), cv2.getAffineTransform(dst, src), scale


def _heatmap(value: np.ndarray, color_map: int = cv2.COLORMAP_JET) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    finite = np.nan_to_num(value, copy=True)
    lo, hi = float(finite.min()), float(finite.max())
    normalized = np.zeros_like(finite) if hi - lo < 1e-7 else (finite - lo) / (hi - lo)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), color_map)


def expand_binary_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Expand a binary segmentation mask by approximately ``pixels`` pixels."""
    binary = np.asarray(mask, dtype=np.uint8)
    radius = max(0, int(pixels))
    if radius == 0:
        return binary.astype(bool)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.dilate(binary, kernel, iterations=1).astype(bool)


def opencv_grasp_rectangle(grasp: Sequence[float]):
    """Convert a decoded grasp to the image convention used by the lab GUI."""
    x, y, grasp_width, grasp_height, theta = grasp[:5]
    # The decoded theta already follows the RealSense image coordinates used by
    # the validated 22-class GUI.  Negating it mirrors the displayed rectangle.
    # Adding 180 degrees, as the legacy GUI did, is geometrically equivalent.
    return (
        (float(x), float(y)),
        (float(grasp_width), float(grasp_height)),
        float(theta),
    )


class ToolRGSInference:
    """Load one experiment config/checkpoint and predict pixel-space grasps."""

    def __init__(self, deployment_cfg: Dict[str, Any]):
        self.deployment_cfg = deployment_cfg
        self.model_cfg = deployment_cfg["model"]
        self.repo_root = Path(deployment_cfg["_repo_root"])
        experiment_path = resolve_repo_path(self.model_cfg["config"], self.repo_root)
        checkpoint_path = resolve_repo_path(self.model_cfg["checkpoint"], self.repo_root)
        if experiment_path is None or not experiment_path.is_file():
            raise FileNotFoundError(f"Experiment config does not exist: {experiment_path}")
        if checkpoint_path is None:
            raise ValueError("model.checkpoint cannot be empty")
        checkpoint_path = ensure_deployment_checkpoint(
            checkpoint_path,
            self.model_cfg.get("checkpoint_url", ""),
            self.model_cfg.get("checkpoint_sha256", ""),
        )

        self.cfg = load_cfg_from_cfg_file(str(experiment_path))
        for key, value in dict(self.model_cfg.get("overrides", {})).items():
            setattr(self.cfg, key, value)
        if str(getattr(self.cfg, "architecture", "")).lower() == "detris":
            raise ValueError(
                "DETRIS is a segmentation backbone, not a dense grasp model; "
                "select a ToolRGS grasp architecture"
            )
        self._resolve_pretrained_paths()
        self.device = torch.device(str(self.model_cfg.get("device", "cuda:0")))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({self.device}) but is not available")

        self.cfg.gpu = self.device.index or 0
        self.cfg.rank = 0
        self.model, _ = build_model(self.cfg)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.grasp_size_activation = resolve_grasp_size_activation(
            self.model_cfg.get(
                "grasp_size_activation",
                getattr(self.cfg, "grasp_size_activation", "auto"),
            ),
            checkpoint=checkpoint,
            model=self.model,
        )
        self.grasp_quality_activation = resolve_grasp_quality_activation(
            self.model_cfg.get(
                "grasp_quality_activation",
                getattr(self.cfg, "grasp_quality_activation", "auto"),
            ),
            checkpoint=checkpoint,
            model=self.model,
        )
        if isinstance(checkpoint, dict):
            state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        else:
            state = checkpoint
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint payload in {checkpoint_path}")
        _load_state(self.model, state)
        self.model.to(self.device).eval()
        self.input_hw = _input_size(self.cfg.input_size)
        self._build_postprocessor()

    def _build_postprocessor(self) -> None:
        postprocessor_cfg = dict(self.model_cfg.get("postprocessor", {}))
        postprocessor_cfg.setdefault("type", "dense_grasp")
        postprocessor_cfg.setdefault(
            "quality_threshold", float(self.model_cfg.get("quality_threshold", 0.4))
        )
        postprocessor_cfg.setdefault(
            "num_grasps", int(self.model_cfg.get("num_grasps", 1))
        )
        self.postprocessor = POSTPROCESSORS.build(postprocessor_cfg)

    def update_postprocessing(
        self,
        *,
        grasp_height: float,
        use_mask: bool,
        mask_threshold: float,
        mask_expand_px: int,
        filter_grasps_by_mask: bool,
    ) -> None:
        """Apply GUI post-processing controls without reloading model weights."""
        threshold = float(mask_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("mask_threshold must be between 0 and 1")
        height = float(grasp_height)
        if height <= 0:
            raise ValueError("grasp_height must be positive")
        expand_px = int(mask_expand_px)
        if expand_px < 0:
            raise ValueError("mask_expand_px cannot be negative")

        postprocessor_cfg = self.model_cfg.setdefault("postprocessor", {})
        postprocessor_cfg["grasp_height"] = height
        self.model_cfg["use_mask_postprocessing"] = bool(use_mask)
        self.model_cfg["mask_threshold"] = threshold
        self.model_cfg["mask_expand_px"] = expand_px
        self.model_cfg["filter_grasps_by_mask"] = bool(filter_grasps_by_mask)
        # Preserve compatibility with older deployment YAML files.
        self.model_cfg["gate_quality_by_mask"] = bool(filter_grasps_by_mask)
        self._build_postprocessor()

    def _resolve_pretrained_paths(self) -> None:
        for key in (
            "clip_pretrain", "dino_pretrain", "mamba_pretrain", "depth_pretrain"
        ):
            value = getattr(self.cfg, key, None)
            if not value or str(value).startswith(("http://", "https://")):
                continue
            path = resolve_repo_path(value, self.repo_root)
            setattr(self.cfg, key, str(ensure_pretrained(path)))

    def _preprocess(
        self, frame_bgr: np.ndarray, prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, float]:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image with three channels")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        matrix, inverse, scale = _affine_matrices(rgb.shape[:2], self.input_hw)
        inp_h, inp_w = self.input_hw
        rgb = cv2.warpAffine(
            rgb,
            matrix,
            (inp_w, inp_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0)
        mean = torch.as_tensor(CLIP_MEAN).view(3, 1, 1)
        std = torch.as_tensor(CLIP_STD).view(3, 1, 1)
        tensor = tensor.sub_(mean).div_(std).unsqueeze(0).to(self.device)
        words = tokenize(
            prompt,
            context_length=int(self.cfg.word_len),
            truncate=True,
        ).to(self.device)
        return tensor, words, matrix, inverse, scale

    def _map_to_original(
        self,
        prediction: torch.Tensor,
        inverse: np.ndarray,
        output_hw: Tuple[int, int],
        activation: str = "none",
    ) -> np.ndarray:
        inp_h, inp_w = self.input_hw
        ori_h, ori_w = output_hw
        resized = F.interpolate(
            prediction,
            size=(inp_h, inp_w),
            mode="bilinear" if prediction.shape[1] > 1 else "bicubic",
            align_corners=False,
        )
        if activation == "sigmoid":
            resized = torch.sigmoid(resized)
        elif activation == "quality":
            resized = (
                torch.sigmoid(resized)
                if self.grasp_quality_activation == "sigmoid"
                else resized.clamp(0.0, 1.0)
            )
        elif activation == "size":
            resized = (
                torch.sigmoid(resized)
                if self.grasp_size_activation == "sigmoid"
                else resized.clamp(0.0, 1.0)
            )
        array = resized[0].detach().float().cpu().numpy()
        if array.shape[0] == 1:
            return cv2.warpAffine(
                array[0], inverse, (ori_w, ori_h),
                flags=cv2.INTER_LINEAR, borderValue=0.0,
            )
        channels = [
            cv2.warpAffine(
                channel, inverse, (ori_w, ori_h),
                flags=cv2.INTER_LINEAR, borderValue=0.0,
            )
            for channel in array
        ]
        return np.stack(channels, axis=0)

    def predict(self, frame_bgr: np.ndarray, prompt: str) -> GraspPrediction:
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("The language prompt cannot be empty")
        image, words, matrix, inverse, scale = self._preprocess(frame_bgr, prompt)
        with torch.inference_mode():
            raw = self.model(image, words)
        predictions = GraspModelResult.from_legacy(
            raw, model=self.model
        ).predictions
        segmentation = self._map_to_original(
            predictions.segmentation, inverse, frame_bgr.shape[:2], "sigmoid"
        )
        quality = self._map_to_original(
            predictions.quality, inverse, frame_bgr.shape[:2], "quality"
        )
        sine = self._map_to_original(
            predictions.sine, inverse, frame_bgr.shape[:2]
        )
        cosine = self._map_to_original(
            predictions.cosine, inverse, frame_bgr.shape[:2]
        )
        width = self._map_to_original(
            predictions.width, inverse, frame_bgr.shape[:2], "size"
        )
        short_side = (
            self._map_to_original(
                predictions.short_side, inverse, frame_bgr.shape[:2], "size"
            )
            if predictions.short_side is not None else None
        )
        mask = expand_binary_mask(
            segmentation >= float(self.model_cfg.get("mask_threshold", 0.35)),
            int(self.model_cfg.get("mask_expand_px", 0)),
        )
        use_mask = bool(self.model_cfg.get("use_mask_postprocessing", True))
        filter_by_mask = bool(
            self.model_cfg.get(
                "filter_grasps_by_mask",
                self.model_cfg.get("gate_quality_by_mask", True),
            )
        )
        if use_mask and filter_by_mask:
            quality = quality * mask.astype(np.float32)
        angle = np.arctan2(sine, cosine) / 2.0

        source_scale = 1.0 / max(scale, 1e-8) if self.model_cfg.get(
            "scale_grasp_to_source", True
        ) else 1.0
        detections = self.postprocessor(
            quality,
            sine,
            cosine,
            width,
            short_side=short_side,
            spatial_scale=source_scale,
        )
        grasps: List[List[float]] = []
        model_grasps: List[List[float]] = []
        scores: List[float] = []
        offset = (
            self._map_to_original(
                predictions.offset, inverse, frame_bgr.shape[:2]
            )
            if predictions.offset is not None else None
        )
        radius = float(getattr(self.cfg, "offset_r", 0.0) or 0.0) * source_scale
        ori_h, ori_w = frame_bgr.shape[:2]
        for detection in detections:
            row, col = detection.row, detection.column
            x, y = detection.x, detection.y
            if offset is not None and offset.shape[0] >= 2 and radius > 0:
                x += float(offset[0, row, col]) * radius
                y += float(offset[1, row, col]) * radius
            x = float(np.clip(x, 0, ori_w - 1))
            y = float(np.clip(y, 0, ori_h - 1))
            if (
                use_mask
                and filter_by_mask
                and not mask[int(round(y)), int(round(x))]
            ):
                continue
            theta = detection.angle_degrees
            grasp_width = detection.width
            grasp_height = detection.height
            grasps.append([x, y, grasp_width, grasp_height, theta])
            model_x, model_y = matrix @ np.array([x, y, 1.0], dtype=np.float32)
            model_grasps.append(
                [
                    float(model_x),
                    float(model_y),
                    grasp_width / source_scale,
                    grasp_height / source_scale,
                    theta,
                ]
            )
            scores.append(detection.score)

        annotated = frame_bgr.copy()
        if use_mask:
            overlay = annotated.copy()
            overlay[mask] = (35, 180, 35)
            annotated = cv2.addWeighted(annotated, 0.72, overlay, 0.28, 0.0)
        for index, grasp in enumerate(grasps):
            x, y, grasp_width, grasp_height, theta = grasp
            rectangle = opencv_grasp_rectangle(grasp)
            points = np.intp(cv2.boxPoints(rectangle))
            cv2.polylines(annotated, [points], True, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(annotated, (round(x), round(y)), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"{index + 1}: {scores[index]:.2f}",
                (round(x) + 8, round(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return GraspPrediction(
            prompt=prompt,
            annotated_bgr=annotated,
            segmentation=mask.astype(np.uint8),
            quality=quality,
            angle=angle,
            width=width,
            short_side=short_side,
            grasps=grasps,
            model_grasps=model_grasps,
            scores=scores,
        )

    @staticmethod
    def visualization_maps(prediction: GraspPrediction) -> Dict[str, np.ndarray]:
        maps = {
            "segmentation": prediction.segmentation.astype(np.uint8) * 255,
            "quality": _heatmap(prediction.quality),
            "angle": _heatmap(prediction.angle, cv2.COLORMAP_HSV),
            "width": _heatmap(prediction.width),
        }
        if prediction.short_side is not None:
            maps["short_side"] = _heatmap(prediction.short_side)
        return maps
