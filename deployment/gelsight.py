"""Optional OpenCV GelSight stream and checkpoint-backed classifier."""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .config import resolve_repo_path
from .sources import FrameSource, build_source
from toolrgs.registry import TACTILE_INPUTS


@dataclass(frozen=True)
class GelSightPrediction:
    label: str
    confidence: float
    topk: List[Tuple[str, float]]
    annotated_bgr: np.ndarray


class GelSightClassifier:
    def __init__(self, cfg: Dict[str, Any], repo_root: str):
        try:
            import torch
            from torchvision import models, transforms
        except ImportError as exc:
            raise RuntimeError("GelSight classification requires torch and torchvision") from exc
        self.torch = torch
        self.transforms = transforms
        self.source: FrameSource = None
        requested = str(cfg.get("device", "cuda:0"))
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"GelSight requested {requested}, but CUDA is unavailable")
        self.device = torch.device(requested)
        checkpoint_path = resolve_repo_path(cfg.get("checkpoint"), repo_root)
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(f"GelSight checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(
                str(checkpoint_path), map_location=self.device, weights_only=False
            )
        except TypeError:  # PyTorch < 2.0 has no weights_only argument.
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("GelSight checkpoint must be a dictionary")
        classes = checkpoint.get("classes")
        if not isinstance(classes, dict):
            raise ValueError("GelSight checkpoint requires a classes mapping")
        self.class_names = [
            str(classes.get(index, classes.get(str(index)))) for index in range(len(classes))
        ]
        if any(name == "None" for name in self.class_names):
            raise ValueError("GelSight classes mapping must contain contiguous indices")
        architecture = checkpoint.get("arch", "resnet18")
        if isinstance(architecture, bytes):
            architecture = architecture.decode("utf-8")
        self.model = self._build_model(architecture, len(self.class_names), models).to(
            self.device
        )
        state = checkpoint.get("model", checkpoint.get("state_dict"))
        if not isinstance(state, dict):
            raise ValueError("GelSight checkpoint requires model or state_dict weights")
        cleaned = {
            (key[7:] if str(key).startswith("module.") else key): value
            for key, value in state.items()
        }
        self.model.load_state_dict(cleaned, strict=True)
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.momentum = 0.0
        self.threshold = float(cfg.get("confidence_threshold", 0.90))
        self.nothing_label = str(cfg.get("nothing_label", "Nothing"))
        image_size = int(cfg.get("image_size", 320))
        mean = list(cfg.get("mean", [0.428, 0.524, 0.580]))
        std = list(cfg.get("std", [0.134, 0.057, 0.118]))
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(round(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        # Open hardware only after the checkpoint has been fully validated.
        self.source = build_source(cfg["camera"], repo_root)
        self.last_time = time.monotonic()

    @staticmethod
    def _build_model(architecture, num_classes, models):
        import torch.nn as nn

        name = str(architecture).strip().lower().replace("-", "_")
        if name == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif name == "resnet34":
            model = models.resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif name == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, num_classes
            )
        else:
            raise ValueError(
                "GelSight architecture must be resnet18, resnet34, resnet50, "
                "or efficientnet_b0"
            )
        return model

    def predict(self, topk: int = 3) -> GelSightPrediction:
        ok, frame = self.source.read()
        if not ok or frame is None:
            raise RuntimeError("GelSight camera returned no frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(tensor), dim=1)[0]
        count = min(max(1, int(topk)), len(self.class_names))
        values, indices = probabilities.topk(count)
        ranked = [
            (self.class_names[int(index)], float(value))
            for value, index in zip(values.cpu(), indices.cpu())
        ]
        raw_label, confidence = ranked[0]
        label = raw_label if confidence >= self.threshold else self.nothing_label
        now = time.monotonic()
        fps = 1.0 / max(now - self.last_time, 1e-6)
        self.last_time = now
        annotated = self._draw(frame.copy(), label, confidence, fps)
        return GelSightPrediction(label, confidence, ranked, annotated)

    @staticmethod
    def _draw(frame, label, confidence, fps):
        text = f"GelSight: {label} ({confidence:.2f})  {fps:.1f} FPS"
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 58), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0)
        cv2.putText(
            frame, text, (16, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
            (0, 255, 255), 2, cv2.LINE_AA,
        )
        return frame

    def close(self) -> None:
        if self.source is not None:
            self.source.close()


@TACTILE_INPUTS.register_module(name="classifier", aliases=("gelsight_classifier",))
def _build_gelsight_classifier(cfg, repo_root):
    return GelSightClassifier(cfg, repo_root)


def build_gelsight(cfg: Dict[str, Any], repo_root: str) -> GelSightClassifier:
    component_type = str(cfg.get("type", "classifier")).strip().lower()
    try:
        factory = TACTILE_INPUTS.require(component_type)
    except KeyError as exc:
        available = ", ".join(sorted(TACTILE_INPUTS.keys()))
        raise ValueError(
            f"Unknown GelSight component {component_type!r}; available: {available}"
        ) from exc
    return factory(cfg, repo_root)
