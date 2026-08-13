"""Deployment configuration loading with repository-relative paths."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml


DETECTOR_CLASSES = [
    "box",
    "clamps",
    "clip",
    "crimp tool",
    "hex key",
    "mallet",
    "marker",
    "screwdriver",
    "sponge",
    "spool",
    "tape",
    "tape measure",
    "wrench",
]

DETECTOR_PALETTE = [
    [220, 20, 60],
    [119, 11, 32],
    [0, 0, 142],
    [0, 0, 230],
    [106, 0, 228],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 70],
    [0, 0, 192],
    [250, 170, 30],
    [100, 170, 30],
    [220, 220, 0],
    [175, 116, 175],
]


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "config": "config/grasp_tools/drogoff.yaml",
        "checkpoint": "exp/grasp_tools/drogoff_grasp_tools/best_jindex_model.pth",
        "checkpoint_url": "",
        "checkpoint_sha256": "",
        "device": "cuda:0",
        "prompt": "the tool",
        "mask_threshold": 0.35,
        "quality_threshold": 0.4,
        "num_grasps": 1,
        "postprocessor": {
            "type": "dense_grasp",
            "min_distance": 2,
            "width_factor": 100.0,
            "grasp_height": 20.0,
        },
        "gate_quality_by_mask": True,
        "scale_grasp_to_source": True,
        "overrides": {},
    },
    "camera": {
        "backend": "opencv",
        "device": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "image_path": "",
        "video_path": "",
        "gstreamer_pipeline": "",
    },
    "robot": {
        "type": "legacy_tcp",
        "enabled": False,
        "host": "192.168.38.10",
        "port": 3000,
        "timeout_s": 2.0,
        "auto_connect": False,
        "auto_arm": False,
        "auto_send": False,
        "auto_send_interval_s": 2.0,
        "default_depth": 0,
        "coordinate_space": "source",
        "width_policy": {
            "type": "model",
            "step": 0.5,
            "safety_margin": 30.0,
            "maximum": None,
            "exclude": ["tape", "cable"],
        },
        "theta_policy": {
            "sign": 1.0,
            "offset_degrees": 0.0,
            "normalization": "signed_90",
        },
        "depth_policy": {
            "multiple_matches": "max",
            "class_tiers": {},
        },
        "limits": {
            "x": [0, 1280],
            "y": [0, 720],
            "theta": [-90, 90],
            "width": [1, 600],
            "depth": [-1, 1],
        },
    },
    "detector": {
        "type": "mmdetection",
        "enabled": False,
        "config": "config/deployment/faster-rcnn-13.py",
        "checkpoint": "weights/epoch_48_13.pth",
        "checkpoint_url": "",
        "checkpoint_sha256": "",
        "trusted_checkpoint": False,
        "device": "cuda:0",
        "score_threshold": 0.7,
        "max_detections": 100,
        "inference_interval_ms": 400,
        "box_thickness": 2,
        "text_scale": 0.55,
        "classes": DETECTOR_CLASSES,
        "palette": DETECTOR_PALETTE,
    },
    "audio": {
        "type": "whisper",
        "enabled": False,
        "model": "small",
        "device": "cuda",
        "sample_rate": 16000,
        "duration_s": 4.0,
        "language": "en",
    },
    "gelsight": {
        "type": "classifier",
        "enabled": False,
        "checkpoint": "weights/gelsight_best.pt",
        "device": "cuda:0",
        "image_size": 320,
        "confidence_threshold": 0.90,
        "nothing_label": "Nothing",
        "topk": 3,
        "mean": [0.428, 0.524, 0.580],
        "std": [0.134, 0.057, 0.118],
        "camera": {
            "type": "opencv",
            "device": 1,
            "width": 480,
            "height": 480,
            "fps": 30,
        },
    },
    "gui": {
        "title": "ToolRGS Real-world Grasp Demo",
        "window_width": 1500,
        "window_height": 900,
        "camera_interval_ms": 33,
        "inference_interval_ms": 400,
        "continuous_inference": True,
    },
}


def _deep_merge(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(
    value: Union[str, Path, None], repo_root: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """Resolve a deployment path relative to the ToolRGS repository root."""
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = Path(repo_root).resolve() if repo_root else repository_root()
    return (root / path).resolve()


def load_deployment_config(
    path: Union[str, Path], repo_root: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Deployment config does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Deployment config must contain a YAML mapping: {path}")
    cfg = _deep_merge(DEFAULT_CONFIG, raw)
    profiles = raw.get("model_profiles", {})
    if profiles:
        if not isinstance(profiles, Mapping):
            raise TypeError("model_profiles must be a YAML mapping")
        resolved_profiles = {
            str(name): _deep_merge(DEFAULT_CONFIG["model"], profile)
            for name, profile in profiles.items()
        }
        active = str(raw.get("active_model") or next(iter(resolved_profiles)))
        if active not in resolved_profiles:
            raise KeyError(
                f"active_model {active!r} is not present in model_profiles"
            )
        cfg["model"] = deepcopy(resolved_profiles[active])
        cfg["_model_profiles"] = resolved_profiles
        cfg["_active_model"] = active
    else:
        cfg["_model_profiles"] = {}
        cfg["_active_model"] = "model"
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(
        Path(repo_root).resolve() if repo_root else repository_root()
    )
    return cfg


def activate_model_profile(config: Mapping[str, Any], name: str) -> Dict[str, Any]:
    """Return a deployment config selecting one already validated model profile."""
    profiles = config.get("_model_profiles", {})
    if name not in profiles:
        raise KeyError(f"Unknown deployment model profile: {name}")
    selected = deepcopy(dict(config))
    selected["model"] = deepcopy(profiles[name])
    selected["_active_model"] = str(name)
    return selected
