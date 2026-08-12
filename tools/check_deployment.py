"""Preflight a ToolRGS deployment without sending robot commands."""

import argparse
import importlib.util
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.config import load_deployment_config, resolve_repo_path
from utils.pretrained import ARTIFACTS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/deployment/lab.yaml")
    parser.add_argument(
        "--probe-camera", action="store_true", help="Open the camera and read one frame"
    )
    parser.add_argument(
        "--build-model", action="store_true", help="Load all model weights on the configured device"
    )
    parser.add_argument(
        "--build-detector",
        action="store_true",
        help="Load the optional MMDetection model and its checkpoint",
    )
    parser.add_argument(
        "--probe-gelsight", action="store_true",
        help="Load the GelSight classifier and classify one tactile frame",
    )
    return parser.parse_args()


class Report:
    def __init__(self):
        self.failures = 0

    def ok(self, message):
        print(f"[PASS] {message}")

    def warn(self, message):
        print(f"[WARN] {message}")

    def fail(self, message):
        self.failures += 1
        print(f"[FAIL] {message}")


def require_module(report, module, label=None):
    if importlib.util.find_spec(module) is None:
        report.fail(f"Missing Python package: {label or module}")
    else:
        report.ok(f"Python package available: {label or module}")


def main() -> int:
    args = parse_args()
    report = Report()
    try:
        cfg = load_deployment_config(args.config)
    except Exception as exc:
        report.fail(str(exc))
        return 1

    require_module(report, "torch")
    require_module(report, "cv2", "opencv-python")
    require_module(report, "PyQt5")
    require_module(report, "yaml", "PyYAML")

    model_cfg = cfg["model"]
    experiment = resolve_repo_path(model_cfg["config"], cfg["_repo_root"])
    checkpoint = resolve_repo_path(model_cfg["checkpoint"], cfg["_repo_root"])
    if experiment is not None and experiment.is_file():
        report.ok(f"Experiment config: {experiment}")
    else:
        report.fail(f"Experiment config not found: {experiment}")
    if checkpoint is not None and checkpoint.is_file():
        report.ok(f"Checkpoint: {checkpoint}")
    elif model_cfg.get("checkpoint_url"):
        report.warn(
            f"Checkpoint is missing and will be downloaded on model build: {checkpoint}"
        )
    else:
        report.fail(f"Checkpoint not found: {checkpoint}")

    if experiment is not None and experiment.is_file():
        with experiment.open("r", encoding="utf-8") as stream:
            experiment_yaml = yaml.safe_load(stream) or {}
        flat = {
            key: value
            for section in experiment_yaml.values()
            if isinstance(section, dict)
            for key, value in section.items()
        }
        for key in ("clip_pretrain", "dino_pretrain", "mamba_pretrain"):
            value = flat.get(key)
            if not value or str(value).startswith(("http://", "https://")):
                continue
            path = resolve_repo_path(value, cfg["_repo_root"])
            if path is not None and path.is_file():
                report.ok(f"{key}: {path}")
            elif path is not None and path.name in {
                artifact.filename for artifact in ARTIFACTS.values()
            }:
                report.warn(f"{key} will be downloaded on first model build: {path}")
            else:
                report.fail(f"{key} not found: {path}")

    backend = str(
        cfg["camera"].get("type", cfg["camera"].get("backend", "opencv"))
    ).lower()
    if backend == "realsense":
        require_module(report, "pyrealsense2")
    if backend == "gstreamer":
        require_module(report, "cv2", "OpenCV with GStreamer support")
        report.warn("Confirm cv2.getBuildInformation() reports GStreamer: YES")
    detector_cfg = cfg.get("detector", {})
    if detector_cfg.get("enabled") or args.build_detector:
        require_module(report, "mmdet")
        detector_config = resolve_repo_path(
            detector_cfg.get("config"), cfg["_repo_root"]
        )
        detector_checkpoint = resolve_repo_path(
            detector_cfg.get("checkpoint"), cfg["_repo_root"]
        )
        if detector_config is not None and detector_config.is_file():
            report.ok(f"Detector config: {detector_config}")
        else:
            report.fail(f"Detector config not found: {detector_config}")
        if detector_checkpoint is not None and detector_checkpoint.is_file():
            report.ok(f"Detector checkpoint: {detector_checkpoint}")
        elif detector_cfg.get("checkpoint_url"):
            report.warn(
                "Detector checkpoint is missing and will be downloaded on "
                f"build: {detector_checkpoint}"
            )
        else:
            report.fail(f"Detector checkpoint not found: {detector_checkpoint}")
        classes = list(detector_cfg.get("classes") or [])
        palette = list(detector_cfg.get("palette") or [])
        if len(classes) != 13:
            report.fail(
                f"Detector requires the checkpoint's 13 classes, got {len(classes)}"
            )
        elif len(set(classes)) != len(classes):
            report.fail("Detector class names must be unique")
        else:
            report.ok("Detector class order: 13 classes")
        if palette and len(palette) != len(classes):
            report.fail("Detector palette length must match detector class count")
        threshold = float(detector_cfg.get("score_threshold", 0.7))
        if not 0.0 <= threshold <= 1.0:
            report.fail("Detector score_threshold must be between 0 and 1")
    if cfg.get("audio", {}).get("enabled"):
        require_module(report, "sounddevice")
        require_module(report, "whisper", "openai-whisper")
    gelsight_cfg = cfg.get("gelsight", {})
    if gelsight_cfg.get("enabled"):
        require_module(report, "torchvision")
        gelsight_checkpoint = resolve_repo_path(
            gelsight_cfg.get("checkpoint"), cfg["_repo_root"]
        )
        if gelsight_checkpoint is not None and gelsight_checkpoint.is_file():
            report.ok(f"GelSight checkpoint: {gelsight_checkpoint}")
        else:
            report.fail(f"GelSight checkpoint not found: {gelsight_checkpoint}")

    robot_cfg = cfg["robot"]
    if robot_cfg.get("enabled"):
        report.warn(
            "Robot output is enabled in YAML, but this preflight intentionally does not connect or send"
        )
    else:
        report.ok("Robot output is disabled (safe dry-run state)")
    if str(robot_cfg.get("coordinate_space", "source")).lower() not in {"source", "model"}:
        report.fail("robot.coordinate_space must be source or model")
    for field in ("x", "y", "theta", "width", "depth"):
        bounds = robot_cfg.get("limits", {}).get(field)
        if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] >= bounds[1]:
            report.fail(f"robot.limits.{field} must be an increasing [minimum, maximum] pair")
    if str(robot_cfg.get("width_policy", {}).get("type", "model")) not in {
        "model", "mask_span"
    }:
        report.fail("robot.width_policy.type must be model or mask_span")
    theta_normalization = str(
        robot_cfg.get("theta_policy", {}).get("normalization", "signed_90")
    )
    if theta_normalization not in {"signed_90", "zero_180", "zero_360", "none"}:
        report.fail(
            "robot.theta_policy.normalization must be signed_90, zero_180, "
            "zero_360, or none"
        )
    depth_cfg = robot_cfg.get("depth_policy", {})
    if str(depth_cfg.get("multiple_matches", "max")) not in {"max", "min", "first"}:
        report.fail("robot.depth_policy.multiple_matches must be max, min, or first")
    invalid_tiers = {
        str(name): tier
        for name, tier in dict(depth_cfg.get("class_tiers", {})).items()
        if str(tier).upper() not in {"L1", "L2", "L3"}
    }
    if invalid_tiers:
        report.fail(f"Invalid robot depth tiers: {invalid_tiers}")

    if args.probe_camera and report.failures == 0:
        from deployment.sources import build_source

        source = None
        try:
            source = build_source(cfg["camera"], cfg["_repo_root"])
            ok, frame = source.read()
            if not ok or frame is None:
                raise RuntimeError("camera returned no frame")
            report.ok(f"Camera frame: shape={frame.shape}, dtype={frame.dtype}")
            expected = (
                int(cfg["camera"].get("height", 0)),
                int(cfg["camera"].get("width", 0)),
            )
            if all(expected) and tuple(frame.shape[:2]) != expected:
                report.fail(
                    f"Camera returned {frame.shape[1]}x{frame.shape[0]}, expected "
                    f"{expected[1]}x{expected[0]}"
                )
        except Exception as exc:
            report.fail(f"Camera probe failed: {exc}")
        finally:
            if source is not None:
                source.close()

    if args.build_model and report.failures == 0:
        try:
            from deployment.inference import ToolRGSInference

            ToolRGSInference(cfg)
            report.ok("Model and checkpoint loaded successfully")
        except Exception as exc:
            report.fail(f"Model build failed: {exc}")

    if args.build_detector and report.failures == 0:
        try:
            from deployment.detector import build_detector

            build_detector(detector_cfg, cfg["_repo_root"])
            report.ok("Detector config and checkpoint loaded successfully")
        except Exception as exc:
            report.fail(f"Detector build failed: {exc}")

    if args.probe_gelsight and report.failures == 0:
        if not gelsight_cfg.get("enabled"):
            report.fail("--probe-gelsight requires gelsight.enabled: true")
        else:
            tactile = None
            try:
                from deployment.gelsight import build_gelsight

                tactile = build_gelsight(gelsight_cfg, cfg["_repo_root"])
                prediction = tactile.predict(int(gelsight_cfg.get("topk", 3)))
                report.ok(
                    f"GelSight frame/classification: {prediction.label} "
                    f"({prediction.confidence:.3f})"
                )
            except Exception as exc:
                report.fail(f"GelSight probe failed: {exc}")
            finally:
                if tactile is not None:
                    tactile.close()

    print(f"\nPreflight completed with {report.failures} failure(s).")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
