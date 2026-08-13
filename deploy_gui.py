"""Launch the ToolRGS real-world grasp GUI."""

import argparse
from pathlib import Path

from deployment.config import load_deployment_config
from deployment.gui import run_gui


DEFAULT_SAMPLE_IMAGE = "assets/grasp_tools/graspall/000000000000.jpg"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/deployment/lab.yaml",
        help="Deployment YAML (default: ready-to-run grasp + detector lab profile)",
    )
    parser.add_argument(
        "--allow-robot",
        action="store_true",
        help="Permit the GUI to connect to a robot receiver when robot.enabled is true",
    )
    parser.add_argument(
        "--image",
        nargs="?",
        const=DEFAULT_SAMPLE_IMAGE,
        default=None,
        metavar="PATH",
        help=(
            "Run without a camera using one image. When PATH is omitted, use the "
            f"repository sample {DEFAULT_SAMPLE_IMAGE}."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override the initial language instruction shown in the GUI.",
    )
    return parser.parse_args(argv)


def apply_runtime_overrides(config, args):
    """Apply safe, temporary CLI overrides without rewriting the YAML file."""
    if args.image is not None:
        image_path = Path(args.image).expanduser()
        if not image_path.is_absolute():
            image_path = Path(config["_repo_root"]) / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"GUI sample image does not exist: {image_path}")
        config["camera"].update(
            {
                "type": "image",
                "backend": "image",
                "image_path": str(image_path),
            }
        )
        # A static image should be inferred on demand instead of every 400 ms.
        config["gui"]["continuous_inference"] = False
    if args.prompt is not None:
        config["model"]["prompt"] = args.prompt
    return config


def main() -> int:
    args = parse_args()
    config = load_deployment_config(args.config)
    config = apply_runtime_overrides(config, args)
    return run_gui(config, allow_robot=args.allow_robot)


if __name__ == "__main__":
    raise SystemExit(main())
