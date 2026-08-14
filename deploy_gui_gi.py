"""Launch ToolRGS with the lab GI/GStreamer RealSense shared-memory stream."""

from deploy_gui import apply_camera_preset, apply_runtime_overrides, parse_args
from deployment.config import load_deployment_config
from deployment.gui import run_gui


def prepare_gi_config(config):
    """Apply the designed GI transport and model-predicted robot width."""
    config = apply_camera_preset(config, "gi")
    config.setdefault("robot", {}).setdefault("width_policy", {})[
        "type"
    ] = "model"
    return config


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_deployment_config(args.config)
    config = prepare_gi_config(config)
    config = apply_runtime_overrides(config, args)
    return run_gui(config, allow_robot=args.allow_robot)


if __name__ == "__main__":
    raise SystemExit(main())
