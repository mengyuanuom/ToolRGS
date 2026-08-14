"""Launch the safe RealSense-only ToolRGS demo (no GI and no robot output)."""

from deploy_gui import apply_camera_preset, apply_runtime_overrides, parse_args
from deployment.config import load_deployment_config
from deployment.gui import run_gui


def prepare_realsense_demo_config(config):
    """Open RealSense directly and disable every robot side effect."""
    config = apply_camera_preset(config, "realsense")
    robot = config.setdefault("robot", {})
    robot.update(
        {
            "enabled": False,
            "auto_connect": False,
            "auto_arm": False,
            "auto_send": False,
        }
    )
    return config


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.allow_robot:
        raise ValueError(
            "deploy_gui_realsense.py is a camera-only demo and never sends "
            "robot commands. Use deploy_gui_gi.py --allow-robot for the "
            "laboratory robot workflow."
        )
    config = load_deployment_config(args.config)
    config = prepare_realsense_demo_config(config)
    config = apply_runtime_overrides(config, args)
    return run_gui(config, allow_robot=False)


if __name__ == "__main__":
    raise SystemExit(main())
