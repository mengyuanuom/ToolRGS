"""Launch the old laboratory GI GUI layout with the current ToolRGS kernel."""

from deploy_gui import apply_camera_preset, apply_runtime_overrides, parse_args
from deployment.config import load_deployment_config
from deployment.gui import run_gui


def prepare_legacy_gi_config(config):
    """Apply the validated old-GUI transport and interaction contract."""
    config = apply_camera_preset(config, "gi")
    config["gui"].update(
        {
            "layout": "legacy",
            "title": "Object Detection & Grasping GUI",
            "continuous_inference": True,
            "legacy_send_every_frames": 50,
        }
    )
    # The old GUI used a blocking persistent TCP client. Connection now runs
    # in a background thread so the Qt window remains responsive while waiting.
    config["robot"]["timeout_s"] = None
    return config


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_deployment_config(args.config)
    config = prepare_legacy_gi_config(config)
    config = apply_runtime_overrides(config, args)
    return run_gui(config, allow_robot=args.allow_robot)


if __name__ == "__main__":
    raise SystemExit(main())
