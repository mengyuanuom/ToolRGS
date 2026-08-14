"""Launch ToolRGS by opening the RealSense camera directly with pyrealsense2."""

from deploy_gui import main


if __name__ == "__main__":
    raise SystemExit(main(camera_preset="realsense"))
