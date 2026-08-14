"""Launch ToolRGS with the lab GI/GStreamer RealSense shared-memory stream."""

from deploy_gui import main


if __name__ == "__main__":
    raise SystemExit(main(camera_preset="gi"))
