"""Qt runtime setup for environments where OpenCV and PyQt bundle different Qt builds."""

import os
from pathlib import Path
from typing import Optional


def configure_pyqt5_plugins() -> Optional[Path]:
    """Point Qt at PyQt5's plugins after importing OpenCV.

    The non-headless OpenCV wheel sets QT_QPA_PLATFORM_PLUGIN_PATH to its own
    bundled plugins on Linux. Those plugins are not ABI-compatible with every
    PyQt5 build and can make QApplication abort before Python can raise an
    exception. Resolve the plugin root from the active PyQt5 installation and
    replace only that runtime path.
    """
    override = str(os.environ.get("TOOLRGS_QT_PLUGIN_PATH", "")).strip()
    if override:
        plugin_root = Path(override).expanduser().resolve()
    else:
        try:
            from PyQt5.QtCore import QLibraryInfo
        except ImportError:
            return None
        if hasattr(QLibraryInfo, "path"):
            raw_path = QLibraryInfo.path(QLibraryInfo.PluginsPath)
        else:
            raw_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        plugin_root = Path(raw_path).expanduser().resolve()

    platforms = plugin_root / "platforms"
    if not platforms.is_dir():
        raise RuntimeError(
            "PyQt5 platform plugins directory does not exist: "
            f"{platforms}. Reinstall PyQt5 or set TOOLRGS_QT_PLUGIN_PATH to "
            "the PyQt5 plugin root."
        )
    os.environ["QT_PLUGIN_PATH"] = str(plugin_root)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
    return platforms
