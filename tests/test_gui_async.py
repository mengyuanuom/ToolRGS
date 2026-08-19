import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
    )
except ImportError:  # pragma: no cover - PyQt is an optional deployment extra.
    QApplication = None


@unittest.skipIf(QApplication is None, "PyQt5 is not installed")
class AsyncGuiTest(unittest.TestCase):
    def test_model_switch_keeps_event_loop_responsive_and_styles_popup(self):
        construction_count = 0
        construction_lock = threading.Lock()
        source_reads = 0
        prediction_calls = 0

        class FakeInference:
            def __init__(self, _config):
                nonlocal construction_count
                with construction_lock:
                    construction_count += 1
                    current = construction_count
                if current > 1:
                    time.sleep(0.4)

            def predict(self, _frame, _prompt):
                nonlocal prediction_calls
                prediction_calls += 1
                raise AssertionError("empty prompt must not reach inference")

        class FakeSource:
            def read(self):
                nonlocal source_reads
                source_reads += 1
                return True, np.zeros((48, 64, 3), dtype=np.uint8)

            def close(self):
                pass

        class FakeRobot:
            connected = False

        inference_module = types.ModuleType("deployment.inference")
        inference_module.GraspPrediction = object
        inference_module.ToolRGSInference = FakeInference
        detector_module = types.ModuleType("deployment.detector")
        detector_module.build_detector = lambda *_args, **_kwargs: None
        audio_module = types.ModuleType("deployment.audio")
        audio_module.build_audio_input = lambda *_args, **_kwargs: None
        gelsight_module = types.ModuleType("deployment.gelsight")
        gelsight_module.build_gelsight = lambda *_args, **_kwargs: None
        policy_module = types.ModuleType("deployment.grasp_policy")
        policy_module.command_theta = lambda value, _cfg: value
        policy_module.command_width = lambda value, *_args: value
        policy_module.mask_span_width = lambda *_args, **_kwargs: 1.0
        qt_module = types.ModuleType("deployment.qt")
        qt_module.configure_pyqt5_plugins = lambda: None
        robot_module = types.ModuleType("deployment.robot")
        robot_module.GraspCommand = object
        robot_module.LegacyTCPGraspClient = FakeRobot
        robot_module.build_robot_client = lambda _cfg: FakeRobot()
        robot_module.find_tool_classes = lambda *_args, **_kwargs: []
        robot_module.semantic_depth = lambda *_args, **_kwargs: 0
        sources_module = types.ModuleType("deployment.sources")
        sources_module.FrameSource = FakeSource
        sources_module.build_source = lambda *_args, **_kwargs: FakeSource()

        replacements = {
            "deployment.inference": inference_module,
            "deployment.detector": detector_module,
            "deployment.audio": audio_module,
            "deployment.gelsight": gelsight_module,
            "deployment.grasp_policy": policy_module,
            "deployment.qt": qt_module,
            "deployment.robot": robot_module,
            "deployment.sources": sources_module,
        }
        sys.modules.pop("deployment.gui", None)
        with mock.patch.dict(sys.modules, replacements):
            from deployment.gui import run_gui

            config = {
                "_repo_root": ".",
                "_active_model": "model-a",
                "_model_profiles": {
                    "model-a": {"prompt": "the screwdriver"},
                    "model-b": {"prompt": "the marker"},
                },
                "model": {"prompt": "the screwdriver"},
                "camera": {},
                "detector": {"enabled": False},
                "audio": {"enabled": False},
                "gelsight": {"enabled": False},
                "robot": {
                    "enabled": False,
                    "auto_connect": False,
                    "auto_send": False,
                },
                "gui": {
                    "title": "ToolRGS GUI test",
                    "window_width": 1200,
                    "window_height": 760,
                    "camera_interval_ms": 20,
                    "inference_interval_ms": 1000,
                    "continuous_inference": False,
                },
            }
            app = QApplication.instance() or QApplication([])
            observations = {}

            def submit_empty_prompt():
                window = next(widget for widget in app.topLevelWidgets() if widget.isVisible())
                prompt = window.findChild(QLineEdit)
                button = window.findChild(QPushButton, "PrimaryButton")
                prompt.setText("   ")
                button.click()
                observations["prompt_enabled"] = prompt.isEnabled()
                observations["prompt_focused"] = prompt.hasFocus()
                observations["empty_status"] = window.status.text()

            def open_model_popup():
                window = next(widget for widget in app.topLevelWidgets() if widget.isVisible())
                combo = window.findChild(QComboBox, "ModelSelector")
                self.assertIsNotNone(combo)
                self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["model-a", "model-b"])
                view_palette = combo.view().palette()
                observations["text"] = view_palette.color(QPalette.Text).name()
                observations["base"] = view_palette.color(QPalette.Base).name()
                combo.showPopup()
                observations["reads_at_open"] = source_reads

            def choose_second_model():
                window = next(widget for widget in app.topLevelWidgets() if widget.isVisible())
                combo = window.findChild(QComboBox, "ModelSelector")
                observations["reads_during_popup"] = source_reads
                observations["switch_started"] = time.monotonic()
                combo.setCurrentIndex(1)
                combo.hidePopup()
                combo.activated[int].emit(1)

            def heartbeat():
                observations["heartbeat"] = (
                    time.monotonic() - observations["switch_started"]
                )

            def finish():
                window = next(widget for widget in app.topLevelWidgets() if widget.isVisible())
                combo = window.findChild(QComboBox, "ModelSelector")
                badge = window.findChild(QLabel, "ModelBadge")
                observations["enabled"] = combo.isEnabled()
                observations["badge"] = badge.text()
                window.close()
                app.quit()

            QTimer.singleShot(60, submit_empty_prompt)
            QTimer.singleShot(120, open_model_popup)
            QTimer.singleShot(250, choose_second_model)
            QTimer.singleShot(330, heartbeat)
            QTimer.singleShot(800, finish)
            with mock.patch.object(QMessageBox, "critical") as critical:
                self.assertEqual(run_gui(config), 0)
                critical.assert_not_called()

        self.assertLess(observations["heartbeat"], 0.25)
        self.assertEqual(
            observations["reads_at_open"], observations["reads_during_popup"]
        )
        self.assertNotEqual(observations["text"], observations["base"])
        self.assertTrue(observations["enabled"])
        self.assertEqual(observations["badge"], "READY")
        self.assertEqual(construction_count, 2)
        self.assertEqual(prediction_calls, 0)
        self.assertTrue(observations["prompt_enabled"])
        self.assertTrue(observations["prompt_focused"])
        self.assertIn("prompt is empty", observations["empty_status"])


if __name__ == "__main__":
    unittest.main()
