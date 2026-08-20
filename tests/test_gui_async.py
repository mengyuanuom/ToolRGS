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
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QPalette
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QLabel,
        QLineEdit,
        QMessageBox,
    )
except ImportError:  # pragma: no cover - PyQt is an optional deployment extra.
    QApplication = None


@unittest.skipIf(QApplication is None, "PyQt5 is not installed")
class AsyncGuiTest(unittest.TestCase):
    def test_model_switch_keeps_event_loop_responsive_and_styles_popup(self):
        construction_count = 0
        construction_lock = threading.Lock()
        source_reads = 0
        detector_updates = []
        prediction_calls = 0
        prediction_prompts = []

        class FakePrediction:
            annotated_bgr = np.zeros((48, 64, 3), dtype=np.uint8)
            grasps = []

        class FakeInference:
            def __init__(self, _config):
                nonlocal construction_count
                with construction_lock:
                    construction_count += 1
                    current = construction_count
                if current > 1:
                    time.sleep(0.4)

            def predict(self, _frame, prompt):
                nonlocal prediction_calls
                prediction_calls += 1
                prediction_prompts.append(prompt)
                return FakePrediction()

            @staticmethod
            def visualization_maps(_prediction):
                return {}

        class FakeSource:
            def read(self):
                nonlocal source_reads
                source_reads += 1
                return True, np.zeros((48, 64, 3), dtype=np.uint8)

            def close(self):
                pass

        class FakeRobot:
            connected = False

        class FakeDetector:
            def predict(self, frame):
                return frame

            def update_postprocessing(
                self, score_threshold, nms_threshold, max_detections
            ):
                detector_updates.append(
                    (score_threshold, nms_threshold, max_detections)
                )

        inference_module = types.ModuleType("deployment.inference")
        inference_module.GraspPrediction = object
        inference_module.ToolRGSInference = FakeInference
        detector_module = types.ModuleType("deployment.detector")
        detector_module.build_detector = lambda *_args, **_kwargs: FakeDetector()
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
            from deployment.gui import format_grasp_prompt, run_gui

            self.assertEqual(
                format_grasp_prompt("screwdriver", "Grasp {}"),
                "screwdriver",
            )
            self.assertEqual(format_grasp_prompt(""), "")
            self.assertEqual(format_grasp_prompt("Grasp Grasp"), "Grasp Grasp")

            config = {
                "_repo_root": ".",
                "_active_model": "model-a",
                "_model_profiles": {
                    "model-a": {"prompt": "the screwdriver"},
                    "model-b": {"prompt": "the marker"},
                },
                "model": {"prompt": "the screwdriver"},
                "camera": {},
                "detector": {
                    "enabled": True,
                    "score_threshold": 0.7,
                    "nms_threshold": 0.5,
                    "max_detections": 100,
                    "inference_interval_ms": 400,
                },
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
                    "inference_interval_ms": 40,
                    "continuous_inference": True,
                },
            }
            app = QApplication.instance() or QApplication([])
            observations = {}
            qt_errors = []
            original_excepthook = sys.excepthook

            def capture_qt_error(_error_type, value, _traceback):
                qt_errors.append(value)
                app.quit()

            sys.excepthook = capture_qt_error

            def main_window():
                return next(
                    widget
                    for widget in app.topLevelWidgets()
                    if hasattr(widget, "settings_pages")
                )

            def submit_empty_prompt():
                window = main_window()
                prompt = window.findChild(QLineEdit, "GraspPrompt")
                prompt.setText("Grasp Grasp")
                prompt.selectAll()
                QTest.keyClick(prompt, Qt.Key_Delete)
                QTest.keyClick(prompt, Qt.Key_Return)
                observations["prompt_enabled"] = prompt.isEnabled()
                observations["prompt_deleted_to_empty"] = prompt.text()
                observations["empty_status"] = window.status.text()
                observations["calls_after_empty"] = prediction_calls

            def open_model_popup():
                window = main_window()
                theme_selector = window.theme_selector
                self.assertEqual(
                    [
                        theme_selector.itemText(i)
                        for i in range(theme_selector.count())
                    ],
                    [
                        "Midnight Teal",
                        "Ocean Blue",
                        "Violet Night",
                        "Graphite Amber",
                    ],
                )
                original_stylesheet = window.styleSheet()
                theme_selector.setCurrentText("Ocean Blue")
                self.assertNotEqual(window.styleSheet(), original_stylesheet)
                self.assertIn("#5ba8ff", window.styleSheet())
                self.assertEqual(window.settings_pages.currentIndex(), 0)
                self.assertIn("Detection Post-processing", window.settings_button.text())
                window.detection_score_input.setValue(0.8)
                window.detection_nms_input.setValue(0.35)
                window.detection_max_input.setValue(7)
                window._switch_mode(1)
                self.assertEqual(window.settings_pages.currentIndex(), 1)
                self.assertIn(
                    "Grasp Model & Post-processing", window.settings_button.text()
                )
                combo = window.findChild(QComboBox, "ModelSelector")
                self.assertIsNotNone(combo)
                self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["model-a", "model-b"])
                view_palette = combo.view().palette()
                observations["text"] = view_palette.color(QPalette.Text).name()
                observations["base"] = view_palette.color(QPalette.Base).name()
                combo.showPopup()
                observations["reads_at_open"] = source_reads

            def edit_prompt_with_keyboard():
                window = main_window()
                prompt = window.findChild(QLineEdit, "GraspPrompt")
                prompt.setText("screwdriver")
                prompt.setFocus(Qt.OtherFocusReason)
                prompt.setCursorPosition(len(prompt.text()))
                QTest.keyClick(prompt, Qt.Key_Backspace)
                prompt.setSelection(0, 1)
                QTest.keyClick(prompt, Qt.Key_Delete)
                observations["prompt_after_delete"] = prompt.text()

            def capture_stable_prompt():
                window = main_window()
                prompt = window.findChild(QLineEdit, "GraspPrompt")
                observations["stable_prompt"] = prompt.text()

            def choose_second_model():
                window = main_window()
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
                window = main_window()
                combo = window.findChild(QComboBox, "ModelSelector")
                badge = window.findChild(QLabel, "ModelBadge")
                observations["enabled"] = combo.isEnabled()
                observations["badge"] = badge.text()
                window.close()
                app.quit()

            QTimer.singleShot(60, submit_empty_prompt)
            QTimer.singleShot(120, open_model_popup)
            QTimer.singleShot(155, edit_prompt_with_keyboard)
            QTimer.singleShot(225, capture_stable_prompt)
            QTimer.singleShot(250, choose_second_model)
            QTimer.singleShot(330, heartbeat)
            QTimer.singleShot(800, finish)
            try:
                with mock.patch.object(QMessageBox, "critical") as critical:
                    self.assertEqual(run_gui(config), 0)
                    critical.assert_not_called()
            finally:
                sys.excepthook = original_excepthook
            if qt_errors:
                raise qt_errors[0]

        self.assertLess(observations["heartbeat"], 0.25)
        self.assertEqual(
            observations["reads_at_open"], observations["reads_during_popup"]
        )
        self.assertNotEqual(observations["text"], observations["base"])
        self.assertTrue(observations["enabled"])
        self.assertEqual(observations["badge"], "READY")
        self.assertEqual(construction_count, 2)
        self.assertEqual(detector_updates[-1], (0.8, 0.35, 7))
        self.assertEqual(observations["calls_after_empty"], 0)
        self.assertEqual(observations["prompt_deleted_to_empty"], "")
        self.assertGreater(prediction_calls, 0)
        self.assertNotIn("Grasp Grasp", prediction_prompts)
        self.assertEqual(observations["prompt_after_delete"], "crewdrive")
        self.assertEqual(observations["stable_prompt"], "crewdrive")
        self.assertTrue(observations["prompt_enabled"])
        self.assertIn("prompt is empty", observations["empty_status"])


if __name__ == "__main__":
    unittest.main()
