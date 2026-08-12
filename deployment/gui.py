"""PyQt5 GUI for camera, ToolRGS inference, and explicitly armed robot output."""

import sys
import time
import queue
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .inference import GraspPrediction, ToolRGSInference
from .config import activate_model_profile
from .detector import build_detector
from .audio import build_audio_input
from .gelsight import build_gelsight
from .grasp_policy import command_theta, command_width
from .robot import GraspCommand, LegacyTCPGraspClient, build_robot_client, semantic_depth
from .sources import FrameSource, build_source


def run_gui(config: Dict[str, Any], allow_robot: bool = False) -> int:
    try:
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The deployment GUI requires PyQt5; install requirement-deploy.txt"
        ) from exc

    inference = ToolRGSInference(config)
    detector = (
        build_detector(config["detector"], config["_repo_root"])
        if config.get("detector", {}).get("enabled")
        else None
    )
    source = build_source(config["camera"], config["_repo_root"])
    robot_cfg = config["robot"]
    gui_cfg = config["gui"]
    audio = build_audio_input(config["audio"]) if config.get("audio", {}).get("enabled") else None
    gelsight_cfg = config.get("gelsight", {})
    gelsight = (
        build_gelsight(gelsight_cfg, config["_repo_root"])
        if gelsight_cfg.get("enabled") else None
    )

    class MainWindow(QMainWindow):
        def __init__(self, frame_source: FrameSource):
            super().__init__()
            self.source = frame_source
            self.current_frame: Optional[np.ndarray] = None
            self.prediction: Optional[GraspPrediction] = None
            self.last_inference_at = 0.0
            self.last_detection_at = 0.0
            self.last_send_at = 0.0
            self.last_gelsight_at = 0.0
            self.gelsight_available = gelsight is not None
            self.inference_busy = False
            self.inference = inference
            self.active_model = str(config.get("_active_model", "model"))
            self.audio_results = queue.Queue()
            self.robot: Optional[LegacyTCPGraspClient] = None
            self.setWindowTitle(str(gui_cfg["title"]))
            self.resize(int(gui_cfg["window_width"]), int(gui_cfg["window_height"]))
            self._build_ui()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._next_frame)
            self.timer.start(int(gui_cfg["camera_interval_ms"]))

        def _build_ui(self) -> None:
            root = QWidget(self)
            layout = QVBoxLayout(root)

            prompt_row = QHBoxLayout()
            profiles = config.get("_model_profiles", {})
            if profiles:
                prompt_row.addWidget(QLabel("Model:"))
                self.model_selector = QComboBox()
                self.model_selector.addItems(list(profiles))
                self.model_selector.setCurrentText(self.active_model)
                self.model_selector.currentTextChanged.connect(
                    self._change_model
                )
                prompt_row.addWidget(self.model_selector)
            else:
                self.model_selector = None
            prompt_row.addWidget(QLabel("Language instruction:"))
            self.prompt = QLineEdit(str(config["model"]["prompt"]))
            self.prompt.returnPressed.connect(self._predict_now)
            prompt_row.addWidget(self.prompt, 1)
            self.predict_button = QPushButton("Predict now")
            self.predict_button.clicked.connect(self._predict_now)
            prompt_row.addWidget(self.predict_button)
            self.audio_button = QPushButton("Record instruction")
            self.audio_button.clicked.connect(self._record_instruction)
            self.audio_button.setVisible(audio is not None)
            prompt_row.addWidget(self.audio_button)
            self.continuous = QCheckBox("Continuous")
            self.continuous.setChecked(bool(gui_cfg["continuous_inference"]))
            prompt_row.addWidget(self.continuous)
            layout.addLayout(prompt_row)

            self.tabs = QTabWidget()
            self.live_label = self._image_label("Waiting for camera frame")
            self.tabs.addTab(self.live_label, "Live grasp")
            self.detector_tab_index = -1
            self.detection_label = None
            if detector is not None:
                self.detection_label = self._image_label("Waiting for detector")
                self.detector_tab_index = self.tabs.addTab(
                    self.detection_label, "Object detection"
                )
            maps_page = QWidget()
            maps_layout = QGridLayout(maps_page)
            self.map_labels = {}
            for index, name in enumerate(
                ("segmentation", "quality", "angle", "width", "short_side")
            ):
                label = self._image_label(name)
                maps_layout.addWidget(label, index // 2, index % 2)
                self.map_labels[name] = label
            self.tabs.addTab(maps_page, "Dense maps")
            self.gelsight_tab_index = -1
            self.gelsight_label = None
            if gelsight is not None:
                self.gelsight_label = self._image_label("Waiting for GelSight frame")
                self.gelsight_tab_index = self.tabs.addTab(
                    self.gelsight_label, "GelSight"
                )
            layout.addWidget(self.tabs, 1)

            robot_row = QHBoxLayout()
            self.connect_button = QPushButton("Connect receiver")
            self.connect_button.clicked.connect(self._connect_robot)
            robot_allowed = bool(robot_cfg.get("enabled")) and allow_robot
            self.connect_button.setEnabled(robot_allowed)
            robot_row.addWidget(self.connect_button)
            self.disconnect_button = QPushButton("Disconnect receiver")
            self.disconnect_button.clicked.connect(self._disconnect_robot)
            self.disconnect_button.setEnabled(False)
            robot_row.addWidget(self.disconnect_button)
            self.arm = QCheckBox("Arm robot output")
            self.arm.setEnabled(False)
            self.arm.toggled.connect(self._refresh_send_state)
            robot_row.addWidget(self.arm)
            self.send_button = QPushButton("Send current grasp")
            self.send_button.clicked.connect(self._send_grasp)
            self.send_button.setEnabled(False)
            robot_row.addWidget(self.send_button)
            self.status = QLabel()
            if robot_cfg.get("enabled") and not allow_robot:
                self.status.setText("DRY RUN: restart with --allow-robot to permit a connection")
            elif not robot_cfg.get("enabled"):
                self.status.setText("DRY RUN: robot.enabled is false")
            else:
                self.status.setText("Robot output permitted but disconnected")
            robot_row.addWidget(self.status, 1)
            layout.addLayout(robot_row)
            self.setCentralWidget(root)

        def _change_model(self, name: str) -> None:
            name = str(name)
            if not name or name == self.active_model or self.inference_busy:
                return
            self.inference_busy = True
            self.status.setText(f"Loading model profile: {name} ...")
            QApplication.processEvents()
            try:
                selected = activate_model_profile(config, name)
                self.inference = ToolRGSInference(selected)
                self.active_model = name
                self.prompt.setText(str(selected["model"].get("prompt", "")))
                self.prediction = None
                self.status.setText(f"Loaded model profile: {name}")
            except Exception as exc:
                if self.model_selector is not None:
                    self.model_selector.blockSignals(True)
                    self.model_selector.setCurrentText(self.active_model)
                    self.model_selector.blockSignals(False)
                self._error("Model loading error", exc)
            finally:
                self.inference_busy = False

        @staticmethod
        def _image_label(text: str):
            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(320, 240)
            label.setStyleSheet("background: #181818; color: #dddddd;")
            return label

        @staticmethod
        def _pixmap(image: np.ndarray, label: QLabel) -> QPixmap:
            if image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            height, width = rgb.shape[:2]
            qimage = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(qimage).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        def _next_frame(self) -> None:
            self._poll_audio()
            try:
                ok, frame = self.source.read()
            except Exception as exc:
                self.timer.stop()
                self._error("Camera error", exc)
                return
            if not ok or frame is None:
                return
            self.current_frame = frame
            display = self.prediction.annotated_bgr if self.prediction is not None else frame
            self.live_label.setPixmap(self._pixmap(display, self.live_label))
            interval_s = int(gui_cfg["inference_interval_ms"]) / 1000.0
            now = time.monotonic()
            if (
                self.gelsight_available
                and self.tabs.currentIndex() == self.gelsight_tab_index
                and now - self.last_gelsight_at >= interval_s
            ):
                try:
                    tactile = gelsight.predict(int(gelsight_cfg.get("topk", 3)))
                    self.gelsight_label.setPixmap(
                        self._pixmap(tactile.annotated_bgr, self.gelsight_label)
                    )
                    self.status.setText(
                        f"GelSight: {tactile.label} ({tactile.confidence:.2f})"
                    )
                    self.last_gelsight_at = now
                except Exception as exc:
                    self.gelsight_available = False
                    gelsight.close()
                    self._error("GelSight error", exc)
            elif (
                detector is not None
                and self.tabs.currentIndex() == self.detector_tab_index
                and now - self.last_detection_at >= interval_s
            ):
                try:
                    detected = detector.predict(frame)
                    self.detection_label.setPixmap(self._pixmap(detected, self.detection_label))
                    self.last_detection_at = now
                except Exception as exc:
                    self._error("Detection error", exc)
            elif (
                self.tabs.currentIndex() == 0
                and self.continuous.isChecked()
                and now - self.last_inference_at >= interval_s
            ):
                self._predict_now()

        def _record_instruction(self) -> None:
            if audio is None:
                return
            self.audio_button.setEnabled(False)
            self.status.setText("Recording and transcribing instruction...")

            def worker():
                try:
                    self.audio_results.put((True, audio.transcribe_once()))
                except Exception as exc:
                    self.audio_results.put((False, exc))

            threading.Thread(target=worker, daemon=True).start()

        def _poll_audio(self) -> None:
            try:
                ok, value = self.audio_results.get_nowait()
            except queue.Empty:
                return
            self.audio_button.setEnabled(True)
            if ok:
                self.prompt.setText(str(value))
                self.status.setText(f"Transcribed instruction: {value}")
                self._predict_now()
            else:
                self._error("Audio transcription error", value)

        def _predict_now(self) -> None:
            if self.current_frame is None or self.inference_busy:
                return
            self.inference_busy = True
            self.predict_button.setEnabled(False)
            QApplication.processEvents()
            try:
                self.prediction = self.inference.predict(
                    self.current_frame.copy(), self.prompt.text()
                )
                self.last_inference_at = time.monotonic()
                self.live_label.setPixmap(
                    self._pixmap(self.prediction.annotated_bgr, self.live_label)
                )
                for name, image in self.inference.visualization_maps(
                    self.prediction
                ).items():
                    if name in self.map_labels:
                        self.map_labels[name].setPixmap(
                            self._pixmap(image, self.map_labels[name])
                        )
                if self.prediction.grasps:
                    grasp = self.prediction.grasps[0]
                    self.status.setText(
                        f"Prediction: x={grasp[0]:.1f}, y={grasp[1]:.1f}, "
                        f"angle={grasp[4]:.1f}, width={grasp[2]:.1f}"
                    )
                    if (
                        robot_cfg.get("auto_send")
                        and self.arm.isChecked()
                        and time.monotonic() - self.last_send_at
                        >= float(robot_cfg.get("auto_send_interval_s", 2.0))
                    ):
                        self._send_grasp()
                else:
                    self.status.setText("No grasp peak passed the quality threshold")
                self._refresh_send_state()
            except Exception as exc:
                self._error("Inference error", exc)
            finally:
                self.inference_busy = False
                self.predict_button.setEnabled(True)

        def _connect_robot(self) -> None:
            if not (bool(robot_cfg.get("enabled")) and allow_robot):
                return
            try:
                self.robot = build_robot_client(robot_cfg)
                self.robot.connect()
                self.connect_button.setEnabled(False)
                self.disconnect_button.setEnabled(True)
                self.arm.setEnabled(True)
                self.status.setText(
                    f"Receiver connected: {robot_cfg['host']}:{robot_cfg['port']}"
                )
            except Exception as exc:
                if self.robot is not None:
                    self.robot.close()
                self.robot = None
                self.connect_button.setEnabled(True)
                self.disconnect_button.setEnabled(False)
                self._error("Robot receiver connection failed", exc)

        def _disconnect_robot(self) -> None:
            self.arm.setChecked(False)
            self.arm.setEnabled(False)
            if self.robot is not None:
                self.robot.close()
            self.robot = None
            self.disconnect_button.setEnabled(False)
            self.connect_button.setEnabled(
                bool(robot_cfg.get("enabled")) and allow_robot
            )
            self._refresh_send_state()
            self.status.setText("Robot receiver disconnected")

        def _refresh_send_state(self) -> None:
            can_send = bool(
                self.robot
                and self.robot.connected
                and self.arm.isChecked()
                and self.prediction
                and self.prediction.grasps
            )
            self.send_button.setEnabled(can_send)

        def _send_grasp(self) -> None:
            if not (
                self.robot
                and self.robot.connected
                and self.arm.isChecked()
                and self.prediction
                and self.prediction.grasps
            ):
                return
            coordinate_space = str(robot_cfg.get("coordinate_space", "source")).lower()
            if coordinate_space not in {"source", "model"}:
                self._error(
                    "Robot configuration error",
                    ValueError("robot.coordinate_space must be source or model"),
                )
                return
            source_grasp = self.prediction.grasps[0]
            model_grasp = self.prediction.model_grasps[0]
            grasp = (
                model_grasp
                if coordinate_space == "model"
                else source_grasp
            )
            x, y, width, _height, theta = grasp
            source_width = command_width(
                source_grasp[2],
                self.prediction.segmentation,
                source_grasp[:2],
                source_grasp[4],
                self.prediction.prompt,
                robot_cfg.get("width_policy", {}),
            )
            if coordinate_space == "model":
                width_scale = model_grasp[2] / max(source_grasp[2], 1e-8)
                width = source_width * width_scale
            else:
                width = source_width
            theta = command_theta(theta, robot_cfg.get("theta_policy", {}))
            depth_cfg = robot_cfg.get("depth_policy", {})
            command = GraspCommand(
                # Preserve the deployed socket contract: pixel fields are ints,
                # theta remains a float, and semantic depth is an integer tier.
                x=int(x),
                y=int(y),
                theta=theta,
                width=int(width),
                depth=semantic_depth(
                    self.prediction.prompt,
                    int(robot_cfg.get("default_depth", 0)),
                    class_tiers=depth_cfg.get("class_tiers", {}),
                    policy=str(depth_cfg.get("multiple_matches", "max")),
                ),
            )
            try:
                command.validate_limits(robot_cfg.get("limits", {}))
                self.robot.send(command)
                self.last_send_at = time.monotonic()
                self.status.setText(f"Sent: {command.to_wire().decode('ascii').strip()}")
            except Exception as exc:
                self._disconnect_robot()
                self._error("Robot command failed", exc)

        def _error(self, title: str, exc: Exception) -> None:
            self.status.setText(f"{title}: {exc}")
            QMessageBox.critical(self, title, str(exc))

        def closeEvent(self, event) -> None:
            self.timer.stop()
            self.source.close()
            if gelsight is not None:
                gelsight.close()
            if self.robot is not None:
                self.robot.close()
            event.accept()

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(source)
    window.show()
    return app.exec_()
