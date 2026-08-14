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
from .qt import configure_pyqt5_plugins
from .robot import GraspCommand, LegacyTCPGraspClient, build_robot_client, semantic_depth
from .sources import FrameSource, build_source


def build_grasp_command(
    prediction: GraspPrediction, robot_cfg: Dict[str, Any]
) -> Optional[GraspCommand]:
    """Build the one command shared by GUI preview and TCP transmission."""
    if not (prediction and prediction.grasps):
        return None
    coordinate_space = str(robot_cfg.get("coordinate_space", "source")).lower()
    if coordinate_space not in {"source", "model"}:
        raise ValueError("robot.coordinate_space must be source or model")
    source_grasp = prediction.grasps[0]
    model_grasp = prediction.model_grasps[0]
    grasp = model_grasp if coordinate_space == "model" else source_grasp
    x, y, _model_width, _height, theta = grasp
    source_width = command_width(
        source_grasp[2],
        prediction.segmentation,
        source_grasp[:2],
        source_grasp[4],
        prediction.prompt,
        robot_cfg.get("width_policy", {}),
    )
    if coordinate_space == "model":
        width_scale = model_grasp[2] / max(source_grasp[2], 1e-8)
        width = source_width * width_scale
    else:
        width = source_width
    depth_cfg = robot_cfg.get("depth_policy", {})
    return GraspCommand(
        x=int(x),
        y=int(y),
        theta=command_theta(theta, robot_cfg.get("theta_policy", {})),
        width=int(width),
        depth=semantic_depth(
            prediction.prompt,
            int(robot_cfg.get("default_depth", 0)),
            class_tiers=depth_cfg.get("class_tiers", {}),
            policy=str(depth_cfg.get("multiple_matches", "max")),
        ),
    )


def run_gui(config: Dict[str, Any], allow_robot: bool = False) -> int:
    qt_platforms = configure_pyqt5_plugins()
    if qt_platforms is not None:
        print(f"[gui] PyQt5 platform plugins: {qt_platforms}")
    try:
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QColorDialog,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSizePolicy,
            QStackedWidget,
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
    legacy_layout = str(gui_cfg.get("layout", "modern")).lower() == "legacy"
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
            self.robot_connect_results = queue.Queue()
            self.robot: Optional[LegacyTCPGraspClient] = None
            self.robot_connecting = False
            self.robot_connect_generation = 0
            self.current_mode = 0
            self.frame_count = 0
            self.model_selector = None
            self.setWindowTitle(str(gui_cfg["title"]))
            if legacy_layout:
                self.setMinimumSize(1600, 1000)
                self.resize(1600, 1000)
            else:
                self.setMinimumSize(1200, 760)
                self.resize(int(gui_cfg["window_width"]), int(gui_cfg["window_height"]))
            self._build_ui()
            if (
                bool(robot_cfg.get("enabled"))
                and allow_robot
                and bool(robot_cfg.get("auto_connect", False))
            ):
                # Defer the connection until the window/event loop is ready.
                QTimer.singleShot(0, self._connect_robot)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._next_frame)
            self.timer.start(int(gui_cfg["camera_interval_ms"]))

        def _build_ui(self) -> None:
            root = QWidget(self)
            root.setObjectName("AppRoot")
            layout = QVBoxLayout(root)
            layout.setContentsMargins(18, 16, 18, 16)
            layout.setSpacing(12)

            # Match the deployed 22-class GUI: three large mode buttons on top,
            # one central page at a time, and no tab bar.
            mode_row = QHBoxLayout()
            mode_row.setContentsMargins(6, 6, 6, 6)
            mode_row.setSpacing(10)
            mode_bar = QFrame()
            mode_bar.setObjectName("TopBar")
            mode_bar.setLayout(mode_row)
            self.object_button = QPushButton("Object Detection")
            self.grasping_button = QPushButton("Grasping Points Detection")
            self.gelsight_button = QPushButton("GelSight")
            for button in (
                self.object_button,
                self.grasping_button,
                self.gelsight_button,
            ):
                button.setMinimumHeight(40)
                button.setCheckable(True)
                button.setObjectName("ModeButton")
                mode_row.addWidget(button, 1)
            if legacy_layout:
                self.appearance_button = QPushButton("Appearance")
                self.appearance_button.setMinimumHeight(40)
                self.appearance_button.setObjectName("ModeButton")
                self.appearance_button.clicked.connect(self._choose_accent)
                mode_row.addWidget(self.appearance_button, 1)
            self.object_button.clicked.connect(lambda: self._switch_mode(0))
            self.grasping_button.clicked.connect(lambda: self._switch_mode(1))
            self.gelsight_button.clicked.connect(lambda: self._switch_mode(2))
            self.object_button.setEnabled(detector is not None)
            self.gelsight_button.setEnabled(gelsight is not None)
            layout.addWidget(mode_bar)

            self.pages = QStackedWidget()

            # Object detection page.
            detection_page = QWidget()
            detection_layout = QVBoxLayout(detection_page)
            self.detection_label = self._image_label(
                "Waiting for object detection"
                if detector is not None
                else "Object detector is disabled in the deployment config"
            )
            self.detection_label.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding
            )
            if legacy_layout:
                self.detection_label.setFixedSize(1280, 720)
            detection_layout.addWidget(self.detection_label)
            self.pages.addWidget(detection_page)

            # Grasp page: two large images above three small dense maps, matching
            # realsense_object_grasp_detection_gelsight_*_widthdepth_22.py.
            grasp_page = QWidget()
            grasp_layout = QVBoxLayout(grasp_page)
            grasp_grid = QGridLayout()
            self.live_label = self._image_label("Grasp Result")
            self.mask_label = self._image_label("Segmentation Mask")
            if legacy_layout:
                self.live_label.setFixedSize(640, 480)
                self.mask_label.setFixedSize(640, 480)
            grasp_grid.addWidget(
                self._labeled_image(self.live_label, "Grasp Result"), 0, 0, 1, 3
            )
            grasp_grid.addWidget(
                self._labeled_image(self.mask_label, "Segmentation Mask"), 0, 3, 1, 3
            )
            self.map_labels = {}
            for column, (name, title) in enumerate(
                (("quality", "Quality Mask"), ("angle", "Angle Mask"), ("width", "Width Mask"))
            ):
                label = self._image_label(title, minimum=(160, 120))
                if legacy_layout:
                    label.setFixedSize(160, 120)
                else:
                    label.setMaximumHeight(180)
                self.map_labels[name] = label
                grasp_grid.addWidget(
                    self._labeled_image(label, title), 1, column * 2, 1, 2
                )
            grasp_layout.addLayout(grasp_grid, 1)

            self.sentence_label = QLabel(
                f"Current target: {config['model']['prompt']}"
            )
            self.sentence_label.setAlignment(Qt.AlignCenter)
            self.sentence_label.setObjectName("TargetLabel")
            grasp_layout.addWidget(self.sentence_label)

            self.audio_button = QPushButton("Speak")
            self.audio_button.clicked.connect(self._record_instruction)
            self.audio_button.setEnabled(audio is not None)
            self.prompt = QLineEdit(str(config["model"]["prompt"]))
            self.prompt.setPlaceholderText("Type a language instruction and press Enter")
            self.prompt.returnPressed.connect(self._predict_now)
            self.predict_button = QPushButton("Predict now")
            self.predict_button.setObjectName("PrimaryButton")
            self.predict_button.clicked.connect(self._predict_now)
            if legacy_layout:
                self.audio_button.setMinimumHeight(40)
                self.prompt.setMinimumHeight(46)
                grasp_layout.addWidget(self.audio_button)
                grasp_layout.addWidget(self.prompt)
                self.predict_button.setVisible(False)
            else:
                prompt_row = QHBoxLayout()
                prompt_row.addWidget(self.audio_button)
                prompt_row.addWidget(self.prompt, 1)
                prompt_row.addWidget(self.predict_button)
                prompt_card = QFrame()
                prompt_card.setObjectName("ControlCard")
                prompt_card.setLayout(prompt_row)
                grasp_layout.addWidget(prompt_card)
            self.pages.addWidget(grasp_page)

            # GelSight page.
            gelsight_page = QWidget()
            gelsight_layout = QVBoxLayout(gelsight_page)
            self.gelsight_label = self._image_label(
                "Waiting for GelSight frame"
                if gelsight is not None
                else "GelSight is disabled in the deployment config"
            )
            self.gelsight_label.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding
            )
            if legacy_layout:
                self.gelsight_label.setFixedSize(640, 480)
            gelsight_layout.addWidget(self.gelsight_label)
            self.pages.addWidget(gelsight_page)

            layout.addWidget(self.pages, 1)

            # Command preview stays visible in dry-run mode. Connecting and
            # sending remain guarded by robot.enabled and --allow-robot.
            robot_controls = QFrame()
            robot_controls.setObjectName("RobotCard")
            robot_layout = QVBoxLayout(robot_controls)
            robot_layout.setContentsMargins(14, 10, 14, 10)
            robot_layout.setSpacing(8)

            command_row = QHBoxLayout()
            command_row.setSpacing(10)
            command_title = QLabel("ROBOT COMMAND")
            command_title.setObjectName("SectionTitle")
            command_row.addWidget(command_title)
            self.connection_badge = QLabel("OFFLINE")
            self.connection_badge.setObjectName("ConnectionBadge")
            self.connection_badge.setProperty("connected", False)
            command_row.addWidget(self.connection_badge)
            self.command_fields = {}
            for field, label_text in (
                ("x", "X"),
                ("y", "Y"),
                ("theta", "Theta"),
                ("width", "Width"),
                ("depth", "Depth"),
            ):
                field_box = QFrame()
                field_box.setObjectName("CommandField")
                field_layout = QHBoxLayout(field_box)
                field_layout.setContentsMargins(9, 5, 9, 5)
                field_layout.setSpacing(6)
                name_label = QLabel(label_text)
                name_label.setObjectName("CommandName")
                value_label = QLabel("--")
                value_label.setObjectName("CommandValue")
                value_label.setMinimumWidth(48)
                value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                field_layout.addWidget(name_label)
                field_layout.addWidget(value_label)
                command_row.addWidget(field_box)
                self.command_fields[field] = value_label
            self.wire_preview = QLabel("{--, --, --, --, --}")
            self.wire_preview.setObjectName("WirePreview")
            self.wire_preview.setToolTip("Exact TCP payload sent to the receiver")
            command_row.addWidget(self.wire_preview, 1)
            robot_layout.addLayout(command_row)

            robot_row = QHBoxLayout()
            robot_row.setSpacing(8)
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
            self.send_button.setObjectName("SendButton")
            self.send_button.clicked.connect(self._send_grasp)
            self.send_button.setEnabled(False)
            robot_row.addWidget(self.send_button)
            self.status = QLabel()
            self.status.setObjectName("StatusText")
            if robot_cfg.get("enabled") and not allow_robot:
                self.status.setText("DRY RUN: restart with --allow-robot to permit a connection")
            elif not robot_cfg.get("enabled"):
                self.status.setText("DRY RUN: robot.enabled is false")
            else:
                self.status.setText("Robot output permitted but disconnected")
            robot_row.addWidget(self.status, 1)
            robot_layout.addLayout(robot_row)
            robot_controls.setVisible(not legacy_layout)
            layout.addWidget(robot_controls)
            self.setCentralWidget(root)
            self._apply_theme()
            self._switch_mode(0 if detector is not None else 1)

        @staticmethod
        def _labeled_image(label, title: str):
            container = QFrame()
            container.setObjectName("ImageCard")
            box = QVBoxLayout(container)
            box.setContentsMargins(8, 8, 8, 7)
            box.setSpacing(5)
            box.addWidget(label, 1)
            caption = QLabel(title)
            caption.setAlignment(Qt.AlignCenter)
            caption.setObjectName("ImageCaption")
            box.addWidget(caption)
            return container

        def _switch_mode(self, index: int) -> None:
            self.current_mode = int(index)
            self.pages.setCurrentIndex(self.current_mode)
            for button, active in (
                (self.object_button, self.current_mode == 0),
                (self.grasping_button, self.current_mode == 1),
                (self.gelsight_button, self.current_mode == 2),
            ):
                button.setChecked(active)

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
        def _image_label(text: str, minimum=(320, 240)):
            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(*minimum)
            label.setObjectName("PreviewCanvas")
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

        def _apply_theme(self) -> None:
            """Apply a restrained dark laboratory theme without changing layout."""
            self.setStyleSheet(
                """
                QWidget#AppRoot {
                    background: #0b1017;
                    color: #dce7f3;
                    font-family: "Inter", "Segoe UI", Arial, sans-serif;
                    font-size: 13px;
                }
                QFrame#TopBar, QFrame#ControlCard, QFrame#RobotCard {
                    background: #111a25;
                    border: 1px solid #223247;
                    border-radius: 10px;
                }
                QPushButton {
                    min-height: 30px;
                    padding: 5px 14px;
                    color: #d7e3ef;
                    background: #172333;
                    border: 1px solid #2a3d55;
                    border-radius: 7px;
                    font-weight: 600;
                }
                QPushButton:hover { background: #203249; border-color: #3f6388; }
                QPushButton:pressed { background: #122033; }
                QPushButton:disabled { color: #627184; background: #101720; border-color: #1b2735; }
                QPushButton#ModeButton:checked {
                    color: #06131b;
                    background: #55d6be;
                    border-color: #79ead5;
                }
                QPushButton#PrimaryButton, QPushButton#SendButton {
                    color: #06131b;
                    background: #55d6be;
                    border-color: #79ead5;
                    font-weight: 700;
                }
                QPushButton#PrimaryButton:hover, QPushButton#SendButton:hover {
                    background: #71e4ce;
                }
                QPushButton#SendButton:disabled {
                    color: #607180;
                    background: #18242f;
                    border-color: #243442;
                }
                QLineEdit {
                    min-height: 32px;
                    padding: 3px 11px;
                    color: #e6eef7;
                    background: #0c141e;
                    border: 1px solid #2a3d55;
                    border-radius: 7px;
                    selection-background-color: #347e78;
                }
                QLineEdit:focus { border-color: #55d6be; }
                QFrame#ImageCard {
                    background: #101823;
                    border: 1px solid #203148;
                    border-radius: 10px;
                }
                QLabel#PreviewCanvas {
                    color: #718297;
                    background: #05090e;
                    border: 1px solid #1b2a3d;
                    border-radius: 7px;
                }
                QLabel#ImageCaption {
                    color: #a9b9ca;
                    font-size: 12px;
                    font-weight: 600;
                }
                QLabel#TargetLabel {
                    color: #edf5fc;
                    font-size: 17px;
                    font-weight: 700;
                    padding: 5px;
                }
                QLabel#SectionTitle {
                    color: #8294a8;
                    font-size: 10px;
                    font-weight: 800;
                    letter-spacing: 1px;
                }
                QFrame#CommandField {
                    background: #0b131d;
                    border: 1px solid #24374d;
                    border-radius: 6px;
                }
                QLabel#CommandName { color: #71859a; font-size: 11px; font-weight: 700; }
                QLabel#CommandValue { color: #f0f6fb; font-family: "Consolas", monospace; font-weight: 700; }
                QLabel#WirePreview {
                    color: #7fe1cf;
                    background: #09121a;
                    border: 1px solid #25463f;
                    border-radius: 6px;
                    padding: 6px 10px;
                    font-family: "Consolas", monospace;
                }
                QLabel#ConnectionBadge {
                    color: #ffb4ad;
                    background: #3a1f25;
                    border: 1px solid #6e343d;
                    border-radius: 7px;
                    padding: 3px 8px;
                    font-size: 10px;
                    font-weight: 800;
                }
                QLabel#ConnectionBadge[connected="true"] {
                    color: #95f0d6;
                    background: #14332d;
                    border-color: #2e7566;
                }
                QLabel#StatusText { color: #90a3b7; padding-left: 6px; }
                QCheckBox { color: #b7c5d4; spacing: 7px; }
                """
            )

        def _choose_accent(self) -> None:
            """Keep the legacy Appearance control without changing its layout."""
            color = QColorDialog.getColor(parent=self, title="Choose accent color")
            if not color.isValid():
                return
            accent = color.name()
            self.setStyleSheet(
                self.styleSheet()
                + f"""
                QPushButton#ModeButton:checked,
                QPushButton#PrimaryButton,
                QPushButton#SendButton {{
                    background: {accent};
                    border-color: {accent};
                }}
                """
            )

        def _set_connection_badge(self, connected: bool) -> None:
            self.connection_badge.setText("CONNECTED" if connected else "OFFLINE")
            self.connection_badge.setProperty("connected", bool(connected))
            self.connection_badge.style().unpolish(self.connection_badge)
            self.connection_badge.style().polish(self.connection_badge)

        def _command_from_prediction(self) -> Optional[GraspCommand]:
            """Build exactly the command that preview/send share."""
            return build_grasp_command(self.prediction, robot_cfg)

        def _update_command_preview(self) -> None:
            try:
                command = self._command_from_prediction()
            except Exception as exc:
                command = None
                self.status.setText(f"Command preview error: {exc}")
            if command is None:
                for label in self.command_fields.values():
                    label.setText("--")
                self.wire_preview.setText("{--, --, --, --, --}")
                return
            values = {
                "x": f"{command.x:g}",
                "y": f"{command.y:g}",
                "theta": f"{command.theta:.1f} deg",
                "width": f"{command.width:g} px",
                "depth": f"{command.depth:d}",
            }
            for name, value in values.items():
                self.command_fields[name].setText(value)
            self.wire_preview.setText(command.to_wire().decode("ascii").strip())

        def _next_frame(self) -> None:
            self._poll_audio()
            self._poll_robot_connection()
            try:
                ok, frame = self.source.read()
            except Exception as exc:
                self.timer.stop()
                self._error("Camera error", exc)
                return
            if not ok or frame is None:
                return
            self.frame_count += 1
            self.current_frame = frame
            display = self.prediction.annotated_bgr if self.prediction is not None else frame
            if self.current_mode == 1:
                self.live_label.setPixmap(self._pixmap(display, self.live_label))
            interval_s = int(gui_cfg["inference_interval_ms"]) / 1000.0
            now = time.monotonic()
            if (
                self.gelsight_available
                and self.current_mode == 2
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
                and self.current_mode == 0
                and now - self.last_detection_at
                >= int(
                    config.get("detector", {}).get(
                        "inference_interval_ms",
                        gui_cfg["inference_interval_ms"],
                    )
                )
                / 1000.0
            ):
                try:
                    detected = detector.predict(frame)
                    self.detection_label.setPixmap(self._pixmap(detected, self.detection_label))
                    self.last_detection_at = now
                except Exception as exc:
                    self._error("Detection error", exc)
            elif (
                self.current_mode == 1
                and bool(gui_cfg["continuous_inference"])
                and now - self.last_inference_at >= interval_s
            ):
                self._predict_now()
            if (
                legacy_layout
                and self.current_mode == 1
                and self.frame_count
                % max(1, int(gui_cfg.get("legacy_send_every_frames", 50)))
                == 0
                and robot_cfg.get("auto_send")
                and self.arm.isChecked()
            ):
                self._send_grasp()

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
                self.sentence_label.setText(
                    f"Current target: {self.prompt.text()}"
                )
                self.last_inference_at = time.monotonic()
                self.live_label.setPixmap(
                    self._pixmap(self.prediction.annotated_bgr, self.live_label)
                )
                for name, image in self.inference.visualization_maps(
                    self.prediction
                ).items():
                    if name == "segmentation":
                        self.mask_label.setPixmap(
                            self._pixmap(image, self.mask_label)
                        )
                    if name in self.map_labels:
                        self.map_labels[name].setPixmap(
                            self._pixmap(image, self.map_labels[name])
                        )
                if self.prediction.grasps:
                    grasp = self.prediction.grasps[0]
                    self._update_command_preview()
                    self.status.setText(
                        f"Prediction: x={grasp[0]:.1f}, y={grasp[1]:.1f}, "
                        f"angle={grasp[4]:.1f}, width={grasp[2]:.1f}"
                    )
                    if (
                        not legacy_layout
                        and robot_cfg.get("auto_send")
                        and self.arm.isChecked()
                        and time.monotonic() - self.last_send_at
                        >= float(robot_cfg.get("auto_send_interval_s", 2.0))
                    ):
                        self._send_grasp()
                else:
                    self._update_command_preview()
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
            if self.robot_connecting or (self.robot and self.robot.connected):
                return
            self.robot_connecting = True
            self.robot_connect_generation += 1
            generation = self.robot_connect_generation
            client = build_robot_client(robot_cfg)
            self.connect_button.setEnabled(False)
            self.status.setText(
                f"Waiting for receiver: {robot_cfg['host']}:{robot_cfg['port']}"
            )

            def worker():
                try:
                    client.connect()
                    self.robot_connect_results.put((generation, client, None))
                except Exception as exc:
                    client.close()
                    self.robot_connect_results.put((generation, None, exc))

            threading.Thread(target=worker, daemon=True).start()

        def _poll_robot_connection(self) -> None:
            try:
                generation, client, error = self.robot_connect_results.get_nowait()
            except queue.Empty:
                return
            if generation != self.robot_connect_generation:
                if client is not None:
                    client.close()
                return
            self.robot_connecting = False
            if error is None:
                self.robot = client
                self.connect_button.setEnabled(False)
                self.disconnect_button.setEnabled(True)
                self.arm.setEnabled(True)
                if bool(robot_cfg.get("auto_arm", False)):
                    self.arm.setChecked(True)
                self._set_connection_badge(True)
                self.status.setText(
                    f"Receiver connected: {robot_cfg['host']}:{robot_cfg['port']}"
                )
                self.statusBar().showMessage(self.status.text())
            else:
                self.robot = None
                self._set_connection_badge(False)
                self.connect_button.setEnabled(True)
                self.disconnect_button.setEnabled(False)
                self._error("Robot receiver connection failed", error)

        def _disconnect_robot(self) -> None:
            self.robot_connect_generation += 1
            self.robot_connecting = False
            self.arm.setChecked(False)
            self.arm.setEnabled(False)
            if self.robot is not None:
                self.robot.close()
            self.robot = None
            self._set_connection_badge(False)
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
            try:
                command = self._command_from_prediction()
                if command is None:
                    return
                command.validate_limits(robot_cfg.get("limits", {}))
                self.robot.send(command)
                self.last_send_at = time.monotonic()
                self.status.setText(f"Sent: {command.to_wire().decode('ascii').strip()}")
                self.statusBar().showMessage(self.status.text())
            except Exception as exc:
                self._disconnect_robot()
                self._error("Robot command failed", exc)

        def _error(self, title: str, exc: Exception) -> None:
            self.status.setText(f"{title}: {exc}")
            QMessageBox.critical(self, title, str(exc))

        def closeEvent(self, event) -> None:
            self.timer.stop()
            self.robot_connect_generation += 1
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
