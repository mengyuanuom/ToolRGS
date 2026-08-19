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


GUI_THEMES = {
    "Midnight Teal": {},
    "Ocean Blue": {
        "#0b1017": "#071426",
        "#111a25": "#0d2038",
        "#0e1721": "#0a1a2e",
        "#172333": "#132b48",
        "#203249": "#1b416c",
        "#122033": "#10243b",
        "#0c141e": "#09182a",
        "#2a3d55": "#2b527d",
        "#101823": "#0b1b30",
        "#05090e": "#030b16",
        "#203148": "#23476e",
        "#1b2a3d": "#1d3d61",
        "#55d6be": "#5ba8ff",
        "#79ead5": "#8ac3ff",
        "#71e4ce": "#79b8ff",
        "#347e78": "#326fa9",
    },
    "Violet Night": {
        "#0b1017": "#100e18",
        "#111a25": "#1b1728",
        "#0e1721": "#171321",
        "#172333": "#261f38",
        "#203249": "#382d51",
        "#122033": "#1d172b",
        "#0c141e": "#15111f",
        "#2a3d55": "#493a63",
        "#101823": "#191523",
        "#05090e": "#09070d",
        "#203148": "#3b3151",
        "#1b2a3d": "#302740",
        "#55d6be": "#b794f6",
        "#79ead5": "#d0b8ff",
        "#71e4ce": "#c6a8ff",
        "#347e78": "#7656a5",
    },
    "Graphite Amber": {
        "#0b1017": "#111111",
        "#111a25": "#1b1b1b",
        "#0e1721": "#181818",
        "#172333": "#292929",
        "#203249": "#393939",
        "#122033": "#202020",
        "#0c141e": "#161616",
        "#2a3d55": "#484848",
        "#101823": "#1a1a1a",
        "#05090e": "#080808",
        "#203148": "#3c3c3c",
        "#1b2a3d": "#303030",
        "#55d6be": "#f2b84b",
        "#79ead5": "#ffd176",
        "#71e4ce": "#ffc963",
        "#347e78": "#946b21",
    },
}


def format_grasp_prompt(value: str, template: str = "Grasp {}") -> str:
    """Turn a target name into the grasp instruction expected by the model."""
    value = normalize_prompt_text(value)
    if not value:
        raise ValueError("Enter a grasp target, for example: Grasp screwdriver")
    lowered = value.casefold()
    instruction_prefixes = (
        "grasp ",
        "pick ",
        "select ",
        "take ",
        "reach ",
        "find ",
        "locate ",
    )
    if lowered.startswith(instruction_prefixes):
        return value
    template = str(template or "Grasp {}").strip()
    if template.count("{}") != 1:
        raise ValueError("model.prompt_template must contain exactly one {} placeholder")
    return template.format(value)


def normalize_prompt_text(value: Any) -> str:
    """Normalize GUI prompt input without accepting whitespace-only text."""
    return str(value or "").strip()


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
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListView,
            QMainWindow,
            QMessageBox,
            QProgressBar,
            QPushButton,
            QSizePolicy,
            QSpinBox,
            QStackedWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The deployment GUI requires PyQt5; install requirement-deploy.txt"
        ) from exc

    class StableModelComboBox(QComboBox):
        """A model selector that does not fight the live-preview refresh loop."""

        popup_visibility_changed = pyqtSignal(bool)

        def showPopup(self) -> None:
            self.view().setMinimumWidth(max(300, self.width()))
            self.popup_visibility_changed.emit(True)
            super().showPopup()

        def hidePopup(self) -> None:
            super().hidePopup()
            self.popup_visibility_changed.emit(False)

        def wheelEvent(self, event) -> None:
            # Prevent a scroll over the closed selector from silently changing
            # the active model. The wheel still works inside the open list.
            if self.view().isVisible():
                super().wheelEvent(event)
            else:
                event.ignore()

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
            self.prompt_template = str(
                config.get("model", {}).get("prompt_template", "Grasp {}")
            )
            self.prompt_missing = not bool(
                normalize_prompt_text(config["model"].get("prompt", ""))
            )
            self.audio_results = queue.Queue()
            self.robot_connect_results = queue.Queue()
            self.inference_results = queue.Queue()
            self.model_load_results = queue.Queue()
            self.robot: Optional[LegacyTCPGraspClient] = None
            self.robot_connecting = False
            self.robot_connect_generation = 0
            self.current_mode = 0
            self.frame_count = 0
            self.model_selector = None
            self.model_popup_open = False
            self.settings_panel = None
            self.model_load_progress = None
            self.model_badge = None
            self.theme_selector = None
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
            self.theme_selector = QComboBox()
            self.theme_selector.setObjectName("ThemeSelector")
            self.theme_selector.setMinimumWidth(150)
            self.theme_selector.setToolTip("Choose the GUI colour theme")
            self.theme_selector.addItems(list(GUI_THEMES))
            configured_theme = str(gui_cfg.get("theme", "Midnight Teal"))
            if configured_theme not in GUI_THEMES:
                configured_theme = "Midnight Teal"
            self.theme_selector.setCurrentText(configured_theme)
            self.theme_selector.currentTextChanged.connect(self._change_theme)
            mode_row.addWidget(self.theme_selector)
            self.settings_button = QPushButton("Post-processing  ▾")
            self.settings_button.setMinimumHeight(40)
            self.settings_button.setCheckable(True)
            self.settings_button.setObjectName("SettingsButton")
            self.settings_button.toggled.connect(self._toggle_settings)
            mode_row.addWidget(self.settings_button)
            self.object_button.clicked.connect(lambda: self._switch_mode(0))
            self.grasping_button.clicked.connect(lambda: self._switch_mode(1))
            self.gelsight_button.clicked.connect(lambda: self._switch_mode(2))
            self.object_button.setEnabled(detector is not None)
            self.gelsight_button.setEnabled(gelsight is not None)
            layout.addWidget(mode_bar)

            self.settings_panel = self._build_settings_panel()
            self.settings_panel.setVisible(False)
            layout.addWidget(self.settings_panel)

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
            self.prompt.setObjectName("GraspPrompt")
            self.prompt.setPlaceholderText("Grasp {}")
            self.prompt.setToolTip(
                "Enter a target name or a full instruction; a bare name uses Grasp {}"
            )
            self.prompt.textChanged.connect(self._prompt_changed)
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

        def _build_settings_panel(self):
            panel = QFrame()
            panel.setObjectName("SettingsPanel")
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(14, 10, 14, 10)
            layout.setSpacing(0)
            self.settings_pages = QStackedWidget()
            self.settings_pages.setObjectName("PostprocessingPages")
            layout.addWidget(self.settings_pages)

            detection_page = QWidget()
            detection_row = QHBoxLayout(detection_page)
            detection_row.setContentsMargins(0, 0, 0, 0)
            detection_row.setSpacing(14)
            detector_cfg = config.get("detector", {})

            score_label = QLabel("Score threshold")
            score_label.setObjectName("FieldLabel")
            detection_row.addWidget(score_label)
            self.detection_score_input = QDoubleSpinBox()
            self.detection_score_input.setObjectName("DetectionScoreThreshold")
            self.detection_score_input.setRange(0.0, 1.0)
            self.detection_score_input.setDecimals(2)
            self.detection_score_input.setSingleStep(0.05)
            self.detection_score_input.setValue(
                float(detector_cfg.get("score_threshold", 0.7))
            )
            detection_row.addWidget(self.detection_score_input)

            nms_label = QLabel("NMS IoU")
            nms_label.setObjectName("FieldLabel")
            detection_row.addWidget(nms_label)
            self.detection_nms_input = QDoubleSpinBox()
            self.detection_nms_input.setObjectName("DetectionNmsThreshold")
            self.detection_nms_input.setRange(0.0, 1.0)
            self.detection_nms_input.setDecimals(2)
            self.detection_nms_input.setSingleStep(0.05)
            self.detection_nms_input.setValue(
                float(detector_cfg.get("nms_threshold", 0.5))
            )
            self.detection_nms_input.setToolTip(
                "Suppress overlapping boxes whose IoU exceeds this value"
            )
            detection_row.addWidget(self.detection_nms_input)

            max_label = QLabel("Max detections")
            max_label.setObjectName("FieldLabel")
            detection_row.addWidget(max_label)
            self.detection_max_input = QSpinBox()
            self.detection_max_input.setObjectName("DetectionMaxDetections")
            self.detection_max_input.setRange(1, 1000)
            self.detection_max_input.setValue(
                int(detector_cfg.get("max_detections", 100))
            )
            detection_row.addWidget(self.detection_max_input)
            detection_row.addStretch(1)
            for control in (
                self.detection_score_input,
                self.detection_nms_input,
                self.detection_max_input,
            ):
                control.setEnabled(detector is not None)
                control.valueChanged.connect(
                    self._apply_detection_postprocessing_controls
                )
            self.settings_pages.addWidget(detection_page)

            grasp_page = QWidget()
            row = QHBoxLayout(grasp_page)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(14)

            model_label = QLabel("GRASP MODEL")
            model_label.setObjectName("FieldLabel")
            row.addWidget(model_label)
            self.model_selector = StableModelComboBox()
            self.model_selector.setObjectName("ModelSelector")
            popup = QListView(self.model_selector)
            popup.setObjectName("ModelPopup")
            popup.setUniformItemSizes(True)
            popup.setSpacing(1)
            self.model_selector.setView(popup)
            profiles = config.get("_model_profiles", {})
            if not profiles:
                profiles = {self.active_model: config["model"]}
            self.model_selector.setMinimumWidth(300)
            self.model_selector.setMaxVisibleItems(12)
            self.model_selector.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon
            )
            for name, profile in profiles.items():
                index = self.model_selector.count()
                self.model_selector.addItem(str(name), str(name))
                detail = str(
                    profile.get("architecture")
                    or profile.get("config")
                    or name
                )
                self.model_selector.setItemData(index, detail, Qt.ToolTipRole)
            active_index = self.model_selector.findData(self.active_model)
            self.model_selector.setCurrentIndex(max(0, active_index))
            self.model_selector.activated[int].connect(
                self._model_selection_changed
            )
            self.model_selector.popup_visibility_changed.connect(
                self._set_model_popup_open
            )
            row.addWidget(self.model_selector)
            self.model_badge = QLabel("READY")
            self.model_badge.setObjectName("ModelBadge")
            self.model_badge.setProperty("loading", False)
            row.addWidget(self.model_badge)
            self.model_load_progress = QProgressBar()
            self.model_load_progress.setObjectName("ModelLoadProgress")
            self.model_load_progress.setRange(0, 0)
            self.model_load_progress.setTextVisible(False)
            self.model_load_progress.setFixedWidth(80)
            self.model_load_progress.setVisible(False)
            row.addWidget(self.model_load_progress)

            height_label = QLabel("Gripper height")
            height_label.setObjectName("FieldLabel")
            row.addWidget(height_label)
            self.grasp_height_input = QDoubleSpinBox()
            self.grasp_height_input.setRange(1.0, 300.0)
            self.grasp_height_input.setDecimals(1)
            self.grasp_height_input.setSingleStep(1.0)
            self.grasp_height_input.setSuffix(" px")
            self.grasp_height_input.setToolTip(
                "Short side of the decoded grasp rectangle in source-image pixels"
            )
            row.addWidget(self.grasp_height_input)

            self.use_mask_input = QCheckBox("Use mask")
            self.use_mask_input.setToolTip(
                "Apply the thresholded segmentation mask during post-processing"
            )
            row.addWidget(self.use_mask_input)

            threshold_label = QLabel("Mask threshold")
            threshold_label.setObjectName("FieldLabel")
            row.addWidget(threshold_label)
            self.mask_threshold_input = QDoubleSpinBox()
            self.mask_threshold_input.setRange(0.0, 1.0)
            self.mask_threshold_input.setDecimals(2)
            self.mask_threshold_input.setSingleStep(0.05)
            row.addWidget(self.mask_threshold_input)

            expand_label = QLabel("Expand")
            expand_label.setObjectName("FieldLabel")
            row.addWidget(expand_label)
            self.mask_expand_input = QSpinBox()
            self.mask_expand_input.setRange(0, 100)
            self.mask_expand_input.setSuffix(" px")
            self.mask_expand_input.setToolTip(
                "Dilate the thresholded mask by this radius in source-image pixels"
            )
            row.addWidget(self.mask_expand_input)

            self.filter_grasps_input = QCheckBox("Filter grasp points")
            self.filter_grasps_input.setToolTip(
                "Reject quality peaks and offset grasp centres outside the mask"
            )
            row.addWidget(self.filter_grasps_input)
            row.addStretch(1)

            self._load_postprocessing_controls(config["model"])
            self.use_mask_input.toggled.connect(self._mask_controls_changed)
            self.filter_grasps_input.toggled.connect(
                self._apply_postprocessing_controls
            )
            self.grasp_height_input.valueChanged.connect(
                self._apply_postprocessing_controls
            )
            self.mask_threshold_input.valueChanged.connect(
                self._apply_postprocessing_controls
            )
            self.mask_expand_input.valueChanged.connect(
                self._apply_postprocessing_controls
            )
            self._sync_mask_control_state()
            self.settings_pages.addWidget(grasp_page)

            gelsight_page = QWidget()
            gelsight_row = QHBoxLayout(gelsight_page)
            gelsight_row.setContentsMargins(0, 0, 0, 0)
            gelsight_message = QLabel(
                "GelSight uses its configured classifier settings."
            )
            gelsight_message.setObjectName("FieldLabel")
            gelsight_row.addWidget(gelsight_message)
            gelsight_row.addStretch(1)
            self.settings_pages.addWidget(gelsight_page)
            return panel

        def _toggle_settings(self, expanded: bool) -> None:
            self.settings_panel.setVisible(bool(expanded))
            self._refresh_settings_button()

        def _refresh_settings_button(self) -> None:
            titles = (
                "Detection Post-processing",
                "Grasp Model & Post-processing",
                "GelSight Settings",
            )
            suffix = "▴" if self.settings_button.isChecked() else "▾"
            self.settings_button.setText(f"{titles[self.current_mode]}  {suffix}")

        def _apply_detection_postprocessing_controls(self, *_args) -> None:
            if detector is None:
                return
            detector.update_postprocessing(
                score_threshold=self.detection_score_input.value(),
                nms_threshold=self.detection_nms_input.value(),
                max_detections=self.detection_max_input.value(),
            )
            detector_cfg = config.setdefault("detector", {})
            detector_cfg["score_threshold"] = self.detection_score_input.value()
            detector_cfg["nms_threshold"] = self.detection_nms_input.value()
            detector_cfg["max_detections"] = self.detection_max_input.value()
            self.last_detection_at = 0.0
            if hasattr(self, "status"):
                self.status.setText(
                    "Detection post-processing updated; the next frame uses these settings"
                )

        def _load_postprocessing_controls(self, model_cfg: Dict[str, Any]) -> None:
            postprocessor = dict(model_cfg.get("postprocessor", {}))
            controls = (
                self.grasp_height_input,
                self.use_mask_input,
                self.mask_threshold_input,
                self.mask_expand_input,
                self.filter_grasps_input,
            )
            for control in controls:
                control.blockSignals(True)
            self.grasp_height_input.setValue(
                float(postprocessor.get("grasp_height", 20.0))
            )
            self.use_mask_input.setChecked(
                bool(model_cfg.get("use_mask_postprocessing", True))
            )
            self.mask_threshold_input.setValue(
                float(model_cfg.get("mask_threshold", 0.35))
            )
            self.mask_expand_input.setValue(
                int(model_cfg.get("mask_expand_px", 0))
            )
            self.filter_grasps_input.setChecked(
                bool(
                    model_cfg.get(
                        "filter_grasps_by_mask",
                        model_cfg.get("gate_quality_by_mask", True),
                    )
                )
            )
            for control in controls:
                control.blockSignals(False)
            self._sync_mask_control_state()

        def _sync_mask_control_state(self) -> None:
            enabled = self.use_mask_input.isChecked()
            self.mask_threshold_input.setEnabled(enabled)
            self.mask_expand_input.setEnabled(enabled)
            self.filter_grasps_input.setEnabled(enabled)

        def _mask_controls_changed(self, _checked: bool) -> None:
            self._sync_mask_control_state()
            self._apply_postprocessing_controls()

        def _apply_postprocessing_controls(self, *_args) -> None:
            self.inference.update_postprocessing(
                grasp_height=self.grasp_height_input.value(),
                use_mask=self.use_mask_input.isChecked(),
                mask_threshold=self.mask_threshold_input.value(),
                mask_expand_px=self.mask_expand_input.value(),
                filter_grasps_by_mask=self.filter_grasps_input.isChecked(),
            )
            self.prediction = None
            if hasattr(self, "status"):
                self.status.setText(
                    "Post-processing updated; the next prediction uses these settings"
                )

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
            self.settings_pages.setCurrentIndex(self.current_mode)
            for button, active in (
                (self.object_button, self.current_mode == 0),
                (self.grasping_button, self.current_mode == 1),
                (self.gelsight_button, self.current_mode == 2),
            ):
                button.setChecked(active)
            self._refresh_settings_button()

        def _model_selection_changed(self, index: int) -> None:
            if self.model_selector is None or index < 0:
                return
            name = self.model_selector.itemData(index)
            self._change_model(str(name or self.model_selector.itemText(index)))

        def _set_model_popup_open(self, visible: bool) -> None:
            self.model_popup_open = bool(visible)

        def _select_active_model(self) -> None:
            if self.model_selector is None:
                return
            index = self.model_selector.findData(self.active_model)
            self.model_selector.blockSignals(True)
            self.model_selector.setCurrentIndex(max(0, index))
            self.model_selector.blockSignals(False)

        def _set_busy(self, busy: bool, loading_model: bool = False) -> None:
            self.inference_busy = bool(busy)
            if self.model_selector is not None:
                self.model_selector.setEnabled(not busy)
            if hasattr(self, "predict_button"):
                self.predict_button.setEnabled(not busy)
            if self.model_load_progress is not None:
                self.model_load_progress.setVisible(bool(loading_model))
            if self.model_badge is not None:
                self.model_badge.setText("LOADING" if loading_model else "READY")
                self.model_badge.setProperty("loading", bool(loading_model))
                self.model_badge.style().unpolish(self.model_badge)
                self.model_badge.style().polish(self.model_badge)

        def _change_model(self, name: str) -> None:
            name = str(name).strip()
            if not name or name == self.active_model:
                return
            if self.inference_busy:
                self._select_active_model()
                self.status.setText("Please wait for the current operation to finish")
                return
            self._set_busy(True, loading_model=True)
            self.status.setText(f"Loading model profile: {name} ...")
            result_queue = self.model_load_results

            def worker():
                try:
                    selected = activate_model_profile(config, name)
                    loaded = ToolRGSInference(selected)
                    result_queue.put((True, name, selected, loaded))
                except Exception as exc:
                    result_queue.put((False, name, None, exc))

            threading.Thread(target=worker, daemon=True).start()

        def _poll_model_load(self) -> None:
            try:
                ok, name, selected, value = self.model_load_results.get_nowait()
            except queue.Empty:
                return
            if ok:
                previous = self.inference
                self.inference = value
                self.active_model = name
                self._load_postprocessing_controls(selected["model"])
                self.prompt_template = str(
                    selected["model"].get("prompt_template", "Grasp {}")
                )
                self.prompt.setText(str(selected["model"].get("prompt", "")))
                self.prediction = None
                self._update_command_preview()
                self.status.setText(f"Loaded model profile: {name}")
                # Let the old CUDA model release outside the Qt event thread.
                def release_old_model(old):
                    # Ensure the GUI callback has returned before dropping the
                    # final reference and releasing a large CUDA allocation.
                    time.sleep(0.05)
                    del old

                threading.Thread(
                    target=release_old_model, args=(previous,), daemon=True
                ).start()
            else:
                self._select_active_model()
                self._error("Model loading error", value)
            self._set_busy(False)

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
            """Apply the selected colour palette without changing layout."""
            stylesheet = """
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
                QFrame#SettingsPanel {
                    background: #0e1721;
                    border: 1px solid #294058;
                    border-radius: 9px;
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
                QComboBox, QSpinBox, QDoubleSpinBox {
                    min-height: 30px;
                    padding: 2px 8px;
                    color: #e6eef7;
                    background: #0c141e;
                    border: 1px solid #2a3d55;
                    border-radius: 7px;
                }
                QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                    border-color: #55d6be;
                }
                QComboBox#ModelSelector {
                    min-height: 34px;
                    padding: 3px 38px 3px 12px;
                    color: #eaf4ff;
                    background: #0c141e;
                    border: 1px solid #36516d;
                    border-radius: 7px;
                    font-weight: 650;
                }
                QComboBox#ModelSelector:hover { border-color: #55d6be; background: #101d29; }
                QComboBox#ModelSelector:focus { border: 1px solid #79ead5; }
                QComboBox#ModelSelector:disabled { color: #718297; background: #101720; }
                QComboBox#ModelSelector::drop-down {
                    width: 32px;
                    border: 0;
                    border-left: 1px solid #2a3d55;
                }
                QComboBox#ModelSelector QAbstractItemView {
                    color: #eaf4ff;
                    background: #111a25;
                    selection-color: #06131b;
                    selection-background-color: #55d6be;
                    show-decoration-selected: 1;
                    border: 1px solid #36516d;
                    border-radius: 6px;
                    outline: 0;
                    padding: 4px;
                }
                QComboBox#ModelSelector QAbstractItemView::item {
                    min-height: 34px;
                    padding: 5px 10px;
                }
                QComboBox#ModelSelector QAbstractItemView::item:selected {
                    color: #06131b;
                    background: #55d6be;
                }
                QListView#ModelPopup {
                    color: #eaf4ff;
                    background: #111a25;
                    selection-color: #06131b;
                    selection-background-color: #55d6be;
                    border: 1px solid #36516d;
                    outline: 0;
                    padding: 4px;
                }
                QListView#ModelPopup::item {
                    min-height: 34px;
                    padding: 5px 10px;
                }
                QListView#ModelPopup::item:selected {
                    color: #06131b;
                    background: #55d6be;
                }
                QLabel#ModelBadge {
                    color: #95f0d6;
                    background: #14332d;
                    border: 1px solid #2e7566;
                    border-radius: 7px;
                    padding: 4px 9px;
                    font-size: 10px;
                    font-weight: 800;
                }
                QLabel#ModelBadge[loading="true"] {
                    color: #ffe3a1;
                    background: #3a3018;
                    border-color: #7c6425;
                }
                QProgressBar#ModelLoadProgress {
                    min-height: 7px;
                    max-height: 7px;
                    background: #0a1119;
                    border: 1px solid #26394e;
                    border-radius: 4px;
                }
                QProgressBar#ModelLoadProgress::chunk {
                    background: #55d6be;
                    border-radius: 3px;
                }
                QLabel#FieldLabel {
                    color: #8fa3b7;
                    font-size: 11px;
                    font-weight: 700;
                }
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
            theme_name = (
                self.theme_selector.currentText()
                if self.theme_selector is not None
                else "Midnight Teal"
            )
            for source, target in GUI_THEMES.get(theme_name, {}).items():
                stylesheet = stylesheet.replace(source, target)
            self.setStyleSheet(stylesheet)

        def _change_theme(self, theme_name: str) -> None:
            if theme_name not in GUI_THEMES:
                return
            gui_cfg["theme"] = theme_name
            self._apply_theme()
            if hasattr(self, "status"):
                self.status.setText(f"GUI theme: {theme_name}")

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
            # Updating large preview pixmaps at camera frame-rate causes some
            # Linux Qt styles to repeatedly repaint/reposition combo popups.
            # Freeze preview work for the brief period in which the user is
            # choosing a model; the camera/source itself remains open.
            if self.model_popup_open:
                return
            self._poll_model_load()
            self._poll_inference()
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
                and not self.prompt_missing
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
            raw_prompt = normalize_prompt_text(self.prompt.text())
            if not raw_prompt:
                # A modal QMessageBox here is immediately reopened by the
                # continuous-inference timer and effectively traps the user.
                # Keep the editor usable and wait for valid text instead.
                self.prompt_missing = True
                self.last_inference_at = time.monotonic()
                self.status.setText(
                    "Language prompt is empty — enter a target to resume inference"
                )
                self.statusBar().showMessage(self.status.text())
                self.prompt.setFocus(Qt.OtherFocusReason)
                return
            try:
                prompt = format_grasp_prompt(
                    raw_prompt, self.prompt_template
                )
            except ValueError as exc:
                self._error("Invalid grasp prompt", exc)
                return
            self.prompt.setText(prompt)
            self._set_busy(True)
            frame = self.current_frame.copy()
            engine = self.inference
            result_queue = self.inference_results

            def worker():
                try:
                    prediction = engine.predict(frame, prompt)
                    maps = engine.visualization_maps(prediction)
                    result_queue.put((True, prompt, prediction, maps))
                except Exception as exc:
                    result_queue.put((False, prompt, exc, None))

            threading.Thread(target=worker, daemon=True).start()

        def _prompt_changed(self, value: str) -> None:
            has_prompt = bool(normalize_prompt_text(value))
            was_missing = self.prompt_missing
            self.prompt_missing = not has_prompt
            if has_prompt and was_missing and hasattr(self, "status"):
                self.last_inference_at = 0.0
                self.status.setText(
                    "Language prompt ready — inference will resume automatically"
                )
                self.statusBar().showMessage(self.status.text())
            elif not has_prompt and hasattr(self, "status"):
                self.last_inference_at = time.monotonic()
                self.status.setText(
                    "Language prompt is empty — enter a target to resume inference"
                )
                self.statusBar().showMessage(self.status.text())

        def _poll_inference(self) -> None:
            try:
                ok, prompt, value, maps = self.inference_results.get_nowait()
            except queue.Empty:
                return
            current_prompt = normalize_prompt_text(self.prompt.text())
            if prompt != current_prompt:
                # The user edited/cleared the prompt while the worker was
                # running. Never display or auto-send a stale grasp result.
                self._set_busy(False)
                if current_prompt:
                    self.last_inference_at = 0.0
                return
            if not ok:
                self._error("Inference error", value)
                self._set_busy(False)
                return
            try:
                self.prediction = value
                self.sentence_label.setText(
                    f"Current target: {prompt}"
                )
                self.last_inference_at = time.monotonic()
                self.live_label.setPixmap(
                    self._pixmap(self.prediction.annotated_bgr, self.live_label)
                )
                for name, image in maps.items():
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
                self._set_busy(False)

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
