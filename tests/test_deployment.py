import tempfile
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

from deployment.config import activate_model_profile, load_deployment_config
from deployment.detector import (
    _trusted_checkpoint_load_context,
    _trusted_mmengine_checkpoint_context,
)
from deployment.grasp_policy import command_theta, command_width, mask_span_width
from deployment.gui import build_grasp_command
from deployment.inference import GraspPrediction, opencv_grasp_rectangle
from deployment.qt import configure_pyqt5_plugins
from deployment.robot import GraspCommand, LegacyTCPGraspClient, semantic_depth
from deployment.weights import ensure_deployment_checkpoint
from deploy_gui import (
    DEFAULT_SAMPLE_IMAGE,
    apply_camera_preset,
    apply_runtime_overrides,
    parse_args,
)
from deploy_gui_legacy_gi import prepare_legacy_gi_config


class DeploymentContractTest(unittest.TestCase):
    def test_image_cli_uses_bundled_sample_by_default(self):
        args = parse_args(["--image"])
        self.assertEqual(args.image, DEFAULT_SAMPLE_IMAGE)

    def test_image_cli_switches_off_camera_and_continuous_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.jpg"
            image.write_bytes(b"placeholder")
            config = {
                "_repo_root": directory,
                "camera": {"type": "realsense", "image_path": ""},
                "gui": {"continuous_inference": True},
                "model": {"prompt": "the tool"},
            }
            args = SimpleNamespace(image=str(image), prompt="the sponge")
            updated = apply_runtime_overrides(config, args)
        self.assertEqual(updated["camera"]["type"], "image")
        self.assertEqual(updated["camera"]["backend"], "image")
        self.assertFalse(updated["gui"]["continuous_inference"])
        self.assertEqual(updated["model"]["prompt"], "the sponge")

    def test_direct_realsense_camera_preset(self):
        config = {"camera": {"type": "image"}}
        updated = apply_camera_preset(config, "realsense")
        self.assertEqual(updated["camera"]["type"], "realsense")
        self.assertEqual(
            (updated["camera"]["width"], updated["camera"]["height"]),
            (1280, 720),
        )

    def test_gi_camera_preset_uses_validated_shared_memory_socket(self):
        config = {"camera": {"type": "realsense"}}
        updated = apply_camera_preset(config, "gi")
        self.assertEqual(updated["camera"]["type"], "gstreamer")
        self.assertIn(
            "socket-path=/home/raico-hri/v1/kinova_rs_grasp/foo/fooA",
            updated["camera"]["gstreamer_pipeline"],
        )
        self.assertIn("format=BGR,width=1280,height=720", updated["camera"]["gstreamer_pipeline"])

    def test_legacy_gi_entry_keeps_old_layout_and_blocking_socket(self):
        config = {
            "camera": {"type": "realsense"},
            "gui": {},
            "robot": {"timeout_s": 2.0},
        }
        updated = prepare_legacy_gi_config(config)
        self.assertEqual(updated["camera"]["type"], "gstreamer")
        self.assertEqual(updated["gui"]["layout"], "legacy")
        self.assertEqual(updated["gui"]["legacy_send_every_frames"], 50)
        self.assertIsNone(updated["robot"]["timeout_s"])

    def test_legacy_wire_protocol(self):
        command = GraspCommand(12.5, 33, -45.25, 80, 1)
        self.assertEqual(command.to_wire(), b"{12.5, 33, -45.25, 80, 1}\n")

    def test_gui_preview_command_matches_legacy_socket_payload(self):
        prediction = GraspPrediction(
            prompt="grasp the screwdriver",
            annotated_bgr=np.zeros((720, 1280, 3), dtype=np.uint8),
            segmentation=np.ones((720, 1280), dtype=np.uint8),
            quality=np.zeros((720, 1280), dtype=np.float32),
            angle=np.zeros((720, 1280), dtype=np.float32),
            width=np.zeros((720, 1280), dtype=np.float32),
            short_side=None,
            grasps=[[640.9, 360.2, 150.0, 20.0, -25.0]],
            model_grasps=[[224.0, 224.0, 52.5, 20.0, -25.0]],
            scores=[0.9],
        )
        command = build_grasp_command(
            prediction,
            {
                "coordinate_space": "source",
                "width_policy": {"type": "model"},
                "theta_policy": {
                    "offset_degrees": 180,
                    "normalization": "zero_360",
                },
                "depth_policy": {},
                "default_depth": 0,
            },
        )
        self.assertEqual(command.to_wire(), b"{640, 360, 155, 150, -1}\n")

    @mock.patch("deployment.robot.socket.create_connection")
    def test_tcp_socket_connects_and_sends_legacy_payload(self, create_connection):
        fake_socket = mock.Mock()
        create_connection.return_value = fake_socket
        client = LegacyTCPGraspClient("192.168.38.10", 3000, timeout_s=2.0)
        client.connect()
        client.send(GraspCommand(640, 360, 155.5, 82, -1))

        create_connection.assert_called_once_with(
            ("192.168.38.10", 3000), timeout=2.0
        )
        fake_socket.settimeout.assert_called_once_with(2.0)
        fake_socket.sendall.assert_called_once_with(
            b"{640, 360, 155.5, 82, -1}\n"
        )
        client.close()
        fake_socket.close.assert_called_once()

    @mock.patch("deployment.robot.socket.create_connection")
    def test_tcp_socket_supports_blocking_lab_mode(self, create_connection):
        fake_socket = mock.Mock()
        create_connection.return_value = fake_socket
        client = LegacyTCPGraspClient(
            "192.168.38.10", 3000, timeout_s=None
        )

        client.connect()

        create_connection.assert_called_once_with(
            ("192.168.38.10", 3000), timeout=None
        )
        fake_socket.settimeout.assert_called_once_with(None)
        client.close()
        fake_socket.close.assert_called_once()

    def test_invalid_width_is_rejected(self):
        with self.assertRaises(ValueError):
            GraspCommand(1, 2, 3, 0, 0).to_wire()

    def test_command_limits_reject_out_of_frame_center(self):
        command = GraspCommand(1300, 300, 0, 80, 0)
        with self.assertRaises(ValueError):
            command.validate_limits(
                {
                    "x": [0, 1280],
                    "y": [0, 720],
                    "theta": [-90, 90],
                    "width": [1, 600],
                    "depth": [-1, 1],
                }
            )

    def test_semantic_depth_matches_server_demo(self):
        self.assertEqual(semantic_depth("pick up the screwdriver"), -1)
        self.assertEqual(semantic_depth("use the mallet"), 1)
        self.assertEqual(semantic_depth("pick up the box"), 1)
        self.assertEqual(semantic_depth("pick up the wrench"), -1)
        self.assertEqual(semantic_depth("pick up the crimp tool"), 0)
        self.assertEqual(semantic_depth("unknown item", default=-1), -1)

    def test_semantic_depth_accepts_lab_overrides(self):
        self.assertEqual(
            semantic_depth(
                "pick up the screwdriver",
                class_tiers={"screwdriver": "L2"},
            ),
            0,
        )

    def test_mask_span_width_follows_grasp_axis(self):
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[15:25, 10:50] = 1
        self.assertAlmostEqual(
            mask_span_width(mask, (30, 20), 0, step=1, safety_margin=4),
            43.0,
        )
        self.assertAlmostEqual(
            mask_span_width(mask, (30, 20), 90, step=1, safety_margin=4),
            13.0,
        )

    def test_mask_span_policy_can_exclude_tape(self):
        mask = np.ones((20, 20), dtype=np.uint8)
        cfg = {
            "type": "mask_span",
            "step": 1,
            "safety_margin": 3,
            "exclude": ["tape", "cable"],
        }
        self.assertEqual(command_width(7, mask, (10, 10), 0, "the tape", cfg), 7)
        self.assertGreater(
            command_width(7, mask, (10, 10), 0, "the tape measure", cfg), 7
        )
        self.assertGreater(command_width(7, mask, (10, 10), 0, "the box", cfg), 7)

    def test_gui_preview_does_not_mirror_decoded_angle(self):
        rectangle = opencv_grasp_rectangle([640, 360, 120, 20, 35])
        self.assertEqual(rectangle, ((640.0, 360.0), (120.0, 20.0), 35.0))
        self.assertEqual(
            command_theta(35, {"offset_degrees": 180, "normalization": "zero_360"}),
            215,
        )

    def test_theta_policy_reproduces_legacy_wire_convention(self):
        self.assertEqual(
            command_theta(
                -25,
                {"offset_degrees": 180, "normalization": "zero_360"},
            ),
            155,
        )

    def test_config_defaults_keep_robot_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deploy.yaml"
            path.write_text(yaml.safe_dump({"model": {"prompt": "wrench"}}), encoding="utf-8")
            cfg = load_deployment_config(path, repo_root=directory)
        self.assertFalse(cfg["robot"]["enabled"])
        self.assertFalse(cfg["robot"]["auto_send"])
        self.assertEqual(cfg["robot"]["coordinate_space"], "source")
        self.assertEqual(cfg["robot"]["width_policy"]["type"], "model")
        self.assertFalse(cfg["gelsight"]["enabled"])
        self.assertEqual(cfg["model"]["prompt"], "wrench")

    def test_lab_profile_uses_physical_realsense_resolution(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_deployment_config(
            str(repo_root / "config" / "deployment" / "lab.yaml")
        )
        self.assertEqual(cfg["camera"]["type"], "realsense")
        self.assertEqual((cfg["camera"]["width"], cfg["camera"]["height"]), (1280, 720))
        self.assertEqual(cfg["robot"]["coordinate_space"], "source")
        self.assertEqual(cfg["robot"]["theta_policy"]["offset_degrees"], 180.0)
        self.assertEqual(len(cfg["detector"]["classes"]), 13)
        self.assertEqual(cfg["detector"]["classes"][0], "box")
        self.assertEqual(cfg["detector"]["classes"][-1], "wrench")
        self.assertEqual(cfg["detector"]["checkpoint"], "weights/epoch_48_13.pth")
        self.assertTrue(cfg["detector"]["enabled"])
        self.assertTrue(cfg["detector"]["trusted_checkpoint"])
        self.assertTrue(cfg["robot"]["enabled"])
        self.assertTrue(cfg["robot"]["auto_connect"])
        self.assertTrue(cfg["robot"]["auto_arm"])
        self.assertTrue(cfg["robot"]["auto_send"])

    def test_detector_live_inference_pipeline_needs_no_annotations(self):
        repo_root = Path(__file__).resolve().parents[1]
        detector_cfg = runpy.run_path(
            str(repo_root / "config" / "deployment" / "faster-rcnn-13.py")
        )
        pipeline_types = [
            transform["type"] for transform in detector_cfg["test_pipeline"]
        ]
        self.assertNotIn("LoadAnnotations", pipeline_types)
        self.assertEqual(len(detector_cfg["metainfo"]["classes"]), 13)
        self.assertEqual(
            detector_cfg["model"]["roi_head"]["bbox_head"]["num_classes"], 13
        )

    def test_trusted_detector_allowlists_only_history_buffer(self):
        fake_context = mock.Mock()
        fake_torch = mock.Mock()
        fake_torch.serialization.safe_globals.return_value = fake_context
        fake_history_module = mock.Mock()

        class HistoryBuffer:
            pass

        fake_history_module.HistoryBuffer = HistoryBuffer
        with mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "mmengine": mock.Mock(),
                "mmengine.logging": mock.Mock(),
                "mmengine.logging.history_buffer": fake_history_module,
            },
        ):
            context = _trusted_mmengine_checkpoint_context(True)
        self.assertIs(context, fake_context)
        fake_torch.serialization.safe_globals.assert_called_once_with(
            [HistoryBuffer]
        )

    def test_untrusted_detector_keeps_default_safe_context(self):
        with _trusted_mmengine_checkpoint_context(False):
            pass

    def test_trusted_detector_disables_weights_only_for_exact_file(self):
        with tempfile.TemporaryDirectory() as directory:
            trusted = Path(directory) / "detector.pth"
            other = Path(directory) / "other.pth"
            original_load = mock.Mock(return_value={"state_dict": {}})
            fake_torch = mock.Mock()
            fake_torch.load = original_load
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with _trusted_checkpoint_load_context(True, trusted):
                    fake_torch.load(str(trusted), map_location="cpu")
                    fake_torch.load(str(other), map_location="cpu")
            self.assertIs(fake_torch.load, original_load)
            self.assertEqual(
                original_load.call_args_list,
                [
                    mock.call(
                        str(trusted), map_location="cpu", weights_only=False
                    ),
                    mock.call(str(other), map_location="cpu"),
                ],
            )

    def test_model_profiles_can_be_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deploy.yaml"
            path.write_text(
                "active_model: drogoff\n"
                "model_profiles:\n"
                "  crog:\n"
                "    config: config/vcot/crog.yaml\n"
                "    checkpoint: weights/crog.pth\n"
                "  drogoff:\n"
                "    config: config/vcot/drogoff.yaml\n"
                "    checkpoint: weights/drogoff.pth\n",
                encoding="utf-8",
            )
            cfg = load_deployment_config(path)
            self.assertEqual(cfg["_active_model"], "drogoff")
            selected = activate_model_profile(cfg, "crog")
            self.assertEqual(selected["_active_model"], "crog")
            self.assertEqual(
                selected["model"]["checkpoint"], "weights/crog.pth"
            )

    def test_existing_checkpoint_is_accepted_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            path.write_bytes(b"toolrgs-test")
            self.assertEqual(ensure_deployment_checkpoint(path), path)

    def test_qt_plugin_path_prefers_active_pyqt_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin_root = Path(directory) / "plugins"
            platforms = plugin_root / "platforms"
            platforms.mkdir(parents=True)

            class FakeLibraryInfo:
                PluginsPath = object()

                @staticmethod
                def location(_):
                    return str(plugin_root)

            fake_qtcore = mock.Mock()
            fake_qtcore.QLibraryInfo = FakeLibraryInfo
            with mock.patch.dict(
                sys.modules,
                {
                    "PyQt5": mock.Mock(),
                    "PyQt5.QtCore": fake_qtcore,
                },
            ), mock.patch.dict(
                "os.environ",
                {"QT_QPA_PLATFORM_PLUGIN_PATH": "/bad/cv2/qt/plugins"},
                clear=False,
            ):
                resolved = configure_pyqt5_plugins()
                self.assertEqual(resolved, platforms.resolve())
                self.assertEqual(
                    __import__("os").environ["QT_QPA_PLATFORM_PLUGIN_PATH"],
                    str(platforms.resolve()),
                )


if __name__ == "__main__":
    unittest.main()
