import hashlib
import inspect
import io
import tempfile
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml
import torch

from deployment.config import activate_model_profile, load_deployment_config
from deployment.detector import (
    MMDetectionAdapter,
    _trusted_checkpoint_load_context,
    _trusted_mmengine_checkpoint_context,
)
from deployment.grasp_policy import command_theta, command_width, mask_span_width
from deployment.gui import build_grasp_command, format_grasp_prompt, normalize_prompt_text
from deployment.inference import (
    GraspPrediction,
    expand_binary_mask,
    opencv_grasp_rectangle,
)
from deployment.qt import configure_pyqt5_plugins
from deployment.robot import GraspCommand, LegacyTCPGraspClient, semantic_depth
from deployment.weights import ensure_deployment_checkpoint
from model.crog import grasp_width_for_loss
from model.etrg.model import ETRG, grasp_width_for_loss as etrg_grasp_width_for_loss
from utils.config import resolve_grasp_size_activation
from deploy_gui import (
    DEFAULT_SAMPLE_IMAGE,
    apply_camera_preset,
    apply_runtime_overrides,
    parse_args,
)
from deploy_gui_gi import prepare_gi_config
from deploy_gui_realsense import prepare_realsense_demo_config

check_deployment_symbols = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "check_deployment.py")
)
parse_check_args = check_deployment_symbols["parse_args"]
experiment_values = check_deployment_symbols["_experiment_values"]


class DeploymentContractTest(unittest.TestCase):
    def test_etrg_width_training_contract_is_sigmoid(self):
        raw = torch.tensor([-2.0, 0.0, 2.0])
        expected = torch.sigmoid(raw)
        self.assertTrue(torch.equal(etrg_grasp_width_for_loss(raw), expected))

    def test_etrg_rgb_forward_accepts_common_two_argument_contract(self):
        signature = inspect.signature(ETRG.forward)
        self.assertIsNone(signature.parameters["word"].default)

    def test_explicit_grasp_activation_must_match_checkpoint_metadata(self):
        self.assertEqual(
            resolve_grasp_size_activation(
                "clamp", checkpoint={"grasp_size_activation": "clamp"}
            ),
            "clamp",
        )
        with self.assertRaisesRegex(ValueError, "conflicts with checkpoint metadata"):
            resolve_grasp_size_activation(
                "sigmoid", checkpoint={"grasp_size_activation": "clamp"}
            )

    def test_image_cli_uses_bundled_sample_by_default(self):
        args = parse_args(["--image"])
        self.assertEqual(args.image, DEFAULT_SAMPLE_IMAGE)

    def test_gui_prompt_normalization_rejects_whitespace(self):
        self.assertEqual(normalize_prompt_text("  the screwdriver  "), "the screwdriver")
        self.assertEqual(normalize_prompt_text(" \t\n "), "")
        self.assertEqual(normalize_prompt_text(None), "")
        self.assertEqual(format_grasp_prompt("Grasp"), "Grasp")
        self.assertEqual(format_grasp_prompt(""), "")

    def test_deployment_check_downloads_missing_weights_by_default(self):
        self.assertTrue(parse_check_args([]).download_weights)
        self.assertFalse(parse_check_args(["--no-download-weights"]).download_weights)

    def test_deployment_check_resolves_inherited_pretrained_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text(
                "TRAIN:\n  clip_pretrain: pretrain/RN50.pt\n  epochs: 36\n",
                encoding="utf-8",
            )
            child = root / "child.yaml"
            child.write_text(
                "_base_: base.yaml\nTRAIN:\n  epochs: 12\n",
                encoding="utf-8",
            )
            values = experiment_values(child)
        self.assertEqual(values["clip_pretrain"], "pretrain/RN50.pt")
        self.assertEqual(values["epochs"], 12)

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

    def test_realsense_demo_disables_robot_side_effects(self):
        config = {
            "camera": {"type": "image"},
            "robot": {
                "enabled": True,
                "auto_connect": True,
                "auto_arm": True,
                "auto_send": True,
            },
        }
        updated = prepare_realsense_demo_config(config)
        self.assertEqual(updated["camera"]["type"], "realsense")
        self.assertFalse(updated["robot"]["enabled"])
        self.assertFalse(updated["robot"]["auto_connect"])
        self.assertFalse(updated["robot"]["auto_arm"])
        self.assertFalse(updated["robot"]["auto_send"])

    def test_gi_camera_preset_uses_validated_shared_memory_socket(self):
        config = {"camera": {"type": "realsense"}}
        updated = apply_camera_preset(config, "gi")
        self.assertEqual(updated["camera"]["type"], "gstreamer")
        self.assertIn(
            "socket-path=/home/raico-hri/v1/kinova_rs_grasp/foo/fooA",
            updated["camera"]["gstreamer_pipeline"],
        )
        self.assertIn("format=BGR,width=1280,height=720", updated["camera"]["gstreamer_pipeline"])
        self.assertEqual(updated["robot"]["timeout_s"], 2.0)

    def test_designed_gi_uses_model_width_and_two_second_timeout(self):
        config = {
            "camera": {"type": "realsense"},
            "robot": {
                "timeout_s": None,
                "width_policy": {"type": "mask_span"},
            },
        }
        updated = prepare_gi_config(config)
        self.assertEqual(updated["camera"]["type"], "gstreamer")
        self.assertEqual(updated["robot"]["timeout_s"], 2.0)
        self.assertEqual(updated["robot"]["width_policy"]["type"], "model")

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

    def test_mask_expansion_uses_source_pixel_radius(self):
        mask = np.zeros((11, 11), dtype=np.uint8)
        mask[5, 5] = 1
        expanded = expand_binary_mask(mask, 2)
        self.assertTrue(expanded[5, 3])
        self.assertTrue(expanded[5, 7])
        self.assertTrue(expanded[3, 5])
        self.assertTrue(expanded[7, 5])
        self.assertFalse(expanded[5, 2])

    def test_aligned_crog_width_loss_uses_sigmoid_contract(self):
        raw = torch.tensor([-2.0, 0.0, 2.0])
        self.assertTrue(
            torch.equal(grasp_width_for_loss(raw, "raw"), raw)
        )
        bounded = grasp_width_for_loss(raw, "sigmoid")
        self.assertTrue(torch.all((bounded > 0.0) & (bounded < 1.0)))
        self.assertAlmostEqual(float(bounded[1]), 0.5)

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
        self.assertNotIn("prompt_template", cfg["model"])
        self.assertEqual(cfg["detector"]["nms_threshold"], 0.5)
        self.assertEqual(cfg["gui"]["theme"], "Midnight Teal")

    def test_grasp_prompt_is_free_form(self):
        self.assertEqual(format_grasp_prompt("screwdriver"), "screwdriver")
        self.assertEqual(
            format_grasp_prompt("the left wrench"), "the left wrench"
        )
        self.assertEqual(
            format_grasp_prompt("Grasp the left wrench"),
            "Grasp the left wrench",
        )
        self.assertEqual(format_grasp_prompt("", "Grasp {}"), "")

    def test_detector_runtime_controls_update_mmdet_test_cfg(self):
        adapter = MMDetectionAdapter.__new__(MMDetectionAdapter)
        roi_test_cfg = {
            "score_thr": 0.05,
            "nms": {"type": "nms", "iou_threshold": 0.5},
            "max_per_img": 100,
        }
        adapter.model = SimpleNamespace(
            test_cfg={
                "rcnn": {
                    "score_thr": 0.05,
                    "nms": {"type": "nms", "iou_threshold": 0.5},
                    "max_per_img": 100,
                }
            },
            roi_head=SimpleNamespace(test_cfg=roi_test_cfg),
        )

        adapter.update_postprocessing(0.8, 0.35, 7)

        rcnn_cfg = adapter.model.test_cfg["rcnn"]
        self.assertEqual(adapter.threshold, 0.8)
        self.assertEqual(adapter.nms_threshold, 0.35)
        self.assertEqual(adapter.max_detections, 7)
        self.assertEqual(rcnn_cfg["score_thr"], 0.8)
        self.assertEqual(rcnn_cfg["nms"]["iou_threshold"], 0.35)
        self.assertEqual(rcnn_cfg["max_per_img"], 7)
        self.assertEqual(roi_test_cfg["score_thr"], 0.8)
        self.assertEqual(roi_test_cfg["nms"]["iou_threshold"], 0.35)
        self.assertEqual(roi_test_cfg["max_per_img"], 7)

    def test_detector_runtime_controls_validate_values(self):
        adapter = MMDetectionAdapter.__new__(MMDetectionAdapter)
        adapter.model = SimpleNamespace(test_cfg={})
        for values in ((-0.1, 0.5, 10), (0.5, 1.1, 10), (0.5, 0.5, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                adapter.update_postprocessing(*values)

    def test_lab_profile_uses_physical_realsense_resolution(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_deployment_config(
            str(repo_root / "config" / "deployment" / "lab.yaml")
        )
        self.assertEqual(cfg["camera"]["type"], "realsense")
        self.assertEqual((cfg["camera"]["width"], cfg["camera"]["height"]), (1280, 720))
        self.assertEqual(cfg["robot"]["coordinate_space"], "source")
        self.assertEqual(cfg["robot"]["theta_policy"]["offset_degrees"], 180.0)
        self.assertEqual(len(cfg["detector"]["classes"]), 22)
        self.assertEqual(cfg["detector"]["classes"][0], "tape measure")
        self.assertEqual(cfg["detector"]["classes"][-1], "cable")
        self.assertEqual(
            cfg["detector"]["checkpoint"],
            "weights/faster_rcnn_r50_fpn_grasp_tools_v2_best.pth",
        )
        self.assertEqual(
            cfg["detector"]["checkpoint_sha256"],
            "76a4a09164f5de1a410957f4439f801328cf543f28f34fa6fee24a7f7eb49e74",
        )
        self.assertTrue(cfg["detector"]["enabled"])
        self.assertTrue(cfg["detector"]["trusted_checkpoint"])
        self.assertEqual(cfg["detector"]["nms_threshold"], 0.5)
        self.assertTrue(cfg["robot"]["enabled"])
        self.assertTrue(cfg["robot"]["auto_connect"])
        self.assertTrue(cfg["robot"]["auto_arm"])
        self.assertTrue(cfg["robot"]["auto_send"])
        self.assertEqual(
            list(cfg["_model_profiles"]),
            [
                "V3-DROG-OFF-V1",
                "V3-CROG",
                "drogoff-grasptools-v2-original300",
                "crog-aligned-grasptools-v2-original300",
            ],
        )
        self.assertTrue(cfg["model"]["use_mask_postprocessing"])
        self.assertTrue(cfg["model"]["filter_grasps_by_mask"])
        self.assertNotIn("prompt_template", cfg["model"])
        self.assertEqual(cfg["model"]["prompt"], "Grasp the screwdriver")
        self.assertEqual(cfg["model"]["mask_expand_px"], 0)
        crog = activate_model_profile(
            cfg, "crog-aligned-grasptools-v2-original300"
        )
        self.assertEqual(
            crog["model"]["postprocessor"]["grasp_height"], 20.0
        )
        self.assertEqual(
            crog["model"]["postprocessor"]["size_coordinate"], "original"
        )
        self.assertEqual(
            crog["model"]["checkpoint_sha256"],
            "6b2f1059448d5c4fc7486c5c66e51929acaf12bafeb827bb759f2e8f941935e2",
        )
        v3_v1 = activate_model_profile(
            cfg, "V3-DROG-OFF-V1"
        )
        self.assertEqual(
            v3_v1["model"]["config"],
            "config/grasp_tools/v3_drogoff_v1_grasp_tools_15k_original_scale.yaml",
        )
        self.assertEqual(
            v3_v1["model"]["checkpoint_sha256"],
            "5f15b5f59e783b9daf3b34bf1d467274591c15fc6f590c36653f128b90dff340",
        )
        self.assertEqual(v3_v1["model"]["grasp_size_activation"], "sigmoid")
        v3_crog = activate_model_profile(cfg, "V3-CROG")
        self.assertEqual(
            v3_crog["model"]["config"],
            "config/grasp_tools/v3_crog_grasp_tools_15k_original_scale.yaml",
        )
        self.assertEqual(
            v3_crog["model"]["checkpoint_sha256"],
            "2d1270024beedde710b8a78b83c83591d3166debed479ad20450a88b80530a4f",
        )
        self.assertEqual(v3_crog["model"]["grasp_size_activation"], "clamp")

    def test_detector_live_inference_pipeline_needs_no_annotations(self):
        repo_root = Path(__file__).resolve().parents[1]
        detector_cfg = runpy.run_path(
            str(
                repo_root
                / "configs"
                / "detection"
                / "faster_rcnn_r50_fpn_grasp_tools_v2_24e.py"
            )
        )
        pipeline_types = [
            transform["type"] for transform in detector_cfg["test_pipeline"]
        ]
        self.assertNotIn("LoadAnnotations", pipeline_types)
        self.assertEqual(len(detector_cfg["metainfo"]["classes"]), 22)
        self.assertEqual(
            detector_cfg["model"]["roi_head"]["bbox_head"]["num_classes"], 22
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

    def test_stale_checkpoint_is_atomically_replaced_from_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            path.write_bytes(b"old-checkpoint")
            payload = b"new-verified-checkpoint"
            expected = hashlib.sha256(payload).hexdigest()
            with mock.patch(
                "urllib.request.urlopen", return_value=io.BytesIO(payload)
            ):
                resolved = ensure_deployment_checkpoint(
                    path, "https://example.invalid/model.pth", expected
                )
            self.assertEqual(resolved, path)
            self.assertEqual(path.read_bytes(), payload)
            self.assertFalse(path.with_suffix(".pth.part").exists())

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
