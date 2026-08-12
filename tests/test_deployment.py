import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import yaml

from deployment.config import activate_model_profile, load_deployment_config
from deployment.grasp_policy import command_theta, command_width, mask_span_width
from deployment.robot import GraspCommand, LegacyTCPGraspClient, semantic_depth
from deployment.weights import ensure_deployment_checkpoint


class DeploymentContractTest(unittest.TestCase):
    def test_legacy_wire_protocol(self):
        command = GraspCommand(12.5, 33, -45.25, 80, 1)
        self.assertEqual(command.to_wire(), b"{12.5, 33, -45.25, 80, 1}\n")

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
            str(repo_root / "config" / "deployment" / "lab.example.yaml")
        )
        self.assertEqual(cfg["camera"]["type"], "realsense")
        self.assertEqual((cfg["camera"]["width"], cfg["camera"]["height"]), (1280, 720))
        self.assertEqual(cfg["robot"]["coordinate_space"], "source")
        self.assertEqual(cfg["robot"]["theta_policy"]["offset_degrees"], 180.0)

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


if __name__ == "__main__":
    unittest.main()
