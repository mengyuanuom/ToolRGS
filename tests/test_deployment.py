import tempfile
from pathlib import Path
import unittest

import yaml

from deployment.config import activate_model_profile, load_deployment_config
from deployment.robot import GraspCommand, semantic_depth
from deployment.weights import ensure_deployment_checkpoint


class DeploymentContractTest(unittest.TestCase):
    def test_legacy_wire_protocol(self):
        command = GraspCommand(12.5, 33, -45.25, 80, 1)
        self.assertEqual(command.to_wire(), b"{12.5, 33, -45.25, 80, 1}\n")

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
        self.assertEqual(semantic_depth("pick up the screwdriver"), 0)
        self.assertEqual(semantic_depth("use the mallet"), 1)
        self.assertEqual(semantic_depth("unknown item", default=-1), -1)

    def test_config_defaults_keep_robot_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deploy.yaml"
            path.write_text(yaml.safe_dump({"model": {"prompt": "wrench"}}), encoding="utf-8")
            cfg = load_deployment_config(path, repo_root=directory)
        self.assertFalse(cfg["robot"]["enabled"])
        self.assertFalse(cfg["robot"]["auto_send"])
        self.assertEqual(cfg["robot"]["coordinate_space"], "source")
        self.assertEqual(cfg["model"]["prompt"], "wrench")

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
