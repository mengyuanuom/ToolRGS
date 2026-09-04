import ast
from pathlib import Path
import unittest

from utils.config import load_cfg_from_cfg_file


ROOT = Path(__file__).resolve().parents[1]


class MapleGraspContractTests(unittest.TestCase):
    def test_unified_stage2_uses_fresh_sigmoid_contract(self):
        cfg = load_cfg_from_cfg_file(
            str(
                ROOT
                / "config"
                / "grasp_tools"
                / "maplegrasp_v3_stage2_unified_original300_retrain.yaml"
            )
        )
        self.assertTrue(cfg.align_grasp_quality_loss)
        self.assertTrue(cfg.align_grasp_size_loss)
        self.assertEqual(cfg.grasp_quality_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_activation, "sigmoid")
        self.assertEqual(cfg.resume, None)
        self.assertEqual(
            cfg.exp_name,
            "maplegrasp_stage2_grasp_tools_v3_15k_unified_original300_sigmoid_consistent",
        )

    def test_quality_and_width_losses_apply_sigmoid(self):
        tree = ast.parse((ROOT / "model" / "maplegrasp.py").read_text())
        sigmoid_inputs = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "sigmoid"
                and node.args
            ):
                continue
            argument = ast.unparse(node.args[0])
            if argument in {"outputs[1]", "outputs[index]"}:
                sigmoid_inputs.add(argument)
        self.assertEqual(sigmoid_inputs, {"outputs[1]", "outputs[index]"})


if __name__ == "__main__":
    unittest.main()
