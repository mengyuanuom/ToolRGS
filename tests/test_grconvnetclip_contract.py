import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GRConvNetCLIPContractTest(unittest.TestCase):
    def test_quality_and_width_training_losses_use_sigmoid(self):
        source = (ROOT / "model" / "grconvnetclip.py").read_text(
            encoding="utf-8"
        )
        module = ast.parse(source)
        wrapper = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "GenerativeResnet_CLIP"
        )

        metadata = {}
        for node in wrapper.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(
                    node.value, ast.Constant
                ):
                    metadata[target.id] = node.value.value
        self.assertEqual(metadata["grasp_quality_loss_activation"], "sigmoid")
        self.assertEqual(metadata["grasp_size_loss_activation"], "sigmoid")

        forward = next(
            node
            for node in wrapper.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        sigmoid_loss_inputs = set()
        for node in ast.walk(forward):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr != "smooth_l1_loss" or not node.args:
                continue
            prediction = node.args[0]
            if (
                isinstance(prediction, ast.Call)
                and isinstance(prediction.func, ast.Attribute)
                and isinstance(prediction.func.value, ast.Name)
                and prediction.func.value.id == "torch"
                and prediction.func.attr == "sigmoid"
                and prediction.args
                and isinstance(prediction.args[0], ast.Name)
            ):
                sigmoid_loss_inputs.add(prediction.args[0].id)

        self.assertEqual(sigmoid_loss_inputs, {"qua_pred", "wid_pred"})


if __name__ == "__main__":
    unittest.main()
