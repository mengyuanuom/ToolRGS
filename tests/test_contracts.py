import glob
import unittest

import numpy as np
import torch
import yaml

from model import MODEL_REGISTRY
from model.layers import OffsetMultiTaskProjector
from utils.dataset import make_dense_offset_with_radius_np


class ToolRGSContractsTest(unittest.TestCase):
    def test_experiment_configs_reference_registered_models(self):
        expected = {
            "crog",
            "crogoff",
            "drog",
            "drogoff",
            "ggcnnclip",
            "grconvnetclip",
        }
        self.assertTrue(expected.issubset(MODEL_REGISTRY))
        paths = glob.glob("config/grasp_tools/*.yaml")
        self.assertGreaterEqual(len(paths), 6)
        for path in paths:
            with open(path, encoding="utf-8") as stream:
                cfg = yaml.safe_load(stream)
            architecture = cfg["MODEL"]["architecture"]
            self.assertIn(architecture, MODEL_REGISTRY, path)

    def test_offset_projector_output_contract(self):
        projector = OffsetMultiTaskProjector(word_dim=512, in_dim=256)
        features = torch.randn(2, 512, 8, 8)
        text_state = torch.randn(2, 512)
        outputs = projector(features, text_state)
        self.assertEqual(len(outputs), 6)
        for output in outputs[:5]:
            self.assertEqual(tuple(output.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(outputs[5].shape), (2, 2, 32, 32))
        self.assertLessEqual(outputs[5].abs().max().item(), 1.0)

    def test_dense_offset_points_toward_grasp_center(self):
        center = np.array([[8.0, 9.0]], dtype=np.float32)
        offset, weight = make_dense_offset_with_radius_np(
            centers_xy=center,
            img_size_hw=(20, 20),
            r_pix=4.0,
            use_gaussian=True,
        )
        self.assertEqual(offset.shape, (2, 20, 20))
        self.assertEqual(weight.shape, (1, 20, 20))
        np.testing.assert_allclose(offset[:, 9, 8], 0.0, atol=1e-6)
        self.assertGreater(offset[0, 9, 6], 0.0)
        self.assertGreater(weight[0, 9, 8], weight[0, 9, 6])
        self.assertLessEqual(np.abs(offset).max(), 1.0)


if __name__ == "__main__":
    unittest.main()
