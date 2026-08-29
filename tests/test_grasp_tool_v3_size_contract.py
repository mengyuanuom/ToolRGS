import os
import tempfile
import unittest

import numpy as np

from utils.grasp_tool_dataset import GraspToolDataset, GraspToolTransforms


class GraspToolV3SizeContractTest(unittest.TestCase):
    def test_dataset_accepts_original_size_contract(self):
        with tempfile.TemporaryDirectory() as root:
            split_root = os.path.join(root, "train")
            os.makedirs(split_root)
            with open(
                os.path.join(split_root, "index.jsonl"), "w", encoding="utf-8"
            ) as stream:
                stream.write(
                    '{"image":"sample.jpg","annotation":"sample.json",'
                    '"query_index":0}\n'
                )
            dataset = GraspToolDataset(
                root,
                split="train",
                grasp_size_factor=300.0,
                grasp_size_coordinate="original",
            )
        self.assertEqual(dataset.grasp_transform.width_factor, 300.0)
        self.assertEqual(dataset.grasp_size_coordinate, "original")

    def test_masks_separate_canvas_geometry_from_original_size(self):
        transform = GraspToolTransforms(width_factor=300.0, width=32, height=32)
        canvas_grasp = np.array(
            [[16.0, 16.0, 20.0, 8.0, 0.0, 0.0]], dtype=np.float32
        )
        original_size = np.array(
            [[16.0, 16.0, 150.0, 30.0, 0.0, 0.0]], dtype=np.float32
        )
        raw = transform.generate_masks(
            canvas_grasp, size_rectangles=original_size
        )
        self.assertAlmostEqual(float(raw["wid"][16, 16]), 0.5, places=5)
        self.assertGreater(float(raw["qua"][16, 16]), 0.9)


if __name__ == "__main__":
    unittest.main()
