import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools.dataset_converters.grasp_tools.to_coco_detection import (
    CLASSES,
    convert_split,
)


class GraspToolsDetectionConverterTest(unittest.TestCase):
    def test_objects_are_converted_once_and_queries_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "train"
            split.mkdir()
            image = np.zeros((64, 96, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(split / "scene.jpg"), image))
            objects = [
                {
                    "category": name,
                    "bbox": [10, 10, 30, 30],
                    "mask": [[10, 10], [30, 10], [30, 30], [10, 30]],
                }
                for name in CLASSES
            ]
            payload = {
                "objects": objects,
                "queries": [
                    {"text": "the tool", "target_idx": 0},
                    {"text": "pick the same tool", "target_idx": 0},
                ],
            }
            (split / "scene.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            output = root / "annotations" / "train.json"
            summary = convert_split(root, "train", output)
            coco = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary["images"], 1)
        self.assertEqual(summary["annotations"], len(CLASSES))
        self.assertEqual([item["name"] for item in coco["categories"]], list(CLASSES))
        self.assertEqual(coco["annotations"][0]["bbox"], [10.0, 10.0, 20.0, 20.0])
        self.assertEqual(coco["annotations"][0]["area"], 400.0)


if __name__ == "__main__":
    unittest.main()
