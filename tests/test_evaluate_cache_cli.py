import sys
import unittest
from unittest.mock import patch

import evaluate


class EvaluateCacheCliTest(unittest.TestCase):
    def test_score_cache_does_not_require_model_config(self):
        with patch.object(
            sys,
            "argv",
            ["evaluate.py", "--score-cache", "predictions.npz"],
        ):
            args = evaluate.parse_args()

        self.assertEqual(args.score_cache, "predictions.npz")
        self.assertIsNone(args.config)
        self.assertEqual(args.grasp_topk, (1, 5))
        self.assertEqual(args.grasp_iou_thresholds, (0.25, 0.50, 0.75))
        self.assertEqual(
            args.grasp_angle_thresholds,
            (5.0, 10.0, 20.0, 30.0),
        )

    def test_model_inference_still_requires_config(self):
        with patch.object(
            sys,
            "argv",
            ["evaluate.py", "--checkpoint", "best.pth"],
        ):
            with self.assertRaises(SystemExit):
                evaluate.parse_args()


if __name__ == "__main__":
    unittest.main()
