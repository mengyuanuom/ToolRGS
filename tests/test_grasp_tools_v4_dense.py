import random
from pathlib import Path
import unittest

from tools.dataset_converters.grasp_tools.augment import (
    SuccessBalancedCategorySampler,
    adaptive_scale_values,
)
from utils.config import load_cfg_from_cfg_file


ROOT = Path(__file__).resolve().parents[1]


class GraspToolsV4DenseTest(unittest.TestCase):
    def test_adaptive_scales_stop_at_inclusive_floor(self):
        self.assertEqual(
            adaptive_scale_values(0.6, 0.3, 0.1),
            (0.6, 0.5, 0.4, 0.3),
        )
        self.assertEqual(adaptive_scale_values(0.3, 0.3, 0.1), (0.3,))

    def test_failed_category_proposals_do_not_consume_balance(self):
        sampler = SuccessBalancedCategorySampler(["a", "b"], random.Random(7))
        first = sampler.propose(set(), {})
        self.assertIn(first, {"a", "b"})
        self.assertEqual(dict(sampler.success_counts), {})
        sampler.commit([first])
        second = sampler.propose(set(), {})
        self.assertNotEqual(first, second)

    def test_v4_training_profiles_are_short_sigmoid_consistent_runs(self):
        profiles = (
            ("drogoff_v1_v4_dense_15k_sigmoid_e12.yaml", "drogoff", 8),
            ("crog_v4_dense_15k_sigmoid_e12.yaml", "crog", 32),
        )
        for filename, architecture, batch_size in profiles:
            with self.subTest(filename=filename):
                cfg = load_cfg_from_cfg_file(
                    ROOT / "config" / "grasp_tools" / filename
                )
                self.assertEqual(cfg.architecture, architecture)
                self.assertTrue(cfg.root_path.endswith("aug_graspall_v4_dense_15k"))
                self.assertEqual(cfg.epochs, 12)
                self.assertEqual(cfg.milestones, [10])
                self.assertEqual(cfg.batch_size, batch_size)
                self.assertEqual(cfg.grasp_quality_loss_activation, "sigmoid")
                self.assertEqual(cfg.grasp_width_loss_activation, "sigmoid")
                self.assertEqual(cfg.grasp_quality_activation, "sigmoid")
                self.assertEqual(cfg.grasp_size_activation, "sigmoid")
                self.assertIsNone(cfg.weight)
                self.assertIsNone(cfg.resume)


if __name__ == "__main__":
    unittest.main()
