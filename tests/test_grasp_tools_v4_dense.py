import random
import unittest

from tools.dataset_converters.grasp_tools.augment import (
    SuccessBalancedCategorySampler,
    adaptive_scale_values,
)


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


if __name__ == "__main__":
    unittest.main()
