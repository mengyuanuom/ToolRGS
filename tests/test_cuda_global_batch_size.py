from types import SimpleNamespace
import unittest

from toolrgs.engine.runner import CUDAGraspRunner


class CUDAGlobalBatchSizeTest(unittest.TestCase):
    def test_opt_in_global_batch_is_divided_across_workers(self):
        runner = CUDAGraspRunner.__new__(CUDAGraspRunner)
        runner.cfg = SimpleNamespace(
            batch_size=128,
            batch_size_val=32,
            batch_size_is_global=True,
            world_size=8,
        )
        runner._configure_batch_sizes()
        self.assertEqual(runner.train_batch_size_per_process, 16)
        self.assertEqual(runner.val_batch_size_per_process, 4)

    def test_legacy_configs_keep_per_process_batch_semantics(self):
        runner = CUDAGraspRunner.__new__(CUDAGraspRunner)
        runner.cfg = SimpleNamespace(
            batch_size=24,
            batch_size_val=8,
            batch_size_is_global=False,
            world_size=8,
        )
        runner._configure_batch_sizes()
        self.assertEqual(runner.train_batch_size_per_process, 24)
        self.assertEqual(runner.val_batch_size_per_process, 8)

    def test_global_batch_must_divide_world_size(self):
        runner = CUDAGraspRunner.__new__(CUDAGraspRunner)
        runner.cfg = SimpleNamespace(
            batch_size=130,
            batch_size_val=32,
            batch_size_is_global=True,
            world_size=8,
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            runner._configure_batch_sizes()


if __name__ == "__main__":
    unittest.main()
