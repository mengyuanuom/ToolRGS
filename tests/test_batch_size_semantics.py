import unittest

from toolrgs.engine.batch import per_process_batch_size


class BatchSizeSemanticsTest(unittest.TestCase):
    def test_global_batch_is_split_across_eight_ranks(self):
        self.assertEqual(per_process_batch_size(128, 8, "batch_size"), 16)

    def test_non_divisible_global_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            per_process_batch_size(10, 8, "batch_size")


if __name__ == "__main__":
    unittest.main()
