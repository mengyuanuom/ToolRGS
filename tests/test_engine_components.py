import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from toolrgs.engine.hooks import Hook, HookList, LoopState
from toolrgs.evaluation import (
    BinarySegmentationMetric,
    DenseGraspPostProcessor,
    GraspSuccessMetric,
    GraspThresholdGridMetric,
    load_prediction_cache,
    save_prediction_cache,
    score_prediction_cache,
    corners_to_five,
    five_to_corners,
    inverse_warp,
    refine_with_grasp_relative_offset,
    refine_with_offset,
    resample_grasp_geometry,
    targets_to_six,
)
from toolrgs.registry import HOOKS, METRICS, POSTPROCESSORS
from utils.grasp_eval import (
    calculate_grasp_matches,
    calculate_jacquard_from_matches,
    calculate_jacquard_index,
)


class EvaluationComponentTest(unittest.TestCase):
    def test_rotated_rectangle_round_trip_preserves_geometry(self):
        rectangle = np.array([32.0, 24.0, 18.0, 8.0, 27.0], dtype=np.float32)
        restored = corners_to_five(five_to_corners(rectangle))
        np.testing.assert_allclose(restored[:4], rectangle[:4], atol=1e-4)
        self.assertAlmostEqual(abs(restored[4]), abs(rectangle[4]), places=4)

    def test_inverse_warp_and_offset_refinement_share_coordinate_contract(self):
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        source = np.arange(16, dtype=np.float32).reshape(4, 4)
        np.testing.assert_array_equal(inverse_warp(source, identity, (4, 4)), source)

        offset = np.zeros((2, 32, 32), dtype=np.float32)
        offset[0] = 0.5
        offset[1] = -0.25
        refined = refine_with_offset(
            [[10.0, 20.0, 8.0, 4.0, 0.0]], offset, identity, radius=20
        )
        np.testing.assert_allclose(refined[0][:2], [20.0, 15.0], atol=1e-4)
        np.testing.assert_allclose(refined[0][2:4], [8.0, 4.0], atol=1e-4)

    def test_geometry_is_bilinearly_resampled_at_refined_center(self):
        sine = np.zeros((4, 4), dtype=np.float32)
        cosine = np.ones((4, 4), dtype=np.float32)
        width = np.zeros((4, 4), dtype=np.float32)
        sine[1:3, 1:3] = 1.0
        cosine[1:3, 1:3] = 0.0
        width[1:3, 1:3] = np.array([[0.2, 0.4], [0.6, 0.8]])

        refined = resample_grasp_geometry(
            [[1.5, 1.5, 7.0, 4.0, 0.0]],
            sine,
            cosine,
            width,
            width_factor=100.0,
        )
        np.testing.assert_allclose(refined[0][:2], [1.5, 1.5], atol=1e-6)
        self.assertAlmostEqual(refined[0][2], 50.0, places=5)
        self.assertAlmostEqual(refined[0][3], 4.0, places=5)
        self.assertAlmostEqual(refined[0][4], 45.0, places=5)

    def test_grasp_relative_offset_uses_predicted_rectangle_scale(self):
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        offset = np.zeros((2, 16, 16), dtype=np.float32)
        offset[0] = 1.0
        refined = refine_with_grasp_relative_offset(
            [[5.0, 5.0, 8.0, 6.0, 0.0]], offset, identity
        )
        expected_x = 5.0 + np.hypot(8.0 * 0.25, 6.0 * 0.5)
        np.testing.assert_allclose(
            refined[0][:2], [expected_x, 5.0], atol=1e-4
        )
        np.testing.assert_allclose(refined[0][2:4], [8.0, 6.0], atol=1e-4)

    def test_target_adapter_keeps_six_values_and_expands_other_formats(self):
        six = np.array([10, 20, 30, 40, 50, 7], dtype=np.float32)
        converted = targets_to_six([six, [10, 20, 30, 40, 50]])
        np.testing.assert_array_equal(converted[0], six)
        np.testing.assert_array_equal(
            converted[1], np.array([10, 20, 30, 40, 50, 0], dtype=np.float32)
        )

    def test_dense_grasp_postprocessor_decodes_peak_angle_and_width(self):
        quality = np.zeros((8, 8), dtype=np.float32)
        quality[3, 4] = 0.9
        sine = np.ones_like(quality)
        cosine = np.zeros_like(quality)
        width = np.full_like(quality, 0.5)
        processor = DenseGraspPostProcessor(num_grasps=1)

        detections = processor(
            quality, sine, cosine, width, spatial_scale=2.0
        )
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual((detection.x, detection.y), (4.0, 3.0))
        self.assertAlmostEqual(detection.angle_degrees, 45.0, places=5)
        self.assertAlmostEqual(detection.width, 100.0)
        self.assertAlmostEqual(detection.height, 20.0)

    def test_original_coordinate_width_is_not_scaled_twice(self):
        quality = np.zeros((8, 8), dtype=np.float32)
        quality[3, 4] = 0.9
        sine = np.zeros_like(quality)
        cosine = np.ones_like(quality)
        width = np.full_like(quality, 0.5)
        detection = DenseGraspPostProcessor(
            num_grasps=1,
            width_factor=300.0,
            grasp_height=20.0,
            size_coordinate="original",
        )(
            quality, sine, cosine, width, spatial_scale=1280.0 / 448.0
        )[0]
        self.assertAlmostEqual(detection.width, 150.0)
        self.assertAlmostEqual(detection.height, 20.0)

    def test_dense_grasp_postprocessor_decodes_predicted_short_side(self):
        quality = np.zeros((8, 8), dtype=np.float32)
        quality[3, 4] = 0.9
        sine = np.zeros_like(quality)
        cosine = np.ones_like(quality)
        width = np.full_like(quality, 0.6)
        short_side = np.full_like(quality, 0.2)
        detection = DenseGraspPostProcessor(num_grasps=1)(
            quality, sine, cosine, width, short_side=short_side
        )[0]
        self.assertAlmostEqual(detection.width, 60.0, places=4)
        self.assertAlmostEqual(detection.height, 20.0, places=4)

    def test_binary_segmentation_metric_uses_per_sample_iou(self):
        prediction = np.array(
            [[[1, 1], [0, 0]], [[1, 0], [0, 0]]], dtype=np.float32
        )
        target = np.array(
            [[[1, 0], [0, 0]], [[1, 0], [0, 0]]], dtype=np.uint8
        )
        metric = BinarySegmentationMetric(iou_thresholds=(0.5, 0.9))
        metric.update(prediction, target)
        result = metric.compute()
        self.assertAlmostEqual(result["iou"], 0.75, places=5)
        self.assertEqual(result["precision"], {"Pr@50": 0.5, "Pr@90": 0.5})

    def test_grasp_success_metric_aggregates_each_topk(self):
        metric = GraspSuccessMetric(topk=(1, 5))
        metric.update(1, True)
        metric.update(1, False)
        metric.update(5, True)
        self.assertEqual(metric.compute(), {"J@1": 0.5, "J@5": 1.0})

    def test_grasp_threshold_grid_computes_each_cell_and_msr(self):
        metric = GraspThresholdGridMetric(
            iou_thresholds=(0.25, 0.5),
            angle_thresholds=(10.0,),
            topk=(1, 5),
        )
        for success in (True, False):
            metric.update(0.25, 10.0, 1, success)
        metric.update(0.25, 10.0, 5, True)
        metric.update(0.5, 10.0, 1, False)
        metric.update(0.5, 10.0, 5, True)
        result = metric.compute()
        self.assertEqual(result["rows"][0]["values"], {1: 0.5, 5: 1.0})
        self.assertEqual(result["rows"][1]["values"], {1: 0.0, 5: 1.0})
        self.assertEqual(result["msr"], {1: 0.25, 5: 1.0})

    def test_jacquard_uses_configured_original_width_cap_without_mutation(self):
        predictions = np.array([[100, 100, 250, 20, 0]], dtype=np.float32)
        targets = np.array([[100, 100, 250, 40, 0, 0]], dtype=np.float32)
        observed = []

        def record_iou(_prediction, target, **_kwargs):
            observed.append(target.copy())
            return 0.0

        with patch("utils.grasp_eval.calculate_iou", side_effect=record_iou):
            calculate_jacquard_index(
                predictions,
                targets,
                target_width_cap=300.0,
                target_height=20.0,
            )
        np.testing.assert_array_equal(targets[0], [100, 100, 250, 40, 0, 0])
        np.testing.assert_array_equal(observed[0], [100, 100, 250, 20, 0, 0])

    def test_threshold_grid_reuses_each_rasterized_iou(self):
        predictions = np.array(
            [[100, 100, 80, 20, 5], [120, 100, 80, 20, 25]],
            dtype=np.float32,
        )
        targets = np.array([[100, 100, 80, 40, 0, 0]], dtype=np.float32)
        with patch("utils.grasp_eval.calculate_iou", return_value=0.6) as iou:
            matches = calculate_grasp_matches(predictions, targets)
        self.assertEqual(iou.call_count, 2)
        self.assertEqual(
            calculate_jacquard_from_matches(matches, 1, 0.5, 10.0), 1
        )
        self.assertEqual(
            calculate_jacquard_from_matches(matches, 1, 0.75, 30.0), 0
        )
        self.assertEqual(
            calculate_jacquard_from_matches(matches, 5, 0.5, 20.0), 1
        )

    def test_prediction_cache_can_be_rescored_without_inference(self):
        records = [
            {
                "segmentation_iou": 0.8,
                "rectangles": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
                "targets": [[1, 2, 3, 4, 5, 0]],
                "matches": [[0, 0.6, 5], [1, 0.8, 20]],
                "target_width_cap": 300,
                "target_height": 20,
            },
            {
                "segmentation_iou": 0.4,
                "rectangles": [],
                "targets": [],
                "matches": [],
                "target_width_cap": 300,
                "target_height": 20,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "predictions.npz")
            save_prediction_cache(path, records, {"max_topk": 5, "split": "val"})
            cache = load_prediction_cache(path)
            scores = score_prediction_cache(
                cache,
                topk=(1, 5),
                grasp_iou_threshold=0.5,
                grasp_angle_threshold=10,
                grasp_iou_thresholds=(0.5, 0.75),
                grasp_angle_thresholds=(10,),
                segmentation_iou_thresholds=(0.5,),
            )
        self.assertEqual(scores["num_samples"], 2)
        self.assertAlmostEqual(scores["iou"], 0.6, places=6)
        self.assertEqual(scores["precision"], {"Pr@50": 0.5})
        self.assertEqual(scores["j_index"], [0.5, 0.5])
        self.assertEqual(scores["msr"], {1: 0.25, 5: 0.5})

    def test_evaluation_components_are_registered(self):
        self.assertIn("binary_segmentation", METRICS)
        self.assertIn("grasp_success", METRICS)
        self.assertIn("grasp_threshold_grid", METRICS)
        self.assertIn("dense_grasp", POSTPROCESSORS)


class HookLifecycleTest(unittest.TestCase):
    def test_hooks_run_in_priority_order(self):
        calls = []

        class LateHook(Hook):
            priority = 80

            def before_epoch(self, loop, state):
                calls.append("late")

        class EarlyHook(Hook):
            priority = 10

            def before_epoch(self, loop, state):
                calls.append("early")

        HookList([LateHook(), EarlyHook()]).call(
            "before_epoch", loop=object(), state=LoopState(epoch=1)
        )
        self.assertEqual(calls, ["early", "late"])
        self.assertIn("noop", HOOKS)


if __name__ == "__main__":
    unittest.main()
