import unittest

import numpy as np

from deployment.grasp_overlay import (
    display_grasp_corners, draw_grasp_overlay, validate_display_padding,
)


class GraspOverlayTest(unittest.TestCase):
    def test_extension_follows_rotated_opening_axis_and_preserves_input(self):
        for angle in (0, 30, 90, -45):
            grasp = [150., 150., 100., 20., float(angle)]
            original = list(grasp)
            before = display_grasp_corners(grasp, 0)
            after = display_grasp_corners(grasp, 20)
            axis = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
            shifts = after - before
            np.testing.assert_allclose(np.abs(shifts @ axis), 20, atol=2e-5)
            np.testing.assert_allclose(shifts @ np.array([-axis[1], axis[0]]), 0, atol=2e-5)
            np.testing.assert_allclose(after.mean(axis=0), grasp[:2], atol=2e-5)
            self.assertEqual(grasp, original)

    def test_jaws_magenta_connectors_yellow(self):
        image = np.zeros((240, 240, 3), np.uint8)
        draw_grasp_overlay(image, [120, 120, 80, 40, 0], 20)
        for x in (60, 180):
            np.testing.assert_array_equal(image[120, x], [255, 0, 255])
        for y in (100, 140):
            np.testing.assert_array_equal(image[y, 120], [0, 255, 255])

    def test_invalid_padding(self):
        for value in (-1, 301, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                validate_display_padding(value)


if __name__ == '__main__':
    unittest.main()
