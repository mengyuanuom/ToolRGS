import unittest

import numpy as np
import torch

from model import MODEL_REGISTRY
from model.droglr import (
    DROGLRProjector,
    decode_direct_lr_geometry,
    decode_lr_geometry,
)
from toolrgs.evaluation.asymmetric_geometry import (
    generate_asymmetric_grasp_targets,
)


class DROGLRTest(unittest.TestCase):
    def test_model_is_registered(self):
        self.assertIn("droglr", MODEL_REGISTRY)

    def test_projector_and_bounded_lr_decode(self):
        projector = DROGLRProjector(word_dim=32, in_dim=8, hidden_dim=16)
        output = projector(torch.randn(2, 16, 8, 8), torch.randn(2, 32))
        self.assertEqual(tuple(output["segmentation"].shape), (2, 1, 32, 32))
        self.assertEqual(tuple(output["total_width"].shape), (2, 1, 32, 32))
        self.assertEqual(tuple(output["left_fraction"].shape), (2, 1, 32, 32))

        total = torch.logit(torch.tensor([[[[0.8]]]]))
        fraction = torch.logit(torch.tensor([[[[0.25]]]]))
        sine = torch.zeros(1, 1, 1, 1)
        cosine = torch.ones(1, 1, 1, 1)
        width, left, right, offset = decode_lr_geometry(
            total,
            fraction,
            sine,
            cosine,
            offset_size_factor=10.0,
            offset_radius=5.0,
        )
        torch.testing.assert_close(width, torch.tensor([[[[0.8]]]]))
        torch.testing.assert_close(left, torch.tensor([[[[0.2]]]]))
        torch.testing.assert_close(right, torch.tensor([[[[0.6]]]]))
        torch.testing.assert_close(offset, torch.tensor([[[[0.4]], [[0.0]]]]))

    def test_targets_keep_original_size_but_canvas_balance(self):
        targets = generate_asymmetric_grasp_targets(
            [[20.0, 20.0, 16.0, 8.0, 0.0]],
            (48, 48),
            size_factor=10.0,
            size_rectangles=[[10.0, 10.0, 8.0, 4.0, 0.0]],
        )
        # At the transformed center: 4 original pixels on either side.
        np.testing.assert_allclose(targets["ltrb"][:2, 20, 20], [0.4, 0.4])
        # Four canvas pixels to the right are two original pixels: 6/2 split.
        np.testing.assert_allclose(targets["ltrb"][:2, 20, 24], [0.6, 0.2])

    def test_direct_heads_and_proportional_total_width_cap(self):
        projector = DROGLRProjector(
            word_dim=32,
            in_dim=8,
            hidden_dim=16,
            parameterization="direct",
        )
        output = projector(torch.randn(2, 16, 8, 8), torch.randn(2, 32))
        self.assertEqual(tuple(output["left_width"].shape), (2, 1, 32, 32))
        self.assertEqual(tuple(output["right_width"].shape), (2, 1, 32, 32))
        self.assertNotIn("total_width", output)
        self.assertNotIn("left_fraction", output)

        left_logits = torch.logit(torch.tensor([[[[0.8]]]]))
        right_logits = torch.logit(torch.tensor([[[[0.7]]]]))
        sine = torch.zeros(1, 1, 1, 1)
        cosine = torch.ones(1, 1, 1, 1)
        width, left, right, offset, raw_left, raw_right = (
            decode_direct_lr_geometry(
                left_logits,
                right_logits,
                sine,
                cosine,
                offset_size_factor=10.0,
                offset_radius=5.0,
                max_total_normalized=1.0,
            )
        )
        torch.testing.assert_close(raw_left, torch.tensor([[[[0.8]]]]))
        torch.testing.assert_close(raw_right, torch.tensor([[[[0.7]]]]))
        torch.testing.assert_close(width, torch.ones_like(width))
        torch.testing.assert_close(left / right, raw_left / raw_right)
        torch.testing.assert_close(
            offset,
            torch.tensor([[[[-1.0 / 15.0]], [[0.0]]]]),
        )


if __name__ == "__main__":
    unittest.main()
