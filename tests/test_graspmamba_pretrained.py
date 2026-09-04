import argparse
from pathlib import Path
import unittest
from unittest import mock

import torch

from model import graspmamba


class GraspMambaPretrainedContractTest(unittest.TestCase):
    def test_official_t_model_has_registered_artifact(self):
        self.assertEqual(
            graspmamba._OFFICIAL_MAMBAVISION_ARTIFACTS["mamba_vision_T"],
            "mambavision-t",
        )

    def test_verified_checkpoint_context_is_narrow(self):
        with graspmamba._official_checkpoint_context(
            Path("mambavision_tiny_1k.pth.tar"),
            "mambavision-t",
        ):
            pass

    def test_no_checkpoint_uses_noop_context(self):
        with graspmamba._official_checkpoint_context(None, None):
            value = argparse.Namespace(ok=True)
        self.assertTrue(value.ok)

    def test_pytorch_without_safe_globals_remains_supported(self):
        with mock.patch.object(
            torch.serialization,
            "safe_globals",
            None,
            create=True,
        ):
            with graspmamba._official_checkpoint_context(
                Path("mambavision_tiny_1k.pth.tar"),
                "mambavision-t",
            ):
                pass


if __name__ == "__main__":
    unittest.main()
