from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CUDARunnerSourceContractTest(unittest.TestCase):
    def test_runner_and_optimization_components_are_registered(self):
        registry = (ROOT / "toolrgs" / "registry.py").read_text(encoding="utf-8-sig")
        runner = (ROOT / "toolrgs" / "engine" / "runner.py").read_text(
            encoding="utf-8-sig"
        )
        optim = (ROOT / "toolrgs" / "engine" / "optim.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn('RUNNERS = Registry("runners")', registry)
        self.assertIn('@RUNNERS.register_module(name="cuda_grasp"', runner)
        self.assertIn('@OPTIM_WRAPPERS.register_module(name="cuda_amp"', optim)
        self.assertIn("PARAM_SCHEDULERS.register_module", optim)

    def test_cuda_runner_does_not_import_npu_runtime(self):
        paths = (
            ROOT / "toolrgs" / "engine" / "runner.py",
            ROOT / "toolrgs" / "engine" / "optim.py",
            ROOT / "tools" / "train.py",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8-sig")
            self.assertNotIn("torch_npu", source, path)
            self.assertNotIn('"hccl"', source, path)
        runner = paths[0].read_text(encoding="utf-8-sig")
        self.assertIn('backend="nccl"', runner)
        self.assertIn("torch.cuda.set_device", runner)


if __name__ == "__main__":
    unittest.main()
