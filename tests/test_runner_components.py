from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NPURunnerSourceContractTest(unittest.TestCase):
    def test_runner_and_optimization_components_are_registered(self):
        registry = (ROOT / "toolrgs" / "registry.py").read_text(encoding="utf-8-sig")
        runner = (ROOT / "toolrgs" / "engine" / "runner.py").read_text(
            encoding="utf-8-sig"
        )
        optim = (ROOT / "toolrgs" / "engine" / "optim.py").read_text(
            encoding="utf-8-sig"
        )
        runtime = (ROOT / "toolrgs" / "runtime" / "device.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn('RUNNERS = Registry("runners")', registry)
        self.assertIn('@RUNNERS.register_module(name="npu_grasp"', runner)
        self.assertIn('@OPTIM_WRAPPERS.register_module(name="npu_amp"', optim)
        self.assertIn("PARAM_SCHEDULERS.register_module", optim)
        self.assertIn("class NoOpGradScaler", runtime)
        self.assertNotIn("transfer_to_npu", runtime)

    def test_npu_runner_uses_explicit_runtime_and_hccl(self):
        paths = (
            ROOT / "toolrgs" / "engine" / "runner.py",
            ROOT / "toolrgs" / "engine" / "optim.py",
            ROOT / "tools" / "train.py",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8-sig")
            self.assertNotIn("transfer_to_npu", source, path)
        runner = paths[0].read_text(encoding="utf-8-sig")
        self.assertIn('backend="hccl"', runner)
        self.assertIn("set_device(cfg.npu)", runner)
        self.assertNotIn("torch.cuda", runner)


if __name__ == "__main__":
    unittest.main()
