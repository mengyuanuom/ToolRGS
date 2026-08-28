from pathlib import Path
import tempfile
import unittest

from utils.config import load_cfg_from_cfg_file, merge_cfg_from_list


ROOT = Path(__file__).resolve().parents[1]


class ConfigInheritanceTest(unittest.TestCase):
    def test_grasp_tools_v3_graspmamba_profile_uses_sigmoid_contract(self):
        cfg = load_cfg_from_cfg_file(
            ROOT
            / "config"
            / "grasp_tools"
            / "graspmamba_grasp_tools_v3_15k_original_scale.yaml"
        )
        self.assertEqual(cfg.architecture, "graspmamba")
        self.assertEqual(cfg.grasp_quality_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_coordinate, "original")
        self.assertEqual(cfg.grasp_size_factor, 300.0)
        self.assertEqual(cfg.word_len, 32)
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.base_lr, 0.0001)
        self.assertFalse(cfg.amp)
        self.assertEqual(cfg.epochs, 36)

    def test_grasp_tools_v3_ggcnn_profile_uses_clamp_contract(self):
        cfg = load_cfg_from_cfg_file(
            ROOT
            / "config"
            / "grasp_tools"
            / "ggcnnclip_grasp_tools_v3_15k_original_scale.yaml"
        )
        self.assertEqual(cfg.architecture, "ggcnnclip")
        self.assertEqual(cfg.grasp_size_activation, "clamp")
        self.assertEqual(cfg.grasp_size_coordinate, "original")
        self.assertEqual(cfg.grasp_size_factor, 300.0)
        self.assertEqual(cfg.word_len, 32)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.epochs, 36)

    def test_grasp_tools_v3_grconvnetclip_profile_uses_clamp_contract(self):
        cfg = load_cfg_from_cfg_file(
            ROOT
            / "config"
            / "grasp_tools"
            / "grconvnetclip_grasp_tools_v3_15k_original_scale.yaml"
        )
        self.assertEqual(cfg.architecture, "grconvnetclip")
        self.assertEqual(cfg.grasp_quality_activation, "clamp")
        self.assertEqual(cfg.grasp_size_activation, "clamp")
        self.assertEqual(cfg.grasp_size_coordinate, "original")
        self.assertEqual(cfg.grasp_size_factor, 300.0)
        self.assertEqual(cfg.word_len, 32)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.epochs, 36)

    def test_etrg_experiment_composes_four_base_configs(self):
        cfg = load_cfg_from_cfg_file(
            ROOT / "configs" / "etrg" / "etrg_r50_ocid_vlg.yaml"
        )
        self.assertEqual(cfg.architecture, "etrg")
        self.assertEqual(cfg.dataset, "OCID-VLG")
        self.assertEqual(cfg.runner["type"], "cuda_grasp")
        self.assertEqual(cfg.optim_wrapper["type"], "cuda_amp")
        self.assertEqual(cfg.param_scheduler["milestones"], [35])
        self.assertEqual(cfg.epochs, 40)
        self.assertEqual(cfg.sections.MODEL.word_dim, 1024)
        self.assertEqual(cfg.etrg_input_mode, "rgb")
        self.assertFalse(cfg.with_depth)

    def test_cli_override_updates_both_config_views(self):
        cfg = load_cfg_from_cfg_file(
            ROOT / "configs" / "etrg" / "etrg_r101_ocid_vlg.yaml"
        )
        updated = merge_cfg_from_list(
            cfg, ["TRAIN.batch_size", "4", "DATA.root_path", "/tmp/ocid"]
        )
        self.assertEqual(updated.batch_size, 4)
        self.assertEqual(updated.sections.TRAIN.batch_size, 4)
        self.assertEqual(updated.root_path, "/tmp/ocid")
        self.assertEqual(updated.sections.DATA.root_path, "/tmp/ocid")

    def test_cli_override_supports_deep_component_paths(self):
        cfg = load_cfg_from_cfg_file(
            ROOT / "configs" / "etrg" / "etrg_r50_ocid_vlg.yaml"
        )
        updated = merge_cfg_from_list(
            cfg, ["RUNTIME.param_scheduler.milestones", "[20, 30]"]
        )
        self.assertEqual(updated.param_scheduler["milestones"], [20, 30])
        self.assertEqual(
            updated.sections.RUNTIME.param_scheduler.milestones, [20, 30]
        )

    def test_delete_replaces_inherited_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text(
                "MODEL:\n  neck:\n    type: old\n    width: 64\n",
                encoding="utf-8",
            )
            (root / "child.yaml").write_text(
                "_base_: base.yaml\nMODEL:\n  neck:\n    _delete_: true\n    type: new\n",
                encoding="utf-8",
            )
            cfg = load_cfg_from_cfg_file(root / "child.yaml")
            self.assertEqual(cfg.neck, {"type": "new"})

    def test_deployment_sections_remain_namespaced(self):
        cfg = load_cfg_from_cfg_file(
            ROOT / "config" / "deployment" / "lab.example.yaml"
        )
        self.assertEqual(cfg.camera["type"], "realsense")
        self.assertEqual((cfg.camera["width"], cfg.camera["height"]), (1280, 720))
        self.assertEqual(cfg.robot["type"], "legacy_tcp")
        self.assertNotIn("type", cfg)


if __name__ == "__main__":
    unittest.main()
