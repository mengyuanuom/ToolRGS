import glob
from pathlib import Path
import tempfile
import unittest

from utils.config import (
    load_cfg_from_cfg_file,
    merge_cfg_from_list,
    resolve_grasp_training_activation,
)


ROOT = Path(__file__).resolve().parents[1]


class ConfigInheritanceTest(unittest.TestCase):
    def test_training_activation_contract_rejects_mismatched_inference(self):
        self.assertEqual(
            resolve_grasp_training_activation("raw", "auto", name="quality"),
            ("raw", "clamp"),
        )
        self.assertEqual(
            resolve_grasp_training_activation(
                "sigmoid", "sigmoid", name="quality"
            ),
            ("sigmoid", "sigmoid"),
        )
        with self.assertRaisesRegex(ValueError, "train/inference activation mismatch"):
            resolve_grasp_training_activation("raw", "sigmoid", name="quality")

    def test_all_grasp_tools_v3_profiles_share_original_300_size_contract(self):
        matched = []
        for path in glob.glob(
            str(ROOT / "config" / "grasp_tools" / "*.yaml")
        ):
            cfg = load_cfg_from_cfg_file(path)
            if "aug_graspall_v3_15k" not in str(getattr(cfg, "root_path", "")):
                continue
            matched.append(path)
            self.assertEqual(cfg.grasp_size_coordinate, "original", path)
            self.assertEqual(cfg.grasp_size_factor, 300.0, path)
        self.assertGreaterEqual(len(matched), 9)

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

    def test_grasp_tools_v3_grconvnetclip_profile_uses_sigmoid_contract(self):
        cfg = load_cfg_from_cfg_file(
            ROOT
            / "config"
            / "grasp_tools"
            / "grconvnetclip_grasp_tools_v3_15k_original_scale.yaml"
        )
        self.assertEqual(cfg.architecture, "grconvnetclip")
        self.assertEqual(cfg.grasp_quality_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_coordinate, "original")
        self.assertEqual(cfg.grasp_size_factor, 300.0)
        self.assertEqual(cfg.word_len, 32)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.epochs, 36)

    def test_crog_unified_v3_sigmoid_profile_is_fresh_and_consistent(self):
        cfg = load_cfg_from_cfg_file(
            ROOT
            / "config"
            / "grasp_tools"
            / "v3_crog_unified_original300_sigmoid_retrain.yaml"
        )
        self.assertEqual(cfg.architecture, "crog")
        self.assertEqual(
            cfg.exp_name,
            "crog_grasp_tools_v3_15k_unified_original300_sigmoid",
        )
        self.assertIsNone(cfg.weight)
        self.assertIsNone(cfg.resume)
        self.assertEqual(cfg.grasp_quality_loss_activation, "sigmoid")
        self.assertEqual(cfg.grasp_width_loss_activation, "sigmoid")
        self.assertEqual(cfg.grasp_quality_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_activation, "sigmoid")
        self.assertEqual(cfg.evaluation_protocol, "toolrgs")
        self.assertEqual(cfg.grasp_size_coordinate, "original")
        self.assertEqual(cfg.grasp_size_factor, 300.0)
        self.assertFalse(cfg.restore_grasp_size_scale)
        self.assertEqual(cfg.epochs, 36)
        self.assertEqual(cfg.batch_size, 32)

    def test_grconvnetclip_unified_v3_profile_is_a_fresh_toolrgs_run(self):
        cfg = load_cfg_from_cfg_file(
            ROOT
            / "config"
            / "grasp_tools"
            / "grconvnetclip_v3_unified_original300_retrain.yaml"
        )
        self.assertEqual(cfg.architecture, "grconvnetclip")
        self.assertEqual(
            cfg.exp_name,
            "grconvnetclip_grasp_tools_v3_15k_unified_original300",
        )
        self.assertIsNone(cfg.weight)
        self.assertIsNone(cfg.resume)
        self.assertEqual(cfg.evaluation_protocol, "toolrgs")
        self.assertFalse(cfg.restore_grasp_size_scale)
        self.assertEqual(cfg.grasp_quality_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_coordinate, "original")
        self.assertEqual(cfg.grasp_size_factor, 300.0)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.epochs, 36)

    def test_etrg_unified_v3_profile_is_sigmoid_and_geometry_masked(self):
        cfg = load_cfg_from_cfg_file(
            ROOT
            / "config"
            / "grasp_tools"
            / "etrg_r50_rgb_v3_unified_original300_sigmoid_masked_retrain.yaml"
        )
        self.assertEqual(cfg.architecture, "etrg")
        self.assertEqual(
            cfg.exp_name,
            "etrg_r50_rgb_grasp_tools_v3_15k_unified_original300_sigmoid_masked",
        )
        self.assertIsNone(cfg.weight)
        self.assertIsNone(cfg.resume)
        self.assertEqual(cfg.grasp_quality_loss_activation, "sigmoid")
        self.assertEqual(cfg.grasp_width_loss_activation, "sigmoid")
        self.assertEqual(cfg.grasp_quality_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_activation, "sigmoid")
        self.assertEqual(cfg.etrg_quality_positive_threshold, 0.05)
        self.assertEqual(cfg.etrg_geometry_mask_threshold, 0.000001)
        self.assertEqual(cfg.evaluation_protocol, "toolrgs")
        self.assertEqual(cfg.grasp_size_coordinate, "original")
        self.assertEqual(cfg.grasp_size_factor, 300.0)
        self.assertFalse(cfg.restore_grasp_size_scale)
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.epochs, 36)

    def test_lgd_unified_v3_profile_matches_the_common_data_contract(self):
        cfg = load_cfg_from_cfg_file(
            ROOT
            / "config"
            / "grasp_tools"
            / "lgd_v3_unified_original300_retrain.yaml"
        )
        self.assertEqual(cfg.architecture, "lgd")
        self.assertEqual(
            cfg.exp_name, "lgd_grasp_tools_v3_15k_unified_original300"
        )
        self.assertIsNone(cfg.weight)
        self.assertIsNone(cfg.resume)
        self.assertEqual(cfg.evaluation_protocol, "toolrgs")
        self.assertFalse(cfg.restore_grasp_size_scale)
        self.assertEqual(cfg.grasp_quality_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_activation, "clamp")
        self.assertEqual(cfg.grasp_size_coordinate, "original")
        self.assertEqual(cfg.grasp_size_factor, 300.0)
        self.assertEqual(cfg.word_len, 32)
        self.assertEqual(cfg.batch_size, 16)
        self.assertEqual(cfg.epochs, 36)

    def test_native_v3_lora_unified_profile_is_fresh_and_uncropped(self):
        cfg = load_cfg_from_cfg_file(
            ROOT
            / "config"
            / "grasp_tools"
            / "drogoff_native_v3_lora_r24_12l_v3_unified_original300_retrain.yaml"
        )
        self.assertEqual(cfg.architecture, "drogoff")
        self.assertEqual(cfg.native_variant, "v3")
        self.assertEqual(cfg.native_lora_rank, 24)
        self.assertEqual(cfg.native_visual_lora_layers, list(range(12)))
        self.assertEqual(cfg.native_text_lora_layers, list(range(12)))
        self.assertEqual(
            cfg.exp_name,
            "drogoff_native_v3_lora_r24_12l_grasp_tools_v3_15k_unified_original300_sigmoid_consistent",
        )
        self.assertIsNone(cfg.weight)
        self.assertIsNone(cfg.resume)
        self.assertEqual(cfg.evaluation_protocol, "toolrgs")
        self.assertFalse(cfg.restore_grasp_size_scale)
        self.assertEqual(cfg.grasp_quality_loss_activation, "sigmoid")
        self.assertEqual(cfg.grasp_width_loss_activation, "sigmoid")
        self.assertEqual(cfg.grasp_quality_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_coordinate, "original")
        self.assertEqual(cfg.grasp_size_factor, 300.0)
        self.assertEqual(cfg.epochs, 24)
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.world_size, 1)

    def test_drog_unified_v3_profile_is_fresh_and_sigmoid_consistent(self):
        cfg = load_cfg_from_cfg_file(
            ROOT
            / "config"
            / "grasp_tools"
            / "drog_v3_unified_original300_sigmoid_retrain.yaml"
        )
        self.assertEqual(cfg.architecture, "drog")
        self.assertEqual(
            cfg.exp_name,
            "drog_grasp_tools_v3_15k_unified_original300_sigmoid",
        )
        self.assertIsNone(cfg.weight)
        self.assertIsNone(cfg.resume)
        self.assertEqual(cfg.grasp_quality_loss_activation, "sigmoid")
        self.assertEqual(cfg.grasp_width_loss_activation, "sigmoid")
        self.assertEqual(cfg.grasp_quality_activation, "sigmoid")
        self.assertEqual(cfg.grasp_size_activation, "sigmoid")
        self.assertEqual(cfg.evaluation_protocol, "toolrgs")
        self.assertEqual(cfg.grasp_size_coordinate, "original")
        self.assertEqual(cfg.grasp_size_factor, 300.0)
        self.assertFalse(cfg.restore_grasp_size_scale)
        self.assertEqual(cfg.word_len, 32)
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.epochs, 36)
        self.assertEqual(cfg.world_size, 1)

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
