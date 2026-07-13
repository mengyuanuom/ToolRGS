"""Configuration-driven dataset registry for ToolRGS."""

from utils.dataset import GraspToolDataset
from utils.vcot_dataset import VCoTDataset


DATASET_REGISTRY = {
    "grasptool": GraspToolDataset,
    "grasp_tool": GraspToolDataset,
    "grasp_tools": GraspToolDataset,
    "vcot": VCoTDataset,
    "vcot_grasp": VCoTDataset,
}


def _normalise_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def build_dataset(cfg, split, with_offset=False):
    """Build the configured dataset without coupling train/eval to its class."""
    name = _normalise_name(cfg.dataset)
    try:
        dataset_class = DATASET_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown DATA.dataset {cfg.dataset!r}; available: {available}") from error

    common = dict(
        root_dir=cfg.root_path,
        input_size=cfg.input_size,
        split=split,
        word_length=cfg.word_len,
        with_offset=with_offset,
        offset_radius=getattr(cfg, "offset_r", 20.0),
        offset_sigma=getattr(cfg, "offset_sigma", None),
    )
    if dataset_class is VCoTDataset:
        common.update(
            split_root=getattr(cfg, "split_root", None),
            prompt_template=getattr(cfg, "prompt_template", "Grasp the {object_name}"),
        )
    return dataset_class(**common)
