"""Dataset wrapper that adds DARG rotated-FCOS supervision maps."""

import torch
from torch.utils.data import Dataset

from toolrgs.evaluation.asymmetric_geometry import (
    ASYMMETRIC_TARGET_KEYS,
    generate_asymmetric_grasp_targets,
    transform_grasps_to_input,
)


class AsymmetricGraspDataset(Dataset):
    """Decorate any ToolRGS grasp dataset with full-rectangle geometry maps."""

    def __init__(self, dataset, size_factor=100.0):
        self.dataset = dataset
        self.size_factor = float(size_factor)
        if self.size_factor <= 0:
            raise ValueError("Asymmetric grasp size_factor must be positive")

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name in {"dataset", "size_factor"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __getitem__(self, index):
        sample = self.dataset[index]
        rectangles = transform_grasps_to_input(
            sample.get("grasps", ()), sample["inverse"]
        )
        targets = generate_asymmetric_grasp_targets(
            rectangles,
            sample["mask"].shape[-2:],
            size_factor=self.size_factor,
            size_rectangles=(
                sample.get("grasps", ())
                if str(getattr(self.dataset, "grasp_size_coordinate", "canvas"))
                == "original"
                else None
            ),
        )
        sample = dict(sample)
        sample["grasp_masks"] = dict(sample["grasp_masks"])
        for name, value in targets.items():
            sample["grasp_masks"][name] = torch.from_numpy(value).float()
        return sample

    def collate_fn(self, batch):
        collated = self.dataset.collate_fn(batch)
        for name in ASYMMETRIC_TARGET_KEYS:
            collated["grasp_masks"][name] = torch.stack(
                [sample["grasp_masks"][name] for sample in batch]
            )
        return collated
