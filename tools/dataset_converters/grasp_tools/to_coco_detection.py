"""Convert Grasp-Tools V2 scene annotations to COCO object detection JSON.

The V2 generator stores every physical object once in ``objects`` and stores
language queries separately. Detection conversion must therefore iterate over
``objects`` rather than duplicating the target of every language query.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


CLASSES: Tuple[str, ...] = (
    "tape measure",
    "T-hex key",
    "L-hex key",
    "marker",
    "wrench",
    "pliers",
    "mallet",
    "screwdriver",
    "clamps",
    "spool",
    "sponge",
    "clip",
    "crimp tool",
    "screw",
    "tape",
    "box",
    "nut",
    "ruler",
    "file",
    "stapler",
    "scissors",
    "cable",
)

ALIASES = {
    "plier": "pliers",
    "hex key": "t-hex key",
    "t hex key": "t-hex key",
    "t-handle hex key": "t-hex key",
    "l hex key": "l-hex key",
    "l-shaped hex key": "l-hex key",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _category_key(value: str) -> str:
    key = str(value or "").strip().lower().replace("_", " ").replace("-", "-")
    key = " ".join(key.split())
    return ALIASES.get(key, key)


CLASS_BY_KEY = {_category_key(name): name for name in CLASSES}


def _polygon_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2.0


def _image_files(split_dir: Path) -> Iterable[Path]:
    for path in sorted(split_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def convert_split(dataset_root: Path, split: str, output: Path) -> Dict[str, object]:
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Grasp-Tools split does not exist: {split_dir}")

    category_ids = {name: index + 1 for index, name in enumerate(CLASSES)}
    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []
    counts: Counter[str] = Counter()
    empty_images = 0
    annotation_id = 1

    for image_id, image_path in enumerate(_image_files(split_dir), start=1):
        json_path = image_path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing paired annotation: {json_path}")
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = frame.shape[:2]
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        objects = list(payload.get("objects") or [])
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": int(width),
                "height": int(height),
            }
        )
        valid_in_image = 0
        for object_index, obj in enumerate(objects):
            key = _category_key(obj.get("category", ""))
            if key not in CLASS_BY_KEY:
                raise ValueError(
                    f"Unknown category {obj.get('category')!r} in "
                    f"{json_path.name} object {object_index}"
                )
            category = CLASS_BY_KEY[key]
            polygon = np.asarray(obj.get("mask") or [], dtype=np.float64)
            if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
                raise ValueError(
                    f"Invalid polygon in {json_path.name} object {object_index}"
                )
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

            raw_bbox = list(obj.get("bbox") or [])
            if len(raw_bbox) == 4:
                x1, y1, x2, y2 = (float(value) for value in raw_bbox)
            else:
                x1, y1 = polygon.min(axis=0)
                x2, y2 = polygon.max(axis=0)
            x1 = float(np.clip(x1, 0, width - 1))
            y1 = float(np.clip(y1, 0, height - 1))
            x2 = float(np.clip(x2, x1 + 1, width))
            y2 = float(np.clip(y2, y1 + 1, height))
            bbox_width, bbox_height = x2 - x1, y2 - y1
            if bbox_width <= 0 or bbox_height <= 0:
                raise ValueError(
                    f"Degenerate bbox in {json_path.name} object {object_index}"
                )
            area = _polygon_area(polygon)
            if area <= 0:
                area = bbox_width * bbox_height

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[category],
                    "bbox": [x1, y1, bbox_width, bbox_height],
                    "area": float(area),
                    "segmentation": [polygon.reshape(-1).astype(float).tolist()],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            valid_in_image += 1
            counts[category] += 1
        if valid_in_image == 0:
            empty_images += 1

    if not images:
        raise RuntimeError(f"No images found in {split_dir}")
    missing_categories = [name for name in CLASSES if counts[name] == 0]
    if missing_categories:
        raise RuntimeError(
            f"Split {split!r} contains no objects for: {', '.join(missing_categories)}"
        )

    coco = {
        "info": {
            "description": f"Grasp-Tools V2 {split} object detection split",
            "version": "2.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_ids[name], "name": name, "supercategory": "tool"}
            for name in CLASSES
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return {
        "split": split,
        "images": len(images),
        "annotations": len(annotations),
        "empty_images": empty_images,
        "class_counts": dict(counts),
        "output": str(output),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/grasp-tools/aug_graspall_v2"),
    )
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: DATASET_ROOT/annotations",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "annotations"
    )
    summaries = []
    for split in args.splits:
        output = output_dir / f"grasp_tools_instances_{split}.json"
        summary = convert_split(dataset_root, str(split), output)
        summaries.append(summary)
        print(
            f"[{split}] images={summary['images']} "
            f"objects={summary['annotations']} empty={summary['empty_images']}\n"
            f"  -> {summary['output']}"
        )
    print(json.dumps({"classes": CLASSES, "splits": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
