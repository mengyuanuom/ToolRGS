#!/usr/bin/env python3
"""Validate the hard contracts of a generated Grasp-Tools V4 Dense dataset."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--objects-min", type=int, default=10)
    parser.add_argument("--objects-max", type=int, default=12)
    parser.add_argument("--scale-min", type=float, default=0.3)
    parser.add_argument("--scale-max", type=float, default=0.6)
    parser.add_argument("--angle-bins", type=int, default=24)
    parser.add_argument(
        "--max-category-imbalance",
        type=int,
        default=1,
        help="Maximum allowed max-minus-min category count inside each split.",
    )
    return parser.parse_args()


def circular_distance_degrees(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def validate_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    root = args.dataset_dir.expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = metadata["config"]
    expected_scenes = {
        "train": int(config["train_scenes"]),
        "val": int(config["val_scenes"]),
        "test": int(config["test_scenes"]),
    }
    categories = list(metadata["canonical_categories"])
    category_set = set(categories)
    bin_width = 360.0 / args.angle_bins
    summary: Dict[str, Any] = {}

    for split, expected in expected_scenes.items():
        split_dir = root / split
        annotation_paths = sorted(split_dir.glob(f"{split}_scene_*.json"))
        if len(annotation_paths) != expected:
            raise ValueError(
                f"{split}: expected {expected} annotations, found {len(annotation_paths)}"
            )
        index_lines = [
            line
            for line in (split_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        placements: Counter[str] = Counter()
        queries_by_category: Counter[str] = Counter()
        angles: Dict[str, Counter[int]] = defaultdict(Counter)
        query_total = 0

        for annotation_path in annotation_paths:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if annotation.get("dataset_version") != "grasp-tools-v4-dense":
                raise ValueError(f"{annotation_path}: wrong or missing dataset_version")
            objects = annotation["objects"]
            queries = annotation["queries"]
            requested = int(annotation["requested_object_count"])
            if not args.objects_min <= len(objects) <= args.objects_max:
                raise ValueError(f"{annotation_path}: invalid object count {len(objects)}")
            if len(objects) != requested:
                raise ValueError(f"{annotation_path}: underfilled scene")
            object_categories = [obj["category"] for obj in objects]
            if len(set(object_categories)) != len(object_categories):
                raise ValueError(f"{annotation_path}: duplicate category in scene")
            if not set(object_categories) <= category_set:
                raise ValueError(f"{annotation_path}: unknown canonical category")
            targets = [int(query["target_idx"]) for query in queries]
            if len(queries) != len(objects) or sorted(targets) != list(range(len(objects))):
                raise ValueError(f"{annotation_path}: queries are not one-per-object")
            if any(query["type"] != "category" for query in queries):
                raise ValueError(f"{annotation_path}: non-category query found")

            for obj in objects:
                category = obj["category"]
                transform = obj["transform"]
                scale = float(transform["scale"])
                if not args.scale_min - 1e-8 <= scale <= args.scale_max + 1e-8:
                    raise ValueError(f"{annotation_path}: scale {scale} outside bounds")
                angle_bin = int(transform["angle_bin"])
                if not 0 <= angle_bin < args.angle_bins:
                    raise ValueError(f"{annotation_path}: invalid angle bin {angle_bin}")
                rotation = float(transform["rotation_deg"])
                center = angle_bin * bin_width
                if circular_distance_degrees(rotation, center) > bin_width / 2.0 + 1e-5:
                    raise ValueError(f"{annotation_path}: rotation outside its angle bin")
                placements[category] += 1
                angles[category][angle_bin] += 1
            for query in queries:
                target = objects[int(query["target_idx"])]
                queries_by_category[target["category"]] += 1
            query_total += len(queries)

        counts = [placements[category] for category in categories]
        imbalance = max(counts) - min(counts)
        if imbalance > args.max_category_imbalance:
            raise ValueError(f"{split}: category imbalance is {imbalance}")
        if placements != queries_by_category:
            raise ValueError(f"{split}: category placement/query distributions differ")
        for category in categories:
            bin_counts = [angles[category][index] for index in range(args.angle_bins)]
            if max(bin_counts) - min(bin_counts) > 1:
                raise ValueError(f"{split}/{category}: successful angle bins are imbalanced")
        if len(index_lines) != query_total:
            raise ValueError(
                f"{split}: index has {len(index_lines)} rows for {query_total} queries"
            )
        summary[split] = {
            "scenes": len(annotation_paths),
            "objects": sum(counts),
            "queries": query_total,
            "objects_per_scene": [args.objects_min, args.objects_max],
            "category_count_range": [min(counts), max(counts)],
            "category_imbalance": imbalance,
        }
    return summary


def main() -> int:
    args = parse_args()
    summary = validate_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[ok] Grasp-Tools V4 Dense contracts validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
