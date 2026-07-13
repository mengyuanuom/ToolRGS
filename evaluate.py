"""Evaluate any ToolRGS architecture using one experiment config."""

import argparse
import os

import cv2
from loguru import logger
import torch
from torch.utils.data import DataLoader

import utils.config as config
from engine.engine import validate_with_grasp
from model import build_model
from utils.data_builder import build_dataset
from utils.misc import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ToolRGS")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    cli = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli.config)
    if cli.opts:
        cfg = config.merge_cfg_from_list(cfg, cli.opts)
    cfg.resume = cli.checkpoint
    return cfg


def load_state(model, state):
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass
    cleaned = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=True)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ToolRGS evaluation currently requires a CUDA GPU")
    cv2.setNumThreads(0)
    args.gpu = 0
    args.rank = 0
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    setup_logger(args.output_dir, distributed_rank=0, filename="eval.log", mode="a")

    model, _ = build_model(args)
    model = model.cuda().eval()
    checkpoint = torch.load(args.resume, map_location="cuda:0")
    load_state(model, checkpoint.get("state_dict", checkpoint))

    needs_offset = args.architecture.lower() in {"crogoff", "drogoff"}
    dataset = build_dataset(args, args.val_split, with_offset=needs_offset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers_val,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    iou, precision, j_index = validate_with_grasp(
        loader, model, getattr(args, "start_epoch", 0), args
    )
    logger.info("Final IoU={}, precision={}, J={}", iou, precision, j_index)


if __name__ == "__main__":
    main()
