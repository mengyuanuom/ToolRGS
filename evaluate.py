"""Evaluate any ToolRGS architecture using one experiment config."""

import argparse
import os

import cv2
from loguru import logger
import torch
from torch.utils.data import DataLoader

import utils.config as config
from model import build_model
from toolrgs.engine import GraspValLoop, RealVLGValLoop  # register validation loops
from toolrgs.registry import LOOPS
from utils.data_builder import build_dataset
from utils.misc import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ToolRGS")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", help="Evaluation split override")
    parser.add_argument(
        "--msr-output",
        help="Write a one-forward-pass IoU/angle threshold-grid summary TSV",
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    cli = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli.config)
    if cli.opts:
        cfg = config.merge_cfg_from_list(cfg, cli.opts)
    cfg.resume = cli.checkpoint
    cfg.msr_output = cli.msr_output
    if cli.msr_output:
        cfg.compute_grasp_msr = True
    cfg.eval_split = cli.split or getattr(
        cfg, "test_split", getattr(cfg, "val_split", None)
    )
    if cfg.eval_split is None:
        raise ValueError("TEST.test_split or DATA.val_split must be configured")
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
    dataset = build_dataset(args, args.eval_split, with_offset=needs_offset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers_val,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    val_loop_class = LOOPS.require(getattr(args, "val_loop", "grasp_val"))
    val_loop = val_loop_class(
        dataloader=loader,
        model=model,
        cfg=args,
        hooks=getattr(args, "val_hooks", None),
    )
    iou, precision, j_index = val_loop.run_epoch(getattr(args, "start_epoch", 0))
    protocol = str(getattr(args, "evaluation_protocol", "")).lower()
    if protocol in {"realvlg", "realvlg_source", "realvlg_official"}:
        logger.info(
            "Final RealVLG F_beta={}, metrics={}, gAcc={}",
            iou,
            precision,
            j_index[0] if j_index else 0.0,
        )
    else:
        logger.info("Final IoU={}, precision={}, J={}", iou, precision, j_index)
    if args.msr_output:
        rows = getattr(val_loop, "grasp_threshold_grid", None)
        msr = getattr(val_loop, "grasp_msr", None)
        if not rows or not msr:
            raise RuntimeError("The selected validation loop did not produce an mSR grid")
        output_parent = os.path.dirname(os.path.abspath(args.msr_output))
        os.makedirs(output_parent, exist_ok=True)
        topk = tuple(int(value) for value in args.grasp_topk)
        with open(args.msr_output, "w", encoding="utf-8") as handle:
            handle.write("iou\tangle\t" + "\t".join(f"J@{k}" for k in topk) + "\n")
            for row in rows:
                values = "\t".join(
                    f"{100.0 * row['values'][k]:.4f}" for k in topk
                )
                handle.write(f"{row['iou']:.2f}\t{row['angle']:.1f}\t{values}\n")
            handle.write(
                "mSR\tmean\t"
                + "\t".join(f"{100.0 * msr[k]:.4f}" for k in topk)
                + "\n"
            )
        logger.info(
            "mSR summary: {} ({})",
            args.msr_output,
            "  ".join(f"mSR@{k}={100.0 * msr[k]:.4f}" for k in topk),
        )


if __name__ == "__main__":
    main()
