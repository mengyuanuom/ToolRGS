"""Evaluate a ToolRGS model or rescore its decoded prediction cache."""

import argparse
from collections.abc import Mapping
import os

import cv2
from loguru import logger
import torch
from torch.utils.data import DataLoader

import utils.config as config
from model import build_model
from toolrgs.engine import GraspValLoop, RealVLGValLoop  # register validation loops
from toolrgs.evaluation import (
    load_prediction_cache,
    save_prediction_cache,
    score_prediction_cache,
    write_score_summary,
)
from toolrgs.registry import LOOPS
from utils.data_builder import build_dataset
from utils.misc import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ToolRGS")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", help="Evaluation split override")
    parser.add_argument(
        "--prediction-cache",
        help="Decoded prediction cache output (default: experiment evaluation_cache/)",
    )
    parser.add_argument(
        "--score-cache",
        help="Load this prediction cache and score it without model inference",
    )
    parser.add_argument(
        "--no-save-predictions",
        dest="save_predictions",
        action="store_false",
        help="Disable the default decoded-prediction cache",
    )
    parser.set_defaults(save_predictions=True)
    parser.add_argument(
        "--msr-output",
        help="Threshold-grid summary TSV (default: next to the prediction cache)",
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    cli = parser.parse_args()
    if not cli.score_cache and not cli.checkpoint:
        parser.error("--checkpoint is required unless --score-cache is used")
    cfg = config.load_cfg_from_cfg_file(cli.config)
    if cli.opts:
        cfg = config.merge_cfg_from_list(cfg, cli.opts)
    cfg.resume = cli.checkpoint
    cfg.score_cache = cli.score_cache
    cfg.prediction_cache = cli.prediction_cache
    cfg.save_predictions = bool(cli.save_predictions and not cli.score_cache)
    cfg.collect_predictions = cfg.save_predictions
    cfg.msr_output = cli.msr_output
    # Cache-backed scoring computes the complete grid after inference. Preserve
    # the in-loop path only for the explicit cache opt-out case.
    cfg.compute_grasp_msr = bool(cli.msr_output and not cfg.save_predictions)
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


def _segmentation_iou_thresholds(cfg):
    metric = getattr(cfg, "segmentation_metric", None)
    if isinstance(metric, Mapping) and "iou_thresholds" in metric:
        return tuple(float(value) for value in metric["iou_thresholds"])
    return (0.5, 0.6, 0.7, 0.8, 0.9)


def _score_kwargs(cfg):
    return {
        "topk": tuple(getattr(cfg, "grasp_topk", (1, 5))),
        "grasp_iou_threshold": float(
            getattr(cfg, "grasp_iou_threshold", 0.25)
        ),
        "grasp_angle_threshold": float(
            getattr(cfg, "grasp_angle_threshold", 30.0)
        ),
        "grasp_iou_thresholds": tuple(
            getattr(cfg, "grasp_iou_thresholds", (0.25, 0.50, 0.75))
        ),
        "grasp_angle_thresholds": tuple(
            getattr(cfg, "grasp_angle_thresholds", (5.0, 10.0, 20.0, 30.0))
        ),
        "segmentation_iou_thresholds": _segmentation_iou_thresholds(cfg),
    }


def _default_prediction_cache(cfg):
    checkpoint_name = os.path.splitext(os.path.basename(cfg.resume))[0]
    return os.path.join(
        cfg.output_dir,
        "evaluation_cache",
        f"{cfg.eval_split}_{checkpoint_name}_predictions.npz",
    )


def _default_score_output(cache_path):
    return os.path.splitext(os.path.abspath(cache_path))[0] + "_scores.tsv"


def _log_scores(scores, source):
    logger.info(
        "Cached evaluation [{}]: samples={}, IoU={:.4f}, J={}",
        source,
        scores["num_samples"],
        scores["iou"],
        scores["j_index"],
    )
    logger.info(
        "Cached mSR: {}",
        "  ".join(
            f"mSR@{topk}={100.0 * scores['msr'][topk]:.4f}"
            for topk in scores["topk"]
        ),
    )


def _score_and_write(cache_path, cfg):
    cache = load_prediction_cache(cache_path)
    cached_split = cache["metadata"].get("split")
    if cached_split and str(cached_split) != str(cfg.eval_split):
        logger.warning(
            "Scoring cached split {!r} while config requests {!r}",
            cached_split,
            cfg.eval_split,
        )
    scores = score_prediction_cache(cache, **_score_kwargs(cfg))
    score_output = cfg.msr_output or _default_score_output(cache_path)
    write_score_summary(score_output, scores)
    _log_scores(scores, cache_path)
    logger.info("Score summary: {}", score_output)
    return scores


def _cache_metadata(cfg):
    checkpoint = os.path.abspath(cfg.resume)
    checkpoint_stat = os.stat(checkpoint)
    return {
        "architecture": str(cfg.architecture),
        "split": str(cfg.eval_split),
        "checkpoint": checkpoint,
        "checkpoint_size": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "config": str(getattr(cfg, "filename", "")),
        "max_topk": max(int(value) for value in getattr(cfg, "grasp_topk", (1, 5))),
        "evaluation_protocol": str(getattr(cfg, "evaluation_protocol", "")),
        "quality_threshold": float(getattr(cfg, "grasp_quality_threshold", 0.4)),
        "min_distance": int(getattr(cfg, "grasp_min_distance", 2)),
        "size_factor": float(getattr(cfg, "grasp_size_factor", 100.0)),
        "grasp_height": float(getattr(cfg, "grasp_height", 20.0)),
        "size_coordinate": str(getattr(cfg, "grasp_size_coordinate", "canvas")),
        "offset_decode_mode": str(getattr(cfg, "offset_decode_mode", "radius")),
    }


def main():
    args = parse_args()
    cv2.setNumThreads(0)
    args.gpu = 0
    args.rank = 0
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    setup_logger(args.output_dir, distributed_rank=0, filename="eval.log", mode="a")

    if args.score_cache:
        _score_and_write(args.score_cache, args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("ToolRGS model inference currently requires a CUDA GPU")

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

    prediction_records = getattr(val_loop, "prediction_records", None)
    if args.save_predictions and prediction_records is not None:
        cache_path = args.prediction_cache or _default_prediction_cache(args)
        save_prediction_cache(
            cache_path,
            prediction_records,
            metadata=_cache_metadata(args),
        )
        logger.info(
            "Prediction cache: {} ({} samples)",
            cache_path,
            len(prediction_records),
        )
        _score_and_write(cache_path, args)
    elif args.save_predictions:
        logger.warning(
            "Validation loop {!r} does not expose decoded prediction records; "
            "cache was not written",
            getattr(args, "val_loop", "grasp_val"),
        )
    elif args.msr_output:
        rows = getattr(val_loop, "grasp_threshold_grid", None)
        msr = getattr(val_loop, "grasp_msr", None)
        if not rows or not msr:
            raise RuntimeError("The selected validation loop did not produce an mSR grid")
        scores = {
            "topk": tuple(int(value) for value in args.grasp_topk),
            "grasp_threshold_grid": rows,
            "msr": msr,
        }
        write_score_summary(args.msr_output, scores)


if __name__ == "__main__":
    main()
