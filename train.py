"""Single configuration-driven trainer for ToolRGS models and datasets."""

import argparse
import datetime
import os
from functools import partial
from pathlib import Path
import shutil
import time

import cv2
from loguru import logger
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utils.config as config
from model import build_model
from toolrgs.engine import GraspTrainLoop, GraspValLoop  # register default loops
from toolrgs.models.base import model_requires_depth
from toolrgs.registry import LOOPS
from utils.data_builder import build_dataset
from utils.misc import init_random_seed, set_random_seed, setup_logger, worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train ToolRGS")
    parser.add_argument("--config", required=True, help="Experiment YAML file")
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    cli = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli.config)
    if cli.opts:
        cfg = config.merge_cfg_from_list(cfg, cli.opts)
    return cfg


def setup_distributed(args):
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        torch.cuda.set_device(0)
    args.distributed = distributed
    return distributed


def _checked_file(value, label):
    """Resolve one configured local file and fail with an actionable message."""
    if not value:
        return None
    text = str(value)
    if text.startswith(("http://", "https://")):
        raise ValueError(
            f"Configured {label} must be a local file, not a URL: {text}\n"
            "Download it first and update the YAML path."
        )
    path = Path(text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Configured {label} does not exist: {path.resolve()}\n"
            f"Update the corresponding YAML value or download the required file."
        )
    return str(path)


def validate_configured_files(args):
    """Check model/checkpoint files before expensive model construction."""
    # MambaVision deliberately owns its optional automatic download path, so
    # only the weights that are synchronously opened by ToolRGS are checked.
    for key in ("clip_pretrain", "dino_pretrain", "depth_pretrain"):
        value = getattr(args, key, None)
        if value:
            setattr(args, key, _checked_file(value, key))
    for key in ("weight", "resume"):
        value = getattr(args, key, None)
        if value:
            setattr(args, key, _checked_file(value, key))


def load_initial_weight(model, filename):
    """Load model initialization without restoring optimizer/epoch state."""
    checkpoint = torch.load(filename, map_location="cpu")
    state = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported initial weight payload: {filename}")
    cleaned = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(cleaned, strict=False)
    logger.info("Loaded initial model weight: {}", filename)
    if incompatible.missing_keys:
        logger.warning(
            "Initial weight did not contain {} model keys (expected for staged heads): {}",
            len(incompatible.missing_keys),
            incompatible.missing_keys[:10],
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "Initial weight contained {} unused keys: {}",
            len(incompatible.unexpected_keys),
            incompatible.unexpected_keys[:10],
        )


def main():
    args = parse_args()
    validate_configured_files(args)
    if not torch.cuda.is_available():
        raise RuntimeError("ToolRGS training currently requires a CUDA GPU")

    cv2.setNumThreads(0)
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)
    distributed = setup_distributed(args)
    is_main = args.rank == 0

    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    setup_logger(args.output_dir, distributed_rank=args.rank,
                 filename="train.log", mode="a")
    logger.info(args)

    model, parameter_groups = build_model(args)
    if model_requires_depth(model) and not bool(getattr(args, "with_depth", False)):
        raise ValueError(
            f"Model {args.architecture!r} requires aligned depth input, but "
            "DATA.with_depth is false or missing. ETRG-A is currently supported "
            "with the OCID-VLG RGB-D dataset."
        )
    if getattr(args, "weight", None):
        load_initial_weight(model, args.weight)
    if args.sync_bn and distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(args.gpu)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    optimizer = torch.optim.Adam(
        parameter_groups, lr=args.base_lr, weight_decay=args.weight_decay
    )
    scheduler = MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_decay
    )
    scaler = torch.cuda.amp.GradScaler()

    needs_offset = args.architecture.lower() in {"crogoff", "drogoff"}
    train_data = build_dataset(args, args.train_split, with_offset=needs_offset)
    val_data = build_dataset(args, args.val_split, with_offset=needs_offset)

    train_sampler = DistributedSampler(train_data, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_data, shuffle=False) if distributed else None
    init_fn = partial(
        worker_init_fn,
        num_workers=args.workers,
        rank=args.rank,
        seed=args.manual_seed,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn if args.workers else None,
        collate_fn=train_data.collate_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers_val,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_data.collate_fn,
    )

    best_iou = 0.0
    best_j = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.gpu}")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        args.start_epoch = checkpoint["epoch"]
        best_iou = checkpoint.get("best_iou", 0.0)
        best_j = checkpoint.get("best_j_index", 0.0)

    train_loop_class = LOOPS.require(getattr(args, "train_loop", "grasp_train"))
    train_loop = train_loop_class(
        dataloader=train_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        cfg=args,
        hooks=getattr(args, "hooks", None),
    )
    val_loop_class = LOOPS.require(getattr(args, "val_loop", "grasp_val"))
    val_loop = val_loop_class(
        dataloader=val_loader,
        model=model,
        cfg=args,
        hooks=getattr(args, "val_hooks", None),
    )

    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_number = epoch + 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_number)

        train_loop.run_epoch(epoch_number)
        iou, precision, j_index = val_loop.run_epoch(epoch_number)

        if is_main:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            last_path = os.path.join(args.output_dir, "last_model.pth")
            torch.save(
                {
                    "epoch": epoch_number,
                    "best_iou": best_iou,
                    "best_j_index": best_j,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "precision": precision,
                    "j_index": j_index,
                },
                last_path,
            )
            if iou >= best_iou:
                best_iou = iou
                shutil.copyfile(last_path, os.path.join(args.output_dir, "best_iou_model.pth"))
            if j_index[0] >= best_j:
                best_j = j_index[0]
                shutil.copyfile(last_path, os.path.join(args.output_dir, "best_jindex_model.pth"))
        scheduler.step()

    logger.info("Training time: {}", datetime.timedelta(seconds=int(time.time() - start)))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
