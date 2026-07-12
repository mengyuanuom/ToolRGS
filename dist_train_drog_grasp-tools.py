import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial
from collections import OrderedDict

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = "99ee90fdefff711f21b8b40a0fac1bdb95da2aa5"

import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
from utils.dataset import GraspToolDataset
from engine.engine import train_with_grasp, validate_with_grasp, validate_without_grasp
from model import build_drog
from utils.misc import (
    init_random_seed, set_random_seed, setup_logger, worker_init_fn
)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pytorch Referring Expression Segmentation"
    )

    parser.add_argument(
        "--config",
        default="config/grasp-tools/drog_dino.yaml",
        type=str,
        help="config file"
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="override some settings in the config."
    )

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    # -------------------------
    # Parse cfg & set seed
    # -------------------------
    args = get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    # -------------------------
    # Torchrun distributed env
    # -------------------------
    # torchrun 会自动设置:
    # RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    # 为了兼容偶尔 python 直跑，这里给默认值
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.gpu = int(os.environ.get("LOCAL_RANK", 0))
    args.ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    # -------------------------
    # basic logging / exp dir
    # -------------------------
    args.exp_name = "_".join([
        args.exp_name,
        str(args.ladder_dim),
        str(args.nhead),
        str(args.dim_ffn),
        str(args.multi_stage)
    ])
    args.exp_name = args.exp_name + datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
    args.output_dir = os.path.join(args.output_folder, args.exp_name)

    args.gpu = gpu
    torch.cuda.set_device(args.gpu)

    # logger
    setup_logger(
        args.output_dir,
        distributed_rank=args.rank,  # 用全局rank更合理
        filename="train.log",
        mode="a"
    )

    # -------------------------
    # dist init (env://)
    # -------------------------
    distributed = args.world_size > 1
    if distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank
        )
        dist.barrier()

    # -------------------------
    # build model
    # -------------------------
    model, param_list = build_drog(args)
    if args.sync_bn and distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logger.info(model)
    logger.info(args)

    # -------------------------
    # optimizer & scheduler
    # -------------------------
    optimizer = torch.optim.Adam(
        param_list,
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.lr_decay
    )
    scaler = amp.GradScaler()

    # DDP wrap
    model = model.cuda()
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True
        )

    # -------------------------
    # build dataset
    # -------------------------
    # torchrun 下每个进程只吃自己的 batch
    if distributed:
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / args.ngpus_per_node)
        args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)

    train_data = GraspToolDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            split='train')
    val_data = GraspToolDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            split='val')

    init_fn = partial(
        worker_init_fn,
        num_workers=args.workers,
        rank=args.rank,
        seed=args.manual_seed
    )

    if distributed:
        train_sampler = data.distributed.DistributedSampler(train_data, shuffle=True)
        val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)
        shuffle_train = False
        shuffle_val = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True
        shuffle_val = False

    train_loader = data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=init_fn,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=GraspToolDataset.collate_fn
    )
    val_loader = data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=shuffle_val,
        num_workers=args.workers_val,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=GraspToolDataset.collate_fn
    )

    best_IoU = 0.0
    best_j_index = 0.0

    # -------------------------
    # resume
    # -------------------------
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            map_location = {"cuda:%d" % 0: "cuda:%d" % gpu}
            checkpoint = torch.load(args.resume, map_location=map_location)
            args.start_epoch = checkpoint["epoch"]
            best_IoU = checkpoint["best_iou"]
            best_j_index = checkpoint["best_j_index"]
            ckpt_state =checkpoint["state_dict"]
            # 如果是 DDP，真正的模型在 model.module 里；否则就是 model 本身
            target_model = getattr(model, "module", model)

            model_state = target_model.state_dict()

            # 只保留：名字相同 且 shape 完全一致 的参数
            compatible_state = {}
            for k, v in ckpt_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    compatible_state[k] = v
                # else:
                #     print(f"[skip] {k}: not in model or shape mismatch "
                #           f"ckpt={tuple(v.shape)}, model={tuple(model_state.get(k, torch.empty(0)).shape)}")

            # 加载筛选后的权重，strict=False 忽略没覆盖到的
            msg = target_model.load_state_dict(compatible_state, strict=False)

            print("[load_state_dict] loaded keys:", len(compatible_state))
            print("[load_state_dict] missing keys:", msg.missing_keys)
            print("[load_state_dict] unexpected keys:", msg.unexpected_keys)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            del checkpoint
            torch.cuda.empty_cache()
        else:
            raise ValueError(
                f"=> resume failed! no checkpoint found at '{args.resume}'. Please check args.resume again!"
            )

    # -------------------------
    # start training
    # -------------------------
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        if distributed:
            train_sampler.set_epoch(epoch_log)

        train_with_grasp(
            train_loader, model, optimizer, scheduler, scaler, epoch_log, args
        )

        if args.use_grasp_masks:
            iou, prec_dict, j_index = validate_with_grasp(val_loader, model, epoch_log, args)
        else:
            iou, prec_dict, j_index = validate_without_grasp(val_loader, model, epoch_log, args)

        # save model only on rank0
        if (not distributed) or dist.get_rank() == 0:
            lastname = os.path.join(args.output_dir, "last_model.pth")
            torch.save(
                {
                    "epoch": epoch_log,
                    "cur_iou": iou,
                    "best_iou": best_IoU,
                    "best_j_index": best_j_index,
                    "prec": prec_dict,
                    "j_index": j_index,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                },
                lastname
            )

            if iou >= best_IoU:
                best_IoU = iou
                bestname = os.path.join(args.output_dir, "best_iou_model.pth")
                shutil.copyfile(lastname, bestname)

            if j_index[0] >= best_j_index:
                best_j_index = j_index[0]
                bestname = os.path.join(args.output_dir, "best_jindex_model.pth")
                shutil.copyfile(lastname, bestname)

        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    logger.info(f"* Best IoU={best_IoU} *")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"* Training time {total_time_str} *")


if __name__ == "__main__":
    main()
    sys.exit(0)
