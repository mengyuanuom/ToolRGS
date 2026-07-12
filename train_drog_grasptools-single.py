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
os.environ["WANDB_API_KEY"] = '99ee90fdefff711f21b8b40a0fac1bdb95da2aa5'

import cv2
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
from utils.dataset import GraspToolDataset
from engine.engine import train_with_grasp, validate_with_grasp, validate_without_grasp
from model import build_drog
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch DROG Training (non-distributed)'
    )
    parser.add_argument('--config',
                        default='config/grasp-tools/drog_dino.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)

    return cfg


@logger.catch
def main():
    args = get_parser()

    # random seed
    args.manual_seed = init_random_seed(getattr(args, "manual_seed", None))
    set_random_seed(args.manual_seed, deterministic=False)

    # --------- GPU 设置：在代码/配置里指定要用哪几块卡 ---------
    all_gpu_num = torch.cuda.device_count()
    logger.info(f"Detected {all_gpu_num} CUDA device(s).")

    # config 里建议加一项：gpu_ids: [0, 1] 或 [0]
    if not hasattr(args, "gpu_ids") or args.gpu_ids is None:
        # 如果没配，默认用所有可用 GPU
        args.gpu_ids = list(range(all_gpu_num))

    # 兼容 int 写法：gpu_ids: 0
    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    # 过滤非法 id
    args.gpu_ids = [int(g) for g in args.gpu_ids]
    args.gpu_ids = [g for g in args.gpu_ids if 0 <= g < all_gpu_num]

    if torch.cuda.is_available():
        if len(args.gpu_ids) == 0:
            raise ValueError(
                f"No valid GPU id in args.gpu_ids, but CUDA is available. "
                f"Detected {all_gpu_num} device(s)."
            )
    else:
        logger.warning("CUDA not available, will run on CPU (very slow).")
        args.gpu_ids = []

    args.ngpus_per_node = len(args.gpu_ids) if torch.cuda.is_available() else 0
    args.rank = 0
    args.world_size = 1
    args.distributed = False

    logger.info(
        f"Local training. world_size={args.world_size}, use_gpu={torch.cuda.is_available()}, "
        f"gpu_ids={args.gpu_ids}"
    )

    main_worker(args)


def main_worker(args):
    # output dir
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=0,
                 filename="train.log",
                 mode="a")

    # build model
    model, param_list = build_drog(args)
    if getattr(args, "sync_bn", False):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logger.info(model)
    logger.info(args)

    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,
                            gamma=args.lr_decay)
    scaler = amp.GradScaler()

    # ---- GPU 包装：单机单/多卡，使用 gpu_ids ----
    if torch.cuda.is_available() and len(args.gpu_ids) > 0:
        if len(args.gpu_ids) > 1:
            logger.info(f"Using nn.DataParallel on GPUs: {args.gpu_ids}")
            torch.cuda.set_device(args.gpu_ids[0])
            model = nn.DataParallel(model, device_ids=args.gpu_ids)
            model = model.cuda(args.gpu_ids[0])
        else:
            logger.info(f"Using single GPU: {args.gpu_ids[0]}")
            torch.cuda.set_device(args.gpu_ids[0])
            model = model.cuda(args.gpu_ids[0])
    else:
        logger.warning("Using CPU for training.")

    # ---- dataset & dataloader ----
    # 不再按 GPU 数量缩放 batch，直接用 config 配好的
    train_data = GraspToolDataset(root_dir=args.root_path,
                                  input_size=args.input_size,
                                  split='train')
    val_data = GraspToolDataset(root_dir=args.root_path,
                                input_size=args.input_size,
                                split='val')

    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)

    train_loader = data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=init_fn if args.workers > 0 else None,
        drop_last=True,
        collate_fn=GraspToolDataset.collate_fn
    )

    val_loader = data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers_val,
        pin_memory=True,
        drop_last=False,
        collate_fn=GraspToolDataset.collate_fn
    )

    logger.info(">>> Try to load one batch from train_loader for debug...")
    try:
        _ = next(iter(train_loader))
        logger.info(">>> Successfully loaded one batch from train_loader.")
    except Exception as e:
        logger.error(f"!!! Error when loading batch from train_loader: {e}")
        raise

    best_IoU = 0.0
    best_j_index = 0.0

    # 起始 epoch
    if not hasattr(args, "start_epoch"):
        args.start_epoch = 0

    # ---- resume ----
    if getattr(args, "resume", None):
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = torch.load(args.resume, map_location=map_location)

            args.start_epoch = checkpoint.get('epoch', 0)
            best_IoU = checkpoint.get("best_iou", 0.0)
            best_j_index = checkpoint.get("best_j_index", 0.0)

            # 如果之前是 DDP/DataParallel 保存的，state_dict 可能有 module. 前缀
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[len("module."):]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)

            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {args.start_epoch})")

            del checkpoint
            torch.cuda.empty_cache()
        else:
            raise ValueError(
                f"=> resume failed! no checkpoint found at '{args.resume}'. "
                f"Please check args.resume again!"
            )

    # ---- start training ----
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # train
        train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch_log, args)

        # evaluation
        if args.use_grasp_masks:
            iou, prec_dict, j_index = validate_with_grasp(val_loader, model, epoch_log, args)
        else:
            iou, prec_dict, j_index = validate_without_grasp(val_loader, model, epoch_log, args)

        # save model
        lastname = os.path.join(args.output_dir, "last_model.pth")
        torch.save(
            {
                'epoch': epoch_log,
                'cur_iou': iou,
                'best_iou': best_IoU,
                'best_j_index': best_j_index,
                'prec': prec_dict,
                'j_index': j_index,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, lastname)

        if iou >= best_IoU:
            best_IoU = iou
            bestname = os.path.join(args.output_dir, "best_iou_model.pth")
            shutil.copyfile(lastname, bestname)

        if j_index[0] >= best_j_index:
            best_j_index = j_index[0]
            bestname = os.path.join(args.output_dir, "best_jindex_model.pth")
            shutil.copyfile(lastname, bestname)

        # update lr
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(2)

    logger.info(f"* Best IoU={best_IoU} *")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'* Training time {total_time_str} *')


if __name__ == '__main__':
    main()
    sys.exit(0)
