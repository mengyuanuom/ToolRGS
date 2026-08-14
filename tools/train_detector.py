"""Train an MMDetection detector from a repository config."""

import argparse
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="MMDetection config file")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument(
        "--launcher", choices=("none", "pytorch", "slurm", "mpi"), default="none"
    )
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, help="Override config values"
    )
    args = parser.parse_args()
    if not Path(args.config).is_file():
        parser.error(f"config does not exist: {args.config}")
    return args


def main() -> int:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    cfg.launcher = args.launcher
    if args.resume:
        cfg.resume = True
        cfg.load_from = None if args.resume == "auto" else args.resume
    Runner.from_cfg(cfg).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
