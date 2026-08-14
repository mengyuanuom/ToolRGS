"""Evaluate an MMDetection detector checkpoint on its configured test split."""

import argparse
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="MMDetection config file")
    parser.add_argument("checkpoint", help="Detector checkpoint")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument(
        "--launcher", choices=("none", "pytorch", "slurm", "mpi"), default="none"
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    for value in (args.config, args.checkpoint):
        if not Path(value).is_file():
            parser.error(f"file does not exist: {value}")
    return args


def main() -> int:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.load_from = args.checkpoint
    cfg.launcher = args.launcher
    if args.work_dir:
        cfg.work_dir = args.work_dir
    Runner.from_cfg(cfg).test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
