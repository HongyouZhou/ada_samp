import argparse
import os

from accelerate import Accelerator
from omegaconf import OmegaConf

import wandb
from src.trainers import get_trainer

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gmflow.yaml", help="Path to base config")
    parser.add_argument("--enable_sweep", action="store_true", help="Enable wandb sweep")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb")
    return parser.parse_args()


def load_config(config_path: str, enable_sweep: bool = False):
    cfg = OmegaConf.load(config_path)

    if wandb.run is not None and wandb.config and enable_sweep:
        print("[train.py] Detected W&B sweep — merging wandb.config into base config")
        sweep_cfg = OmegaConf.create(dict(wandb.config))
        cfg = OmegaConf.merge(cfg, sweep_cfg)
        cfg._sweep_mode = True
    else:
        cfg._sweep_mode = False

    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config, args.enable_sweep)

    if args.enable_wandb:
        wandb.init(
            project=cfg.get("project", "gmflow_project"),
            config=cfg,
        )

    if cfg._sweep_mode and "lr" in cfg.train:
        wandb.run.name = f"sweep_lr{cfg.train.lr:.0e}"

    accelerator = Accelerator()

    TrainerClass = get_trainer(cfg)
    trainer = TrainerClass(cfg, accelerator)
    if cfg.train.resume_checkpoint:
        trainer.load_checkpoint()
    trainer.train()
    trainer.test()
    accelerator.end_training()


if __name__ == "__main__":
    main()
