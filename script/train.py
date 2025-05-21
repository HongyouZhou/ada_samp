import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf

from src.trainers import get_trainer


def main():
    cfg = OmegaConf.load("configs/gmflow.yaml")

    wandb.init(project="gmflow_sweep", config=cfg)

    cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(wandb.config)))

    accelerator = Accelerator()
    trainer = get_trainer(cfg.method)(cfg, accelerator)
    trainer.train()


if __name__ == "__main__":
    main()
