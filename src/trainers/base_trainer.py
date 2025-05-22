from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from src.samplers.registry import build_noise_sampler, build_timestep_sampler

if TYPE_CHECKING:
    import torch.nn as nn
    from accelerate import Accelerator
    from omegaconf import DictConfig
    from torch.utils.data import DataLoader

    from samplers.noise_sampler import NoiseSampler
    from samplers.timestep_sampler import TimestepSampler


class BaseTrainer(ABC):
    cfg: "DictConfig"
    accelerator: "Accelerator"
    model: Optional["nn.Module"]
    train_dataloader: Optional["DataLoader"]
    val_dataloader: Optional["DataLoader"]
    timestep_sampler: Optional["TimestepSampler"]
    noise_sampler: Optional["NoiseSampler"]

    def __init__(self, cfg: "DictConfig", accelerator: "Accelerator"):
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device
        self.is_main = accelerator.is_main_process

        self.model = None
        self.train_dataloader = None
        self.val_dataloader = None
        
    def train(self, train_loader, val_loader=None):
        for epoch in range(self.cfg.epochs):
            self.train_one_epoch(train_loader, epoch)
            if val_loader:
                self.evaluate(val_loader, epoch)
            if self.is_main:
                self.save_checkpoint(epoch)

    @abstractmethod
    def train_one_epoch(self, train_loader, epoch):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def setup_dataloaders(self):
        raise NotImplementedError

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def log_metrics(self, metrics: dict, step: Optional[int] = None, prefix: str = "train"):
        if self.is_main:
            try:
                import wandb

                wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)
            except ImportError:
                print("[BaseTrainer] wandb not installed; skipping logging.")
