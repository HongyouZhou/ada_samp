from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

if TYPE_CHECKING:
    import torch.nn as nn
    from accelerate import Accelerator
    from omegaconf import DictConfig
    from torch.utils.data import DataLoader

    from samplers.noise_sampler import NoiseSampler
    from samplers.timestep_sampler import TimestepSampler


class ProgressManager:
    def __init__(self, total_epochs: int, total_steps: int):
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.console = Console()

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[progress.completed]{task.completed}/{task.total} epochs"),
            TextColumn("•"),
            TextColumn("[yellow]Loss: {task.fields[loss]:.4f}"),
            TimeRemainingColumn(),
            transient=True,
            console=self.console
        )

        self.live = Live(
            renderable=self.progress,
            refresh_per_second=50,
            transient=True
        )

        self.epoch_task = self.progress.add_task(
            "[green]Epoch Progress", 
            total=total_epochs,
            loss=0.0
        )
        self.batch_task = None

    def start(self):
        self.live.start()

    def stop(self):
        self.live.stop()

    def update_epoch(self, advance: int = 1):
        self.progress.update(self.epoch_task, advance=advance)

    def update_epoch_loss(self, loss: float):
        self.progress.update(self.epoch_task, loss=loss)

    def add_batch_task(self, total_steps: int):
        if self.batch_task is not None:
            self.progress.remove_task(self.batch_task)

        self.batch_task = self.progress.add_task(
            "[blue]Batch Progress", 
            total=total_steps,
            loss=0.0
        )

    def update_batch(self, advance: int = 1, loss: float = None):
        if self.batch_task is not None:
            if loss is not None:
                self.progress.update(self.batch_task, advance=advance, loss=loss)
            else:
                self.progress.update(self.batch_task, advance=advance)

    def reset_batch(self):
        if self.batch_task is not None:
            self.progress.reset(self.batch_task)


class BaseTrainer(ABC):
    cfg: "DictConfig"
    accelerator: "Accelerator"
    model: Optional["nn.Module"]
    train_dataloader: Optional["DataLoader"]
    val_dataloader: Optional["DataLoader"]
    timestep_sampler: Optional["TimestepSampler"]
    noise_sampler: Optional["NoiseSampler"]
    progress_manager: Optional["ProgressManager"]

    def __init__(self, cfg: "DictConfig", accelerator: "Accelerator"):
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device
        self.is_main = accelerator.is_main_process
        self.current_epoch = 0

        self.model = None
        self.train_dataloader = None
        self.val_dataloader = None

    def train(self):
        if self.is_main:
            total_steps = self.cfg.train.epochs * len(self.train_dataloader)
            self.progress_manager = ProgressManager(self.cfg.train.epochs, total_steps)
            self.progress_manager.update_epoch(self.current_epoch)
            self.progress_manager.start()

        for epoch in range(self.current_epoch + 1, self.cfg.train.epochs + 1):
            self.current_epoch = epoch
            loss_dict = self.train_epoch(epoch)
            loss_tensor = torch.tensor(loss_dict["total_loss"], device=self.device)
            gathered = self.accelerator.gather(loss_tensor)
            avg_epoch_loss = gathered.float().mean().item()
            avg_epoch_loss /= self.accelerator.num_processes
            avg_epoch_loss /= len(self.train_dataloader)
            if self.val_dataloader:
                self.evaluate(epoch)
            if self.is_main:
                self.log_metrics(loss_dict, step=epoch, prefix="train")
                if self.cfg.train.save_checkpoint and epoch % self.cfg.train.save_interval == 0:
                    self.save_checkpoint(epoch)
                self.progress_manager.update_epoch()
                self.progress_manager.update_epoch_loss(avg_epoch_loss)
        if self.is_main:
            self.progress_manager.stop()

    @abstractmethod
    def train_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, epoch):
        metrics = self._evaluate_local(epoch)
        for k, v in metrics.items():
            t = torch.tensor(v, device=self.device)
            gathered = self.accelerator.gather(t)
            metrics[k] = gathered.float().mean().item()
        if self.is_main:
            self.log_metrics(metrics, step=epoch, prefix="val")
        return metrics

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def load_checkpoint(self):
        pass

    def log_metrics(self, metrics: dict, step: int | None = None, prefix: str = "train"):
        if self.cfg.debug:
            return

        if self.is_main:
            try:
                import wandb

                wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)
            except ImportError:
                print("[BaseTrainer] wandb not installed; skipping logging.")
