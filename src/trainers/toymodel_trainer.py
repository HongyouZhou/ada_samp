from typing import TYPE_CHECKING

import torch as th
from accelerate import Accelerator
from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, ProgressColumn, TextColumn, TimeRemainingColumn
from torch.utils.data import DataLoader, DistributedSampler

from src.datasets.registry import get_dataset
from src.models.gmflow import GMFlow
from src.trainers.base_trainer import BaseTrainer

if TYPE_CHECKING:
    from accelerate import Accelerator


class CheckboardTrainer(BaseTrainer):
    def __init__(self, cfg, accelerator: "Accelerator"):
        super().__init__(cfg, accelerator)
        self.model = GMFlow(**cfg.model)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=cfg.train.lr)
        self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.train.epochs)

        if cfg.get("resume", None) is not None:
            self.load_checkpoint(cfg.resume)

        self.data = get_dataset(cfg.data.train)
        self.train_dataloader = DataLoader(self.data, **cfg.data.train_dataloader, sampler=DistributedSampler(self.data))
        self.val_dataloader = None

        self.model, self.optimizer, self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
        )

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0

        if hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(epoch)

        if self.is_main:
            self.progress_manager.add_batch_task(len(self.train_dataloader))

        for step, batch in enumerate(self.train_dataloader):
            loss = self.train_step(batch)
            total_loss += loss

            if self.is_main:
                self.progress_manager.update_batch(loss=loss)

        self.accelerator.wait_for_everyone()

        if self.is_main:
            avg_loss = total_loss / len(self.train_dataloader)
            self.accelerator.print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
            self.progress_manager.update_epoch(loss=avg_loss)

    def train_step(self, batch):
        bs = batch["x"].size(0)

        self.optimizer.zero_grad()

        with self.accelerator.autocast():
            loss, log_vars = self.model(batch["x"].reshape(bs, 2, 1, 1), return_loss=True)

        self.accelerator.backward(loss)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def evaluate(self):
        pass

    def load_checkpoint(self):
        pass

    def save_checkpoint(self, epoch: int):
        pass

    def setup_dataloaders(self):
        pass
