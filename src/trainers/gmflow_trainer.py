from typing import TYPE_CHECKING

import torch as th
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler

from src.datasets.registry import get_dataset
from src.models.gmflow import GMFlow
from src.trainers.base_trainer import BaseTrainer
from src.trainers.registry import register_trainer

if TYPE_CHECKING:
    from accelerate import Accelerator


@register_trainer("gmflow")
class GMFlowTrainer(BaseTrainer):
    def __init__(self, cfg, accelerator: "Accelerator"):
        super().__init__(cfg, accelerator)
        self.model = GMFlow(**cfg.model)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=cfg.train.lr)
        self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.train.epochs)

        if cfg.get("resume", None) is not None:
            self.load_checkpoint(cfg.resume)

        self.data = get_dataset(cfg.data.train)
        self.train_loader = DataLoader(self.data, **cfg.data.train_loader, sampler=DistributedSampler(self.data))
        self.val_loader = None

        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        )

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0

        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        for step, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            with self.accelerator.autocast():
                loss = self.compute_loss(batch)

            self.accelerator.backward(loss)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if self.accelerator.is_main_process and step % self.cfg.data.train_loader.batch_size == 0:
                print(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f}")

        avg_loss = self.accelerator.gather(th.tensor([total_loss], device=self.accelerator.device)).mean().item()

        if self.accelerator.is_main_process:
            print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")

    def evaluate(self):
        pass

    def load_checkpoint(self):
        pass

    def save_checkpoint(self, epoch: int):
        pass

    def setup_dataloaders(self):
        pass
