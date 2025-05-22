from typing import TYPE_CHECKING

import torch as th
from accelerate import Accelerator

from models.gmflow import GMFlow

from .base_trainer import BaseTrainer

if TYPE_CHECKING:
    from accelerate import Accelerator


class GMFlowTrainer(BaseTrainer):
    def __init__(self, cfg, accelerator: "Accelerator"):
        super().__init__(cfg, accelerator)
        self.model = GMFlow(cfg.model)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=cfg.train.lr)
        self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.train.epochs)

        if cfg.resume:
            self.load_checkpoint(cfg.resume)

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        for batch in train_loader:
            self.optimizer.zero_grad()
            self.optimizer.step()
            self.scheduler.step()

    def evaluate(self):
        pass

    def load_checkpoint(self):
        pass

    def save_checkpoint(self):
        pass

    def setup_dataloaders(self):
        pass
