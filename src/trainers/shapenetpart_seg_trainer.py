import os
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
import torch.nn as nn
from accelerate import Accelerator
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from torch.utils.data import DataLoader, DistributedSampler

from src.datasets.registry import get_dataset
from src.models.gmflow import GMFlow
from src.trainers.base_trainer import BaseTrainer
from src.trainers.registry import register_trainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if TYPE_CHECKING:
    from accelerate import Accelerator


@register_trainer("shapenetpart_seg")
class ShapenetPartSegTrainer(BaseTrainer):
    def __init__(self, cfg, accelerator: "Accelerator", **kwargs):
        super().__init__(cfg, accelerator, **kwargs)
        self.model = GMFlow(**cfg.trainer.diffusion, train_cfg=cfg.train, test_cfg=cfg.test_cfg)
        self.optimizer = th.optim.Adam([{"params": self.model.parameters()}], lr=cfg.train.lr)
        self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.train.epochs)

        if cfg.train.resume_checkpoint:
            self.load_checkpoint(cfg.resume)

        self.train_data = get_dataset(cfg.data.train)
        self.val_data = get_dataset(cfg.data.val)
        if self.accelerator.distributed_type == "NO":
            self.train_dataloader = DataLoader(self.train_data, **cfg.data.train_dataloader)
            self.val_dataloader = DataLoader(self.val_data, **cfg.data.val_dataloader)
        else:
            self.train_dataloader = DataLoader(self.train_data, **cfg.data.train_dataloader, sampler=DistributedSampler(self.train_data))
            self.val_dataloader = DataLoader(self.val_data, **cfg.data.val_dataloader, sampler=DistributedSampler(self.val_data))

        self.num_points = cfg.data.train.num_points

        self.model, self.optimizer, self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
        )

        self.val_metrics = {
            "accuracy": torch.nn.CrossEntropyLoss(),
            "iou": torch.nn.CrossEntropyLoss(),
            "dice": torch.nn.CrossEntropyLoss(),
        }

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0

        if hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(epoch)

        if self.is_main:
            self.progress_manager.add_batch_task(len(self.train_dataloader))

        for step, batch in enumerate(self.train_dataloader):
            loss_dict = self.train_step(batch)
            total_loss += loss_dict["loss"]

            global_step = (epoch - 1) * self.num_processes * len(self.train_dataloader) + step * self.num_processes + self.process_index
            for key, value in loss_dict.items():
                self.log_metrics({f"train/{key}": value}, global_step)
            if self.scheduler is not None:
                self.log_metrics({"lr": self.scheduler.get_last_lr()[0]}, global_step)

            if self.is_main:
                self.progress_manager.update_batch(loss=loss_dict["loss"])

        self.accelerator.wait_for_everyone()

        return {"total_loss": total_loss}

    def train_step(self, batch):
        pc, lb, seg, n, f = batch
        batch_size = seg.shape[0]
        self.optimizer.zero_grad()

        seg = seg.reshape(batch_size, self.num_points, 1, 1, 1)  # input to GMFlow has to be (B, C, H, W, D)
        pc = pc.reshape(batch_size, self.num_points, 3, 1, 1)  # input to GMFlow has to be (B, C, H, W, D)

        with self.accelerator.autocast():
            flow_loss, log_vars = self.model(seg, pc=pc, return_loss=True)

        self.accelerator.backward(flow_loss)

        self.optimizer.step()
        self.scheduler.step()

        return {"loss": flow_loss.item()}

    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        metric_values = {name: 0.0 for name in self.val_metrics}

        if hasattr(self.val_dataloader.sampler, "set_epoch"):
            self.val_dataloader.sampler.set_epoch(epoch)

        if self.is_main:
            self.progress_manager.add_batch_task(len(self.val_dataloader))

        with torch.no_grad():
            for step, batch in enumerate(self.val_dataloader):
                pc, lb, seg, n, f = batch

                batch_size = seg.shape[0]

                seg = seg.reshape(batch_size, self.num_points, 1, 1, 1).float()
                pc = pc.reshape(batch_size, self.num_points, 3, 1, 1)
                with self.accelerator.autocast():
                    seg_emb_pred, velocity_data = self.model(seg, pc=pc, return_loss=False, return_velocity=True)

                if self.is_main:
                    self.progress_manager.update_batch(loss=0)

        self.accelerator.wait_for_everyone()

        num_batches = len(self.val_dataloader)
        avg_metrics = {name: value / num_batches for name, value in metric_values.items()}

        return {"val_loss": total_loss, **avg_metrics}

    def test(self):
        pass
