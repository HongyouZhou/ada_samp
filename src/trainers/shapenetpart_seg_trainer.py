import os
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch as th
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
    def __init__(self, cfg, accelerator: "Accelerator"):
        super().__init__(cfg, accelerator)
        self.model = GMFlow(**cfg.trainer.diffusion, train_cfg=cfg.train, test_cfg=cfg.test_cfg)
        self.seg_emb = nn.Embedding(cfg.trainer.diffusion.embedding.num_classes, cfg.trainer.diffusion.embedding.embed_dim)
        # Point cloud embedding network
        self.pc_embed = nn.Sequential(
            nn.Linear(3, cfg.trainer.diffusion.embedding.pc_embed_dim),  # Assuming point cloud has 3 coordinates (x,y,z)
            nn.SiLU(),
            nn.Linear(cfg.trainer.diffusion.embedding.pc_embed_dim, cfg.trainer.diffusion.embedding.pc_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.trainer.diffusion.embedding.pc_embed_dim, cfg.trainer.diffusion.embedding.embed_dim),
        )
        self.optimizer = th.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.seg_emb.parameters()},
                {"params": self.pc_embed.parameters()},
            ],
            lr=cfg.train.lr,
        )
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

        self.batch_size = cfg.data.train_dataloader.batch_size
        self.num_points = cfg.data.train.num_points

        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.seg_emb, self.pc_embed = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.seg_emb,
            self.pc_embed,
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

        return {"total_loss": total_loss}

    def train_step(self, batch):
        pc, lb, seg, n, f = batch

        seg_emb = self.seg_emb(seg)
        pc_emb = self.pc_embed(pc)
        # input to GMFlow has to be (B, C, H, W)
        # TODO: Proper handling of 3D tensor and point cloud
        seg_emb = seg_emb.reshape(self.batch_size, self.num_points, self.cfg.trainer.diffusion.embedding.embed_dim, 1, 1)
        pc_emb = pc_emb.reshape(self.batch_size, self.num_points, self.cfg.trainer.diffusion.embedding.embed_dim, 1, 1)

        self.optimizer.zero_grad()

        with self.accelerator.autocast():
            loss, log_vars = self.model(seg_emb, pc_emb=pc_emb, return_loss=True)

        self.accelerator.backward(loss)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def evaluate(self, epoch: int):
        pass

    def test(self):
        pass
