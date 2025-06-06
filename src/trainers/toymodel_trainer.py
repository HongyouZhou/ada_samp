import os
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
from accelerate import Accelerator
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from torch.utils.data import DataLoader, DistributedSampler

from src.datasets.registry import get_dataset
from src.models.gmflow import GMFlow
from src.trainers.base_trainer import BaseTrainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

        return {"total_loss": total_loss}

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
    
    def test(self):
        self.model.eval()
        samples = []
        
        # Create a rich progress bar
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        
        with progress:
            task = progress.add_task("Generating samples...", total=int(self.cfg.test.n_samples // self.cfg.train.batch_size))
            for _ in range(int(self.cfg.test.n_samples // self.cfg.train.batch_size)):
                noise = torch.randn((self.cfg.train.batch_size, 2, 1, 1), device=self.device)
                samples.append(self.model(x_0=noise, return_loss=False).reshape(self.cfg.train.batch_size, 2).cpu().detach().numpy())
                progress.update(task, advance=1)
                
        samples = np.concatenate(samples, axis=0)

        histo, _, _ = np.histogram2d(
            samples[:, 0], samples[:, 1], bins=200, range=[[-4.2, 4.2], [-4.2, 4.2]])
        histo_image = (histo.T[::-1] / 160).clip(0, 1)
        histo_image = cm.viridis(histo_image)
        histo_image = np.round(histo_image * 255).clip(min=0, max=255).astype(np.uint8)

        out_path = os.path.abspath(self.cfg.test.out_path)
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        plt.imsave(out_path, histo_image)

        print(f'Sample histogram saved to {out_path}.')

    def load_checkpoint(self):
        # find the latest checkpoint
        checkpoint_dir = os.path.join(self.cfg.train.output_dir, "checkpoints")
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        self.accelerator.load_state(latest_checkpoint)

    def save_checkpoint(self, epoch: int):
        self.accelerator.save_state(os.path.join(self.cfg.train.output_dir, f"checkpoint_{epoch}.pt"))
