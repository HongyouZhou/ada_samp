import os
import time
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
from src.trainers.registry import register_trainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if TYPE_CHECKING:
    from accelerate import Accelerator

@register_trainer("toymodel")
class CheckboardTrainer(BaseTrainer):
    def __init__(self, cfg, accelerator: "Accelerator", **kwargs):
        super().__init__(cfg, accelerator, **kwargs)
        self.model = GMFlow(**cfg.trainer.diffusion, test_cfg=cfg.test_cfg)
        self.optimizer = th.optim.AdamW(self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        self.data = get_dataset(cfg.data.train)
        self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.train.epochs * len(self.data))

        if cfg.train.resume_checkpoint:
            self.load_checkpoint()

        if self.accelerator.distributed_type == "NO":
            self.train_dataloader = DataLoader(self.data, **cfg.data.train_dataloader)
        else:
            self.train_dataloader = DataLoader(self.data, **cfg.data.train_dataloader, sampler=DistributedSampler(self.data))
        self.val_dataloader = None
        
        self.batch_size = cfg.data.train_dataloader.batch_size

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
        x = batch["x"].reshape(self.batch_size, 2, 1, 1)

        self.optimizer.zero_grad()
        with self.accelerator.autocast():
            loss, log_vars = self.model(x, return_loss=True)

        self.accelerator.backward(loss.mean())

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def validate(self):
        pass

    def test(self):
        self.model.eval()
        with torch.no_grad():
            if self.is_main:
                self.test_step()
            # self.accelerator.wait_for_everyone()

    def test_step(self):
        samples = []

        # Create a rich progress bar
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )

        with progress:
            task = progress.add_task("Generating samples...", total=int(self.cfg.test.n_samples // self.cfg.data.train_dataloader.batch_size))
            for _ in range(int(self.cfg.test.n_samples // self.cfg.data.train_dataloader.batch_size)):
                noise = torch.randn((self.cfg.data.train_dataloader.batch_size, 2, 1, 1), device=self.device)
                x_t, velocity_data = self.model(x_0=noise, return_loss=False, return_velocity=True)
                samples.append(x_t.reshape(self.cfg.data.train_dataloader.batch_size, 2).cpu().detach().numpy())
                progress.update(task, advance=1)

        samples = np.concatenate(samples, axis=0)

        histo, _, _ = np.histogram2d(
            samples[:, 0],
            samples[:, 1],
            bins=200,
            range=[
                [-self.cfg.data.train.scale - 0.2, self.cfg.data.train.scale + 0.2],
                [-self.cfg.data.train.scale - 0.2, self.cfg.data.train.scale + 0.2],
            ],
        )
        histo_image = (histo.T[::-1] / 160).clip(0, 1)
        histo_image = cm.viridis(histo_image)
        histo_image = np.round(histo_image * 255).clip(min=0, max=255).astype(np.uint8)
        plt.imsave(os.path.join(self.out_dir, "sample_histogram.png"), histo_image)

        print(f"Sample histogram saved to {os.path.join(self.out_dir, 'sample_histogram.png')}.")

        self.visualize_velocity_gmm(velocity_data, timestep_idx=0, batch_idx=0, save_path=os.path.join(self.out_dir, "velocity_gmm.png"))

        self.visualize_velocity_evolution(velocity_data, batch_idx=0, num_timesteps=8, save_dir=self.out_dir)

    def visualize_velocity_gmm(self, velocity_data, timestep_idx=0, batch_idx=0, save_path=None):
        """
        Visualize the velocity GMM at a specific timestep using a density plot style.

        Args:
            velocity_data (dict): Dictionary containing velocity GMM data
            timestep_idx (int): Index of the timestep to visualize
            batch_idx (int): Index of the batch to visualize
            channel_idx (int): Index of the channel to visualize
            save_path (str): Path to save the visualization
        """
        # Extract GMM parameters for the specified indices
        means = velocity_data["velocities"]["means"][timestep_idx, batch_idx, :, :, 0, 0].cpu().numpy()
        logstds = velocity_data["velocities"]["logstds"][timestep_idx, batch_idx, :, :, 0, 0].cpu().numpy()
        logweights = velocity_data["velocities"]["logweights"][timestep_idx, batch_idx, :, 0, 0].cpu().numpy()

        # Convert log weights to probabilities
        weights = np.exp(logweights)
        weights = weights / weights.sum()  # Normalize weights

        # Create a grid for the density plot
        x = np.linspace(-self.cfg.data.train.scale - 0.5, self.cfg.data.train.scale + 0.5, 200)
        y = np.linspace(-self.cfg.data.train.scale - 0.5, self.cfg.data.train.scale + 0.5, 200)
        X, Y = np.meshgrid(x, y)
        positions = np.stack([X, Y], axis=-1)

        # Calculate the density for each point
        density = np.zeros_like(X)
        for i in range(len(means)):
            std = np.exp(logstds[0])
            # Calculate Gaussian density
            diff = positions - means[i]
            exponent = -0.5 * np.sum(diff * diff, axis=-1) / (std * std)
            gaussian = np.exp(exponent) / (2 * np.pi * std * std)
            density += weights[i] * gaussian

        # Normalize and create the image
        density = density.T[::-1]  # Transpose and flip vertically
        density = (density / density.max()).clip(0, 1)  # Normalize to [0, 1]
        density_image = cm.viridis(density)
        density_image = np.round(density_image * 255).clip(min=0, max=255).astype(np.uint8)

        if save_path:
            plt.imsave(save_path, density_image)
        else:
            plt.imshow(density_image)
            plt.axis("off")
            plt.show()

    def visualize_velocity_evolution(self, velocity_data, batch_idx=0, num_timesteps=5, save_dir=None):
        """
        Visualize the evolution of velocity GMM over multiple timesteps using density plots.

        Args:
            velocity_data (dict): Dictionary containing velocity GMM data
            batch_idx (int): Index of the batch to visualize
            channel_idx (int): Index of the channel to visualize
            num_timesteps (int): Number of timesteps to visualize
            save_dir (str): Directory to save the visualizations
        """
        # Get timestep indices to visualize
        total_timesteps = len(velocity_data["timesteps"])
        timestep_indices = np.linspace(0, total_timesteps - 1, num_timesteps, dtype=int)

        # Create a grid for the density plot
        x = np.linspace(-self.cfg.data.train.scale - 0.5, self.cfg.data.train.scale + 0.5, 200)
        y = np.linspace(-self.cfg.data.train.scale - 0.5, self.cfg.data.train.scale + 0.5, 200)
        X, Y = np.meshgrid(x, y)
        positions = np.stack([X, Y], axis=-1)

        # Create a figure with subplots
        fig, axes = plt.subplots(1, num_timesteps, figsize=(5 * num_timesteps, 5))
        if num_timesteps == 1:
            axes = [axes]

        for i, timestep_idx in enumerate(timestep_indices):
            # Extract GMM parameters for the specified indices
            means = velocity_data["velocities"]["means"][timestep_idx, batch_idx, :, :, 0, 0].cpu().numpy()
            logstds = velocity_data["velocities"]["logstds"][timestep_idx, batch_idx, :, :, 0, 0].cpu().numpy()
            logweights = velocity_data["velocities"]["logweights"][timestep_idx, batch_idx, :, 0, 0].cpu().numpy()

            # Convert log weights to probabilities
            weights = np.exp(logweights)
            weights = weights / weights.sum()  # Normalize weights

            # Calculate the density for each point
            density = np.zeros_like(X)
            for j in range(len(means)):
                std = np.exp(logstds[0])
                # Calculate Gaussian density
                diff = positions - means[j]
                exponent = -0.5 * np.sum(diff * diff, axis=-1) / (std * std)
                gaussian = np.exp(exponent) / (2 * np.pi * std * std)
                density += weights[j] * gaussian

            # Normalize and create the image
            density = density.T[::-1]  # Transpose and flip vertically
            density = (density / density.max()).clip(0, 1)  # Normalize to [0, 1]
            density_image = cm.viridis(density)
            density_image = np.round(density_image * 255).clip(min=0, max=255).astype(np.uint8)

            # Display the image
            axes[i].imshow(density_image)
            axes[i].axis("off")
            axes[i].set_title(f"Timestep {timestep_idx}")

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/velocity_evolution.png")
            plt.close()
        else:
            plt.show()
