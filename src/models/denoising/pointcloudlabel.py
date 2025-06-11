# src/models/pointcloud_label_denoiser.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .registry import register_model
from .toymodel import Timesteps


@register_model("GMFlowMLPLabelDenoiser")
class GMFlowMLPLabelDenoiser(nn.Module):
    def __init__(
        self,
        num_gaussians=32,
        embed_dim=2048,
        hidden_dim=512,
        pos_min_period=5e-3,
        pos_max_period=50,
        constant_logstd=None,
        num_layers=5,
        num_points=2048, # number of points in each data
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.constant_logstd = constant_logstd
        self.time_proj = Timesteps(num_channels=embed_dim, flip_sin_to_cos=True, downscale_freq_shift=1)

        # Increased input dimension to include point cloud embedding
        in_dim = embed_dim * 5
        mlp = []
        for _ in range(num_layers):
            mlp.append(nn.Linear(in_dim, hidden_dim))
            mlp.append(nn.SiLU())
            in_dim = hidden_dim
        self.net = nn.Sequential(*mlp)
        self.out_means = nn.Linear(hidden_dim, num_gaussians * num_points)
        self.out_logweights = nn.Linear(hidden_dim, num_gaussians * 1)
        if constant_logstd is None:
            self.out_logstds = nn.Linear(hidden_dim, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_logweights.weight)
        if self.constant_logstd is None:
            nn.init.zeros_(self.out_logstds.weight)

    def forward(self, seg, timestep, pc):
        """
        Args:
            seg (torch.Tensor): Shape (B, N, 1, 1, 1) - Segmentation labels
            pc (torch.Tensor): Shape (B, N, 3, 1, 1) - Point cloud coordinates
            timestep (torch.Tensor): Shape (B, N,) - Timesteps
        """
        bs, n, emb_dim= seg.shape[:3]
        extra_dims = seg.shape[3:]
        
        # Process timestep embedding
        t_emb = self.time_proj(timestep).to(seg) # (B, embed_dim)
        seg = seg.reshape(bs, n)
        pc = pc.reshape(bs, n * 3)

        # sum all embeddings
        embeddings = torch.cat([t_emb, pc, seg], dim=-1)
        
        feat = self.net(embeddings)
        
        feat = feat.reshape(bs, -1)
        
        means = self.out_means(feat).reshape(bs, self.num_gaussians, n, 1, *extra_dims) # (B, num_gaussians, num_points, 1, *extra_dims)
        logweights = self.out_logweights(feat).log_softmax(dim=-1).reshape(bs, self.num_gaussians, 1, 1, *extra_dims) # (B, num_gaussians, 1, *extra_dims)
        
        if self.constant_logstd is None:
            logstds = self.out_logstds(feat).reshape(bs, 1, 1, 1, *extra_dims) # (B, 1, n, 1, *extra_dims) isotropic gaussian logstd
        else:
            logstds = torch.full((bs, 1, 1, 1, *extra_dims), self.constant_logstd, dtype=seg.dtype, device=seg.device)
            
        return dict(means=means, logweights=logweights, logstds=logstds)
