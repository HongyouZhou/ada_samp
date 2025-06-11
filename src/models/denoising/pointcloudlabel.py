# src/models/pointcloud_label_denoiser.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .registry import register_model
from .toymodel import Timesteps, get_1d_sincos_pos_embed


class SinCos1DPosEmbed(nn.Module):
    def __init__(self, num_channels=256, min_period=1e-3, max_period=10):
        super().__init__()
        self.num_channels = num_channels
        self.min_period = min_period
        self.max_period = max_period
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states (torch.Tensor): Shape (B, 1)

        Returns:
            torch.Tensor: Shape (B, num_channels)
        """
        return get_1d_sincos_pos_embed(self.num_channels, hidden_states, self.min_period, self.max_period)


@register_model("GMFlowMLPLabelDenoiser")
class GMFlowMLPLabelDenoiser(nn.Module):
    def __init__(
        self,
        num_gaussians=32,
        embed_dim=256,
        hidden_dim=512,
        constant_logstd=None,
        num_layers=5,
        num_classes=10,
        num_points=2048, # number of points in each data
        pc_embed_dim=256,  # New parameter for point cloud embedding dimension
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.constant_logstd = constant_logstd
        self.time_proj = Timesteps(num_channels=embed_dim, flip_sin_to_cos=True, downscale_freq_shift=1)
        

        # Increased input dimension to include point cloud embedding
        in_dim = embed_dim
        mlp = []
        for _ in range(num_layers):
            mlp.append(nn.Linear(in_dim, hidden_dim))
            mlp.append(nn.SiLU())
            in_dim = hidden_dim
        self.net = nn.Sequential(*mlp)
        self.out_means = nn.Linear(hidden_dim, num_gaussians * 1)
        self.out_logweights = nn.Linear(hidden_dim, num_gaussians)
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

    def forward(self, seg_emb, timestep, pc_emb):
        """
        Args:
            seg_emb (torch.Tensor): Shape (B, N, embed_dim, 1, 1) - Segmentation labels
            pc_emb (torch.Tensor): Shape (B, N, embed_dim, 1, 1) - Point cloud coordinates
            timestep (torch.Tensor): Shape (B, N,) - Timesteps
        """
        bs, n, emb_dim= seg_emb.shape[:3]
        extra_dims = seg_emb.shape[3:]
        
        # Process timestep embedding
        t_emb = self.time_proj(timestep).unsqueeze(1).to(seg_emb)
        seg_emb = seg_emb.reshape(bs, n, emb_dim)
        pc_emb = pc_emb.reshape(bs, n, emb_dim)
        # sum all embeddings
        embeddings = t_emb + seg_emb + pc_emb
        embeddings = embeddings.reshape(bs, n, emb_dim)
        
        # Process through MLP
        feat = self.net(embeddings)
        
        # Generate output distributions
        means = self.out_means(feat).reshape(bs, self.num_gaussians, n, 1, *extra_dims) # (B, num_gaussians, num_points, 1, *extra_dims)
        logweights = self.out_logweights(feat).log_softmax(dim=-1).reshape(bs, self.num_gaussians, n, 1, *extra_dims) # (B, num_gaussians, num_points, 1, *extra_dims)
        
        if self.constant_logstd is None:
            logstds = self.out_logstds(feat).reshape(bs, 1, n, 1, *extra_dims) # (B, 1, n, 1, *extra_dims) isotropic gaussian logstd
        else:
            logstds = torch.full((bs, 1, n, 1, *extra_dims), self.constant_logstd, dtype=seg_emb.dtype, device=seg_emb.device)
            
        return dict(means=means, logweights=logweights, logstds=logstds)
