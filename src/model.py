from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError(f"dim must be a positive even integer, got {dim}")
        k = dim // 2
        i = torch.arange(k, dtype=torch.float32)
        freqs = torch.exp(-i * math.log(10000.0) / (k - 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        args = t * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowMLP(nn.Module):
    def __init__(
        self,
        data_dim: int,
        time_embed_dim: int = 128,
        hidden_dim: int = 256,
        num_hidden_layers: int = 5,
    ):
        super().__init__()
        if data_dim <= 0:
            raise ValueError(f"data_dim must be > 0, got {data_dim}")
        if time_embed_dim < 2 or time_embed_dim % 2 != 0:
            raise ValueError(f"time_embed_dim must be a positive even integer, got {time_embed_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

        self.time_embed = SinusoidalTimeEmbedding(dim=time_embed_dim)

        layers: list[nn.Module] = []
        in_dim = data_dim + time_embed_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        e_t = self.time_embed(t)
        x = torch.cat([z, e_t], dim=-1)
        return self.net(x)


class MeanFlowMLP(nn.Module):
    def __init__(
        self,
        data_dim: int,
        time_embed_dim: int = 128,
        horizon_embed_dim: int = 128,
        hidden_dim: int = 256,
        num_hidden_layers: int = 5,
    ):
        super().__init__()
        if data_dim <= 0:
            raise ValueError(f"data_dim must be > 0, got {data_dim}")
        if time_embed_dim < 2 or time_embed_dim % 2 != 0:
            raise ValueError(f"time_embed_dim must be a positive even integer, got {time_embed_dim}")
        if horizon_embed_dim < 2 or horizon_embed_dim % 2 != 0:
            raise ValueError(f"horizon_embed_dim must be a positive even integer, got {horizon_embed_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

        self.time_embed = SinusoidalTimeEmbedding(dim=time_embed_dim)
        self.horizon_embed = SinusoidalTimeEmbedding(dim=horizon_embed_dim)

        layers: list[nn.Module] = []
        in_dim = data_dim + time_embed_dim + horizon_embed_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """model(z, r, t) with internal (t, h=t-r) conditioning."""
        h = t - r
        e_t = self.time_embed(t)
        e_h = self.horizon_embed(h)
        x = torch.cat([z, e_t, e_h], dim=-1)
        return self.net(x)
