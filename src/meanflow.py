from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.flow import sample_time


def sample_meanflow_times(
    batch_size: int,
    device: torch.device | str,
    eps: float = 0.005,
    flow_matching_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample r and t for MeanFlow training. Both shapes [B, 1].

    mask=True  (flow_matching_ratio fraction): r = t  → standard FM (h=0)
    mask=False: r ~ Uniform(0, t)              → mean-flow interval
    """
    t = sample_time(batch_size, device, eps=eps, schedule="uniform")

    u_rand = torch.rand(batch_size, 1, device=device)
    r = u_rand * t

    mask = torch.rand(batch_size, 1, device=device) < flow_matching_ratio
    r = torch.where(mask, t, r)

    return r, t


def make_meanflow_batch(
    x: torch.Tensor,
    time_eps: float = 0.005,
    flow_matching_ratio: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Build training tensors for one MeanFlow batch."""
    B, device = x.shape[0], x.device

    eps = torch.randn_like(x)
    r, t = sample_meanflow_times(
        B, device, eps=time_eps, flow_matching_ratio=flow_matching_ratio
    )

    z_t = (1.0 - t) * x + t * eps
    v = eps - x

    return {"x": x, "eps": eps, "z_t": z_t, "v": v, "r": r, "t": t}


def meanflow_loss(
    model: nn.Module,
    x: torch.Tensor,
    time_eps: float = 0.005,
    flow_matching_ratio: float = 0.5,
    loss_p: float = 1.0,
    loss_c: float = 1e-3,
) -> torch.Tensor:
    """MeanFlow training loss — paper Algorithm 1.

    Model interface: model(z, r, t)
    JVP tangents: (dz/dt, dr/dt, dt/dt) = (v, 0, 1)
      r is held fixed during the JVP, so dr/dt = 0.

    Target: u_tgt = v - (t - r) * dudt
    When r = t: u_tgt = v  →  reduces to standard Flow Matching.
    """
    batch = make_meanflow_batch(x, time_eps=time_eps, flow_matching_ratio=flow_matching_ratio)
    z, r, t, v = batch["z_t"], batch["r"], batch["t"], batch["v"]

    def fn(z_in: torch.Tensor, r_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        return model(z_in, r_in, t_in)

    u, dudt = torch.func.jvp(
        fn,
        (z, r, t),
        (v, torch.zeros_like(r), torch.ones_like(t)),
    )

    u_tgt = (v - (t - r) * dudt).detach()

    error = u - u_tgt
    per_sample_l2 = error.pow(2).sum(dim=1)

    if loss_p == 0.0:
        loss = per_sample_l2.mean()
    else:
        weight = 1.0 / (per_sample_l2.detach() + loss_c).pow(loss_p)
        loss = (weight * per_sample_l2).mean()

    return loss


@torch.no_grad()
def sample_meanflow(
    model: nn.Module,
    num_samples: int,
    dim: int,
    num_steps: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Generate samples using MeanFlow ODE.

    Starts from z ~ N(0,I) at t=1 and steps toward t=0.
    For each interval [r_i, t_i]:
        u = model(z, r_batch, t_batch)
        z = z - (t - r) * u

    Example with num_steps=5:
        t=1.0, r=0.8 → t=0.8, r=0.6 → ... → t=0.2, r=0.0
    """
    was_training = model.training
    model.eval()

    z = torch.randn(num_samples, dim, device=device)
    times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_i = float(times[i].item())
        r_i = float(times[i + 1].item())

        t_batch = torch.full((num_samples, 1), t_i, device=device)
        r_batch = torch.full((num_samples, 1), r_i, device=device)

        u = model(z, r_batch, t_batch)
        z = z - (t_batch - r_batch) * u

    if was_training:
        model.train()

    return z
