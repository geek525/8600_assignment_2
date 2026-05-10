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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample t, r, h for MeanFlow training. All shapes [B, 1].

    h = t - r.
    For flow-matching samples (mask=True), h=0 and r=t,
    reducing to standard instantaneous Flow Matching.
    """
    t = sample_time(batch_size, device, eps=eps, schedule="uniform")

    u = torch.rand(batch_size, 1, device=device)
    r = u * t
    h = t - r

    # flow_matching_ratio fraction of samples use h=0 (standard FM)
    mask = torch.rand(batch_size, 1, device=device) < flow_matching_ratio
    r = torch.where(mask, t, r)
    h = torch.where(mask, torch.zeros_like(h), h)

    return t, r, h


def make_meanflow_batch(
    x: torch.Tensor,
    time_eps: float = 0.005,
    flow_matching_ratio: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Build training tensors for one MeanFlow batch."""
    B, device = x.shape[0], x.device

    eps = torch.randn_like(x)
    t, r, h = sample_meanflow_times(
        B, device, eps=time_eps, flow_matching_ratio=flow_matching_ratio
    )

    z_t = (1.0 - t) * x + t * eps
    v = eps - x

    return {"x": x, "eps": eps, "z_t": z_t, "v": v, "t": t, "r": r, "h": h}


def meanflow_loss(
    model: nn.Module,
    x: torch.Tensor,
    time_eps: float = 0.005,
    flow_matching_ratio: float = 0.5,
    loss_p: float = 1.0,
    loss_c: float = 1e-3,
) -> torch.Tensor:
    """Compute the MeanFlow training loss using torch.func.jvp.

    Model interface: model(z, t, h)
    JVP tangents for (z, t, h): (v, ones, ones)
      - dz/dt = v   (velocity along the linear path)
      - dt/dt = 1
      - dh/dt = 1   (since h = t - r and r is held fixed)

    Target: u_tgt = v - h * dudt
    Loss: MSE(u, stop_gradient(u_tgt)), optionally with adaptive weighting.

    When h=0, u_tgt = v, reducing to standard Flow Matching.
    """
    batch = make_meanflow_batch(x, time_eps=time_eps, flow_matching_ratio=flow_matching_ratio)
    z, t, h, v = batch["z_t"], batch["t"], batch["h"], batch["v"]

    def fn(z_in: torch.Tensor, t_in: torch.Tensor, h_in: torch.Tensor) -> torch.Tensor:
        return model(z_in, t_in, h_in)

    u, dudt = torch.func.jvp(
        fn,
        (z, t, h),
        (v, torch.ones_like(t), torch.ones_like(h)),
    )

    u_tgt = (v - h * dudt).detach()

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
    Update: z_r = z_t - h * u(z_t, t, h)
    """
    was_training = model.training
    model.eval()

    z = torch.randn(num_samples, dim, device=device)
    times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_i = float(times[i].item())
        h_i = float(times[i].item() - times[i + 1].item())

        t_batch = torch.full((num_samples, 1), t_i, device=device)
        h_batch = torch.full((num_samples, 1), h_i, device=device)

        u = model(z, t_batch, h_batch)
        z = z - h_batch * u

    if was_training:
        model.train()

    return z
