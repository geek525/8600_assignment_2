from __future__ import annotations

import torch
from torch import nn

from src.flow import sample_time


def sample_meanflow_times(
    batch_size: int,
    device: torch.device | str,
    eps: float = 0.005,
    flow_matching_ratio: float = 0.5,
    time_sampler: str = "logit_normal_pair",
    p_mean: float = -0.4,
    p_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample r and t for MeanFlow training. Both shapes [B, 1].

    time_sampler="uniform_conditional":
        t ~ Uniform(eps, 1-eps), r ~ Uniform(0, t)
    time_sampler="logit_normal_pair":
        (r, t) = sorted sigmoid of two independent logit-normal draws

    After sampling, flow_matching_ratio fraction of samples get r = t (h=0),
    which reduces to standard Flow Matching for those samples.
    """
    if time_sampler == "logit_normal_pair":
        y = torch.randn(2, batch_size, 1, device=device)
        times = torch.sigmoid(y * p_std + p_mean)
        times = times.clamp(eps, 1.0 - eps)
        sorted_times = torch.sort(times, dim=0).values
        r = sorted_times[0]   # smaller value → left endpoint
        t = sorted_times[1]   # larger value  → current time
    else:  # "uniform_conditional"
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
    time_sampler: str = "logit_normal_pair",
    p_mean: float = -0.4,
    p_std: float = 1.0,
    debug_time_intervals: bool = False,
) -> dict[str, torch.Tensor]:
    """Build training tensors for one MeanFlow batch."""
    B, device = x.shape[0], x.device

    noise = torch.randn_like(x)
    r, t = sample_meanflow_times(
        B, device,
        eps=time_eps,
        flow_matching_ratio=flow_matching_ratio,
        time_sampler=time_sampler,
        p_mean=p_mean,
        p_std=p_std,
    )

    if debug_time_intervals:
        h = t - r
        frac_zero = (h.abs() < 1e-6).float().mean().item()
        print(
            f"[debug intervals] mean(t)={t.mean():.3f}  mean(r)={r.mean():.3f}  "
            f"mean(h)={h.mean():.3f}  "
            f"h in [{h.min():.3f}, {h.max():.3f}]  "
            f"frac(h≈0)={frac_zero:.2f}  (flow_matching_ratio={flow_matching_ratio})"
        )

    z_t = (1.0 - t) * x + t * noise
    v = noise - x

    return {"x": x, "eps": noise, "z_t": z_t, "v": v, "r": r, "t": t}


def meanflow_loss(
    model: nn.Module,
    x: torch.Tensor,
    time_eps: float = 0.005,
    flow_matching_ratio: float = 0.5,
    loss_p: float = 1.0,
    loss_c: float = 1e-3,
    time_sampler: str = "logit_normal_pair",
    p_mean: float = -0.4,
    p_std: float = 1.0,
    debug_time_intervals: bool = False,
) -> torch.Tensor:
    """MeanFlow training loss — paper Algorithm 1.

    Model interface: model(z, r, t)
    JVP tangents: (dz/dt, dr/dt, dt/dt) = (v, 0, 1)
      r is held fixed during the JVP, so dr/dt = 0.

    Target: u_tgt = v - (t - r) * dudt
    When r = t (h=0): u_tgt = v  →  reduces to standard Flow Matching.
    """
    batch = make_meanflow_batch(
        x,
        time_eps=time_eps,
        flow_matching_ratio=flow_matching_ratio,
        time_sampler=time_sampler,
        p_mean=p_mean,
        p_std=p_std,
        debug_time_intervals=debug_time_intervals,
    )
    z, r, t, v = batch["z_t"], batch["r"], batch["t"], batch["v"]

    def fn(z_in: torch.Tensor, r_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        return model(z_in, r_in, t_in)

    u, dudt = torch.func.jvp(
        fn,
        (z, r, t),
        (v, torch.zeros_like(r), torch.ones_like(t)),
    )

    assert u.shape == z.shape, f"u must be a vector field [B,D]: got {u.shape}, expected {z.shape}"
    assert dudt.shape == z.shape, f"dudt must be a vector field [B,D]: got {dudt.shape}, expected {z.shape}"

    u_tgt = (v - (t - r) * dudt).detach()

    assert u_tgt.shape == z.shape, f"u_tgt must be a vector field [B,D]: got {u_tgt.shape}, expected {z.shape}"

    error = u - u_tgt

    # FIX 2: .mean(dim=1) 而不是 .sum(dim=1)
    # sum 会随数据维度 D 线性增大，导致 adaptive weight 趋近 0，梯度信号消失。
    # mean 保持量纲稳定，与 loss_c=1e-3 的默认值匹配。
    per_sample_l2 = error.pow(2).mean(dim=1)

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
    debug_time_intervals: bool = False,
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

        if debug_time_intervals:
            print(f"[debug sampling] t={t_i:.4f} -> r={r_i:.4f}, h={t_i - r_i:.4f}")

        t_batch = torch.full((num_samples, 1), t_i, device=device)
        r_batch = torch.full((num_samples, 1), r_i, device=device)

        u = model(z, r_batch, t_batch)
        assert u.shape == z.shape, f"sampling u must be [B,D]: got {u.shape}, expected {z.shape}"
        z = z - (t_batch - r_batch) * u

    if was_training:
        model.train()

    return z
