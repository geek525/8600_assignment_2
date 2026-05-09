from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

PredictionType = Literal["x", "v"]
LossType = Literal["x", "v"]


def sample_time(
    batch_size: int,
    device: torch.device | str,
    eps: float = 1e-5,
) -> torch.Tensor:
    return torch.rand(batch_size, 1, device=device) * (1 - 2 * eps) + eps


def sample_noise(x: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(x)


def make_noisy_sample(
    x: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    if t.ndim == 1:
        t = t[:, None]
    return (1 - t) * x + t * eps


def velocity_target(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return eps - x


def output_to_x(
    model_output: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    prediction_type: PredictionType,
) -> torch.Tensor:
    if t.ndim == 1:
        t = t[:, None]
    if prediction_type == "x":
        return model_output
    elif prediction_type == "v":
        return z_t - t * model_output
    else:
        raise ValueError(f"prediction_type must be 'x' or 'v', got '{prediction_type}'")


def output_to_v(
    model_output: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    prediction_type: PredictionType,
    eps_clip: float = 1e-5,
) -> torch.Tensor:
    if t.ndim == 1:
        t = t[:, None]
    if prediction_type == "v":
        return model_output
    elif prediction_type == "x":
        t_safe = t.clamp(min=eps_clip)
        return (z_t - model_output) / t_safe
    else:
        raise ValueError(f"prediction_type must be 'x' or 'v', got '{prediction_type}'")


def compute_targets(
    x: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return x, velocity_target(x, eps)


def compute_flow_loss(
    model_output: torch.Tensor,
    x: torch.Tensor,
    eps: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    prediction_type: PredictionType,
    loss_type: LossType,
) -> torch.Tensor:
    if loss_type == "x":
        x_pred = output_to_x(model_output, z_t, t, prediction_type)
        return F.mse_loss(x_pred, x)
    elif loss_type == "v":
        v_pred = output_to_v(model_output, z_t, t, prediction_type)
        v_target = velocity_target(x, eps)
        return F.mse_loss(v_pred, v_target)
    else:
        raise ValueError(f"loss_type must be 'x' or 'v', got '{loss_type}'")


def make_training_batch(
    x: torch.Tensor,
    eps_time: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = torch.randn_like(x)
    t = sample_time(x.shape[0], x.device, eps=eps_time)
    z_t = make_noisy_sample(x, eps, t)
    v_tgt = velocity_target(x, eps)
    return z_t, t, eps, v_tgt


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    num_samples: int,
    data_dim: int,
    prediction_type: PredictionType,
    num_steps: int = 50,
    device: torch.device | str = "cpu",
    eps_clip: float = 1e-5,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    z = torch.randn(num_samples, data_dim, device=device)
    times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_cur = times[i].item()
        t_next = times[i + 1].item()
        dt = t_next - t_cur

        t_batch = torch.full((num_samples, 1), t_cur, device=device)
        model_output = model(z, t_batch)
        v_pred = output_to_v(
            model_output=model_output,
            z_t=z,
            t=t_batch,
            prediction_type=prediction_type,
            eps_clip=eps_clip,
        )
        z = z + dt * v_pred

    if was_training:
        model.train()

    return z
