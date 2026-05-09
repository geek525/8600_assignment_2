from __future__ import annotations

import argparse
import csv
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataloader import ToyDiffusionDataset
from src.flow import (
    compute_flow_loss,
    euler_sample,
    make_noisy_sample,
    sample_noise,
    sample_time,
)
from src.model import FlowMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a standard Flow Matching model")

    parser.add_argument("--dataset", type=str, default="swiss_roll",
                        choices=["swiss_roll", "gaussians", "circles"])
    parser.add_argument("--dim", type=int, default=2, choices=[2, 8, 32])
    parser.add_argument("--prediction", type=str, default="v", choices=["x", "v"])
    parser.add_argument("--loss", type=str, default="v", choices=["x", "v"])
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--sample_interval", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--time_eps", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=str, default="auto")

    args = parser.parse_args()

    if args.num_workers != "auto":
        try:
            val = int(args.num_workers)
            if val < 0:
                raise ValueError
        except ValueError:
            parser.error("--num_workers must be a non-negative integer or 'auto'")

    if args.out_dir is None:
        args.out_dir = (
            f"outputs/{args.dataset}_D{args.dim}"
            f"_pred-{args.prediction}_loss-{args.loss}"
        )

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_num_workers(num_workers_arg: str) -> int:
    if num_workers_arg != "auto":
        value = int(num_workers_arg)
        if value < 0:
            raise ValueError("--num_workers must be non-negative or 'auto'")
        return value

    system = platform.system().lower()
    if system == "windows":
        return 0
    if system == "linux":
        return 4
    if system == "darwin":
        return 0
    return 0


def save_checkpoint(path: Path, model: FlowMLP, optimizer: torch.optim.Optimizer,
                    step: int, config: dict, resolved_config: dict, loss: float) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config,
        "resolved_config": resolved_config,
        "loss": loss,
    }, path)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    num_workers = resolve_num_workers(args.num_workers)
    print(f"Using num_workers={num_workers}")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    samples_dir = out_dir / "samples"
    logs_dir = out_dir / "logs"
    for d in [ckpt_dir, samples_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    resolved_config = {"device": str(device), "num_workers": num_workers}
    with open(out_dir / "config.json", "w") as f:
        json.dump({**config, **resolved_config}, f, indent=2)

    dataset = ToyDiffusionDataset(
        name=args.dataset,
        dim=args.dim,
        data_dir=args.data_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    model = FlowMLP(data_dim=args.dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_rows: list[tuple[int, float]] = []
    step = 0
    loss = torch.tensor(0.0)

    while step < args.steps:
        for x in loader:
            if step >= args.steps:
                break

            step += 1
            x = x.to(device).float()

            t = sample_time(x.shape[0], device=x.device, eps=args.time_eps)
            eps = sample_noise(x)
            z_t = make_noisy_sample(x, eps, t)
            model_output = model(z_t, t)
            loss = compute_flow_loss(
                model_output=model_output,
                x=x,
                eps=eps,
                z_t=z_t,
                t=t,
                prediction_type=args.prediction,
                loss_type=args.loss,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            log_rows.append((step, loss_val))

            if step % args.log_interval == 0:
                print(
                    f"step {step}/{args.steps} | loss {loss_val:.6f} | "
                    f"dataset {args.dataset} | D={args.dim} | "
                    f"pred={args.prediction} | loss_type={args.loss}"
                )

            if step % args.save_interval == 0:
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, step, config, resolved_config, loss_val)

    save_checkpoint(ckpt_dir / "final.pt", model, optimizer, step, config, resolved_config, float(loss.item()))

    with open(logs_dir / "train_loss.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        writer.writerows(log_rows)

    print("Generating final samples...")
    samples = euler_sample(
        model=model,
        num_samples=args.num_samples,
        data_dim=args.dim,
        prediction_type=args.prediction,
        num_steps=args.sample_steps,
        device=device,
    )
    samples_np = samples.detach().cpu().numpy()
    np.save(samples_dir / "final_samples.npy", samples_np)

    samples_2d = dataset.to_2d(samples_np)
    np.save(samples_dir / "final_samples_2d.npy", samples_2d)

    print(f"Done. Results saved to {out_dir}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
