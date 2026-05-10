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
from src.meanflow import meanflow_loss, sample_meanflow
from src.model import MeanFlowMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MeanFlow model")

    parser.add_argument("--dataset", type=str, default="swiss_roll",
                        choices=["swiss_roll", "gaussians", "circles"])
    parser.add_argument("--dim", type=int, default=32, choices=[2, 8, 32])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=str, default="auto")
    parser.add_argument("--time_eps", type=float, default=0.005)
    parser.add_argument("--flow_matching_ratio", type=float, default=0.5)
    parser.add_argument("--loss_p", type=float, default=1.0)
    parser.add_argument("--loss_c", type=float, default=1e-3)
    parser.add_argument("--time_sampler", type=str, default="logit_normal_pair",
                        choices=["uniform_conditional", "logit_normal_pair"])
    parser.add_argument("--p_mean", type=float, default=-0.4)
    parser.add_argument("--p_std", type=float, default=1.0)
    parser.add_argument("--debug_time_intervals", action="store_true", default=False)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=5)
    parser.add_argument("--time_embed_dim", type=int, default=128)
    parser.add_argument("--horizon_embed_dim", type=int, default=128)

    parser.add_argument("--sample_steps", type=int, default=1,
                        help="Euler steps for post-training sample preview")
    parser.add_argument("--num_samples", type=int, default=5000)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)

    args = parser.parse_args()

    if args.num_workers != "auto":
        try:
            val = int(args.num_workers)
            if val < 0:
                raise ValueError
        except ValueError:
            parser.error("--num_workers must be a non-negative integer or 'auto'")

    if args.out_dir is None:
        args.out_dir = f"outputs/{args.dataset}_D{args.dim}_meanflow"

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


def save_checkpoint(
    path: Path,
    model: MeanFlowMLP,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    resolved_config: dict,
    loss: float,
) -> None:
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
    print(f"time_eps: {args.time_eps}")
    print(f"flow_matching_ratio: {args.flow_matching_ratio}")

    num_workers = resolve_num_workers(args.num_workers)
    print(f"Using num_workers={num_workers}")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    samples_dir = out_dir / "samples"
    logs_dir = out_dir / "logs"
    for d in [ckpt_dir, samples_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config = {
        **vars(args),
        "model_type": "meanflow_rt",
        "conditioning": "t_h",
        "prediction": "x",
        "loss": "meanflow",
    }
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

    model = MeanFlowMLP(
        data_dim=args.dim,
        time_embed_dim=args.time_embed_dim,
        horizon_embed_dim=args.horizon_embed_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    )
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

            loss = meanflow_loss(
                model=model,
                x=x,
                time_eps=args.time_eps,
                flow_matching_ratio=args.flow_matching_ratio,
                loss_p=args.loss_p,
                loss_c=args.loss_c,
                time_sampler=args.time_sampler,
                p_mean=args.p_mean,
                p_std=args.p_std,
                debug_time_intervals=(step == 1 and args.debug_time_intervals),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            log_rows.append((step, loss_val))

            if step % args.log_interval == 0:
                print(
                    f"step {step}/{args.steps} | loss {loss_val:.6f} | "
                    f"dataset {args.dataset} | D={args.dim} | MeanFlow"
                )

            if step % args.save_interval == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{step:06d}.pt",
                    model, optimizer, step, config, resolved_config, loss_val,
                )

    save_checkpoint(
        ckpt_dir / "final.pt",
        model, optimizer, step, config, resolved_config, float(loss.item()),
    )

    with open(logs_dir / "train_loss.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        writer.writerows(log_rows)

    print("Generating final samples...")
    samples = sample_meanflow(
        model=model,
        num_samples=args.num_samples,
        dim=args.dim,
        num_steps=args.sample_steps,
        device=device,
        debug_time_intervals=args.debug_time_intervals,
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