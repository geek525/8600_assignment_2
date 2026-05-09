from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.dataloader import ToyDiffusionDataset
from src.flow import euler_sample
from src.model import FlowMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained Flow Matching model")

    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint file (final.pt)")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--make_plot", action="store_true", default=False)

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        parser.error(f"Checkpoint not found: {args.ckpt}")
    if args.num_samples <= 0:
        parser.error("--num_samples must be > 0")
    if args.sample_steps <= 0:
        parser.error("--sample_steps must be > 0")
    if args.device not in ("auto", "cuda", "cpu"):
        parser.error("--device must be 'auto', 'cuda', or 'cpu'")

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


def infer_out_dir(ckpt_path: Path, sample_steps: int) -> Path:
    experiment_dir = ckpt_path.parent.parent
    return experiment_dir / "samples" / f"steps_{sample_steps:03d}"


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def run_sampling(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path, device)

    state_dict = ckpt["model_state_dict"]
    config = ckpt["config"]

    dataset_name = config["dataset"]
    dim = int(config["dim"])
    prediction_type = config["prediction"]
    loss_type = config.get("loss", None)
    step = ckpt.get("step", "?")

    print(
        f"Loaded checkpoint:\n"
        f"  dataset: {dataset_name}\n"
        f"  dim: {dim}\n"
        f"  prediction: {prediction_type}\n"
        f"  loss: {loss_type}\n"
        f"  training step: {step}\n"
        f"Sampling:\n"
        f"  num_samples: {args.num_samples}\n"
        f"  sample_steps: {args.sample_steps}"
    )

    model = FlowMLP(data_dim=dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    samples = euler_sample(
        model=model,
        num_samples=args.num_samples,
        data_dim=dim,
        prediction_type=prediction_type,
        num_steps=args.sample_steps,
        device=device,
    )

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = infer_out_dir(ckpt_path, args.sample_steps)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_np = samples.detach().cpu().numpy()
    np.save(out_dir / "samples.npy", samples_np)

    dataset = ToyDiffusionDataset(
        name=dataset_name,
        dim=dim,
        data_dir=args.data_dir,
    )
    samples_2d = dataset.to_2d(samples_np)
    np.save(out_dir / "samples_2d.npy", samples_2d)

    meta = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset_name,
        "dim": dim,
        "prediction": prediction_type,
        "loss": loss_type,
        "num_samples": args.num_samples,
        "sample_steps": args.sample_steps,
        "device": str(device),
        "seed": args.seed,
    }
    with open(out_dir / "sample_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{dataset_name} | D={dim} | pred={prediction_type} | steps={args.sample_steps}")
        plt.savefig(out_dir / "samples_2d.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {out_dir / 'samples_2d.png'}")

    print(f"Done. Results saved to {out_dir}")


def main() -> None:
    args = parse_args()
    run_sampling(args)


if __name__ == "__main__":
    main()
