from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dataloader import ToyDiffusionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot generated samples vs ground truth")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="train.py output directory (contains config.json and samples/)")
    parser.add_argument("--sample_dir", type=str, default=None,
                        help="sample.py output directory (contains sample_config.json and samples_2d.npy)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["swiss_roll", "gaussians", "circles"])
    parser.add_argument("--dim", type=int, default=None, choices=[2, 8, 32])
    parser.add_argument("--prediction", type=str, default=None, choices=["x", "v"])
    parser.add_argument("--loss", type=str, default=None, choices=["x", "v"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="report/figures/results")
    parser.add_argument("--num_points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.experiment_dir is None and args.sample_dir is None:
        parser.error("Provide --experiment_dir or --sample_dir")

    return args


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def maybe_subsample(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    if len(points) <= num_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=num_points, replace=False)
    return points[idx]


def scatter_ax(ax: plt.Axes, points: np.ndarray, title: str) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def load_real_2d(dataset_name: str, dim: int, data_dir: str) -> np.ndarray:
    dataset = ToyDiffusionDataset(name=dataset_name, dim=dim, data_dir=data_dir)
    return dataset.to_2d(dataset.data.numpy())


def plot_generated_vs_ground_truth(
    real_2d: np.ndarray,
    gen_2d: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    scatter_ax(axes[0], real_2d, "Ground truth")
    scatter_ax(axes[1], gen_2d, "Generated")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_from_experiment_dir(args: argparse.Namespace) -> None:
    exp_dir = Path(args.experiment_dir)
    config = load_json(exp_dir / "config.json")

    dataset_name = args.dataset or config["dataset"]
    dim = args.dim or int(config["dim"])
    prediction = args.prediction or config.get("prediction")
    loss = args.loss or config.get("loss")

    gen_2d = np.load(exp_dir / "samples" / "final_samples_2d.npy")
    real_2d = load_real_2d(dataset_name, dim, args.data_dir)

    real_2d = maybe_subsample(real_2d, args.num_points, args.seed)
    gen_2d = maybe_subsample(gen_2d, args.num_points, args.seed)

    time_schedule = config.get("time_schedule")

    parts = [dataset_name, f"D={dim}"]
    if prediction:
        parts.append(f"pred={prediction}")
    if loss:
        parts.append(f"loss={loss}")
    if time_schedule is not None:
        parts.append(f"schedule={time_schedule}")
    title = " | ".join(parts)

    stem_parts = [dataset_name, f"D{dim}"]
    if prediction:
        stem_parts.append(f"pred-{prediction}")
    if loss:
        stem_parts.append(f"loss-{loss}")
    if time_schedule is not None:
        stem_parts.append(f"schedule-{time_schedule}")
    stem = "_".join(stem_parts) + "_generated_vs_ground_truth"

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    plot_generated_vs_ground_truth(real_2d, gen_2d, title, out_dir / f"{stem}.png")


def plot_from_sample_dir(args: argparse.Namespace) -> None:
    sample_dir = Path(args.sample_dir)
    config = load_json(sample_dir / "sample_config.json")

    dataset_name = args.dataset or config["dataset"]
    dim = args.dim or int(config["dim"])
    prediction = args.prediction or config.get("prediction")
    loss = args.loss or config.get("loss")
    sample_steps = config.get("sample_steps")

    gen_2d = np.load(sample_dir / "samples_2d.npy")
    real_2d = load_real_2d(dataset_name, dim, args.data_dir)

    real_2d = maybe_subsample(real_2d, args.num_points, args.seed)
    gen_2d = maybe_subsample(gen_2d, args.num_points, args.seed)

    time_schedule = config.get("time_schedule")

    parts = [dataset_name, f"D={dim}"]
    if prediction:
        parts.append(f"pred={prediction}")
    if loss:
        parts.append(f"loss={loss}")
    if time_schedule is not None:
        parts.append(f"schedule={time_schedule}")
    if sample_steps is not None:
        parts.append(f"steps={sample_steps}")
    title = " | ".join(parts)

    stem_parts = [dataset_name, f"D{dim}"]
    if prediction:
        stem_parts.append(f"pred-{prediction}")
    if loss:
        stem_parts.append(f"loss-{loss}")
    if time_schedule is not None:
        stem_parts.append(f"schedule-{time_schedule}")
    if sample_steps is not None:
        stem_parts.append(f"steps-{sample_steps:03d}")
    stem = "_".join(stem_parts) + "_generated_vs_ground_truth"

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    plot_generated_vs_ground_truth(real_2d, gen_2d, title, out_dir / f"{stem}.png")


def main() -> None:
    args = parse_args()
    if args.experiment_dir is not None:
        plot_from_experiment_dir(args)
    else:
        plot_from_sample_dir(args)


if __name__ == "__main__":
    main()
