from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dataloader import ToyDiffusionDataset

DATASETS = ["swiss_roll", "gaussians", "circles"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize toy datasets (D=2 and D=32 projected)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="report/figures/data_visualization")
    parser.add_argument("--num_points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def save_individual_figures(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    for name in DATASETS:
        ds_2d = ToyDiffusionDataset(name=name, dim=2, data_dir=args.data_dir)
        ds_32d = ToyDiffusionDataset(name=name, dim=32, data_dir=args.data_dir)

        x_2d = maybe_subsample(ds_2d.data.numpy(), args.num_points, args.seed)
        x_32d_raw = maybe_subsample(ds_32d.data.numpy(), args.num_points, args.seed)
        x_32d_back = ds_32d.to_2d(x_32d_raw)

        for points, suffix, title in [
            (x_2d, "D2_original", f"{name}: original D=2"),
            (x_32d_back, "D32_back_projected", f"{name}: D=32 projected to 2D"),
        ]:
            fig, ax = plt.subplots(figsize=(4, 4))
            scatter_ax(ax, points, title)
            plt.savefig(out_dir / f"{name}_{suffix}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)


def save_combined_figure(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    fig, axes = plt.subplots(3, 2, figsize=(8, 11))

    for row, name in enumerate(DATASETS):
        ds_2d = ToyDiffusionDataset(name=name, dim=2, data_dir=args.data_dir)
        ds_32d = ToyDiffusionDataset(name=name, dim=32, data_dir=args.data_dir)

        x_2d = maybe_subsample(ds_2d.data.numpy(), args.num_points, args.seed)
        x_32d_raw = maybe_subsample(ds_32d.data.numpy(), args.num_points, args.seed)
        x_32d_back = ds_32d.to_2d(x_32d_raw)

        scatter_ax(axes[row, 0], x_2d, f"{name}: original D=2")
        scatter_ax(axes[row, 1], x_32d_back, f"{name}: D=32 projected to 2D")

    fig.suptitle("Toy Datasets: Original D=2 vs D=32 Back-Projected", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "all_datasets_original_vs_back_projected.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    save_individual_figures(args)
    save_combined_figure(args)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
