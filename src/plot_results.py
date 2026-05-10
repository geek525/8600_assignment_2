from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dataloader import ToyDiffusionDataset

_DEFAULT_LR = 1e-3
_DEFAULT_HIDDEN_DIM = 256
_DEFAULT_STEPS = 25000
_DEFAULT_TIME_EPS = 0.005


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


def format_float_for_filename(value: float | str) -> str:
    v = float(value)
    s = f"{v:.2e}"
    # Remove trailing zeros in coefficient: "1.00e-03" -> "1e-03"
    s = re.sub(r"(\d)\.?0+e", r"\1e", s)
    # Remove leading zeros in exponent: "1e-03" -> "1e-3"
    s = re.sub(r"e([+-])0*(\d+)", lambda m: f"e{m.group(1)}{m.group(2)}", s)
    return s.replace("-", "m").replace("+", "").replace(".", "p")


def build_labels(
    config: dict,
    dataset_name: str,
    dim: int,
    prediction: str | None,
    loss: str | None,
    sample_steps: int | None = None,
    dir_name: str = "",
) -> tuple[str, str]:
    """Return (plot_title, filename_stem) based on config metadata."""
    title_parts = [dataset_name, f"D={dim}"]
    stem_parts = [dataset_name, f"D{dim}"]

    if prediction:
        title_parts.append(f"pred={prediction}")
        stem_parts.append(f"pred-{prediction}")
    if loss:
        title_parts.append(f"loss={loss}")
        stem_parts.append(f"loss-{loss}")

    time_schedule = config.get("time_schedule")
    if time_schedule is not None and (time_schedule != "uniform" or "schedule" in dir_name):
        title_parts.append(f"schedule={time_schedule}")
        stem_parts.append(f"schedule-{time_schedule}")

    lr = config.get("lr")
    if lr is not None and ("lr" in dir_name or float(lr) != _DEFAULT_LR):
        title_parts.append(f"lr={lr}")
        stem_parts.append(f"lr-{format_float_for_filename(lr)}")

    hidden_dim = config.get("hidden_dim")
    if hidden_dim is not None and ("hidden" in dir_name or int(hidden_dim) != _DEFAULT_HIDDEN_DIM):
        title_parts.append(f"hidden={hidden_dim}")
        stem_parts.append(f"hidden-{hidden_dim}")

    steps = config.get("steps")
    if steps is not None and int(steps) != _DEFAULT_STEPS:
        title_parts.append(f"train_steps={steps}")
        stem_parts.append(f"steps-{steps}")

    time_eps = config.get("time_eps")
    if time_eps is not None and float(time_eps) != _DEFAULT_TIME_EPS:
        title_parts.append(f"time_eps={time_eps}")
        stem_parts.append(f"timeeps-{format_float_for_filename(time_eps)}")

    if sample_steps is not None:
        title_parts.append(f"steps={sample_steps}")
        stem_parts.append(f"steps-{sample_steps:03d}")

    return " | ".join(title_parts), "_".join(stem_parts) + "_generated_vs_ground_truth"


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

    title, stem = build_labels(
        config=config,
        dataset_name=dataset_name,
        dim=dim,
        prediction=prediction,
        loss=loss,
        dir_name=exp_dir.name,
    )

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

    title, stem = build_labels(
        config=config,
        dataset_name=dataset_name,
        dim=dim,
        prediction=prediction,
        loss=loss,
        sample_steps=sample_steps,
        dir_name=sample_dir.name,
    )

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
