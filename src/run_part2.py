from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ALL_DATASETS = ["swiss_roll", "gaussians", "circles"]
ALL_DIMS = [2, 8, 32]
ALL_PREDICTIONS = ["x", "v"]
ALL_LOSSES = ["x", "v"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for Part 2 Flow Matching experiments")

    parser.add_argument("--root_out_dir", type=str, default="outputs/part2")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=str, default="auto")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--continue_on_error", action="store_true", default=False)

    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--dims", type=str, default="all")
    parser.add_argument("--predictions", type=str, default="all")
    parser.add_argument("--losses", type=str, default="all")

    return parser.parse_args()


def parse_string_filter(value: str, valid_values: list[str], name: str) -> list[str]:
    if value == "all":
        return valid_values
    selected = [v.strip() for v in value.split(",")]
    for s in selected:
        if s not in valid_values:
            raise ValueError(f"Invalid {name} '{s}'. Choose from {valid_values} or 'all'")
    return selected


def parse_dim_filter(value: str) -> list[int]:
    if value == "all":
        return ALL_DIMS
    selected = [int(v.strip()) for v in value.split(",")]
    for d in selected:
        if d not in ALL_DIMS:
            raise ValueError(f"Invalid dim '{d}'. Choose from {ALL_DIMS} or 'all'")
    return selected


def is_experiment_complete(out_dir: Path) -> bool:
    return (
        (out_dir / "checkpoints" / "final.pt").exists()
        and (out_dir / "samples" / "final_samples_2d.npy").exists()
    )


def build_out_dir(root: Path, dataset: str, dim: int, prediction: str, loss: str) -> Path:
    return root / f"{dataset}_D{dim}_pred-{prediction}_loss-{loss}"


def build_train_command(
    args: argparse.Namespace,
    dataset: str,
    dim: int,
    prediction: str,
    loss: str,
    out_dir: Path,
    seed: int,
) -> list[str]:
    return [
        sys.executable, "-m", "src.train",
        "--dataset", dataset,
        "--dim", str(dim),
        "--prediction", prediction,
        "--loss", loss,
        "--steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num_workers", str(args.num_workers),
        "--sample_steps", str(args.sample_steps),
        "--num_samples", str(args.num_samples),
        "--data_dir", str(args.data_dir),
        "--seed", str(seed),
        "--out_dir", str(out_dir),
    ]


def run_part2(args: argparse.Namespace) -> None:
    datasets = parse_string_filter(args.datasets, ALL_DATASETS, "dataset")
    dims = parse_dim_filter(args.dims)
    predictions = parse_string_filter(args.predictions, ALL_PREDICTIONS, "prediction")
    losses = parse_string_filter(args.losses, ALL_LOSSES, "loss")

    root = Path(args.root_out_dir)

    configs = [
        (dataset, dim, prediction, loss)
        for dataset in datasets
        for dim in dims
        for prediction in predictions
        for loss in losses
    ]
    total = len(configs)

    print(f"Part 2 batch runner")
    print(f"Datasets: {datasets}")
    print(f"Dims: {dims}")
    print(f"Predictions: {predictions}")
    print(f"Losses: {losses}")
    print(f"Total experiments: {total}")
    print(f"Root output directory: {root}")

    failed: list[str] = []

    for idx, (dataset, dim, prediction, loss) in enumerate(configs, start=1):
        out_dir = build_out_dir(root, dataset, dim, prediction, loss)
        label = f"[{idx}/{total}]"

        if args.skip_existing and is_experiment_complete(out_dir):
            print(f"{label} Skipping existing experiment: {out_dir}")
            continue

        print(f"\n{'=' * 60}")
        print(f"{label} Training dataset={dataset}, D={dim}, pred={prediction}, loss={loss}")
        print(f"Output: {out_dir}")
        print(f"{'=' * 60}")

        experiment_seed = args.seed + idx - 1
        cmd = build_train_command(args, dataset, dim, prediction, loss, out_dir, experiment_seed)

        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            if args.continue_on_error:
                print(f"ERROR: experiment failed (exit code {e.returncode}). Continuing.")
                failed.append(str(out_dir))
            else:
                raise

    if failed:
        print(f"\n{len(failed)} experiment(s) failed:")
        for f in failed:
            print(f"  {f}")


def main() -> None:
    args = parse_args()
    run_part2(args)


if __name__ == "__main__":
    main()
