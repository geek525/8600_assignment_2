from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCHEDULES = ["uniform", "low_noise", "high_noise", "middle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part 3 time schedule batch runner (swiss_roll, D=32, pred=v, loss=v)"
    )
    parser.add_argument("--root_out_dir", type=str, default="outputs/part3_time_schedule")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=str, default="auto")
    parser.add_argument("--time_eps", type=float, default=0.005)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


def is_experiment_complete(out_dir: Path) -> bool:
    return (
        (out_dir / "checkpoints" / "final.pt").exists()
        and (out_dir / "samples" / "final_samples_2d.npy").exists()
    )


def build_command(args: argparse.Namespace, schedule: str, out_dir: Path, seed: int) -> list[str]:
    return [
        sys.executable, "-m", "src.train",
        "--dataset", "swiss_roll",
        "--dim", "32",
        "--prediction", "v",
        "--loss", "v",
        "--steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num_workers", str(args.num_workers),
        "--time_eps", str(args.time_eps),
        "--time_schedule", schedule,
        "--sample_steps", str(args.sample_steps),
        "--num_samples", str(args.num_samples),
        "--data_dir", str(args.data_dir),
        "--seed", str(seed),
        "--out_dir", str(out_dir),
    ]


def main() -> None:
    args = parse_args()
    root = Path(args.root_out_dir)
    total = len(SCHEDULES)

    print("Part 3 time schedule runner")
    print(f"Schedules: {SCHEDULES}")
    print(f"Total experiments: {total}")
    print(f"Root output directory: {root}")

    for idx, schedule in enumerate(SCHEDULES, start=1):
        out_dir = root / f"swiss_roll_D32_pred-v_loss-v_schedule-{schedule}"
        label = f"[{idx}/{total}]"

        if args.skip_existing and is_experiment_complete(out_dir):
            print(f"{label} Skipping existing experiment: {out_dir}")
            continue

        print(f"\n{'=' * 60}")
        print(f"{label} Training schedule={schedule}")
        print(f"Output: {out_dir}")
        print(f"{'=' * 60}")

        cmd = build_command(args, schedule, out_dir, args.seed + idx - 1)

        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
            continue

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
