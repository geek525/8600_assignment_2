from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.dataloader import ToyDiffusionDataset
from src.flow import euler_sample
from src.model import FlowMLP

_PRED = "x"
_LOSS = "x"
_DIM = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part 4.1: sampling efficiency evaluation for trained Part 2 models"
    )
    parser.add_argument("--part2_root", type=str,
                        default="/content/drive/MyDrive/Flowmatching/part2_eps0005")
    parser.add_argument("--out_root", type=str,
                        default="/content/drive/MyDrive/Flowmatching/part4_sampling_efficiency")
    parser.add_argument("--fig_root", type=str,
                        default="/content/drive/MyDrive/Flowmatching/report_figures/part4_sampling_efficiency")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--datasets", type=str, default="swiss_roll,gaussians,circles",
                        help="Comma-separated dataset names")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--sample_steps_list", type=str, default="1,2,5,10,20,50,100,200",
                        help="Comma-separated Euler step counts to evaluate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true", default=False,
                        help="Skip sampling if output files already exist")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Print planned operations without executing")
    return parser.parse_args()


def parse_datasets(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_steps_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def is_sample_complete(sample_out_dir: Path) -> bool:
    return (
        (sample_out_dir / "config.json").exists()
        and (sample_out_dir / "samples" / "final_samples_2d.npy").exists()
    )


def sample_and_save(
    part2_exp_dir: Path,
    sample_out_dir: Path,
    dataset: str,
    sample_steps: int,
    num_samples: int,
    data_dir: str,
    device: torch.device,
    seed: int,
) -> None:
    config = json.loads((part2_exp_dir / "config.json").read_text())
    dim = int(config["dim"])
    prediction = config["prediction"]
    loss = config.get("loss")

    model = FlowMLP(
        data_dim=dim,
        time_embed_dim=int(config.get("time_embed_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_hidden_layers=int(config.get("num_hidden_layers", 5)),
    )
    ckpt = torch.load(
        part2_exp_dir / "checkpoints" / "final.pt",
        map_location=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    samples = euler_sample(
        model=model,
        num_samples=num_samples,
        data_dim=dim,
        prediction_type=prediction,
        num_steps=sample_steps,
        device=device,
    )
    samples_np = samples.detach().cpu().numpy()

    ds = ToyDiffusionDataset(name=dataset, dim=dim, data_dir=data_dir)
    samples_2d = ds.to_2d(samples_np)

    samples_dir = sample_out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    np.save(samples_dir / "final_samples.npy", samples_np)
    np.save(samples_dir / "final_samples_2d.npy", samples_2d)

    out_config = {
        "dataset": dataset,
        "dim": dim,
        "prediction": prediction,
        "loss": loss,
        "source_experiment_dir": str(part2_exp_dir),
        "sample_steps": sample_steps,
        "num_samples": num_samples,
        "part": "part4_sampling_efficiency",
    }
    (sample_out_dir / "config.json").write_text(json.dumps(out_config, indent=2))
    print(f"  Saved samples to {sample_out_dir}")


def plot_result(
    sample_out_dir: Path,
    fig_root: Path,
    data_dir: str,
    sample_steps: int,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable, "-m", "src.plot_results",
        "--experiment_dir", str(sample_out_dir),
        "--data_dir", data_dir,
        "--out_dir", str(fig_root),
        "--title_suffix", f"sample_steps={sample_steps}",
        "--filename_suffix", f"steps-{sample_steps}",
    ]
    if dry_run:
        print("DRY RUN plot:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def run(args: argparse.Namespace) -> None:
    datasets = parse_datasets(args.datasets)
    steps_list = parse_steps_list(args.sample_steps_list)
    device = resolve_device(args.device)

    part2_root = Path(args.part2_root)
    out_root = Path(args.out_root)
    fig_root = Path(args.fig_root)

    total = len(datasets) * len(steps_list)
    print("Part 4.1 Sampling Efficiency")
    print(f"Datasets:    {datasets}")
    print(f"Step counts: {steps_list}")
    print(f"Total:       {total} experiments")
    print(f"Device:      {device}")
    print(f"Part 2 root: {part2_root}")
    print(f"Output root: {out_root}")
    print(f"Figure root: {fig_root}")

    count = 0
    for dataset in datasets:
        exp_name = f"{dataset}_D{_DIM}_pred-{_PRED}_loss-{_LOSS}"
        part2_exp_dir = part2_root / exp_name

        if not part2_exp_dir.exists():
            print(f"\nWARNING: Part 2 experiment not found, skipping: {part2_exp_dir}")
            count += len(steps_list)
            continue

        for sample_steps in steps_list:
            count += 1
            out_name = f"{exp_name}_steps-{sample_steps}"
            sample_out_dir = out_root / out_name
            label = f"[{count}/{total}]"

            print(f"\n{'=' * 60}")
            print(f"{label} dataset={dataset}  steps={sample_steps}")
            print(f"{'=' * 60}")

            # --- sampling ---
            if args.skip_existing and is_sample_complete(sample_out_dir):
                print("  Skipping sampling (already complete)")
            elif args.dry_run:
                print(f"  DRY RUN: would sample {args.num_samples} points "
                      f"with {sample_steps} Euler steps -> {sample_out_dir}")
            else:
                sample_and_save(
                    part2_exp_dir=part2_exp_dir,
                    sample_out_dir=sample_out_dir,
                    dataset=dataset,
                    sample_steps=sample_steps,
                    num_samples=args.num_samples,
                    data_dir=args.data_dir,
                    device=device,
                    seed=args.seed,
                )

            # --- plotting ---
            plot_result(
                sample_out_dir=sample_out_dir,
                fig_root=fig_root,
                data_dir=args.data_dir,
                sample_steps=sample_steps,
                dry_run=args.dry_run,
            )


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
