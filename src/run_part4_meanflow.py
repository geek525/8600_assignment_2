from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.dataloader import ToyDiffusionDataset
from src.meanflow import sample_meanflow
from src.model import MeanFlowMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 4.2 MeanFlow batch runner")

    parser.add_argument("--train_root", type=str,
                        default="/content/drive/MyDrive/Flowmatching/part4_meanflow_train_fix3_sampler")
    parser.add_argument("--sample_root", type=str,
                        default="/content/drive/MyDrive/Flowmatching/part4_meanflow_samples_fix3_sampler")
    parser.add_argument("--fig_root", type=str,
                        default="/content/drive/MyDrive/Flowmatching/report_figures/part4_meanflow_fix3_sampler")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--datasets", type=str, default="swiss_roll,gaussians,circles")
    parser.add_argument("--dim", type=int, default=32)

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

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=5)
    parser.add_argument("--time_embed_dim", type=int, default=128)
    parser.add_argument("--horizon_embed_dim", type=int, default=128)

    parser.add_argument("--sample_steps_list", type=str, default="1,2,5")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)

    return parser.parse_args()


def parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_steps_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def is_train_complete(train_dir: Path) -> bool:
    return (train_dir / "checkpoints" / "final.pt").exists()


def is_sample_complete(sample_dir: Path) -> bool:
    return (
        (sample_dir / "config.json").exists()
        and (sample_dir / "samples" / "final_samples_2d.npy").exists()
    )


def build_train_cmd(args: argparse.Namespace, dataset: str, out_dir: Path) -> list[str]:
    return [
        sys.executable, "-m", "src.train_meanflow",
        "--dataset", dataset,
        "--dim", str(args.dim),
        "--data_dir", args.data_dir,
        "--out_dir", str(out_dir),
        "--steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num_workers", str(args.num_workers),
        "--time_eps", str(args.time_eps),
        "--flow_matching_ratio", str(args.flow_matching_ratio),
        "--loss_p", str(args.loss_p),
        "--loss_c", str(args.loss_c),
        "--hidden_dim", str(args.hidden_dim),
        "--num_hidden_layers", str(args.num_hidden_layers),
        "--time_embed_dim", str(args.time_embed_dim),
        "--horizon_embed_dim", str(args.horizon_embed_dim),
        "--device", args.device,
        "--seed", str(args.seed),
        "--sample_steps", "1",
        "--num_samples", str(args.num_samples),
        "--time_sampler", args.time_sampler,
        "--p_mean", str(args.p_mean),
        "--p_std", str(args.p_std),
    ]


def do_sample_and_save(
    train_dir: Path,
    sample_out_dir: Path,
    dataset: str,
    sample_steps: int,
    num_samples: int,
    data_dir: str,
    device: torch.device,
    seed: int,
) -> None:
    config = json.loads((train_dir / "config.json").read_text())
    dim = int(config["dim"])

    model = MeanFlowMLP(
        data_dim=dim,
        time_embed_dim=int(config.get("time_embed_dim", 128)),
        horizon_embed_dim=int(config.get("horizon_embed_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_hidden_layers=int(config.get("num_hidden_layers", 5)),
    )
    ckpt = torch.load(train_dir / "checkpoints" / "final.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    samples = sample_meanflow(
        model=model,
        num_samples=num_samples,
        dim=dim,
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

    # loss=null keeps it out of the plot title; suffix carries all identity
    out_config = {
        "dataset": dataset,
        "dim": dim,
        "prediction": "x",
        "loss": None,
        "model_type": "meanflow_rt",
        "conditioning": "r_t",
        "source_train_dir": str(train_dir),
        "sample_steps": sample_steps,
        "num_samples": num_samples,
        "part": "part4_meanflow_rt",
    }
    (sample_out_dir / "config.json").write_text(json.dumps(out_config, indent=2))
    print(f"  Saved samples to {sample_out_dir}")


def do_plot(
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
        "--title_suffix", f"MeanFlow | sample_steps={sample_steps}",
        "--filename_suffix", f"meanflow_steps-{sample_steps}",
    ]
    if dry_run:
        print("DRY RUN plot:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def run(args: argparse.Namespace) -> None:
    datasets = parse_list(args.datasets)
    steps_list = parse_steps_list(args.sample_steps_list)
    device = resolve_device(args.device)

    train_root = Path(args.train_root)
    sample_root = Path(args.sample_root)
    fig_root = Path(args.fig_root)

    print("Part 4.2 MeanFlow Runner")
    print(f"Datasets:    {datasets}")
    print(f"Step counts: {steps_list}")
    print(f"Device:      {device}")
    print(f"Train root:  {train_root}")
    print(f"Sample root: {sample_root}")
    print(f"Figure root: {fig_root}")

    for dataset in datasets:
        train_name = f"{dataset}_D{args.dim}_meanflow"
        train_dir = train_root / train_name

        print(f"\n{'#' * 60}")
        print(f"Dataset: {dataset}  ->  {train_dir}")
        print(f"{'#' * 60}")

        if is_train_complete(train_dir) and args.skip_existing:
            print("Skipping training (already complete)")
        else:
            cmd = build_train_cmd(args, dataset, train_dir)
            if args.dry_run:
                print("DRY RUN train:", " ".join(cmd))
            else:
                print(f"Training...")
                subprocess.run(cmd, check=True)

        for sample_steps in steps_list:
            sample_name = f"{train_name}_steps-{sample_steps}"
            sample_out_dir = sample_root / sample_name

            print(f"\n  steps={sample_steps}  ->  {sample_out_dir}")

            if args.skip_existing and is_sample_complete(sample_out_dir):
                print("  Skipping sampling (already complete)")
            elif args.dry_run:
                print(f"  DRY RUN: would sample {args.num_samples} points "
                      f"with {sample_steps} MeanFlow steps")
            else:
                do_sample_and_save(
                    train_dir=train_dir,
                    sample_out_dir=sample_out_dir,
                    dataset=dataset,
                    sample_steps=sample_steps,
                    num_samples=args.num_samples,
                    data_dir=args.data_dir,
                    device=device,
                    seed=args.seed,
                )

            do_plot(
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
