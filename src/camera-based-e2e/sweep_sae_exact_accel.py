from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


DEFAULT_TARGETS = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0]


def format_target_slug(target: float) -> str:
    if float(target).is_integer():
        return str(int(target)).replace("-", "neg")
    return str(target).replace("-", "neg").replace(".", "p")


def read_summary_means(summary_csv: Path) -> dict[str, float]:
    with summary_csv.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows found in {summary_csv}")

    def mean_of(field: str) -> float:
        values = [float(row[field]) for row in rows]
        return sum(values) / len(values)

    return {
        "target_accel": mean_of("target_accel_mps2"),
        "edited_accel_mean": mean_of("edited_accel_mean"),
        "max_abs_accel_error": mean_of("max_abs_accel_error"),
        "mean_abs_accel_error": mean_of("mean_abs_accel_error"),
        "mean_edited_accel_std": mean_of("edited_accel_std"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an exact-acceleration sweep and aggregate control-fidelity metrics."
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=DEFAULT_TARGETS,
        help="target acceleration values in m/s^2",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/gilbreth/chang899/waymo_data/waymo_open_dataset_end_to_end_camera_v_1_0_0",
    )
    parser.add_argument("--index_file", type=str, default="index_val.pkl")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/camera-e2e-epoch=04-val_loss=2.90.ckpt",
    )
    parser.add_argument(
        "--sae_ckpt",
        type=str,
        default="/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/output/sae_control_pred.pt",
    )
    parser.add_argument("--n_items", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--accel_eps", type=float, default=1e-4)
    parser.add_argument(
        "--max_plots",
        type=int,
        default=0,
        help="per-target plots; default 0 for sweep runs",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/output/intervention_exact_accel_sweep",
    )
    parser.add_argument(
        "--aggregate_csv",
        type=str,
        default=None,
        help="optional path for the final sweep CSV; defaults to <output_root>/sweep_metrics.csv",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_csv = Path(args.aggregate_csv) if args.aggregate_csv else output_root / "sweep_metrics.csv"

    aggregate_rows = []
    for target in args.targets:
        target_slug = format_target_slug(target)
        target_output_dir = output_root / f"target_{target_slug}"
        cmd = [
            sys.executable,
            "intervene_sae_exact_accel.py",
            "--data_dir",
            args.data_dir,
            "--index_file",
            args.index_file,
            "--model_ckpt",
            args.model_ckpt,
            "--sae_ckpt",
            args.sae_ckpt,
            "--target_accel_mps2",
            str(target),
            "--n_items",
            str(args.n_items),
            "--batch_size",
            str(args.batch_size),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--dt",
            str(args.dt),
            "--accel_eps",
            str(args.accel_eps),
            "--max_plots",
            str(args.max_plots),
            "--output_dir",
            str(target_output_dir),
        ]

        print(f"\n=== Running target_accel_mps2={target} ===")
        subprocess.run(cmd, cwd=script_dir, check=True)

        summary_csv = target_output_dir / "sample_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Expected per-target summary CSV at {summary_csv}")
        aggregate_rows.append(read_summary_means(summary_csv))

    aggregate_rows.sort(key=lambda row: row["target_accel"])
    with aggregate_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_accel",
                "edited_accel_mean",
                "max_abs_accel_error",
                "mean_abs_accel_error",
                "mean_edited_accel_std",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"\nSaved sweep metrics to {aggregate_csv}")
