import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib_{os.getuid()}")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("FC_CACHEDIR", f"/tmp/fontconfig_{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["FC_CACHEDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "best_class":
                    parsed[key] = value
                elif key.startswith("selected_"):
                    parsed[key] = parse_bool(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return sorted(rows, key=lambda item: int(item["feature_idx"]))


def require_columns(rows: list[dict], columns: list[str], csv_path: Path) -> None:
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")


def top_abs_indices(values: list[float], top_k: int) -> set[int]:
    ranked = sorted(range(len(values)), key=lambda idx: abs(values[idx]), reverse=True)
    return set(ranked[: min(top_k, len(ranked))])


def symmetric_ylim(*series: list[float]) -> tuple[float, float]:
    max_abs = max((abs(value) for values in series for value in values), default=1.0)
    if max_abs == 0:
        max_abs = 1.0
    pad = max_abs * 0.08
    return -(max_abs + pad), max_abs + pad


def draw_delta_stems(
    ax: plt.Axes,
    feature_idx: list[int],
    delta: list[float],
    top_positions: set[int],
    title: str,
    ylabel: str,
) -> None:
    base_color = "#5B9BD5"
    highlight_color = "#ff2f5f"

    ax.vlines(feature_idx, [0.0], delta, color=base_color, linewidth=1.0, alpha=0.9)
    if top_positions:
        top_x = [feature_idx[pos] for pos in sorted(top_positions)]
        top_y = [delta[pos] for pos in sorted(top_positions)]
        ax.vlines(top_x, [0.0], top_y, color=highlight_color, linewidth=1.5, alpha=0.95)

    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.6)
    ax.margins(x=0.01)


def write_top_summary(
    output_csv: Path,
    rows: list[dict],
    feature_idx: list[int],
    delta_accel: list[float],
    delta_decel: list[float],
    top_accel: set[int],
    top_decel: set[int],
) -> None:
    fieldnames = [
        "feature_idx",
        "delta_decelerating_to_accelerating",
        "delta_accelerating_to_decelerating",
        "mean_accelerating",
        "mean_decelerating",
        "active_rate_accelerating",
        "active_rate_decelerating",
        "r_accelerating",
        "r_decelerating",
        "top_decelerating_to_accelerating",
        "top_accelerating_to_decelerating",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pos, row in enumerate(rows):
            writer.writerow(
                {
                    "feature_idx": feature_idx[pos],
                    "delta_decelerating_to_accelerating": delta_accel[pos],
                    "delta_accelerating_to_decelerating": delta_decel[pos],
                    "mean_accelerating": row["mean_accelerating"],
                    "mean_decelerating": row["mean_decelerating"],
                    "active_rate_accelerating": row["active_rate_accelerating"],
                    "active_rate_decelerating": row["active_rate_decelerating"],
                    "r_accelerating": row["r_accelerating"],
                    "r_decelerating": row["r_decelerating"],
                    "top_decelerating_to_accelerating": pos in top_accel,
                    "top_accelerating_to_decelerating": pos in top_decel,
                }
            )


def write_ranked_top_k(
    output_csv: Path,
    rows: list[dict],
    feature_idx: list[int],
    delta_accel: list[float],
    delta_decel: list[float],
    top_k: int,
) -> None:
    ranked_accel = sorted(range(len(rows)), key=lambda pos: abs(delta_accel[pos]), reverse=True)[:top_k]
    ranked_decel = sorted(range(len(rows)), key=lambda pos: abs(delta_decel[pos]), reverse=True)[:top_k]

    fieldnames = [
        "transition",
        "rank",
        "feature_idx",
        "delta",
        "mean_accelerating",
        "mean_decelerating",
        "active_rate_accelerating",
        "active_rate_decelerating",
        "r_accelerating",
        "r_decelerating",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for transition, ranked, deltas in (
            ("decelerating_to_accelerating", ranked_accel, delta_accel),
            ("accelerating_to_decelerating", ranked_decel, delta_decel),
        ):
            for rank, pos in enumerate(ranked, start=1):
                row = rows[pos]
                writer.writerow(
                    {
                        "transition": transition,
                        "rank": rank,
                        "feature_idx": feature_idx[pos],
                        "delta": deltas[pos],
                        "mean_accelerating": row["mean_accelerating"],
                        "mean_decelerating": row["mean_decelerating"],
                        "active_rate_accelerating": row["active_rate_accelerating"],
                        "active_rate_decelerating": row["active_rate_decelerating"],
                        "r_accelerating": row["r_accelerating"],
                        "r_decelerating": row["r_decelerating"],
                    }
                )


def plot_accel_decel(csv_path: Path, output_path: Path, top_k: int, title: str) -> tuple[Path, Path]:
    rows = load_rows(csv_path)
    require_columns(
        rows,
        [
            "feature_idx",
            "mean_accelerating",
            "mean_decelerating",
            "active_rate_accelerating",
            "active_rate_decelerating",
            "r_accelerating",
            "r_decelerating",
        ],
        csv_path,
    )

    feature_idx = [int(row["feature_idx"]) for row in rows]
    mean_accel = [row["mean_accelerating"] for row in rows]
    mean_decel = [row["mean_decelerating"] for row in rows]
    delta_accel = [accel - decel for accel, decel in zip(mean_accel, mean_decel)]
    delta_decel = [-value for value in delta_accel]
    top_accel = top_abs_indices(delta_accel, top_k)
    top_decel = top_abs_indices(delta_decel, top_k)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 6.8), sharex=True)
    fig.suptitle(
        f"{title}\nPositive = latent increases when switching to target speed-change class",
        fontsize=12,
    )

    draw_delta_stems(
        axes[0],
        feature_idx,
        delta_accel,
        top_accel,
        f"decelerating -> accelerating (top-{top_k} by |delta| in red)",
        "Mean delta z",
    )
    draw_delta_stems(
        axes[1],
        feature_idx,
        delta_decel,
        top_decel,
        f"accelerating -> decelerating (top-{top_k} by |delta| in red)",
        "Mean delta z",
    )
    ymin, ymax = symmetric_ylim(delta_accel, delta_decel)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    axes[-1].set_xlabel("Latent index")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_path.with_suffix(".top_latents.csv")
    write_top_summary(summary_path, rows, feature_idx, delta_accel, delta_decel, top_accel, top_decel)
    ranked_summary_path = output_path.with_suffix(f".top{top_k}_ranked.csv")
    write_ranked_top_k(ranked_summary_path, rows, feature_idx, delta_accel, delta_decel, top_k)
    return summary_path, ranked_summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SAE latent activation deltas for accelerating and decelerating trajectories."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("output/analysis_control/sae_control_speed_change.csv"),
        help="CSV produced by analyze_sae_control.py for speed-change labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sae_output/analysis_control/plots/sae_accel_decel_latent_deltas.png"),
        help="Output figure path.",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of strongest latents to highlight per panel.")
    parser.add_argument(
        "--title",
        default="Mean SAE latent delta per acceleration transition",
        help="Figure title.",
    )
    args = parser.parse_args()

    summary_path, ranked_summary_path = plot_accel_decel(args.input_csv, args.output, args.top_k, args.title)
    print(f"Saved plot to {args.output}")
    print(f"Saved latent-delta summary to {summary_path}")
    print(f"Saved ranked top-{args.top_k} summary to {ranked_summary_path}")


if __name__ == "__main__":
    main()
