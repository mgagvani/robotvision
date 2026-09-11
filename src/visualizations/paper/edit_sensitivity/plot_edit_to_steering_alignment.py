#!/usr/bin/env python3
"""Plot edit-to-steering alignment for SAE features.

This figure joins two independently generated analyses:

1. Visual counterfactual sensitivity from ``analyze_sae_visual_gen_pt2.py``:
   mean signed SAE activation change per edit direction.
2. Latent steering from ``analyze_sae_control.py``:
   signed behavioral change when manually increasing each SAE feature.

For each edit direction, a point is one SAE feature. The x-axis is the mean
feature change caused by that edit. The y-axis is the feature's steering effect
on braking, signed so positive values mean "more edit-consistent behavior":
stop-inducing edits use +Delta brake, while go-inducing edits use -Delta brake.
Thus features in quadrants I and III are aligned: the edit moves the feature in
the same direction that manual steering moves behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
E2E_ROOT = SCRIPT_PATH.parents[3]

DEFAULT_VISUAL_SUMMARY = Path(
    "/work/hdd/bgxf/mgagvani/visual_gen_1000/sae_analysis/"
    "sae_visual_gen_feature_summary_block_3.csv"
)
DEFAULT_CONTROL_STAT_ROWS = SCRIPT_PATH.parent / "sae_control_stat_rows_block_3_val.csv"
DEFAULT_OUTPUT_PREFIX = SCRIPT_PATH.parent / "fig5b_edit_to_steering_alignment"

DEFAULT_DIRECTIONS = (
    "green_to_red",
    "red_to_green",
    "add_stop_sign",
    "add_pedestrian",
)

EDIT_LABELS = {
    "green_to_red": "green → red",
    "red_to_green": "red → green",
    "yellow_to_green": "yellow → green",
    "add_stop_sign": "add stop sign",
    "remove_stop_sign": "remove stop sign",
    "add_pedestrian": "add pedestrian",
    "remove_pedestrian": "remove pedestrian",
    "add_traffic_light": "add traffic light",
    "remove_traffic_light": "remove traffic light",
}

BEHAVIOR_SIGN = {
    "green_to_red": 1.0,
    "add_stop_sign": 1.0,
    "add_pedestrian": 1.0,
    "add_traffic_light": 1.0,
    "red_to_green": -1.0,
    "yellow_to_green": -1.0,
    "remove_stop_sign": -1.0,
    "remove_pedestrian": -1.0,
    "remove_traffic_light": -1.0,
}

TEXT = "#222222"
MUTED = "#666666"
GRID = "#D8DCE2"
BLUE = "#2F6C9F"
RED = "#B94A48"
ORANGE = "#C45A2A"
GREEN = "#5E8C61"
PURPLE = "#7A5FA8"
BROWN = "#8C6D62"
TEAL = "#3B8C8A"
LIGHT_GRAY = "#C9CDD3"
CLUSTER_COLORS = [BLUE, ORANGE, GREEN, PURPLE, LIGHT_GRAY]
CLUSTER_LABELS = {
    4: "unclustered",
    5: "unclustered",
}
X_AXIS_LABEL = r"Visual edit response (normalized $\Delta z_i$)"
Y_AXIS_LABEL = "Standardized steering effect on braking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual_summary", type=Path, default=DEFAULT_VISUAL_SUMMARY)
    parser.add_argument("--control_stat_rows", type=Path, default=DEFAULT_CONTROL_STAT_ROWS)
    parser.add_argument("--output_prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--directions",
        type=str,
        default=",".join(DEFAULT_DIRECTIONS),
        help="Comma-separated edit directions to plot.",
    )
    parser.add_argument("--top_k_per_direction", type=int, default=80)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--control_metric", type=str, default="brake_mag")
    parser.add_argument(
        "--visual_metric",
        type=str,
        default="mean_delta_scale_units",
        help="Visual-summary column for signed feature change.",
    )
    parser.add_argument(
        "--steering_metric",
        type=str,
        default="std_effect_max",
        choices=["std_effect_max", "mean_delta_max"],
        help="Control-stat column for signed steering effect.",
    )
    return parser.parse_args()


def read_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fget(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def iget(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.1,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.4,
            "figure.dpi": 160,
            "savefig.dpi": 960,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def corr(x: np.ndarray, y: np.ndarray, *, spearman: bool = False) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if spearman:
        x = rankdata(x)
        y = rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def cosine_alignment(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    denom = float(np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def deterministic_kmeans(x: np.ndarray, n_clusters: int, *, max_iter: int = 100) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros(0, dtype=int)
    n_clusters = max(1, min(n_clusters, x.shape[0]))
    norms = np.linalg.norm(x, axis=1)
    centers = [x[int(np.argmax(norms))]]
    for _ in range(1, n_clusters):
        dist_sq = np.min(
            np.stack([np.sum((x - center) ** 2, axis=1) for center in centers], axis=1),
            axis=1,
        )
        centers.append(x[int(np.argmax(dist_sq))])
    centers = np.stack(centers, axis=0)
    labels = np.zeros(x.shape[0], dtype=int)
    for _ in range(max_iter):
        dist = np.stack([np.sum((x - center) ** 2, axis=1) for center in centers], axis=1)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_idx in range(n_clusters):
            mask = labels == cluster_idx
            if np.any(mask):
                centers[cluster_idx] = x[mask].mean(axis=0)
    return labels


def assign_clusters(
    feature_ids: list[int],
    directions: list[str],
    visual_by_direction: dict[str, dict[int, float]],
    n_clusters: int,
) -> dict[int, int]:
    if not feature_ids:
        return {}
    matrix = np.array(
        [[visual_by_direction.get(direction, {}).get(feature_id, 0.0) for direction in directions]
         for feature_id in feature_ids],
        dtype=float,
    )
    scale = np.std(matrix, axis=0)
    scale[scale == 0] = 1.0
    matrix = (matrix - np.mean(matrix, axis=0)) / scale
    try:
        from sklearn.cluster import KMeans

        labels = KMeans(
            n_clusters=max(1, min(n_clusters, len(feature_ids))),
            random_state=0,
            n_init=20,
        ).fit_predict(matrix)
    except Exception:
        labels = deterministic_kmeans(matrix, n_clusters)
    return {feature_id: int(label) for feature_id, label in zip(feature_ids, labels)}


def build_visual_maps(
    rows: list[dict],
    directions: list[str],
    visual_metric: str,
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, dict]]]:
    values: dict[str, dict[int, float]] = {direction: {} for direction in directions}
    raw_rows: dict[str, dict[int, dict]] = {direction: {} for direction in directions}
    for row in rows:
        if row.get("group_kind") != "edit_direction":
            continue
        direction = str(row.get("group_value"))
        if direction not in values:
            continue
        feature_idx = iget(row, "feature_idx")
        values[direction][feature_idx] = fget(row, visual_metric)
        raw_rows[direction][feature_idx] = row
    return values, raw_rows


def build_control_map(
    rows: list[dict],
    *,
    control_metric: str,
    steering_metric: str,
) -> dict[int, dict]:
    out = {}
    for row in rows:
        if row.get("stat_name") != control_metric:
            continue
        out[iget(row, "feature_idx")] = row
    if not out:
        raise ValueError(f"No rows found for control metric {control_metric!r}")
    for feature_idx, row in out.items():
        if steering_metric not in row:
            raise KeyError(f"Control row for feature {feature_idx} lacks {steering_metric!r}")
    return out


def selected_feature_ids_by_direction(
    visual_by_direction: dict[str, dict[int, float]],
    directions: list[str],
    top_k: int,
) -> dict[str, list[int]]:
    selected = {}
    for direction in directions:
        ranked = sorted(
            visual_by_direction.get(direction, {}).items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        selected[direction] = [feature_idx for feature_idx, _ in ranked[:top_k]]
    return selected


def build_point_rows(
    *,
    directions: list[str],
    visual_by_direction: dict[str, dict[int, float]],
    visual_raw: dict[str, dict[int, dict]],
    control_by_feature: dict[int, dict],
    selected_by_direction: dict[str, list[int]],
    clusters: dict[int, int],
    steering_metric: str,
) -> list[dict]:
    rows = []
    for direction in directions:
        sign = BEHAVIOR_SIGN.get(direction)
        if sign is None:
            raise KeyError(f"No behavior-sign convention for direction {direction!r}")
        for feature_idx in selected_by_direction.get(direction, []):
            if feature_idx not in visual_by_direction.get(direction, {}):
                continue
            if feature_idx not in control_by_feature:
                continue
            visual_row = visual_raw[direction][feature_idx]
            control_row = control_by_feature[feature_idx]
            raw_brake_effect = fget(control_row, steering_metric)
            edit_consistent_effect = sign * raw_brake_effect
            mean_delta = visual_by_direction[direction][feature_idx]
            rows.append(
                {
                    "edit_direction": direction,
                    "feature_idx": feature_idx,
                    "cluster_id": clusters.get(feature_idx, -1),
                    "cluster_label": CLUSTER_LABELS.get(clusters.get(feature_idx, -1), f"C{clusters.get(feature_idx, -1)}"),
                    "mean_delta_z_scale_units": mean_delta,
                    "abs_mean_delta_z_scale_units": abs(mean_delta),
                    "raw_brake_steering_effect": raw_brake_effect,
                    "edit_consistent_brake_steering_effect": edit_consistent_effect,
                    "aligned_sign": int(mean_delta * edit_consistent_effect > 0),
                    "visual_n": iget(visual_row, "n"),
                    "visual_paired_t": fget(visual_row, "paired_t"),
                    "visual_sign_consistency": fget(visual_row, "sign_consistency"),
                    "control_relevant_scene_count": iget(control_row, "relevant_scene_count"),
                    "control_mean_rho": fget(control_row, "mean_rho"),
                    "control_frac_consistent": fget(control_row, "frac_consistent"),
                    "control_score": fget(control_row, "control_score"),
                }
            )
    return rows


def summarize_points(rows: list[dict], directions: list[str]) -> list[dict]:
    out = []
    for direction in directions + ["pooled"]:
        group = rows if direction == "pooled" else [row for row in rows if row["edit_direction"] == direction]
        x = np.array([float(row["mean_delta_z_scale_units"]) for row in group], dtype=float)
        y = np.array([float(row["edit_consistent_brake_steering_effect"]) for row in group], dtype=float)
        nonzero = (x != 0) & (y != 0) & np.isfinite(x) & np.isfinite(y)
        out.append(
            {
                "edit_direction": direction,
                "n_points": int(len(group)),
                "pearson_r": corr(x, y),
                "spearman_r": corr(x, y, spearman=True),
                "cosine_alignment": cosine_alignment(x, y),
                "sign_match_rate": float(np.mean(x[nonzero] * y[nonzero] > 0)) if np.any(nonzero) else float("nan"),
                "median_abs_edit_delta": float(np.median(np.abs(x))) if len(x) else float("nan"),
                "median_abs_steering_effect": float(np.median(np.abs(y))) if len(y) else float("nan"),
            }
        )
    return out


def symmetric_limit(values: np.ndarray, *, floor: float) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return floor
    limit = float(np.quantile(finite, 0.985))
    return max(floor, limit * 1.12)


def draw_panel(
    ax: plt.Axes,
    rows: list[dict],
    summary: dict,
    *,
    xlim: float,
    ylim: float,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    ax.axhline(0, color="#9EA5B1", linewidth=0.65, zorder=0)
    ax.axvline(0, color="#9EA5B1", linewidth=0.65, zorder=0)
    for cluster_idx in sorted({int(row["cluster_id"]) for row in rows}):
        cluster_rows = [row for row in rows if int(row["cluster_id"]) == cluster_idx]
        x = [float(row["mean_delta_z_scale_units"]) for row in cluster_rows]
        y = [float(row["edit_consistent_brake_steering_effect"]) for row in cluster_rows]
        color = CLUSTER_COLORS[cluster_idx % len(CLUSTER_COLORS)]
        ax.scatter(
            x,
            y,
            s=13,
            color=color,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.25,
            label=CLUSTER_LABELS.get(cluster_idx, f"C{cluster_idx}"),
        )

    x = np.array([float(row["mean_delta_z_scale_units"]) for row in rows], dtype=float)
    y = np.array([float(row["edit_consistent_brake_steering_effect"]) for row in rows], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) >= 2 and np.std(x[mask]) > 0:
        slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
        xx = np.array([-xlim, xlim])
        ax.plot(xx, slope * xx + intercept, color=TEXT, linewidth=0.75, alpha=0.75)

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.9)
    ax.set_title(EDIT_LABELS.get(str(summary["edit_direction"]), str(summary["edit_direction"])), fontweight="bold")
    if show_xlabel:
        ax.set_xlabel(X_AXIS_LABEL)
    if show_ylabel:
        ax.set_ylabel(Y_AXIS_LABEL)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def make_figure(point_rows: list[dict], summary_rows: list[dict], directions: list[str], output_prefix: Path) -> None:
    setup_style()
    x_values = np.array([float(row["mean_delta_z_scale_units"]) for row in point_rows], dtype=float)
    y_values = np.array([float(row["edit_consistent_brake_steering_effect"]) for row in point_rows], dtype=float)
    xlim = symmetric_limit(x_values, floor=0.2)
    ylim = symmetric_limit(y_values, floor=0.02)

    ncols = 3 if len(directions) == 3 else 2
    nrows = int(math.ceil(len(directions) / ncols))
    figsize = (7.25, 2.55) if len(directions) == 3 else (7.25, 5.15)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    if len(directions) == 3:
        fig.subplots_adjust(left=0.078, right=0.992, top=0.82, bottom=0.23, wspace=0.28)
    else:
        fig.subplots_adjust(left=0.078, right=0.988, top=0.92, bottom=0.12, wspace=0.24, hspace=0.34)

    summary_by_direction = {row["edit_direction"]: row for row in summary_rows}
    for idx, direction in enumerate(directions):
        ax = axes[idx // ncols][idx % ncols]
        rows = [row for row in point_rows if row["edit_direction"] == direction]
        draw_panel(
            ax,
            rows,
            summary_by_direction[direction],
            xlim=xlim,
            ylim=ylim,
            show_xlabel=idx // ncols == nrows - 1,
            show_ylabel=idx % ncols == 0,
        )

    for idx in range(len(directions), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=min(len(unique), 6),
        frameon=False,
        bbox_to_anchor=(0.54, 1.02 if len(directions) == 3 else 0.985),
        handletextpad=0.25,
        columnspacing=0.8,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    directions = [part.strip() for part in args.directions.split(",") if part.strip()]

    visual_rows = read_csv_dicts(args.visual_summary)
    control_rows = read_csv_dicts(args.control_stat_rows)
    visual_by_direction, visual_raw = build_visual_maps(visual_rows, directions, args.visual_metric)
    control_by_feature = build_control_map(
        control_rows,
        control_metric=args.control_metric,
        steering_metric=args.steering_metric,
    )

    selected_by_direction = selected_feature_ids_by_direction(
        visual_by_direction,
        directions,
        args.top_k_per_direction,
    )
    feature_ids = sorted(
        {feature_idx for values in selected_by_direction.values() for feature_idx in values}
    )
    feature_ids = [feature_idx for feature_idx in feature_ids if feature_idx in control_by_feature]
    selected_by_direction = {
        direction: [feature_idx for feature_idx in values if feature_idx in control_by_feature]
        for direction, values in selected_by_direction.items()
    }
    clusters = assign_clusters(feature_ids, directions, visual_by_direction, args.n_clusters)
    point_rows = build_point_rows(
        directions=directions,
        visual_by_direction=visual_by_direction,
        visual_raw=visual_raw,
        control_by_feature=control_by_feature,
        selected_by_direction=selected_by_direction,
        clusters=clusters,
        steering_metric=args.steering_metric,
    )
    if not point_rows:
        raise ValueError("No joined visual/control points to plot.")

    summary_rows = summarize_points(point_rows, directions)
    output_prefix = args.output_prefix
    write_csv(output_prefix.with_name(f"{output_prefix.name}_points.csv"), point_rows)
    write_csv(output_prefix.with_name(f"{output_prefix.name}_summary.csv"), summary_rows)
    make_figure(point_rows, summary_rows, directions, output_prefix)

    pooled = next(row for row in summary_rows if row["edit_direction"] == "pooled")
    print(f"Saved {output_prefix.with_suffix('.png')}")
    print(f"Saved {output_prefix.with_suffix('.pdf')}")
    print(
        "Pooled alignment: "
        f"n={pooled['n_points']} pearson={pooled['pearson_r']:+.3f} "
        f"spearman={pooled['spearman_r']:+.3f} "
        f"sign_match={100.0 * pooled['sign_match_rate']:.1f}%"
    )


if __name__ == "__main__":
    main()
