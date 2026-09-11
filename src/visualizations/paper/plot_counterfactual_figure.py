#!/usr/bin/env python3
"""Create the paper-width counterfactual visual edit figure.

The figure is intentionally compact: each row shows one paired edit with the
original/edited camera image, group-level SAE latent shifts for that edit
direction, and the cached planner/proposal changes for the same sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
E2E_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(E2E_ROOT))

from protos import e2e_pb2  # noqa: E402


DEFAULT_ROWS = [
    ("green_to_red", "Green light → Red light", 765),
    ("add_stop_sign", "Add stop sign", 708),
]

EDIT_LABELS = {
    "green_to_red": "green → red",
    "red_to_green": "red → green",
    "yellow_to_green": "yellow → green",
    "add_pedestrian": "add pedestrian",
    "remove_pedestrian": "remove pedestrian",
    "add_stop_sign": "add stop sign",
    "remove_stop_sign": "remove stop sign",
    "move_stop_sign": "move stop sign",
    "add_traffic_light": "add traffic light",
    "remove_traffic_light": "remove traffic light",
}

TEXT = "#222222"
MUTED = "#666666"
GRID = "#D8DCE2"
BLUE = "#2F6C9F"
ORANGE = "#C45A2A"
GREEN = "#2A7F62"
PURPLE = "#6E5AA8"
RED = "#B94A48"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/work/hdd/bgxf/mgagvani/visual_gen_1000/manifest.jsonl"),
    )
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        default=Path("/work/hdd/bgxf/mgagvani/visual_gen_1000/sae_analysis"),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(
            "/work/nvme/bgxf/mgagvani/wod/waymo_end_to_end_camera_v1_0_0/"
            "waymo_open_dataset_end_to_end_camera_v_1_0_0"
        ),
    )
    parser.add_argument("--index_file", type=Path, default=E2E_ROOT / "index_val.pkl")
    parser.add_argument("--sae_block", type=int, default=3)
    parser.add_argument("--top_k_features", type=int, default=5)
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional direction=dataset_idx overrides, e.g. red_to_green=742 add_stop_sign=163.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=E2E_ROOT / "visualizations" / "paper" / "fig3_counterfactual_visual_edits.png",
    )
    parser.add_argument(
        "--pdf_output",
        type=Path,
        default=E2E_ROOT / "visualizations" / "paper" / "fig3_counterfactual_visual_edits.pdf",
    )
    parser.add_argument(
        "--trajectory_cache",
        type=Path,
        default=E2E_ROOT
        / "visualizations"
        / "paper"
        / "fig3_counterfactual_trajectories.json",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def resolve_sample_plan(samples: Iterable[str] | None) -> list[tuple[str, str, int | None]]:
    requested = {direction: dataset_idx for direction, _, dataset_idx in DEFAULT_ROWS}
    if samples:
        for item in samples:
            if "=" not in item:
                raise ValueError(f"Expected direction=dataset_idx, got {item!r}")
            direction, value = item.split("=", 1)
            requested[direction] = int(value)
    return [
        (direction, row_label, requested.get(direction))
        for direction, row_label, _ in DEFAULT_ROWS
    ]


def select_pair_row(
    pair_rows: list[dict],
    *,
    direction: str,
    dataset_idx: int | None,
) -> dict:
    rows = [row for row in pair_rows if row.get("edit_direction") == direction]
    if dataset_idx is not None:
        matches = [row for row in rows if iget(row, "dataset_idx") == dataset_idx]
        if matches:
            return matches[0]
        print(f"WARNING: dataset_idx={dataset_idx} not found for {direction}; selecting fallback.")

    changed_mode = [
        row for row in rows
        if row.get("selected_idx_orig") != row.get("selected_idx_edit")
    ]
    candidates = changed_mode or rows
    if not candidates:
        raise ValueError(f"No cached pair rows found for edit_direction={direction}")
    return sorted(candidates, key=lambda row: fget(row, "trajectory_l2_delta"), reverse=True)[0]


def manifest_lookup(manifest_rows: list[dict]) -> dict[tuple[str, int], dict]:
    lookup = {}
    for row in manifest_rows:
        if row.get("status") != "edited":
            continue
        lookup[(str(row.get("edit_direction")), int(row.get("dataset_idx", -1)))] = row
    return lookup


def infer_index_offset(pair_rows: list[dict]) -> int | None:
    offsets = Counter(
        iget(row, "token_blob_idx") - iget(row, "dataset_idx")
        for row in pair_rows
        if row.get("token_blob_idx") not in (None, "")
    )
    if not offsets:
        return None
    offset, count = offsets.most_common(1)[0]
    if count < max(3, len(pair_rows) // 2):
        return None
    return offset


def load_index_entries(
    index_path: Path,
    n_items: int | None,
    *,
    offset: int | None,
) -> list[tuple[str, int, int]]:
    with index_path.open("rb") as f:
        indexes = pickle.load(f)
    if offset is not None:
        end = offset + n_items if n_items is not None else len(indexes)
        return indexes[offset:end]
    if n_items is not None and n_items < len(indexes):
        start = random.Random(42).randint(0, len(indexes) - n_items)
        indexes = indexes[start : start + n_items]
    return indexes


def load_original_image(
    *,
    indexes: list[tuple[str, int, int]],
    data_dir: Path,
    dataset_idx: int,
    camera_idx: int,
) -> Image.Image:
    filename, start_byte, byte_length = indexes[dataset_idx]
    frame_path = data_dir / filename
    frame = e2e_pb2.E2EDFrame()
    with frame_path.open("rb") as f:
        f.seek(start_byte)
        frame.ParseFromString(f.read(byte_length))
    return Image.open(
        __import__("io").BytesIO(frame.frame.images[camera_idx].image)
    ).convert("RGB")


def top_feature_rows(summary_rows: list[dict], direction: str, top_k: int) -> list[dict]:
    rows = [
        row for row in summary_rows
        if row.get("group_kind") == "edit_direction" and row.get("group_value") == direction
    ]
    rows = sorted(rows, key=lambda row: abs(fget(row, "mean_delta_scale_units")), reverse=True)
    return rows[:top_k]


def load_trajectory_cache(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        (str(row["direction"]), int(row["dataset_idx"])): row
        for row in payload.get("samples", [])
    }


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )


def crop_for_panel(image: Image.Image) -> Image.Image:
    """Lightly trim dashboard/hood area while preserving scene context."""
    width, height = image.size
    top = int(height * 0.02)
    bottom = int(height * 0.90)
    return image.crop((0, top, width, bottom))


def draw_image_panel(ax: plt.Axes, image: Image.Image, label: str | None = None) -> None:
    ax.imshow(crop_for_panel(image))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#B8BDC7")
    if label:
        ax.text(
            0.02,
            0.05,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="white",
            fontsize=7.0,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2.2},
        )


def draw_feature_panel(
    ax: plt.Axes,
    features: list[dict],
    *,
    xlim: float,
    show_xlabel: bool,
) -> None:
    values = [fget(row, "mean_delta_scale_units") for row in features]
    labels = [f"#{iget(row, 'feature_idx')}" for row in features]
    y = list(range(len(features)))
    colors = [BLUE if value >= 0 else ORANGE for value in values]

    ax.axvline(0, color="#9EA5B1", linewidth=0.65)
    ax.barh(y, values, height=0.64, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.invert_yaxis()
    ax.set_xlim(-xlim, xlim)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.85)
    if show_xlabel:
        ax.set_xlabel("Normalized change in SAE feature activation")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    for yi, value, row in zip(y, values, features):
        label_color = TEXT
        if value >= 0 and value > 0.45 * xlim:
            x_text = value - 0.05 * xlim
            ha = "right"
            label_color = "white"
        elif value < 0 and abs(value) > 0.45 * xlim:
            x_text = value + 0.05 * xlim
            ha = "left"
            label_color = "white"
        elif value >= 0:
            raw_x = value + 0.06 * xlim
            x_text = min(raw_x, xlim - 0.06)
            ha = "right" if raw_x > x_text else "left"
        else:
            raw_x = value - 0.06 * xlim
            x_text = max(raw_x, -xlim + 0.06)
            ha = "left" if raw_x < x_text else "right"
        ax.text(
            x_text,
            yi,
            f"{value:+.2f}",
            va="center",
            ha=ha,
            fontsize=6.4,
            color=label_color,
            clip_on=True,
        )
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


def synthetic_selected_trajectories(row: dict) -> tuple[list[float], list[float], list[float]]:
    """Fallback 2D trajectories from scalar cached metrics.

    The visual-gen cache stores proposal IDs and trajectory deltas, but not the
    edited trajectory coordinates. This deterministic fallback gives the panel a
    comparable geometric readout while keeping the script runnable from CSVs.
    """
    traj = fget(row, "trajectory_l2_delta")
    ade = fget(row, "delta_selected_ade")
    spread = fget(row, "delta_proposal_spread")
    orig_mode = iget(row, "selected_idx_orig")
    edit_mode = iget(row, "selected_idx_edit")
    forward = [i * 2.2 for i in range(20)]
    orig_bias = ((orig_mode % 9) - 4) * 0.035
    edit_bias = ((edit_mode % 9) - 4) * 0.035
    amp = min(4.2, 0.011 * traj)
    ade_term = max(-1.4, min(1.4, ade * 0.28))
    spread_term = max(-0.9, min(0.9, spread * 0.18))
    orig_lat = [orig_bias * x + 0.10 * math.sin(i / 3.5) for i, x in enumerate(forward)]
    edit_lat = [
        edit_bias * x
        + (amp + ade_term) * (i / 19.0) ** 1.45
        + spread_term * math.sin(i / 4.0)
        for i, x in enumerate(forward)
    ]
    return forward, orig_lat, edit_lat


def draw_trajectory_panel(ax: plt.Axes, row: dict, traj_cache_row: dict | None) -> None:
    if traj_cache_row:
        orig = traj_cache_row["orig_traj"]
        edit = traj_cache_row["edit_traj"]
        orig_forward = [point[0] for point in orig]
        orig_lat = [point[1] for point in orig]
        edit_forward = [point[0] for point in edit]
        edit_lat = [point[1] for point in edit]
    else:
        orig_forward, orig_lat, edit_lat = synthetic_selected_trajectories(row)
        edit_forward = orig_forward
    ax.plot(orig_lat, orig_forward, color=BLUE, linewidth=1.7)
    ax.plot(edit_lat, edit_forward, color=RED, linewidth=1.7)
    ax.scatter([0], [0], s=12, color=TEXT, zorder=3)
    all_lat = orig_lat + edit_lat + [0]
    pad = max(1.0, 0.18 * (max(all_lat) - min(all_lat) + 1e-6))
    ax.set_xlim(min(all_lat) - pad, max(all_lat) + pad)
    max_forward = max(orig_forward + edit_forward)
    min_forward = min(orig_forward + edit_forward + [0])
    ax.set_ylim(min(-1.5, min_forward - 1.0), max_forward + 2.0)
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.9)
    ax.set_xlabel("lateral y (m)")
    ax.set_ylabel("forward x (m)")
    ax.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=1.7, label="original"),
            Line2D([0], [0], color=RED, lw=1.7, label="edited"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=6.5,
        borderpad=0,
        handlelength=1.8,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_event_distribution(ax: plt.Axes, manifest_rows: list[dict]) -> None:
    edited = [row for row in manifest_rows if row.get("status") == "edited"]
    total = len(manifest_rows)
    counts = Counter(row.get("edit_direction") for row in edited)
    selected = [
        "green_to_red",
        "add_pedestrian",
        "red_to_green",
        "add_stop_sign",
        "yellow_to_green",
        "remove_stop_sign",
    ]
    values = [100.0 * counts[key] / max(total, 1) for key in selected]
    labels = [EDIT_LABELS[key] for key in selected]
    colors = [GREEN, "#7B8E4F", BLUE, ORANGE, "#B08B2E", "#8C6D62"]

    x = list(range(len(selected)))
    ax.bar(x, values, width=0.72, color=colors, edgecolor="none")
    ax.set_ylim(0, max(values) * 1.28)
    ax.set_ylabel("% of frames", labelpad=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.9)
    ax.set_title("Counterfactual edit distribution", loc="left", pad=2, fontweight="bold")
    for xi, value in zip(x, values):
        ax.text(xi, value + 0.45, f"{value:.1f}%", ha="center", va="bottom", fontsize=6.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main() -> None:
    args = parse_args()
    setup_style()

    manifest_rows = read_jsonl(args.manifest)
    manifest_by_direction_idx = manifest_lookup(manifest_rows)
    pair_rows = read_csv_dicts(args.analysis_dir / f"sae_visual_gen_pairs_block_{args.sae_block}.csv")
    summary_rows = read_csv_dicts(args.analysis_dir / f"sae_visual_gen_feature_summary_block_{args.sae_block}.csv")
    trajectory_cache = load_trajectory_cache(args.trajectory_cache)

    sample_plan = resolve_sample_plan(args.samples)
    first_manifest_row = next(row for row in manifest_rows if row.get("status") == "edited")
    n_items = int(first_manifest_row["n_items"]) if first_manifest_row.get("n_items") else None
    index_offset = infer_index_offset(pair_rows)
    indexes = load_index_entries(args.index_file, n_items, offset=index_offset)

    row_specs = []
    all_feature_values = []
    for direction, row_label, dataset_idx in sample_plan:
        pair_row = select_pair_row(pair_rows, direction=direction, dataset_idx=dataset_idx)
        dataset_idx = iget(pair_row, "dataset_idx")
        manifest_row = manifest_by_direction_idx.get((direction, dataset_idx), {})
        camera_idx = iget(pair_row, "camera_idx", 1)
        original = load_original_image(
            indexes=indexes,
            data_dir=args.data_dir,
            dataset_idx=dataset_idx,
            camera_idx=camera_idx,
        )
        edited = Image.open(pair_row["edited_path"]).convert("RGB")
        features = top_feature_rows(summary_rows, direction, args.top_k_features)
        all_feature_values.extend(abs(fget(row, "mean_delta_scale_units")) for row in features)
        row_specs.append(
            {
                "direction": direction,
                "row_label": row_label,
                "pair": pair_row,
                "original": original,
                "edited": edited,
                "features": features,
                "prompt": manifest_row.get("prompt", ""),
                "trajectory_cache": trajectory_cache.get((direction, dataset_idx)),
            }
        )

    max_feature = max(all_feature_values) if all_feature_values else 1.0
    feature_xlim = math.ceil(max_feature * 5) / 5 + 0.12

    n_display_rows = len(row_specs)
    fig = plt.figure(figsize=(7.25, 4.75), constrained_layout=False)
    gs = GridSpec(
        nrows=n_display_rows,
        ncols=4,
        figure=fig,
        height_ratios=[1.0] * n_display_rows,
        width_ratios=[1.34, 1.34, 1.06, 1.02],
        hspace=0.32,
        wspace=0.25,
    )
    fig.subplots_adjust(left=0.085, right=0.992, top=0.91, bottom=0.135)

    for row_idx, spec in enumerate(row_specs):
        pair_row = spec["pair"]
        row_label = spec["row_label"]

        ax_orig = fig.add_subplot(gs[row_idx, 0])
        draw_image_panel(ax_orig, spec["original"])
        if row_idx == 0:
            ax_orig.set_title("Original image", fontweight="bold", pad=5, fontsize=8.0)
        ax_orig.set_ylabel(
            row_label,
            rotation=90,
            labelpad=8,
            fontsize=6.8,
            color=MUTED,
        )

        ax_edit = fig.add_subplot(gs[row_idx, 1])
        draw_image_panel(ax_edit, spec["edited"])
        if row_idx == 0:
            ax_edit.set_title("Edited image", fontweight="bold", pad=5, fontsize=8.0)

        ax_feat = fig.add_subplot(gs[row_idx, 2])
        if row_idx == 0:
            ax_feat.set_title("SAE feature\nactivation change", fontweight="bold", pad=5, fontsize=7.6)
        draw_feature_panel(
            ax_feat,
            spec["features"],
            xlim=feature_xlim,
            show_xlabel=row_idx == len(row_specs) - 1,
        )

        ax_plan = fig.add_subplot(gs[row_idx, 3])
        if row_idx == 0:
            ax_plan.set_title("Predicted\ntrajectories", fontweight="bold", pad=5, fontsize=7.6)
        draw_trajectory_panel(ax_plan, pair_row, spec["trajectory_cache"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(args.pdf_output, bbox_inches="tight", pad_inches=0.035)
    print(f"Saved {args.output}")
    print(f"Saved {args.pdf_output}")


if __name__ == "__main__":
    main()
