#!/usr/bin/env python3
"""Supplement figures S4.3 (additional examples) and S4.4 (failure cases).

Both figures are built from real cached artifacts:
  - real planner trajectories (original/edited) from the expected-gallery cache,
  - real edited images from the visual-edit manifest,
  - real per-direction SAE feature deltas from the visual-gen summary CSV.

Style follows visualizations/paper/FIGURE_STYLE_GUIDE.md and reuses the
helpers from plot_counterfactual_figure.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from PIL import Image

SCRIPT_PATH = Path(__file__).resolve()
PAPER_DIR = SCRIPT_PATH.parent
E2E_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(E2E_ROOT))
sys.path.insert(0, str(PAPER_DIR))

from protos import e2e_pb2  # noqa: E402

# Reuse the Fig. 3 style + drawing helpers so the supplement matches the paper.
from plot_counterfactual_figure import (  # noqa: E402
    BLUE,
    EDIT_LABELS,
    GRID,
    MUTED,
    RED,
    TEXT,
    crop_for_panel,
    draw_feature_panel,
    draw_image_panel,
    fget,
    read_csv_dicts,
    setup_style,
    top_feature_rows,
)

GT_GREEN = "#2A7F62"

# One example per edit direction (second-ranked in the gallery cache; the
# first-ranked scene per direction is skipped). add_pedestrian uses the
# next scene after #313 because the night edit is hard to interpret.
DEFAULT_EXAMPLES = [
    ("green_to_red", 477, "green \u2192 red"),
    ("red_to_green", 603, "red \u2192 green"),
    ("add_stop_sign", 236, "add stop sign"),
    ("add_pedestrian", 202, "add pedestrian"),
]

DEFAULT_FAILURES = [
    {
        "direction": "add_stop_sign",
        "dataset_idx": 391,
        "label": "Unrealistic / non-minimal edit",
    },
    {
        "direction": "add_pedestrian",
        "dataset_idx": 496,
        "label": "Irrelevant-object edit",
    },
    {
        "direction": "add_stop_sign",
        "dataset_idx": 236,
        "label": "Visible edit, weak trajectory change",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory_cache",
        type=Path,
        default=PAPER_DIR / "counterfactual_gallery_expected_trajectories.json",
    )
    parser.add_argument(
        "--failures_trajectory_cache",
        type=Path,
        default=PAPER_DIR / "supp_edit_failures_trajectories.json",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path(
            "/work/hdd/bgxf/mgagvani/visual_gen_1000/sae_analysis/"
            "sae_visual_gen_feature_summary_block_3.csv"
        ),
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
    parser.add_argument("--top_k_features", type=int, default=5)
    parser.add_argument(
        "--examples",
        nargs="*",
        default=None,
        help="Override successes as direction=dataset_idx (uses default labels).",
    )
    parser.add_argument(
        "--pairs_csv",
        type=Path,
        default=Path(
            "/work/hdd/bgxf/mgagvani/visual_gen_1000/sae_analysis/"
            "sae_visual_gen_pairs_block_3.csv"
        ),
        help="Per-example pairs CSV; maps dataset_idx -> token_blob_idx/edited_path for image-only failures.",
    )
    parser.add_argument("--examples_output", type=Path, default=PAPER_DIR / "supp_edit_examples.png")
    parser.add_argument("--failures_output", type=Path, default=PAPER_DIR / "supp_edit_failures.png")
    return parser.parse_args()


def load_cache(path: Path) -> dict[tuple[str, int], dict]:
    payload = json.loads(path.read_text())
    return {
        (str(row["direction"]), int(row["dataset_idx"])): row
        for row in payload.get("samples", [])
    }


def load_original_image(
    indexes: list[tuple[str, int, int]],
    data_dir: Path,
    token_blob_idx: int,
    camera_idx: int,
) -> Image.Image:
    filename, start_byte, byte_length = indexes[token_blob_idx]
    frame = e2e_pb2.E2EDFrame()
    with (data_dir / filename).open("rb") as f:
        f.seek(start_byte)
        frame.ParseFromString(f.read(byte_length))
    import io

    return Image.open(io.BytesIO(frame.frame.images[camera_idx].image)).convert("RGB")


def draw_trajectory_panel(ax: plt.Axes, row: dict, *, show_gt: bool, show_legend: bool) -> None:
    orig = row["orig_traj"]
    edit = row["edit_traj"]
    future = row.get("future") if show_gt else None

    ax.plot([p[1] for p in orig], [p[0] for p in orig], color=BLUE, linewidth=1.7)
    ax.plot([p[1] for p in edit], [p[0] for p in edit], color=RED, linewidth=1.7)
    handles = [
        Line2D([0], [0], color=BLUE, lw=1.7, label="original"),
        Line2D([0], [0], color=RED, lw=1.7, label="edited"),
    ]
    if future:
        ax.plot(
            [p[1] for p in future],
            [p[0] for p in future],
            color=GT_GREEN,
            linewidth=1.1,
            linestyle="--",
        )
        handles.append(Line2D([0], [0], color=GT_GREEN, lw=1.1, ls="--", label="ground truth"))
    ax.scatter([0], [0], s=12, color=TEXT, zorder=3)

    lat = [p[1] for p in orig] + [p[1] for p in edit] + [0.0]
    fwd = [p[0] for p in orig] + [p[0] for p in edit] + [0.0]
    pad = max(1.0, 0.18 * (max(lat) - min(lat) + 1e-6))
    ax.set_xlim(min(lat) - pad, max(lat) + pad)
    ax.set_ylim(min(-1.5, min(fwd) - 1.0), max(fwd) + 2.0)
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.9)
    ax.set_xlabel("lateral y (m)")
    ax.set_ylabel("forward x (m)")
    if show_legend:
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6.2,
                  borderpad=0, handlelength=1.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def resolve_overrides(items, defaults):
    if not items:
        return defaults
    label_by_dir = {d: lbl for d, _, lbl in defaults}
    out = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"Expected direction=dataset_idx, got {it!r}")
        d, v = it.split("=", 1)
        out.append((d, int(v), label_by_dir.get(d, EDIT_LABELS.get(d, d))))
    return out


def load_pairs_lookup(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    out = {}
    for row in read_csv_dicts(path):
        try:
            key = (str(row["edit_direction"]), int(float(row["dataset_idx"])))
        except (KeyError, ValueError):
            continue
        out[key] = row
    return out


def build_failure_specs(plan, *, cache, indexes, data_dir):
    specs = []
    for item in plan:
        direction = item["direction"]
        dataset_idx = int(item["dataset_idx"])
        row = cache.get((direction, dataset_idx))
        if row is None:
            print(f"WARNING: no cached trajectory for {direction}={dataset_idx}; skipping.")
            continue
        original = load_original_image(
            indexes, data_dir, int(row["token_blob_idx"]), int(row.get("camera_idx", 1))
        )
        edited = Image.open(row["edited_path"]).convert("RGB")
        specs.append(
            {
                "direction": direction,
                "label": item["label"],
                "original": original,
                "edited": edited,
                "row": row,
            }
        )
    return specs


def build_specs(plan, *, cache, indexes, data_dir, summary_rows, top_k):
    specs = []
    feat_vals = []
    for direction, dataset_idx, label in plan:
        row = cache.get((direction, dataset_idx))
        if row is None:
            print(f"WARNING: no cached trajectory for {direction}={dataset_idx}; skipping.")
            continue
        original = load_original_image(
            indexes, data_dir, int(row["token_blob_idx"]), int(row.get("camera_idx", 1))
        )
        edited = Image.open(row["edited_path"]).convert("RGB")
        features = top_feature_rows(summary_rows, direction, top_k) if summary_rows else []
        feat_vals.extend(abs(fget(r, "mean_delta_scale_units")) for r in features)
        specs.append(
            {
                "direction": direction,
                "label": label,
                "row": row,
                "original": original,
                "edited": edited,
                "features": features,
            }
        )
    return specs, feat_vals


def make_examples_figure(specs, feat_xlim, output: Path) -> None:
    n = len(specs)
    has_feat = any(s["features"] for s in specs)
    ncols = 4 if has_feat else 3
    width_ratios = [1.34, 1.34, 1.06, 1.02] if has_feat else [1.4, 1.4, 1.05]
    fig = plt.figure(figsize=(7.25, 1.18 * n + 0.35))
    gs = GridSpec(
        nrows=n, ncols=ncols, figure=fig,
        width_ratios=width_ratios, hspace=0.34, wspace=0.26,
    )
    fig.subplots_adjust(left=0.085, right=0.992, top=0.955, bottom=0.06)

    prev_dir = None
    for i, spec in enumerate(specs):
        ax_o = fig.add_subplot(gs[i, 0])
        draw_image_panel(ax_o, spec["original"])
        if i == 0:
            ax_o.set_title("Original image", fontweight="bold", pad=5, fontsize=8.0)
        side = spec["label"] if spec["direction"] != prev_dir else ""
        prev_dir = spec["direction"]
        ax_o.set_ylabel(side, rotation=90, labelpad=8, fontsize=6.8, color=MUTED)

        ax_e = fig.add_subplot(gs[i, 1])
        draw_image_panel(ax_e, spec["edited"])
        if i == 0:
            ax_e.set_title("Edited image", fontweight="bold", pad=5, fontsize=8.0)

        col = 2
        if has_feat:
            ax_f = fig.add_subplot(gs[i, col])
            if i == 0:
                ax_f.set_title("SAE feature\nactivation change", fontweight="bold", pad=5, fontsize=7.4)
            draw_feature_panel(ax_f, spec["features"], xlim=feat_xlim, show_xlabel=(i == n - 1))
            col += 1

        ax_t = fig.add_subplot(gs[i, col])
        if i == 0:
            ax_t.set_title("Predicted\ntrajectories", fontweight="bold", pad=5, fontsize=7.4)
        draw_trajectory_panel(ax_t, spec["row"], show_gt=False, show_legend=(i == 0))

    _save(fig, output)


def make_failures_figure(specs, output: Path) -> None:
    n = len(specs)
    fig = plt.figure(figsize=(7.25, 1.62 * n + 0.3))
    gs = GridSpec(nrows=n, ncols=3, figure=fig, width_ratios=[1.4, 1.4, 1.15],
                  hspace=0.5, wspace=0.24)
    fig.subplots_adjust(left=0.06, right=0.992, top=0.93, bottom=0.06)

    for i, spec in enumerate(specs):
        ax_o = fig.add_subplot(gs[i, 0])
        draw_image_panel(ax_o, spec["original"])
        if i == 0:
            ax_o.set_title("Original image", fontweight="bold", pad=5, fontsize=8.0)
        ax_o.set_ylabel(
            f"{chr(97 + i)}.  {spec['label']}",
            rotation=90, labelpad=8, fontsize=6.8, color=MUTED, va="center",
        )

        ax_e = fig.add_subplot(gs[i, 1])
        draw_image_panel(ax_e, spec["edited"])
        if i == 0:
            ax_e.set_title("Edited image", fontweight="bold", pad=5, fontsize=8.0)

        ax_third = fig.add_subplot(gs[i, 2])
        if i == 0:
            ax_third.set_title("Predicted\ntrajectories", fontweight="bold", pad=5, fontsize=7.4)
        draw_trajectory_panel(
            ax_third, spec["row"], show_gt=True, show_legend=(i == 0),
        )

    _save(fig, output)


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    fig.savefig(output, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {pdf}")


def main() -> None:
    args = parse_args()
    setup_style()

    cache = load_cache(args.trajectory_cache)
    with args.index_file.open("rb") as f:
        indexes = pickle.load(f)
    summary_rows = read_csv_dicts(args.summary_csv) if args.summary_csv.exists() else []
    if not summary_rows:
        print(f"WARNING: feature summary not found at {args.summary_csv}; omitting SAE panels.")

    import math

    example_plan = resolve_overrides(args.examples, DEFAULT_EXAMPLES)

    ex_specs, ex_feat = build_specs(
        example_plan, cache=cache, indexes=indexes, data_dir=args.data_dir,
        summary_rows=summary_rows, top_k=args.top_k_features,
    )
    failures_cache = load_cache(args.failures_trajectory_cache)
    fail_specs = build_failure_specs(
        DEFAULT_FAILURES, cache=failures_cache, indexes=indexes, data_dir=args.data_dir,
    )

    feat_xlim = (math.ceil(max(ex_feat) * 5) / 5 + 0.12) if ex_feat else 1.0

    if ex_specs:
        make_examples_figure(ex_specs, feat_xlim, args.examples_output)
    if fail_specs:
        make_failures_figure(fail_specs, args.failures_output)


if __name__ == "__main__":
    main()
