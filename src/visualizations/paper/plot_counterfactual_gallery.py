#!/usr/bin/env python3
"""Make a rough contact sheet for choosing counterfactual examples."""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


SCRIPT_PATH = Path(__file__).resolve()
E2E_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(E2E_ROOT))

from protos import e2e_pb2  # noqa: E402


LABELS = {
    "green_to_red": "Green → Red",
    "red_to_green": "Red → Green",
    "add_stop_sign": "Add stop sign",
    "remove_stop_sign": "Remove stop sign",
    "add_pedestrian": "Add pedestrian",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory_cache",
        type=Path,
        default=E2E_ROOT
        / "visualizations"
        / "paper"
        / "counterfactual_gallery_trajectories.json",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=E2E_ROOT / "visualizations" / "paper" / "counterfactual_gallery.png",
    )
    parser.add_argument(
        "--pdf_output",
        type=Path,
        default=E2E_ROOT / "visualizations" / "paper" / "counterfactual_gallery.pdf",
    )
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "savefig.dpi": 180,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def decode_jpeg_bytes(data: bytes) -> np.ndarray:
    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    image = torchvision.io.decode_jpeg(tensor, mode=torchvision.io.ImageReadMode.RGB)
    return image.permute(1, 2, 0).cpu().numpy()


def load_original_image(indexes: list[tuple[str, int, int]], data_dir: Path, token_idx: int, camera_idx: int) -> np.ndarray:
    filename, start_byte, byte_length = indexes[token_idx]
    frame = e2e_pb2.E2EDFrame()
    with (data_dir / filename).open("rb") as f:
        f.seek(start_byte)
        frame.ParseFromString(f.read(byte_length))
    return decode_jpeg_bytes(frame.frame.images[camera_idx].image)


def load_edited_image(path: str) -> np.ndarray:
    return decode_jpeg_bytes(Path(path).read_bytes())


def crop(image: np.ndarray) -> np.ndarray:
    height = image.shape[0]
    return image[int(0.02 * height): int(0.90 * height)]


def draw_image(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(crop(image))
    ax.set_title(title, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#B8BDC7")


def draw_traj(ax: plt.Axes, row: dict) -> None:
    orig = row["orig_traj"]
    edit = row["edit_traj"]
    future = row.get("future")

    ax.plot([p[1] for p in orig], [p[0] for p in orig], color="#2F6C9F", lw=1.6, label="orig")
    ax.plot([p[1] for p in edit], [p[0] for p in edit], color="#B94A48", lw=1.6, label="edit")
    if future:
        ax.plot([p[1] for p in future], [p[0] for p in future], color="#2A7F62", lw=1.1, ls="--", label="gt")
    ax.scatter([0], [0], s=12, color="#222222", zorder=4)
    ax.grid(True, color="#D8DCE2", lw=0.45)
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.legend(loc="best", frameon=False, fontsize=6, handlelength=1.4)
    ax.set_title(
        f"mode {row['selected_idx_orig']}→{row['selected_idx_edit']} | "
        f"Δx {row.get('final_x_delta', 0.0):+.1f} | "
        f"L2 {row['trajectory_l2_delta']:.1f}",
        pad=2,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main() -> None:
    args = parse_args()
    setup_style()
    payload = json.loads(args.trajectory_cache.read_text())
    rows = payload["samples"]
    with args.index_file.open("rb") as f:
        indexes = pickle.load(f)

    n = len(rows)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=3,
        figsize=(12.5, max(2.0 * n, 8.0)),
        gridspec_kw={"width_ratios": [1.2, 1.2, 0.9], "wspace": 0.08, "hspace": 0.32},
    )
    if n == 1:
        axes = [axes]

    prev_direction = None
    for idx, (row, ax_row) in enumerate(zip(rows, axes)):
        direction = row["direction"]
        camera_idx = int(row.get("camera_idx", 1))
        original = load_original_image(indexes, args.data_dir, int(row["token_blob_idx"]), camera_idx)
        edited = load_edited_image(row["edited_path"])
        prefix = LABELS.get(direction, direction)
        marker = "     "
        if direction != prev_direction:
            marker = f"{prefix}     "
            prev_direction = direction
        row_title = f"{marker}#{idx + 1} dataset {row['dataset_idx']}"
        draw_image(ax_row[0], original, row_title)
        draw_image(ax_row[1], edited, "edited")
        draw_traj(ax_row[2], row)

    fig.suptitle("Counterfactual candidate gallery", y=0.997, fontsize=12, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(args.pdf_output, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {args.output}")
    print(f"Saved {args.pdf_output}")


if __name__ == "__main__":
    main()
