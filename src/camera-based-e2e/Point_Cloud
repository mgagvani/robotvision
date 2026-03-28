"""Offline BEV preprocessing script.

Iterates over the Waymo E2E dataset, generates per-frame Bird's Eye View
feature maps from multi-view camera images + monocular depth, and saves
them to disk alongside the dataset.

Usage:
    python create_bev.py \
        --data_dir /scratch/gilbreth/svelmuru/waymo_end_to_end_dataset/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
        --split train \
        --batch_size 8

Output:
    {data_dir}/../bev/{split}/bev_{idx:07d}.npy   — per-frame BEV  (4, 200, 200) float32
    {data_dir}/../bev/bev_index_{split}.pkl        — list mapping dataset index → .npy path
"""

import argparse
import os
import pickle

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from protos import e2e_pb2
from point_cloud_gpu import create_bev_from_frame_gpu, BEV_SIZE

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def load_depth_model(device):
    """Load the Depth-Anything-V2 model and processor."""
    print(f"Loading depth model: {DEPTH_MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device)
    model.eval()
    # FP16 for ~2x faster inference on modern GPUs
    if device.type == "cuda":
        model = model.half()
        print("Using FP16 inference")
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Generate offline BEV maps from Waymo E2E camera data")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to Waymo E2E data directory (same as used by loader.py)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                        help="Which split to process")
    parser.add_argument("--n_items", type=int, default=None,
                        help="Limit to first N items (for debugging)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Resume from this index (skip already-processed frames)")
    parser.add_argument("--index_dir", type=str, default=".",
                        help="Directory containing index_*.pkl files (default: current dir)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="Use 'gpu' (default) for fast PyTorch pipeline or 'cpu' for legacy Open3D")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Output directory: {data_dir}/../bev/{split}/ ──
    bev_base = Path(args.data_dir).parent / "bev"
    bev_split_dir = bev_base / args.split
    bev_split_dir.mkdir(parents=True, exist_ok=True)
    print(f"BEV output directory: {bev_split_dir}")

    # ── Load index ──
    index_file = os.path.join(args.index_dir, f"index_{args.split}.pkl")
    print(f"Loading index: {index_file}")
    with open(index_file, "rb") as f:
        indexes = pickle.load(f)

    # Compute processing range: [start_idx, end_idx)
    end_idx = len(indexes)
    if args.n_items is not None:
        end_idx = min(args.start_idx + args.n_items, end_idx)
    end_idx = max(end_idx, args.start_idx)  # ensure end >= start
    print(f"Processing frames {args.start_idx} to {end_idx - 1} ({end_idx - args.start_idx} frames)")

    # ── Load depth model ──
    depth_model, depth_processor = load_depth_model(device)

    # ── Process frames ──
    bev_index = []  # list of (dataset_idx, bev_path)
    open_file = None
    open_filename = ""

    viz_dir = Path("./visualizations")
    viz_dir.mkdir(exist_ok=True)
    saved_first_viz = False

    for idx in tqdm(range(args.start_idx, end_idx), desc=f"BEV [{args.split}]"):
        filename, start_byte, byte_length = indexes[idx]

        # Reuse file handle when reading from the same file
        if open_filename != filename:
            if open_file is not None:
                open_file.close()
            open_file = open(os.path.join(args.data_dir, filename), "rb")
            open_filename = filename

        open_file.seek(start_byte)
        protobuf = open_file.read(byte_length)

        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(protobuf)

        # Skip if already processed (checkpoint/resume support)
        bev_path = bev_split_dir / f"bev_{idx:07d}.npy"
        if bev_path.exists():
            bev_index.append((idx, str(bev_path)))
            continue

        # Generate BEV
        if args.backend == "gpu":
            bev = create_bev_from_frame_gpu(frame, depth_model, depth_processor, device)
        else:
            from point_cloud import create_bev_from_frame
            bev = create_bev_from_frame(frame, depth_model, depth_processor, device)
        assert bev.shape == (4, BEV_SIZE, BEV_SIZE), f"Unexpected BEV shape: {bev.shape}"

        # Save
        np.save(str(bev_path), bev)
        bev_index.append((idx, str(bev_path)))

        # Save a visualization of the very first BEV for sanity checking
        if not saved_first_viz:
            _save_bev_visualization(bev, viz_dir / f"bev_sample_{args.split}.png")
            saved_first_viz = True

    if open_file is not None:
        open_file.close()

    # ── Save index mapping ──
    index_path = bev_base / f"bev_index_{args.split}.pkl"
    with open(str(index_path), "wb") as f:
        pickle.dump(bev_index, f)
    print(f"\nSaved BEV index ({len(bev_index)} entries) to {index_path}")
    print("Done!")


def _save_bev_visualization(bev, out_path):
    """Save a 2×2 visualization of the 4 BEV channels."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    titles = ["Max Height", "Mean Height", "Log Point Density", "Mean Luminance"]
    cmaps = ["viridis", "viridis", "hot", "gray"]

    for i, (ax, title, cmap) in enumerate(zip(axes.flat, titles, cmaps)):
        im = ax.imshow(bev[i], cmap=cmap, origin="upper")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("BEV Channels (ego at center, +X up, +Y left)", fontsize=14)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved BEV visualization to {out_path}")


if __name__ == "__main__":
    main()
