"""Quick sanity-check visualizer for occupancy grids.

Loads a few occ_{idx}.npy files and shows:
  1. Top-down 2D slice (XY at ground level) — colored by class
  2. Side view (XZ slice through center)
  3. Class distribution bar chart

Does NOT require Open3D. Uses matplotlib only.

Usage (on login node — no GPU needed):
    python viz_occ.py \
        --occ_dir /scratch/gilbreth/kumar753/waymo_occ/train \
        --indices 0 100 500 1000 5000

Classes:
    0 = free (white)
    1 = vehicle (red)
    2 = pedestrian (blue)
    3 = cyclist (orange)
    4 = road (gray)
    5 = static (green)
  255 = unknown (black)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Class colors (RGB, normalized)
CLASS_COLORS = {
    255: (0.0,  0.0,  0.0),   # unknown — black
    0:   (1.0,  1.0,  1.0),   # free — white
    1:   (0.9,  0.1,  0.1),   # vehicle — red
    2:   (0.1,  0.3,  0.9),   # pedestrian — blue
    3:   (1.0,  0.6,  0.0),   # cyclist — orange
    4:   (0.5,  0.5,  0.5),   # road — gray
    5:   (0.2,  0.7,  0.2),   # static — green
}
CLASS_NAMES = {255: "unknown", 0: "free", 1: "vehicle", 2: "pedestrian",
               3: "cyclist", 4: "road", 5: "static"}

VOX_XY_SIZE = 100
VOX_Z_SIZE  = 16
VOX_Z_MIN   = -3.0
VOX_Z_RES   = 0.5


def grid_to_rgb(slice_2d):
    """Convert a 2D (H, W) uint8 class grid to (H, W, 3) RGB image."""
    h, w = slice_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        mask = slice_2d == cls
        rgb[mask] = color
    return rgb


def ground_level_z():
    """Return the voxel Z index closest to Z=0 (ground level)."""
    return int((0.0 - VOX_Z_MIN) / VOX_Z_RES)


def visualize_occ(occ_path, ax_row):
    """Visualize one occupancy grid on a row of 3 axes."""
    grid = np.load(occ_path)   # (100, 100, 16)
    idx = os.path.basename(occ_path).replace("occ_", "").replace(".npy", "")

    # ── Slice 1: top-down XY at ground level ──
    z_ground = ground_level_z()
    xy_slice = grid[:, :, z_ground]   # (100, 100)
    ax_row[0].imshow(grid_to_rgb(xy_slice), origin="upper")
    ax_row[0].set_title(f"idx={idx} XY@Z=0m", fontsize=9)
    ax_row[0].axis("off")

    # ── Slice 2: top-down XY — max occupied class along Z ──
    # For each XY cell take the most common non-unknown, non-free class
    best = np.full((VOX_XY_SIZE, VOX_XY_SIZE), 255, dtype=np.uint8)
    for z in range(VOX_Z_SIZE):
        layer = grid[:, :, z]
        occupied = (layer >= 1) & (layer <= 5)
        best[occupied] = layer[occupied]
    ax_row[1].imshow(grid_to_rgb(best), origin="upper")
    ax_row[1].set_title(f"idx={idx} XY max-Z", fontsize=9)
    ax_row[1].axis("off")

    # ── Slice 3: class distribution ──
    classes = [0, 1, 2, 3, 4, 5, 255]
    counts = [int((grid == c).sum()) for c in classes]
    colors = [CLASS_COLORS[c] for c in classes]
    labels = [CLASS_NAMES[c] for c in classes]
    bars = ax_row[2].bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax_row[2].set_title(f"idx={idx} voxel counts", fontsize=9)
    ax_row[2].tick_params(axis="x", labelsize=7, rotation=30)
    ax_row[2].tick_params(axis="y", labelsize=7)

    # Print summary
    total = grid.size
    known = (grid != 255).sum()
    free = (grid == 0).sum()
    occupied_sum = sum((grid == c).sum() for c in range(1, 6))
    print(f"  idx={idx}: known={known/total:.1%}  free={free/total:.1%}  "
          f"occupied={occupied_sum/total:.1%}  unknown={((grid==255).sum())/total:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--occ_dir", type=str, required=True,
                        help="Directory containing occ_{idx}.npy files")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 100, 500, 1000, 5000],
                        help="Frame indices to visualize")
    parser.add_argument("--out", type=str, default="occ_viz.png",
                        help="Output image path")
    args = parser.parse_args()

    # Filter to indices that actually exist
    paths = []
    for idx in args.indices:
        p = os.path.join(args.occ_dir, f"occ_{idx:07d}.npy")
        if os.path.exists(p):
            paths.append(p)
        else:
            print(f"  Warning: {p} not found, skipping")

    if not paths:
        print("No valid occ files found. Check --occ_dir and --indices.")
        return

    n = len(paths)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    print(f"\nVoxel statistics:")
    for i, path in enumerate(paths):
        visualize_occ(path, axes[i])

    # Legend
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c], linewidth=0.5,
                       edgecolor="black")
        for c in [255, 0, 1, 2, 3, 4, 5]
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=7,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))

    plt.suptitle("Occupancy grid sanity check\n"
                 "Left: XY slice at Z=0m  |  Middle: XY max-Z projection  |  Right: class counts",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

