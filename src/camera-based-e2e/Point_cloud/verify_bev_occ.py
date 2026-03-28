"""
Compare BEV density map with OCC semantic labels for the same frame.
Usage:
    python verify_bev_occ.py --idx 1
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

VOX_XY_RANGE = 25.0
VOX_XY_RES   = 0.5

CLASS_COLORS = {
    0: [1.0, 1.0, 1.0],   # free - white
    1: [1.0, 0.0, 0.0],   # vehicle - red
    2: [0.0, 0.3, 1.0],   # pedestrian - blue
    3: [1.0, 0.5, 0.0],   # cyclist - orange
    4: [0.4, 0.4, 0.4],   # road - grey
    5: [0.2, 0.7, 0.2],   # static - green
    255: [0.0, 0.0, 0.0], # unknown - black
}
CLASS_NAMES = {0:'free', 1:'vehicle', 2:'pedestrian',
               3:'cyclist', 4:'road', 5:'static', 255:'unknown'}

BEV_DIR = "/scratch/gilbreth/kumar753/waymo_bev/train"
OCC_DIR = "/scratch/gilbreth/kumar753/waymo_occ/train"


def occ_to_topdown(occ):
    """Collapse (100,100,16) to (100,100) taking dominant semantic class along Z."""
    topdown = np.full((100, 100), 255, dtype=np.uint8)
    # First pass: road/static
    for z in range(16):
        layer = occ[:, :, z]
        mask = (layer == 4) | (layer == 5)
        topdown[mask] = layer[mask]
    # Second pass: dynamic objects on top
    for z in range(16):
        layer = occ[:, :, z]
        mask = (layer >= 1) & (layer <= 3)
        topdown[mask] = layer[mask]
    # Third pass: free space
    for z in range(16):
        layer = occ[:, :, z]
        mask = (layer == 0) & (topdown == 255)
        topdown[mask] = 0
    return topdown


def grid_to_rgb(grid_2d):
    h, w = grid_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        rgb[grid_2d == cls] = color
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=1)
    parser.add_argument("--out", type=str, default="verify.png")
    args = parser.parse_args()

    bev_path = f"{BEV_DIR}/bev_{args.idx:07d}.npy"
    occ_path = f"{OCC_DIR}/occ_{args.idx:07d}.npy"

    bev = np.load(bev_path)  # (4, 200, 200)
    occ = np.load(occ_path)  # (100, 100, 16)

    occ_top = occ_to_topdown(occ)  # (100, 100)

    # Upsample OCC to 200x200 to match BEV
    occ_up = np.kron(occ_top, np.ones((2, 2), dtype=np.uint8))  # (200, 200)
    occ_rgb = grid_to_rgb(occ_up)  # (200, 200, 3)

    # BEV channels
    bev_density  = bev[2]  # log point density
    bev_height   = bev[0]  # max height
    bev_lumin    = bev[3]  # mean luminance

    # Normalize for display
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(norm(bev_density), cmap='hot', origin='upper')
    axes[0, 0].set_title(f'BEV Density (idx={args.idx})')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(norm(bev_height), cmap='viridis', origin='upper')
    axes[0, 1].set_title('BEV Max Height')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(norm(bev_lumin), cmap='gray', origin='upper')
    axes[0, 2].set_title('BEV Luminance')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(occ_rgb, origin='upper')
    axes[1, 0].set_title('OCC Top-Down Semantic')
    axes[1, 0].axis('off')

    # Overlay OCC on BEV density
    density_rgb = plt.cm.hot(norm(bev_density))[:, :, :3]
    alpha = 0.5
    overlay = np.where(
        (occ_rgb > 0).any(axis=2, keepdims=True),
        alpha * occ_rgb + (1 - alpha) * density_rgb,
        density_rgb
    )
    axes[1, 1].imshow(overlay, origin='upper')
    axes[1, 1].set_title('OCC overlaid on BEV Density')
    axes[1, 1].axis('off')

    # OCC overlaid on luminance
    lumin_rgb = np.stack([norm(bev_lumin)] * 3, axis=-1)
    overlay2 = np.where(
        (occ_rgb > 0).any(axis=2, keepdims=True),
        alpha * occ_rgb + (1 - alpha) * lumin_rgb,
        lumin_rgb
    )
    axes[1, 2].imshow(overlay2, origin='upper')
    axes[1, 2].set_title('OCC overlaid on BEV Luminance')
    axes[1, 2].axis('off')

    # Legend
    patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c], edgecolor='black', linewidth=0.5)
        for c in [0, 1, 2, 3, 4, 5, 255]
    ]
    fig.legend(handles=patches, loc='lower center', ncol=7, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))

    plt.suptitle(f'BEV vs OCC Alignment Check — frame {args.idx:07d}', fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
