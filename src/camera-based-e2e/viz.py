import argparse, os, pathlib
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

from loader import WaymoE2E
from models.monocular import MonocularModel, SAMFeatures
from torch import nn

def gen_viz_data(model: nn.Module, data_root: str, num_samples: int, device: torch.device):
    """Generate the gt and pred trajectories along with past states"""
    dataset = WaymoE2E(
        batch_size=1,
        indexFile="index_val.pkl",
        data_dir=data_root,
        images=True,
        n_items=num_samples,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )
    model.eval()

    pred_trajectories = []
    gt_trajectories = []
    past_trajectories = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating trajectory data"):
            past, future, images, intent = batch['PAST'], batch['FUTURE'], batch['IMAGES'], batch['INTENT']
            past = past.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            intent = intent.to(device, non_blocking=True)
            images = [img.to(device, non_blocking=True) for img in images]
            B, T, F = past.shape

            # Extract past positions (x, y) from the first 2 features
            past_positions = past[:, :, :2].squeeze(0).cpu()  # (T, 2)
            past_trajectories.append(past_positions)

            pred_future = model({"PAST": past, "IMAGES": images, "INTENT": intent})
            pred_future = pred_future.view(B, -1, 2)
            pred_trajectories.append(pred_future.squeeze(0).cpu())
            gt_trajectories.append(future.squeeze(0).cpu())

    return past_trajectories, pred_trajectories, gt_trajectories


def create_animated_trajectory_plot(
    past_trajectories: List[torch.Tensor],
    pred_trajectories: List[torch.Tensor],
    gt_trajectories: List[torch.Tensor],
    save_path: Optional[str] = None,
    interval: int = 200,
):
    """
    Args:
        past_trajectories: List of past position tensors [(T_past, 2), ...]
        pred_trajectories: List of predicted future position tensors [(T_future, 2), ...]
        gt_trajectories: List of ground truth future position tensors [(T_future, 2), ...]
        save_path: path to save
        interval: 1/framerate (ms)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate global bounds once from all trajectories to prevent jumping
    all_x_coords = []
    all_y_coords = []
    for past, pred, gt in zip(past_trajectories, pred_trajectories, gt_trajectories):
        all_x_coords.extend([past[:, 0].numpy(), pred[:, 0].numpy(), gt[:, 0].numpy()])
        all_y_coords.extend([past[:, 1].numpy(), pred[:, 1].numpy(), gt[:, 1].numpy()])

    global_x = np.concatenate(all_x_coords)
    global_y = np.concatenate(all_y_coords)
    margin = 5.0
    x_min, x_max = global_x.min() - margin, global_x.max() + margin
    y_min, y_max = global_y.min() - margin, global_y.max() + margin

    past_color = "blue"
    pred_color = "red"
    gt_color = "green"

    (past_line,) = ax.plot(
        [],
        [],
        "o-",
        color=past_color,
        linewidth=2,
        markersize=4,
        label="Past Trajectory",
        alpha=0.8,
    )
    (pred_line,) = ax.plot(
        [],
        [],
        "o-",
        color=pred_color,
        linewidth=2,
        markersize=4,
        label="Predicted Future",
        alpha=0.8,
    )
    (gt_line,) = ax.plot(
        [],
        [],
        "o-",
        color=gt_color,
        linewidth=2,
        markersize=4,
        label="Ground Truth Future",
        alpha=0.8,
    )

    (current_pos,) = ax.plot([], [], "ko", markersize=8, label="Current Position")

    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    def init():
        ax.clear()
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Vehicle Trajectory Animation: Past → Present → Future")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return past_line, pred_line, gt_line, current_pos, text

    def animate(frame):
        # Update progress bar for frame generation
        if not hasattr(animate, 'pbar'):
            animate.pbar = tqdm(total=len(past_trajectories), desc="Generating animation frames", unit="frame", position=1, leave=False)
        animate.pbar.update(1)
        
        ax.clear()
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Vehicle Trajectory Animation: Past → Present → Future")
        ax.grid(True, alpha=0.3)

        sample_idx = frame % len(past_trajectories)

        past = past_trajectories[sample_idx].numpy()
        pred = pred_trajectories[sample_idx].numpy()
        gt = gt_trajectories[sample_idx].numpy()

        ax.plot(
            past[:, 0],
            past[:, 1],
            "o-",
            color=past_color,
            linewidth=2,
            markersize=4,
            label="Past Trajectory",
            alpha=0.8,
        )
        ax.plot(
            pred[:, 0],
            pred[:, 1],
            "o-",
            color=pred_color,
            linewidth=2,
            markersize=4,
            label="Predicted Future",
            alpha=0.8,
        )
        ax.plot(
            gt[:, 0],
            gt[:, 1],
            "o-",
            color=gt_color,
            linewidth=2,
            markersize=4,
            label="Ground Truth Future",
            alpha=0.8,
        )

        current_x, current_y = past[-1, 0], past[-1, 1]
        ax.plot(current_x, current_y, "ko", markersize=8, label="Current Position")

        # Use static bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.text(
            0.02,
            0.98,
            f"Sample {sample_idx + 1}/{len(past_trajectories)}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        ax.legend()
        ax.set_aspect("equal", adjustable="box")

        return []

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(past_trajectories),
        interval=interval,
        blit=False,
        repeat=True,
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=3)

    return anim


def calculate_metrics(
    pred_trajectories: List[torch.Tensor], gt_trajectories: List[torch.Tensor]
) -> dict:
    """Calculate trajectory prediction metrics"""
    ade_scores = []  # Average Displacement Error
    fde_scores = []  # Final Displacement Error

    for pred, gt in zip(pred_trajectories, gt_trajectories):
        # ADE: Average Euclidean distance over all time steps
        distances = torch.norm(pred - gt, dim=1)
        ade = torch.mean(distances).item()
        ade_scores.append(ade)

        # FDE: Euclidean distance at final time step
        fde = torch.norm(pred[-1] - gt[-1]).item()
        fde_scores.append(fde)

    return {
        "ADE_mean": np.mean(ade_scores),
        "ADE_std": np.std(ade_scores),
        "ADE_p1": np.percentile(ade_scores, 1),
        "ADE_p50": np.percentile(ade_scores, 50),
        "ADE_p99": np.percentile(ade_scores, 99),
        "FDE_mean": np.mean(fde_scores),
        "FDE_std": np.std(fde_scores),
        "FDE_p1": np.percentile(fde_scores, 1),
        "FDE_p50": np.percentile(fde_scores, 50),
        "FDE_p99": np.percentile(fde_scores, 99),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the root directory of the Waymo E2E dataset.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to animate."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualizations.",
    )
    parser.add_argument(
        "--animation_interval",
        type=int,
        default=1000,
        help="Animation interval in milliseconds.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = 16 * 6  # Past: (B, 16, 6)
    out_dim = 20 * 2  # Future: (B, 20, 2)
    model = MonocularModel(in_dim=in_dim, out_dim=out_dim, feature_extractor=SAMFeatures())

    # HACK: fix for loading torch.compile()d models
    ckpt = torch.load(args.model_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    mapped = {}
    for k, v in state.items():
        k = k.replace("model._orig_mod.", "").replace("model.", "")
        if k.startswith("features.sam_pos_embed"):
            k = k.replace("features.sam_pos_embed", "features.sam_model.model.pos_embed")
        elif k.startswith("features.sam_pos_embed_window"):
            k = k.replace("features.sam_pos_embed_window", "features.sam_model.model.pos_embed_window")
        elif k.startswith("features.sam_patch_embed"):
            k = k.replace("features.sam_patch_embed", "features.sam_model.model.patch_embed")
        elif k.startswith("features.sam_blocks"):
            k = k.replace("features.sam_blocks", "features.sam_model.model.blocks")
        mapped[k] = v

    model.load_state_dict(mapped, strict=True)
    model.to(device)
    model.eval()


    past_trajectories, pred_trajectories, gt_trajectories = gen_viz_data(
        model, args.data_root, args.num_samples, device
    )

    metrics = calculate_metrics(pred_trajectories, gt_trajectories)
    print(f"ADE @ 5s: {metrics['ADE_mean']:.3f} ± {metrics['ADE_std']:.3f} meters")
    print(f"ADE percentiles - 1%: {metrics['ADE_p1']:.3f}, 50%: {metrics['ADE_p50']:.3f}, 99%: {metrics['ADE_p99']:.3f}")
    print(f"FDE: {metrics['FDE_mean']:.3f} ± {metrics['FDE_std']:.3f} m")
    print(f"FDE percentiles - 1%: {metrics['FDE_p1']:.3f}, 50%: {metrics['FDE_p50']:.3f}, 99%: {metrics['FDE_p99']:.3f}")

    anim = create_animated_trajectory_plot(
        past_trajectories,
        pred_trajectories,
        gt_trajectories,
        save_path=os.path.join(args.output_dir, "trajectory_animation.gif"),
        interval=args.animation_interval,
    )
