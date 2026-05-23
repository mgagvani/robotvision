from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models.initial_sae import SparseAutoEncoder, load_torch_file


CAMERA_FRONT = 1

INTENT_NAMES = {
    0: "UNKNOWN",
    1: "GO_STRAIGHT",
    2: "GO_LEFT",
    3: "GO_RIGHT",
}

MOTION_NAMES = {
    0: "BACKWARD",
    1: "STOPPING",
    2: "FORWARD",
}

SPEED_CHANGE_NAMES = {
    -1: "NEUTRAL",
    0: "DECELERATING",
    1: "ACCELERATING",
}


@dataclass(frozen=True)
class PairSelection:
    anchor_idx: int
    partner_idx: int
    correlation: float
    conditional_prob: float
    output_dir: Path
    examples: list[dict]


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sae_model(ckpt_path: Path, device: torch.device) -> SparseAutoEncoder:
    ckpt = load_torch_file(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(ckpt).__name__}")
    if ckpt.get("mode") != "control":
        raise ValueError(f"Checkpoint mode must be 'control', got {ckpt.get('mode')!r}")

    model = SparseAutoEncoder(
        in_dims=int(ckpt["in_dims"]),
        expansion=int(ckpt["expansion"]),
        sparsity=float(ckpt.get("sparsity", 1e-4)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def select_best_controls(control_pred: torch.Tensor, scores_predicted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scores_predicted.ndim != 2:
        raise ValueError(f"Expected scores_predicted shape (N, K), got {tuple(scores_predicted.shape)}")
    if control_pred.shape[:2] != scores_predicted.shape:
        raise ValueError(
            "control_pred and scores_predicted disagree on (N, K): "
            f"{tuple(control_pred.shape[:2])} vs {tuple(scores_predicted.shape)}"
        )

    best_idx = scores_predicted.argmin(dim=1)
    row_idx = torch.arange(control_pred.shape[0], dtype=torch.long)
    best_control = control_pred[row_idx, best_idx]
    return best_control.contiguous(), best_idx


def select_best_trajectories(
    trajectory_predicted: Optional[torch.Tensor],
    best_idx: torch.Tensor,
    n_proposals: int,
) -> Optional[torch.Tensor]:
    if trajectory_predicted is None:
        return None
    if trajectory_predicted.ndim != 2:
        raise ValueError(
            f"Expected trajectory_predicted shape (N, K*T*2), got {tuple(trajectory_predicted.shape)}"
        )
    if trajectory_predicted.shape[0] != best_idx.shape[0]:
        raise ValueError("trajectory_predicted and best_idx disagree on number of scenes")
    if trajectory_predicted.shape[1] % (n_proposals * 2) != 0:
        raise ValueError(
            "trajectory_predicted second dim must be divisible by n_proposals * 2, got "
            f"{trajectory_predicted.shape[1]} vs {n_proposals * 2}"
        )

    horizon = trajectory_predicted.shape[1] // (n_proposals * 2)
    traj = trajectory_predicted.view(trajectory_predicted.shape[0], n_proposals, horizon, 2)
    row_idx = torch.arange(best_idx.shape[0], dtype=torch.long)
    return traj[row_idx, best_idx].contiguous()


def encode_latents(
    model: SparseAutoEncoder,
    best_control: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    flat = best_control.reshape(best_control.shape[0], -1).contiguous()
    outputs = []
    with torch.inference_mode():
        for start in range(0, flat.shape[0], batch_size):
            batch = flat[start : start + batch_size].to(device, non_blocking=True)
            latents = model(batch)["latents"].cpu()
            outputs.append(latents)
    return torch.cat(outputs, dim=0)


def derive_motion_labels(
    future: torch.Tensor,
    past: torch.Tensor,
    stop_disp_thresh: float,
    backward_disp_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    start_xy = past[:, -1, 0:2]
    final_xy = future[:, -1]
    displacement = final_xy - start_xy
    total_disp = torch.linalg.norm(displacement, dim=-1)

    vx0 = past[:, -1, 2]
    vy0 = past[:, -1, 3]
    heading0 = torch.atan2(vy0, vx0)
    heading_vec = torch.stack([torch.cos(heading0), torch.sin(heading0)], dim=-1)
    longitudinal_disp = (displacement * heading_vec).sum(dim=-1)

    labels = torch.full((future.shape[0],), 1, dtype=torch.long)
    stop_mask = total_disp < stop_disp_thresh
    backward_mask = (~stop_mask) & (longitudinal_disp < -backward_disp_thresh)
    forward_mask = (~stop_mask) & (~backward_mask)

    labels[backward_mask] = 0
    labels[forward_mask] = 2
    return labels, total_disp


def derive_speed_change_labels(
    future: torch.Tensor,
    past: torch.Tensor,
    dt: float,
    speed_delta_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start_xy = past[:, -1, 0:2]
    speed0 = torch.linalg.norm(past[:, -1, 2:4], dim=-1)

    prev_xy = torch.cat([start_xy.unsqueeze(1), future[:, :-1]], dim=1)
    step_speed = torch.linalg.norm(future - prev_xy, dim=-1) / dt
    final_speed = step_speed[:, -1]
    delta_speed = final_speed - speed0

    labels = torch.full((future.shape[0],), -1, dtype=torch.long)
    labels[delta_speed <= -speed_delta_thresh] = 0
    labels[delta_speed >= speed_delta_thresh] = 1
    return labels, speed0, final_speed, delta_speed


def compute_correlation_matrix(latents: torch.Tensor) -> torch.Tensor:
    values = latents.to(torch.float64)
    centered = values - values.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(values.shape[0], 1)
    var = torch.diag(cov).clamp_min(0.0)
    std = torch.sqrt(var)
    denom = std[:, None] * std[None, :]
    corr = torch.zeros_like(cov)
    valid = denom > 0
    corr[valid] = cov[valid] / denom[valid]
    corr.fill_diagonal_(1.0)
    return corr


def resolve_path(candidate: str, data_path: Path) -> Path:
    path = Path(candidate)
    if path.is_absolute() and path.exists():
        return path

    search_roots = [
        Path.cwd(),
        data_path.parent,
        data_path.parent.parent,
        Path(__file__).resolve().parent,
    ]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Could not resolve '{candidate}'. Checked cwd, data parent, repo script dir, and parent dirs."
    )


def build_front_camera_lookup(
    scene_names: set[str],
    data_dir: Path,
    index_path: Path,
) -> dict[str, np.ndarray]:
    from protos import e2e_pb2

    if not scene_names:
        return {}

    with index_path.open("rb") as handle:
        index = pickle.load(handle)
    if not isinstance(index, list):
        raise TypeError(f"Expected index pickle to be a list, got {type(index).__name__}")

    remaining = set(scene_names)
    lookup: dict[str, np.ndarray] = {}
    current_filename = None
    current_file = None

    try:
        for entry in index:
            if not remaining:
                break

            filename, start_byte, byte_length = entry
            if filename != current_filename:
                if current_file is not None:
                    current_file.close()
                current_file = (data_dir / filename).open("rb")
                current_filename = filename

            current_file.seek(start_byte)
            protobuf = current_file.read(byte_length)
            frame = e2e_pb2.E2EDFrame()
            frame.ParseFromString(protobuf)
            scene_name = frame.frame.context.name
            if scene_name not in remaining:
                continue

            front_image = None
            for image in frame.frame.images:
                if image.name == CAMERA_FRONT:
                    front_image = np.array(Image.open(BytesIO(image.image)).convert("RGB"))
                    break

            if front_image is not None:
                lookup[scene_name] = front_image
            remaining.remove(scene_name)
    finally:
        if current_file is not None:
            current_file.close()

    if remaining:
        warnings.warn(f"Missing {len(remaining)} requested scene(s) in index lookup; montages will omit those images.")
    return lookup


def save_anchor_summary_plot(anchor_idx: int, partner_rows: list[dict], output_path: Path) -> None:
    if not partner_rows:
        return

    labels = [f"f{int(row['partner_idx'])}" for row in reversed(partner_rows)]
    corr = [row["correlation"] for row in reversed(partner_rows)]
    cond = [row["partner_given_anchor"] for row in reversed(partner_rows)]

    fig_h = max(4.0, 0.55 * len(partner_rows) + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    bars = ax.barh(labels, corr, color="tab:blue", edgecolor="black", linewidth=0.4)
    ax.set_title(f"Anchor f{anchor_idx}: Top Co-Activated Partners")
    ax.set_xlabel("Pearson correlation")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    for bar, cond_prob in zip(bars, cond):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x + 0.01, y, f"P(partner|anchor)={cond_prob:.2f}", va="center", fontsize=8)

    ax.set_xlim(0, max(corr) * 1.25 if corr else 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_pair_montage(
    pair: PairSelection,
    front_images: dict[str, np.ndarray],
    best_traj_by_scene: dict[str, np.ndarray],
    past_by_scene: dict[str, np.ndarray],
    future_by_scene: dict[str, np.ndarray],
) -> None:
    if not pair.examples:
        return

    fig, axes = plt.subplots(len(pair.examples), 2, figsize=(13, 4.2 * len(pair.examples)))
    if len(pair.examples) == 1:
        axes = np.array([axes])

    fig.suptitle(
        f"Anchor f{pair.anchor_idx} with partner f{pair.partner_idx} | "
        f"corr={pair.correlation:.3f} | P(partner|anchor)={pair.conditional_prob:.3f}",
        fontsize=13,
        y=0.995,
    )

    for row_axes, example in zip(axes, pair.examples):
        ax_img, ax_traj = row_axes
        scene_name = example["scene_name"]
        image = front_images.get(scene_name)
        if image is None:
            ax_img.text(0.5, 0.5, "Front camera unavailable", ha="center", va="center", fontsize=11)
            ax_img.set_facecolor("#f0f0f0")
        else:
            ax_img.imshow(image)
        ax_img.set_title(
            f"{scene_name}\nanchor={example['anchor_activation']:.3f} "
            f"partner={example['partner_activation']:.3f} joint={example['joint_score']:.3f}",
            fontsize=9,
        )
        ax_img.axis("off")

        past = past_by_scene[scene_name]
        future = future_by_scene[scene_name]
        best_traj = best_traj_by_scene.get(scene_name)

        ax_traj.plot(past[:, 0], past[:, 1], color="black", marker="o", linewidth=1.4, label="Past")
        ax_traj.plot(future[:, 0], future[:, 1], color="green", marker="o", linewidth=2.0, label="GT future")
        if best_traj is not None:
            ax_traj.plot(best_traj[:, 0], best_traj[:, 1], color="tab:red", marker="o", linewidth=1.8, label="Best pred")
        ax_traj.scatter([past[-1, 0]], [past[-1, 1]], color="black", s=30)
        ax_traj.grid(True, alpha=0.25)
        ax_traj.axis("equal")
        ax_traj.set_xlabel("x (m)")
        ax_traj.set_ylabel("y (m)")
        ax_traj.set_title(
            f"intent={example['intent']} | motion={example['motion_label']} | speed={example['speed_label']}\n"
            f"v0={example['initial_speed_mps']:.2f} m/s | dv={example['speed_delta_mps']:.2f} m/s "
            f"| disp={example['total_disp_m']:.2f} m",
            fontsize=9,
        )
        ax_traj.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(pair.output_dir / f"pair_f{pair.anchor_idx:04d}_f{pair.partner_idx:04d}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze control-SAE neuron co-activation on best proposals per scene.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to extracted control activations (.pt).")
    parser.add_argument("--sae_ckpt", type=str, required=True, help="Path to the control SAE checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for CSV/JSON summaries and figures.")
    parser.add_argument("--data_dir", type=str, default=None, help="Waymo data directory for optional front-camera rendering.")
    parser.add_argument("--index_file", type=str, default=None, help="Index pickle path. Falls back to meta['index_file'].")
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--sae_batch_size", type=int, default=4096)
    parser.add_argument("--sample_unit", type=str, choices=["best"], default="best")
    parser.add_argument("--correlation_metric", type=str, choices=["pearson"], default="pearson")
    parser.add_argument("--top_neurons", type=int, default=20)
    parser.add_argument("--top_partners", type=int, default=10)
    parser.add_argument("--top_scenes", type=int, default=8)
    parser.add_argument("--min_active_count", type=int, default=25)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of scenes to analyze.")
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--stop_disp_thresh", type=float, default=1.0)
    parser.add_argument("--backward_disp_thresh", type=float, default=0.5)
    parser.add_argument("--speed_delta_thresh", type=float, default=0.5)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    blob = load_torch_file(str(data_path), map_location="cpu")
    if not isinstance(blob, dict):
        raise TypeError(f"Expected extracted activation dict, got {type(blob).__name__}")

    required_keys = ["control_pred", "scores_predicted", "past", "future", "intent", "names"]
    missing_keys = [key for key in required_keys if key not in blob]
    if missing_keys:
        available = ", ".join(sorted(blob.keys()))
        raise KeyError(
            f"Missing required key(s) {missing_keys} in extraction blob. Available keys: {available}"
        )

    control_pred = blob["control_pred"].float()
    scores_predicted = blob["scores_predicted"].float()
    past = blob["past"].float()
    future = blob["future"].float()
    intent = blob["intent"].long()
    names = list(blob["names"])
    trajectory_predicted = blob.get("trajectory_predicted")
    meta = blob.get("meta", {})

    if args.max_samples is not None:
        control_pred = control_pred[: args.max_samples]
        scores_predicted = scores_predicted[: args.max_samples]
        past = past[: args.max_samples]
        future = future[: args.max_samples]
        intent = intent[: args.max_samples]
        names = names[: args.max_samples]
        if trajectory_predicted is not None:
            trajectory_predicted = trajectory_predicted[: args.max_samples]

    num_scenes = control_pred.shape[0]
    if not (scores_predicted.shape[0] == past.shape[0] == future.shape[0] == intent.shape[0] == len(names) == num_scenes):
        raise ValueError("Mismatch between scene counts across control_pred, scores_predicted, past, future, intent, and names.")

    best_control, best_idx = select_best_controls(control_pred, scores_predicted)
    best_traj = select_best_trajectories(trajectory_predicted, best_idx, n_proposals=control_pred.shape[1])

    model = load_sae_model(Path(args.sae_ckpt), device)
    latents = encode_latents(model, best_control, batch_size=args.sae_batch_size, device=device)
    latent_dim = latents.shape[1]

    motion_labels, total_disp = derive_motion_labels(
        future=future,
        past=past,
        stop_disp_thresh=args.stop_disp_thresh,
        backward_disp_thresh=args.backward_disp_thresh,
    )
    speed_labels, initial_speed, final_speed, speed_delta = derive_speed_change_labels(
        future=future,
        past=past,
        dt=args.dt,
        speed_delta_thresh=args.speed_delta_thresh,
    )

    latents64 = latents.to(torch.float64)
    active_mask = latents64 > 0
    mean_activation = latents64.mean(dim=0)
    active_count = active_mask.sum(dim=0)
    active_rate = active_mask.to(torch.float64).mean(dim=0)
    corr = compute_correlation_matrix(latents64)

    valid_anchor_mask = (mean_activation > 0) & (active_count >= args.min_active_count)
    valid_anchor_idx = torch.nonzero(valid_anchor_mask, as_tuple=False).flatten()
    if valid_anchor_idx.numel() == 0:
        raise ValueError(
            "No anchor neurons satisfy the min activity filter. "
            f"Try lowering --min_active_count (currently {args.min_active_count})."
        )
    anchor_scores = mean_activation[valid_anchor_idx]
    sorted_anchor_idx = valid_anchor_idx[torch.argsort(anchor_scores, descending=True)]
    top_anchor_idx = sorted_anchor_idx[: args.top_neurons].tolist()

    top_neuron_rows: list[dict] = []
    pair_rows: list[dict] = []
    example_rows: list[dict] = []
    pair_selections: list[PairSelection] = []
    report = {
        "args": vars(args),
        "num_scenes": num_scenes,
        "latent_dim": latent_dim,
        "anchors": [],
    }

    for rank, anchor_idx in enumerate(top_anchor_idx, start=1):
        anchor_name = f"f{anchor_idx}"
        anchor_dir = output_dir / f"anchor_{anchor_idx:04d}"
        anchor_dir.mkdir(parents=True, exist_ok=True)
        anchor_active = active_mask[:, anchor_idx]
        anchor_active_count = int(active_count[anchor_idx].item())

        top_neuron_rows.append(
            {
                "anchor_rank": rank,
                "neuron_idx": anchor_idx,
                "mean_activation": float(mean_activation[anchor_idx].item()),
                "active_rate": float(active_rate[anchor_idx].item()),
                "active_count": anchor_active_count,
            }
        )

        candidate_corr = corr[anchor_idx].clone()
        candidate_corr[anchor_idx] = 0.0
        candidate_corr[torch.isnan(candidate_corr)] = 0.0
        positive_candidates = torch.nonzero(candidate_corr > 0, as_tuple=False).flatten()
        positive_sorted = positive_candidates[torch.argsort(candidate_corr[positive_candidates], descending=True)]
        top_partner_idx = positive_sorted[: args.top_partners].tolist()

        anchor_report = {
            "anchor_idx": anchor_idx,
            "anchor_rank": rank,
            "mean_activation": float(mean_activation[anchor_idx].item()),
            "active_rate": float(active_rate[anchor_idx].item()),
            "active_count": anchor_active_count,
            "summary_plot": str((anchor_dir / "summary.png").relative_to(output_dir)),
            "partners": [],
        }

        partner_rows_for_plot: list[dict] = []
        for partner_rank, partner_idx in enumerate(top_partner_idx, start=1):
            partner_active = active_mask[:, partner_idx]
            co_active = anchor_active & partner_active
            co_active_count = int(co_active.sum().item())
            partner_active_count = int(active_count[partner_idx].item())
            conditional_prob = float(co_active_count / anchor_active_count) if anchor_active_count > 0 else 0.0
            mean_partner_when_anchor = (
                float(latents64[anchor_active, partner_idx].mean().item()) if anchor_active_count > 0 else 0.0
            )
            pair_row = {
                "anchor_rank": rank,
                "anchor_idx": anchor_idx,
                "partner_rank": partner_rank,
                "partner_idx": partner_idx,
                "correlation": float(candidate_corr[partner_idx].item()),
                "anchor_mean_activation": float(mean_activation[anchor_idx].item()),
                "partner_mean_activation": float(mean_activation[partner_idx].item()),
                "anchor_active_rate": float(active_rate[anchor_idx].item()),
                "partner_active_rate": float(active_rate[partner_idx].item()),
                "anchor_active_count": anchor_active_count,
                "partner_active_count": partner_active_count,
                "co_active_count": co_active_count,
                "partner_given_anchor": conditional_prob,
                "mean_partner_activation_when_anchor_active": mean_partner_when_anchor,
            }
            pair_rows.append(pair_row)
            partner_rows_for_plot.append(pair_row)

            joint_score = torch.minimum(latents64[:, anchor_idx], latents64[:, partner_idx])
            selected_scene_idx = torch.nonzero(co_active, as_tuple=False).flatten()
            sorted_scene_idx = selected_scene_idx[torch.argsort(joint_score[selected_scene_idx], descending=True)]
            top_scene_idx = sorted_scene_idx[: args.top_scenes].tolist()

            selected_examples: list[dict] = []
            for scene_rank, scene_idx in enumerate(top_scene_idx, start=1):
                scene_name = str(names[scene_idx])
                row = {
                    "anchor_idx": anchor_idx,
                    "partner_idx": partner_idx,
                    "pair_rank": partner_rank,
                    "scene_rank": scene_rank,
                    "scene_name": scene_name,
                    "joint_score": float(joint_score[scene_idx].item()),
                    "anchor_activation": float(latents64[scene_idx, anchor_idx].item()),
                    "partner_activation": float(latents64[scene_idx, partner_idx].item()),
                    "intent": INTENT_NAMES.get(int(intent[scene_idx].item()), f"INTENT_{int(intent[scene_idx].item())}"),
                    "motion_label": MOTION_NAMES[int(motion_labels[scene_idx].item())],
                    "speed_label": SPEED_CHANGE_NAMES[int(speed_labels[scene_idx].item())],
                    "initial_speed_mps": float(initial_speed[scene_idx].item()),
                    "final_speed_mps": float(final_speed[scene_idx].item()),
                    "speed_delta_mps": float(speed_delta[scene_idx].item()),
                    "total_disp_m": float(total_disp[scene_idx].item()),
                    "anchor_dir": anchor_dir.name,
                    "montage_path": str(
                        (anchor_dir / f"pair_f{anchor_idx:04d}_f{partner_idx:04d}.png").relative_to(output_dir)
                    ),
                    "rendered": False,
                }
                example_rows.append(row)
                selected_examples.append(row)

            pair_selections.append(
                PairSelection(
                    anchor_idx=anchor_idx,
                    partner_idx=partner_idx,
                    correlation=float(candidate_corr[partner_idx].item()),
                    conditional_prob=conditional_prob,
                    output_dir=anchor_dir,
                    examples=selected_examples,
                )
            )
            anchor_report["partners"].append(
                {
                    "partner_rank": partner_rank,
                    "partner_idx": partner_idx,
                    "correlation": float(candidate_corr[partner_idx].item()),
                    "co_active_count": co_active_count,
                    "partner_given_anchor": conditional_prob,
                    "mean_partner_activation_when_anchor_active": mean_partner_when_anchor,
                    "montage_path": str(
                        (anchor_dir / f"pair_f{anchor_idx:04d}_f{partner_idx:04d}.png").relative_to(output_dir)
                    ),
                    "examples": selected_examples,
                }
            )

        save_anchor_summary_plot(anchor_idx=anchor_idx, partner_rows=partner_rows_for_plot, output_path=anchor_dir / "summary.png")
        report["anchors"].append(anchor_report)

    if args.data_dir is None:
        warnings.warn("No --data_dir provided; skipping front-camera scene rendering and leaving rendered=false in examples CSV.")
    else:
        index_candidate = args.index_file or meta.get("index_file")
        if not index_candidate:
            warnings.warn("No --index_file provided and extraction meta has no index_file; skipping scene rendering.")
        else:
            data_dir = Path(args.data_dir)
            index_path = resolve_path(str(index_candidate), data_path)
            needed_scene_names = {row["scene_name"] for row in example_rows}
            front_images = build_front_camera_lookup(needed_scene_names, data_dir=data_dir, index_path=index_path)

            past_by_scene = {str(names[idx]): past[idx, :, 0:2].numpy() for idx in range(num_scenes)}
            future_by_scene = {str(names[idx]): future[idx].numpy() for idx in range(num_scenes)}
            best_traj_by_scene = (
                {str(names[idx]): best_traj[idx].numpy() for idx in range(num_scenes)} if best_traj is not None else {}
            )

            for pair in pair_selections:
                render_pair_montage(
                    pair=pair,
                    front_images=front_images,
                    best_traj_by_scene=best_traj_by_scene,
                    past_by_scene=past_by_scene,
                    future_by_scene=future_by_scene,
                )

            rendered_pairs = {
                (pair.anchor_idx, pair.partner_idx)
                for pair in pair_selections
                if any(example["scene_name"] in front_images for example in pair.examples)
            }
            for row in example_rows:
                row["rendered"] = (row["anchor_idx"], row["partner_idx"]) in rendered_pairs

    write_csv(output_dir / "top_neurons.csv", top_neuron_rows)
    write_csv(output_dir / "coactivation_pairs.csv", pair_rows)
    write_csv(output_dir / "coactivation_examples.csv", example_rows)
    (output_dir / "coactivation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scenes analyzed: {num_scenes}")
    print(f"Latent dim: {latent_dim}")
    print(f"Top anchors saved to {output_dir / 'top_neurons.csv'}")
    print(f"Pair stats saved to {output_dir / 'coactivation_pairs.csv'}")
    print(f"Example stats saved to {output_dir / 'coactivation_examples.csv'}")
    print(f"JSON report saved to {output_dir / 'coactivation_report.json'}")


if __name__ == "__main__":
    main()
