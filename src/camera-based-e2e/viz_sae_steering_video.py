"""
Create supplemental SAE steering videos.

Each generated MP4 shows a continuous Waymo validation segment window with the
best-scoring SAE-steered trajectory projected into the front-camera panorama.
Narrative clips can also render a compact front-camera plus top-down trajectory
layout for supplemental videos.

Example:
    conda run -n amdgpu python src/camera-based-e2e/viz_sae_steering_video.py \
        --run_root /scratch/negishi/mgagvani/robotvision_scratch/sae_repro_20260515_123234 \
        --model_path src/camera-based-e2e/camera-e2e-epoch=04-val_loss=2.90.ckpt \
        --clusters 0 --window_frames 8 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from extract_planner_tok import load_model
from protos import e2e_pb2
from sae_utils import (
    DEFAULT_SAE_BLOCK,
    build_sae_from_checkpoint,
    load_sae_bundle,
    resolve_token_tensor,
)
from viz_camera_projection import (
    CAMERA_FRONT,
    FRONT3_CAMERAS,
    create_video,
    decode_image,
    draw_trajectory_on_image,
    get_camera_calibration,
    project_trajectory_to_image,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = Path("/scratch/negishi/mgagvani/robotvision_scratch/sae_repro_20260515_123234")
DEFAULT_MODEL_PATH = PROJECT_DIR / "camera-e2e-epoch=04-val_loss=2.90.ckpt"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "visualizations" / "paper" / "videos"
DEFAULT_STEERING_DIR = PROJECT_DIR / "visualizations" / "paper" / "steering"

INTENT_NAMES = {0: "UNKNOWN", 1: "STRAIGHT", 2: "LEFT", 3: "RIGHT"}
STAT_LABELS = {
    "final_lateral_disp": "Lateral displacement",
    "avg_curvature": "Curvature",
    "brake_mag": "Brake magnitude",
    "accel_mag": "Acceleration",
    "score_margin": "Score margin",
    "proposal_spread": "Proposal spread",
}

BASELINE_COLOR = (230, 38, 38)
STEERED_COLOR = (42, 214, 221)
LEFT_TURN_COLOR = (208, 72, 226)
RIGHT_TURN_COLOR = (42, 214, 221)
BASELINE_FREEZE_COLOR = (244, 130, 58)
GROUND_TRUTH_COLOR = (35, 190, 80)
PROPOSAL_COLOR = (255, 170, 170)
TEXT_COLOR = (35, 41, 52)
MUTED_TEXT_COLOR = (95, 103, 116)
PANEL_BG = (248, 249, 251)
PANEL_LINE = (206, 211, 219)
HEATMAP_BG = (20, 24, 31)


@dataclass(frozen=True)
class SegmentFrame:
    token_idx: int
    segment_hash: str
    segment_frame_idx: int
    filename: str
    start_byte: int
    byte_length: int


@dataclass(frozen=True)
class ClipSpec:
    cluster_id: int
    feature_idx: int
    stat_name: str
    seed_scene_idx: int
    segment_hash: str
    segment_start_frame_idx: int
    segment_end_frame_idx: int
    frames: list[SegmentFrame]
    cluster_features: list[int]


@dataclass(frozen=True)
class ActivationGroup:
    cluster_id: int
    label: str
    feature_indices: list[int]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_feature_list(text: str) -> list[int]:
    return [int(part) for part in text.split() if part]


def parse_clusters(text: str, available_clusters: Iterable[int]) -> list[int]:
    raw = text.strip().lower()
    if raw in {"all", "*"}:
        return sorted(set(available_clusters))
    return parse_int_list(text)


def parse_activation_group_specs(text: str) -> list[tuple[int, str]]:
    specs = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            cluster_text, label = part.split(":", 1)
            specs.append((int(cluster_text.strip()), label.strip()))
        else:
            cluster_id = int(part)
            specs.append((cluster_id, f"C{cluster_id}"))
    return specs


def parse_scene_idx_by_cluster(text: str | None) -> dict[int, int]:
    if text is None or not text.strip():
        return {}
    mapping = {}
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        cluster_text, scene_text = part.split(":", 1)
        mapping[int(cluster_text.strip())] = int(scene_text.strip())
    return mapping


def parse_optional_int_list(text: str | None) -> list[int]:
    if text is None or not text.strip():
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def analysis_dir(run_root: Path, block: int) -> Path:
    return run_root / "analysis" / f"block_{block}"


def cluster_top_scenes_path(run_root: Path, block: int, split: str, threshold: str) -> Path:
    return analysis_dir(run_root, block) / f"sae_cluster_top_scenes_block_{block}_{split}_thresh{threshold}.csv"


def cluster_summary_path(run_root: Path, block: int, split: str, threshold: str) -> Path:
    return analysis_dir(run_root, block) / f"sae_correlation_clusters_block_{block}_{split}_thresh{threshold}.csv"


def cluster_control_summary_path(output_dir: Path, block: int, split: str) -> Path:
    filename = f"cluster_control_summary_block_{block}_{split}.csv"
    default_path = DEFAULT_STEERING_DIR / filename
    if default_path.exists():
        return default_path
    return output_dir.parent / "steering" / filename


def run_control_summary_path(run_root: Path, block: int, split: str) -> Path:
    return analysis_dir(run_root, block) / f"sae_control_summary_block_{block}_{split}.csv"


def load_cluster_features(
    run_root: Path,
    block: int,
    split: str,
    threshold: str,
) -> dict[int, list[int]]:
    return {
        int(row["cluster_id"]): parse_feature_list(row["features"])
        for row in read_csv_rows(cluster_summary_path(run_root, block, split, threshold))
    }


def build_activation_groups(
    cluster_features: dict[int, list[int]],
    group_spec: str,
) -> list[ActivationGroup]:
    groups = []
    for cluster_id, label in parse_activation_group_specs(group_spec):
        if cluster_id not in cluster_features:
            raise KeyError(f"Activation cluster {cluster_id} not found in cluster summary")
        groups.append(
            ActivationGroup(
                cluster_id=cluster_id,
                label=label,
                feature_indices=cluster_features[cluster_id],
            )
        )
    return groups


def load_segment_index(index_file: Path) -> dict[str, list[tuple[int, str, int, int]]]:
    cache_path = index_file.with_suffix(".segments.pkl")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing segment cache {cache_path}. Run viz_camera_projection.py once to build it."
        )
    with cache_path.open("rb") as f:
        segments = pickle.load(f)
    return segments


def load_index(index_file: Path) -> list[tuple[str, int, int]]:
    with index_file.open("rb") as f:
        return pickle.load(f)


def build_entry_maps(
    index_entries: list[tuple[str, int, int]],
    segments: dict[str, list[tuple[int, str, int, int]]],
) -> tuple[
    dict[tuple[str, int, int], int],
    dict[tuple[str, int, int], tuple[str, int]],
]:
    entry_to_token_idx = {entry: idx for idx, entry in enumerate(index_entries)}
    entry_to_segment: dict[tuple[str, int, int], tuple[str, int]] = {}
    for segment_hash, frames in segments.items():
        for segment_frame_idx, filename, start_byte, byte_length in frames:
            entry_to_segment[(filename, start_byte, byte_length)] = (
                segment_hash,
                segment_frame_idx,
            )
    return entry_to_token_idx, entry_to_segment


def clamp_window_start(position: int, window_frames: int, segment_len: int) -> int:
    if window_frames >= segment_len:
        return 0
    return min(max(position - window_frames // 2, 0), segment_len - window_frames)


def make_window_for_scene(
    scene_idx: int,
    *,
    index_entries: list[tuple[str, int, int]],
    segments: dict[str, list[tuple[int, str, int, int]]],
    entry_to_token_idx: dict[tuple[str, int, int], int],
    entry_to_segment: dict[tuple[str, int, int], tuple[str, int]],
    window_frames: int,
) -> tuple[str, list[SegmentFrame]]:
    if scene_idx < 0 or scene_idx >= len(index_entries):
        raise IndexError(f"scene_idx={scene_idx} outside index range [0, {len(index_entries)})")

    entry = index_entries[scene_idx]
    if entry not in entry_to_segment:
        raise KeyError(f"Scene {scene_idx} did not map to any segment frame")

    segment_hash, _ = entry_to_segment[entry]
    segment_entries = segments[segment_hash]
    position = next(
        idx
        for idx, (_, filename, start_byte, byte_length) in enumerate(segment_entries)
        if (filename, start_byte, byte_length) == entry
    )
    start_pos = clamp_window_start(position, window_frames, len(segment_entries))
    window = segment_entries[start_pos : start_pos + min(window_frames, len(segment_entries))]

    frames = []
    for segment_frame_idx, filename, start_byte, byte_length in window:
        window_entry = (filename, start_byte, byte_length)
        frames.append(
            SegmentFrame(
                token_idx=entry_to_token_idx[window_entry],
                segment_hash=segment_hash,
                segment_frame_idx=segment_frame_idx,
                filename=filename,
                start_byte=start_byte,
                byte_length=byte_length,
            )
        )
    return segment_hash, frames


def encode_all_latents(
    *,
    sae,
    token_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for start in tqdm(range(0, token_tensor.shape[0], batch_size), desc="Encoding SAE latents"):
            x = token_tensor[start : start + batch_size].float().to(device)
            chunks.append(sae.encode(x).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def choose_bidirectional_scene_overrides(
    *,
    token_tensor: torch.Tensor,
    past_cpu: torch.Tensor,
    future_cpu: torch.Tensor,
    sae,
    cluster_features: dict[int, list[int]],
    activation_groups: list[ActivationGroup],
    clusters: list[int],
    window_frames: int,
    batch_size: int,
    device: torch.device,
    segments: dict[str, list[tuple[int, str, int, int]]],
    entry_to_token_idx: dict[tuple[str, int, int], int],
) -> dict[int, int]:
    if len(activation_groups) < 2:
        raise ValueError("--auto_bidirectional requires at least two --activation_clusters rows")

    z_all = encode_all_latents(
        sae=sae,
        token_tensor=token_tensor,
        batch_size=batch_size,
        device=device,
    )

    group_feature_lists = []
    for group in activation_groups[:2]:
        valid = [idx for idx in group.feature_indices if 0 <= idx < z_all.shape[1]]
        if not valid:
            raise ValueError(f"Activation group {group.label} has no valid feature indices")
        group_feature_lists.append(valid)

    rel_future_y = (future_cpu[:, -1, 1] - past_cpu[:, -1, 1]).numpy()
    best: tuple[float, str, int, int, float, float, float, float] | None = None

    for segment_hash, segment_entries in segments.items():
        token_indices = []
        for _, filename, start_byte, byte_length in segment_entries:
            token_idx = entry_to_token_idx.get((filename, start_byte, byte_length))
            if token_idx is not None:
                token_indices.append(token_idx)
        if len(token_indices) < 2:
            continue

        token_indices_arr = np.asarray(token_indices, dtype=np.int64)
        z_segment = z_all[token_indices_arr]
        left_score = z_segment[:, group_feature_lists[0]].sum(axis=1)
        right_score = z_segment[:, group_feature_lists[1]].sum(axis=1)
        y_segment = rel_future_y[token_indices_arr]

        n = len(token_indices)
        win = min(window_frames, n)
        for start in range(0, n - win + 1):
            end = start + win
            left_peak = float(left_score[start:end].max())
            right_peak = float(right_score[start:end].max())
            pos_y = float(y_segment[start:end].max())
            neg_y = float((-y_segment[start:end]).max())
            if pos_y <= 0.0 or neg_y <= 0.0:
                continue
            lateral_balance = min(pos_y, neg_y)
            lateral_range = pos_y + neg_y
            activation_balance = min(left_peak, right_peak)
            activation_total = left_peak + right_peak
            score = activation_balance + 0.25 * activation_total + 0.03 * lateral_range + 0.05 * lateral_balance
            if best is None or score > best[0]:
                center = start + win // 2
                best = (score, segment_hash, token_indices[center], win, left_peak, right_peak, pos_y, neg_y)

    if best is None:
        raise RuntimeError("Could not find a window with both positive and negative lateral motion")

    score, segment_hash, scene_idx, win, left_peak, right_peak, pos_y, neg_y = best
    print(
        "Auto bidirectional clip: "
        f"scene={scene_idx} segment={segment_hash[:12]}... frames={win} "
        f"left_peak={left_peak:.2f} right_peak={right_peak:.2f} "
        f"y+={pos_y:.2f} y-={neg_y:.2f} score={score:.2f}"
    )
    return {cluster_id: scene_idx for cluster_id in clusters}


def choose_clip_specs(
    *,
    run_root: Path,
    output_dir: Path,
    block: int,
    split: str,
    threshold: str,
    clusters_arg: str,
    clips_per_cluster: int,
    window_frames: int,
    scene_idx_override: int | None,
    scene_idx_by_cluster: dict[int, int],
    feature_idx_override: int | None,
    index_entries: list[tuple[str, int, int]],
    segments: dict[str, list[tuple[int, str, int, int]]],
    entry_to_token_idx: dict[tuple[str, int, int], int],
    entry_to_segment: dict[tuple[str, int, int], tuple[str, int]],
) -> list[ClipSpec]:
    cluster_features = load_cluster_features(run_root, block, split, threshold)

    cluster_control_rows = read_csv_rows(cluster_control_summary_path(output_dir, block, split))
    control_by_cluster = {
        int(row["cluster_id"]): row
        for row in cluster_control_rows
    }

    top_scene_rows = read_csv_rows(cluster_top_scenes_path(run_root, block, split, threshold))
    scenes_by_cluster: dict[int, list[dict[str, str]]] = {}
    for row in top_scene_rows:
        scenes_by_cluster.setdefault(int(row["cluster_id"]), []).append(row)
    for rows in scenes_by_cluster.values():
        rows.sort(key=lambda row: int(row["rank"]))

    clusters = parse_clusters(clusters_arg, cluster_features.keys())
    specs: list[ClipSpec] = []
    used_segments: set[str] = set()

    for cluster_id in clusters:
        if cluster_id not in cluster_features:
            raise KeyError(f"Cluster {cluster_id} not found in cluster summary")
        control_row = control_by_cluster.get(cluster_id)
        if control_row is None and feature_idx_override is None:
            raise KeyError(f"Cluster {cluster_id} not found in cluster control summary")

        default_feature = int(control_row["top_feature_idx"]) if control_row else cluster_features[cluster_id][0]
        feature_idx = feature_idx_override if feature_idx_override is not None else default_feature
        stat_name = control_row["dominant_control_stat"] if control_row else "proposal_spread"

        cluster_scene_override = scene_idx_by_cluster.get(cluster_id)
        if scene_idx_override is not None or cluster_scene_override is not None:
            scene_idx = scene_idx_override if scene_idx_override is not None else cluster_scene_override
            candidate_scene_rows = [{"scene_idx": str(scene_idx), "rank": "1"}]
        else:
            candidate_scene_rows = scenes_by_cluster.get(cluster_id, [])
            if cluster_id == 1 and candidate_scene_rows and "future_final_y" in candidate_scene_rows[0]:
                candidate_scene_rows = sorted(
                    candidate_scene_rows,
                    key=lambda row: (-float(row["future_final_y"]), int(row["rank"])),
                )
            elif cluster_id == 3 and candidate_scene_rows and "future_final_y" in candidate_scene_rows[0]:
                candidate_scene_rows = sorted(
                    candidate_scene_rows,
                    key=lambda row: (float(row["future_final_y"]), int(row["rank"])),
                )
        if not candidate_scene_rows:
            raise ValueError(f"No top scenes available for cluster {cluster_id}")

        selected_for_cluster = 0
        fallback_spec: ClipSpec | None = None
        for row in candidate_scene_rows:
            scene_idx = int(row["scene_idx"])
            segment_hash, frames = make_window_for_scene(
                scene_idx,
                index_entries=index_entries,
                segments=segments,
                entry_to_token_idx=entry_to_token_idx,
                entry_to_segment=entry_to_segment,
                window_frames=window_frames,
            )
            spec = ClipSpec(
                cluster_id=cluster_id,
                feature_idx=feature_idx,
                stat_name=stat_name,
                seed_scene_idx=scene_idx,
                segment_hash=segment_hash,
                segment_start_frame_idx=frames[0].segment_frame_idx,
                segment_end_frame_idx=frames[-1].segment_frame_idx,
                frames=frames,
                cluster_features=cluster_features[cluster_id],
            )
            if fallback_spec is None:
                fallback_spec = spec
            if segment_hash in used_segments and scene_idx_override is None:
                continue
            specs.append(spec)
            used_segments.add(segment_hash)
            selected_for_cluster += 1
            if selected_for_cluster >= clips_per_cluster:
                break

        if selected_for_cluster == 0 and fallback_spec is not None:
            specs.append(fallback_spec)
            used_segments.add(fallback_spec.segment_hash)

    return specs


def load_feature_scales(run_root: Path, block: int, split: str) -> dict[int, float]:
    scales: dict[int, float] = {}
    path = run_control_summary_path(run_root, block, split)
    for row in read_csv_rows(path):
        scales[int(row["feature_idx"])] = float(row["intervention_scale"])
    return scales


def reshape_trajectory(
    trajectory: torch.Tensor,
    *,
    num_proposals: int,
    horizon: int,
) -> torch.Tensor:
    if trajectory.ndim == 4:
        return trajectory
    return trajectory.view(trajectory.size(0), num_proposals, horizon, 2)


def selected_metrics(
    trajectories: torch.Tensor,
    scores: torch.Tensor,
    future: torch.Tensor,
) -> dict[str, torch.Tensor]:
    row_idx = torch.arange(trajectories.size(0), device=trajectories.device)
    selected_idx = scores.argmin(dim=1)
    selected = trajectories[row_idx, selected_idx]
    dist = torch.norm(selected - future, dim=-1)
    return {
        "selected_idx": selected_idx,
        "selected_traj": selected,
        "ade": dist.mean(dim=1),
        "fde": dist[:, -1],
    }


def run_steering(
    *,
    planner_model,
    sae,
    token_tensor: torch.Tensor,
    past_cpu: torch.Tensor,
    future_cpu: torch.Tensor,
    expected_trajectory_cpu: torch.Tensor,
    expected_scores_cpu: torch.Tensor,
    indices: list[int],
    feature_idx: int,
    feature_scale: float,
    alpha: float,
    batch_size: int,
    device: torch.device,
    num_proposals: int,
    horizon: int,
) -> dict[str, torch.Tensor | float]:
    base_x_cpu = token_tensor[indices].float()
    past = past_cpu[indices].float()
    future = future_cpu[indices].float()
    expected_trajectory = expected_trajectory_cpu[indices].float()
    expected_scores = expected_scores_cpu[indices].float()

    baseline_traj_chunks = []
    baseline_score_chunks = []
    steered_traj_chunks = []
    steered_score_chunks = []
    z_chunks = []
    max_baseline_diff = 0.0

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            base_x = base_x_cpu[start : start + batch_size].to(device)
            past_batch = past[start : start + batch_size].to(device)
            future_batch = future[start : start + batch_size].to(device)

            raw_out = planner_model.forward_from_planner_query_tok(base_x, past_batch)
            expected_traj_batch = expected_trajectory[start : start + batch_size].to(device)
            expected_scores_batch = expected_scores[start : start + batch_size].to(device)
            max_baseline_diff = max(
                max_baseline_diff,
                float((raw_out["trajectory"] - expected_traj_batch).abs().max().item()),
                float((raw_out["scores"] - expected_scores_batch).abs().max().item()),
            )
            raw_traj = reshape_trajectory(
                raw_out["trajectory"].detach().cpu(),
                num_proposals=num_proposals,
                horizon=horizon,
            )
            raw_scores = raw_out["scores"].detach().cpu()
            baseline_traj_chunks.append(raw_traj)
            baseline_score_chunks.append(raw_scores)

            z = sae.encode(base_x)
            z_chunks.append(z.detach().cpu())
            if alpha == 0.0:
                steered_out = raw_out
            else:
                z_mod = z.clone()
                z_mod[:, feature_idx] = (z_mod[:, feature_idx] + alpha * feature_scale).clamp_min(0.0)
                recon_query = sae.decode_to_input(z_mod, reference_x=base_x)
                steered_out = planner_model.forward_from_planner_query_tok(recon_query, past_batch)

            steered_traj_chunks.append(
                reshape_trajectory(
                    steered_out["trajectory"].detach().cpu(),
                    num_proposals=num_proposals,
                    horizon=horizon,
                )
            )
            steered_score_chunks.append(steered_out["scores"].detach().cpu())

            _ = future_batch

    baseline_traj = torch.cat(baseline_traj_chunks, dim=0)
    baseline_scores = torch.cat(baseline_score_chunks, dim=0)
    steered_traj = torch.cat(steered_traj_chunks, dim=0)
    steered_scores = torch.cat(steered_score_chunks, dim=0)
    z_window = torch.cat(z_chunks, dim=0)

    baseline = selected_metrics(baseline_traj, baseline_scores, future)
    steered = selected_metrics(steered_traj, steered_scores, future)
    return {
        "baseline_traj": baseline_traj,
        "baseline_scores": baseline_scores,
        "baseline_selected": baseline["selected_traj"],
        "baseline_selected_idx": baseline["selected_idx"],
        "baseline_ade": baseline["ade"],
        "baseline_fde": baseline["fde"],
        "steered_traj": steered_traj,
        "steered_scores": steered_scores,
        "steered_selected": steered["selected_traj"],
        "steered_selected_idx": steered["selected_idx"],
        "steered_ade": steered["ade"],
        "steered_fde": steered["fde"],
        "future": future,
        "past": past,
        "z_window": z_window,
        "baseline_check_max_abs": max_baseline_diff,
    }


def compute_alpha_sweep_trajectories(
    *,
    planner_model,
    sae,
    token_tensor: torch.Tensor,
    past_cpu: torch.Tensor,
    future_cpu: torch.Tensor,
    expected_trajectory_cpu: torch.Tensor,
    expected_scores_cpu: torch.Tensor,
    spec: ClipSpec,
    freeze_indices: list[int],
    feature_idx: int,
    feature_scale: float,
    max_alpha: float,
    freeze_frame_count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, list[np.ndarray]], list[float]]:
    if not freeze_indices:
        return {}, []

    sweep_alphas = np.linspace(0.0, max_alpha, freeze_frame_count, dtype=np.float32).tolist()
    token_indices = [spec.frames[idx].token_idx for idx in freeze_indices]
    trajectories_by_freeze = {idx: [] for idx in freeze_indices}

    for alpha_value in tqdm(sweep_alphas, desc="Computing alpha sweep"):
        sweep_data = run_steering(
            planner_model=planner_model,
            sae=sae,
            token_tensor=token_tensor,
            past_cpu=past_cpu,
            future_cpu=future_cpu,
            expected_trajectory_cpu=expected_trajectory_cpu,
            expected_scores_cpu=expected_scores_cpu,
            indices=token_indices,
            feature_idx=feature_idx,
            feature_scale=feature_scale,
            alpha=float(alpha_value),
            batch_size=batch_size,
            device=device,
            num_proposals=planner_model.n_proposals,
            horizon=planner_model.horizon,
        )
        selected = sweep_data["steered_selected"].numpy()
        for row_idx, freeze_idx in enumerate(freeze_indices):
            trajectories_by_freeze[freeze_idx].append(selected[row_idx])

    return trajectories_by_freeze, [float(value) for value in sweep_alphas]


class FrameReader:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._handles: dict[str, object] = {}

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def read(self, frame: SegmentFrame) -> dict:
        handle = self._handles.get(frame.filename)
        if handle is None:
            handle = open(self.data_root / frame.filename, "rb")
            self._handles[frame.filename] = handle

        handle.seek(frame.start_byte)
        protobuf = handle.read(frame.byte_length)
        e2e_frame = e2e_pb2.E2EDFrame()
        e2e_frame.ParseFromString(protobuf)

        cam_images = {}
        for img in e2e_frame.frame.images:
            if img.name in FRONT3_CAMERAS:
                cam_images[img.name] = decode_image(img.image)

        calibrations = {}
        for cam_id in FRONT3_CAMERAS:
            try:
                intr, extr, dist, width, height = get_camera_calibration(
                    e2e_frame.frame.context.camera_calibrations,
                    cam_id,
                )
                calibrations[cam_id] = {
                    "intrinsic": intr,
                    "extrinsic": extr,
                    "dist_coeffs": dist,
                    "width": width,
                    "height": height,
                }
            except ValueError:
                continue

        return {
            "name": e2e_frame.frame.context.name,
            "intent": int(e2e_frame.intent),
            "cam_images": cam_images,
            "calibrations": calibrations,
        }


def draw_multi_trajectory_on_cam(
    cam_img: np.ndarray,
    calib: dict,
    selected_trajectory: np.ndarray,
) -> np.ndarray:
    intrinsic = calib["intrinsic"]
    extrinsic = calib["extrinsic"]
    dist_coeffs = calib["dist_coeffs"]
    width = calib["width"]
    height = calib["height"]

    img = cam_img.copy()
    pixels = project_trajectory_to_image(selected_trajectory, intrinsic, extrinsic, dist_coeffs, width, height)
    return draw_route_ribbon_on_image(img, pixels)


def draw_route_ribbon_on_image(image: np.ndarray, trajectory_pixels: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(trajectory_pixels).any(axis=1)
    if valid.sum() < 2:
        return image

    route = np.round(trajectory_pixels[valid]).astype(np.int32)
    if len(route) < 2:
        return image

    overlay = image.copy()
    cv2.polylines(overlay, [route], False, STEERED_COLOR, 22, cv2.LINE_AA)
    return overlay


def densify_vehicle_trajectory(trajectory: np.ndarray, max_spacing_m: float = 0.75) -> np.ndarray:
    if trajectory.size == 0:
        return trajectory
    points = np.asarray(trajectory, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return points

    anchors = np.concatenate([np.zeros((1, 2), dtype=np.float32), points], axis=0)
    dense = [anchors[0]]
    for start, end in zip(anchors[:-1], anchors[1:]):
        dist = float(np.linalg.norm(end - start))
        steps = max(1, int(np.ceil(dist / max_spacing_m)))
        for step in range(1, steps + 1):
            dense.append(start + (end - start) * (step / steps))
    return np.stack(dense, axis=0)


def trajectory_edge_points(trajectory: np.ndarray, width_m: float) -> tuple[np.ndarray, np.ndarray]:
    if trajectory.shape[0] < 2:
        return trajectory, trajectory
    tangents = np.zeros_like(trajectory, dtype=np.float32)
    tangents[0] = trajectory[1] - trajectory[0]
    tangents[-1] = trajectory[-1] - trajectory[-2]
    if trajectory.shape[0] > 2:
        tangents[1:-1] = trajectory[2:] - trajectory[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-6)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    half_width = 0.5 * width_m
    return trajectory + normals * half_width, trajectory - normals * half_width


def project_ground_points_to_stitched_pano(
    points: np.ndarray,
    calibrations: dict[int, dict],
    panel_layout: list[tuple[int, int, int, float]],
    *,
    cam_id: int = CAMERA_FRONT,
) -> np.ndarray:
    layout = [item for item in panel_layout if item[0] == cam_id]
    if not layout or cam_id not in calibrations:
        return np.full((points.shape[0], 2), np.nan, dtype=np.float32)

    _, x_offset, _, scale = layout[0]
    calib = calibrations[cam_id]
    num_points = points.shape[0]
    trajectory_3d = np.hstack([points, np.zeros((num_points, 1), dtype=np.float32)])
    pts_vehicle = np.hstack([trajectory_3d, np.ones((num_points, 1), dtype=np.float32)])
    vehicle_to_camera = np.linalg.inv(calib["extrinsic"])
    pts_waymo_cam = (vehicle_to_camera @ pts_vehicle.T).T[:, :3]

    pts_opencv_cam = np.zeros_like(pts_waymo_cam)
    pts_opencv_cam[:, 0] = -pts_waymo_cam[:, 1]
    pts_opencv_cam[:, 1] = -pts_waymo_cam[:, 2]
    pts_opencv_cam[:, 2] = pts_waymo_cam[:, 0]

    valid_depth = pts_opencv_cam[:, 2] > 0.75
    pixels = np.full((num_points, 2), np.nan, dtype=np.float32)
    if not np.any(valid_depth):
        return pixels

    pts_2d, _ = cv2.projectPoints(
        pts_opencv_cam[valid_depth].reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        calib["intrinsic"],
        calib["dist_coeffs"],
    )
    pts_2d = pts_2d.reshape(-1, 2).astype(np.float32)
    pts_2d[:, 0] = x_offset + pts_2d[:, 0] * scale
    pts_2d[:, 1] = pts_2d[:, 1] * scale
    pixels[np.where(valid_depth)[0]] = pts_2d
    return pixels


def draw_ground_route_band(
    image: np.ndarray,
    left_pixels: np.ndarray,
    right_pixels: np.ndarray,
    *,
    color: tuple[int, int, int] = STEERED_COLOR,
    alpha: float = 0.72,
) -> np.ndarray:
    overlay = image.copy()
    filled_any = False
    max_step_px = 0.24 * image.shape[1]
    max_width_px = 0.42 * image.shape[1]
    max_quad_area = 0.10 * image.shape[0] * image.shape[1]
    min_ground_y = 0.28 * image.shape[0]

    for idx in range(left_pixels.shape[0] - 1):
        quad = np.array(
            [
                left_pixels[idx],
                left_pixels[idx + 1],
                right_pixels[idx + 1],
                right_pixels[idx],
            ],
            dtype=np.float32,
        )
        if np.isnan(quad).any() or not np.isfinite(quad).all():
            continue
        center_a = 0.5 * (left_pixels[idx] + right_pixels[idx])
        center_b = 0.5 * (left_pixels[idx + 1] + right_pixels[idx + 1])
        if np.linalg.norm(center_b - center_a) > max_step_px:
            continue
        if np.min(quad[:, 1]) < min_ground_y:
            continue
        width_a = float(np.linalg.norm(left_pixels[idx] - right_pixels[idx]))
        width_b = float(np.linalg.norm(left_pixels[idx + 1] - right_pixels[idx + 1]))
        if max(width_a, width_b) > max_width_px:
            continue
        area = abs(float(cv2.contourArea(quad.astype(np.float32))))
        if area > max_quad_area:
            continue
        cv2.fillConvexPoly(overlay, np.round(quad).astype(np.int32), color, cv2.LINE_AA)
        filled_any = True

    if not filled_any:
        return image
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def project_trajectory_to_stitched_pano(
    trajectory: np.ndarray,
    calibrations: dict[int, dict],
    panel_layout: list[tuple[int, int, int, float]],
    pano_width: int,
    pano_height: int,
) -> np.ndarray:
    per_step_candidates: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(len(trajectory))]
    panel_by_cam = {cam_id: (x_offset, panel_width) for cam_id, x_offset, panel_width, _ in panel_layout}
    for cam_id, x_offset, _, scale in panel_layout:
        if cam_id not in calibrations:
            continue
        calib = calibrations[cam_id]
        pixels = project_trajectory_to_image(
            trajectory,
            calib["intrinsic"],
            calib["extrinsic"],
            calib["dist_coeffs"],
            calib["width"],
            calib["height"],
        )
        valid = ~np.isnan(pixels).any(axis=1)
        for step_idx, pixel in enumerate(pixels):
            if not valid[step_idx]:
                continue
            stitched = np.array([x_offset + pixel[0] * scale, pixel[1] * scale], dtype=np.float32)
            if 0 <= stitched[0] < pano_width and 0 <= stitched[1] < pano_height:
                per_step_candidates[step_idx].append((cam_id, stitched))

    chosen_points = np.full((len(trajectory), 2), np.nan, dtype=np.float32)
    chosen_cams = np.full((len(trajectory),), -1, dtype=np.int32)
    def shared_seam_x(cam_a: int, cam_b: int) -> float | None:
        if cam_a == cam_b:
            return None
        if cam_a == CAMERA_FRONT or cam_b == CAMERA_FRONT:
            side_cam = cam_b if cam_a == CAMERA_FRONT else cam_a
            if side_cam not in panel_by_cam or CAMERA_FRONT not in panel_by_cam:
                return None
            side_x, side_w = panel_by_cam[side_cam]
            front_x, _ = panel_by_cam[CAMERA_FRONT]
            if side_x < front_x:
                return float(front_x)
            return float(side_x)
        return None

    def is_allowed_camera_transition(
        prev_cam: int,
        next_cam: int,
        prev_point: np.ndarray,
        next_point: np.ndarray,
    ) -> bool:
        if prev_cam < 0 or prev_cam == next_cam:
            return True
        seam_x = shared_seam_x(prev_cam, next_cam)
        if seam_x is None:
            return False
        _, front_w = panel_by_cam.get(CAMERA_FRONT, (0, pano_width / 3))
        seam_tol = 0.22 * front_w
        return abs(float(prev_point[0]) - seam_x) <= seam_tol and abs(float(next_point[0]) - seam_x) <= seam_tol

    previous = np.array([pano_width * 0.5, pano_height - 1], dtype=np.float32)
    previous_cam = -1
    for step_idx, candidates in enumerate(per_step_candidates):
        if not candidates:
            continue
        front_candidates = [(cam_id, point) for cam_id, point in candidates if cam_id == CAMERA_FRONT]
        candidate_pairs = front_candidates if front_candidates else candidates
        candidate_pairs = [
            (cam_id, point)
            for cam_id, point in candidate_pairs
            if is_allowed_camera_transition(previous_cam, cam_id, previous, point)
        ]
        if not candidate_pairs:
            break
        candidate_array = np.stack([point for _, point in candidate_pairs], axis=0)
        choice_idx = int(np.argmin(np.linalg.norm(candidate_array - previous[None, :], axis=1)))
        chosen_cams[step_idx] = candidate_pairs[choice_idx][0]
        chosen_points[step_idx] = candidate_pairs[choice_idx][1]
        previous = chosen_points[step_idx]
        previous_cam = int(chosen_cams[step_idx])

    anchor = np.array([pano_width * 0.5, pano_height - 1], dtype=np.float32)
    front_visible = np.where(chosen_cams == CAMERA_FRONT)[0]
    valid = np.where(~np.isnan(chosen_points).any(axis=1))[0]
    if front_visible.size:
        distances = np.linalg.norm(chosen_points[front_visible] - anchor[None, :], axis=1)
        start_idx = int(front_visible[int(np.argmin(distances))])
    elif valid.size:
        distances = np.linalg.norm(chosen_points[valid] - anchor[None, :], axis=1)
        start_idx = int(valid[int(np.argmin(distances))])
    else:
        return chosen_points

    route = [anchor]
    previous = anchor
    max_step_px = 0.24 * pano_width
    for point in chosen_points[start_idx:]:
        if np.isnan(point).any():
            continue
        if len(route) > 1 and np.linalg.norm(point - previous) > max_step_px:
            break
        route.append(point)
        previous = point
    return np.stack(route, axis=0) if len(route) >= 2 else chosen_points


def build_front3_pano_and_layout(
    cam_images: dict[int, np.ndarray],
    calibrations: dict[int, dict],
) -> tuple[np.ndarray, list[tuple[int, int, int, float]]]:
    if CAMERA_FRONT not in cam_images:
        raise RuntimeError("FRONT camera image missing")
    target_h = cam_images[CAMERA_FRONT].shape[0]
    panels = []
    panel_layout = []
    x_offset = 0
    for cam_id in FRONT3_CAMERAS:
        if cam_id not in cam_images:
            continue
        panel = cam_images[cam_id].copy()
        h, w = panel.shape[:2]
        scale = 1.0
        if h != target_h:
            scale = target_h / h
            panel = cv2.resize(panel, (int(w * scale), target_h), interpolation=cv2.INTER_LINEAR)
        if cam_id in calibrations:
            panel_layout.append((cam_id, x_offset, panel.shape[1], scale))
        panels.append(panel)
        x_offset += panel.shape[1]
    if not panels:
        raise RuntimeError("No front camera panels available")
    return np.concatenate(panels, axis=1), panel_layout


def draw_vehicle_band_on_pano(
    pano: np.ndarray,
    calibrations: dict[int, dict],
    panel_layout: list[tuple[int, int, int, float]],
    trajectory: np.ndarray,
    trajectory_width_m: float,
    *,
    color: tuple[int, int, int] = STEERED_COLOR,
    alpha: float = 0.72,
) -> np.ndarray:
    trajectory = densify_vehicle_trajectory(trajectory)
    left_edge, right_edge = trajectory_edge_points(trajectory, trajectory_width_m)
    left_pixels = project_ground_points_to_stitched_pano(
        left_edge,
        calibrations,
        panel_layout,
    )
    right_pixels = project_ground_points_to_stitched_pano(
        right_edge,
        calibrations,
        panel_layout,
    )
    return draw_ground_route_band(pano, left_pixels, right_pixels, color=color, alpha=alpha)


def build_front_camera_and_layout(
    cam_images: dict[int, np.ndarray],
    calibrations: dict[int, dict],
) -> tuple[np.ndarray, list[tuple[int, int, int, float]]]:
    if CAMERA_FRONT not in cam_images:
        raise RuntimeError("FRONT camera image missing")
    front = cam_images[CAMERA_FRONT].copy()
    panel_layout = []
    if CAMERA_FRONT in calibrations:
        panel_layout.append((CAMERA_FRONT, 0, front.shape[1], 1.0))
    return front, panel_layout


def draw_vehicle_band_on_front(
    front: np.ndarray,
    calibrations: dict[int, dict],
    panel_layout: list[tuple[int, int, int, float]],
    trajectory: np.ndarray,
    trajectory_width_m: float,
    *,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    if trajectory is None or trajectory.size == 0:
        return front
    return draw_vehicle_band_on_pano(
        front,
        calibrations,
        panel_layout,
        trajectory,
        trajectory_width_m,
        color=color,
        alpha=alpha,
    )


def trajectory_topdown_points(
    trajectory: np.ndarray,
    *,
    scale: float,
    origin: tuple[int, int],
) -> np.ndarray:
    if trajectory is None or trajectory.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    points = np.asarray(trajectory, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.int32)
    points = np.concatenate([np.zeros((1, 2), dtype=np.float32), points], axis=0)
    px = origin[0] - points[:, 1] * scale
    py = origin[1] - points[:, 0] * scale
    return np.round(np.stack([px, py], axis=1)).astype(np.int32)


def draw_topdown_trajectory_panel(
    *,
    height: int,
    width: int,
    baseline_trajectory: np.ndarray,
    steered_trajectory: np.ndarray | None,
    steered_color: tuple[int, int, int],
    phase_label: str | None = None,
    intervention_alpha: float | None = None,
    max_alpha: float | None = None,
) -> np.ndarray:
    panel = np.full((height, width, 3), (247, 249, 252), dtype=np.uint8)
    pad = max(28, int(round(0.04 * min(height, width))))
    plot_top = pad + 78
    plot_bottom = height - pad
    plot_left = pad
    plot_right = width - pad
    origin = ((plot_left + plot_right) // 2, plot_bottom - 18)

    trajectories = [np.asarray(baseline_trajectory, dtype=np.float32)]
    if steered_trajectory is not None:
        trajectories.append(np.asarray(steered_trajectory, dtype=np.float32))
    finite = [traj for traj in trajectories if traj.ndim == 2 and traj.shape[1] == 2 and traj.size]
    if finite:
        all_points = np.concatenate(finite, axis=0)
        max_forward = max(8.0, float(np.nanmax(all_points[:, 0])) + 4.0)
        max_lateral = max(4.0, float(np.nanmax(np.abs(all_points[:, 1]))) + 2.0)
    else:
        max_forward = 30.0
        max_lateral = 8.0
    scale = min(
        (plot_bottom - plot_top - 28) / max_forward,
        (plot_right - plot_left) / (2.0 * max_lateral),
    )

    cv2.rectangle(panel, (plot_left, plot_top), (plot_right, plot_bottom), (255, 255, 255), -1)
    cv2.rectangle(panel, (plot_left, plot_top), (plot_right, plot_bottom), (209, 216, 226), 1)
    for meters in range(10, int(max_forward) + 1, 10):
        y = int(round(origin[1] - meters * scale))
        if plot_top <= y <= plot_bottom:
            cv2.line(panel, (plot_left, y), (plot_right, y), (232, 236, 242), 1, cv2.LINE_AA)
            put_text(panel, f"{meters}m", (plot_left + 10, y - 6), scale=0.42, color=(118, 128, 142), thickness=1)
    cv2.line(panel, (origin[0], plot_top), (origin[0], plot_bottom), (226, 231, 238), 1, cv2.LINE_AA)

    baseline_pts = trajectory_topdown_points(baseline_trajectory, scale=scale, origin=origin)
    if len(baseline_pts) >= 2:
        cv2.polylines(panel, [baseline_pts], False, BASELINE_FREEZE_COLOR, 10, cv2.LINE_AA)
        cv2.polylines(panel, [baseline_pts], False, (255, 222, 191), 3, cv2.LINE_AA)
    if steered_trajectory is not None:
        steered_pts = trajectory_topdown_points(steered_trajectory, scale=scale, origin=origin)
        if len(steered_pts) >= 2:
            cv2.polylines(panel, [steered_pts], False, steered_color, 10, cv2.LINE_AA)
            cv2.polylines(panel, [steered_pts], False, tuple(min(255, c + 42) for c in steered_color), 3, cv2.LINE_AA)

    cv2.circle(panel, origin, 10, (28, 35, 45), -1, cv2.LINE_AA)
    cv2.arrowedLine(
        panel,
        (origin[0], origin[1] - 4),
        (origin[0], max(plot_top + 18, origin[1] - int(round(5 * scale)))),
        (28, 35, 45),
        3,
        cv2.LINE_AA,
        tipLength=0.28,
    )

    title = phase_label if phase_label else "trajectory"
    put_text(panel, title, (pad, pad + 18), scale=0.78, color=(20, 27, 36), thickness=2)
    legend_y = pad + 50
    cv2.line(panel, (pad, legend_y), (pad + 52, legend_y), BASELINE_FREEZE_COLOR, 8, cv2.LINE_AA)
    put_text(panel, "original", (pad + 66, legend_y + 6), scale=0.48, color=(62, 70, 84), thickness=1)
    if steered_trajectory is not None:
        cv2.line(panel, (pad + 190, legend_y), (pad + 242, legend_y), steered_color, 8, cv2.LINE_AA)
        put_text(panel, "steered", (pad + 256, legend_y + 6), scale=0.48, color=(62, 70, 84), thickness=1)
    if intervention_alpha is not None and max_alpha is not None:
        text = f"alpha {intervention_alpha:.1f}/{max_alpha:.1f}"
        put_text(panel, text, (width - pad - 170, pad + 18), scale=0.48, color=(82, 92, 108), thickness=1)
        bar_w = width - 2 * pad
        bar_y = height - max(12, pad // 2)
        cv2.rectangle(panel, (pad, bar_y), (pad + bar_w, bar_y + 5), (214, 220, 229), -1)
        filled = int(round(bar_w * float(intervention_alpha) / max(float(max_alpha), 1e-6)))
        cv2.rectangle(panel, (pad, bar_y), (pad + filled, bar_y + 5), steered_color, -1)
    return panel


def make_front_topdown_frame(
    *,
    cam_images: dict[int, np.ndarray],
    calibrations: dict[int, dict],
    baseline_trajectory: np.ndarray,
    steered_trajectory: np.ndarray | None,
    trajectory_width_m: float,
    steered_color: tuple[int, int, int],
    phase_label: str | None = None,
    intervention_alpha: float | None = None,
    max_alpha: float | None = None,
    draw_steered_on_camera: bool = True,
) -> np.ndarray:
    front, panel_layout = build_front_camera_and_layout(cam_images, calibrations)
    front = draw_vehicle_band_on_front(
        front,
        calibrations,
        panel_layout,
        baseline_trajectory,
        trajectory_width_m,
        color=BASELINE_FREEZE_COLOR,
        alpha=0.42,
    )
    if draw_steered_on_camera and steered_trajectory is not None:
        front = draw_vehicle_band_on_front(
            front,
            calibrations,
            panel_layout,
            steered_trajectory,
            trajectory_width_m,
            color=steered_color,
            alpha=0.58,
        )
    panel_w = max(520, int(round(front.shape[0] * 0.78)))
    topdown = draw_topdown_trajectory_panel(
        height=front.shape[0],
        width=panel_w,
        baseline_trajectory=baseline_trajectory,
        steered_trajectory=steered_trajectory,
        steered_color=steered_color,
        phase_label=phase_label,
        intervention_alpha=intervention_alpha,
        max_alpha=max_alpha,
    )
    return np.concatenate([front, topdown], axis=1)


def draw_freeze_labels(
    image: np.ndarray,
    *,
    freeze_elapsed: int,
    freeze_total: int,
    intervention_alpha: float,
    max_alpha: float,
) -> np.ndarray:
    out = image.copy()
    pad = 24
    box_w = 520
    box_h = 118
    cv2.rectangle(out, (pad, pad), (pad + box_w, pad + box_h), (18, 24, 32), -1)
    cv2.rectangle(out, (pad, pad), (pad + box_w, pad + box_h), (238, 242, 248), 2)
    put_text(out, "SAE intervention", (pad + 22, pad + 34), scale=0.72, color=(245, 248, 252), thickness=2)
    put_text(
        out,
        f"alpha {intervention_alpha:.1f}/{max_alpha:.1f}",
        (pad + box_w - 164, pad + 34),
        scale=0.5,
        color=(210, 218, 229),
        thickness=1,
    )
    cv2.line(out, (pad + 26, pad + 66), (pad + 92, pad + 66), (244, 130, 58), 12, cv2.LINE_AA)
    put_text(out, "baseline", (pad + 112, pad + 72), scale=0.58, color=(235, 239, 245), thickness=1)
    cv2.line(out, (pad + 260, pad + 66), (pad + 326, pad + 66), STEERED_COLOR, 12, cv2.LINE_AA)
    put_text(out, "steered", (pad + 346, pad + 72), scale=0.58, color=(235, 239, 245), thickness=1)
    progress_w = box_w - 44
    progress_x = pad + 22
    progress_y = pad + box_h - 22
    cv2.rectangle(out, (progress_x, progress_y), (progress_x + progress_w, progress_y + 5), (75, 85, 99), -1)
    filled = int(round(progress_w * (freeze_elapsed + 1) / max(1, freeze_total)))
    cv2.rectangle(out, (progress_x, progress_y), (progress_x + filled, progress_y + 5), STEERED_COLOR, -1)
    return out


def draw_turn_comparison_freeze_labels(
    image: np.ndarray,
    *,
    freeze_elapsed: int,
    freeze_total: int,
    intervention_alpha: float,
    max_alpha: float,
) -> np.ndarray:
    out = image.copy()
    pad = 24
    box_w = 650
    box_h = 132
    cv2.rectangle(out, (pad, pad), (pad + box_w, pad + box_h), (18, 24, 32), -1)
    cv2.rectangle(out, (pad, pad), (pad + box_w, pad + box_h), (238, 242, 248), 2)
    put_text(out, "SAE turn interventions", (pad + 22, pad + 34), scale=0.7, color=(245, 248, 252), thickness=2)
    put_text(
        out,
        f"alpha {intervention_alpha:.1f}/{max_alpha:.1f}",
        (pad + box_w - 170, pad + 34),
        scale=0.5,
        color=(210, 218, 229),
        thickness=1,
    )
    legend_y = pad + 70
    legend = [
        ("baseline", BASELINE_FREEZE_COLOR),
        ("left turn", LEFT_TURN_COLOR),
        ("right turn", RIGHT_TURN_COLOR),
    ]
    x = pad + 28
    for label, color in legend:
        cv2.line(out, (x, legend_y - 4), (x + 54, legend_y - 4), color, 10, cv2.LINE_AA)
        put_text(out, label, (x + 72, legend_y + 2), scale=0.52, color=(235, 239, 245), thickness=1)
        x += 205

    progress_w = box_w - 44
    progress_x = pad + 22
    progress_y = pad + box_h - 24
    cv2.rectangle(out, (progress_x, progress_y), (progress_x + progress_w, progress_y + 5), (75, 85, 99), -1)
    filled = int(round(progress_w * (freeze_elapsed + 1) / max(1, freeze_total)))
    cv2.rectangle(out, (progress_x, progress_y), (progress_x + filled, progress_y + 5), RIGHT_TURN_COLOR, -1)
    return out


def draw_single_turn_freeze_labels(
    image: np.ndarray,
    *,
    label: str,
    color: tuple[int, int, int],
    freeze_elapsed: int,
    freeze_total: int,
    intervention_alpha: float,
    max_alpha: float,
) -> np.ndarray:
    out = image.copy()
    pad = 24
    box_w = 570
    box_h = 132
    cv2.rectangle(out, (pad, pad), (pad + box_w, pad + box_h), (18, 24, 32), -1)
    cv2.rectangle(out, (pad, pad), (pad + box_w, pad + box_h), (238, 242, 248), 2)
    put_text(out, f"SAE {label}", (pad + 22, pad + 34), scale=0.7, color=(245, 248, 252), thickness=2)
    put_text(
        out,
        f"alpha {intervention_alpha:.1f}/{max_alpha:.1f}",
        (pad + box_w - 170, pad + 34),
        scale=0.5,
        color=(210, 218, 229),
        thickness=1,
    )
    legend_y = pad + 72
    cv2.line(out, (pad + 28, legend_y - 4), (pad + 82, legend_y - 4), BASELINE_FREEZE_COLOR, 10, cv2.LINE_AA)
    put_text(out, "baseline", (pad + 100, legend_y + 2), scale=0.52, color=(235, 239, 245), thickness=1)
    cv2.line(out, (pad + 260, legend_y - 4), (pad + 314, legend_y - 4), color, 10, cv2.LINE_AA)
    put_text(out, label, (pad + 332, legend_y + 2), scale=0.52, color=(235, 239, 245), thickness=1)

    progress_w = box_w - 44
    progress_x = pad + 22
    progress_y = pad + box_h - 24
    cv2.rectangle(out, (progress_x, progress_y), (progress_x + progress_w, progress_y + 5), (75, 85, 99), -1)
    filled = int(round(progress_w * (freeze_elapsed + 1) / max(1, freeze_total)))
    cv2.rectangle(out, (progress_x, progress_y), (progress_x + filled, progress_y + 5), color, -1)
    return out


def stitch_front3_multi(
    cam_images: dict[int, np.ndarray],
    calibrations: dict[int, dict],
    selected_trajectory: np.ndarray | None,
    trajectory_width_m: float,
) -> np.ndarray:
    pano, panel_layout = build_front3_pano_and_layout(cam_images, calibrations)
    if selected_trajectory is None:
        return pano
    return draw_vehicle_band_on_pano(
        pano,
        calibrations,
        panel_layout,
        selected_trajectory,
        trajectory_width_m,
        color=STEERED_COLOR,
        alpha=0.72,
    )


def stitch_front3_intervention_freeze(
    cam_images: dict[int, np.ndarray],
    calibrations: dict[int, dict],
    baseline_trajectory: np.ndarray,
    steered_trajectory: np.ndarray,
    trajectory_width_m: float,
    *,
    freeze_elapsed: int,
    freeze_total: int,
    intervention_alpha: float,
    max_alpha: float,
) -> np.ndarray:
    pano, panel_layout = build_front3_pano_and_layout(cam_images, calibrations)
    pano = draw_vehicle_band_on_pano(
        pano,
        calibrations,
        panel_layout,
        baseline_trajectory,
        trajectory_width_m,
        color=BASELINE_FREEZE_COLOR,
        alpha=0.58,
    )
    pano = draw_vehicle_band_on_pano(
        pano,
        calibrations,
        panel_layout,
        steered_trajectory,
        trajectory_width_m,
        color=STEERED_COLOR,
        alpha=0.64,
    )
    return draw_freeze_labels(
        pano,
        freeze_elapsed=freeze_elapsed,
        freeze_total=freeze_total,
        intervention_alpha=intervention_alpha,
        max_alpha=max_alpha,
    )


def stitch_front3_turn_comparison_freeze(
    cam_images: dict[int, np.ndarray],
    calibrations: dict[int, dict],
    baseline_trajectory: np.ndarray,
    left_trajectory: np.ndarray,
    right_trajectory: np.ndarray,
    trajectory_width_m: float,
    *,
    freeze_elapsed: int,
    freeze_total: int,
    intervention_alpha: float,
    max_alpha: float,
) -> np.ndarray:
    pano, panel_layout = build_front3_pano_and_layout(cam_images, calibrations)
    pano = draw_vehicle_band_on_pano(
        pano,
        calibrations,
        panel_layout,
        baseline_trajectory,
        trajectory_width_m,
        color=BASELINE_FREEZE_COLOR,
        alpha=0.42,
    )
    pano = draw_vehicle_band_on_pano(
        pano,
        calibrations,
        panel_layout,
        right_trajectory,
        trajectory_width_m,
        color=RIGHT_TURN_COLOR,
        alpha=0.52,
    )
    pano = draw_vehicle_band_on_pano(
        pano,
        calibrations,
        panel_layout,
        left_trajectory,
        trajectory_width_m,
        color=LEFT_TURN_COLOR,
        alpha=0.58,
    )
    return draw_turn_comparison_freeze_labels(
        pano,
        freeze_elapsed=freeze_elapsed,
        freeze_total=freeze_total,
        intervention_alpha=intervention_alpha,
        max_alpha=max_alpha,
    )


def stitch_front3_single_turn_freeze(
    cam_images: dict[int, np.ndarray],
    calibrations: dict[int, dict],
    baseline_trajectory: np.ndarray,
    intervention_trajectory: np.ndarray,
    trajectory_width_m: float,
    *,
    label: str,
    color: tuple[int, int, int],
    freeze_elapsed: int,
    freeze_total: int,
    intervention_alpha: float,
    max_alpha: float,
) -> np.ndarray:
    pano, panel_layout = build_front3_pano_and_layout(cam_images, calibrations)
    pano = draw_vehicle_band_on_pano(
        pano,
        calibrations,
        panel_layout,
        baseline_trajectory,
        trajectory_width_m,
        color=BASELINE_FREEZE_COLOR,
        alpha=0.42,
    )
    pano = draw_vehicle_band_on_pano(
        pano,
        calibrations,
        panel_layout,
        intervention_trajectory,
        trajectory_width_m,
        color=color,
        alpha=0.62,
    )
    return draw_single_turn_freeze_labels(
        pano,
        label=label,
        color=color,
        freeze_elapsed=freeze_elapsed,
        freeze_total=freeze_total,
        intervention_alpha=intervention_alpha,
        max_alpha=max_alpha,
    )


def stitch_front3_raw(cam_images: dict[int, np.ndarray]) -> np.ndarray:
    if CAMERA_FRONT not in cam_images:
        raise RuntimeError("FRONT camera image missing")
    target_h = cam_images[CAMERA_FRONT].shape[0]
    panels = []
    for cam_id in FRONT3_CAMERAS:
        if cam_id not in cam_images:
            continue
        panel = cam_images[cam_id]
        h, w = panel.shape[:2]
        if h != target_h:
            scale = target_h / h
            panel = cv2.resize(panel, (int(w * scale), target_h), interpolation=cv2.INTER_LINEAR)
        panels.append(panel)
    if not panels:
        raise RuntimeError("No front camera panels available")
    return np.concatenate(panels, axis=1)


def put_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.75,
    color: tuple[int, int, int] = TEXT_COLOR,
    thickness: int = 2,
) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_sparkline(
    panel: np.ndarray,
    values: np.ndarray,
    *,
    current_idx: int,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    cv2.rectangle(panel, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), PANEL_LINE, 1)
    put_text(panel, label, (x, y - 10), scale=0.52, color=MUTED_TEXT_COLOR, thickness=1)

    if values.size == 0:
        return
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    denom = max(max_v - min_v, 1e-6)
    points = []
    for idx, value in enumerate(values):
        px = x + int(round(idx * (w - 1) / max(1, values.size - 1)))
        py = y + h - int(round((float(value) - min_v) / denom * (h - 1)))
        points.append((px, py))
    if len(points) >= 2:
        cv2.polylines(panel, [np.array(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    cursor_x = x + int(round(current_idx * (w - 1) / max(1, values.size - 1)))
    cv2.line(panel, (cursor_x, y), (cursor_x, y + h), (45, 49, 57), 1, cv2.LINE_AA)
    put_text(panel, f"{values[current_idx]:.2f}", (x + w - 86, y + h - 10), scale=0.5, color=color, thickness=1)


def draw_legend(panel: np.ndarray, x: int, y: int) -> None:
    items = [
        ("baseline", BASELINE_COLOR),
        ("steered", STEERED_COLOR),
        ("ground truth", GROUND_TRUTH_COLOR),
    ]
    for idx, (label, color) in enumerate(items):
        yy = y + idx * 32
        cv2.line(panel, (x, yy), (x + 34, yy), color, 5, cv2.LINE_AA)
        put_text(panel, label, (x + 46, yy + 7), scale=0.55, color=TEXT_COLOR, thickness=1)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    values = values.astype(np.float32)
    row_min = values.min(axis=1, keepdims=True)
    row_max = values.max(axis=1, keepdims=True)
    return (values - row_min) / np.maximum(row_max - row_min, 1e-6)


def colorize_heatmap(values: np.ndarray) -> np.ndarray:
    scaled = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def compute_group_activation_rows(
    z_window: np.ndarray,
    activation_groups: list[ActivationGroup],
) -> tuple[np.ndarray, np.ndarray]:
    raw_rows = []
    for group in activation_groups:
        valid = [idx for idx in group.feature_indices if 0 <= idx < z_window.shape[1]]
        if valid:
            raw_rows.append(z_window[:, valid].sum(axis=1))
        else:
            raw_rows.append(np.zeros(z_window.shape[0], dtype=np.float32))
    raw = np.stack(raw_rows, axis=0).astype(np.float32) if raw_rows else np.zeros((0, z_window.shape[0]), dtype=np.float32)
    return raw, normalize_rows(raw)


def make_filmstrip_thumbnails(
    *,
    spec: ClipSpec,
    data_root: Path,
    count: int,
) -> tuple[list[np.ndarray], list[int]]:
    if count <= 0:
        return [], []
    sample_indices = np.linspace(0, len(spec.frames) - 1, min(count, len(spec.frames))).round().astype(int).tolist()
    thumbnails = []
    reader = FrameReader(data_root)
    try:
        for idx in sample_indices:
            sample = reader.read(spec.frames[idx])
            thumbnails.append(stitch_front3_raw(sample["cam_images"]))
    finally:
        reader.close()
    return thumbnails, sample_indices


def draw_label_box(img: np.ndarray, text: str, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), (238, 242, 238), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (58, 65, 74), 1)
    font_scale = min(0.82, max(0.48, 0.82 * w / max(1, 18 * len(text))))
    put_text(img, text, (x + 12, y + int(h * 0.68)), scale=font_scale, color=(0, 0, 0), thickness=2)


def make_left_right_timeline_strip(
    *,
    width: int,
    spec: ClipSpec,
    frame_local_idx: int,
    frame: SegmentFrame,
    sample_name: str,
    intent: int,
    alpha: float,
    feature_scale: float,
    activation_groups: list[ActivationGroup],
    group_activation_norm: np.ndarray,
    active_group_counts: np.ndarray,
    filmstrip_thumbnails: list[np.ndarray],
    filmstrip_indices: list[int],
    baseline_ade: np.ndarray,
    steered_ade: np.ndarray,
    baseline_fde: np.ndarray,
    steered_fde: np.ndarray,
) -> np.ndarray:
    strip_h = 270
    strip = np.full((strip_h, width, 3), HEATMAP_BG, dtype=np.uint8)
    _ = (spec, frame, sample_name, intent, alpha, feature_scale, baseline_ade, steered_ade, baseline_fde, steered_fde)

    pad = 16
    label_w = 132
    gap = 8

    film_y = 16
    film_h = 96
    film_x = pad + label_w + gap
    film_w = width - film_x - pad
    if filmstrip_thumbnails:
        n = len(filmstrip_thumbnails)
        thumb_w = max(1, (film_w - gap * (n - 1)) // n)
        nearest_thumb = int(np.argmin(np.abs(np.array(filmstrip_indices) - frame_local_idx)))
        for thumb_idx, thumb in enumerate(filmstrip_thumbnails):
            x = film_x + thumb_idx * (thumb_w + gap)
            resized = cv2.resize(thumb, (thumb_w, film_h), interpolation=cv2.INTER_AREA)
            strip[film_y : film_y + film_h, x : x + thumb_w] = resized
            border_color = (255, 255, 255) if thumb_idx == nearest_thumb else (88, 96, 108)
            thickness = 4 if thumb_idx == nearest_thumb else 1
            cv2.rectangle(strip, (x, film_y), (x + thumb_w, film_y + film_h), border_color, thickness)
            if thumb_idx == 0:
                put_text(strip, "t=0", (x + 8, film_y + film_h - 10), scale=0.56, color=(255, 255, 255), thickness=2)
            elif thumb_idx == n - 1:
                put_text(strip, "t=1", (x + 8, film_y + film_h - 10), scale=0.56, color=(255, 255, 255), thickness=2)
    draw_label_box(strip, "Frame", pad, film_y, label_w, film_h)

    row_y = film_y + film_h + 14
    row_h = 40
    timeline_x = film_x
    timeline_w = film_w
    cursor_x = timeline_x + int(round(frame_local_idx * (timeline_w - 1) / max(1, len(spec.frames) - 1)))
    for row_idx, group in enumerate(activation_groups):
        y = row_y + row_idx * (row_h + 8)
        label = group.label
        if row_idx < active_group_counts.shape[0]:
            label = f"{label}  {int(active_group_counts[row_idx, frame_local_idx])}/{len(group.feature_indices)}"
        draw_label_box(strip, label, pad, y, label_w, row_h)
        if group_activation_norm.shape[0] > row_idx:
            heat = colorize_heatmap(group_activation_norm[row_idx : row_idx + 1])
            heat = cv2.resize(heat, (timeline_w, row_h), interpolation=cv2.INTER_NEAREST)
        else:
            heat = np.full((row_h, timeline_w, 3), (42, 48, 58), dtype=np.uint8)
        strip[y : y + row_h, timeline_x : timeline_x + timeline_w] = heat
        cv2.rectangle(strip, (timeline_x, y), (timeline_x + timeline_w, y + row_h), (218, 224, 234), 1)

    cursor_top = film_y + film_h - 2
    cursor_bottom = min(strip_h - 42, row_y + len(activation_groups) * (row_h + 8))
    cv2.line(strip, (cursor_x, cursor_top), (cursor_x, cursor_bottom), (255, 255, 255), 3, cv2.LINE_AA)
    put_text(strip, "t", (pad, strip_h - 20), scale=0.74, color=(255, 255, 255), thickness=2)
    cv2.arrowedLine(strip, (pad + 34, strip_h - 25), (pad + 150, strip_h - 25), (255, 255, 255), 4, cv2.LINE_AA, tipLength=0.25)
    return strip


def make_heatmap_strip(
    *,
    width: int,
    spec: ClipSpec,
    frame_local_idx: int,
    frame: SegmentFrame,
    sample_name: str,
    intent: int,
    alpha: float,
    feature_scale: float,
    cluster_heatmap: np.ndarray,
    selected_feature_row: int | None,
    active_cluster_counts: np.ndarray,
    baseline_ade: np.ndarray,
    steered_ade: np.ndarray,
    baseline_fde: np.ndarray,
    steered_fde: np.ndarray,
) -> np.ndarray:
    strip_h = 250
    strip = np.full((strip_h, width, 3), HEATMAP_BG, dtype=np.uint8)
    pad = 28
    header_y = 34
    put_text(
        strip,
        f"C{spec.cluster_id}  feature {spec.feature_idx}  {STAT_LABELS.get(spec.stat_name, spec.stat_name)}",
        (pad, header_y),
        scale=0.68,
        color=(244, 246, 249),
        thickness=2,
    )
    put_text(
        strip,
        f"alpha={alpha:g} scale={feature_scale:.3f}  scene={frame.token_idx} frame={frame.segment_frame_idx}  intent={INTENT_NAMES.get(intent, str(intent))}",
        (pad, header_y + 30),
        scale=0.48,
        color=(185, 193, 205),
        thickness=1,
    )

    delta_ade = steered_ade[frame_local_idx] - baseline_ade[frame_local_idx]
    delta_fde = steered_fde[frame_local_idx] - baseline_fde[frame_local_idx]
    metric_color = (118, 217, 151) if delta_ade <= 0 else (245, 138, 138)
    metric_text = (
        f"ADE {baseline_ade[frame_local_idx]:.2f}->{steered_ade[frame_local_idx]:.2f}m "
        f"({delta_ade:+.2f})   FDE {baseline_fde[frame_local_idx]:.2f}->{steered_fde[frame_local_idx]:.2f}m "
        f"({delta_fde:+.2f})   active {int(active_cluster_counts[frame_local_idx])}/{len(spec.cluster_features)}"
    )
    put_text(strip, metric_text, (pad, header_y + 60), scale=0.48, color=metric_color, thickness=1)

    heatmap_x = pad
    heatmap_y = 106
    heatmap_w = width - 2 * pad
    heatmap_h = 110
    if cluster_heatmap.size:
        heatmap_img = colorize_heatmap(cluster_heatmap)
        heatmap_img = cv2.resize(heatmap_img, (heatmap_w, heatmap_h), interpolation=cv2.INTER_NEAREST)
    else:
        heatmap_img = np.full((heatmap_h, heatmap_w, 3), (42, 48, 58), dtype=np.uint8)
    strip[heatmap_y : heatmap_y + heatmap_h, heatmap_x : heatmap_x + heatmap_w] = heatmap_img

    cv2.rectangle(strip, (heatmap_x, heatmap_y), (heatmap_x + heatmap_w, heatmap_y + heatmap_h), (218, 224, 234), 1)
    cursor_x = heatmap_x + int(round(frame_local_idx * (heatmap_w - 1) / max(1, len(spec.frames) - 1)))
    cv2.line(strip, (cursor_x, heatmap_y - 8), (cursor_x, heatmap_y + heatmap_h + 8), (255, 255, 255), 2, cv2.LINE_AA)

    if selected_feature_row is not None and cluster_heatmap.shape[0] > 0:
        row_y = heatmap_y + int(round((selected_feature_row + 0.5) * heatmap_h / cluster_heatmap.shape[0]))
        cv2.line(strip, (heatmap_x, row_y), (heatmap_x + heatmap_w, row_y), (236, 72, 153), 2, cv2.LINE_AA)
        put_text(strip, f"f{spec.feature_idx}", (heatmap_x + heatmap_w - 72, max(heatmap_y + 18, row_y - 5)), scale=0.42, color=(255, 236, 246), thickness=1)

    put_text(strip, "cluster features", (pad, heatmap_y - 10), scale=0.44, color=(185, 193, 205), thickness=1)
    put_text(strip, "clip time", (heatmap_x + heatmap_w - 76, heatmap_y + heatmap_h + 28), scale=0.44, color=(185, 193, 205), thickness=1)

    legend_x = pad
    legend_y = strip_h - 22
    legend_items = [
        ("baseline", BASELINE_COLOR),
        ("steered", STEERED_COLOR),
        ("ground truth", GROUND_TRUTH_COLOR),
    ]
    for label, color in legend_items:
        cv2.line(strip, (legend_x, legend_y), (legend_x + 28, legend_y), color, 4, cv2.LINE_AA)
        put_text(strip, label, (legend_x + 36, legend_y + 5), scale=0.42, color=(225, 230, 238), thickness=1)
        legend_x += 142
    short_name = sample_name if len(sample_name) <= 58 else f"{sample_name[:55]}..."
    put_text(strip, short_name, (width - pad - 430, legend_y + 5), scale=0.38, color=(150, 158, 171), thickness=1)
    return strip


def make_telemetry_panel(
    *,
    height: int,
    width: int,
    spec: ClipSpec,
    frame_local_idx: int,
    frame: SegmentFrame,
    sample_name: str,
    intent: int,
    alpha: float,
    feature_scale: float,
    cluster_scores: np.ndarray,
    feature_activations: np.ndarray,
    active_cluster_counts: np.ndarray,
    baseline_ade: np.ndarray,
    steered_ade: np.ndarray,
    baseline_fde: np.ndarray,
    steered_fde: np.ndarray,
) -> np.ndarray:
    panel = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    pad = 30
    y = 44
    put_text(panel, f"SAE cluster C{spec.cluster_id}", (pad, y), scale=0.95, thickness=2)
    y += 42
    stat_label = STAT_LABELS.get(spec.stat_name, spec.stat_name)
    put_text(panel, f"feature {spec.feature_idx} -> {stat_label}", (pad, y), scale=0.58, color=MUTED_TEXT_COLOR, thickness=1)
    y += 34
    put_text(panel, f"alpha={alpha:g}   scale={feature_scale:.3f}", (pad, y), scale=0.58, color=MUTED_TEXT_COLOR, thickness=1)
    y += 42

    put_text(panel, f"scene {frame.token_idx}  frame {frame.segment_frame_idx}", (pad, y), scale=0.62, thickness=1)
    y += 30
    put_text(panel, f"intent {INTENT_NAMES.get(intent, str(intent))}", (pad, y), scale=0.62, thickness=1)
    y += 30
    short_name = sample_name if len(sample_name) <= 42 else f"{sample_name[:39]}..."
    put_text(panel, short_name, (pad, y), scale=0.46, color=MUTED_TEXT_COLOR, thickness=1)
    y += 44

    cv2.rectangle(panel, (pad, y), (width - pad, y + 128), (255, 255, 255), -1)
    cv2.rectangle(panel, (pad, y), (width - pad, y + 128), PANEL_LINE, 1)
    put_text(panel, "Current frame metrics", (pad + 18, y + 30), scale=0.62, thickness=1)
    delta_ade = steered_ade[frame_local_idx] - baseline_ade[frame_local_idx]
    delta_fde = steered_fde[frame_local_idx] - baseline_fde[frame_local_idx]
    metric_color = (36, 132, 70) if delta_ade <= 0 else (188, 62, 62)
    put_text(panel, f"ADE  {baseline_ade[frame_local_idx]:.2f} -> {steered_ade[frame_local_idx]:.2f} m", (pad + 18, y + 62), scale=0.58, thickness=1)
    put_text(panel, f"FDE  {baseline_fde[frame_local_idx]:.2f} -> {steered_fde[frame_local_idx]:.2f} m", (pad + 18, y + 91), scale=0.58, thickness=1)
    put_text(panel, f"delta ADE {delta_ade:+.2f} m   delta FDE {delta_fde:+.2f} m", (pad + 18, y + 120), scale=0.52, color=metric_color, thickness=1)
    y += 172

    graph_w = width - 2 * pad
    draw_sparkline(
        panel,
        cluster_scores,
        current_idx=frame_local_idx,
        x=pad,
        y=y,
        w=graph_w,
        h=120,
        color=(47, 108, 159),
        label="cluster activation trace",
    )
    y += 174
    draw_sparkline(
        panel,
        feature_activations,
        current_idx=frame_local_idx,
        x=pad,
        y=y,
        w=graph_w,
        h=110,
        color=STEERED_COLOR,
        label=f"feature {spec.feature_idx} activation",
    )
    y += 156

    put_text(
        panel,
        f"active cluster features: {int(active_cluster_counts[frame_local_idx])}/{len(spec.cluster_features)}",
        (pad, y),
        scale=0.58,
        color=TEXT_COLOR,
        thickness=1,
    )
    y += 48
    draw_legend(panel, pad, y)
    return panel


def render_clip(
    *,
    spec: ClipSpec,
    render_data: dict[str, torch.Tensor | float],
    activation_groups: list[ActivationGroup],
    data_root: Path,
    output_path: Path,
    fps: int,
    video_scale: float,
    top_k: int,
    alpha: float,
    feature_scale: float,
    trajectory_width_m: float,
    narrative: bool,
    freeze_frame_idx: int | None,
    freeze_frame_idxs: list[int] | None,
    freeze_seconds: float,
    narrative_sweep_trajectories: dict[int, list[np.ndarray]] | None = None,
    narrative_sweep_alphas: list[float] | None = None,
    turn_comparison_sweeps: dict[str, dict[int, list[np.ndarray]]] | None = None,
    turn_comparison_features: dict[str, int] | None = None,
    sequential_turn_interventions: bool = False,
    non_intervention_stride: int = 1,
    narrative_layout: str = "timeline",
) -> dict:
    z_window = render_data["z_window"].numpy()
    cluster_feature_indices = [idx for idx in spec.cluster_features if 0 <= idx < z_window.shape[1]]
    if cluster_feature_indices:
        cluster_scores = z_window[:, cluster_feature_indices].sum(axis=1)
        active_cluster_counts = (z_window[:, cluster_feature_indices] > 0).sum(axis=1)
        cluster_heatmap = normalize_rows(z_window[:, cluster_feature_indices].T)
    else:
        cluster_scores = np.zeros(len(spec.frames), dtype=np.float32)
        active_cluster_counts = np.zeros(len(spec.frames), dtype=np.int64)
        cluster_heatmap = np.zeros((0, len(spec.frames)), dtype=np.float32)
    feature_activations = z_window[:, spec.feature_idx]
    selected_feature_row = None
    if spec.feature_idx in cluster_feature_indices:
        selected_feature_row = cluster_feature_indices.index(spec.feature_idx)
    group_activation_raw, group_activation_norm = compute_group_activation_rows(z_window, activation_groups)
    active_group_counts = np.zeros_like(group_activation_raw, dtype=np.int64)
    for row_idx, group in enumerate(activation_groups):
        valid = [idx for idx in group.feature_indices if 0 <= idx < z_window.shape[1]]
        if valid:
            active_group_counts[row_idx] = (z_window[:, valid] > 0).sum(axis=1)

    baseline_traj = render_data["baseline_traj"].numpy()
    baseline_scores = render_data["baseline_scores"].numpy()
    baseline_selected = render_data["baseline_selected"].numpy()
    steered_selected = render_data["steered_selected"].numpy()
    future = render_data["future"].numpy()
    past = render_data["past"].numpy()
    baseline_ade = render_data["baseline_ade"].numpy()
    steered_ade = render_data["steered_ade"].numpy()
    baseline_fde = render_data["baseline_fde"].numpy()
    steered_fde = render_data["steered_fde"].numpy()

    use_front_topdown = narrative and narrative_layout == "front_topdown"
    if use_front_topdown:
        filmstrip_thumbnails, filmstrip_indices = [], []
    else:
        filmstrip_thumbnails, filmstrip_indices = make_filmstrip_thumbnails(
            spec=spec,
            data_root=data_root,
            count=10,
        )

    viz_frames = []
    if not narrative:
        freeze_indices = []
    elif freeze_frame_idx is not None:
        freeze_indices = [freeze_frame_idx]
    elif freeze_frame_idxs:
        freeze_indices = freeze_frame_idxs
    else:
        freeze_indices = [len(spec.frames) // 2]
    freeze_indices = sorted({min(max(idx, 0), len(spec.frames) - 1) for idx in freeze_indices})
    first_freeze_idx = freeze_indices[0] if freeze_indices else len(spec.frames)
    freeze_set = set(freeze_indices)
    freeze_frame_count = max(1, int(round(freeze_seconds * fps)))
    fallback_sweep_alphas = np.linspace(0.0, alpha, freeze_frame_count, dtype=np.float32).tolist()
    sweep_alphas = narrative_sweep_alphas or [float(value) for value in fallback_sweep_alphas]
    if len(sweep_alphas) < freeze_frame_count:
        sweep_alphas = sweep_alphas + [sweep_alphas[-1] if sweep_alphas else alpha] * (
            freeze_frame_count - len(sweep_alphas)
        )
    non_intervention_stride = max(1, int(non_intervention_stride))
    reader = FrameReader(data_root)
    try:
        for frame_local_idx, frame in enumerate(tqdm(spec.frames, desc=f"Rendering C{spec.cluster_id}")):
            sample = reader.read(frame)
            current_xy = past[frame_local_idx, -1, :2]
            should_append_normal = (
                not narrative
                or frame_local_idx in freeze_set
                or frame_local_idx % non_intervention_stride == 0
            )
            if should_append_normal:
                baseline_vf = baseline_selected[frame_local_idx] - current_xy
                if use_front_topdown:
                    combined = make_front_topdown_frame(
                        cam_images=sample["cam_images"],
                        calibrations=sample["calibrations"],
                        baseline_trajectory=baseline_vf,
                        steered_trajectory=None,
                        trajectory_width_m=trajectory_width_m,
                        steered_color=STEERED_COLOR,
                        phase_label="baseline plan",
                        draw_steered_on_camera=False,
                    )
                else:
                    if turn_comparison_sweeps:
                        selected_vf = None
                    elif narrative and frame_local_idx < first_freeze_idx:
                        selected_vf = None
                    else:
                        selected_vf = steered_selected[frame_local_idx] - current_xy
                    pano = stitch_front3_multi(
                        sample["cam_images"],
                        sample["calibrations"],
                        selected_vf,
                        trajectory_width_m,
                    )
                    heatmap_strip = make_left_right_timeline_strip(
                        width=pano.shape[1],
                        spec=spec,
                        frame_local_idx=frame_local_idx,
                        frame=frame,
                        sample_name=sample["name"],
                        intent=sample["intent"],
                        alpha=alpha,
                        feature_scale=feature_scale,
                        activation_groups=activation_groups,
                        group_activation_norm=group_activation_norm,
                        active_group_counts=active_group_counts,
                        filmstrip_thumbnails=filmstrip_thumbnails,
                        filmstrip_indices=filmstrip_indices,
                        baseline_ade=baseline_ade,
                        steered_ade=steered_ade,
                        baseline_fde=baseline_fde,
                        steered_fde=steered_fde,
                    )
                    combined = np.concatenate([pano, heatmap_strip], axis=0)
                viz_frames.append(
                    {
                        "image": combined,
                        "ade": float(baseline_ade[frame_local_idx]),
                        "fde": float(baseline_fde[frame_local_idx]),
                        "name": sample["name"],
                    }
                )
            if narrative and frame_local_idx in freeze_set:
                baseline_vf = baseline_selected[frame_local_idx] - current_xy
                sweep_for_frame = None
                if narrative_sweep_trajectories:
                    sweep_for_frame = narrative_sweep_trajectories.get(frame_local_idx)
                if turn_comparison_sweeps and sequential_turn_interventions:
                    phase_specs = [
                        (
                            "left",
                            "left turn",
                            LEFT_TURN_COLOR,
                            turn_comparison_sweeps.get("left", {}).get(frame_local_idx),
                        ),
                        (
                            "right",
                            "right turn",
                            RIGHT_TURN_COLOR,
                            turn_comparison_sweeps.get("right", {}).get(frame_local_idx),
                        ),
                    ]
                    for _, phase_label, phase_color, phase_sweep in phase_specs:
                        for freeze_elapsed in range(freeze_frame_count):
                            if phase_sweep and freeze_elapsed < len(phase_sweep):
                                intervention_vf = phase_sweep[freeze_elapsed] - current_xy
                            else:
                                intervention_vf = baseline_vf
                            intervention_alpha = sweep_alphas[freeze_elapsed]
                            if use_front_topdown:
                                freeze_image = make_front_topdown_frame(
                                    cam_images=sample["cam_images"],
                                    calibrations=sample["calibrations"],
                                    baseline_trajectory=baseline_vf,
                                    steered_trajectory=intervention_vf,
                                    trajectory_width_m=trajectory_width_m,
                                    steered_color=phase_color,
                                    phase_label=f"SAE {phase_label}",
                                    intervention_alpha=float(intervention_alpha),
                                    max_alpha=alpha,
                                )
                            else:
                                freeze_pano = stitch_front3_single_turn_freeze(
                                    sample["cam_images"],
                                    sample["calibrations"],
                                    baseline_vf,
                                    intervention_vf,
                                    trajectory_width_m,
                                    label=phase_label,
                                    color=phase_color,
                                    freeze_elapsed=freeze_elapsed,
                                    freeze_total=freeze_frame_count,
                                    intervention_alpha=float(intervention_alpha),
                                    max_alpha=alpha,
                                )
                                freeze_strip = make_left_right_timeline_strip(
                                    width=freeze_pano.shape[1],
                                    spec=spec,
                                    frame_local_idx=frame_local_idx,
                                    frame=frame,
                                    sample_name=sample["name"],
                                    intent=sample["intent"],
                                    alpha=alpha,
                                    feature_scale=feature_scale,
                                    activation_groups=activation_groups,
                                    group_activation_norm=group_activation_norm,
                                    active_group_counts=active_group_counts,
                                    filmstrip_thumbnails=filmstrip_thumbnails,
                                    filmstrip_indices=filmstrip_indices,
                                    baseline_ade=baseline_ade,
                                    steered_ade=steered_ade,
                                    baseline_fde=baseline_fde,
                                    steered_fde=steered_fde,
                                )
                                freeze_image = np.concatenate([freeze_pano, freeze_strip], axis=0)
                            viz_frames.append(
                                {
                                    "image": freeze_image,
                                    "ade": float(steered_ade[frame_local_idx]),
                                    "fde": float(steered_fde[frame_local_idx]),
                                    "name": sample["name"],
                                }
                            )
                    continue
                for freeze_elapsed in range(freeze_frame_count):
                    if turn_comparison_sweeps:
                        left_sweep = turn_comparison_sweeps.get("left", {}).get(frame_local_idx)
                        right_sweep = turn_comparison_sweeps.get("right", {}).get(frame_local_idx)
                        if left_sweep and freeze_elapsed < len(left_sweep):
                            left_vf = left_sweep[freeze_elapsed] - current_xy
                        else:
                            left_vf = baseline_vf
                        if right_sweep and freeze_elapsed < len(right_sweep):
                            right_vf = right_sweep[freeze_elapsed] - current_xy
                        else:
                            right_vf = baseline_vf
                        intervention_alpha = sweep_alphas[freeze_elapsed]
                        if use_front_topdown:
                            freeze_pano = make_front_topdown_frame(
                                cam_images=sample["cam_images"],
                                calibrations=sample["calibrations"],
                                baseline_trajectory=baseline_vf,
                                steered_trajectory=left_vf,
                                trajectory_width_m=trajectory_width_m,
                                steered_color=LEFT_TURN_COLOR,
                                phase_label="SAE left turn",
                                intervention_alpha=float(intervention_alpha),
                                max_alpha=alpha,
                            )
                        else:
                            freeze_pano = stitch_front3_turn_comparison_freeze(
                                sample["cam_images"],
                                sample["calibrations"],
                                baseline_vf,
                                left_vf,
                                right_vf,
                                trajectory_width_m,
                                freeze_elapsed=freeze_elapsed,
                                freeze_total=freeze_frame_count,
                                intervention_alpha=float(intervention_alpha),
                                max_alpha=alpha,
                            )
                    elif sweep_for_frame and freeze_elapsed < len(sweep_for_frame):
                        steered_vf = sweep_for_frame[freeze_elapsed] - current_xy
                        intervention_alpha = sweep_alphas[freeze_elapsed]
                        if use_front_topdown:
                            freeze_pano = make_front_topdown_frame(
                                cam_images=sample["cam_images"],
                                calibrations=sample["calibrations"],
                                baseline_trajectory=baseline_vf,
                                steered_trajectory=steered_vf,
                                trajectory_width_m=trajectory_width_m,
                                steered_color=STEERED_COLOR,
                                phase_label="SAE intervention",
                                intervention_alpha=float(intervention_alpha),
                                max_alpha=alpha,
                            )
                        else:
                            freeze_pano = stitch_front3_intervention_freeze(
                                sample["cam_images"],
                                sample["calibrations"],
                                baseline_vf,
                                steered_vf,
                                trajectory_width_m,
                                freeze_elapsed=freeze_elapsed,
                                freeze_total=freeze_frame_count,
                                intervention_alpha=float(intervention_alpha),
                                max_alpha=alpha,
                            )
                    else:
                        steered_vf = steered_selected[frame_local_idx] - current_xy
                        intervention_alpha = alpha
                        if use_front_topdown:
                            freeze_pano = make_front_topdown_frame(
                                cam_images=sample["cam_images"],
                                calibrations=sample["calibrations"],
                                baseline_trajectory=baseline_vf,
                                steered_trajectory=steered_vf,
                                trajectory_width_m=trajectory_width_m,
                                steered_color=STEERED_COLOR,
                                phase_label="SAE intervention",
                                intervention_alpha=float(intervention_alpha),
                                max_alpha=alpha,
                            )
                        else:
                            freeze_pano = stitch_front3_intervention_freeze(
                                sample["cam_images"],
                                sample["calibrations"],
                                baseline_vf,
                                steered_vf,
                                trajectory_width_m,
                                freeze_elapsed=freeze_elapsed,
                                freeze_total=freeze_frame_count,
                                intervention_alpha=float(intervention_alpha),
                                max_alpha=alpha,
                            )
                    if use_front_topdown:
                        freeze_image = freeze_pano
                    else:
                        freeze_strip = make_left_right_timeline_strip(
                            width=freeze_pano.shape[1],
                            spec=spec,
                            frame_local_idx=frame_local_idx,
                            frame=frame,
                            sample_name=sample["name"],
                            intent=sample["intent"],
                            alpha=alpha,
                            feature_scale=feature_scale,
                            activation_groups=activation_groups,
                            group_activation_norm=group_activation_norm,
                            active_group_counts=active_group_counts,
                            filmstrip_thumbnails=filmstrip_thumbnails,
                            filmstrip_indices=filmstrip_indices,
                            baseline_ade=baseline_ade,
                            steered_ade=steered_ade,
                            baseline_fde=baseline_fde,
                            steered_fde=steered_fde,
                        )
                        freeze_image = np.concatenate([freeze_pano, freeze_strip], axis=0)
                    viz_frames.append(
                        {
                            "image": freeze_image,
                            "ade": float(steered_ade[frame_local_idx]),
                            "fde": float(steered_fde[frame_local_idx]),
                            "name": sample["name"],
                        }
                    )
    finally:
        reader.close()

    create_video(viz_frames, str(output_path), fps=fps, scale=video_scale)

    return {
        "cluster_id": spec.cluster_id,
        "feature_idx": spec.feature_idx,
        "stat_name": spec.stat_name,
        "seed_scene_idx": spec.seed_scene_idx,
        "segment_hash": spec.segment_hash,
        "segment_start_frame_idx": spec.segment_start_frame_idx,
        "segment_end_frame_idx": spec.segment_end_frame_idx,
        "num_frames": len(spec.frames),
        "alpha": alpha,
        "feature_scale": feature_scale,
        "trajectory_width_m": trajectory_width_m,
        "narrative": narrative,
        "freeze_frame_idx": freeze_indices[0] if narrative and freeze_indices else "",
        "freeze_frame_idxs": ",".join(str(idx) for idx in freeze_indices) if narrative else "",
        "freeze_seconds": freeze_seconds if narrative else "",
        "sequential_turn_interventions": bool(sequential_turn_interventions),
        "non_intervention_stride": non_intervention_stride,
        "narrative_layout": narrative_layout if narrative else "",
        "turn_comparison": bool(turn_comparison_sweeps),
        "left_turn_feature_idx": turn_comparison_features.get("left", "") if turn_comparison_features else "",
        "right_turn_feature_idx": turn_comparison_features.get("right", "") if turn_comparison_features else "",
        "mean_baseline_ade": float(np.mean(baseline_ade)),
        "mean_steered_ade": float(np.mean(steered_ade)),
        "mean_delta_ade": float(np.mean(steered_ade - baseline_ade)),
        "mean_baseline_fde": float(np.mean(baseline_fde)),
        "mean_steered_fde": float(np.mean(steered_fde)),
        "mean_delta_fde": float(np.mean(steered_fde - baseline_fde)),
        "max_cluster_score": float(np.max(cluster_scores)) if cluster_scores.size else 0.0,
        "max_feature_activation": float(np.max(feature_activations)) if feature_activations.size else 0.0,
        "baseline_check_max_abs": float(render_data["baseline_check_max_abs"]),
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAE steering supplemental video clips.")
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sae_block", type=int, default=DEFAULT_SAE_BLOCK)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--corr_threshold", type=float, default=0.35)
    parser.add_argument("--clusters", type=str, default="1")
    parser.add_argument(
        "--activation_clusters",
        type=str,
        default="1:Left,3:Right",
        help="Comma-separated cluster_id:label rows to show in the activation timeline.",
    )
    parser.add_argument("--clips_per_cluster", type=int, default=1)
    parser.add_argument("--window_frames", type=int, default=96)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--video_scale", type=float, default=0.75)
    parser.add_argument(
        "--trajectory_width_m",
        type=float,
        default=3.0,
        help="Ground-plane width in meters for the projected trajectory band.",
    )
    parser.add_argument(
        "--narrative",
        action="store_true",
        help="Insert a freeze-frame intervention segment with baseline and steered trajectories.",
    )
    parser.add_argument(
        "--freeze_frame_idx",
        type=int,
        default=None,
        help="Single local frame index for narrative freeze. Overrides --freeze_frame_idxs.",
    )
    parser.add_argument(
        "--freeze_frame_idxs",
        type=str,
        default="24,48,72",
        help="Comma-separated local frame indices for repeated narrative intervention freezes.",
    )
    parser.add_argument(
        "--freeze_seconds",
        type=float,
        default=2.0,
        help="Duration of the narrative freeze segment in seconds.",
    )
    parser.add_argument(
        "--compare_turn_interventions",
        action="store_true",
        help="During narrative freezes, show baseline plus left-turn and right-turn SAE interventions.",
    )
    parser.add_argument(
        "--sequential_turn_interventions",
        action="store_true",
        help="With --compare_turn_interventions, render left and right intervention sweeps sequentially.",
    )
    parser.add_argument(
        "--non_intervention_stride",
        type=int,
        default=1,
        help="Keep every Nth non-intervention source frame; use 2 for 2x playback between freezes.",
    )
    parser.add_argument(
        "--narrative_layout",
        type=str,
        default="timeline",
        choices=["timeline", "front_topdown"],
        help="Narrative video layout: existing timeline strip or front camera plus top-down trajectory panel.",
    )
    parser.add_argument(
        "--left_turn_cluster",
        type=int,
        default=1,
        help="Cluster whose top control feature is used for the left-turn intervention comparison.",
    )
    parser.add_argument(
        "--right_turn_cluster",
        type=int,
        default=3,
        help="Cluster whose top control feature is used for the right-turn intervention comparison.",
    )
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--index_file", type=Path, default=None)
    parser.add_argument("--scene_idx", type=int, default=None)
    parser.add_argument(
        "--scene_idx_by_cluster",
        type=str,
        default="1:96098",
        help="Comma-separated cluster_id:scene_idx overrides. Default chooses a bidirectional left/right window.",
    )
    parser.add_argument(
        "--auto_bidirectional",
        action="store_true",
        help="Scan validation segments for a window with both activation rows and left/right lateral motion.",
    )
    parser.add_argument("--feature_idx", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_tag", type=str, default="", help="Optional filename tag inserted after narrative prefix.")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=auto_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sae_block != DEFAULT_SAE_BLOCK:
        raise ValueError(
            f"This video workflow currently supports sae_block={DEFAULT_SAE_BLOCK}; got {args.sae_block}."
        )
    if args.window_frames <= 0:
        raise ValueError("--window_frames must be positive")
    if args.clips_per_cluster <= 0:
        raise ValueError("--clips_per_cluster must be positive")

    threshold = f"{args.corr_threshold:g}"
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAE bundle from {args.run_root}")
    bundle = load_sae_bundle(args.run_root, args.split, args.sae_block, map_location="cpu")
    token_blob = bundle["token_blob"]
    token_tensor, token_key = resolve_token_tensor(token_blob, args.sae_block)

    meta = token_blob.get("meta", {})
    data_root_value = args.data_root or meta.get("data_dir")
    index_file_value = args.index_file or meta.get("index_file")
    if data_root_value is None:
        raise ValueError("Could not infer data_root; pass --data_root")
    if index_file_value is None:
        raise ValueError("Could not infer index_file; pass --index_file")
    data_root = Path(data_root_value).resolve()
    index_file = Path(index_file_value).resolve()

    print(f"Using token key {token_key}")
    print(f"Loading index from {index_file}")
    index_entries = load_index(index_file)
    segments = load_segment_index(index_file)
    entry_to_token_idx, entry_to_segment = build_entry_maps(index_entries, segments)

    cluster_features = load_cluster_features(args.run_root, args.sae_block, args.split, threshold)
    activation_groups = build_activation_groups(cluster_features, args.activation_clusters)
    print(
        "Activation rows: "
        + ", ".join(f"{group.label}=C{group.cluster_id}" for group in activation_groups)
    )

    sae = build_sae_from_checkpoint(bundle["ckpt"], bundle["legacy_norm"]).to(device)
    sae.eval()

    scene_idx_by_cluster = parse_scene_idx_by_cluster(args.scene_idx_by_cluster)
    if args.auto_bidirectional and args.scene_idx is None:
        clusters = parse_clusters(args.clusters, cluster_features.keys())
        auto_overrides = choose_bidirectional_scene_overrides(
            token_tensor=token_tensor,
            past_cpu=token_blob["past"].float(),
            future_cpu=token_blob["future"].float(),
            sae=sae,
            cluster_features=cluster_features,
            activation_groups=activation_groups,
            clusters=clusters,
            window_frames=args.window_frames,
            batch_size=args.batch_size,
            device=device,
            segments=segments,
            entry_to_token_idx=entry_to_token_idx,
        )
        scene_idx_by_cluster.update(auto_overrides)

    specs = choose_clip_specs(
        run_root=args.run_root,
        output_dir=args.output_dir,
        block=args.sae_block,
        split=args.split,
        threshold=threshold,
        clusters_arg=args.clusters,
        clips_per_cluster=args.clips_per_cluster,
        window_frames=args.window_frames,
        scene_idx_override=args.scene_idx,
        scene_idx_by_cluster=scene_idx_by_cluster,
        feature_idx_override=args.feature_idx,
        index_entries=index_entries,
        segments=segments,
        entry_to_token_idx=entry_to_token_idx,
        entry_to_segment=entry_to_segment,
    )
    if not specs:
        raise RuntimeError("No clips selected")
    print(f"Selected {len(specs)} clip(s)")

    print(f"Loading planner model from {args.model_path}")
    planner_model, _ = load_model(str(args.model_path), device=device)
    planner_model.eval()
    feature_scales = load_feature_scales(args.run_root, args.sae_block, args.split)
    cluster_control_rows = {
        int(row["cluster_id"]): row
        for row in read_csv_rows(cluster_control_summary_path(args.output_dir, args.sae_block, args.split))
    }

    past_cpu = token_blob["past"].float()
    future_cpu = token_blob["future"].float()
    expected_trajectory_cpu = token_blob["trajectory"].float()
    expected_scores_cpu = token_blob["scores"].float()
    manifest_rows = []

    for spec in specs:
        indices = [frame.token_idx for frame in spec.frames]
        feature_scale = feature_scales.get(spec.feature_idx)
        if feature_scale is None:
            raise KeyError(f"No intervention scale found for feature {spec.feature_idx}")
        print(
            f"Rendering C{spec.cluster_id} feature={spec.feature_idx} "
            f"scene={spec.seed_scene_idx} segment={spec.segment_hash[:12]}..."
        )
        render_data = run_steering(
            planner_model=planner_model,
            sae=sae,
            token_tensor=token_tensor,
            past_cpu=past_cpu,
            future_cpu=future_cpu,
            expected_trajectory_cpu=expected_trajectory_cpu,
            expected_scores_cpu=expected_scores_cpu,
            indices=indices,
            feature_idx=spec.feature_idx,
            feature_scale=feature_scale,
            alpha=args.alpha,
            batch_size=args.batch_size,
            device=device,
            num_proposals=planner_model.n_proposals,
            horizon=planner_model.horizon,
        )
        freeze_frame_idxs = parse_optional_int_list(args.freeze_frame_idxs)
        if args.narrative:
            if args.freeze_frame_idx is not None:
                requested_freeze_indices = [args.freeze_frame_idx]
            elif freeze_frame_idxs:
                requested_freeze_indices = freeze_frame_idxs
            else:
                requested_freeze_indices = [len(spec.frames) // 2]
            requested_freeze_indices = sorted(
                {min(max(idx, 0), len(spec.frames) - 1) for idx in requested_freeze_indices}
            )
            freeze_frame_count = max(1, int(round(args.freeze_seconds * args.fps)))
            if args.compare_turn_interventions:
                try:
                    left_feature_idx = int(cluster_control_rows[args.left_turn_cluster]["top_feature_idx"])
                    right_feature_idx = int(cluster_control_rows[args.right_turn_cluster]["top_feature_idx"])
                except KeyError as exc:
                    raise KeyError(f"Missing turn-comparison cluster in control summary: {exc}") from exc
                left_scale = feature_scales.get(left_feature_idx)
                right_scale = feature_scales.get(right_feature_idx)
                if left_scale is None:
                    raise KeyError(f"No intervention scale found for left feature {left_feature_idx}")
                if right_scale is None:
                    raise KeyError(f"No intervention scale found for right feature {right_feature_idx}")
                left_sweep, sweep_alphas = compute_alpha_sweep_trajectories(
                    planner_model=planner_model,
                    sae=sae,
                    token_tensor=token_tensor,
                    past_cpu=past_cpu,
                    future_cpu=future_cpu,
                    expected_trajectory_cpu=expected_trajectory_cpu,
                    expected_scores_cpu=expected_scores_cpu,
                    spec=spec,
                    freeze_indices=requested_freeze_indices,
                    feature_idx=left_feature_idx,
                    feature_scale=left_scale,
                    max_alpha=args.alpha,
                    freeze_frame_count=freeze_frame_count,
                    batch_size=args.batch_size,
                    device=device,
                )
                right_sweep, sweep_alphas = compute_alpha_sweep_trajectories(
                    planner_model=planner_model,
                    sae=sae,
                    token_tensor=token_tensor,
                    past_cpu=past_cpu,
                    future_cpu=future_cpu,
                    expected_trajectory_cpu=expected_trajectory_cpu,
                    expected_scores_cpu=expected_scores_cpu,
                    spec=spec,
                    freeze_indices=requested_freeze_indices,
                    feature_idx=right_feature_idx,
                    feature_scale=right_scale,
                    max_alpha=args.alpha,
                    freeze_frame_count=freeze_frame_count,
                    batch_size=args.batch_size,
                    device=device,
                )
                sweep_trajectories = None
                turn_comparison_sweeps = {"left": left_sweep, "right": right_sweep}
                turn_comparison_features = {"left": left_feature_idx, "right": right_feature_idx}
            else:
                sweep_trajectories, sweep_alphas = compute_alpha_sweep_trajectories(
                    planner_model=planner_model,
                    sae=sae,
                    token_tensor=token_tensor,
                    past_cpu=past_cpu,
                    future_cpu=future_cpu,
                    expected_trajectory_cpu=expected_trajectory_cpu,
                    expected_scores_cpu=expected_scores_cpu,
                    spec=spec,
                    freeze_indices=requested_freeze_indices,
                    feature_idx=spec.feature_idx,
                    feature_scale=feature_scale,
                    max_alpha=args.alpha,
                    freeze_frame_count=freeze_frame_count,
                    batch_size=args.batch_size,
                    device=device,
                )
                turn_comparison_sweeps = None
                turn_comparison_features = None
        else:
            requested_freeze_indices = []
            sweep_trajectories = None
            sweep_alphas = None
            turn_comparison_sweeps = None
            turn_comparison_features = None
        output_prefix = "narrative_" if args.narrative else ""
        if args.output_tag:
            output_prefix += f"{args.output_tag}_"
        output_path = args.output_dir / (
            f"{output_prefix}cluster_{spec.cluster_id}_feature_{spec.feature_idx}_"
            f"scene_{spec.seed_scene_idx}_alpha_{args.alpha:g}.mp4"
        )
        row = render_clip(
            spec=spec,
            render_data=render_data,
            activation_groups=activation_groups,
            data_root=data_root,
            output_path=output_path,
            fps=args.fps,
            video_scale=args.video_scale,
            top_k=args.top_k,
            alpha=args.alpha,
            feature_scale=feature_scale,
            trajectory_width_m=args.trajectory_width_m,
            narrative=args.narrative,
            freeze_frame_idx=args.freeze_frame_idx,
            freeze_frame_idxs=requested_freeze_indices,
            freeze_seconds=args.freeze_seconds,
            narrative_sweep_trajectories=sweep_trajectories,
            narrative_sweep_alphas=sweep_alphas,
            turn_comparison_sweeps=turn_comparison_sweeps,
            turn_comparison_features=turn_comparison_features,
            sequential_turn_interventions=args.sequential_turn_interventions,
            non_intervention_stride=args.non_intervention_stride,
            narrative_layout=args.narrative_layout,
        )
        manifest_rows.append(row)

    manifest_path = args.output_dir / "manifest.csv"
    write_csv_rows(manifest_path, manifest_rows)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
