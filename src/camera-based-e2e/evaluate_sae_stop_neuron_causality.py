"""
Evaluate whether amplifying stop-associated SAE features reduces predicted motion.

The intended workflow is:

1. Run ``analyze_sae_object_neurons.py --label_mode stop_with_object`` to obtain
   candidate SAE features associated with stopping at stop signs.
2. Feed that analysis JSON into this script.
3. Over a matched set of stop-sign-present, currently-moving frames, compare
   baseline vs feature-intervened predicted trajectories.

The script reports paired changes in:

- endpoint distance from the current position
- path length
- mean displacement from the current position
- predicted tail speed
- predicted stop rate
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from analyze_sae_masked_object_neurons import prepare_model_and_sae, select_best_trajectory
from analyze_sae_object_neurons import (
    default_index_file,
    frame_has_object,
    load_detection_artifacts,
    past_current_speed,
    resolve_object_label,
)
from loader import WaymoE2E, collate_with_images
from models.sae import SparseAutoencoder
from new_sae_utils import get_sae_target_layer

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def resolve_input_path(
    raw_path: str | Path,
    *,
    description: str,
) -> Path:
    path = Path(raw_path).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([Path.cwd() / path, SCRIPT_DIR / path, REPO_ROOT / path])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    locations = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {description} '{raw_path}'. Checked: {locations}")


def load_analysis_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_csv_ints(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def feature_ids_from_analysis(analysis: dict, key: str, feature_count: int) -> List[int]:
    rows = analysis.get(key, [])
    return [int(row["feature_idx"]) for row in rows[:feature_count]]


def choose_feature_ids(analysis: dict, feature_key: str, feature_count: int) -> List[int]:
    if feature_key != "auto":
        feature_ids = feature_ids_from_analysis(analysis, feature_key, feature_count)
        if not feature_ids:
            raise ValueError(f"Analysis JSON has no rows under '{feature_key}'")
        return feature_ids

    for key in (
        "top_positive_association",
        "top_joint_positive_features",
        "top_presence_positive_features",
        "top_trajectory_positive_features",
    ):
        feature_ids = feature_ids_from_analysis(analysis, key, feature_count)
        if feature_ids:
            return feature_ids
    raise ValueError("Could not find any positive feature list in analysis JSON")


@contextlib.contextmanager
def intervene_selected_features_hook(
    target_layer,
    sae: SparseAutoencoder,
    feature_ids: Sequence[int],
    mode: str,
    amplify_factor: float,
    absolute_value: float,
):
    feature_ids_t = torch.as_tensor(feature_ids, dtype=torch.long)

    def replace_with_modified_sae(module, inputs, output):
        del module, inputs
        flat_output = output.reshape(-1, output.size(-1))
        hidden, preprocess_stats = sae.encode(
            flat_output,
            return_preprocess_stats=True,
        )
        if feature_ids_t.numel() > 0:
            target_ids = feature_ids_t.to(device=hidden.device)
            hidden = hidden.clone()
            if mode == "zero":
                hidden[:, target_ids] = 0.0
            elif mode == "amplify":
                hidden[:, target_ids] = hidden[:, target_ids] * amplify_factor
            elif mode == "set_absolute":
                hidden[:, target_ids] = absolute_value
            else:
                raise ValueError(f"Unsupported intervention mode: {mode}")
        reconstructed = sae.decode_to_input(hidden, preprocess_stats=preprocess_stats)
        return reconstructed.view_as(output)

    handle = target_layer.register_forward_hook(replace_with_modified_sae)
    try:
        yield
    finally:
        handle.remove()


def run_baseline_and_patched(
    *,
    model,
    sae: SparseAutoencoder,
    target_layer,
    past: torch.Tensor,
    intent: torch.Tensor,
    images,
    feature_ids: Sequence[int],
    intervention_mode: str,
    amplify_factor: float,
    absolute_value: float,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_inputs = {"PAST": past, "IMAGES": images, "INTENT": intent}
    with torch.no_grad():
        baseline_output = model(model_inputs)
        baseline_traj = select_best_trajectory(baseline_output).detach().cpu()
        with intervene_selected_features_hook(
            target_layer,
            sae,
            feature_ids=feature_ids,
            mode=intervention_mode,
            amplify_factor=amplify_factor,
            absolute_value=absolute_value,
        ):
            patched_output = model(model_inputs)
        patched_traj = select_best_trajectory(patched_output).detach().cpu()
    return baseline_traj, patched_traj


def endpoint_distance(traj: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
    current_xy = past[:, -1, :2]
    return torch.norm(traj[:, -1] - current_xy, dim=-1)


def path_length(traj: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
    prev = torch.cat([past[:, -1:, :2], traj[:, :-1]], dim=1)
    return torch.norm(traj - prev, dim=-1).sum(dim=-1)


def mean_displacement(traj: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
    current_xy = past[:, -1:, :2]
    return torch.norm(traj - current_xy, dim=-1).mean(dim=-1)


def predicted_tail_speed(
    traj: torch.Tensor,
    past: torch.Tensor,
    *,
    dt: float,
    tail_steps: int,
) -> torch.Tensor:
    prev = torch.cat([past[:, -1:, :2], traj[:, :-1]], dim=1)
    step_speeds = torch.norm(traj - prev, dim=-1) / dt
    tail = step_speeds[:, -min(tail_steps, step_speeds.size(1)) :]
    return tail.mean(dim=-1)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_stat_summary(diff: torch.Tensor) -> dict:
    diff64 = diff.to(torch.float64)
    n = int(diff64.numel())
    mean = float(diff64.mean().item()) if n else 0.0
    if n <= 1:
        std = 0.0
        sem = 0.0
    else:
        std = float(diff64.std(unbiased=True).item())
        sem = std / math.sqrt(n) if std > 0.0 else 0.0
    if sem > 0.0:
        z_value = mean / sem
        p_less = normal_cdf(z_value)
        p_two_sided = 2.0 * min(p_less, 1.0 - p_less)
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem
    else:
        z_value = float("-inf") if mean < 0 else (float("inf") if mean > 0 else 0.0)
        p_less = 0.0 if mean < 0 else (1.0 if mean > 0 else 0.5)
        p_two_sided = 0.0 if mean != 0.0 else 1.0
        ci_low = mean
        ci_high = mean
    return {
        "n": n,
        "mean_diff": mean,
        "std_diff": std,
        "sem_diff": sem,
        "z_value": z_value,
        "p_value_one_sided_decrease": p_less,
        "p_value_two_sided": p_two_sided,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "fraction_decreased": float((diff64 < 0).to(torch.float64).mean().item()) if n else 0.0,
        "fraction_increased": float((diff64 > 0).to(torch.float64).mean().item()) if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_json", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--sae_checkpoint_path", type=str, required=True, help="SAE checkpoint .pt, extracted directory, or sae_checkpoints.tar.gz archive")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--model_checkpoint_path",
        default="./pretrained/camera-e2e-epoch=04-val_loss=2.90.ckpt",
        type=str,
    )
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--index_file", type=str, default=None)
    parser.add_argument("--n_items", type=int, default=None, help="Optional cap on raw frames scanned")
    parser.add_argument("--max_examples", type=int, default=1000, help="Maximum eligible frames to evaluate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--block_idx", type=int, default=3, help="Transformer block index whose output query state is SAE-modeled")
    parser.add_argument("--camera_indices", type=str, default=None)
    parser.add_argument("--feature_key", type=str, default="auto")
    parser.add_argument("--feature_count", type=int, default=8)
    parser.add_argument("--intervention_mode", type=str, default="amplify", choices=["amplify", "zero", "set_absolute"])
    parser.add_argument("--amplify_factor", type=float, default=10.0)
    parser.add_argument("--absolute_value", type=float, default=1.0, help="Value to set features to when intervention_mode is 'set_absolute'")
    parser.add_argument("--moving_speed_thresh", type=float, default=None)
    parser.add_argument("--tail_steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--stop_speed_thresh", type=float, default=None)
    parser.add_argument(
        "--require_object_presence",
        action="store_true",
        help="Only evaluate frames with the target object present",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    analysis_json_path = resolve_input_path(args.analysis_json, description="analysis json")
    detections_path = resolve_input_path(args.detections, description="detections artifact")
    sae_checkpoint_path = resolve_input_path(args.sae_checkpoint_path, description="SAE checkpoint")
    model_checkpoint_path = resolve_input_path(
        args.model_checkpoint_path,
        description="model checkpoint",
    )
    data_dir = resolve_input_path(args.data_dir, description="dataset directory")

    analysis = load_analysis_json(analysis_json_path)
    split = args.split or analysis.get("split", "train")
    index_file = resolve_input_path(
        args.index_file or analysis.get("index_file", default_index_file(split)),
        description="dataset index file",
    )
    dt = float(args.dt if args.dt is not None else analysis.get("dt", 0.25))
    tail_steps = int(args.tail_steps if args.tail_steps is not None else analysis.get("tail_steps", 4))
    moving_speed_thresh = float(
        args.moving_speed_thresh
        if args.moving_speed_thresh is not None
        else analysis.get("moving_speed_thresh", 1.0)
    )
    stop_speed_thresh = float(
        args.stop_speed_thresh
        if args.stop_speed_thresh is not None
        else analysis.get("stop_speed_thresh", 0.5)
    )

    raw_camera_indices = parse_csv_ints(args.camera_indices)
    if raw_camera_indices:
        camera_indices = raw_camera_indices
    else:
        camera_indices = analysis.get("camera_indices", [1]) or [1]

    categories, frame_to_record = load_detection_artifacts(str(detections_path))
    label_id, label_name = resolve_object_label(
        categories,
        analysis.get("object_label_name", "stop sign"),
        analysis.get("object_label_id"),
    )
    feature_ids = choose_feature_ids(analysis, args.feature_key, args.feature_count)

    model, sae, _ = prepare_model_and_sae(
        model_checkpoint_path=str(model_checkpoint_path),
        sae_checkpoint_path=str(sae_checkpoint_path),
        block_idx=args.block_idx,
        device=device,
    )
    target_layer = get_sae_target_layer(model, args.block_idx)

    dataset = WaymoE2E(indexFile=str(index_file), data_dir=str(data_dir), n_items=None)
    if args.n_items is not None:
        dataset.indexes = dataset.indexes[: args.n_items]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_with_images,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    results: List[dict] = []
    raw_frames_scanned = 0
    eligible_frames = 0
    missing_detection_frames = 0
    skipped_no_object = 0
    skipped_low_speed = 0
    running_dataset_idx = 0

    pbar = tqdm(total=args.max_examples, desc="Evaluating stop-neuron causality")
    with torch.no_grad():
        for batch in loader:
            if eligible_frames >= args.max_examples:
                break

            frame_names = batch["NAME"]
            batch_size = len(frame_names)
            raw_frames_scanned += batch_size

            accepted_indices: List[int] = []
            accepted_rows: List[dict] = []

            past_cpu = batch["PAST"]
            for sample_idx, frame_name in enumerate(frame_names):
                if eligible_frames + len(accepted_indices) >= args.max_examples:
                    break

                record = frame_to_record.get(frame_name)
                if record is None:
                    missing_detection_frames += 1
                    continue

                has_object = frame_has_object(
                    record=record,
                    label_id=label_id,
                    score_thresh=float(analysis.get("score_thresh", 0.4)),
                    camera_indices=camera_indices,
                )
                if args.require_object_presence and not has_object:
                    skipped_no_object += 1
                    continue

                current_speed = past_current_speed(past_cpu[sample_idx])
                if current_speed < moving_speed_thresh:
                    skipped_low_speed += 1
                    continue

                accepted_indices.append(sample_idx)
                accepted_rows.append(
                    {
                        "dataset_idx": running_dataset_idx + sample_idx,
                        "frame_name": frame_name,
                        "has_object": bool(has_object),
                        "current_speed": float(current_speed),
                    }
                )

            if not accepted_indices:
                running_dataset_idx += batch_size
                continue

            past = batch["PAST"][accepted_indices].to(device)
            intent = batch["INTENT"][accepted_indices].to(device)
            images = model.decode_batch_jpeg(
                [batch["IMAGES_JPEG"][sample_idx] for sample_idx in accepted_indices],
                device=device,
            )

            baseline_traj, patched_traj = run_baseline_and_patched(
                model=model,
                sae=sae,
                target_layer=target_layer,
                past=past,
                intent=intent,
                images=images,
                feature_ids=feature_ids,
                intervention_mode=args.intervention_mode,
                amplify_factor=args.amplify_factor,
                absolute_value=args.absolute_value,
                args=args,
            )

            past_eval = batch["PAST"][accepted_indices].detach().cpu()
            baseline_endpoint = endpoint_distance(baseline_traj, past_eval)
            patched_endpoint = endpoint_distance(patched_traj, past_eval)
            baseline_path = path_length(baseline_traj, past_eval)
            patched_path = path_length(patched_traj, past_eval)
            baseline_disp = mean_displacement(baseline_traj, past_eval)
            patched_disp = mean_displacement(patched_traj, past_eval)
            baseline_tail_speed = predicted_tail_speed(
                baseline_traj,
                past_eval,
                dt=dt,
                tail_steps=tail_steps,
            )
            patched_tail_speed = predicted_tail_speed(
                patched_traj,
                past_eval,
                dt=dt,
                tail_steps=tail_steps,
            )

            for local_idx, row in enumerate(accepted_rows):
                results.append(
                    {
                        **row,
                        "baseline_endpoint_distance": float(baseline_endpoint[local_idx].item()),
                        "patched_endpoint_distance": float(patched_endpoint[local_idx].item()),
                        "endpoint_distance_diff": float(
                            (patched_endpoint[local_idx] - baseline_endpoint[local_idx]).item()
                        ),
                        "baseline_path_length": float(baseline_path[local_idx].item()),
                        "patched_path_length": float(patched_path[local_idx].item()),
                        "path_length_diff": float((patched_path[local_idx] - baseline_path[local_idx]).item()),
                        "baseline_mean_displacement": float(baseline_disp[local_idx].item()),
                        "patched_mean_displacement": float(patched_disp[local_idx].item()),
                        "mean_displacement_diff": float(
                            (patched_disp[local_idx] - baseline_disp[local_idx]).item()
                        ),
                        "baseline_tail_speed": float(baseline_tail_speed[local_idx].item()),
                        "patched_tail_speed": float(patched_tail_speed[local_idx].item()),
                        "tail_speed_diff": float(
                            (patched_tail_speed[local_idx] - baseline_tail_speed[local_idx]).item()
                        ),
                        "baseline_predicted_stop": bool(
                            baseline_tail_speed[local_idx].item() <= stop_speed_thresh
                        ),
                        "patched_predicted_stop": bool(
                            patched_tail_speed[local_idx].item() <= stop_speed_thresh
                        ),
                    }
                )

            eligible_frames += len(accepted_rows)
            pbar.update(len(accepted_rows))
            running_dataset_idx += batch_size
    pbar.close()

    if not results:
        raise RuntimeError("No eligible frames were evaluated")

    endpoint_diff = torch.tensor([row["endpoint_distance_diff"] for row in results], dtype=torch.float64)
    path_diff = torch.tensor([row["path_length_diff"] for row in results], dtype=torch.float64)
    disp_diff = torch.tensor([row["mean_displacement_diff"] for row in results], dtype=torch.float64)
    tail_speed_diff = torch.tensor([row["tail_speed_diff"] for row in results], dtype=torch.float64)
    baseline_endpoint = torch.tensor([row["baseline_endpoint_distance"] for row in results], dtype=torch.float64)
    patched_endpoint = torch.tensor([row["patched_endpoint_distance"] for row in results], dtype=torch.float64)
    baseline_path = torch.tensor([row["baseline_path_length"] for row in results], dtype=torch.float64)
    patched_path = torch.tensor([row["patched_path_length"] for row in results], dtype=torch.float64)
    baseline_disp = torch.tensor([row["baseline_mean_displacement"] for row in results], dtype=torch.float64)
    patched_disp = torch.tensor([row["patched_mean_displacement"] for row in results], dtype=torch.float64)
    baseline_tail = torch.tensor([row["baseline_tail_speed"] for row in results], dtype=torch.float64)
    patched_tail = torch.tensor([row["patched_tail_speed"] for row in results], dtype=torch.float64)
    baseline_stop = torch.tensor([row["baseline_predicted_stop"] for row in results], dtype=torch.float64)
    patched_stop = torch.tensor([row["patched_predicted_stop"] for row in results], dtype=torch.float64)

    results.sort(key=lambda row: row["endpoint_distance_diff"])
    summary = {
        "analysis_json": str(analysis_json_path),
        "detections": str(detections_path),
        "sae_checkpoint_path": str(sae_checkpoint_path),
        "model_checkpoint_path": str(model_checkpoint_path),
        "data_dir": str(data_dir),
        "index_file": str(index_file),
        "split": split,
        "object_label_name": label_name,
        "object_label_id": label_id,
        "feature_ids": feature_ids,
        "feature_key": args.feature_key,
        "feature_count": args.feature_count,
        "intervention_mode": args.intervention_mode,
        "amplify_factor": args.amplify_factor,
        "camera_indices": camera_indices,
        "dt": dt,
        "tail_steps": tail_steps,
        "moving_speed_thresh": moving_speed_thresh,
        "stop_speed_thresh": stop_speed_thresh,
        "require_object_presence": args.require_object_presence,
        "raw_frames_scanned": raw_frames_scanned,
        "eligible_frames_evaluated": len(results),
        "missing_detection_frames": missing_detection_frames,
        "skipped_no_object": skipped_no_object,
        "skipped_low_speed": skipped_low_speed,
        "baseline_means": {
            "endpoint_distance": float(baseline_endpoint.mean().item()),
            "path_length": float(baseline_path.mean().item()),
            "mean_displacement": float(baseline_disp.mean().item()),
            "tail_speed": float(baseline_tail.mean().item()),
            "predicted_stop_rate": float(baseline_stop.mean().item()),
        },
        "patched_means": {
            "endpoint_distance": float(patched_endpoint.mean().item()),
            "path_length": float(patched_path.mean().item()),
            "mean_displacement": float(patched_disp.mean().item()),
            "tail_speed": float(patched_tail.mean().item()),
            "predicted_stop_rate": float(patched_stop.mean().item()),
        },
        "paired_tests": {
            "endpoint_distance": paired_stat_summary(endpoint_diff),
            "path_length": paired_stat_summary(path_diff),
            "mean_displacement": paired_stat_summary(disp_diff),
            "tail_speed": paired_stat_summary(tail_speed_diff),
            "predicted_stop_indicator_diff": paired_stat_summary(patched_stop - baseline_stop),
        },
        "top_endpoint_reductions": results[:25],
        "top_endpoint_increases": list(reversed(results[-25:])),
    }

    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    print(f"Evaluated {len(results)} eligible frame(s)")
    print(
        "Endpoint distance: "
        f"{summary['baseline_means']['endpoint_distance']:.4f} -> "
        f"{summary['patched_means']['endpoint_distance']:.4f} "
        f"(mean diff {summary['paired_tests']['endpoint_distance']['mean_diff']:.4f}, "
        f"p_decrease={summary['paired_tests']['endpoint_distance']['p_value_one_sided_decrease']:.3g})"
    )
    print(
        "Path length: "
        f"{summary['baseline_means']['path_length']:.4f} -> "
        f"{summary['patched_means']['path_length']:.4f} "
        f"(mean diff {summary['paired_tests']['path_length']['mean_diff']:.4f}, "
        f"p_decrease={summary['paired_tests']['path_length']['p_value_one_sided_decrease']:.3g})"
    )
    print(
        "Tail speed: "
        f"{summary['baseline_means']['tail_speed']:.4f} -> "
        f"{summary['patched_means']['tail_speed']:.4f} "
        f"(mean diff {summary['paired_tests']['tail_speed']['mean_diff']:.4f}, "
        f"p_decrease={summary['paired_tests']['tail_speed']['p_value_one_sided_decrease']:.3g})"
    )
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
