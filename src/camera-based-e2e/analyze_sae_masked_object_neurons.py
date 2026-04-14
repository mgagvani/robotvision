"""
Measure SAE feature changes when detector boxes are masked from camera frames.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from analyze_sae_object_neurons import (
    compute_hidden_activations,
    default_index_file,
    get_sae_state_dict,
    load_detection_artifacts,
    resolve_object_label,
)
from loader import WaymoE2E, collate_with_images
from models.base_model import LitModel
from models.feature_extractors import SAMFeatures
from models.monocular import DeepMonocularModel
from sparseAE import SparseAE

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


def parse_csv_strings(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def parse_csv_ints(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def resolve_object_labels_multi(
    categories: Sequence[str],
    object_names: Sequence[str],
    label_ids: Sequence[int],
) -> Tuple[List[int], List[str]]:
    resolved_ids: List[int] = []
    resolved_names: List[str] = []
    seen_ids = set()

    for label_id in label_ids:
        _, label_name = resolve_object_label(categories, None, int(label_id))
        if label_id in seen_ids:
            continue
        seen_ids.add(label_id)
        resolved_ids.append(int(label_id))
        resolved_names.append(label_name)

    for object_name in object_names:
        label_id, label_name = resolve_object_label(categories, object_name, None)
        if label_id in seen_ids:
            continue
        seen_ids.add(label_id)
        resolved_ids.append(label_id)
        resolved_names.append(label_name)

    if not resolved_ids:
        raise ValueError("Pass at least one label via --object_names or --label_ids")

    return resolved_ids, resolved_names


def select_best_trajectory(output: dict) -> torch.Tensor:
    pred = output["trajectory"]
    scores = output.get("scores", None)
    bsz = pred.size(0)
    pred = pred.view(bsz, -1, pred.size(-2), 2)
    if scores is not None and pred.size(1) > 1:
        best_idx = scores.argmin(dim=1)
    else:
        best_idx = torch.zeros(bsz, dtype=torch.long, device=pred.device)
    return pred[torch.arange(bsz, device=pred.device), best_idx]


def prepare_model_and_sae(
    model_checkpoint_path: str,
    sae_checkpoint_path: str,
    block_idx: int,
    device: torch.device,
):
    submodel = DeepMonocularModel(
        feature_extractor=SAMFeatures(
            model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True
        ),
        out_dim=40,
        n_blocks=4,
    )
    model = LitModel.load_from_checkpoint(model_checkpoint_path, model=submodel)
    model = model.to(device)
    model.eval()

    target_layer = model.model.blocks[block_idx].mlp[2]
    sae_checkpoint = torch.load(sae_checkpoint_path, map_location="cpu")
    sae_state = get_sae_state_dict(sae_checkpoint)
    encoder_weight = sae_state["encoder.weight"]
    dict_size, input_dim = encoder_weight.shape
    sae = SparseAE.build_from_state_dict(
        sae_state,
        target_model=model,
        input_dim=input_dim,
        dict_size=dict_size,
        compile_sae=False,
    )
    sae = sae.to(device)
    sae.eval()
    target_layer.register_forward_hook(sae.hook_fn)
    return model, sae, dict_size


def filter_detection_matches(
    detection_record: dict,
    target_label_ids: Sequence[int],
    score_thresh: float,
) -> Tuple[List[List[float]], List[int], List[str], List[float]]:
    target_set = {int(label_id) for label_id in target_label_ids}
    boxes: List[List[float]] = []
    labels: List[int] = []
    label_names: List[str] = []
    scores: List[float] = []
    for box, label, label_name, score in zip(
        detection_record.get("boxes", []),
        detection_record.get("labels", []),
        detection_record.get("label_names", []),
        detection_record.get("scores", []),
    ):
        if int(label) not in target_set or float(score) < score_thresh:
            continue
        boxes.append(box)
        labels.append(int(label))
        label_names.append(label_name)
        scores.append(float(score))
    return boxes, labels, label_names, scores


def collect_mask_boxes(
    record: Optional[dict],
    target_label_ids: Sequence[int],
    score_thresh: float,
    camera_indices: Optional[Sequence[int]],
) -> Tuple[Dict[int, List[List[float]]], List[str], int]:
    if record is None:
        return {}, [], 0

    allowed = None if camera_indices is None else {str(cam_idx) for cam_idx in camera_indices}
    boxes_by_camera: Dict[int, List[List[float]]] = {}
    present_names = set()
    total_boxes = 0

    for cam_idx_str, detection_record in record["detections"].items():
        if allowed is not None and cam_idx_str not in allowed:
            continue
        boxes, _, label_names, _ = filter_detection_matches(
            detection_record=detection_record,
            target_label_ids=target_label_ids,
            score_thresh=score_thresh,
        )
        if not boxes:
            continue
        cam_idx = int(cam_idx_str)
        boxes_by_camera[cam_idx] = boxes
        present_names.update(label_names)
        total_boxes += len(boxes)

    return boxes_by_camera, sorted(present_names), total_boxes


def clamp_box(box: Sequence[float], height: int, width: int, margin: int = 0) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    left = max(0, min(width, math.floor(float(x1)) - margin))
    top = max(0, min(height, math.floor(float(y1)) - margin))
    right = max(0, min(width, math.ceil(float(x2)) + margin))
    bottom = max(0, min(height, math.ceil(float(y2)) + margin))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def mask_boxes_on_tensor(
    image: torch.Tensor,
    boxes: Sequence[Sequence[float]],
    fill_value: float = 0.0,
    margin: int = 0,
) -> torch.Tensor:
    masked = image.clone()
    _, height, width = masked.shape
    for box in boxes:
        clamped = clamp_box(box, height=height, width=width, margin=margin)
        if clamped is None:
            continue
        left, top, right, bottom = clamped
        masked[:, top:bottom, left:right] = fill_value
    return masked


def apply_detection_masks(
    decoded_images: Sequence[torch.Tensor],
    frame_names: Sequence[str],
    frame_to_record: Dict[str, dict],
    target_label_ids: Sequence[int],
    score_thresh: float,
    camera_indices: Optional[Sequence[int]],
    fill_value: float = 0.0,
    margin: int = 0,
) -> Tuple[List[torch.Tensor], List[int], List[List[str]]]:
    masked_images = [camera_batch.clone() for camera_batch in decoded_images]
    masked_box_counts = [0 for _ in frame_names]
    masked_label_names: List[List[str]] = [[] for _ in frame_names]

    for sample_idx, frame_name in enumerate(frame_names):
        boxes_by_camera, present_names, total_boxes = collect_mask_boxes(
            record=frame_to_record.get(frame_name),
            target_label_ids=target_label_ids,
            score_thresh=score_thresh,
            camera_indices=camera_indices,
        )
        masked_box_counts[sample_idx] = total_boxes
        masked_label_names[sample_idx] = present_names
        if total_boxes == 0:
            continue
        for cam_idx, boxes in boxes_by_camera.items():
            if cam_idx >= len(masked_images):
                continue
            masked_images[cam_idx][sample_idx] = mask_boxes_on_tensor(
                masked_images[cam_idx][sample_idx],
                boxes=boxes,
                fill_value=fill_value,
                margin=margin,
            )

    return masked_images, masked_box_counts, masked_label_names


def run_model_hidden_and_trajectory(
    model: LitModel,
    sae: SparseAE,
    past: torch.Tensor,
    intent: torch.Tensor,
    images: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    sae.internal_acts = None
    outputs = model({"PAST": past, "IMAGES": images, "INTENT": intent})
    if sae.internal_acts is None:
        raise RuntimeError("No activations captured from target model hook")
    hidden = compute_hidden_activations(sae, sae.internal_acts).detach().cpu().to(torch.float64)
    trajectory = select_best_trajectory(outputs).detach().cpu()
    return hidden, trajectory


def summarize_masking_effects(
    baseline_sum: torch.Tensor,
    masked_sum: torch.Tensor,
    baseline_active_sum: torch.Tensor,
    masked_active_sum: torch.Tensor,
    frame_count: int,
    top_k: int,
) -> dict:
    baseline_mean = baseline_sum / max(frame_count, 1)
    masked_mean = masked_sum / max(frame_count, 1)
    delta_mean = baseline_mean - masked_mean

    baseline_active_rate = baseline_active_sum / max(frame_count, 1)
    masked_active_rate = masked_active_sum / max(frame_count, 1)
    delta_active_rate = baseline_active_rate - masked_active_rate

    top_reduced = torch.topk(delta_mean, k=min(top_k, delta_mean.numel())).indices.tolist()
    top_increased = torch.topk(-delta_mean, k=min(top_k, delta_mean.numel())).indices.tolist()

    def build_rows(indices: List[int]) -> List[dict]:
        rows = []
        for idx in indices:
            rows.append(
                {
                    "feature_idx": int(idx),
                    "baseline_mean": float(baseline_mean[idx].item()),
                    "masked_mean": float(masked_mean[idx].item()),
                    "delta_mean": float(delta_mean[idx].item()),
                    "baseline_active_rate": float(baseline_active_rate[idx].item()),
                    "masked_active_rate": float(masked_active_rate[idx].item()),
                    "delta_active_rate": float(delta_active_rate[idx].item()),
                }
            )
        return rows

    return {
        "top_reduced_by_masking": build_rows(top_reduced),
        "top_increased_by_masking": build_rows(top_increased),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=str, required=True, help="Glob or path for saved detection artifacts")
    parser.add_argument("--sae_checkpoint_path", type=str, required=True, help="Trained SparseAE checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Waymo dataset directory")
    parser.add_argument(
        "--object_names",
        type=str,
        default="stop sign,traffic light",
        help="Comma-separated detector category names to mask",
    )
    parser.add_argument(
        "--label_ids",
        type=str,
        default=None,
        help="Optional comma-separated detector label ids to mask",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the analysis JSON")
    parser.add_argument("--model_checkpoint_path", default="./pretrained/camera-e2e-epoch=04-val_loss=2.90.ckpt", type=str)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--index_file", type=str, default=None, help="Override split index file")
    parser.add_argument("--n_items", type=int, default=None, help="Number of frames from the head of the split")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--block_idx", type=int, default=3, help="Transformer block index whose mlp[2] is SAE-modeled")
    parser.add_argument("--score_thresh", type=float, default=0.4, help="Detection score threshold for masking")
    parser.add_argument("--camera_indices", type=str, default="1", help="Comma-separated camera subset to mask")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--mask_fill_value", type=float, default=0.0, help="Pixel value used inside masked boxes")
    parser.add_argument("--mask_margin", type=int, default=0, help="Extra pixels to expand each masked box")
    parser.add_argument(
        "--top_examples",
        type=int,
        default=25,
        help="Number of highest trajectory-shift examples to keep in the JSON summary",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    camera_indices = parse_csv_ints(args.camera_indices)
    camera_indices = camera_indices if camera_indices else None

    categories, frame_to_record = load_detection_artifacts(args.detections)
    object_label_ids, object_label_names = resolve_object_labels_multi(
        categories=categories,
        object_names=parse_csv_strings(args.object_names),
        label_ids=parse_csv_ints(args.label_ids),
    )

    model, sae, dict_size = prepare_model_and_sae(
        model_checkpoint_path=args.model_checkpoint_path,
        sae_checkpoint_path=args.sae_checkpoint_path,
        block_idx=args.block_idx,
        device=device,
    )

    index_file = args.index_file or default_index_file(args.split)
    dataset = WaymoE2E(indexFile=index_file, data_dir=args.data_dir, n_items=None)
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

    baseline_sum = torch.zeros(dict_size, dtype=torch.float64)
    masked_sum = torch.zeros(dict_size, dtype=torch.float64)
    baseline_active_sum = torch.zeros(dict_size, dtype=torch.float64)
    masked_active_sum = torch.zeros(dict_size, dtype=torch.float64)
    masked_frame_count = 0
    total_masked_boxes = 0
    missing_detection_frames = 0
    skipped_unmasked_frames = 0
    trajectory_shift_sum = 0.0
    top_examples: List[dict] = []
    running_dataset_idx = 0

    with torch.no_grad():
        for batch in loader:
            frame_names = batch["NAME"]
            batch_size = len(frame_names)

            missing_in_batch = 0
            unmasked_in_batch = 0
            record_cache: List[Optional[dict]] = []
            for frame_name in frame_names:
                record = frame_to_record.get(frame_name)
                record_cache.append(record)
                if record is None:
                    missing_in_batch += 1
                    continue
                _, _, total_boxes = collect_mask_boxes(
                    record=record,
                    target_label_ids=object_label_ids,
                    score_thresh=args.score_thresh,
                    camera_indices=camera_indices,
                )
                if total_boxes == 0:
                    unmasked_in_batch += 1

            missing_detection_frames += missing_in_batch
            skipped_unmasked_frames += unmasked_in_batch
            if missing_in_batch + unmasked_in_batch == batch_size:
                running_dataset_idx += batch_size
                continue

            past = batch["PAST"].to(device)
            intent = batch["INTENT"].to(device)
            images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
            masked_images, masked_box_counts, masked_label_names = apply_detection_masks(
                decoded_images=images,
                frame_names=frame_names,
                frame_to_record=frame_to_record,
                target_label_ids=object_label_ids,
                score_thresh=args.score_thresh,
                camera_indices=camera_indices,
                fill_value=args.mask_fill_value,
                margin=args.mask_margin,
            )

            baseline_hidden, baseline_traj = run_model_hidden_and_trajectory(
                model=model,
                sae=sae,
                past=past,
                intent=intent,
                images=images,
            )
            masked_hidden, masked_traj = run_model_hidden_and_trajectory(
                model=model,
                sae=sae,
                past=past,
                intent=intent,
                images=masked_images,
            )

            trajectory_shift = torch.norm(baseline_traj - masked_traj, dim=-1).mean(dim=-1)
            for sample_idx, frame_name in enumerate(frame_names):
                if record_cache[sample_idx] is None or masked_box_counts[sample_idx] == 0:
                    continue

                baseline_row = baseline_hidden[sample_idx]
                masked_row = masked_hidden[sample_idx]
                baseline_sum += baseline_row
                masked_sum += masked_row
                baseline_active_sum += (baseline_row > 0).to(torch.float64)
                masked_active_sum += (masked_row > 0).to(torch.float64)
                masked_frame_count += 1
                total_masked_boxes += masked_box_counts[sample_idx]

                traj_shift_value = float(trajectory_shift[sample_idx].item())
                trajectory_shift_sum += traj_shift_value
                top_examples.append(
                    {
                        "frame_name": frame_name,
                        "dataset_idx": running_dataset_idx + sample_idx,
                        "masked_box_count": int(masked_box_counts[sample_idx]),
                        "masked_label_names": masked_label_names[sample_idx],
                        "trajectory_shift_l2_mean": traj_shift_value,
                    }
                )

            running_dataset_idx += batch_size

    if masked_frame_count == 0:
        raise RuntimeError("No frames contained maskable detections for the requested labels")

    top_examples.sort(key=lambda row: row["trajectory_shift_l2_mean"], reverse=True)
    summary = summarize_masking_effects(
        baseline_sum=baseline_sum,
        masked_sum=masked_sum,
        baseline_active_sum=baseline_active_sum,
        masked_active_sum=masked_active_sum,
        frame_count=masked_frame_count,
        top_k=args.top_k,
    )

    output = {
        "object_label_ids": object_label_ids,
        "object_label_names": object_label_names,
        "masked_frame_count": masked_frame_count,
        "average_masked_boxes_per_frame": total_masked_boxes / masked_frame_count,
        "mean_trajectory_shift_l2": trajectory_shift_sum / masked_frame_count,
        "missing_detection_frames": missing_detection_frames,
        "skipped_unmasked_frames": skipped_unmasked_frames,
        "score_thresh": args.score_thresh,
        "split": args.split,
        "top_k": args.top_k,
        "camera_indices": camera_indices,
        "mask_fill_value": args.mask_fill_value,
        "mask_margin": args.mask_margin,
        "delta_definition": "baseline_hidden - masked_hidden",
        "top_masked_examples": top_examples[: args.top_examples],
        **summary,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print(
        "Analyzed masking for "
        f"{', '.join(object_label_names)} across {masked_frame_count} frame(s); "
        f"mean trajectory shift={output['mean_trajectory_shift_l2']:.6f}"
    )
    if missing_detection_frames:
        print(f"Skipped {missing_detection_frames} frame(s) without detection metadata")
    if skipped_unmasked_frames:
        print(f"Skipped {skipped_unmasked_frames} frame(s) without target detections above threshold")
    print(f"Saved masking-based SAE analysis to {output_path}")


if __name__ == "__main__":
    main()
