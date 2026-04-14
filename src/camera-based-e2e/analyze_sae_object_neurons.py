"""
Find SAE features that differ most between frames with and without a target object.
"""


import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from loader import WaymoE2E, collate_with_images
from models.base_model import LitModel
from models.feature_extractors import SAMFeatures
from models.monocular import DeepMonocularModel
from sparseAE import SparseAE


def get_sae_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        checkpoint_obj = checkpoint_obj["state_dict"]
    if not isinstance(checkpoint_obj, dict):
        raise TypeError("Unsupported SAE checkpoint format")
    return checkpoint_obj


def default_index_file(split: str) -> str:
    if split == "val":
        return "index_val.pkl"
    if split == "test":
        return "index_test.pkl"
    return "index_train.pkl"


def resolve_object_label(
    categories: Sequence[str],
    object_name: Optional[str],
    label_id: Optional[int],
) -> Tuple[int, str]:
    if label_id is not None:
        label_name = categories[label_id] if 0 <= label_id < len(categories) else f"label_{label_id}"
        return label_id, label_name
    if object_name is None:
        raise ValueError("Pass either --object_name or --label_id")

    lowered = object_name.strip().lower()
    for idx, category in enumerate(categories):
        if category.lower() == lowered:
            return idx, category
    raise ValueError(f"Object '{object_name}' was not found in detector categories")


def load_detection_artifacts(path_pattern: str) -> Tuple[Sequence[str], Dict[str, dict]]:
    matches = sorted(glob.glob(path_pattern))
    if not matches:
        raise FileNotFoundError(f"No detection artifacts matched {path_pattern}")

    categories = None
    frame_to_record = {}
    for path in matches:
        artifact = torch.load(path, map_location="cpu")
        if categories is None:
            categories = artifact["categories"]
        elif list(categories) != list(artifact["categories"]):
            raise ValueError(f"Category mismatch across detection artifacts: {path}")

        for record in artifact["records"]:
            frame_to_record[record["frame_name"]] = record

    if categories is None:
        raise RuntimeError("No detector categories were loaded")

    return categories, frame_to_record


def frame_has_object(
    record: dict,
    label_id: int,
    score_thresh: float,
    camera_indices: Optional[Sequence[int]],
) -> bool:
    detections = record["detections"]
    allowed = None if camera_indices is None else {str(cam_idx) for cam_idx in camera_indices}
    for cam_idx, det in detections.items():
        if allowed is not None and cam_idx not in allowed:
            continue
        for det_label, det_score in zip(det["labels"], det["scores"]):
            if det_label == label_id and det_score >= score_thresh:
                return True
    return False


def compute_hidden_activations(sae: SparseAE, hooked_acts: torch.Tensor) -> torch.Tensor:
    x = hooked_acts
    x_centered = x - sae.decoder.bias
    hidden = torch.relu(sae.encoder(x_centered))
    return hidden.reshape(hidden.size(0), -1)


def future_tail_speed(
    past: torch.Tensor,
    future: torch.Tensor,
    dt: float,
    tail_steps: int,
) -> float:
    if future.ndim != 2 or future.size(-1) != 2:
        raise ValueError(f"Expected future shape (T, 2), got {tuple(future.shape)}")

    if tail_steps <= 0:
        raise ValueError(f"tail_steps must be > 0, got {tail_steps}")

    prev = torch.cat([past[-1:, :2], future[:-1]], dim=0)
    step_speeds = torch.norm(future - prev, dim=-1) / dt
    tail = step_speeds[-min(tail_steps, step_speeds.numel()) :]
    return float(tail.mean().item())


def past_current_speed(past: torch.Tensor) -> float:
    if past.ndim != 2 or past.size(-1) < 4:
        raise ValueError(f"Expected past shape (T, >=4), got {tuple(past.shape)}")
    return float(torch.norm(past[-1, 2:4], dim=-1).item())


def classify_sample(
    *,
    has_object: bool,
    past: torch.Tensor,
    future: torch.Tensor,
    label_mode: str,
    dt: float,
    tail_steps: int,
    moving_speed_thresh: float,
    stop_speed_thresh: float,
) -> Optional[bool]:
    if label_mode == "object_presence":
        return has_object

    if not has_object:
        return None

    current_speed = past_current_speed(past)
    tail_speed = future_tail_speed(past, future, dt=dt, tail_steps=tail_steps)

    if current_speed < moving_speed_thresh:
        return None

    if label_mode == "stop_with_object":
        return tail_speed <= stop_speed_thresh
    if label_mode == "move_with_object":
        return tail_speed > stop_speed_thresh

    raise ValueError(f"Unsupported label_mode: {label_mode}")


def summarize_features(
    positive_sum: torch.Tensor,
    negative_sum: torch.Tensor,
    positive_active_sum: torch.Tensor,
    negative_active_sum: torch.Tensor,
    positive_count: int,
    negative_count: int,
    top_k: int,
) -> dict:
    positive_mean = positive_sum / max(positive_count, 1)
    negative_mean = negative_sum / max(negative_count, 1)
    delta_mean = positive_mean - negative_mean

    positive_active_rate = positive_active_sum / max(positive_count, 1)
    negative_active_rate = negative_active_sum / max(negative_count, 1)
    delta_active_rate = positive_active_rate - negative_active_rate

    top_positive = torch.topk(delta_mean, k=min(top_k, delta_mean.numel())).indices.tolist()
    top_negative = torch.topk(-delta_mean, k=min(top_k, delta_mean.numel())).indices.tolist()

    def build_rows(indices: List[int]) -> List[dict]:
        rows = []
        for idx in indices:
            rows.append(
                {
                    "feature_idx": int(idx),
                    "positive_mean": float(positive_mean[idx].item()),
                    "negative_mean": float(negative_mean[idx].item()),
                    "delta_mean": float(delta_mean[idx].item()),
                    "positive_active_rate": float(positive_active_rate[idx].item()),
                    "negative_active_rate": float(negative_active_rate[idx].item()),
                    "delta_active_rate": float(delta_active_rate[idx].item()),
                }
            )
        return rows

    return {
        "top_positive_association": build_rows(top_positive),
        "top_negative_association": build_rows(top_negative),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=str, required=True, help="Glob or path for saved detection artifacts")
    parser.add_argument("--sae_checkpoint_path", type=str, required=True, help="Trained SparseAE checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Waymo dataset directory")
    parser.add_argument("--object_name", type=str, default=None, help="Detector category name to analyze")
    parser.add_argument("--label_id", type=int, default=None, help="Detector category id to analyze")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the analysis JSON")
    parser.add_argument("--model_checkpoint_path", default="./pretrained/camera-e2e-epoch=04-val_loss=2.90.ckpt", type=str)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--index_file", type=str, default=None, help="Override split index file")
    parser.add_argument("--n_items", type=int, default=None, help="Number of frames from the head of the split")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--block_idx", type=int, default=3, help="Transformer block index whose mlp[2] is SAE-modeled")
    parser.add_argument("--score_thresh", type=float, default=0.4, help="Detection score threshold for object presence")
    parser.add_argument("--camera_indices", type=str, default="1", help="Comma-separated camera subset to score")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument(
        "--label_mode",
        type=str,
        default="object_presence",
        choices=["object_presence", "stop_with_object", "move_with_object"],
        help=(
            "How to build positive/negative sets. "
            "'object_presence' compares object vs non-object frames. "
            "'stop_with_object' compares stopping vs non-stopping samples within object-present frames. "
            "'move_with_object' flips that polarity."
        ),
    )
    parser.add_argument("--dt", type=float, default=0.25, help="Trajectory step size in seconds")
    parser.add_argument(
        "--tail_steps",
        type=int,
        default=4,
        help="Number of future steps from the tail used to estimate final speed",
    )
    parser.add_argument(
        "--moving_speed_thresh",
        type=float,
        default=1.0,
        help="Minimum current speed required to include a sample in behavior-conditioned modes",
    )
    parser.add_argument(
        "--stop_speed_thresh",
        type=float,
        default=0.5,
        help="Tail speed threshold in m/s used to define stopping behavior",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    camera_indices = None
    if args.camera_indices:
        camera_indices = [int(piece.strip()) for piece in args.camera_indices.split(",") if piece.strip()]

    categories, frame_to_record = load_detection_artifacts(args.detections)
    label_id, label_name = resolve_object_label(categories, args.object_name, args.label_id)

    submodel = DeepMonocularModel(
        feature_extractor=SAMFeatures(
            model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True
        ),
        out_dim=40,
        n_blocks=4,
    )
    model = LitModel.load_from_checkpoint(args.model_checkpoint_path, model=submodel)
    model = model.to(device)
    model.eval()

    target_layer = model.model.blocks[args.block_idx].mlp[2]

    sae_checkpoint = torch.load(args.sae_checkpoint_path, map_location="cpu")
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
    )

    positive_sum = torch.zeros(dict_size, dtype=torch.float64)
    negative_sum = torch.zeros(dict_size, dtype=torch.float64)
    positive_active_sum = torch.zeros(dict_size, dtype=torch.float64)
    negative_active_sum = torch.zeros(dict_size, dtype=torch.float64)
    positive_count = 0
    negative_count = 0
    missing_detection_frames = 0
    skipped_behavior_frames = 0

    with torch.no_grad():
        for batch in loader:
            past = batch["PAST"].to(device)
            intent = batch["INTENT"].to(device)
            images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
            model_inputs = {"PAST": past, "IMAGES": images, "INTENT": intent}

            sae.internal_acts = None
            _ = model(model_inputs)
            if sae.internal_acts is None:
                raise RuntimeError("No activations captured from target model hook")

            hidden = compute_hidden_activations(sae, sae.internal_acts).detach().cpu().to(torch.float64)

            for sample_idx, frame_name in enumerate(batch["NAME"]):
                record = frame_to_record.get(frame_name)
                if record is None:
                    missing_detection_frames += 1
                    continue

                has_object = frame_has_object(
                    record=record,
                    label_id=label_id,
                    score_thresh=args.score_thresh,
                    camera_indices=camera_indices,
                )
                is_positive = classify_sample(
                    has_object=has_object,
                    past=past[sample_idx].detach().cpu(),
                    future=batch["FUTURE"][sample_idx].detach().cpu(),
                    label_mode=args.label_mode,
                    dt=args.dt,
                    tail_steps=args.tail_steps,
                    moving_speed_thresh=args.moving_speed_thresh,
                    stop_speed_thresh=args.stop_speed_thresh,
                )
                if is_positive is None:
                    skipped_behavior_frames += 1
                    continue
                feature_row = hidden[sample_idx]
                active_row = (feature_row > 0).to(torch.float64)

                if is_positive:
                    positive_sum += feature_row
                    positive_active_sum += active_row
                    positive_count += 1
                else:
                    negative_sum += feature_row
                    negative_active_sum += active_row
                    negative_count += 1

    if positive_count == 0 or negative_count == 0:
        raise RuntimeError(
            f"Need both positive and negative examples, got positive={positive_count}, negative={negative_count}"
        )

    summary = summarize_features(
        positive_sum=positive_sum,
        negative_sum=negative_sum,
        positive_active_sum=positive_active_sum,
        negative_active_sum=negative_active_sum,
        positive_count=positive_count,
        negative_count=negative_count,
        top_k=args.top_k,
    )

    output = {
        "object_label_id": label_id,
        "object_label_name": label_name,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "missing_detection_frames": missing_detection_frames,
        "skipped_behavior_frames": skipped_behavior_frames,
        "score_thresh": args.score_thresh,
        "split": args.split,
        "top_k": args.top_k,
        "label_mode": args.label_mode,
        "dt": args.dt,
        "tail_steps": args.tail_steps,
        "moving_speed_thresh": args.moving_speed_thresh,
        "stop_speed_thresh": args.stop_speed_thresh,
        **summary,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print(
        f"Analyzed '{label_name}' (label_id={label_id}, label_mode={args.label_mode}) with "
        f"{positive_count} positive and {negative_count} negative frame(s)"
    )
    if skipped_behavior_frames:
        print(f"Skipped {skipped_behavior_frames} frame(s) due to behavior-mode filters")
    print(f"Saved SAE object-neuron analysis to {output_path}")


if __name__ == "__main__":
    main()
