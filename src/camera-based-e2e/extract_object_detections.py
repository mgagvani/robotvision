"""Run an object detector on selected cameras and save detections."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)

from loader import WaymoE2E, collate_with_images


DETECTOR_REGISTRY = {
    "fasterrcnn_resnet50_fpn_v2": (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    ),
    "retinanet_resnet50_fpn_v2": (
        retinanet_resnet50_fpn_v2,
        RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
    ),
}


def parse_camera_indices(raw: str) -> List[int]:
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def decode_sample_images(
    images_jpeg: List[List[torch.Tensor]],
    sample_idx: int,
    camera_indices: Iterable[int],
    device: torch.device,
) -> List[torch.Tensor]:
    decoded = []
    for cam_idx in camera_indices:
        jpeg = images_jpeg[sample_idx][cam_idx]
        image = torchvision.io.decode_jpeg(
            jpeg if isinstance(jpeg, torch.Tensor) else torch.as_tensor(jpeg, dtype=torch.uint8),
            mode=torchvision.io.ImageReadMode.RGB,
            device=device,
        )
        decoded.append(image.float().div(255.0))
    return decoded


def build_detector(name: str, device: torch.device):
    model_fn, weights = DETECTOR_REGISTRY[name]
    model = model_fn(weights=weights, box_score_thresh=0.0)
    model.to(device)
    model.eval()
    return model, weights.meta["categories"]


def default_index_file(split: str) -> str:
    if split == "val":
        return "index_val.pkl"
    if split == "test":
        return "index_test.pkl"
    return "index_train.pkl"


def serialize_prediction(
    prediction: Dict[str, torch.Tensor],
    categories: List[str],
    score_thresh: float,
    max_detections: int,
) -> Dict[str, List]:
    scores = prediction["scores"].detach().cpu()
    keep = scores >= score_thresh
    if max_detections > 0:
        topk = min(max_detections, int(keep.sum().item()))
        if topk > 0:
            keep_indices = torch.nonzero(keep, as_tuple=False).flatten()[:topk]
            keep = torch.zeros_like(keep, dtype=torch.bool)
            keep[keep_indices] = True

    boxes = prediction["boxes"][keep].detach().cpu().tolist()
    labels = prediction["labels"][keep].detach().cpu().tolist()
    kept_scores = scores[keep].tolist()
    label_names = [categories[label] if 0 <= label < len(categories) else f"label_{label}" for label in labels]
    return {
        "boxes": boxes,
        "labels": labels,
        "label_names": label_names,
        "scores": kept_scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Waymo dataset directory")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the detection artifact")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--index_file", type=str, default=None, help="Override split index file")
    parser.add_argument("--detector", type=str, default="fasterrcnn_resnet50_fpn_v2", choices=sorted(DETECTOR_REGISTRY))
    parser.add_argument("--camera_indices", type=str, default="1", help="Comma-separated camera indices to analyze")
    parser.add_argument("--n_items", type=int, default=None, help="Number of frames from the head of the split")
    parser.add_argument("--batch_size", type=int, default=4, help="Dataset batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--score_thresh", type=float, default=0.4, help="Minimum detector score to keep")
    parser.add_argument("--max_detections", type=int, default=50, help="Max detections to keep per image")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu, or an explicit device string")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    camera_indices = parse_camera_indices(args.camera_indices)
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

    detector, categories = build_detector(args.detector, device)
    records = []

    with torch.inference_mode():
        for batch_start, batch in enumerate(loader):
            batch_size = len(batch["NAME"])
            flattened_images: List[torch.Tensor] = []
            image_keys = []
            for sample_idx in range(batch_size):
                decoded_images = decode_sample_images(batch["IMAGES_JPEG"], sample_idx, camera_indices, device)
                flattened_images.extend(decoded_images)
                image_keys.extend((sample_idx, cam_idx) for cam_idx in camera_indices)

            predictions = detector(flattened_images)

            per_sample = {
                sample_idx: {
                    "frame_name": batch["NAME"][sample_idx],
                    "detections": {},
                }
                for sample_idx in range(batch_size)
            }

            for (sample_idx, cam_idx), prediction in zip(image_keys, predictions):
                per_sample[sample_idx]["detections"][str(cam_idx)] = serialize_prediction(
                    prediction=prediction,
                    categories=categories,
                    score_thresh=args.score_thresh,
                    max_detections=args.max_detections,
                )

            records.extend(per_sample[sample_idx] for sample_idx in range(batch_size))
            print(f"Processed detection batch {batch_start + 1} with {batch_size} frame(s)")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": args.split,
            "index_file": index_file,
            "detector": args.detector,
            "categories": categories,
            "camera_indices": camera_indices,
            "score_thresh": args.score_thresh,
            "records": records,
        },
        output_path,
    )

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "detector": args.detector,
                "num_records": len(records),
                "camera_indices": camera_indices,
                "categories": categories,
            },
            indent=2,
        )
    )
    print(f"Saved {len(records)} frame detections to {output_path}")
    print(f"Wrote metadata summary to {summary_path}")


if __name__ == "__main__":
    main()
