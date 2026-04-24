#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
E2E_ROOT = REPO_ROOT / "src" / "camera-based-e2e"
if str(E2E_ROOT) not in sys.path:
    sys.path.insert(0, str(E2E_ROOT))

from loader import WaymoE2E  # pylint: disable=wrong-import-position
from models.base_model import collate_with_images  # pylint: disable=wrong-import-position
from models.feature_extractors import SAMFeatures  # pylint: disable=wrong-import-position
from models.monocular import DeepMonocularModel  # pylint: disable=wrong-import-position
from nuscenes_loader import NuScenesDataset  # pylint: disable=wrong-import-position


def _decode_batch_jpeg(images_jpeg: list[list[torch.Tensor]], device: torch.device) -> list[torch.Tensor]:
    flat_encoded = []
    cam_sizes = []
    for cam in images_jpeg:
        cam_sizes.append(len(cam))
        flat_encoded.extend(
            jpg if isinstance(jpg, torch.Tensor) else torch.frombuffer(memoryview(jpg), dtype=torch.uint8)
            for jpg in cam
        )

    flat_decoded = torchvision.io.decode_jpeg(
        flat_encoded,
        mode=torchvision.io.ImageReadMode.UNCHANGED,
        device=device,
    )

    out = []
    idx = 0
    for n in cam_sizes:
        cam_list = flat_decoded[idx : idx + n]
        idx += n
        out.append(torch.stack(cam_list, dim=0))
    return out


def _load_model(checkpoint: str, device: torch.device) -> DeepMonocularModel:
    out_dim = 20 * 2
    model = DeepMonocularModel(
        feature_extractor=SAMFeatures(model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True),
        out_dim=out_dim,
        n_blocks=4,
        n_proposals=50,
    )
    lit_model = LitModel.load_from_checkpoint(
        checkpoint,
        model=model,
        lr=1e-4,
        map_location="cpu",
        weights_only=False,
    )
    model = lit_model.model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _prediction_to_xy(pred: Dict[str, torch.Tensor], horizon: int = 20) -> torch.Tensor:
    traj = pred["trajectory"]  # (B, K*T*2)
    k = pred["scores"].shape[1] if pred.get("scores") is not None else 1
    traj = traj.view(traj.size(0), k, horizon, 2)
    if pred.get("scores") is None:
        best_idx = torch.zeros(traj.size(0), dtype=torch.long, device=traj.device)
    else:
        best_idx = pred["scores"].argmin(dim=1)
    return traj[torch.arange(traj.size(0), device=traj.device), best_idx]


def _ade_fde(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    distances = torch.norm(pred_xy - gt_xy, dim=-1)
    ade = distances.mean(dim=1)
    fde = distances[:, -1]
    return ade, fde


@torch.inference_mode()
def evaluate_open_loop(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint, device)

    if args.dataset == "waymo":
        dataset = WaymoE2E(indexFile=args.index_file, data_dir=args.data_dir, n_items=args.max_items)
    elif args.dataset == "nuscenes":
        dataset = NuScenesDataset(data_dir=args.data_dir, split=args.split, n_items=args.max_items)
    else:
        raise ValueError(f"Unsupported open-loop dataset: {args.dataset}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )

    all_ade = []
    all_fde = []
    sample_count = 0
    for batch in loader:
        past = batch["PAST"].to(device, non_blocking=True)
        future = batch["FUTURE"].to(device, non_blocking=True)
        intent = batch["INTENT"].to(device, non_blocking=True)
        images = _decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)

        pred = model({"PAST": past, "IMAGES": images, "INTENT": intent})
        pred_xy = _prediction_to_xy(pred, horizon=future.shape[1])
        ade, fde = _ade_fde(pred_xy, future)
        all_ade.append(ade.cpu())
        all_fde.append(fde.cpu())
        sample_count += future.shape[0]

    ade_all = torch.cat(all_ade)
    fde_all = torch.cat(all_fde)
    return {
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": int(sample_count),
        "ade_mean": float(ade_all.mean().item()),
        "fde_mean": float(fde_all.mean().item()),
        "ade_p90": float(torch.quantile(ade_all, 0.9).item()),
        "fde_p90": float(torch.quantile(fde_all, 0.9).item()),
    }


def evaluate_navsim(args: argparse.Namespace) -> Dict[str, float]:
    navsim_root = os.environ.get("NAVSIM_DEVKIT_ROOT")
    if not navsim_root:
        raise EnvironmentError("NAVSIM_DEVKIT_ROOT is not set.")

    metric_cache = args.metric_cache_path or f"{os.environ.get('NAVSIM_EXP_ROOT', '/tmp')}/metric_cache"
    cmd = [
        "python",
        f"{navsim_root}/navsim/planning/script/run_pdm_score.py",
        f"train_test_split={args.split}",
        "agent=deep_monocular_agent",
        f"agent.checkpoint_path={args.checkpoint}",
        f"experiment_name={args.experiment_name}",
        f"metric_cache_path={metric_cache}",
        f"synthetic_sensor_path={args.synthetic_sensor_path}",
        f"synthetic_scenes_path={args.synthetic_scenes_path}",
    ]
    subprocess.run(cmd, check=True)

    output_dir = Path(os.environ.get("NAVSIM_EXP_ROOT", ".")) / "pdm_score" / args.experiment_name
    csv_files = sorted(output_dir.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No NAVSIM score CSV found under {output_dir}")
    latest = csv_files[-1]

    final_score = np.nan
    with latest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("token") == "extended_pdm_score_combined":
                final_score = float(row["score"])
                break

    return {
        "dataset": "navsim",
        "split": args.split,
        "score_file": str(latest),
        "extended_pdm_score_combined": float(final_score),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation entrypoint for Waymo, nuScenes, and NAVSIM.")
    parser.add_argument("--dataset", choices=["waymo", "nuscenes", "navsim"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=str(REPO_ROOT / "eval_results.json"))

    # open-loop data args
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--index-file", default="index_val.pkl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=None)

    # navsim args
    parser.add_argument("--experiment-name", default="deep_monocular_eval")
    parser.add_argument("--metric-cache-path", default=None)
    parser.add_argument("--synthetic-sensor-path", default="")
    parser.add_argument("--synthetic-scenes-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset in {"waymo", "nuscenes"}:
        result = evaluate_open_loop(args)
    else:
        if not args.synthetic_sensor_path or not args.synthetic_scenes_path:
            raise ValueError("NAVSIM evaluation requires --synthetic-sensor-path and --synthetic-scenes-path.")
        result = evaluate_navsim(args)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

