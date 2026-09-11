import argparse
import csv
import re
from pathlib import Path

import torch

from loader import WaymoE2E
from models.base_model import collate_with_images
from models.checkpoint_loader import load_model_from_checkpoint


def latest_checkpoint(pattern: str, checkpoint_dir: Path) -> Path:
    matches = sorted(checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No checkpoints matched {checkpoint_dir / pattern}")
    return matches[-1]


def checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"epoch=(\d+)", path.name)
    return int(match.group(1)) if match else None


def selected_trajectory(output: dict[str, torch.Tensor], horizon: int) -> torch.Tensor:
    traj = output["trajectory"]
    scores = output.get("scores")
    if traj.ndim != 2:
        raise ValueError(f"Expected flat trajectory output, got {traj.shape}")

    batch_size = traj.size(0)
    traj = traj.view(batch_size, -1, horizon, 2)
    if scores is None or traj.size(1) == 1:
        return traj[:, 0]

    selected = scores.argmin(dim=1)
    return traj[torch.arange(batch_size, device=traj.device), selected]


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: str,
    val_items: int | None,
    batch_size: int,
    device: torch.device,
    model_type: str = "auto",
) -> dict[str, float | int | str]:
    dataset = WaymoE2E(indexFile="index_val.pkl", data_dir=data_dir, n_items=val_items)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )
    model = load_model_from_checkpoint(
        checkpoint_path.as_posix(),
        device=device,
        model_type=model_type,
    )

    total_count = 0
    ade3_sum = 0.0
    ade5_sum = 0.0
    oracle3_sum = 0.0
    oracle5_sum = 0.0

    with torch.inference_mode():
        for batch in loader:
            past = batch["PAST"].to(device, non_blocking=True)
            future = batch["FUTURE"].to(device, non_blocking=True)
            intent = batch["INTENT"].to(device, non_blocking=True)
            images = [img.to(device, non_blocking=True) if img is not None else None for img in batch["IMAGES"]]

            output = model({"PAST": past, "IMAGES": images, "INTENT": intent})
            pred = selected_trajectory(output, horizon=future.size(1))

            traj = output["trajectory"].view(future.size(0), -1, future.size(1), 2)
            dist = torch.norm(pred - future, dim=-1)
            oracle_dist = torch.norm(traj - future[:, None], dim=-1)

            batch_n = future.size(0)
            ade3_sum += dist[:, :12].mean(dim=1).sum().item()
            ade5_sum += dist[:, :20].mean(dim=1).sum().item()
            oracle3_sum += oracle_dist[:, :, :12].mean(dim=2).min(dim=1).values.sum().item()
            oracle5_sum += oracle_dist[:, :, :20].mean(dim=2).min(dim=1).values.sum().item()
            total_count += batch_n

    return {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_epoch": checkpoint_epoch(checkpoint_path),
        "val_items": val_items,
        "n_eval": total_count,
        "selected_ADE_3s": ade3_sum / total_count,
        "selected_ADE_5s": ade5_sum / total_count,
        "oracle_ADE_3s": oracle3_sum / total_count,
        "oracle_ADE_5s": oracle5_sum / total_count,
    }


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    headers = [
        "run",
        "model",
        "train_items",
        "val_items",
        "checkpoint_epoch",
        "selected_ADE_3s",
        "selected_ADE_5s",
        "oracle_ADE_3s",
        "oracle_ADE_5s",
        "checkpoint",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--checkpoint_dir",
        default="/work/nvme/bgxf/mgagvani/wod/waymo_end_to_end_camera_v1_0_0/checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        default="/work/nvme/bgxf/mgagvani/wod/waymo_end_to_end_camera_v1_0_0/ade_eval",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sizes = [50_000, 100_000, 150_000, 200_000, 250_000]
    models = ["gtrs", "drivor"]
    specs = []
    for model in models:
        for train_items in train_sizes:
            val_items = max(train_items // 10, 1)
            specs.append(
                (
                    f"{model}_n{train_items}",
                    model,
                    train_items,
                    val_items,
                    f"camera-e2e-{model}-waymo-n{train_items}-*.ckpt",
                )
            )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")
    rows = []
    for run_name, model, train_items, val_items, pattern in specs:
        ckpt = latest_checkpoint(pattern, checkpoint_dir)
        print(f"Evaluating {run_name}: {ckpt}")
        row = evaluate_checkpoint(
            checkpoint_path=ckpt,
            data_dir=args.data_dir,
            val_items=val_items,
            batch_size=args.batch_size,
            device=device,
            model_type=model,
        )
        row.update({"run": run_name, "model": model, "train_items": train_items})
        rows.append(row)

    csv_path = output_dir / "gtrs_drivor_ade_table.csv"
    md_path = output_dir / "gtrs_drivor_ade_table.md"
    fieldnames = [
        "run",
        "model",
        "train_items",
        "val_items",
        "n_eval",
        "checkpoint_epoch",
        "selected_ADE_3s",
        "selected_ADE_5s",
        "oracle_ADE_3s",
        "oracle_ADE_5s",
        "checkpoint",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, md_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
