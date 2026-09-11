import argparse
import csv
from pathlib import Path

import torch

from evaluate_waymo_ade_table import evaluate_checkpoint


FIELDNAMES = [
    "row_index",
    "run",
    "model",
    "train_items",
    "source_val_items",
    "val_items",
    "n_eval",
    "checkpoint_epoch",
    "selected_ADE_3s",
    "selected_ADE_5s",
    "oracle_ADE_3s",
    "oracle_ADE_5s",
    "checkpoint",
]


def read_markdown_table(path: Path) -> list[dict[str, str]]:
    rows = []
    headers = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if headers is None:
            headers = parts
            continue
        if all(part == "---" for part in parts):
            continue
        rows.append(dict(zip(headers, parts, strict=True)))
    return rows


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    headers = [name for name in FIELDNAMES if name != "row_index"]
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


def write_result(row: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run = row["run"]
    row_index = row["row_index"]
    csv_path = output_dir / f"{int(row_index):02d}_{run}_full_val_ade.csv"
    md_path = output_dir / f"{int(row_index):02d}_{run}_full_val_ade.md"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
    write_markdown([row], md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def aggregate_results(output_dir: Path, csv_name: str, md_name: str) -> None:
    rows = []
    for path in sorted(output_dir.glob("*_full_val_ade.csv")):
        if path.name == csv_name:
            continue
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["row_index"]))

    csv_path = output_dir / csv_name
    md_path = output_dir / md_name
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--row_index", type=int)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "/work/nvme/bgxf/mgagvani/wod/waymo_end_to_end_camera_v1_0_0/ade_eval/full_val"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--csv_name", default="gtrs_drivor_full_val_ade_table.csv")
    parser.add_argument("--md_name", default="gtrs_drivor_full_val_ade_table.md")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_results(args.output_dir, args.csv_name, args.md_name)
        return

    if args.data_dir is None or args.table is None or args.row_index is None:
        parser.error("--data_dir, --table, and --row_index are required unless --aggregate is set")

    table_rows = read_markdown_table(args.table)
    source = table_rows[args.row_index]
    checkpoint = Path(source["checkpoint"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")

    print(f"Evaluating full validation row {args.row_index}: {source['run']}")
    print(f"checkpoint: {checkpoint}")
    result = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        data_dir=args.data_dir,
        val_items=None,
        batch_size=args.batch_size,
        device=device,
        model_type=source["model"],
    )
    result.update(
        {
            "row_index": args.row_index,
            "run": source["run"],
            "model": source["model"],
            "train_items": int(source["train_items"]),
            "source_val_items": int(source["val_items"]),
            "val_items": "all",
        }
    )
    write_result(result, args.output_dir)


if __name__ == "__main__":
    main()
