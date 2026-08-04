import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--analysis_dir", type=str, default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    analysis_root = Path(args.analysis_dir) if args.analysis_dir else run_root / "analysis"
    merged_dir = analysis_root / "merged"

    grouped_paths: dict[Path, list[tuple[int, Path]]] = defaultdict(list)
    for block_dir in sorted(analysis_root.glob("block_*")):
        if not block_dir.is_dir():
            continue
        try:
            block_idx = int(block_dir.name.split("_")[-1])
        except ValueError:
            continue
        for csv_path in sorted(block_dir.rglob("*.csv")):
            relative_path = csv_path.relative_to(block_dir)
            grouped_paths[relative_path].append((block_idx, csv_path))

    for relative_path, entries in grouped_paths.items():
        merged_rows = []
        for block_idx, csv_path in entries:
            for row in read_csv_rows(csv_path):
                merged_rows.append({"sae_block": block_idx, **row})
        output_path = merged_dir / relative_path
        write_csv_rows(output_path, merged_rows)
        print(f"Merged {len(entries)} block files into {output_path}")
