"""Publish or fetch the paper artifacts on the Hugging Face Hub.

Two Hub repositories:
  <HF_ORG>/smts-wod-e2e              (model)   planner checkpoint, ADE-table baselines, SAEs, val tokens
  <HF_ORG>/smts-counterfactual-edits (dataset) edit manifest, 615 edited front-camera frames, stage-2 outputs

    source paths.env
    hf auth login                          # once, with a write token
    python artifacts.py upload   --org <HF_ORG> [--private] [--tokens]
    python artifacts.py download --org <HF_ORG>      # lays files out where paths.env expects them

Upload reads from the paths.env locations; download writes to them. Both are idempotent.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODEL_REPO = "smts-wod-e2e"
EDITS_REPO = "smts-counterfactual-edits"
PLANNER_NAME = "camera-e2e-epoch=04-val_loss=2.90.ckpt"


def env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set; `source paths.env` first")
    return Path(value)


def table_checkpoints(output_root: Path) -> list[Path]:
    """The ten GTRS/DrivoR checkpoints behind the ADE table (one per model and training size)."""
    table = output_root / "ade_eval" / "gtrs_drivor_ade_table.md"
    if not table.exists():
        raise SystemExit(f"missing {table}; run scripts/run_waymo_ade_table.slurm first")
    return [Path(p) for p in re.findall(r"\S+\.ckpt", table.read_text())]


def upload(args: argparse.Namespace) -> None:
    api = HfApi()
    run_root, out_root, edits = env("RUN_ROOT"), env("OUTPUT_ROOT"), env("VISUAL_GEN_ROOT")
    model_id, edits_id = f"{args.org}/{MODEL_REPO}", f"{args.org}/{EDITS_REPO}"
    api.create_repo(model_id, repo_type="model", private=args.private, exist_ok=True)
    api.create_repo(edits_id, repo_type="dataset", private=args.private, exist_ok=True)

    files = [(run_root / PLANNER_NAME, f"planner/{PLANNER_NAME}")]
    files += [(p, f"baselines/{p.name}") for p in table_checkpoints(out_root)]
    files += [(Path("sae_checkpoints") / f"sae_block_{b}.pt", f"sae/block_{b}/sae_checkpoint.pt") for b in range(4)]
    if args.tokens:
        files.append((run_root / "tokens" / "planner_tokens_val.pt", "tokens/planner_tokens_val.pt"))
    for src, dst in files:
        print(f"{model_id}: {dst}  <-  {src}")
        api.upload_file(path_or_fileobj=str(src.resolve()), path_in_repo=dst, repo_id=model_id, repo_type="model")

    print(f"{edits_id}: manifest.jsonl, edited/, sae_analysis/  <-  {edits}")
    api.upload_folder(folder_path=str(edits), repo_id=edits_id, repo_type="dataset",
                      allow_patterns=["manifest.jsonl", "edited/*", "sae_analysis/*"])


def download(args: argparse.Namespace) -> None:
    run_root, ckpt_dir, edits = env("RUN_ROOT"), env("CHECKPOINT_DIR"), env("VISUAL_GEN_ROOT")
    local = Path(snapshot_download(f"{args.org}/{MODEL_REPO}", repo_type="model"))
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "model").mkdir(exist_ok=True)
    (run_root / "tokens").mkdir(exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    links = [(local / "planner" / PLANNER_NAME, run_root / PLANNER_NAME)]
    links += [(p, ckpt_dir / p.name) for p in sorted((local / "baselines").glob("*.ckpt"))]
    links += [(local / "sae" / f"block_{b}" / "sae_checkpoint.pt", run_root / "model" / f"block_{b}" / "sae_checkpoint.pt") for b in range(4)]
    if (local / "tokens" / "planner_tokens_val.pt").exists():
        links.append((local / "tokens" / "planner_tokens_val.pt", run_root / "tokens" / "planner_tokens_val.pt"))
    for src, dst in links:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        print(f"{dst}  ->  {src}")
    snapshot_download(f"{args.org}/{EDITS_REPO}", repo_type="dataset", local_dir=str(edits))
    print(f"{edits}: manifest.jsonl, edited/, sae_analysis/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upload"); up.add_argument("--org", required=True); up.add_argument("--private", action="store_true")
    up.add_argument("--tokens", action="store_true", help="also upload the 2.4 GB validation token cache")
    dl = sub.add_parser("download"); dl.add_argument("--org", required=True)
    args = parser.parse_args()
    {"upload": upload, "download": download}[args.command](args)
