"""Publish or fetch the paper artifacts on the Hugging Face Hub.

One Hub repository, mgagvani/mech-interp-for-e2e-driving:
  planner/camera-e2e-epoch=04-val_loss=2.90.ckpt   the paper's SMTS checkpoint
  sae/block_{0..3}/sae_checkpoint.pt               the paper's SAEs
  counterfactual_edits/{manifest.jsonl,edited/,sae_analysis/}   edit manifest, 615 edited frames, stage-2 outputs

    source paths.env
    hf auth login                          # once, with a write token
    python artifacts.py upload   [--public]
    python artifacts.py download           # lays files out where paths.env expects them

Upload reads from the paths.env locations; download writes to them. Both are idempotent.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "mgagvani/mech-interp-for-e2e-driving"
PLANNER_NAME = "camera-e2e-epoch=04-val_loss=2.90.ckpt"


def env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set; `source paths.env` first")
    return Path(value)


def upload(args: argparse.Namespace) -> None:
    api = HfApi()
    run_root, edits = env("RUN_ROOT"), env("VISUAL_GEN_ROOT")
    api.create_repo(REPO_ID, repo_type="model", private=not args.public, exist_ok=True)
    files = [(run_root / PLANNER_NAME, f"planner/{PLANNER_NAME}")]
    files += [(Path("sae_checkpoints") / f"sae_block_{b}.pt", f"sae/block_{b}/sae_checkpoint.pt") for b in range(4)]
    for src, dst in files:
        print(f"{REPO_ID}: {dst}  <-  {src}", flush=True)
        api.upload_file(path_or_fileobj=str(src.resolve()), path_in_repo=dst, repo_id=REPO_ID, repo_type="model")
    print(f"{REPO_ID}: counterfactual_edits/  <-  {edits}", flush=True)
    api.upload_folder(folder_path=str(edits), path_in_repo="counterfactual_edits", repo_id=REPO_ID, repo_type="model",
                      allow_patterns=["manifest.jsonl", "edited/*", "sae_analysis/*"])
    print(f"done: https://huggingface.co/{REPO_ID}")


def download(args: argparse.Namespace) -> None:
    run_root, edits = env("RUN_ROOT"), env("VISUAL_GEN_ROOT")
    local = Path(snapshot_download(REPO_ID, repo_type="model"))
    links = [(local / "planner" / PLANNER_NAME, run_root / PLANNER_NAME)]
    links += [(local / "sae" / f"block_{b}" / "sae_checkpoint.pt", run_root / "model" / f"block_{b}" / "sae_checkpoint.pt") for b in range(4)]
    for name in ("manifest.jsonl", "edited", "sae_analysis"):
        links.append((local / "counterfactual_edits" / name, edits / name))
    (run_root / "tokens").mkdir(parents=True, exist_ok=True)
    for src, dst in links:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            raise SystemExit(f"{dst} is a real directory; move it aside and rerun")
        dst.symlink_to(src.resolve())
        print(f"{dst}  ->  {src}")
    print(f"next: sbatch scripts/run_extract_tokens.slurm  (fills {run_root}/tokens from the planner checkpoint)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upload"); up.add_argument("--public", action="store_true", help="create the repo public (default private)")
    sub.add_parser("download")
    args = parser.parse_args()
    {"upload": upload, "download": download}[args.command](args)
