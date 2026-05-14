from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src/camera-based-e2e"))

import train_sae  # noqa: E402
from sae_utils import DRVLA_SAE_VERSION, DRVLA_SOURCE_NOTE, planner_token_key, resolve_topk  # noqa: E402


def build_token_blob(*, seed: int, n_blocks: int = 4, n_samples: int = 8, dim: int = 16) -> dict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    blob = {
        "meta": {
            "n_blocks": n_blocks,
            "checkpoint": "/tmp/fake-planner.ckpt",
            "data_dir": "/tmp/fake-data",
            "index_file": "/tmp/fake-index.pkl",
        }
    }
    for block_idx in range(n_blocks):
        blob[planner_token_key(block_idx)] = torch.randn(n_samples, dim, generator=generator)
    blob["planner_query_tok"] = blob[planner_token_key(n_blocks - 1)].clone()
    return blob


def test_train_sae_writes_drvla_metadata_for_all_blocks(tmp_path: Path) -> None:
    train_path = tmp_path / "planner_tokens_train.pt"
    val_path = tmp_path / "planner_tokens_val.pt"
    output_dir = tmp_path / "model"

    torch.save(build_token_blob(seed=0), train_path)
    torch.save(build_token_blob(seed=1), val_path)

    train_sae.main(
        [
            "--output_dir",
            str(output_dir),
            "--train_dataset",
            str(train_path),
            "--val_dataset",
            str(val_path),
            "--device",
            "cpu",
            "--batch_size",
            "4",
            "--max_epochs",
            "1",
            "--geometric_median_samples",
            "8",
            "--blocks",
            "all",
        ]
    )

    expected_k = resolve_topk(16)
    for block_idx in range(4):
        ckpt_path = output_dir / f"block_{block_idx}" / "sae_checkpoint.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt["block_index"] == block_idx
        assert ckpt["token_key"] == planner_token_key(block_idx)
        assert ckpt["k"] == expected_k
        assert ckpt["k_aux"] == 512
        assert ckpt["lambda_aux"] == 1.0 / 32.0
        assert ckpt["metadata"]["sae_version"] == DRVLA_SAE_VERSION
        assert ckpt["metadata"]["source_note"] == DRVLA_SOURCE_NOTE
        assert ckpt["metadata"]["activation_kind"] == "post_transformer_block_query"
        assert ckpt["metadata"]["resolved_hyperparameters"]["k"] == expected_k
        assert ckpt["metadata"]["train_token_meta"]["n_blocks"] == 4
        assert len(ckpt["history"]) == 1
