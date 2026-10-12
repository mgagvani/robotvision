from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from models.sae import SparseAutoencoder

if TYPE_CHECKING:
    from loader import WaymoE2E


DEFAULT_SAE_BLOCK = 3
DEFAULT_TOPK_RATIO = 16


def planner_token_key(block_idx: int) -> str:
    return f"planner_query_tok_block_{block_idx}"


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_topk(input_dim: int, requested_k: int | None = None) -> int:
    if requested_k is not None and requested_k > 0:
        return max(1, min(requested_k, input_dim))
    return max(1, min(input_dim, round(input_dim / DEFAULT_TOPK_RATIO)))


def parse_blocks(text: str, *, default_blocks: list[int] | None = None) -> list[int]:
    raw = text.strip().lower()
    if raw in {"all", "*"}:
        return list(default_blocks if default_blocks is not None else range(DEFAULT_SAE_BLOCK + 1))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def resolve_token_tensor(token_blob: dict, sae_block: int) -> tuple[torch.Tensor, str]:
    block_key = planner_token_key(sae_block)
    return token_blob[block_key].float(), block_key


def infer_sae_paths(run_root: Path, split: str, sae_block: int) -> tuple[Path, Path]:
    ckpt_path = run_root / "model" / f"block_{sae_block}" / "sae_checkpoint.pt"
    token_path = run_root / "tokens" / f"planner_tokens_{split}.pt"
    return ckpt_path, token_path


def build_sae_from_checkpoint(ckpt: dict) -> SparseAutoencoder:
    model = SparseAutoencoder(
        input_dim=ckpt["input_dim"],
        latent_dim=ckpt["latent_dim"],
        k=ckpt["k"],
        k_aux=ckpt["k_aux"],
        aux_alpha=ckpt["lambda_aux"],
        dead_steps_threshold=ckpt["dead_steps_threshold"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model


def load_sae_bundle(
    run_root: Path,
    split: str,
    sae_block: int,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    ckpt_path, token_path = infer_sae_paths(run_root, split, sae_block)
    ckpt = torch.load(ckpt_path, map_location=map_location)
    token_blob = torch.load(token_path, map_location=map_location)
    return {
        "ckpt": ckpt,
        "token_blob": token_blob,
        "ckpt_path": ckpt_path,
        "token_path": token_path,
        "model_dir": ckpt_path.parent,
    }


def default_analysis_dir(run_root: Path, sae_block: int, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return run_root / "analysis" / f"block_{sae_block}"


def encode_tensor_batchwise(
    sae: SparseAutoencoder,
    token_tensor: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    latents = []
    sae.eval()
    with torch.no_grad():
        for start in range(0, len(token_tensor), batch_size):
            batch_x = token_tensor[start : start + batch_size].to(device)
            latents.append(sae.encode(batch_x).cpu())
    return torch.cat(latents, dim=0)


def decode_latents_to_input(
    sae: SparseAutoencoder,
    z: torch.Tensor,
    reference_x: torch.Tensor,
) -> torch.Tensor:
    return sae.decode_to_input(z, reference_x=reference_x)


def dataset_from_token_blob(
    token_blob: dict,
    *,
    data_dir: str | None = None,
    index_file: str | None = None,
) -> "WaymoE2E":
    from loader import WaymoE2E

    meta = token_blob.get("meta", {})
    resolved_data_dir = data_dir or meta.get("data_dir")
    resolved_index_file = index_file or meta.get("index_file")
    n_items = meta.get("n_items")
    if resolved_data_dir is None or resolved_index_file is None:
        raise ValueError(
            "Need data_dir and index_file to replay planner batches for early-block interventions."
        )
    return WaymoE2E(
        indexFile=resolved_index_file,
        data_dir=resolved_data_dir,
        n_items=n_items,
    )


def collate_dataset_indices(dataset: WaymoE2E, indices: torch.Tensor) -> dict:
    from models.base_model import collate_with_images

    samples = [dataset[int(idx)] for idx in indices.tolist()]
    return collate_with_images(samples)


def planner_inputs_from_collated_batch(batch: dict, *, device: torch.device) -> dict[str, torch.Tensor]:
    """Move a `collate_with_images` batch to `device` in the planner's input format."""
    images = [img.to(device, non_blocking=True) if img is not None else None for img in batch["IMAGES"]]
    return {
        "PAST": batch["PAST"].to(device, non_blocking=True),
        "IMAGES": images,
        "INTENT": batch["INTENT"].to(device, non_blocking=True),
    }


def prepare_replay_context(planner_model, batch: dict, *, device: torch.device) -> dict[str, torch.Tensor]:
    model_inputs = planner_inputs_from_collated_batch(batch, device=device)
    tokens, _ = planner_model.prepare_visual_tokens(model_inputs["IMAGES"])
    return {
        "past": model_inputs["PAST"],
        "tokens": tokens,
    }
