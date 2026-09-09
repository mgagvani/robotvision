from __future__ import annotations

import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from models.base_model import LitModel
from models.feature_extractors import SAMFeatures
from models.monocular import DeepMonocularModel
from models.sae import LEGACY_SAE_TYPE, SparseAutoencoder, TOPK_AUX_SAE_TYPE

if TYPE_CHECKING:
    from loader import WaymoE2E


DEFAULT_SAE_BLOCK = 3
DEFAULT_TOPK_RATIO = 16
DRVLA_SAE_VERSION = "topk_aux_drvla_v1"
DRVLA_SOURCE_NOTE = "Dr.VLA SAE recipe aligned to Table 4 / Appendix A.2."
DRVLA_SOURCE_URLS = (
    "https://drvla.github.io/",
    "https://drvla.github.io/drvla.pdf",
)


class ActivationCapture:
    def __init__(self) -> None:
        self.activations: torch.Tensor | None = None
        self._handle = None

    def clear(self) -> None:
        self.activations = None

    def register(self, module) -> "ActivationCapture":
        """Attach to `module` and retain the handle so the hook can be removed."""
        self.remove()
        self._handle = module.register_forward_hook(self)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __call__(self, module, inputs, output) -> None:
        del module, inputs
        self.activations = output.detach()


def freeze_module(module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def set_eval_mode(*modules) -> None:
    for module in modules:
        module.eval()


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
    if block_key in token_blob:
        return token_blob[block_key].float(), block_key
    if sae_block == DEFAULT_SAE_BLOCK and "planner_query_tok" in token_blob:
        return token_blob["planner_query_tok"].float(), "planner_query_tok"
    raise KeyError(
        f"Could not find block token key for sae_block={sae_block}. "
        f"Expected {block_key} or planner_query_tok for block {DEFAULT_SAE_BLOCK}."
    )


def infer_sae_model_dir(run_root: Path, sae_block: int) -> Path:
    block_dir = run_root / "model" / f"block_{sae_block}"
    if block_dir.exists():
        return block_dir
    if sae_block == DEFAULT_SAE_BLOCK:
        return run_root / "model"
    return block_dir


def infer_sae_paths(run_root: Path, split: str, sae_block: int) -> tuple[Path, Path, Path | None]:
    model_dir = infer_sae_model_dir(run_root, sae_block)
    ckpt_path = model_dir / "sae_checkpoint.pt"
    token_path = run_root / "tokens" / f"planner_tokens_{split}.pt"
    legacy_norm_path = model_dir / "sae_normalization.pt"
    if legacy_norm_path.exists():
        return ckpt_path, token_path, legacy_norm_path
    return ckpt_path, token_path, None


def normalize_compiled_state_dict(state_dict: dict, module) -> dict:
    """Reconcile ``torch.compile``'s ``._orig_mod.`` key prefixes with `module`.

    A checkpoint saved from a compiled SAE carries ``encoder._orig_mod.weight``
    where an eager module expects ``encoder.weight`` (and vice versa). Loading
    across that boundary silently drops every tensor under ``strict=False``,
    leaving a randomly initialised SAE that still "loads" successfully.
    """
    state_compiled = any("._orig_mod." in key for key in state_dict)
    module_compiled = any("._orig_mod." in key for key in module.state_dict())
    if state_compiled == module_compiled:
        return state_dict

    normalized = {}
    for key, value in state_dict.items():
        for prefix in ("encoder.", "decoder."):
            if module_compiled:
                if key.startswith(prefix):
                    key = key.replace(prefix, f"{prefix}_orig_mod.", 1)
                    break
            else:
                compiled_prefix = f"{prefix}_orig_mod."
                if key.startswith(compiled_prefix):
                    key = key.replace(compiled_prefix, prefix, 1)
                    break
        normalized[key] = value
    return normalized


def build_sae_from_checkpoint(ckpt: dict, legacy_norm: dict | None = None) -> SparseAutoencoder:
    sae_type = ckpt.get("sae_type", LEGACY_SAE_TYPE)
    if sae_type == TOPK_AUX_SAE_TYPE:
        model = SparseAutoencoder(
            input_dim=ckpt["input_dim"],
            latent_dim=ckpt["latent_dim"],
            sae_type=TOPK_AUX_SAE_TYPE,
            k=ckpt["k"],
            k_aux=ckpt["k_aux"],
            aux_alpha=ckpt["lambda_aux"],
            dead_steps_threshold=ckpt["dead_steps_threshold"],
            use_encoder_bias=False,
        )
    else:
        model = SparseAutoencoder(
            input_dim=ckpt["input_dim"],
            latent_dim=ckpt["latent_dim"],
            sae_type=LEGACY_SAE_TYPE,
            use_encoder_bias=True,
        )
    model.load_state_dict(normalize_compiled_state_dict(ckpt["state_dict"], model), strict=False)
    if legacy_norm is not None:
        model.set_legacy_normalization(
            mean=legacy_norm["mean"],
            std=legacy_norm["std"],
        )
    return model


def build_default_backbone() -> DeepMonocularModel:
    return DeepMonocularModel(
        feature_extractor=SAMFeatures(
            model_name="timm/vit_pe_spatial_small_patch16_512.fb",
            frozen=True,
        ),
        out_dim=40,
    )


def build_default_lit_model(model_checkpoint_path: str) -> LitModel:
    model = LitModel.load_from_checkpoint(
        model_checkpoint_path,
        model=build_default_backbone(),
        map_location="cpu",
    )
    model.eval()
    return model


def resolve_sae_checkpoint_path(path: str | Path, sae_block: int) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        # Two layouts exist in the wild and both must resolve:
        #   * a flat directory of per-block files, `sae_block_{n}.pt`
        #   * a train_sae.py run root or model dir, which infer_sae_paths()
        #     addresses as `model/block_{n}/sae_checkpoint.pt`
        # Only accepting the first silently rejected valid run roots.
        candidates = (
            checkpoint_path / f"sae_block_{sae_block}.pt",
            checkpoint_path / "sae_checkpoint.pt",
            checkpoint_path / f"block_{sae_block}" / "sae_checkpoint.pt",
            infer_sae_model_dir(checkpoint_path, sae_block) / "sae_checkpoint.pt",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        tried = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"No SAE checkpoint for block {sae_block} under {checkpoint_path}. Tried:\n  {tried}"
        )
    if checkpoint_path.suffix == ".pt":
        return checkpoint_path
    if not checkpoint_path.name.endswith(".tar.gz"):
        raise ValueError(
            f"Unsupported SAE checkpoint artifact '{checkpoint_path}'. Expected a .pt file, "
            "directory, or .tar.gz archive."
        )

    extract_dir = checkpoint_path.parent / f"{checkpoint_path.name[:-7]}_extracted"
    target_path = extract_dir / f"sae_block_{sae_block}.pt"
    if target_path.exists():
        return target_path

    extract_dir.mkdir(parents=True, exist_ok=True)
    member_name = f"sae_block_{sae_block}.pt"
    with tarfile.open(checkpoint_path, "r:gz") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Archive {checkpoint_path} does not contain {member_name}"
            ) from exc
        archive.extract(member, path=extract_dir)
    return target_path


def get_sae_target_layer(model: LitModel, block_idx: int):
    return model.model.blocks[block_idx]


def load_model_and_sae(
    model_checkpoint_path: str,
    sae_checkpoint_path: str | Path,
    block_idx: int,
    *,
    device: torch.device | None = None,
) -> tuple[LitModel, SparseAutoencoder, ActivationCapture, Path]:
    model = build_default_lit_model(model_checkpoint_path)
    resolved_sae_checkpoint_path = resolve_sae_checkpoint_path(sae_checkpoint_path, block_idx)
    sae_checkpoint = torch.load(resolved_sae_checkpoint_path, map_location="cpu")
    checkpoint_block_idx = sae_checkpoint.get("block_index")
    if checkpoint_block_idx is not None and int(checkpoint_block_idx) != block_idx:
        raise ValueError(
            f"SAE checkpoint {resolved_sae_checkpoint_path} was trained for block "
            f"{checkpoint_block_idx}, not requested block {block_idx}."
        )
    sae = build_sae_from_checkpoint(sae_checkpoint)
    sae.eval()

    capture = ActivationCapture().register(get_sae_target_layer(model, block_idx))

    if device is not None:
        model = model.to(device)
        sae = sae.to(device)

    return model, sae, capture, resolved_sae_checkpoint_path


def compute_hidden_activations(sae: SparseAutoencoder, hooked_acts: torch.Tensor) -> torch.Tensor:
    flat_acts = hooked_acts.reshape(-1, hooked_acts.size(-1))
    hidden = sae.encode(flat_acts)
    return hidden.reshape(hooked_acts.size(0), -1)


def load_sae_bundle(
    run_root: Path,
    split: str,
    sae_block: int,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    ckpt_path, token_path, legacy_norm_path = infer_sae_paths(run_root, split, sae_block)
    ckpt = torch.load(ckpt_path, map_location=map_location)
    token_blob = torch.load(token_path, map_location=map_location)
    legacy_norm = torch.load(legacy_norm_path, map_location=map_location) if legacy_norm_path else None
    return {
        "ckpt": ckpt,
        "token_blob": token_blob,
        "legacy_norm": legacy_norm,
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


def planner_inputs_from_collated_batch(
    batch: dict,
    *,
    lit_model,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    # `collate_with_images` decodes JPEGs in the dataloader workers and emits "IMAGES";
    # older collated blobs (and any caller that bypasses the collate fn) still carry the
    # raw "IMAGES_JPEG" bytes, so handle both. Mirrors LitModel.transfer_batch_to_device.
    if "IMAGES" in batch:
        images = [
            img.to(device, non_blocking=True) if img is not None else None
            for img in batch["IMAGES"]
        ]
    elif "IMAGES_JPEG" in batch:
        images = lit_model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
    else:
        raise KeyError("collated batch has neither 'IMAGES' nor 'IMAGES_JPEG'")

    return {
        "PAST": batch["PAST"].to(device, non_blocking=True),
        "IMAGES": images,
        "INTENT": batch["INTENT"].to(device, non_blocking=True),
    }


def prepare_replay_context(
    planner_model,
    lit_model,
    batch: dict,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model_inputs = planner_inputs_from_collated_batch(batch, lit_model=lit_model, device=device)
    tokens, _ = planner_model.prepare_visual_tokens(model_inputs["IMAGES"])
    return {
        "past": model_inputs["PAST"],
        "tokens": tokens,
    }
