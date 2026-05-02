"""Shared helpers for SAE-backed experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.base_model import LitModel
from models.feature_extractors import SAMFeatures
from models.monocular import DeepMonocularModel
from sparseAE import SparseAE


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def set_eval_mode(*modules: nn.Module) -> None:
    for module in modules:
        module.eval()


def get_sae_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        checkpoint_obj = checkpoint_obj["state_dict"]
    if not isinstance(checkpoint_obj, dict):
        raise TypeError("Unsupported SAE checkpoint format")
    return checkpoint_obj


def compute_hidden_activations(sae: SparseAE, hooked_acts: torch.Tensor) -> torch.Tensor:
    hidden = torch.relu(sae.encoder(hooked_acts - sae.decoder.bias))
    return hidden.flatten(start_dim=1)


def build_default_backbone() -> DeepMonocularModel:
    return DeepMonocularModel(
        feature_extractor=SAMFeatures(
            model_name="timm/vit_pe_spatial_small_patch16_512.fb",
            frozen=True,
        ),
        out_dim=40,
        n_blocks=4,
    )


def build_default_lit_model(model_checkpoint_path: str) -> LitModel:
    model = LitModel.load_from_checkpoint(
        model_checkpoint_path,
        model=build_default_backbone(),
        map_location="cpu",
    )
    model.eval()
    return model


def load_model_and_sae(
    model_checkpoint_path: str,
    sae_checkpoint_path: str,
    block_idx: int,
    device: torch.device | None = None,
) -> tuple[LitModel, SparseAE, Any]:
    model = build_default_lit_model(model_checkpoint_path)
    target_layer = model.model.blocks[block_idx].mlp[2]

    sae_checkpoint = torch.load(sae_checkpoint_path, map_location="cpu")
    sae_state = get_sae_state_dict(sae_checkpoint)
    encoder_weight = sae_state["encoder.weight"]
    dict_size, input_dim = encoder_weight.shape

    sae = SparseAE.build_from_state_dict(
        sae_state,
        target_model=model,
        input_dim=input_dim,
        dict_size=dict_size,
        compile_sae=False,
    )
    sae.eval()
    hook_handle = target_layer.register_forward_hook(sae.hook_fn)

    if device is not None:
        model = model.to(device)
        sae = sae.to(device)

    return model, sae, hook_handle
