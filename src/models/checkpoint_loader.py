"""Load trained E2E checkpoints (DeepMonocular, GTRS, DrivoR)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.drivor import DrivoRModel
from models.feature_extractors import SAMFeatures
from models.gtrs import GTRSModel
from models.monocular import DeepMonocularModel


def infer_model_type(hparams: dict, mapped_state: Dict[str, torch.Tensor]) -> str:
    model_name = str(hparams.get("model_name", "")).lower()
    if "drivor" in model_name:
        return "drivor"
    if "gtrs" in model_name:
        return "gtrs"
    if "deepmonocular" in model_name:
        return "deepmonocular"

    keys = list(mapped_state.keys())
    if any(k.startswith("scene_embeds.") or k.startswith("traj_features.") for k in keys):
        return "drivor"
    if any(k.startswith("vocab_embed.") or k.startswith("status_encoding.") for k in keys):
        return "gtrs"
    return "deepmonocular"


def build_model(
    model_type: str,
    feature_model: str = "timm/vit_pe_spatial_small_patch16_512.fb",
) -> nn.Module:
    feature_extractor = SAMFeatures(model_name=feature_model, frozen=True)
    out_dim = 20 * 2
    if model_type == "gtrs":
        return GTRSModel(feature_extractor=feature_extractor, out_dim=out_dim)
    if model_type == "drivor":
        return DrivoRModel(feature_extractor=feature_extractor, out_dim=out_dim)
    if model_type == "deepmonocular":
        return DeepMonocularModel(feature_extractor=feature_extractor, out_dim=out_dim)
    raise ValueError(f"Unsupported model_type={model_type}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device | None = None,
    model_type: str = "auto",
    feature_model: str = "timm/vit_pe_spatial_small_patch16_512.fb",
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
    state = ckpt.get("state_dict", {}) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    mapped_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("model."):
            k = k[6:]
        mapped_state[k] = v

    resolved = model_type
    if resolved == "auto":
        resolved = infer_model_type(hparams, mapped_state)

    if resolved == "deepmonocular" and isinstance(hparams, dict) and "model" in hparams:
        model = hparams["model"]
    else:
        model = build_model(resolved, feature_model=feature_model)

    model.load_state_dict(mapped_state, strict=True)

    if not hasattr(model, "n_proposals"):
        if hasattr(model, "vocab"):
            model.n_proposals = int(model.vocab.shape[0])
        elif hasattr(model, "cfg"):
            model.n_proposals = int(getattr(model.cfg, "proposal_num", 1))
        else:
            model.n_proposals = 1

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
