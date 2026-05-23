from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .initial_sae import load_torch_file


@dataclass(frozen=True)
class PMBlockSAEMetadata:
    block_index: int
    activation_key: str
    input_dim: int
    latent_dim: int
    k: int


class PMBlockTopKSAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        k: int,
        block_index: int,
        activation_key: str,
        legacy_norm_enabled: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k
        self.metadata = PMBlockSAEMetadata(
            block_index=block_index,
            activation_key=activation_key,
            input_dim=input_dim,
            latent_dim=latent_dim,
            k=k,
        )

        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(input_dim))

        self.register_buffer("legacy_mean", torch.zeros(input_dim))
        self.register_buffer("legacy_std", torch.ones(input_dim))
        self.register_buffer("legacy_norm_enabled", torch.tensor(bool(legacy_norm_enabled), dtype=torch.bool))
        self.register_buffer("steps_since_active", torch.zeros(latent_dim, dtype=torch.long))
        self.register_buffer("cmse", torch.tensor(0.0))

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if bool(self.legacy_norm_enabled.item()):
            std = torch.clamp_min(self.legacy_std, 1e-6)
            return (x - self.legacy_mean) / std
        return x

    def _denormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        if bool(self.legacy_norm_enabled.item()):
            return x * self.legacy_std + self.legacy_mean
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self._normalize_input(x)
        pre_acts = F.relu(self.encoder(x_norm - self.b_pre))
        if self.k <= 0 or self.k >= pre_acts.shape[-1]:
            return pre_acts

        topk_vals, topk_idx = torch.topk(pre_acts, k=self.k, dim=-1)
        latents = torch.zeros_like(pre_acts)
        latents.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        return latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        recon_norm = self.decoder(z) + self.b_pre
        return self._denormalize_output(recon_norm)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latents = self.encode(x)
        reconstruction = self.decode(latents)
        return {
            "latents": latents,
            "reconstruction": reconstruction,
        }


def load_pm_block_sae(ckpt_path: Path, device: torch.device) -> PMBlockTopKSAE:
    ckpt = load_torch_file(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(ckpt).__name__}")
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} is missing required key 'state_dict'")

    metadata = ckpt.get("metadata", {})
    resolved = metadata.get("resolved_hyperparameters", {})
    input_dim = int(ckpt.get("input_dim", resolved.get("input_dim")))
    latent_dim = int(ckpt.get("latent_dim", resolved.get("latent_dim")))
    k = int(ckpt.get("k", resolved.get("k", latent_dim)))
    block_index = int(ckpt.get("block_index", metadata.get("block_index", -1)))
    activation_key = str(metadata.get("activation_key", ckpt.get("token_key", "planner_query_tok")))
    legacy_norm_enabled = bool(ckpt["state_dict"].get("legacy_norm_enabled", torch.tensor(False)).item())

    model = PMBlockTopKSAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        k=k,
        block_index=block_index,
        activation_key=activation_key,
        legacy_norm_enabled=legacy_norm_enabled,
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
