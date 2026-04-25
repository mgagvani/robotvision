from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LEGACY_SAE_TYPE = "legacy_relu"
TOPK_AUX_SAE_TYPE = "topk_aux"


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        sae_type: str = TOPK_AUX_SAE_TYPE,
        k: int | None = None,
        k_aux: int = 512,
        aux_alpha: float = 1.0 / 32.0,
        dead_steps_threshold: int = 500,
        use_encoder_bias: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sae_type = sae_type
        self.k = min(k if k is not None else latent_dim, latent_dim)
        self.k_aux = k_aux
        self.aux_alpha = aux_alpha
        self.dead_steps_threshold = dead_steps_threshold
        self.use_encoder_bias = use_encoder_bias

        self.encoder = nn.Linear(input_dim, latent_dim, bias=use_encoder_bias)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
        self.register_buffer("legacy_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("legacy_std", torch.ones(input_dim, dtype=torch.float32))
        self.register_buffer("legacy_norm_enabled", torch.tensor(False))

        if self.sae_type == TOPK_AUX_SAE_TYPE:
            self.b_pre = nn.Parameter(torch.zeros(input_dim))
            self.register_buffer("cmse", torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer(
                "steps_since_active",
                torch.zeros(latent_dim, dtype=torch.long),
            )
            self.reset_topk_parameters()
        elif self.sae_type == LEGACY_SAE_TYPE:
            self.register_parameter("b_pre", None)
            self.register_buffer("cmse", torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer(
                "steps_since_active",
                torch.zeros(latent_dim, dtype=torch.long),
            )
            self.reset_legacy_parameters()
        else:
            raise ValueError(f"Unsupported sae_type={sae_type}")

    def reset_topk_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.decoder.weight, mean=0.0, std=1.0 / math.sqrt(self.input_dim))
            self.normalize_decoder_columns()
            scale = math.sqrt(max(self.k, 1) / float(self.input_dim))
            self.encoder.weight.copy_(self.decoder.weight.t() * scale)
            if self.encoder.bias is not None:
                self.encoder.bias.zero_()
            self.b_pre.zero_()
            self.cmse.fill_(1.0)
            self.steps_since_active.zero_()

    def reset_legacy_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        self.cmse.fill_(1.0)
        self.steps_since_active.zero_()

    def set_preprocessing_state(
        self,
        *,
        b_pre: torch.Tensor,
        cmse: float | torch.Tensor,
    ) -> None:
        if self.sae_type != TOPK_AUX_SAE_TYPE:
            return
        with torch.no_grad():
            self.b_pre.copy_(b_pre.to(self.b_pre.device, dtype=self.b_pre.dtype))
            self.cmse.fill_(float(cmse))

    def set_legacy_normalization(
        self,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.legacy_mean.copy_(mean.to(self.legacy_mean.device, dtype=self.legacy_mean.dtype))
            self.legacy_std.copy_(std.to(self.legacy_std.device, dtype=self.legacy_std.dtype))
            self.legacy_norm_enabled.fill_(True)

    def preprocess_input(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if self.sae_type == LEGACY_SAE_TYPE:
            if bool(self.legacy_norm_enabled.item()):
                return (x - self.legacy_mean) / self.legacy_std.clamp_min(1e-6), None
            return x, None
        if self.sae_type != TOPK_AUX_SAE_TYPE:
            return x, None

        x_shifted = x - self.b_pre
        sample_mean = x_shifted.mean(dim=-1, keepdim=True)
        centered = x_shifted - sample_mean
        sample_norm = torch.norm(centered, dim=-1, keepdim=True).clamp_min(1e-8)
        x_norm = centered / sample_norm
        stats = {
            "sample_mean": sample_mean,
            "sample_norm": sample_norm,
        }
        return x_norm, stats

    def restore_input(self, x_norm: torch.Tensor, stats: dict[str, torch.Tensor] | None) -> torch.Tensor:
        if self.sae_type == LEGACY_SAE_TYPE:
            if bool(self.legacy_norm_enabled.item()):
                return x_norm * self.legacy_std + self.legacy_mean
            return x_norm
        if self.sae_type != TOPK_AUX_SAE_TYPE:
            return x_norm
        if stats is None:
            raise ValueError("TopK SAE restore_input requires preprocessing stats.")
        return x_norm * stats["sample_norm"] + stats["sample_mean"] + self.b_pre

    def topk_masked_relu(self, values: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or values.size(-1) == 0:
            return torch.zeros_like(values)

        k_eff = min(k, values.size(-1))
        top_values, top_indices = torch.topk(values, k=k_eff, dim=-1)
        masked = torch.zeros_like(values)
        masked.scatter_(-1, top_indices, top_values)
        return F.relu(masked)

    def encode_preprocessed(
        self,
        x_preprocessed: torch.Tensor,
        *,
        return_pre_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pre_activations = self.encoder(x_preprocessed)
        if self.sae_type == LEGACY_SAE_TYPE:
            z = F.relu(pre_activations)
        else:
            z = self.topk_masked_relu(pre_activations, self.k)
        if return_pre_activations:
            return z, pre_activations
        return z

    def encode(
        self,
        x: torch.Tensor,
        *,
        return_pre_activations: bool = False,
        return_preprocess_stats: bool = False,
    ):
        x_preprocessed, stats = self.preprocess_input(x)
        encoded = self.encode_preprocessed(
            x_preprocessed,
            return_pre_activations=return_pre_activations,
        )
        if return_pre_activations:
            z, pre_activations = encoded
            if return_preprocess_stats:
                return z, pre_activations, stats
            return z, pre_activations

        if return_preprocess_stats:
            return encoded, stats
        return encoded

    def decode_to_preprocessed(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def decode_to_input(
        self,
        z: torch.Tensor,
        *,
        reference_x: torch.Tensor | None = None,
        preprocess_stats: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x_preprocessed_hat = self.decode_to_preprocessed(z)
        if self.sae_type == LEGACY_SAE_TYPE:
            return x_preprocessed_hat

        if preprocess_stats is None:
            if reference_x is None:
                raise ValueError("TopK SAE decode_to_input requires reference_x or preprocess_stats.")
            _, preprocess_stats = self.preprocess_input(reference_x)
        return self.restore_input(x_preprocessed_hat, preprocess_stats)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_to_preprocessed(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, stats = self.encode(x, return_preprocess_stats=True)
        x_hat = self.decode_to_input(z, preprocess_stats=stats)
        return x_hat, z

    def compute_auxiliary_reconstruction(
        self,
        residual_preprocessed: torch.Tensor,
        dead_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.sae_type != TOPK_AUX_SAE_TYPE or not dead_mask.any():
            return torch.zeros_like(residual_preprocessed)

        aux_pre = self.encoder(residual_preprocessed)
        masked_aux_pre = torch.full_like(aux_pre, float("-inf"))
        masked_aux_pre[:, dead_mask] = aux_pre[:, dead_mask]
        aux_z = self.topk_masked_relu(masked_aux_pre, min(self.k_aux, int(dead_mask.sum().item())))
        return self.decode_to_preprocessed(aux_z)

    def compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_preprocessed, stats = self.preprocess_input(x)
        z, pre_activations = self.encode(
            x,
            return_pre_activations=True,
        )
        x_preprocessed_hat = self.decode_to_preprocessed(z)
        x_hat = self.decode_to_input(z, preprocess_stats=stats)

        recon_loss = F.mse_loss(x_hat, x) / self.cmse.clamp_min(1e-8)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.sae_type == TOPK_AUX_SAE_TYPE:
            dead_mask = self.steps_since_active >= self.dead_steps_threshold
            residual_preprocessed = x_preprocessed - x_preprocessed_hat
            if dead_mask.any():
                aux_hat = self.compute_auxiliary_reconstruction(
                    residual_preprocessed=residual_preprocessed,
                    dead_mask=dead_mask,
                )
                aux_loss = F.mse_loss(aux_hat, residual_preprocessed) / self.cmse.clamp_min(1e-8)

        total_loss = recon_loss + self.aux_alpha * aux_loss
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "aux_loss": aux_loss,
            "x_hat": x_hat,
            "z": z,
            "pre_activations": pre_activations,
        }

    def update_activation_history(self, z: torch.Tensor) -> None:
        with torch.no_grad():
            fired = (z > 0).any(dim=0)
            self.steps_since_active.add_(1)
            self.steps_since_active[fired] = 0

    def project_decoder_gradients(self) -> None:
        if self.decoder.weight.grad is None or self.sae_type != TOPK_AUX_SAE_TYPE:
            return
        with torch.no_grad():
            decoder_weight = self.decoder.weight
            grad = self.decoder.weight.grad
            projection = (grad * decoder_weight).sum(dim=0, keepdim=True)
            grad.sub_(decoder_weight * projection)

    def normalize_decoder_columns(self) -> None:
        with torch.no_grad():
            norms = torch.norm(self.decoder.weight, dim=0, keepdim=True).clamp_min(1e-8)
            self.decoder.weight.div_(norms)

    def dead_fraction(self) -> float:
        if self.latent_dim == 0:
            return 0.0
        return float((self.steps_since_active >= self.dead_steps_threshold).float().mean().item())
