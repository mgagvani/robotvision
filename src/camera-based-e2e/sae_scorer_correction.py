"""
Utilities for training and evaluating an SAE-informed scorer correction model.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from loader import WaymoE2E
from models.base_model import LitModel, collate_with_images
from models.feature_extractors import SAMFeatures
from models.monocular import DeepMonocularModel
from sparseAE import SparseAE

DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_VAL_SPLIT = "val"
DEFAULT_PATIENCE, DEFAULT_MIN_DELTA = 2, 1e-3
DEFAULT_LOG_EVERY_N_STEPS, DEFAULT_SEED = 10, 42
VAL_MONITOR = "val_corrected_picked_ade"
INDEX_FILES = {
    DEFAULT_TRAIN_SPLIT: "index_train.pkl",
    DEFAULT_VAL_SPLIT: "index_val.pkl",
    "test": "index_test.pkl",
}
SCORE_METRIC_NAMES = {
    "baseline_score_mse",
    "corrected_score_mse",
    "baseline_score_mae",
    "corrected_score_mae",
}


def default_num_workers() -> int:
    return os.cpu_count() or 1

def default_index_file(split: str) -> str:
    return INDEX_FILES.get(split, INDEX_FILES[DEFAULT_TRAIN_SPLIT])


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def set_eval_mode(*modules: nn.Module) -> None:
    for module in modules:
        module.eval()


def checkpoint_value(checkpoint: dict[str, Any], key: str) -> Any:
    if key in checkpoint:
        return checkpoint[key]
    args = checkpoint.get("args")
    if isinstance(args, dict):
        return args.get(key)
    return None


def resolve_required_arg(
    cli_value: Any,
    checkpoint: dict[str, Any],
    key: str,
) -> Any:
    if cli_value is not None:
        return cli_value
    checkpoint_val = checkpoint_value(checkpoint, key)
    if checkpoint_val is None:
        raise ValueError(
            f"Missing required argument '{key}'. Pass --{key} or use a residual checkpoint that stores it."
        )
    return checkpoint_val


def build_train_val_loaders(
    *,
    data_dir: str,
    train_items: int | None,
    val_items: int | None,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = WaymoE2E(
        indexFile=default_index_file(DEFAULT_TRAIN_SPLIT),
        data_dir=data_dir,
        n_items=train_items,
        seed=seed,
    )
    val_dataset = WaymoE2E(
        indexFile=default_index_file(DEFAULT_VAL_SPLIT),
        data_dir=data_dir,
        n_items=val_items,
        seed=seed + 1,
    )
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_with_images,
            persistent_workers=False,
            pin_memory=False,
            shuffle=True,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_with_images,
            persistent_workers=False,
            pin_memory=False,
            shuffle=False,
        ),
    )


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


def extract_batch_targets(
    model: LitModel,
    sae: SparseAE,
    batch: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    past = batch["PAST"].to(device)
    future = batch["FUTURE"].to(device)
    intent = batch["INTENT"].to(device)

    if "IMAGES" in batch:
        images = [
            image.to(device) if isinstance(image, torch.Tensor) else image
            for image in batch["IMAGES"]
        ]
    else:
        images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)

    model_inputs = {"PAST": past, "IMAGES": images, "INTENT": intent}
    sae.internal_acts = None
    output = model(model_inputs)

    if not isinstance(output, dict):
        raise TypeError("Expected the backbone model to return a dict with trajectory and scores")
    if output.get("scores") is None:
        raise RuntimeError("Backbone model does not expose scorer predictions in output['scores']")
    if sae.internal_acts is None:
        raise RuntimeError("No SAE activations were captured from the registered hook")

    pred_scores = output["scores"]
    pred_future = output["trajectory"]
    t_steps = future.size(1)
    pred = pred_future.view(pred_future.size(0), -1, t_steps, 2)
    ade_per_mode = torch.norm(pred - future[:, None, :, :], dim=-1).mean(dim=-1)

    sae_hidden = compute_hidden_activations(sae, sae.internal_acts)
    trajectories = pred.reshape(pred.size(0), pred.size(1), -1)

    return {
        "sae_hidden": sae_hidden.detach(),
        "trajectories": trajectories.detach(),
        "score_pred": pred_scores.detach(),
        "true_ade": ade_per_mode.detach(),
        "residual_target": (ade_per_mode - pred_scores).detach(),
    }


def flatten_proposal_batch(batch_targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    sae_hidden = batch_targets["sae_hidden"]
    trajectories = batch_targets["trajectories"]
    score_pred = batch_targets["score_pred"]
    residual_target = batch_targets["residual_target"]
    true_ade = batch_targets["true_ade"]

    batch_size, n_modes, traj_dim = trajectories.shape
    sae_dim = sae_hidden.size(-1)

    return {
        "sae_hidden": sae_hidden[:, None, :].expand(-1, n_modes, -1).reshape(batch_size * n_modes, sae_dim),
        "trajectory": trajectories.reshape(batch_size * n_modes, traj_dim),
        "score_pred": score_pred.reshape(batch_size * n_modes),
        "residual_target": residual_target.reshape(batch_size * n_modes),
        "true_ade": true_ade.reshape(batch_size * n_modes),
    }


def score_comparison_metrics(
    baseline_scores: torch.Tensor,
    corrected_scores: torch.Tensor,
    true_ade: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics = {
        "baseline_score_mse": torch.mean((baseline_scores - true_ade) ** 2),
        "corrected_score_mse": torch.mean((corrected_scores - true_ade) ** 2),
        "baseline_score_mae": torch.mean(torch.abs(baseline_scores - true_ade)),
        "corrected_score_mae": torch.mean(torch.abs(corrected_scores - true_ade)),
    }
    for prefix, scores in (("baseline", baseline_scores), ("corrected", corrected_scores)):
        for key, value in selection_metrics(scores, true_ade).items():
            metrics[f"{prefix}_{key}"] = value
    return metrics


def selection_metrics(score_matrix: torch.Tensor, ade_per_mode: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size = score_matrix.size(0)
    batch_idx = torch.arange(batch_size, device=score_matrix.device)
    picked_idx = score_matrix.argmin(dim=1)
    picked_ade = ade_per_mode[batch_idx, picked_idx]
    oracle_ade = ade_per_mode.min(dim=1).values

    oracle_ranking = ade_per_mode.argsort(dim=1)
    oracle_rank_of = oracle_ranking.argsort(dim=1)
    picked_rank = oracle_rank_of[batch_idx, picked_idx].float()

    n_modes = ade_per_mode.size(1)
    scorer_ranks = score_matrix.argsort(dim=1).argsort(dim=1).float()
    oracle_ranks = oracle_rank_of.float()
    d = oracle_ranks - scorer_ranks
    denom = max(n_modes * (n_modes * n_modes - 1), 1)
    spearman = 1 - 6 * (d.square().sum(dim=1) / denom)

    metrics = {
        "picked_ade": picked_ade.mean(),
        "oracle_ade": oracle_ade.mean(),
        "regret": (picked_ade - oracle_ade).mean(),
        "mean_rank": picked_rank.mean(),
        "spearman": spearman.mean(),
    }
    for topk in (1, 5, 10):
        if topk <= n_modes:
            metrics[f"top{topk}_acc"] = (picked_rank < topk).float().mean()
    return metrics


class SAEScorerResidual(nn.Module):
    def __init__(
        self,
        sae_dim: int,
        traj_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sae_tower = nn.Sequential(
            nn.LayerNorm(sae_dim),
            nn.Linear(sae_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.traj_tower = nn.Sequential(
            nn.LayerNorm(traj_dim),
            nn.Linear(traj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_tower = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fusion_dim = hidden_dim * 2 + hidden_dim // 2
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        sae_hidden: torch.Tensor,
        trajectory: torch.Tensor,
        score_pred: torch.Tensor,
    ) -> torch.Tensor:
        if score_pred.ndim == 1:
            score_pred = score_pred.unsqueeze(-1)
        features = torch.cat(
            [
                self.sae_tower(sae_hidden),
                self.traj_tower(trajectory),
                self.score_tower(score_pred),
            ],
            dim=-1,
        )
        return self.head(features).squeeze(-1)


def infer_residual_dims(model: LitModel, sae: SparseAE) -> tuple[int, int]:
    return sae.encoder.weight.shape[0], model.model.horizon * 2


class LitSAEScorerResidual(pl.LightningModule):
    def __init__(
        self,
        *,
        model_checkpoint_path: str,
        sae_checkpoint_path: str,
        block_idx: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        loss_type: str = "huber",
        topk_train: int = 10,
        topk_weight: float = 3.0,
        rest_weight: float = 1.0,
        rank_loss_weight: float = 0.25,
        rank_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        backbone_model, sae, hook_handle = load_model_and_sae(
            model_checkpoint_path=model_checkpoint_path,
            sae_checkpoint_path=sae_checkpoint_path,
            block_idx=block_idx,
        )
        freeze_module(backbone_model)
        freeze_module(sae)
        set_eval_mode(backbone_model, sae)

        self.backbone_model = backbone_model
        self.sae = sae
        self.hook_handle = hook_handle
        self.sae_dim, self.traj_dim = infer_residual_dims(backbone_model, sae)
        self.residual_model = SAEScorerResidual(
            sae_dim=self.sae_dim,
            traj_dim=self.traj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.save_hyperparameters({
            key: value
            for key, value in locals().items()
            if key not in {"self", "backbone_model", "sae", "hook_handle"}
        } | {"sae_dim": self.sae_dim, "traj_dim": self.traj_dim})

    def _set_backbone_eval(self) -> None:
        set_eval_mode(self.backbone_model, self.sae)

    def remove_hook(self) -> None:
        if getattr(self, "hook_handle", None) is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not isinstance(batch, dict):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        if "IMAGES_JPEG" in batch:
            images_jpeg = batch["IMAGES_JPEG"]
            batch_wo_jpeg = dict(batch)
            batch_wo_jpeg.pop("IMAGES_JPEG", None)
            moved = super().transfer_batch_to_device(batch_wo_jpeg, device, dataloader_idx)
            moved["IMAGES"] = self.backbone_model.decode_batch_jpeg(images_jpeg, device=device)
            return moved

        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_fit_start(self) -> None:
        self._set_backbone_eval()

    def on_train_epoch_start(self) -> None:
        self._set_backbone_eval()

    def on_validation_epoch_start(self) -> None:
        self._set_backbone_eval()

    def on_fit_end(self) -> None:
        self.remove_hook()

    def teardown(self, stage: str | None) -> None:
        self.remove_hook()
        return super().teardown(stage)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.residual_model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def _log_step_metric(
        self,
        name: str,
        value: torch.Tensor,
        *,
        batch_size: int,
        prog_bar: bool = False,
    ) -> None:
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=False,
        )

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        self._set_backbone_eval()

        with torch.no_grad():
            batch_targets = extract_batch_targets(
                self.backbone_model,
                self.sae,
                batch,
                self.device,
            )

        flat_batch = flatten_proposal_batch(batch_targets)
        proposal_weights = proposal_weights_from_ade(
            batch_targets["true_ade"],
            topk=self.hparams.topk_train,
            topk_weight=self.hparams.topk_weight,
            rest_weight=self.hparams.rest_weight,
        )
        flat_weights = proposal_weights.reshape(-1)

        residual_pred_flat = self.residual_model(
            flat_batch["sae_hidden"],
            flat_batch["trajectory"],
            flat_batch["score_pred"],
        )
        residual_pred = residual_pred_flat.view_as(batch_targets["score_pred"])
        reg_loss = correction_loss(
            residual_pred_flat,
            flat_batch["residual_target"],
            loss_type=self.hparams.loss_type,
            weights=flat_weights,
        )
        corrected_scores = batch_targets["score_pred"] + residual_pred
        rank_loss = pairwise_ranking_loss(
            corrected_scores,
            batch_targets["true_ade"],
            proposal_weights=proposal_weights,
            temperature=self.hparams.rank_temperature,
        )
        loss = reg_loss + self.hparams.rank_loss_weight * rank_loss

        sample_count = batch_targets["score_pred"].size(0)
        score_count = batch_targets["score_pred"].numel()
        comparison_metrics = score_comparison_metrics(
            batch_targets["score_pred"],
            corrected_scores,
            batch_targets["true_ade"],
        )

        self._log_step_metric(f"{stage}_loss", loss, batch_size=score_count, prog_bar=(stage == "val"))
        self._log_step_metric(f"{stage}_reg_loss", reg_loss, batch_size=score_count)
        self._log_step_metric(f"{stage}_rank_loss", rank_loss, batch_size=score_count)
        for key, value in comparison_metrics.items():
            self._log_step_metric(
                f"{stage}_{key}",
                value,
                batch_size=score_count if key in SCORE_METRIC_NAMES else sample_count,
                prog_bar=(stage == "val" and key == "corrected_picked_ade"),
            )

        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")


def _load_residual_state_from_lightning_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    state_dict = checkpoint.get("state_dict", {})
    residual_prefix = "residual_model."
    residual_state = {
        key[len(residual_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(residual_prefix)
    }
    if not residual_state:
        raise KeyError("Lightning checkpoint does not contain residual_model.* weights")
    return residual_state, hparams


def build_residual_model_from_metadata(
    metadata: dict[str, Any],
    *,
    device: torch.device,
) -> SAEScorerResidual:
    model = SAEScorerResidual(
        sae_dim=metadata["sae_dim"],
        traj_dim=metadata["traj_dim"],
        hidden_dim=metadata["hidden_dim"],
        dropout=metadata["dropout"],
    )
    model = model.to(device)
    model.eval()
    return model


def load_residual_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[SAEScorerResidual, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model = build_residual_model_from_metadata(checkpoint, device=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model, checkpoint

    residual_state, metadata = _load_residual_state_from_lightning_checkpoint(checkpoint)
    model = build_residual_model_from_metadata(metadata, device=device)
    model.load_state_dict(residual_state, strict=True)
    metadata["args"] = metadata.copy()
    return model, metadata


def save_residual_model(
    output_path: str,
    model: SAEScorerResidual,
    *,
    sae_dim: int,
    traj_dim: int,
    hidden_dim: int,
    dropout: float,
    best_metric: float,
    args: dict[str, Any],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sae_dim": sae_dim,
            "traj_dim": traj_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "best_metric": best_metric,
            "args": args,
        },
        path,
    )


def correction_loss(
    residual_pred: torch.Tensor,
    residual_target: torch.Tensor,
    loss_type: str,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_type == "huber":
        loss = F.smooth_l1_loss(residual_pred, residual_target, reduction="none")
    elif loss_type == "mse":
        loss = F.mse_loss(residual_pred, residual_target, reduction="none")
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    if weights is not None:
        weights = weights.to(loss.device, dtype=loss.dtype)
        loss = loss * weights
        return loss.sum() / weights.sum().clamp_min(1e-8)

    return loss.mean()


def proposal_weights_from_ade(
    ade_per_mode: torch.Tensor,
    *,
    topk: int,
    topk_weight: float,
    rest_weight: float,
) -> torch.Tensor:
    if ade_per_mode.ndim != 2:
        raise ValueError(f"Expected ade_per_mode to be 2D, got shape {tuple(ade_per_mode.shape)}")

    n_modes = ade_per_mode.size(1)
    topk = max(1, min(topk, n_modes))
    oracle_ranks = ade_per_mode.argsort(dim=1).argsort(dim=1)
    weights = torch.full_like(ade_per_mode, float(rest_weight))
    weights = torch.where(
        oracle_ranks < topk,
        torch.full_like(weights, float(topk_weight)),
        weights,
    )
    return weights / weights.mean().clamp_min(1e-8)


def pairwise_ranking_loss(
    score_matrix: torch.Tensor,
    ade_per_mode: torch.Tensor,
    *,
    proposal_weights: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    if score_matrix.ndim != 2 or ade_per_mode.ndim != 2:
        raise ValueError("pairwise_ranking_loss expects 2D score and ADE matrices")
    if score_matrix.shape != ade_per_mode.shape:
        raise ValueError(
            f"Score/ADE shape mismatch: {tuple(score_matrix.shape)} vs {tuple(ade_per_mode.shape)}"
        )

    score_diff = score_matrix[:, :, None] - score_matrix[:, None, :]
    ade_diff = ade_per_mode[:, :, None] - ade_per_mode[:, None, :]
    order_sign = torch.sign(-ade_diff)

    pair_mask = torch.triu(
        torch.ones_like(order_sign, dtype=torch.bool),
        diagonal=1,
    ) & (order_sign != 0)
    pair_loss = F.softplus((order_sign * score_diff) / max(temperature, 1e-6))

    if proposal_weights is not None:
        pair_weights = 0.5 * (
            proposal_weights[:, :, None] + proposal_weights[:, None, :]
        )
        pair_loss = pair_loss * pair_weights

    pair_loss = pair_loss.masked_select(pair_mask)
    if pair_loss.numel() == 0:
        return score_matrix.new_tensor(0.0)
    return pair_loss.mean()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint_path", type=str, required=True)
    parser.add_argument("--sae_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--train_items", type=int, default=250_000)
    parser.add_argument("--val_items", type=int, default=25_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=default_num_workers())
    parser.add_argument("--block_idx", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--loss_type", type=str, default="huber", choices=("mse", "huber"))
    parser.add_argument("--topk_train", type=int, default=10)
    parser.add_argument("--topk_weight", type=float, default=3.0)
    parser.add_argument("--rest_weight", type=float, default=1.0)
    parser.add_argument("--rank_loss_weight", type=float, default=0.25)
    parser.add_argument("--rank_temperature", type=float, default=1.0)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    return parser


def make_run_name() -> str:
    return f"sae_scorer_residual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def export_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **vars(args),
        "train_split": DEFAULT_TRAIN_SPLIT,
        "val_split": DEFAULT_VAL_SPLIT,
        "patience": DEFAULT_PATIENCE,
        "min_delta": DEFAULT_MIN_DELTA,
        "run_name": None,
        "log_every_n_steps": DEFAULT_LOG_EVERY_N_STEPS,
        "wandb_project": "robotvision",
        "seed": DEFAULT_SEED,
    }


def export_best_residual_checkpoint(
    lightning_checkpoint_path: str,
    output_path: str,
    args: dict[str, Any],
    best_metric: float,
) -> None:
    best_module = LitSAEScorerResidual.load_from_checkpoint(lightning_checkpoint_path)
    try:
        save_residual_model(
            output_path,
            best_module.residual_model,
            sae_dim=best_module.sae_dim,
            traj_dim=best_module.traj_dim,
            hidden_dim=best_module.hparams.hidden_dim,
            dropout=best_module.hparams.dropout,
            best_metric=best_metric,
            args=args,
        )
    finally:
        best_module.remove_hook()


def build_loggers(log_dir: Path, run_name: str) -> list[CSVLogger | WandbLogger]:
    loggers: list[CSVLogger | WandbLogger] = [CSVLogger(log_dir.as_posix(), name=run_name)]
    loggers.append(
        WandbLogger(
            name=run_name,
            save_dir=log_dir.as_posix(),
            project="robotvision",
            log_model=True,
        )
    )
    return loggers


def build_callbacks(checkpoint_dir: Path) -> list[ModelCheckpoint | EarlyStopping]:
    return [
        ModelCheckpoint(
            monitor=VAL_MONITOR,
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_dir.as_posix(),
            filename="sae-scorer-residual-{epoch:02d}-{val_corrected_picked_ade:.4f}",
        ),
        EarlyStopping(
            monitor=VAL_MONITOR,
            mode="min",
            patience=DEFAULT_PATIENCE,
            min_delta=DEFAULT_MIN_DELTA,
        ),
    ]


def trainer_precision() -> tuple[str, str]:
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "32-true"
    if accelerator == "cuda":
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        torch.set_float32_matmul_precision("medium")
    return accelerator, precision


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pl.seed_everything(DEFAULT_SEED, workers=True)

    train_loader, val_loader = build_train_val_loaders(
        data_dir=args.data_dir,
        train_items=args.train_items,
        val_items=args.val_items,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=DEFAULT_SEED,
    )
    lit_model = LitSAEScorerResidual(
        model_checkpoint_path=args.model_checkpoint_path,
        sae_checkpoint_path=args.sae_checkpoint_path,
        block_idx=args.block_idx,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        loss_type=args.loss_type,
        topk_train=args.topk_train,
        topk_weight=args.topk_weight,
        rest_weight=args.rest_weight,
        rank_loss_weight=args.rank_loss_weight,
        rank_temperature=args.rank_temperature,
    )

    base_path = Path(args.data_dir).parent
    run_name = make_run_name()
    log_dir = Path(args.log_dir) if args.log_dir is not None else base_path / "logs"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir is not None else base_path / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = build_callbacks(checkpoint_dir)
    checkpoint_callback = callbacks[0]
    accelerator, precision = trainer_precision()

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        logger=build_loggers(log_dir, run_name),
        callbacks=callbacks,
        log_every_n_steps=DEFAULT_LOG_EVERY_N_STEPS,
    )
    trainer.fit(lit_model, train_loader, val_loader)

    if checkpoint_callback.best_model_path:
        best_metric = float(checkpoint_callback.best_model_score.item())
        export_best_residual_checkpoint(
            checkpoint_callback.best_model_path,
            args.output_path,
            export_args(args),
            best_metric=best_metric,
        )
        print(f"Best Lightning checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Exported residual checkpoint: {args.output_path}")
        print(f"Best val corrected pick ADE: {best_metric:.6f}")
    else:
        raise RuntimeError("Trainer finished without producing a best checkpoint")


if __name__ == "__main__":
    main()
