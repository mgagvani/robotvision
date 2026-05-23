from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from analyze_sae_control import ACCEL_BANK_TOP_K_PER_SIGN
from models.initial_sae import SparseAutoEncoder, load_torch_file
from models.pm_block_sae import PMBlockTopKSAE, load_pm_block_sae


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str, device: torch.device):
    from models.base_model import LitModel
    from models.feature_extractors import SAMFeatures
    from models.monocular import DeepMonocularModel

    out_dim = 20 * 2
    model = DeepMonocularModel(
        feature_extractor=SAMFeatures(
            model_name="timm/vit_pe_spatial_small_patch16_512.fb",
            frozen=True,
        ),
        out_dim=out_dim,
        n_blocks=4,
        n_proposals=50,
    )
    lit_model = LitModel.load_from_checkpoint(
        checkpoint_path,
        model=model,
        lr=1e-4,
        map_location="cpu",
        weights_only=False,
    )
    model = lit_model.model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, lit_model


def load_control_sae_model(ckpt_path: Path, device: torch.device) -> SparseAutoEncoder:
    ckpt = load_torch_file(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(ckpt).__name__}")
    if ckpt.get("mode") != "control":
        raise ValueError(f"Checkpoint mode must be 'control', got {ckpt.get('mode')!r}")

    model = SparseAutoEncoder(
        in_dims=int(ckpt["in_dims"]),
        expansion=int(ckpt["expansion"]),
        sparsity=float(ckpt.get("sparsity", 1e-4)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def summarise_metric(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(values.mean().item()),
        "p50": float(values.quantile(0.50).item()),
        "p90": float(values.quantile(0.90).item()),
    }


def save_trajectory_plot(
    out_path: Path,
    past: torch.Tensor,
    future: torch.Tensor,
    base_pred: torch.Tensor,
    sae_pred: torch.Tensor,
    edited_pred: torch.Tensor,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(past[:, 0].numpy(), past[:, 1].numpy(), color="black", marker="o", linewidth=1.5, label="Past")
    ax.plot(future[:, 0].numpy(), future[:, 1].numpy(), color="green", marker="o", linewidth=2.0, label="GT")
    ax.plot(base_pred[:, 0].numpy(), base_pred[:, 1].numpy(), color="tab:blue", marker="o", linewidth=2.0, label="Base")
    ax.plot(sae_pred[:, 0].numpy(), sae_pred[:, 1].numpy(), color="tab:orange", marker="o", linewidth=2.0, label="SAE")
    ax.plot(
        edited_pred[:, 0].numpy(),
        edited_pred[:, 1].numpy(),
        color="tab:red",
        marker="o",
        linewidth=2.0,
        label="Edited",
    )
    ax.scatter([past[-1, 0].item()], [past[-1, 1].item()], color="black", s=40)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_example_title(example: dict) -> str:
    behavior_desc = "neutral"
    if example["accelerating_scene"]:
        behavior_desc = "accelerating->decelerate"
    elif example["decelerating_scene"]:
        behavior_desc = "decelerating->accelerate"
    return (
        f"{example['name']}\n"
        f"kind={example['sae_kind']} block={example['sae_block_index']} flip={example['flip_behaviors']} "
        f"behavior={behavior_desc} success={example.get('flip_success', False)}\n"
        f"target_req={example['target_accel_mps2']:.3f} desired_scene_target={example['desired_accel_mps2']:.3f} "
        f"target_apply={example['applied_accel_mps2']:.3f} "
        f"raw={example['target_raw_accel']:.4f} bank={example['accel_bank_sign']}\n"
        f"base_mean={example['base_accel_mean']:.4f} sae_mean={example['sae_accel_mean']:.4f} "
        f"edited_mean={example['edited_accel_mean']:.4f} std={example['edited_accel_std']:.6f}\n"
        f"max_err={example['max_abs_accel_error']:.6e} template_resid={example['solver_residual_raw']:.6e} "
        f"features={example['selected_feature_count']}\n"
        f"best_idx base/sae/edited={example['best_idx_base']}/{example['best_idx_sae']}/{example['best_idx_edited']} "
        f"| editShiftADE={example['edit_shift_ade']:.3f} "
        f"baseADE={example['base_ade']:.3f} saeADE={example['sae_ade']:.3f} editedADE={example['edited_ade']:.3f}"
    )


def save_example_plots(examples: list[dict], plot_dir: Path, max_plots: int) -> int:
    plot_dir.mkdir(parents=True, exist_ok=True)
    for plot_rank, example in enumerate(examples[:max_plots], start=1):
        safe_name = example["name"].replace("/", "_")
        plot_path = plot_dir / f"{plot_rank:02d}_{safe_name}_shift_{example['edit_shift_ade']:.3f}.png"
        save_trajectory_plot(
            out_path=plot_path,
            past=example["past"][:, 0:2],
            future=example["future"],
            base_pred=example["base_best"],
            sae_pred=example["sae_best"],
            edited_pred=example["edited_best"],
            title=build_example_title(example),
        )
    return min(max_plots, len(examples))


def target_accel_to_raw_control(
    target_accel_mps2: float | torch.Tensor,
    max_accel: float,
    accel_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_accel <= 0:
        raise ValueError(f"max_accel must be positive, got {max_accel}")
    if not (0.0 < accel_eps < 1.0):
        raise ValueError(f"accel_eps must be between 0 and 1, got {accel_eps}")

    target = torch.as_tensor(target_accel_mps2, dtype=torch.float32)
    if torch.any(target.abs() > max_accel):
        raise ValueError(
            f"Requested target acceleration must lie within [-{max_accel}, {max_accel}], got {target.tolist()}"
        )

    normalized = torch.clamp(target / max_accel, -1.0 + accel_eps, 1.0 - accel_eps)
    raw = torch.atanh(normalized)
    applied = torch.tanh(raw) * max_accel
    return raw, applied


def raw_control_to_accel(raw_accel: torch.Tensor, max_accel: float) -> torch.Tensor:
    return torch.tanh(raw_accel) * max_accel


def raw_control_to_omega(raw_omega: torch.Tensor, max_omega: float) -> torch.Tensor:
    return torch.tanh(raw_omega) * max_omega


def derive_gt_speed_change_labels(
    future: torch.Tensor,
    past: torch.Tensor,
    dt: float,
    speed_delta_thresh: float,
) -> torch.Tensor:
    start_xy = past[:, -1, 0:2]
    speed0 = torch.linalg.norm(past[:, -1, 2:4], dim=-1)

    prev_xy = torch.cat([start_xy.unsqueeze(1), future[:, :-1]], dim=1)
    step_speed = torch.linalg.norm(future - prev_xy, dim=-1) / dt
    final_speed = step_speed[:, -1]
    delta_speed = final_speed - speed0

    labels = torch.full((future.size(0),), -1, dtype=torch.long, device=future.device)
    labels[delta_speed <= -speed_delta_thresh] = 0
    labels[delta_speed >= speed_delta_thresh] = 1
    return labels


def gather_best_outputs(
    out: dict[str, torch.Tensor],
    max_accel: float,
    max_omega: float,
) -> dict[str, torch.Tensor]:
    scores = out["scores_predicted"]
    control_pred = out["control_pred"]
    batch_size, n_proposals, horizon, _ = control_pred.shape
    traj = out["trajectory_predicted"].view(batch_size, n_proposals, horizon, 2)
    best_idx = scores.argmin(dim=1)
    row_idx = torch.arange(batch_size, device=scores.device)
    best_traj = traj[row_idx, best_idx]
    best_control = control_pred[row_idx, best_idx]
    best_accel = raw_control_to_accel(best_control[..., 0], max_accel)
    best_omega = raw_control_to_omega(best_control[..., 1], max_omega)
    return {
        "best_idx": best_idx,
        "traj": best_traj,
        "control": best_control,
        "accel": best_accel,
        "omega": best_omega,
    }


def load_accel_bank(accel_bank_pt: Path) -> dict[str, object]:
    bank = load_torch_file(str(accel_bank_pt), map_location="cpu")
    if not isinstance(bank, dict):
        raise TypeError(f"Expected accel bank dict, got {type(bank).__name__}")
    for key in ("positive", "negative", "top_k_per_sign", "horizon"):
        if key not in bank:
            raise KeyError(f"Acceleration bank is missing required key '{key}'")
    return bank


def select_signed_bank(
    accel_bank: dict[str, object],
    target_raw_accel: float,
    top_k_per_sign: int,
) -> dict[str, object]:
    if abs(target_raw_accel) <= 1e-9:
        horizon = int(accel_bank["horizon"])
        return {
            "sign": "zero",
            "feature_indices": [],
            "decoder_accel_submatrix": torch.zeros(horizon, 0, dtype=torch.float32),
            "decoder_omega_submatrix": torch.zeros(horizon, 0, dtype=torch.float32),
        }

    sign_name = "positive" if target_raw_accel > 0.0 else "negative"
    bank = accel_bank[sign_name]
    n_features = min(top_k_per_sign, len(bank["feature_indices"]))
    return {
        "sign": sign_name,
        "feature_indices": list(bank["feature_indices"][:n_features]),
        "decoder_accel_submatrix": bank["decoder_accel_submatrix"][:, :n_features].clone().to(torch.float64),
        "decoder_omega_submatrix": bank["decoder_omega_submatrix"][:, :n_features].clone().to(torch.float64),
    }


def solve_projected_gradient_nnls(
    accel_matrix: torch.Tensor,
    omega_matrix: torch.Tensor,
    target_raw_accel: float,
    omega_penalty: float,
    l2_penalty: float = 1e-4,
    max_iters: int = 600,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, int]:
    n_features = accel_matrix.shape[1]
    if n_features == 0:
        return torch.zeros(0, dtype=accel_matrix.dtype), 0

    gram = accel_matrix.T @ accel_matrix
    if omega_penalty > 0.0:
        gram = gram + omega_penalty * (omega_matrix.T @ omega_matrix)
    gram = gram + l2_penalty * torch.eye(n_features, dtype=accel_matrix.dtype)
    rhs = accel_matrix.T @ torch.full((accel_matrix.shape[0],), target_raw_accel, dtype=accel_matrix.dtype)

    eigvals = torch.linalg.eigvalsh(gram)
    lipschitz = max(float((2.0 * eigvals.max()).item()), 1e-6)
    step = 1.0 / lipschitz
    coeff = torch.zeros(n_features, dtype=accel_matrix.dtype)

    for iteration in range(1, max_iters + 1):
        grad = 2.0 * (gram @ coeff - rhs)
        next_coeff = torch.clamp_min(coeff - step * grad, 0.0)
        if torch.max(torch.abs(next_coeff - coeff)).item() <= tol:
            coeff = next_coeff
            return coeff, iteration
        coeff = next_coeff

    return coeff, max_iters


def solve_control_accel_latent_template(
    accel_bank: dict[str, object],
    target_raw_accel: float,
    top_k_per_sign: int,
    omega_penalty: float,
    l2_penalty: float = 1e-4,
    max_iters: int = 600,
    tol: float = 1e-6,
) -> dict[str, object]:
    signed_bank = select_signed_bank(
        accel_bank=accel_bank,
        target_raw_accel=target_raw_accel,
        top_k_per_sign=top_k_per_sign,
    )
    accel_matrix = signed_bank["decoder_accel_submatrix"]
    omega_matrix = signed_bank["decoder_omega_submatrix"]
    coeff, solver_iters = solve_projected_gradient_nnls(
        accel_matrix=accel_matrix,
        omega_matrix=omega_matrix,
        target_raw_accel=target_raw_accel,
        omega_penalty=omega_penalty,
        l2_penalty=l2_penalty,
        max_iters=max_iters,
        tol=tol,
    )

    decoded_raw_accel = accel_matrix @ coeff
    decoded_raw_omega = omega_matrix @ coeff
    target_vector = torch.full_like(decoded_raw_accel, target_raw_accel)
    residual = decoded_raw_accel - target_vector
    return {
        "sae_kind": "control",
        "bank_sign": signed_bank["sign"],
        "feature_indices": list(signed_bank["feature_indices"]),
        "template_values": coeff.to(torch.float32),
        "solver_residual_raw": float(residual.abs().mean().item()) if residual.numel() else 0.0,
        "omega_leakage_mean": float(decoded_raw_omega.abs().mean().item()) if decoded_raw_omega.numel() else 0.0,
        "selected_feature_count": len(signed_bank["feature_indices"]),
        "solver_iters": solver_iters,
    }


def build_control_latent_template_editor(
    feature_indices: torch.Tensor,
    template_values: torch.Tensor,
    batch_size: int,
    n_proposals: int,
    latent_dim: int,
):
    if feature_indices.ndim != 1 or template_values.ndim != 1:
        raise ValueError("feature_indices and template_values must both be 1-D tensors")
    if feature_indices.numel() != template_values.numel():
        raise ValueError(
            f"feature_indices has {feature_indices.numel()} entries but template_values has {template_values.numel()}"
        )

    def edit(latents: torch.Tensor) -> torch.Tensor:
        expected_shape = (batch_size * n_proposals, latent_dim)
        if latents.shape != expected_shape:
            raise ValueError(f"Expected latent shape {expected_shape}, got {tuple(latents.shape)}")

        edited = latents.clone().view(batch_size, n_proposals, latent_dim)
        if feature_indices.numel() > 0:
            edited[..., feature_indices] = template_values.view(1, 1, -1)
        return edited.view(batch_size * n_proposals, latent_dim)

    return edit


def checkpoint_is_pm_block_sae(raw_ckpt: dict[str, object]) -> bool:
    return "state_dict" in raw_ckpt and ("block_index" in raw_ckpt or "token_key" in raw_ckpt)


def load_sae_for_intervention(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[object, str, dict[str, object]]:
    raw_ckpt = load_torch_file(str(ckpt_path), map_location="cpu")
    if not isinstance(raw_ckpt, dict):
        raise TypeError(f"Expected SAE checkpoint dict, got {type(raw_ckpt).__name__}")

    if raw_ckpt.get("mode") == "control" and "model" in raw_ckpt:
        return load_control_sae_model(ckpt_path, device), "control", raw_ckpt
    if checkpoint_is_pm_block_sae(raw_ckpt):
        return load_pm_block_sae(ckpt_path, device), "planner_query_block", raw_ckpt

    raise ValueError(
        f"Unsupported SAE checkpoint format at {ckpt_path}. "
        f"Available keys: {sorted(raw_ckpt.keys())}"
    )


def build_block_latent_template_editor(
    feature_indices: torch.Tensor,
    template_values: torch.Tensor,
    batch_size: int,
    latent_dim: int,
):
    if feature_indices.ndim != 1 or template_values.ndim != 1:
        raise ValueError("feature_indices and template_values must both be 1-D tensors")
    if feature_indices.numel() != template_values.numel():
        raise ValueError(
            f"feature_indices has {feature_indices.numel()} entries but template_values has {template_values.numel()}"
        )

    def edit(latents: torch.Tensor) -> torch.Tensor:
        expected_shape = (batch_size, latent_dim)
        if latents.shape != expected_shape:
            raise ValueError(f"Expected latent shape {expected_shape}, got {tuple(latents.shape)}")

        edited = latents.clone()
        if feature_indices.numel() > 0:
            edited[:, feature_indices] = template_values.view(1, -1)
        return edited

    return edit


def build_block_flip_latent_editor(
    accel_feature_indices: torch.Tensor,
    accel_template_values: torch.Tensor,
    decel_feature_indices: torch.Tensor,
    decel_template_values: torch.Tensor,
    accelerate_mask: torch.Tensor,
    decelerate_mask: torch.Tensor,
    batch_size: int,
    latent_dim: int,
):
    accelerate_mask = accelerate_mask.to(dtype=torch.bool)
    decelerate_mask = decelerate_mask.to(dtype=torch.bool)

    def edit(latents: torch.Tensor) -> torch.Tensor:
        expected_shape = (batch_size, latent_dim)
        if latents.shape != expected_shape:
            raise ValueError(f"Expected latent shape {expected_shape}, got {tuple(latents.shape)}")

        edited = latents.clone()
        if decelerate_mask.any() and accel_feature_indices.numel() > 0:
            decel_idx = torch.nonzero(decelerate_mask, as_tuple=False).flatten()
            edited[decel_idx[:, None], accel_feature_indices[None, :]] = accel_template_values.view(1, -1)
        if accelerate_mask.any() and decel_feature_indices.numel() > 0:
            accel_idx = torch.nonzero(accelerate_mask, as_tuple=False).flatten()
            edited[accel_idx[:, None], decel_feature_indices[None, :]] = decel_template_values.view(1, -1)
        return edited

    return edit


def build_loader(dataset, batch_size: int):
    from models.base_model import collate_with_images

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_with_images,
        pin_memory=False,
        persistent_workers=False,
    )


def collect_block_accel_calibration(
    model,
    lit_model,
    sae: PMBlockTopKSAE,
    loader,
    device: torch.device,
    block_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_batches = []
    scene_raw_accel_batches = []
    block_key = f"planner_query_tok_block_{block_index}"

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Calibrating block {block_index} latents"):
            past = batch["PAST"].to(device, non_blocking=True)
            intent = batch["INTENT"].to(device, non_blocking=True)
            images = lit_model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
            model_input = {
                "PAST": past,
                "IMAGES": images,
                "INTENT": intent,
            }

            base_out = model(model_input)
            if block_key not in base_out:
                raise KeyError(f"Model output is missing expected key '{block_key}'")

            query_block = base_out[block_key]
            latents = sae(query_block)["latents"]
            base_best = gather_best_outputs(base_out, max_accel=model.max_accel, max_omega=model.max_omega)
            scene_raw_accel = base_best["control"][..., 0].mean(dim=-1)

            latent_batches.append(latents.cpu())
            scene_raw_accel_batches.append(scene_raw_accel.cpu())

    if not latent_batches:
        raise RuntimeError("No calibration batches were processed for block-level SAE intervention.")
    return torch.cat(latent_batches, dim=0), torch.cat(scene_raw_accel_batches, dim=0)


def solve_block_accel_latent_template(
    latents: torch.Tensor,
    scene_raw_accel: torch.Tensor,
    target_raw_accel: float,
    top_k_per_sign: int,
    template_scene_count: int,
) -> dict[str, object]:
    if latents.ndim != 2:
        raise ValueError(f"Expected latents shape (N, D), got {tuple(latents.shape)}")
    if scene_raw_accel.ndim != 1 or scene_raw_accel.shape[0] != latents.shape[0]:
        raise ValueError(
            f"Expected scene_raw_accel shape ({latents.shape[0]},), got {tuple(scene_raw_accel.shape)}"
        )

    n_scenes = latents.shape[0]
    use_scene_count = max(1, min(template_scene_count, n_scenes))
    raw_distance = torch.abs(scene_raw_accel - target_raw_accel)
    nearest_idx = torch.argsort(raw_distance)[:use_scene_count]
    target_latents = latents[nearest_idx]
    target_mean = target_latents.mean(dim=0)

    if target_raw_accel > 0.0:
        ref_mask = scene_raw_accel < 0.0
        ref_target = -abs(target_raw_accel)
        sign_name = "positive"
    elif target_raw_accel < 0.0:
        ref_mask = scene_raw_accel > 0.0
        ref_target = abs(target_raw_accel)
        sign_name = "negative"
    else:
        ref_mask = torch.abs(scene_raw_accel) > 0.5
        ref_target = 0.0
        sign_name = "zero"

    if bool(ref_mask.any().item()):
        ref_candidates = torch.nonzero(ref_mask, as_tuple=False).flatten()
        ref_distance = torch.abs(scene_raw_accel[ref_candidates] - ref_target)
        ref_count = min(use_scene_count, ref_candidates.numel())
        ref_idx = ref_candidates[torch.argsort(ref_distance)[:ref_count]]
    else:
        ref_count = use_scene_count
        ref_idx = torch.argsort(torch.abs(scene_raw_accel))[:ref_count]

    ref_latents = latents[ref_idx]
    ref_mean = ref_latents.mean(dim=0)

    contrast = target_mean - ref_mean
    ranked = torch.argsort(torch.abs(contrast), descending=True)
    selected = ranked[: min(top_k_per_sign, ranked.numel())]

    return {
        "sae_kind": "planner_query_block",
        "bank_sign": sign_name,
        "feature_indices": selected.tolist(),
        "template_values": target_mean[selected].to(torch.float32),
        "solver_residual_raw": float(raw_distance[nearest_idx].mean().item()),
        "omega_leakage_mean": 0.0,
        "selected_feature_count": int(selected.numel()),
        "solver_iters": int(use_scene_count),
        "template_scene_count": int(use_scene_count),
        "reference_scene_count": int(ref_idx.numel()),
    }


def build_template_solution(
    args,
    model,
    lit_model,
    sae,
    sae_kind: str,
    target_raw_accel: float,
    dataset,
    device: torch.device,
) -> tuple[dict[str, object], Optional[int]]:
    if sae_kind == "control":
        if args.accel_bank_pt is None:
            raise ValueError("--accel_bank_pt is required when using a control SAE checkpoint.")
        accel_bank = load_accel_bank(Path(args.accel_bank_pt))
        return (
            solve_control_accel_latent_template(
                accel_bank=accel_bank,
                target_raw_accel=target_raw_accel,
                top_k_per_sign=args.top_k_per_sign,
                omega_penalty=args.omega_penalty,
                l2_penalty=args.solver_l2_penalty,
                max_iters=args.solver_max_iters,
                tol=args.solver_tol,
            ),
            None,
        )

    if not isinstance(sae, PMBlockTopKSAE):
        raise TypeError(f"Expected PMBlockTopKSAE for planner block path, got {type(sae).__name__}")

    block_index = args.planner_sae_block_index
    if block_index is None:
        block_index = sae.metadata.block_index
    num_blocks = len(model.blocks)
    if block_index < 0 or block_index >= num_blocks:
        raise ValueError(f"planner_sae_block_index must be in [0, {num_blocks - 1}], got {block_index}")

    calibration_loader = build_loader(dataset, batch_size=args.batch_size)
    latents, scene_raw_accel = collect_block_accel_calibration(
        model=model,
        lit_model=lit_model,
        sae=sae,
        loader=calibration_loader,
        device=device,
        block_index=block_index,
    )
    return (
        solve_block_accel_latent_template(
            latents=latents,
            scene_raw_accel=scene_raw_accel,
            target_raw_accel=target_raw_accel,
            top_k_per_sign=args.top_k_per_sign,
            template_scene_count=args.template_scene_count,
        ),
        block_index,
    )


def build_flip_template_solutions(
    args,
    model,
    lit_model,
    sae,
    sae_kind: str,
    dataset,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object], Optional[int]]:
    target_mag = abs(float(args.target_accel_mps2))
    if target_mag <= 0.0:
        raise ValueError("--target_accel_mps2 must be non-zero when using --flip_behaviors")

    accel_solution, block_index = build_template_solution(
        args=args,
        model=model,
        lit_model=lit_model,
        sae=sae,
        sae_kind=sae_kind,
        target_raw_accel=float(target_accel_to_raw_control(target_mag, model.max_accel, args.accel_eps)[0].item()),
        dataset=dataset,
        device=device,
    )
    decel_solution, block_index_2 = build_template_solution(
        args=args,
        model=model,
        lit_model=lit_model,
        sae=sae,
        sae_kind=sae_kind,
        target_raw_accel=float(target_accel_to_raw_control(-target_mag, model.max_accel, args.accel_eps)[0].item()),
        dataset=dataset,
        device=device,
    )
    if block_index is None:
        block_index = block_index_2
    return accel_solution, decel_solution, block_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply latent acceleraton interventions using either the legacy control SAE or a PM block SAE."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/gilbreth/chang899/waymo_data/waymo_open_dataset_end_to_end_camera_v_1_0_0",
    )
    parser.add_argument("--index_file", type=str, default="index_val.pkl")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/camera-e2e-epoch=04-val_loss=2.90.ckpt",
    )
    parser.add_argument(
        "--sae_ckpt",
        type=str,
        default="/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/sae_checkpoints/sae_block_3.pt",
    )
    parser.add_argument(
        "--accel_bank_pt",
        type=str,
        default="/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/output/analysis_control/sae_control_accel_bank.pt",
        help="Only used when --sae_ckpt points to a legacy control SAE checkpoint.",
    )
    parser.add_argument("--planner_sae_block_index", type=int, default=None)
    parser.add_argument("--template_scene_count", type=int, default=64)
    parser.add_argument("--target_accel_mps2", type=float, required=True)
    parser.add_argument("--accel_eps", type=float, default=1e-4)
    parser.add_argument("--n_items", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--speed_delta_thresh", type=float, default=0.5)
    parser.add_argument("--flip_behaviors", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_plots", type=int, default=24)
    parser.add_argument("--top_k_per_sign", type=int, default=ACCEL_BANK_TOP_K_PER_SIGN)
    parser.add_argument("--omega_penalty", type=float, default=0.1)
    parser.add_argument("--solver_l2_penalty", type=float, default=1e-4)
    parser.add_argument("--solver_max_iters", type=int, default=600)
    parser.add_argument("--solver_tol", type=float, default=1e-6)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/sae_output/intervention_accel_latent_block3",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    from loader import WaymoE2E

    model, lit_model = load_model(args.model_ckpt, device)
    sae, sae_kind, _ = load_sae_for_intervention(Path(args.sae_ckpt), device)
    model.dt = args.dt

    target_raw_accel_cpu, applied_accel_cpu = target_accel_to_raw_control(
        target_accel_mps2=args.target_accel_mps2,
        max_accel=model.max_accel,
        accel_eps=args.accel_eps,
    )
    target_raw_accel = float(target_raw_accel_cpu.item())
    applied_accel = applied_accel_cpu.to(device=device, dtype=torch.float32)
    flip_applied_accel_mag = float(abs(applied_accel_cpu.item()))

    dataset = WaymoE2E(
        indexFile=args.index_file,
        data_dir=args.data_dir,
        n_items=args.n_items,
        seed=args.seed,
    )
    if args.flip_behaviors:
        accel_template_solution, decel_template_solution, block_index = build_flip_template_solutions(
            args=args,
            model=model,
            lit_model=lit_model,
            sae=sae,
            sae_kind=sae_kind,
            dataset=dataset,
            device=device,
        )
        template_solution = {
            "sae_kind": sae_kind,
            "bank_sign": "flip",
            "feature_indices": [],
            "template_values": torch.empty(0, dtype=torch.float32),
            "solver_residual_raw": 0.5
            * (accel_template_solution["solver_residual_raw"] + decel_template_solution["solver_residual_raw"]),
            "omega_leakage_mean": 0.5
            * (accel_template_solution["omega_leakage_mean"] + decel_template_solution["omega_leakage_mean"]),
            "selected_feature_count": accel_template_solution["selected_feature_count"]
            + decel_template_solution["selected_feature_count"],
            "solver_iters": max(accel_template_solution["solver_iters"], decel_template_solution["solver_iters"]),
            "accel_template_solution": accel_template_solution,
            "decel_template_solution": decel_template_solution,
        }
        accel_feature_indices_cpu = torch.tensor(accel_template_solution["feature_indices"], dtype=torch.long)
        accel_template_values_cpu = accel_template_solution["template_values"].to(torch.float32)
        decel_feature_indices_cpu = torch.tensor(decel_template_solution["feature_indices"], dtype=torch.long)
        decel_template_values_cpu = decel_template_solution["template_values"].to(torch.float32)
    else:
        template_solution, block_index = build_template_solution(
            args=args,
            model=model,
            lit_model=lit_model,
            sae=sae,
            sae_kind=sae_kind,
            target_raw_accel=target_raw_accel,
            dataset=dataset,
            device=device,
        )

        feature_indices_cpu = torch.tensor(template_solution["feature_indices"], dtype=torch.long)
        template_values_cpu = template_solution["template_values"].to(torch.float32)
    loader = build_loader(dataset, batch_size=args.batch_size)

    summary_rows = []
    saved_examples = []
    global_edit_shift_ade = []
    global_edit_shift_fde = []
    global_base_shift_ade = []
    global_base_shift_fde = []
    global_base_ade = []
    global_sae_ade = []
    global_edited_ade = []
    global_accel_error = []
    global_mean_accel_error = []
    global_edited_accel_mean = []
    global_edited_accel_std = []
    global_solver_residual_raw = []
    global_omega_leakage_mean = []
    global_selected_feature_count = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Intervening accel latents")):
            batch_size = batch["PAST"].shape[0]
            past = batch["PAST"].to(device, non_blocking=True)
            future = batch["FUTURE"].to(device, non_blocking=True)
            intent = batch["INTENT"].to(device, non_blocking=True)
            images = lit_model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)

            model_input = {
                "PAST": past,
                "IMAGES": images,
                "INTENT": intent,
            }

            gt_speed_labels = derive_gt_speed_change_labels(
                future=future,
                past=past,
                dt=args.dt,
                speed_delta_thresh=args.speed_delta_thresh,
            )
            accelerating_mask = gt_speed_labels == 1
            decelerating_mask = gt_speed_labels == 0

            base_out = model(model_input)
            if sae_kind == "control":
                sae_out = model(model_input, sae=sae, sae_target="control")
                if args.flip_behaviors:
                    raise ValueError("--flip_behaviors is only supported for planner-query-block SAE checkpoints.")
                latent_editor = build_control_latent_template_editor(
                    feature_indices=feature_indices_cpu.to(device=device),
                    template_values=template_values_cpu.to(device=device),
                    batch_size=batch_size,
                    n_proposals=model.n_proposals,
                    latent_dim=sae.encoder.out_features,
                )
                edited_out = model(model_input, sae=sae, sae_target="control", sae_latent_edit=latent_editor)
            else:
                if block_index is None:
                    raise RuntimeError("block_index must be resolved for planner-query-block SAE intervention.")
                if args.flip_behaviors:
                    latent_editor = build_block_flip_latent_editor(
                        accel_feature_indices=accel_feature_indices_cpu.to(device=device),
                        accel_template_values=accel_template_values_cpu.to(device=device),
                        decel_feature_indices=decel_feature_indices_cpu.to(device=device),
                        decel_template_values=decel_template_values_cpu.to(device=device),
                        accelerate_mask=accelerating_mask,
                        decelerate_mask=decelerating_mask,
                        batch_size=batch_size,
                        latent_dim=sae.latent_dim,
                    )
                else:
                    latent_editor = build_block_latent_template_editor(
                        feature_indices=feature_indices_cpu.to(device=device),
                        template_values=template_values_cpu.to(device=device),
                        batch_size=batch_size,
                        latent_dim=sae.latent_dim,
                    )
                sae_out = model(
                    model_input,
                    sae=sae,
                    sae_target="planner_query_block",
                    sae_block_index=block_index,
                )
                edited_out = model(
                    model_input,
                    sae=sae,
                    sae_target="planner_query_block",
                    sae_block_index=block_index,
                    sae_latent_edit=latent_editor,
                )

            base_best = gather_best_outputs(base_out, max_accel=model.max_accel, max_omega=model.max_omega)
            sae_best = gather_best_outputs(sae_out, max_accel=model.max_accel, max_omega=model.max_omega)
            edited_best = gather_best_outputs(edited_out, max_accel=model.max_accel, max_omega=model.max_omega)

            edit_shift = torch.linalg.norm(edited_best["traj"] - sae_best["traj"], dim=-1)
            base_shift = torch.linalg.norm(edited_best["traj"] - base_best["traj"], dim=-1)
            gt_shift = torch.linalg.norm(future - edited_best["traj"], dim=-1)
            gt_base = torch.linalg.norm(future - base_best["traj"], dim=-1)
            gt_sae = torch.linalg.norm(future - sae_best["traj"], dim=-1)

            edit_shift_ade = edit_shift.mean(dim=-1)
            edit_shift_fde = edit_shift[:, -1]
            base_shift_ade = base_shift.mean(dim=-1)
            base_shift_fde = base_shift[:, -1]
            base_ade = gt_base.mean(dim=-1)
            sae_ade = gt_sae.mean(dim=-1)
            edited_ade = gt_shift.mean(dim=-1)

            base_accel_mean = base_best["accel"].mean(dim=-1)
            sae_accel_mean = sae_best["accel"].mean(dim=-1)
            edited_accel_mean = edited_best["accel"].mean(dim=-1)
            edited_accel_std = edited_best["accel"].std(dim=-1, unbiased=False)
            if args.flip_behaviors:
                desired_applied_accel = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
                desired_applied_accel[decelerating_mask] = flip_applied_accel_mag
                desired_applied_accel[accelerating_mask] = -flip_applied_accel_mag
                max_abs_accel_error = (edited_best["accel"] - desired_applied_accel).abs().max(dim=-1).values
                mean_abs_accel_error = (edited_best["accel"] - desired_applied_accel).abs().mean(dim=-1)
            else:
                max_abs_accel_error = (edited_best["accel"] - applied_accel).abs().max(dim=-1).values
                mean_abs_accel_error = (edited_best["accel"] - applied_accel).abs().mean(dim=-1)

            solver_residual_raw = torch.full(
                (batch_size,),
                fill_value=template_solution["solver_residual_raw"],
                dtype=torch.float32,
            )
            omega_leakage_mean = torch.full(
                (batch_size,),
                fill_value=template_solution["omega_leakage_mean"],
                dtype=torch.float32,
            )
            selected_feature_count = torch.full(
                (batch_size,),
                fill_value=float(template_solution["selected_feature_count"]),
                dtype=torch.float32,
            )

            global_edit_shift_ade.append(edit_shift_ade.cpu())
            global_edit_shift_fde.append(edit_shift_fde.cpu())
            global_base_shift_ade.append(base_shift_ade.cpu())
            global_base_shift_fde.append(base_shift_fde.cpu())
            global_base_ade.append(base_ade.cpu())
            global_sae_ade.append(sae_ade.cpu())
            global_edited_ade.append(edited_ade.cpu())
            global_accel_error.append(max_abs_accel_error.cpu())
            global_mean_accel_error.append(mean_abs_accel_error.cpu())
            global_edited_accel_mean.append(edited_accel_mean.cpu())
            global_edited_accel_std.append(edited_accel_std.cpu())
            global_solver_residual_raw.append(solver_residual_raw)
            global_omega_leakage_mean.append(omega_leakage_mean)
            global_selected_feature_count.append(selected_feature_count)

            for sample_idx in range(batch_size):
                row = {
                    "name": batch["NAME"][sample_idx],
                    "sae_kind": sae_kind,
                    "sae_block_index": block_index if block_index is not None else -1,
                    "flip_behaviors": bool(args.flip_behaviors),
                    "gt_speed_label": int(gt_speed_labels[sample_idx].item()),
                    "accelerating_scene": bool(accelerating_mask[sample_idx].item()),
                    "decelerating_scene": bool(decelerating_mask[sample_idx].item()),
                    "target_accel_mps2": float(args.target_accel_mps2),
                    "desired_accel_mps2": (
                        -flip_applied_accel_mag
                        if bool(accelerating_mask[sample_idx].item())
                        else flip_applied_accel_mag
                        if bool(decelerating_mask[sample_idx].item())
                        else 0.0
                    )
                    if args.flip_behaviors
                    else float(args.target_accel_mps2),
                    "target_raw_accel": target_raw_accel,
                    "applied_accel_mps2": float(applied_accel_cpu.item()),
                    "accel_bank_sign": template_solution["bank_sign"],
                    "base_accel_mean": float(base_accel_mean[sample_idx].item()),
                    "sae_accel_mean": float(sae_accel_mean[sample_idx].item()),
                    "edited_accel_mean": float(edited_accel_mean[sample_idx].item()),
                    "edited_accel_std": float(edited_accel_std[sample_idx].item()),
                    "max_abs_accel_error": float(max_abs_accel_error[sample_idx].item()),
                    "mean_abs_accel_error": float(mean_abs_accel_error[sample_idx].item()),
                    "solver_residual_raw": template_solution["solver_residual_raw"],
                    "omega_leakage_mean": template_solution["omega_leakage_mean"],
                    "selected_feature_count": template_solution["selected_feature_count"],
                    "best_idx_base": int(base_best["best_idx"][sample_idx].item()),
                    "best_idx_sae": int(sae_best["best_idx"][sample_idx].item()),
                    "best_idx_edited": int(edited_best["best_idx"][sample_idx].item()),
                    "edit_shift_ade": float(edit_shift_ade[sample_idx].item()),
                    "edit_shift_fde": float(edit_shift_fde[sample_idx].item()),
                    "base_shift_ade": float(base_shift_ade[sample_idx].item()),
                    "base_shift_fde": float(base_shift_fde[sample_idx].item()),
                    "base_ade": float(base_ade[sample_idx].item()),
                    "sae_ade": float(sae_ade[sample_idx].item()),
                    "edited_ade": float(edited_ade[sample_idx].item()),
                    "ade_delta_vs_base": float((edited_ade[sample_idx] - base_ade[sample_idx]).item()),
                    "ade_delta_vs_sae": float((edited_ade[sample_idx] - sae_ade[sample_idx]).item()),
                }
                if bool(decelerating_mask[sample_idx].item()):
                    row["flip_success"] = float(edited_accel_mean[sample_idx].item()) > 0.0
                    row["flip_margin"] = float(edited_accel_mean[sample_idx].item())
                    row["flip_direction_rank"] = 2
                elif bool(accelerating_mask[sample_idx].item()):
                    row["flip_success"] = float(edited_accel_mean[sample_idx].item()) < 0.0
                    row["flip_margin"] = -float(edited_accel_mean[sample_idx].item())
                    row["flip_direction_rank"] = 1
                else:
                    row["flip_success"] = False
                    row["flip_margin"] = -float(abs(edited_accel_mean[sample_idx].item()))
                    row["flip_direction_rank"] = 0
                summary_rows.append(row)
                include_for_plots = (
                    bool(accelerating_mask[sample_idx].item()) or bool(decelerating_mask[sample_idx].item())
                ) if args.flip_behaviors else True
                if include_for_plots:
                    saved_examples.append(
                        {
                            "rank_metric": row["edit_shift_ade"],
                            "name": row["name"],
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "sae_kind": row["sae_kind"],
                        "sae_block_index": row["sae_block_index"],
                        "flip_behaviors": row["flip_behaviors"],
                        "gt_speed_label": row["gt_speed_label"],
                        "accelerating_scene": row["accelerating_scene"],
                        "decelerating_scene": row["decelerating_scene"],
                        "target_accel_mps2": row["target_accel_mps2"],
                        "desired_accel_mps2": row["desired_accel_mps2"],
                        "target_raw_accel": row["target_raw_accel"],
                        "applied_accel_mps2": row["applied_accel_mps2"],
                        "accel_bank_sign": row["accel_bank_sign"],
                        "base_accel_mean": row["base_accel_mean"],
                        "sae_accel_mean": row["sae_accel_mean"],
                        "edited_accel_mean": row["edited_accel_mean"],
                        "edited_accel_std": row["edited_accel_std"],
                        "max_abs_accel_error": row["max_abs_accel_error"],
                        "mean_abs_accel_error": row["mean_abs_accel_error"],
                        "solver_residual_raw": row["solver_residual_raw"],
                        "omega_leakage_mean": row["omega_leakage_mean"],
                        "selected_feature_count": row["selected_feature_count"],
                        "best_idx_base": row["best_idx_base"],
                        "best_idx_sae": row["best_idx_sae"],
                        "best_idx_edited": row["best_idx_edited"],
                        "flip_success": row["flip_success"],
                        "flip_margin": row["flip_margin"],
                        "flip_direction_rank": row["flip_direction_rank"],
                        "past": batch["PAST"][sample_idx].cpu(),
                        "future": batch["FUTURE"][sample_idx].cpu(),
                        "base_best": base_best["traj"][sample_idx].cpu(),
                        "sae_best": sae_best["traj"][sample_idx].cpu(),
                        "edited_best": edited_best["traj"][sample_idx].cpu(),
                        "base_best_control": base_best["control"][sample_idx].cpu(),
                        "sae_best_control": sae_best["control"][sample_idx].cpu(),
                        "edited_best_control": edited_best["control"][sample_idx].cpu(),
                        "base_best_accel": base_best["accel"][sample_idx].cpu(),
                        "sae_best_accel": sae_best["accel"][sample_idx].cpu(),
                        "edited_best_accel": edited_best["accel"][sample_idx].cpu(),
                        "base_best_omega": base_best["omega"][sample_idx].cpu(),
                        "sae_best_omega": sae_best["omega"][sample_idx].cpu(),
                        "edited_best_omega": edited_best["omega"][sample_idx].cpu(),
                        "edit_shift_ade": row["edit_shift_ade"],
                        "edit_shift_fde": row["edit_shift_fde"],
                        "base_ade": row["base_ade"],
                        "sae_ade": row["sae_ade"],
                        "edited_ade": row["edited_ade"],
                        }
                    )

    if not summary_rows:
        raise RuntimeError("No samples were processed, so no summary could be generated.")

    edit_shift_ade_all = torch.cat(global_edit_shift_ade, dim=0)
    edit_shift_fde_all = torch.cat(global_edit_shift_fde, dim=0)
    base_shift_ade_all = torch.cat(global_base_shift_ade, dim=0)
    base_shift_fde_all = torch.cat(global_base_shift_fde, dim=0)
    base_ade_all = torch.cat(global_base_ade, dim=0)
    sae_ade_all = torch.cat(global_sae_ade, dim=0)
    edited_ade_all = torch.cat(global_edited_ade, dim=0)
    accel_error_all = torch.cat(global_accel_error, dim=0)
    mean_accel_error_all = torch.cat(global_mean_accel_error, dim=0)
    edited_accel_mean_all = torch.cat(global_edited_accel_mean, dim=0)
    edited_accel_std_all = torch.cat(global_edited_accel_std, dim=0)
    solver_residual_raw_all = torch.cat(global_solver_residual_raw, dim=0)
    omega_leakage_mean_all = torch.cat(global_omega_leakage_mean, dim=0)
    selected_feature_count_all = torch.cat(global_selected_feature_count, dim=0)

    print(
        f"\nRequested target acceleration={args.target_accel_mps2:.6f} m/s^2 "
        f"applied={float(applied_accel_cpu.item()):.6f} m/s^2 "
        f"raw={target_raw_accel:.6f} kind={sae_kind} block={block_index if block_index is not None else 'control'}"
    )

    stats = {
        "max_abs_accel_error": summarise_metric(accel_error_all),
        "mean_abs_accel_error": summarise_metric(mean_accel_error_all),
        "edited_accel_mean": summarise_metric(edited_accel_mean_all),
        "edited_accel_std": summarise_metric(edited_accel_std_all),
        "solver_residual_raw": summarise_metric(solver_residual_raw_all),
        "omega_leakage_mean": summarise_metric(omega_leakage_mean_all),
        "selected_feature_count": summarise_metric(selected_feature_count_all),
        "edit_shift_ade": summarise_metric(edit_shift_ade_all),
        "edit_shift_fde": summarise_metric(edit_shift_fde_all),
        "base_shift_ade": summarise_metric(base_shift_ade_all),
        "base_shift_fde": summarise_metric(base_shift_fde_all),
        "base_ade": summarise_metric(base_ade_all),
        "sae_ade": summarise_metric(sae_ade_all),
        "edited_ade": summarise_metric(edited_ade_all),
        "ade_delta_vs_base": summarise_metric(edited_ade_all - base_ade_all),
        "ade_delta_vs_sae": summarise_metric(edited_ade_all - sae_ade_all),
    }
    for metric_name, metric_stats in stats.items():
        print(
            f"  {metric_name:<22} mean={metric_stats['mean']:.6f} "
            f"p50={metric_stats['p50']:.6f} p90={metric_stats['p90']:.6f}"
        )

    summary_csv = output_dir / "sample_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if args.flip_behaviors:
        saved_examples.sort(
            key=lambda item: (
                int(bool(item["flip_success"])),
                int(item["flip_direction_rank"]),
                float(item["flip_margin"]),
                float(item["rank_metric"]),
            ),
            reverse=True,
        )
    else:
        saved_examples.sort(key=lambda item: item["rank_metric"], reverse=True)
    saved_plot_count = save_example_plots(saved_examples, plot_dir, args.max_plots)
    accel_to_decel_count = 0
    decel_to_accel_count = 0
    if args.flip_behaviors:
        accel_to_decel_examples = [
            example
            for example in saved_examples
            if bool(example["accelerating_scene"]) and bool(example.get("flip_success", False))
        ]
        decel_to_accel_examples = [
            example
            for example in saved_examples
            if bool(example["decelerating_scene"]) and bool(example.get("flip_success", False))
        ]
        accel_to_decel_dir = output_dir / "plots_accelerating_to_decelerate"
        decel_to_accel_dir = output_dir / "plots_decelerating_to_accelerate"
        accel_to_decel_count = save_example_plots(accel_to_decel_examples, accel_to_decel_dir, args.max_plots)
        decel_to_accel_count = save_example_plots(decel_to_accel_examples, decel_to_accel_dir, args.max_plots)

    raw_out = output_dir / "intervention_results.pt"
    torch.save(
        {
            "args": vars(args),
            "sae_kind": sae_kind,
            "sae_block_index": block_index,
            "template_solution": template_solution,
            "summary_rows": summary_rows,
            "saved_examples": saved_examples[: args.max_plots],
        },
        raw_out,
    )

    print(f"\nSaved sample summary to {summary_csv}")
    print(f"Saved raw outputs to {raw_out}")
    print(f"Saved up to {saved_plot_count} plots to {plot_dir}")
    if args.flip_behaviors:
        print(
            f"Saved {accel_to_decel_count} accelerating->decelerate plots to "
            f"{output_dir / 'plots_accelerating_to_decelerate'}"
        )
        print(
            f"Saved {decel_to_accel_count} decelerating->accelerate plots to "
            f"{output_dir / 'plots_decelerating_to_accelerate'}"
        )
