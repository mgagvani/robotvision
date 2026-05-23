import argparse
import csv
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.initial_sae import SparseAutoEncoder, load_torch_file


MOTION_NAMES = {
    0: "BACKWARD",
    1: "STOPPING",
    2: "FORWARD",
}

SPEED_CHANGE_NAMES = {
    0: "DECELERATING",
    1: "ACCELERATING",
}

ACCEL_BANK_TOP_K_PER_SIGN = 8


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(num)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out


def class_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def load_sae_model(ckpt_path: Path, device: torch.device) -> SparseAutoEncoder:
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


def rollout_control(
    control_pred: torch.Tensor,
    past: torch.Tensor,
    dt: float,
    max_accel: float,
    max_omega: float,
) -> dict[str, torch.Tensor]:
    if control_pred.ndim != 4:
        raise ValueError(f"Expected control_pred shape (N, K, T, 2), got {tuple(control_pred.shape)}")
    if past.ndim != 3:
        raise ValueError(f"Expected past shape (N, 16, 6), got {tuple(past.shape)}")

    accel = torch.tanh(control_pred[..., 0]) * max_accel  # (N, K, T)
    omega = torch.tanh(control_pred[..., 1]) * max_omega  # (N, K, T)

    x_state = past[:, -1, 0].unsqueeze(1).expand(-1, control_pred.size(1)).clone()
    y_state = past[:, -1, 1].unsqueeze(1).expand(-1, control_pred.size(1)).clone()
    vx0 = past[:, -1, 2]
    vy0 = past[:, -1, 3]
    speed0 = torch.sqrt(vx0 * vx0 + vy0 * vy0 + 1e-6)
    speed_state = speed0.unsqueeze(1).expand(-1, control_pred.size(1)).clone()
    heading0 = torch.atan2(vy0, vx0)
    heading_state = heading0.unsqueeze(1).expand(-1, control_pred.size(1)).clone()

    for t in range(control_pred.size(2)):
        x_state = x_state + speed_state * torch.cos(heading_state) * dt
        y_state = y_state + speed_state * torch.sin(heading_state) * dt
        heading_state = heading_state + omega[:, :, t] * dt
        speed_state = torch.clamp_min(speed_state + accel[:, :, t] * dt, 0.0)

    final_xy = torch.stack([x_state, y_state], dim=-1)  # (N, K, 2)
    start_xy = past[:, -1, 0:2]
    displacement = final_xy - start_xy.unsqueeze(1)
    total_disp = torch.linalg.norm(displacement, dim=-1)
    heading_vec = torch.stack([torch.cos(heading0), torch.sin(heading0)], dim=-1).unsqueeze(1)
    longitudinal_disp = (displacement * heading_vec).sum(dim=-1)

    return {
        "initial_speed": speed0,
        "final_speed": speed_state,
        "delta_speed": speed_state - speed0.unsqueeze(1),
        "total_disp": total_disp,
        "longitudinal_disp": longitudinal_disp,
    }


def derive_motion_labels(
    control_pred: torch.Tensor,
    past: torch.Tensor,
    dt: float,
    max_accel: float,
    max_omega: float,
    stop_disp_thresh: float,
    backward_disp_thresh: float,
) -> torch.Tensor:
    motion = rollout_control(control_pred, past, dt, max_accel, max_omega)
    labels = torch.full(motion["total_disp"].shape, 1, dtype=torch.long)  # STOPPING

    stop_mask = motion["total_disp"] < stop_disp_thresh
    backward_mask = (~stop_mask) & (motion["longitudinal_disp"] < -backward_disp_thresh)
    forward_mask = (~stop_mask) & (~backward_mask)

    labels[backward_mask] = 0
    labels[forward_mask] = 2
    return labels.reshape(-1)


def derive_speed_change_labels(
    control_pred: torch.Tensor,
    past: torch.Tensor,
    dt: float,
    max_accel: float,
    max_omega: float,
    speed_delta_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    motion = rollout_control(control_pred, past, dt, max_accel, max_omega)
    labels = torch.full(motion["delta_speed"].shape, -1, dtype=torch.long)
    labels[motion["delta_speed"] <= -speed_delta_thresh] = 0
    labels[motion["delta_speed"] >= speed_delta_thresh] = 1
    valid_mask = labels >= 0
    return labels.reshape(-1), valid_mask.reshape(-1)


def compute_stats(
    model: SparseAutoEncoder,
    feature_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    dataset = TensorDataset(feature_tensor, label_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    latent_dim = model.encoder.out_features
    all_classes = sorted(torch.unique(label_tensor).tolist())

    total_count = 0
    total_sum = torch.zeros(latent_dim, dtype=torch.float64)
    total_sum_sq = torch.zeros(latent_dim, dtype=torch.float64)
    total_active = torch.zeros(latent_dim, dtype=torch.float64)

    class_counts = {cls: 0 for cls in all_classes}
    class_sums = {cls: torch.zeros(latent_dim, dtype=torch.float64) for cls in all_classes}
    class_active = {cls: torch.zeros(latent_dim, dtype=torch.float64) for cls in all_classes}

    with torch.no_grad():
        for batch_x, batch_label in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_label = batch_label.to(device, non_blocking=True)

            z = model(batch_x)["latents"]
            z_cpu = z.cpu().to(torch.float64)
            label_cpu = batch_label.cpu()

            total_count += z_cpu.shape[0]
            total_sum += z_cpu.sum(dim=0)
            total_sum_sq += (z_cpu * z_cpu).sum(dim=0)
            total_active += (z_cpu > 0).sum(dim=0)

            for cls in all_classes:
                mask = label_cpu == cls
                n = int(mask.sum().item())
                if n == 0:
                    continue
                selected = z_cpu[mask]
                class_counts[cls] += n
                class_sums[cls] += selected.sum(dim=0)
                class_active[cls] += (selected > 0).sum(dim=0)

    total_n = torch.tensor(float(total_count), dtype=torch.float64)
    mean_all = total_sum / total_n
    var_all = safe_div(total_sum_sq, total_n) - mean_all.square()
    var_all = torch.clamp(var_all, min=0.0)
    std_all = torch.sqrt(var_all)
    active_rate_all = total_active / total_n

    ss_total = total_sum_sq - total_n * mean_all.square()
    ss_between = torch.zeros_like(ss_total)
    mean_by_class = {}
    active_rate_by_class = {}
    point_biserial_r = {}

    for cls in all_classes:
        class_n = float(class_counts[cls])
        class_sum = class_sums[cls]
        class_active_sum = class_active[cls]
        class_n_tensor = torch.tensor(class_n, dtype=torch.float64)
        other_n = float(total_count - class_counts[cls])

        mean_cls = safe_div(class_sum, class_n_tensor)
        mean_by_class[cls] = mean_cls
        active_rate_by_class[cls] = safe_div(class_active_sum, class_n_tensor)
        ss_between += class_n_tensor * (mean_cls - mean_all).square()

        if class_n == 0 or other_n == 0:
            point_biserial_r[cls] = torch.zeros(latent_dim, dtype=torch.float64)
            continue

        other_mean = safe_div(total_sum - class_sum, torch.tensor(other_n, dtype=torch.float64))
        p = class_n / total_count
        q = 1.0 - p
        scale = math.sqrt(p * q)
        r = torch.zeros(latent_dim, dtype=torch.float64)
        denom_mask = std_all > 0
        r[denom_mask] = ((mean_cls[denom_mask] - other_mean[denom_mask]) / std_all[denom_mask]) * scale
        point_biserial_r[cls] = r

    eta_sq = torch.zeros_like(ss_total)
    valid_total = ss_total > 0
    eta_sq[valid_total] = ss_between[valid_total] / ss_total[valid_total]

    return {
        "all_classes": all_classes,
        "total_count": total_count,
        "mean_all": mean_all,
        "std_all": std_all,
        "active_rate_all": active_rate_all,
        "mean_by_class": mean_by_class,
        "active_rate_by_class": active_rate_by_class,
        "point_biserial_r": point_biserial_r,
        "eta_sq": eta_sq,
        "class_counts": class_counts,
    }


def write_csv(stats: dict, class_names: dict[int, str], output_csv: Path) -> None:
    all_classes = stats["all_classes"]
    eta_sq = stats["eta_sq"]

    rows = []
    for feature_idx in range(len(eta_sq)):
        row = {
            "feature_idx": feature_idx,
            "eta_sq": float(eta_sq[feature_idx].item()),
            "mean_all": float(stats["mean_all"][feature_idx].item()),
            "std_all": float(stats["std_all"][feature_idx].item()),
            "active_rate_all": float(stats["active_rate_all"][feature_idx].item()),
        }

        best_class = None
        best_abs_r = -1.0
        for cls in all_classes:
            class_name = class_names.get(cls, str(cls))
            slug = class_slug(class_name)
            r_val = float(stats["point_biserial_r"][cls][feature_idx].item())
            mean_val = float(stats["mean_by_class"][cls][feature_idx].item())
            active_rate_val = float(stats["active_rate_by_class"][cls][feature_idx].item())
            row[f"r_{slug}"] = r_val
            row[f"mean_{slug}"] = mean_val
            row[f"active_rate_{slug}"] = active_rate_val
            if abs(r_val) > best_abs_r:
                best_abs_r = abs(r_val)
                best_class = class_name

        row["best_abs_r"] = best_abs_r
        row["best_class"] = best_class
        rows.append(row)

    rows.sort(key=lambda item: item["eta_sq"], reverse=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(stats: dict, class_names: dict[int, str], top_k: int, title: str, dropped_count: int = 0) -> None:
    eta_sq = stats["eta_sq"]
    top_eta = torch.argsort(eta_sq, descending=True)[:top_k].tolist()

    print(f"{title} counts:")
    for cls in stats["all_classes"]:
        name = class_names.get(cls, str(cls))
        print(f"  {name}: {stats['class_counts'][cls]}")
    if dropped_count > 0:
        print(f"  DROPPED_NEUTRAL: {dropped_count}")

    print("")
    print(f"Top {top_k} SAE features by eta^2 for {title}:")
    for rank, feature_idx in enumerate(top_eta, start=1):
        eta_val = float(eta_sq[feature_idx].item())

        best_class = None
        best_r = None
        best_abs_r = -1.0
        per_class_bits = []
        for cls in stats["all_classes"]:
            name = class_names.get(cls, str(cls))
            r_val = float(stats["point_biserial_r"][cls][feature_idx].item())
            mean_val = float(stats["mean_by_class"][cls][feature_idx].item())
            active_rate_val = float(stats["active_rate_by_class"][cls][feature_idx].item())
            per_class_bits.append(f"{name}: r={r_val:+.4f}, mean={mean_val:.4f}, active={active_rate_val:.3f}")
            if abs(r_val) > best_abs_r:
                best_abs_r = abs(r_val)
                best_class = name
                best_r = r_val

        print(
            f"{rank}. feature={feature_idx} eta^2={eta_val:.5f} "
            f"best={best_class} r={best_r:+.4f} active_all={float(stats['active_rate_all'][feature_idx].item()):.3f}"
        )
        print("   " + " | ".join(per_class_bits))

    print("")
    for cls in stats["all_classes"]:
        name = class_names.get(cls, str(cls))
        abs_r = torch.abs(stats["point_biserial_r"][cls])
        top_feats = torch.argsort(abs_r, descending=True)[:top_k].tolist()
        print(f"Top {top_k} features for {name} by |r|:")
        for rank, feature_idx in enumerate(top_feats, start=1):
            r_val = float(stats["point_biserial_r"][cls][feature_idx].item())
            eta_val = float(stats["eta_sq"][feature_idx].item())
            mean_val = float(stats["mean_by_class"][cls][feature_idx].item())
            active_rate_val = float(stats["active_rate_by_class"][cls][feature_idx].item())
            print(
                f"  {rank}. feature={feature_idx} r={r_val:+.4f} "
                f"eta^2={eta_val:.5f} mean={mean_val:.4f} active={active_rate_val:.3f}"
            )
        print("")


def compute_accel_targets(feature_tensor: torch.Tensor, max_accel: float) -> dict[str, torch.Tensor]:
    raw_accel = feature_tensor[:, 0::2]
    accel_mps2 = torch.tanh(raw_accel) * max_accel
    return {
        "raw_accel": raw_accel,
        "mean_raw_accel": raw_accel.mean(dim=-1),
        "mean_accel_mps2": accel_mps2.mean(dim=-1),
        "accel_std": accel_mps2.std(dim=-1, unbiased=False),
    }


def compute_accel_continuous_stats(
    model: SparseAutoEncoder,
    feature_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
    max_accel: float,
) -> dict[str, torch.Tensor]:
    accel_targets = compute_accel_targets(feature_tensor, max_accel=max_accel)
    mean_raw_accel = accel_targets["mean_raw_accel"].to(torch.float64)

    latent_dim = model.encoder.out_features
    total_count = 0
    total_sum = torch.zeros(latent_dim, dtype=torch.float64)
    total_sum_sq = torch.zeros(latent_dim, dtype=torch.float64)
    total_active = torch.zeros(latent_dim, dtype=torch.float64)
    total_sum_xy = torch.zeros(latent_dim, dtype=torch.float64)
    total_sum_x = torch.tensor(0.0, dtype=torch.float64)
    total_sum_x_sq = torch.tensor(0.0, dtype=torch.float64)

    with torch.no_grad():
        for start in range(0, feature_tensor.shape[0], batch_size):
            end = min(start + batch_size, feature_tensor.shape[0])
            batch_x = feature_tensor[start:end].to(device, non_blocking=True)
            batch_target = mean_raw_accel[start:end]

            latents = model(batch_x)["latents"].cpu().to(torch.float64)
            total_count += latents.shape[0]
            total_sum += latents.sum(dim=0)
            total_sum_sq += (latents * latents).sum(dim=0)
            total_active += (latents > 0).sum(dim=0)
            total_sum_xy += (latents * batch_target.unsqueeze(1)).sum(dim=0)
            total_sum_x += batch_target.sum()
            total_sum_x_sq += (batch_target * batch_target).sum()

    total_n = torch.tensor(float(total_count), dtype=torch.float64)
    mean_latent = total_sum / total_n
    var_latent = safe_div(total_sum_sq, total_n) - mean_latent.square()
    var_latent = torch.clamp(var_latent, min=0.0)
    std_latent = torch.sqrt(var_latent)
    active_rate_all = total_active / total_n

    mean_target = total_sum_x / total_n
    var_target = torch.clamp(total_sum_x_sq / total_n - mean_target.square(), min=0.0)
    cov = total_sum_xy / total_n - mean_latent * mean_target

    corr = torch.zeros_like(cov)
    if float(var_target.item()) > 0.0:
        denom_mask = std_latent > 0
        corr[denom_mask] = cov[denom_mask] / (std_latent[denom_mask] * math.sqrt(float(var_target.item())))

    slope = torch.zeros_like(cov)
    slope_mask = var_latent > 0
    slope[slope_mask] = cov[slope_mask] / var_latent[slope_mask]

    return {
        "corr_mean_raw": corr,
        "slope_mean_raw": slope,
        "active_rate_all": active_rate_all,
    }


def compute_decoder_accel_stats(model: SparseAutoEncoder) -> dict[str, torch.Tensor]:
    decoder = model.decoder.weight.detach().cpu().to(torch.float64)
    accel_matrix = decoder[0::2].contiguous()
    omega_matrix = decoder[1::2].contiguous()
    return {
        "decoder_accel_matrix": accel_matrix,
        "decoder_omega_matrix": omega_matrix,
        "decoder_accel_mean": accel_matrix.mean(dim=0),
        "decoder_accel_std": accel_matrix.std(dim=0, unbiased=False),
        "decoder_omega_norm": omega_matrix.norm(dim=0),
    }


def score_accel_features_for_sign(
    sign: int,
    corr_mean_raw: torch.Tensor,
    slope_mean_raw: torch.Tensor,
    decoder_accel_mean: torch.Tensor,
    decoder_accel_std: torch.Tensor,
    decoder_omega_norm: torch.Tensor,
    active_rate_all: torch.Tensor,
) -> torch.Tensor:
    sign_float = float(sign)
    signed_corr = torch.clamp_min(corr_mean_raw * sign_float, 0.0)
    signed_slope = torch.clamp_min(slope_mean_raw * sign_float, 0.0)
    signed_decoder = torch.clamp_min(decoder_accel_mean * sign_float, 0.0)
    denominator = 1.0 + decoder_accel_std + decoder_omega_norm
    return (2.0 * signed_corr + signed_slope + signed_decoder + 0.25 * active_rate_all) / denominator


def select_accel_feature_banks(
    corr_mean_raw: torch.Tensor,
    slope_mean_raw: torch.Tensor,
    decoder_accel_mean: torch.Tensor,
    decoder_accel_std: torch.Tensor,
    decoder_omega_norm: torch.Tensor,
    active_rate_all: torch.Tensor,
    decoder_accel_matrix: torch.Tensor,
    decoder_omega_matrix: torch.Tensor,
    top_k_per_sign: int = ACCEL_BANK_TOP_K_PER_SIGN,
) -> dict[str, object]:
    score_positive = score_accel_features_for_sign(
        sign=1,
        corr_mean_raw=corr_mean_raw,
        slope_mean_raw=slope_mean_raw,
        decoder_accel_mean=decoder_accel_mean,
        decoder_accel_std=decoder_accel_std,
        decoder_omega_norm=decoder_omega_norm,
        active_rate_all=active_rate_all,
    )
    score_negative = score_accel_features_for_sign(
        sign=-1,
        corr_mean_raw=corr_mean_raw,
        slope_mean_raw=slope_mean_raw,
        decoder_accel_mean=decoder_accel_mean,
        decoder_accel_std=decoder_accel_std,
        decoder_omega_norm=decoder_omega_norm,
        active_rate_all=active_rate_all,
    )
    feature_indices = torch.arange(corr_mean_raw.shape[0], dtype=torch.long)

    def build_bank(sign_name: str, sign: int, score: torch.Tensor) -> dict[str, object]:
        candidate_mask = (decoder_accel_mean * float(sign)) > 0
        ranked = torch.argsort(score, descending=True)
        selected = [int(idx) for idx in ranked.tolist() if bool(candidate_mask[idx].item())]
        if not selected:
            selected = ranked.tolist()
        selected = selected[:top_k_per_sign]
        idx_tensor = torch.tensor(selected, dtype=torch.long)
        return {
            "sign": sign_name,
            "feature_indices": selected,
            "scores": score[idx_tensor].to(torch.float32),
            "corr_mean_raw": corr_mean_raw[idx_tensor].to(torch.float32),
            "slope_mean_raw": slope_mean_raw[idx_tensor].to(torch.float32),
            "decoder_accel_mean": decoder_accel_mean[idx_tensor].to(torch.float32),
            "decoder_accel_std": decoder_accel_std[idx_tensor].to(torch.float32),
            "decoder_omega_norm": decoder_omega_norm[idx_tensor].to(torch.float32),
            "active_rate_all": active_rate_all[idx_tensor].to(torch.float32),
            "decoder_accel_submatrix": decoder_accel_matrix[:, idx_tensor].to(torch.float32),
            "decoder_omega_submatrix": decoder_omega_matrix[:, idx_tensor].to(torch.float32),
        }

    return {
        "feature_indices": feature_indices,
        "score_positive": score_positive,
        "score_negative": score_negative,
        "positive": build_bank("positive", 1, score_positive),
        "negative": build_bank("negative", -1, score_negative),
    }


def build_accel_analysis_artifact(
    model: SparseAutoEncoder,
    feature_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
    max_accel: float,
    top_k_per_sign: int = ACCEL_BANK_TOP_K_PER_SIGN,
) -> dict[str, object]:
    continuous_stats = compute_accel_continuous_stats(
        model=model,
        feature_tensor=feature_tensor,
        batch_size=batch_size,
        device=device,
        max_accel=max_accel,
    )
    decoder_stats = compute_decoder_accel_stats(model)
    bank_selection = select_accel_feature_banks(
        corr_mean_raw=continuous_stats["corr_mean_raw"],
        slope_mean_raw=continuous_stats["slope_mean_raw"],
        decoder_accel_mean=decoder_stats["decoder_accel_mean"],
        decoder_accel_std=decoder_stats["decoder_accel_std"],
        decoder_omega_norm=decoder_stats["decoder_omega_norm"],
        active_rate_all=continuous_stats["active_rate_all"],
        decoder_accel_matrix=decoder_stats["decoder_accel_matrix"],
        decoder_omega_matrix=decoder_stats["decoder_omega_matrix"],
        top_k_per_sign=top_k_per_sign,
    )
    return {
        **continuous_stats,
        **decoder_stats,
        **bank_selection,
        "top_k_per_sign": top_k_per_sign,
        "horizon": int(decoder_stats["decoder_accel_matrix"].shape[0]),
    }


def write_accel_csv(accel_artifact: dict[str, object], output_csv: Path) -> None:
    corr_mean_raw = accel_artifact["corr_mean_raw"]
    slope_mean_raw = accel_artifact["slope_mean_raw"]
    decoder_accel_mean = accel_artifact["decoder_accel_mean"]
    decoder_accel_std = accel_artifact["decoder_accel_std"]
    decoder_omega_norm = accel_artifact["decoder_omega_norm"]
    active_rate_all = accel_artifact["active_rate_all"]
    score_positive = accel_artifact["score_positive"]
    score_negative = accel_artifact["score_negative"]
    positive_selected = set(accel_artifact["positive"]["feature_indices"])
    negative_selected = set(accel_artifact["negative"]["feature_indices"])

    rows = []
    for feature_idx in range(len(corr_mean_raw)):
        pos_score = float(score_positive[feature_idx].item())
        neg_score = float(score_negative[feature_idx].item())
        row = {
            "feature_idx": feature_idx,
            "corr_mean_raw": float(corr_mean_raw[feature_idx].item()),
            "slope_mean_raw": float(slope_mean_raw[feature_idx].item()),
            "decoder_accel_mean": float(decoder_accel_mean[feature_idx].item()),
            "decoder_accel_std": float(decoder_accel_std[feature_idx].item()),
            "decoder_omega_norm": float(decoder_omega_norm[feature_idx].item()),
            "active_rate_all": float(active_rate_all[feature_idx].item()),
            "bank_sign": "positive" if pos_score >= neg_score else "negative",
            "score_positive": pos_score,
            "score_negative": neg_score,
            "selected_positive": feature_idx in positive_selected,
            "selected_negative": feature_idx in negative_selected,
        }
        rows.append(row)

    rows.sort(key=lambda row: max(row["score_positive"], row["score_negative"]), reverse=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_accel_bank(accel_artifact: dict[str, object], output_pt: Path) -> None:
    def serialise_bank(bank: dict[str, object]) -> dict[str, object]:
        return {
            "sign": bank["sign"],
            "feature_indices": list(bank["feature_indices"]),
            "scores": bank["scores"].clone(),
            "corr_mean_raw": bank["corr_mean_raw"].clone(),
            "slope_mean_raw": bank["slope_mean_raw"].clone(),
            "decoder_accel_mean": bank["decoder_accel_mean"].clone(),
            "decoder_accel_std": bank["decoder_accel_std"].clone(),
            "decoder_omega_norm": bank["decoder_omega_norm"].clone(),
            "active_rate_all": bank["active_rate_all"].clone(),
            "decoder_accel_submatrix": bank["decoder_accel_submatrix"].clone(),
            "decoder_omega_submatrix": bank["decoder_omega_submatrix"].clone(),
        }

    payload = {
        "top_k_per_sign": int(accel_artifact["top_k_per_sign"]),
        "horizon": int(accel_artifact["horizon"]),
        "positive": serialise_bank(accel_artifact["positive"]),
        "negative": serialise_bank(accel_artifact["negative"]),
    }
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt)


def print_accel_summary(accel_artifact: dict[str, object], top_k: int) -> None:
    print("Acceleration bank summary:")
    for sign_name in ("positive", "negative"):
        bank = accel_artifact[sign_name]
        print(f"Top {min(top_k, len(bank['feature_indices']))} features for {sign_name} acceleration:")
        for rank, feature_idx in enumerate(bank["feature_indices"][:top_k], start=1):
            idx = int(feature_idx)
            print(
                f"  {rank}. feature={idx} score={float(accel_artifact[f'score_{sign_name}'][idx].item()):.5f} "
                f"corr={float(accel_artifact['corr_mean_raw'][idx].item()):+.4f} "
                f"slope={float(accel_artifact['slope_mean_raw'][idx].item()):+.4f} "
                f"dec_mean={float(accel_artifact['decoder_accel_mean'][idx].item()):+.4f} "
                f"dec_std={float(accel_artifact['decoder_accel_std'][idx].item()):.4f} "
                f"omega_norm={float(accel_artifact['decoder_omega_norm'][idx].item()):.4f}"
            )
        print("")


def build_feature_and_labels(
    data_path: Path,
    max_samples: Optional[int],
    dt: float,
    max_accel: float,
    max_omega: float,
    stop_disp_thresh: float,
    backward_disp_thresh: float,
    speed_delta_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    blob = load_torch_file(str(data_path), map_location="cpu")
    if not isinstance(blob, dict):
        raise TypeError(f"Expected extracted activation dict, got {type(blob).__name__}")
    if "control_pred" not in blob or "past" not in blob:
        raise KeyError("Expected keys 'control_pred' and 'past' in extracted control file")

    control_pred = blob["control_pred"].float()
    past = blob["past"].float()
    if max_samples is not None:
        control_pred = control_pred[:max_samples]
        past = past[:max_samples]

    feature_tensor = control_pred.reshape(control_pred.shape[0] * control_pred.shape[1], -1).contiguous()
    motion_labels = derive_motion_labels(
        control_pred=control_pred,
        past=past,
        dt=dt,
        max_accel=max_accel,
        max_omega=max_omega,
        stop_disp_thresh=stop_disp_thresh,
        backward_disp_thresh=backward_disp_thresh,
    )
    speed_change_labels, speed_change_valid = derive_speed_change_labels(
        control_pred=control_pred,
        past=past,
        dt=dt,
        max_accel=max_accel,
        max_omega=max_omega,
        speed_delta_thresh=speed_delta_thresh,
    )

    meta = blob.get("meta", {})
    return feature_tensor, motion_labels, speed_change_labels, speed_change_valid, meta


def run_control_sae_analysis(
    data_path: Path,
    sae_ckpt: Path,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    top_k: int,
    max_samples: Optional[int],
    dt: float,
    max_accel: float,
    max_omega: float,
    stop_disp_thresh: float,
    backward_disp_thresh: float,
    speed_delta_thresh: float,
) -> dict[str, Path]:
    data_path = Path(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_sae_model(Path(sae_ckpt), device)
    feature_tensor, motion_labels, speed_change_labels, speed_change_valid, meta = build_feature_and_labels(
        data_path=data_path,
        max_samples=max_samples,
        dt=dt,
        max_accel=max_accel,
        max_omega=max_omega,
        stop_disp_thresh=stop_disp_thresh,
        backward_disp_thresh=backward_disp_thresh,
        speed_delta_thresh=speed_delta_thresh,
    )

    motion_stats = compute_stats(
        model=model,
        feature_tensor=feature_tensor,
        label_tensor=motion_labels,
        batch_size=batch_size,
        device=device,
    )
    motion_csv = output_dir / "sae_control_motion.csv"
    write_csv(motion_stats, MOTION_NAMES, motion_csv)
    print_summary(motion_stats, MOTION_NAMES, top_k=top_k, title="Motion state")
    print(f"Saved motion CSV to {motion_csv}")

    valid_feature_tensor = feature_tensor[speed_change_valid]
    valid_speed_change_labels = speed_change_labels[speed_change_valid]
    speed_stats = compute_stats(
        model=model,
        feature_tensor=valid_feature_tensor,
        label_tensor=valid_speed_change_labels,
        batch_size=batch_size,
        device=device,
    )
    speed_csv = output_dir / "sae_control_speed_change.csv"
    write_csv(speed_stats, SPEED_CHANGE_NAMES, speed_csv)
    print_summary(
        speed_stats,
        SPEED_CHANGE_NAMES,
        top_k=top_k,
        title="Speed change",
        dropped_count=int((~speed_change_valid).sum().item()),
    )
    print(f"Saved speed-change CSV to {speed_csv}")

    accel_artifact = build_accel_analysis_artifact(
        model=model,
        feature_tensor=feature_tensor,
        batch_size=batch_size,
        device=device,
        max_accel=max_accel,
        top_k_per_sign=ACCEL_BANK_TOP_K_PER_SIGN,
    )
    accel_csv = output_dir / "sae_control_accel.csv"
    write_accel_csv(accel_artifact, accel_csv)
    accel_bank_pt = output_dir / "sae_control_accel_bank.pt"
    save_accel_bank(accel_artifact, accel_bank_pt)
    print_accel_summary(accel_artifact, top_k=min(top_k, ACCEL_BANK_TOP_K_PER_SIGN))
    print(f"Saved accel CSV to {accel_csv}")
    print(f"Saved accel bank to {accel_bank_pt}")

    if meta:
        print("")
        print("Source metadata:")
        for key, value in meta.items():
            print(f"  {key}: {value}")

    return {
        "motion_csv": motion_csv,
        "speed_csv": speed_csv,
        "accel_csv": accel_csv,
        "accel_bank_pt": accel_bank_pt,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/output/extract_train_control.pt",
        help="path to extracted control activations (.pt with control_pred and past)",
    )
    parser.add_argument(
        "--sae_ckpt",
        type=str,
        default="/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/output/sae_control_pred.pt",
        help="path to control SAE checkpoint",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--max_samples", type=int, default=None, help="limit to the first N scenes before flattening K proposals")
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--max_accel", type=float, default=8.0)
    parser.add_argument("--max_omega", type=float, default=1.0)
    parser.add_argument("--stop_disp_thresh", type=float, default=1.0, help="meters over the full horizon to call a proposal stopping")
    parser.add_argument("--backward_disp_thresh", type=float, default=0.5, help="signed longitudinal meters to call a proposal backward")
    parser.add_argument("--speed_delta_thresh", type=float, default=0.5, help="m/s change over the horizon to call accel/decel")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir) if args.output_dir else data_path.parent / "analysis_control"
    device = torch.device(args.device)
    run_control_sae_analysis(
        data_path=data_path,
        sae_ckpt=Path(args.sae_ckpt),
        output_dir=output_dir,
        batch_size=args.batch_size,
        device=device,
        top_k=args.top_k,
        max_samples=args.max_samples,
        dt=args.dt,
        max_accel=args.max_accel,
        max_omega=args.max_omega,
        stop_disp_thresh=args.stop_disp_thresh,
        backward_disp_thresh=args.backward_disp_thresh,
        speed_delta_thresh=args.speed_delta_thresh,
    )
