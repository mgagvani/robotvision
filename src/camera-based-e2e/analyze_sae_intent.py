import argparse
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.sae import SparseAutoencoder


INTENT_NAMES = {
    0: "UNKNOWN",
    1: "GO_STRAIGHT",
    2: "GO_LEFT",
    3: "GO_RIGHT",
}


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(num)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_stats(
    model: SparseAutoencoder,
    token_tensor: torch.Tensor,
    intent_tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    dataset = TensorDataset(token_tensor, intent_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    latent_dim = model.encoder.out_features
    all_intents = sorted(torch.unique(intent_tensor).tolist())

    total_count = 0
    total_sum = torch.zeros(latent_dim, dtype=torch.float64)
    total_sum_sq = torch.zeros(latent_dim, dtype=torch.float64)
    total_active = torch.zeros(latent_dim, dtype=torch.float64)

    class_counts = {intent: 0 for intent in all_intents}
    class_sums = {intent: torch.zeros(latent_dim, dtype=torch.float64) for intent in all_intents}
    class_active = {intent: torch.zeros(latent_dim, dtype=torch.float64) for intent in all_intents}

    model.eval()
    with torch.no_grad():
        for batch_x, batch_intent in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_intent = batch_intent.to(device, non_blocking=True)

            z = model.encode(normalize(batch_x, mean, std))
            z_cpu = z.cpu().to(torch.float64)
            intent_cpu = batch_intent.cpu()

            total_count += z_cpu.shape[0]
            total_sum += z_cpu.sum(dim=0)
            total_sum_sq += (z_cpu * z_cpu).sum(dim=0)
            total_active += (z_cpu > 0).sum(dim=0)

            for intent in all_intents:
                mask = intent_cpu == intent
                n = int(mask.sum().item())
                if n == 0:
                    continue
                selected = z_cpu[mask]
                class_counts[intent] += n
                class_sums[intent] += selected.sum(dim=0)
                class_active[intent] += (selected > 0).sum(dim=0)

    total_n = torch.tensor(float(total_count), dtype=torch.float64)
    mean_all = total_sum / total_n
    var_all = safe_div(total_sum_sq, total_n) - mean_all.square()
    var_all = torch.clamp(var_all, min=0.0)
    std_all = torch.sqrt(var_all)
    active_rate_all = total_active / total_n

    ss_total = total_sum_sq - total_n * mean_all.square()
    ss_between = torch.zeros_like(ss_total)
    mean_by_intent = {}
    active_rate_by_intent = {}
    point_biserial_r = {}

    for intent in all_intents:
        class_n = float(class_counts[intent])
        class_sum = class_sums[intent]
        class_active_sum = class_active[intent]
        class_n_tensor = torch.tensor(class_n, dtype=torch.float64)
        other_n = float(total_count - class_counts[intent])

        mean_intent = safe_div(class_sum, class_n_tensor)
        mean_by_intent[intent] = mean_intent
        active_rate_by_intent[intent] = safe_div(class_active_sum, class_n_tensor)

        ss_between += class_n_tensor * (mean_intent - mean_all).square()

        if class_n == 0 or other_n == 0:
            point_biserial_r[intent] = torch.zeros(latent_dim, dtype=torch.float64)
            continue

        other_mean = safe_div(total_sum - class_sum, torch.tensor(other_n, dtype=torch.float64))
        p = class_n / total_count
        q = 1.0 - p
        scale = math.sqrt(p * q)
        r = torch.zeros(latent_dim, dtype=torch.float64)
        denom_mask = std_all > 0
        r[denom_mask] = ((mean_intent[denom_mask] - other_mean[denom_mask]) / std_all[denom_mask]) * scale
        point_biserial_r[intent] = r

    eta_sq = torch.zeros_like(ss_total)
    valid_total = ss_total > 0
    eta_sq[valid_total] = ss_between[valid_total] / ss_total[valid_total]

    return {
        "all_intents": all_intents,
        "total_count": total_count,
        "mean_all": mean_all,
        "std_all": std_all,
        "active_rate_all": active_rate_all,
        "mean_by_intent": mean_by_intent,
        "active_rate_by_intent": active_rate_by_intent,
        "point_biserial_r": point_biserial_r,
        "eta_sq": eta_sq,
        "class_counts": class_counts,
    }


def write_csv(stats: dict, output_csv: Path) -> None:
    all_intents = stats["all_intents"]
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

        best_intent = None
        best_abs_r = -1.0
        for intent in all_intents:
            intent_name = INTENT_NAMES.get(intent, str(intent))
            r_val = float(stats["point_biserial_r"][intent][feature_idx].item())
            mean_val = float(stats["mean_by_intent"][intent][feature_idx].item())
            active_rate_val = float(stats["active_rate_by_intent"][intent][feature_idx].item())
            row[f"r_{intent_name}"] = r_val
            row[f"mean_{intent_name}"] = mean_val
            row[f"active_rate_{intent_name}"] = active_rate_val
            if abs(r_val) > best_abs_r:
                best_abs_r = abs(r_val)
                best_intent = intent_name

        row["best_abs_r"] = best_abs_r
        row["best_intent"] = best_intent
        rows.append(row)

    rows.sort(key=lambda item: item["eta_sq"], reverse=True)
    fieldnames = list(rows[0].keys())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(stats: dict, top_k: int) -> None:
    eta_sq = stats["eta_sq"]
    top_eta = torch.argsort(eta_sq, descending=True)[:top_k].tolist()

    print("Intent counts:")
    for intent in stats["all_intents"]:
        name = INTENT_NAMES.get(intent, str(intent))
        print(f"  {name}: {stats['class_counts'][intent]}")

    print("")
    print(f"Top {top_k} SAE features by eta^2:")
    for rank, feature_idx in enumerate(top_eta, start=1):
        eta_val = float(eta_sq[feature_idx].item())

        best_intent = None
        best_r = None
        best_abs_r = -1.0
        per_intent_bits = []
        for intent in stats["all_intents"]:
            name = INTENT_NAMES.get(intent, str(intent))
            r_val = float(stats["point_biserial_r"][intent][feature_idx].item())
            mean_val = float(stats["mean_by_intent"][intent][feature_idx].item())
            active_rate_val = float(stats["active_rate_by_intent"][intent][feature_idx].item())
            per_intent_bits.append(
                f"{name}: r={r_val:+.4f}, mean={mean_val:.4f}, active={active_rate_val:.3f}"
            )
            if abs(r_val) > best_abs_r:
                best_abs_r = abs(r_val)
                best_intent = name
                best_r = r_val

        print(
            f"{rank}. feature={feature_idx} eta^2={eta_val:.5f} "
            f"best={best_intent} r={best_r:+.4f} active_all={float(stats['active_rate_all'][feature_idx].item()):.3f}"
        )
        print("   " + " | ".join(per_intent_bits))

    print("")
    for intent in stats["all_intents"]:
        name = INTENT_NAMES.get(intent, str(intent))
        abs_r = torch.abs(stats["point_biserial_r"][intent])
        top_feats = torch.argsort(abs_r, descending=True)[:top_k].tolist()
        print(f"Top {top_k} features for {name} by |r|:")
        for rank, feature_idx in enumerate(top_feats, start=1):
            r_val = float(stats["point_biserial_r"][intent][feature_idx].item())
            eta_val = float(stats["eta_sq"][feature_idx].item())
            mean_val = float(stats["mean_by_intent"][intent][feature_idx].item())
            active_rate_val = float(stats["active_rate_by_intent"][intent][feature_idx].item())
            print(
                f"  {rank}. feature={feature_idx} r={r_val:+.4f} "
                f"eta^2={eta_val:.5f} mean={mean_val:.4f} active={active_rate_val:.3f}"
            )
        print("")


def infer_paths(run_root: Path, split: str) -> tuple[Path, Path, Path]:
    ckpt_path = run_root / "model" / "sae_checkpoint.pt"
    norm_path = run_root / "model" / "sae_normalization.pt"
    token_path = run_root / "tokens" / f"planner_tokens_{split}.pt"
    return ckpt_path, norm_path, token_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path, norm_path, token_path = infer_paths(run_root, args.split)
    device = torch.device(args.device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    norm = torch.load(norm_path, map_location="cpu")
    token_blob = torch.load(token_path, map_location="cpu")

    model = SparseAutoencoder(
        input_dim=ckpt["input_dim"],
        latent_dim=ckpt["latent_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    token_tensor = token_blob["planner_query_tok"].float()
    intent_tensor = token_blob["intent"].long()
    mean = norm["mean"].to(device)
    std = norm["std"].to(device)

    stats = compute_stats(
        model=model,
        token_tensor=token_tensor,
        intent_tensor=intent_tensor,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        device=device,
    )

    output_csv = output_dir / f"sae_intent_correlation_{args.split}.csv"
    write_csv(stats, output_csv)
    print_summary(stats, top_k=args.top_k)
    print(f"Saved CSV to {output_csv}")
