"""Train and evaluate one ladder cell: (condition, encoder_draw, adapter_seed).

Architecture, losses and optimizer come from `sae_scorer_correction.py` (the paper's
adapter pipeline) unchanged. The only substitution is the representation-tower input,
supplied by `adapter_ladder.data` (spec section 1).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from sae_scorer_correction import (
    SAEScorerResidual,
    correction_loss,
    pairwise_ranking_loss,
    proposal_weights_from_ade,
    selection_metrics,
)

from .data import (
    build_representation,
    load_slim_cache,
    proposal_fingerprint,
    topk_relu,
    validate_sae_checkpoint,
    validate_slim_cache,
)


class PaperScorerResidual(nn.Module):
    """Supplement Section 5 three-tower residual adapter.

    Each of the representation, trajectory and scalar-score inputs is embedded to the
    same hidden width.  Their concatenation passes through a two-linear-layer MLP head.
    This intentionally lives beside, rather than replacing, PR #26's legacy adapter so
    that the two recipes cannot be confused in saved artifacts.
    """

    def __init__(self, rep_dim: int, traj_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.rep_tower = nn.Sequential(
            nn.LayerNorm(rep_dim), nn.Linear(rep_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.traj_tower = nn.Sequential(
            nn.LayerNorm(traj_dim), nn.Linear(traj_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.score_tower = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rep: torch.Tensor, trajectory: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        if score.ndim == 1:
            score = score.unsqueeze(-1)
        fused = torch.cat(
            (self.rep_tower(rep), self.traj_tower(trajectory), self.score_tower(score)), dim=-1
        )
        return self.head(fused).squeeze(-1)


def residual_loss(pred, target, loss_type: str, weights):
    """Weighted residual loss.

    The pipeline's `correction_loss` supports huber (smooth L1) and mse; the ladder spec
    writes "weighted L1", so plain L1 is added here rather than editing the shared
    pipeline file. Default stays huber so the ladder reuses the paper's config verbatim.
    """
    if loss_type == "l1":
        loss = torch.abs(pred - target) * weights
        return loss.sum() / weights.sum().clamp_min(1e-8)
    return correction_loss(pred, target, loss_type=loss_type, weights=weights)


def evaluate(model, rep, cache, device, *, encoder=None, chunk=2048, use_bf16=False) -> dict:
    model.eval()
    picked, base_picked, ranks, base_ranks, n = [], [], [], [], cache["h"].size(0)
    with torch.inference_mode():
        for s in range(0, n, chunk):
            h = cache["h"][s : s + chunk]
            tr = cache["traj"][s : s + chunk]
            sc = cache["scores"][s : s + chunk]
            ade = cache["ade"][s : s + chunk]
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=use_bf16 and device.type == "cuda",
            ):
                x = encoder(h) if encoder is not None else rep(h)
                b, k, _ = tr.shape
                res = model(
                    x[:, None, :].expand(b, k, x.size(-1)).reshape(b * k, -1),
                    tr.reshape(b * k, -1),
                    sc.reshape(b * k),
                ).view(b, k)
            corrected = sc + res
            idx = torch.arange(b, device=device)
            picked.append(ade[idx, corrected.argmin(1)])
            base_picked.append(ade[idx, sc.argmin(1)])
            oracle_rank = ade.argsort(1).argsort(1)
            ranks.append(oracle_rank[idx, corrected.argmin(1)].float())
            base_ranks.append(oracle_rank[idx, sc.argmin(1)].float())
    model.train()
    picked = torch.cat(picked)
    base_picked = torch.cat(base_picked)
    return {
        "per_scene_ade": picked.cpu(),
        "per_scene_ade_base": base_picked.cpu(),
        "per_scene_rank": torch.cat(ranks).cpu(),
        "per_scene_rank_base": torch.cat(base_ranks).cpu(),
        "ade": float(picked.mean()),
        "ade_base": float(base_picked.mean()),
        "rel_improvement_pct": float((base_picked.mean() - picked.mean()) / base_picked.mean() * 100.0),
        "mean_rank": float(torch.cat(ranks).mean()),
        "mean_rank_base": float(torch.cat(base_ranks).mean()),
    }


class LearnedEncoder(torch.nn.Module):
    """C6: 384->384 + TopK(k), trained jointly on the residual loss.

    TopK routes gradients through the active units only, exactly as in SAE training, so
    no straight-through estimator is needed (spec section 3).
    """

    def __init__(self, prep, init_weight: torch.Tensor, k: int) -> None:
        super().__init__()
        self.prep = prep
        self.k = k
        self.weight = torch.nn.Parameter(init_weight.clone())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return topk_relu(self.prep(h) @ self.weight.t(), self.k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--encoder_draw", type=int, default=0)
    ap.add_argument("--adapter_seed", type=int, default=0)
    ap.add_argument("--train_cache", required=True)
    ap.add_argument("--val_cache", required=True)
    ap.add_argument("--sae_checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--recipe", choices=("paper", "pr26"), default="paper",
        help="paper = weighted-L1-only supplemental recipe; pr26 = legacy Huber+rank implementation",
    )
    # Paper adapter config (spec section 1) -- do not change across conditions.
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16, help="scenes per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--loss_type", default=None, choices=("mse", "huber", "l1"))
    ap.add_argument("--topk_train", type=int, default=10)
    ap.add_argument("--topk_weight", type=float, default=3.0)
    ap.add_argument("--rest_weight", type=float, default=1.0)
    ap.add_argument("--rank_loss_weight", type=float, default=None)
    ap.add_argument("--rank_temperature", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    ap.add_argument("--train_items", type=int, default=250_000)
    ap.add_argument("--early_stop_val_items", type=int, default=25_000)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--max_train_scenes", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # The paper's stated objective is weighted L1 and contains no auxiliary ranking
    # term.  PR #26 used Huber + 0.25 ranking loss; retain that only behind an explicit
    # recipe name so it remains auditable and cannot silently become the paper control.
    if args.loss_type is None:
        args.loss_type = "l1" if args.recipe == "paper" else "huber"
    if args.rank_loss_weight is None:
        args.rank_loss_weight = 0.0 if args.recipe == "paper" else 0.25
    if args.recipe == "paper" and (args.loss_type != "l1" or args.rank_loss_weight != 0.0):
        raise ValueError("paper recipe requires --loss_type l1 and --rank_loss_weight 0")

    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.adapter_seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
    use_bf16 = args.precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()

    sae_ckpt = torch.load(args.sae_checkpoint, map_location="cpu")
    validate_sae_checkpoint(sae_ckpt)
    rep_seed_base = 1000 + args.adapter_seed if args.condition == "C6" else 1000
    rep, prov = build_representation(
        args.condition, sae_ckpt, device, encoder_draw=args.encoder_draw,
        seed_base=rep_seed_base,
    )
    if args.condition == "C6":
        prov["encoder_init_seed"] = rep_seed_base + args.encoder_draw
    train = load_slim_cache(Path(args.train_cache), device)
    val = load_slim_cache(Path(args.val_cache), device)
    validate_slim_cache(train, "train")
    validate_slim_cache(val, "val")
    val_proposal_fingerprint = proposal_fingerprint(val)
    # Mirror WaymoE2E(n_items=..., seed=...) exactly: it selects one deterministic
    # contiguous window.  The split seed is fixed across adapter seeds; only adapter
    # initialization and epoch shuffle vary in the seed sweep.
    n_train_full = train["h"].size(0)
    train_items = min(args.max_train_scenes or args.train_items, n_train_full)
    train_start = random.Random(args.split_seed).randint(0, n_train_full - train_items)
    train = {key: value[train_start : train_start + train_items] for key, value in train.items()}
    n = train_items

    n_val_full = val["h"].size(0)
    early_val_items = min(args.early_stop_val_items, n_val_full)
    early_val_start = random.Random(args.split_seed + 1).randint(0, n_val_full - early_val_items)
    early_val = {
        key: value[early_val_start : early_val_start + early_val_items]
        for key, value in val.items()
    }

    encoder = None
    if args.condition == "C6":
        encoder = LearnedEncoder(rep.prep, rep.weight, rep.k).to(device)

    rep_dim = rep.dim
    if args.recipe == "paper":
        model = PaperScorerResidual(
            rep_dim=rep_dim, traj_dim=train["traj"].size(-1),
            hidden_dim=args.hidden_dim, dropout=args.dropout,
        ).to(device)
        architecture = "paper_three_tower_equal_width_two_layer_head"
    else:
        model = SAEScorerResidual(
            sae_dim=rep_dim, traj_dim=train["traj"].size(-1),
            hidden_dim=args.hidden_dim, dropout=args.dropout,
        ).to(device)
        architecture = "pr26_asymmetric_score_tower_three_layer_head"
    params = list(model.parameters()) + (list(encoder.parameters()) if encoder else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    g = torch.Generator(device="cpu").manual_seed(args.adapter_seed)
    best, best_state, bad, stop_epoch = float("inf"), None, 0, args.epochs - 1
    history = []
    for epoch in range(args.epochs):
        order = torch.randperm(n, generator=g).to(device)
        run, steps = 0.0, 0
        for s in range(0, n, args.batch_size):
            sel = order[s : s + args.batch_size]
            h, tr = train["h"][sel], train["traj"][sel]
            sc, ade = train["scores"][sel], train["ade"][sel]
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                x = encoder(h) if encoder is not None else rep(h)
                b, k, _ = tr.shape
                w = proposal_weights_from_ade(
                    ade, topk=args.topk_train, topk_weight=args.topk_weight,
                    rest_weight=args.rest_weight,
                )
                res = model(
                    x[:, None, :].expand(b, k, x.size(-1)).reshape(b * k, -1),
                    tr.reshape(b * k, -1),
                    sc.reshape(b * k),
                )
                reg = residual_loss(
                    res, (ade - sc).reshape(b * k), args.loss_type, w.reshape(-1)
                )
                if args.rank_loss_weight:
                    rank = pairwise_ranking_loss(
                        sc + res.view(b, k), ade, proposal_weights=w,
                        temperature=args.rank_temperature,
                    )
                    loss = reg + args.rank_loss_weight * rank
                else:
                    loss = reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.detach()); steps += 1

        m = evaluate(model, rep, early_val, device, encoder=encoder, use_bf16=use_bf16)
        history.append({"epoch": epoch, "train_loss": run / max(steps, 1),
                        "val_ade": m["ade"], "rel_improvement_pct": m["rel_improvement_pct"]})
        print(f"[{args.condition} d{args.encoder_draw} s{args.adapter_seed}] "
              f"epoch {epoch}: loss={run/max(steps,1):.4f} val_ade={m['ade']:.4f} "
              f"rel={m['rel_improvement_pct']:+.2f}% rank={m['mean_rank']:.2f}", flush=True)
        if m["ade"] < best - args.min_delta:
            best, bad, stop_epoch = m["ade"], 0, epoch
            best_state = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()} if encoder else None)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state[0])
        if encoder is not None and best_state[1] is not None:
            encoder.load_state_dict(best_state[1])
    model.to(device)
    final = evaluate(model, rep, val, device, encoder=encoder, use_bf16=use_bf16)

    tag = (
        f"{args.recipe}_{args.condition}_d{args.encoder_draw}_s{args.adapter_seed}"
        f"_h{args.hidden_dim}"
    )
    out = Path(args.output_dir)
    (out / "per_scene").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save(final["per_scene_ade"], out / "per_scene" / f"{tag}.pt")
    torch.save(final["per_scene_rank"], out / "per_scene" / f"{tag}_rank.pt")
    c0_path = out / "per_scene" / "C0.pt"
    c0_rank_path = out / "per_scene" / "C0_rank.pt"
    if args.condition == "C4" and args.encoder_draw == 0 and args.adapter_seed == 42:
        torch.save(final["per_scene_ade_base"], c0_path)
        torch.save(final["per_scene_rank_base"], c0_rank_path)
    if args.condition == "C3" and args.adapter_seed == 42:
        random_dir = out / "random_encoders"
        random_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"weight": rep.weight.cpu(), "provenance": prov},
            random_dir / f"W_rand_d{args.encoder_draw}.pt",
        )
    torch.save(
        {"model_state_dict": model.state_dict(),
         # Top-level dims plus the explicit architecture make checkpoints self-describing.
         # PR-26 checkpoints remain compatible with its legacy loader; paper-recipe
         # checkpoints must be rebuilt as PaperScorerResidual before loading the state.
         "sae_dim": rep_dim, "traj_dim": train["traj"].size(-1),
         "hidden_dim": args.hidden_dim, "dropout": args.dropout,
         "best_metric": best,
         "encoder_state_dict": encoder.state_dict() if encoder else None,
         "W_rand": rep.weight.cpu() if args.condition in ("C3",) else None,
         "architecture": architecture,
         "provenance": prov, "args": vars(args)},
        out / "ckpt" / f"{tag}.pt",
    )
    record = {
        "recipe": args.recipe, "architecture": architecture,
        "condition": args.condition, "encoder_draw": args.encoder_draw,
        "adapter_seed": args.adapter_seed, "hidden_dim": args.hidden_dim,
        "rep_dim": rep_dim, "rep_kind": prov["kind"],
        "ade": final["ade"], "ade_base": final["ade_base"],
        "rel_improvement_pct": final["rel_improvement_pct"],
        "mean_rank": final["mean_rank"], "mean_rank_base": final["mean_rank_base"],
        "early_stop_epoch": stop_epoch,
        "n_train_scenes": n, "wall_time_s": round(time.time() - t0, 1),
        "n_params": sum(p.numel() for p in params),
        "val_proposal_fingerprint_100": val_proposal_fingerprint,
        "train_window": [train_start, train_start + train_items],
        "early_stop_val_window": [early_val_start, early_val_start + early_val_items],
        "precision_resolved": "bf16" if use_bf16 else "fp32",
        "history": history,
    }
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "runs" / f"{tag}.json").write_text(json.dumps(record, indent=2))
    (out / "config_diffs").mkdir(parents=True, exist_ok=True)
    (out / "config_diffs" / f"{tag}.json").write_text(json.dumps({
        "base_condition": "C4",
        "only_representation_change": {
            "condition": args.condition,
            "representation": prov,
            "hidden_dim": args.hidden_dim,
        },
        "fixed_training": {
            "recipe": args.recipe, "architecture": architecture,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "dropout": args.dropout,
            "loss_type": args.loss_type, "topk_train": args.topk_train,
            "topk_weight": args.topk_weight, "rest_weight": args.rest_weight,
            "rank_loss_weight": args.rank_loss_weight,
            "patience": args.patience, "min_delta": args.min_delta,
            "train_items": train_items, "train_window_start": train_start,
            "early_stop_val_items": early_val_items,
            "early_stop_val_window_start": early_val_start,
            "split_seed": args.split_seed,
            "precision": "bf16" if use_bf16 else "fp32",
        },
    }, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
