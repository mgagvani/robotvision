"""Slim caches and frozen representation builders for the capacity-control ladder.

The only thing that varies across conditions is the adapter's representation-tower input
(spec section 1). Everything here produces that input as a frozen, deterministic function
of the cached block-3 activation ``h``, so no condition ever touches a planner or SAE
forward pass during training (spec section 9).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

BLOCK = 3
TOPK = 24
CONDITIONS = ("C1", "C2", "C3", "C4", "C5", "C6")


def build_slim_cache(token_path: Path, out_path: Path, horizon: int = 20) -> dict:
    """Reduce an extracted planner-token blob to what the ladder needs.

    Keeps the block-3 activation, the flattened proposals, the base scores, and the
    per-proposal ADE against the logged future. The proposal set and base scores come
    straight from the frozen planner, so they are shared by every condition by
    construction (spec section 1).
    """
    blob = torch.load(token_path, map_location="cpu")
    h = blob[f"planner_query_tok_block_{BLOCK}"].float()
    n = h.size(0)
    traj = blob["trajectory"].view(n, -1, horizon, 2).float()
    future = blob["future"].float()
    out = {
        "h": h,
        "traj": traj.reshape(n, traj.size(1), -1).contiguous(),
        "scores": blob["scores"].float(),
        "ade": torch.norm(traj - future[:, None], dim=-1).mean(dim=-1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    return {k: tuple(v.shape) for k, v in out.items()}


def load_slim_cache(path: Path, device: torch.device) -> dict:
    blob = torch.load(path, map_location="cpu")
    return {k: v.to(device) for k, v in blob.items()}


class Preprocessor:
    """prep(h) exactly as `SparseAutoencoder.preprocess_input` (spec section 2).

    Subtract the learned bias b_pre, subtract the per-sample mean across features, then
    divide by the centered l2 norm.
    """

    def __init__(self, b_pre: torch.Tensor) -> None:
        self.b_pre = b_pre

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        shifted = h - self.b_pre
        centered = shifted - shifted.mean(dim=-1, keepdim=True)
        return centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def topk_relu(values: torch.Tensor, k: int = TOPK) -> torch.Tensor:
    """ReLU(TopK(.)) -- identical to `SparseAutoencoder.topk_masked_relu`."""
    k_eff = min(k, values.size(-1))
    top_values, top_indices = torch.topk(values, k=k_eff, dim=-1)
    masked = torch.zeros_like(values)
    masked.scatter_(-1, top_indices, top_values)
    return torch.relu(masked)


def sae_encoder_init(input_dim: int, latent_dim: int, k: int, generator: torch.Generator) -> torch.Tensor:
    """Draw W_rand under the SAE's own encoder init scheme (spec section 3, C3).

    `SparseAutoencoder.reset_topk_parameters` draws decoder columns from
    N(0, 1/sqrt(d)), normalizes them to unit norm, then ties the encoder to their
    transpose scaled by sqrt(k/d). Reproducing that -- rather than Kaiming -- keeps C3 a
    true null model: random dictionary directions of the same geometry, unlearned.
    """
    dec = torch.empty(input_dim, latent_dim).normal_(0.0, 1.0 / math.sqrt(input_dim), generator=generator)
    dec = dec / dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return dec.t().contiguous() * math.sqrt(max(k, 1) / float(input_dim))


class Representation:
    """Frozen map from cached ``h`` to a condition's adapter input."""

    def __init__(self, kind: str, prep: Preprocessor, weight: torch.Tensor | None = None, k: int = TOPK):
        self.kind = kind
        self.prep = prep
        self.weight = weight
        self.k = k
        self.dim = 384 if weight is None else weight.size(0)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        if self.kind == "dense_raw":
            return h
        if self.kind == "dense_norm":
            return self.prep(h)
        if self.kind in ("sparse_sae", "sparse_random"):
            return topk_relu(self.prep(h) @ self.weight.t(), self.k)
        raise ValueError(f"unknown representation kind {self.kind!r}")


def build_representation(
    condition: str,
    sae_ckpt: dict,
    device: torch.device,
    *,
    encoder_draw: int = 0,
    seed_base: int = 1000,
) -> tuple[Representation, dict]:
    """Return the frozen representation for a condition, plus a provenance record."""
    state = sae_ckpt["state_dict"]
    b_pre = state["b_pre"].to(device).float()
    prep = Preprocessor(b_pre)
    input_dim, latent_dim, k = sae_dims(sae_ckpt)

    if condition in ("C1", "C5"):
        return Representation("dense_raw", prep), {"kind": "dense_raw"}
    if condition == "C2":
        return Representation("dense_norm", prep), {"kind": "dense_norm"}
    if condition == "C4":
        w = state["encoder.weight"].to(device).float()
        return Representation("sparse_sae", prep, w, k), {"kind": "sparse_sae", "k": k}
    if condition == "C3":
        g = torch.Generator().manual_seed(seed_base + encoder_draw)
        w = sae_encoder_init(input_dim, latent_dim, k, g).to(device)
        return (
            Representation("sparse_random", prep, w, k),
            {"kind": "sparse_random", "k": k, "encoder_draw": encoder_draw,
             "init": "sae reset_topk_parameters (normalized decoder columns, tied, x sqrt(k/d))"},
        )
    if condition == "C6":
        # Trainable encoder; the tensor here is only the initialization.
        g = torch.Generator().manual_seed(seed_base + encoder_draw)
        w = sae_encoder_init(input_dim, latent_dim, k, g).to(device)
        return Representation("sparse_random", prep, w, k), {"kind": "sparse_learned", "k": k}
    raise ValueError(f"unknown condition {condition!r}")


def sae_dims(ckpt: dict) -> tuple[int, int, int]:
    return int(ckpt["input_dim"]), int(ckpt["latent_dim"]), int(ckpt.get("k", TOPK))


def validate_sae_checkpoint(ckpt: dict) -> None:
    """Fail loudly if the SAE is not the frozen block-3 top-k model the ladder assumes."""
    missing = [k for k in ("state_dict", "input_dim", "latent_dim", "k") if k not in ckpt]
    if missing:
        raise ValueError(f"SAE checkpoint missing keys: {missing}")
    state = ckpt["state_dict"]
    for key in ("b_pre", "encoder.weight"):
        if key not in state:
            raise ValueError(f"SAE state_dict missing {key!r}; prep()/C4 would be silently wrong")
    if int(ckpt.get("block_index", BLOCK)) != BLOCK:
        raise ValueError(f"expected block {BLOCK}, got {ckpt.get('block_index')}")
    if int(ckpt["k"]) != TOPK:
        raise ValueError(f"expected k={TOPK}, got {ckpt['k']}")
    d_in, d_lat, _ = sae_dims(ckpt)
    if (d_in, d_lat) != (384, 384):
        raise ValueError(f"expected frozen 384->384 SAE, got {d_in}->{d_lat}")
    if tuple(state["encoder.weight"].shape) != (d_lat, d_in):
        raise ValueError(f"encoder.weight shape {tuple(state['encoder.weight'].shape)} != ({d_lat}, {d_in})")


def validate_slim_cache(cache: dict, name: str) -> None:
    """Shape/finiteness contract for a slim cache (spec section 1 assertions)."""
    for key in ("h", "traj", "scores", "ade"):
        if key not in cache:
            raise ValueError(f"{name} cache missing {key!r}")
    n, k = cache["scores"].shape
    if k != 50 or tuple(cache["h"].shape[1:]) != (384,) or tuple(cache["traj"].shape[2:]) != (40,):
        raise ValueError(
            f"{name} cache violates paper shapes: h={tuple(cache['h'].shape)} "
            f"traj={tuple(cache['traj'].shape)} scores={tuple(cache['scores'].shape)}"
        )
    if cache["h"].shape[0] != n or cache["traj"].shape[:2] != (n, k) or cache["ade"].shape != (n, k):
        raise ValueError(
            f"{name} cache shape mismatch: h={tuple(cache['h'].shape)} "
            f"traj={tuple(cache['traj'].shape)} scores={(n, k)} ade={tuple(cache['ade'].shape)}"
        )
    for key in ("h", "traj", "scores", "ade"):
        if not torch.isfinite(cache[key]).all():
            raise ValueError(f"{name} cache has non-finite values in {key!r}")


def proposal_fingerprint(cache: dict, n_scenes: int = 100, seed: int = 7) -> str:
    """Stable digest of the frozen proposal set and base scores.

    Recorded per run so the spec section 1 identity assertion is checked across every
    condition automatically, not only in a one-off gate.
    """
    import hashlib

    g = torch.Generator().manual_seed(seed)
    n = cache["scores"].shape[0]
    idx = torch.randperm(n, generator=g)[:n_scenes].to(cache["scores"].device)
    payload = torch.cat(
        [cache["traj"][idx].reshape(-1).double(), cache["scores"][idx].reshape(-1).double()]
    ).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]
