"""Manifest, aggregation and paired bootstrap for the ladder (spec sections 5, 8)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

ADAPTER_SEEDS_5 = [42, 43, 44, 45, 46]
ADAPTER_SEEDS_3 = [42, 43, 44]

# Seed protocol (spec section 3): C1 and C4 carry the headline comparison and get n=5;
# everything else n=3. C3 is 5 encoder draws x 3 adapter seeds. C5 is two widths x 3.
MANIFEST_SPEC = [
    ("C4", [0], ADAPTER_SEEDS_5, [256]),
    ("C1", [0], ADAPTER_SEEDS_5, [256]),
    ("C3", [0, 1, 2, 3, 4], ADAPTER_SEEDS_3, [256]),
    ("C2", [0], ADAPTER_SEEDS_3, [256]),
    ("C5", [0], ADAPTER_SEEDS_3, [512, 1024]),
]


def build_manifest(path: Path, include_c6: bool = False) -> list[dict]:
    rows = []
    for cond, draws, seeds, hiddens in MANIFEST_SPEC:
        for d in draws:
            for s in seeds:
                for hd in hiddens:
                    rows.append({"condition": cond, "encoder_draw": d, "adapter_seed": s, "hidden_dim": hd})
    if include_c6:
        rows += [{"condition": "C6", "encoder_draw": 0, "adapter_seed": s, "hidden_dim": 256} for s in ADAPTER_SEEDS_3]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    return rows


def load_runs(out_dir: Path) -> list[dict]:
    runs = []
    for p in sorted((out_dir / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        r.pop("history", None)
        r["tag"] = p.stem
        runs.append(r)
    return runs


def group_key(r: dict) -> str:
    """C5 is reported per width, since width is the manipulated variable there."""
    return f"{r['condition']}_h{r['hidden_dim']}" if r["condition"] == "C5" else r["condition"]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """95% CI on mean(a - b) over scenes.

    Every condition reranks the SAME proposals on the SAME scenes, so per-scene ADE is
    paired and the bootstrap resamples scenes (spec section 5).
    """
    d = a - b
    rng = np.random.default_rng(seed)
    # A monolithic (10k, 106k) int64 index array needs >8 GiB.  Chunk the exact same
    # resampling operation so aggregation remains practical off the large SLURM nodes.
    means = np.empty(n_boot, dtype=np.float64)
    chunk = max(1, min(256, n_boot))
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        idx = rng.integers(0, len(d), size=(stop - start, len(d)))
        means[start:stop] = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def seed_mean_scene_ade(out_dir: Path, tags: list[str]) -> np.ndarray:
    """Average per-scene ADE across a group's seeds -> one paired vector per condition."""
    mats = [torch.load(out_dir / "per_scene" / f"{t}.pt", map_location="cpu").numpy() for t in tags]
    return np.mean(np.stack(mats), axis=0)


def ci_text(values: tuple[float, float, float]) -> str:
    _, lo, hi = values
    return f"[{lo:.5f}, {hi:.5f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--make_manifest", action="store_true")
    ap.add_argument("--include_c6", action="store_true")
    ap.add_argument("--n_boot", type=int, default=10_000)
    args = ap.parse_args()
    out = Path(args.output_dir)

    if args.make_manifest:
        rows = build_manifest(out / "manifest.json", include_c6=args.include_c6)
        print(f"manifest: {len(rows)} runs -> {out/'manifest.json'}")
        for cond in dict.fromkeys(r["condition"] for r in rows):
            print(f"  {cond}: {sum(1 for r in rows if r['condition']==cond)}")
        return

    runs = load_runs(out)
    if not runs:
        raise SystemExit(f"no runs found under {out/'runs'}")

    with (out / "runs.csv").open("w", newline="") as fh:
        keys = ["tag", "recipe", "architecture", "condition", "encoder_draw", "adapter_seed",
                "hidden_dim", "rep_dim", "rep_kind", "ade", "ade_base",
                "rel_improvement_pct", "mean_rank", "mean_rank_base",
                "early_stop_epoch", "n_params", "n_train_scenes", "wall_time_s",
                "val_proposal_fingerprint_100", "train_window", "early_stop_val_window",
                "precision_resolved"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)

    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(group_key(r), []).append(r)

    fingerprints = {r.get("val_proposal_fingerprint_100") for r in runs}
    fingerprints.discard(None)
    if len(fingerprints) > 1:
        raise ValueError(f"proposal-set identity gate failed: {sorted(fingerprints)}")

    base_ade = runs[0]["ade_base"]
    base_rank = runs[0].get("mean_rank_base")
    base_scene_path = out / "per_scene" / "C0.pt"
    if not base_scene_path.exists():
        raise SystemExit(f"missing base per-scene vector {base_scene_path}; rerun with the paper trainer")
    base_scene = torch.load(base_scene_path, map_location="cpu").numpy()

    agg = []
    for key, rs in sorted(groups.items()):
        rel = np.array([r["rel_improvement_pct"] for r in rs])
        rank = np.array([r["mean_rank"] for r in rs])
        agg.append({
            "condition": key, "n_runs": len(rs),
            "rel_improvement_mean": round(float(rel.mean()), 4),
            "rel_improvement_std": round(float(rel.std(ddof=1)) if len(rel) > 1 else 0.0, 4),
            "ade_mean": round(float(np.mean([r["ade"] for r in rs])), 5),
            "mean_rank_mean": round(float(rank.mean()), 4),
            "mean_rank_std": round(float(rank.std(ddof=1)) if len(rank) > 1 else 0.0, 4),
            "n_params": rs[0]["n_params"],
        })
    agg.insert(0, {"condition": "C0", "n_runs": 0, "rel_improvement_mean": 0.0,
                   "rel_improvement_std": 0.0, "ade_mean": round(base_ade, 5),
                   "mean_rank_mean": round(base_rank, 4) if base_rank is not None else None,
                   "mean_rank_std": 0.0, "n_params": 0})

    # Paired bootstrap for the comparisons named in spec section 5.
    scene = {k: seed_mean_scene_ade(out, [r["tag"] for r in rs]) for k, rs in groups.items()}
    pairs = [("C4", "C1"), ("C4", "C2"), ("C4", "C3"), ("C4", "C5_h512"),
             ("C4", "C5_h1024"), ("C1", "C2"), ("C4", "C6")]
    cis = []
    for a, b in pairs:
        if a not in scene or b not in scene:
            continue
        # delta = ADE(b) - ADE(a): positive means `a` is better.
        d, lo, hi = paired_bootstrap(scene[b], scene[a], n_boot=args.n_boot)
        cis.append({"comparison": f"{a} vs {b}", "delta_ade_m": round(d, 5),
                    "ci95_low": round(lo, 5), "ci95_high": round(hi, 5),
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "better": a if d > 0 else b})
    for key in scene:
        d, lo, hi = paired_bootstrap(base_scene, scene[key], n_boot=args.n_boot)
        for row in agg:
            if row["condition"] == key:
                row["delta_vs_C0_m"] = round(d, 5)
                row["ci95_vs_C0"] = f"[{lo:.5f}, {hi:.5f}]"

    # The requested aggregate table carries deltas to both headline references.
    for row in agg:
        key = row["condition"]
        if key == "C0" or key not in scene:
            continue
        for ref in ("C1", "C4"):
            if ref not in scene:
                continue
            values = paired_bootstrap(scene[ref], scene[key], n_boot=args.n_boot)
            row[f"delta_vs_{ref}_m"] = round(values[0], 5)
            row[f"ci95_vs_{ref}"] = ci_text(values)

    with (out / "aggregate_table.csv").open("w", newline="") as fh:
        keys = list({k: None for r in agg for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(agg)
    with (out / "pairwise_cis.csv").open("w", newline="") as fh:
        if cis:
            w = csv.DictWriter(fh, fieldnames=list(cis[0]))
            w.writeheader(); w.writerows(cis)

    # Section 8 plot: seed variation, base at zero, and the paper point estimate.
    try:
        import matplotlib.pyplot as plt

        plot_rows = [r for r in agg if r["condition"] != "C0"]
        labels = ["C0"] + [r["condition"] for r in plot_rows]
        values = [0.0] + [r["rel_improvement_mean"] for r in plot_rows]
        errors = [0.0] + [r["rel_improvement_std"] for r in plot_rows]
        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(labels)), 4.5))
        ax.bar(labels, values, yerr=errors, capsize=4)
        ax.axhline(3.44, color="black", linestyle="--", linewidth=1, label="paper C4: 3.44%")
        ax.set_ylabel("Relative ADE@5s improvement (%)")
        ax.set_xlabel("Condition")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "relative_improvement.png", dpi=200)
        plt.close(fig)
    except ImportError:
        print("warning: matplotlib unavailable; skipped relative_improvement.png")

    print(json.dumps({"aggregate": agg, "pairwise": cis}, indent=2))


if __name__ == "__main__":
    main()
