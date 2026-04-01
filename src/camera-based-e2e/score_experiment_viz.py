"""
Compare scorer-loss experiments across runs and auto-generate rankings.

This script scans Lightning CSV logs under a logs root, extracts final validation
metrics per run, groups by scorer-loss configuration, and writes:
  - summary CSV
  - ranking bar chart
  - oracle-vs-pred scatter (diagnose ranking bottleneck)
  - regret-vs-top1 scatter (diagnose scorer discrimination)
  - markdown report with "what went right/wrong"

Usage:
  python score_experiment_viz.py --logs_root /scratch/.../waymo/logs
  python score_experiment_viz.py --logs_root /scratch/.../waymo/logs --latest_n 8
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_hparams_kv(hparams_path: Path) -> Dict[str, str]:
    """
    Parse a few scalar keys from hparams.yaml without requiring strict YAML parse.
    This is intentionally tolerant because some generated hparams files can contain
    very large content blocks.
    """
    keys = {
        "score_loss_type",
        "score_target_type",
        "score_weight",
        "score_temperature",
        "score_rank_weight",
        "score_margin",
        "score_topk",
        "comfort_jerk_threshold",
        "model_type",
    }
    out: Dict[str, str] = {}
    if not hparams_path.exists():
        return out

    try:
        with hparams_path.open("r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                if k in keys:
                    out[k] = v.strip().strip("'\"")
    except Exception:
        return out
    return out


def _to_float(v: Optional[str], default: float) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: Optional[str], default: int) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _find_runs(logs_root: Path) -> List[Path]:
    runs = sorted(logs_root.glob("camera_e2e_*/version_0/metrics.csv"))
    return runs


def _final_metric(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    s = df[col].dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def collect_run_table(logs_root: Path, latest_n: int = 0) -> pd.DataFrame:
    rows = []
    metrics_files = _find_runs(logs_root)
    if latest_n > 0:
        metrics_files = metrics_files[-latest_n:]

    for mpath in metrics_files:
        run_dir = mpath.parent
        run_name = run_dir.parent.name
        try:
            df = pd.read_csv(mpath)
        except Exception:
            continue

        hp = _read_hparams_kv(run_dir / "hparams.yaml")
        score_loss_type = hp.get("score_loss_type", "bce")

        row = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "score_loss_type": score_loss_type,
            "score_target_type": hp.get("score_target_type", "l1"),
            "score_weight": _to_float(hp.get("score_weight"), 1.0),
            "score_temperature": _to_float(hp.get("score_temperature"), 5.0),
            "score_rank_weight": _to_float(hp.get("score_rank_weight"), 0.0),
            "score_margin": _to_float(hp.get("score_margin"), 0.2),
            "score_topk": _to_int(hp.get("score_topk"), 0),
            "val_ade_pred": _final_metric(df, "val_ade_pred"),
            "val_ade_oracle": _final_metric(df, "val_ade_oracle"),
            "val_ade_regret": _final_metric(df, "val_ade_regret"),
            "val_loss": _final_metric(df, "val_loss"),
            "val_loss_score": _final_metric(df, "val_loss_score"),
            "val_score_top1_acc": _final_metric(df, "val_score_top1_acc"),
            "val_score_gap_best_second": _final_metric(df, "val_score_gap_best_second"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values("run_name").reset_index(drop=True)
    return out


def _approach_label(r: pd.Series) -> str:
    t = r["score_loss_type"]
    tgt = r.get("score_target_type", "l1")
    prefix = f"{t}+{tgt}" if tgt != "l1" else t
    if t == "bce_pairwise":
        return f"{prefix}(w={r['score_rank_weight']:.2f},m={r['score_margin']:.2f},k={int(r['score_topk'])})"
    if t in ("bce", "listnet"):
        return f"{prefix}(tau={r['score_temperature']:.1f})"
    return prefix


def _diagnose_row(r: pd.Series) -> str:
    pred = r["val_ade_pred"]
    oracle = r["val_ade_oracle"]
    regret = r["val_ade_regret"]
    top1 = r.get("val_score_top1_acc", np.nan)

    if np.isnan(pred) or np.isnan(oracle):
        return "incomplete metrics"
    if oracle < 0.8 and regret > 1.0:
        return "strong proposals, weak ranking (scorer bottleneck)"
    if oracle >= 0.8:
        return "trajectory quality bottleneck (oracle not strong enough)"
    if not np.isnan(top1) and top1 < 0.3:
        return "low top1 match; scorer ordering not learning"
    if regret < 0.7:
        return "ranking improved"
    return "mixed behavior"


def _plot_rank_bar(df: pd.DataFrame, out_dir: Path):
    g = df.groupby("approach", as_index=False)["val_ade_pred"].min().sort_values("val_ade_pred")
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(g))
    ax.bar(x, g["val_ade_pred"], color="#2f6db0")
    ax.set_ylabel("Best val_ade_pred (m)")
    ax.set_title("Approach Ranking (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(g["approach"], rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ranking_val_ade_pred.png", dpi=180)
    plt.close(fig)


def _plot_oracle_vs_pred(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, sub in df.groupby("approach"):
        ax.scatter(sub["val_ade_oracle"], sub["val_ade_pred"], label=name, alpha=0.85)
    lim_lo = np.nanmin([df["val_ade_oracle"].min(), df["val_ade_pred"].min()]) * 0.9
    lim_hi = np.nanmax([df["val_ade_oracle"].max(), df["val_ade_pred"].max()]) * 1.1
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="gray", alpha=0.7, label="pred=oracle")
    ax.set_xlabel("val_ade_oracle (m)")
    ax.set_ylabel("val_ade_pred (m)")
    ax.set_title("Oracle vs Selected ADE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "oracle_vs_pred_scatter.png", dpi=180)
    plt.close(fig)


def _plot_regret_vs_top1(df: pd.DataFrame, out_dir: Path):
    if "val_score_top1_acc" not in df.columns or df["val_score_top1_acc"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, sub in df.groupby("approach"):
        ax.scatter(sub["val_score_top1_acc"], sub["val_ade_regret"], label=name, alpha=0.85)
    ax.set_xlabel("val_score_top1_acc")
    ax.set_ylabel("val_ade_regret (m)")
    ax.set_title("Ranking Accuracy vs Regret")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "regret_vs_top1_scatter.png", dpi=180)
    plt.close(fig)


def _write_report(df: pd.DataFrame, out_dir: Path):
    ranked = df.sort_values("val_ade_pred").reset_index(drop=True)
    by_approach = (
        df.groupby("approach", as_index=False)[
            ["val_ade_pred", "val_ade_oracle", "val_ade_regret", "val_loss_score", "val_score_top1_acc"]
        ]
        .agg("mean")
        .sort_values("val_ade_pred")
    )

    lines: List[str] = []
    lines.append("# Scorer Experiment Report")
    lines.append("")
    lines.append("## Overall ranking (by val_ade_pred)")
    lines.append("")
    for i, r in ranked.head(10).iterrows():
        lines.append(
            f"{i+1}. `{r['run_name']}` | approach `{r['approach']}` | "
            f"val_ade_pred={r['val_ade_pred']:.3f}, oracle={r['val_ade_oracle']:.3f}, "
            f"regret={r['val_ade_regret']:.3f} | diagnosis: {r['diagnosis']}"
        )

    lines.append("")
    lines.append("## Approach-level means")
    lines.append("")
    lines.append("| approach | val_ade_pred | val_ade_oracle | val_ade_regret | val_loss_score | val_score_top1_acc |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in by_approach.iterrows():
        lines.append(
            f"| {r['approach']} | {r['val_ade_pred']:.3f} | {r['val_ade_oracle']:.3f} | "
            f"{r['val_ade_regret']:.3f} | {r['val_loss_score']:.3f} | {r['val_score_top1_acc']:.3f} |"
        )

    lines.append("")
    lines.append("## What went wrong / right")
    lines.append("")
    scorer_bad = ranked[(ranked["val_ade_oracle"] < 0.8) & (ranked["val_ade_regret"] > 1.0)]
    if not scorer_bad.empty:
        lines.append("- **Scorer bottleneck persists** in runs where oracle is strong but regret stays high.")
    trajectory_bad = ranked[ranked["val_ade_oracle"] >= 0.8]
    if not trajectory_bad.empty:
        lines.append("- **Trajectory generation bottleneck** appears in some runs (high oracle ADE).")
    improved = ranked[ranked["val_ade_regret"] < 0.7]
    if not improved.empty:
        lines.append("- **Ranking improved** for runs with regret below 0.7.")
    if scorer_bad.empty and trajectory_bad.empty and improved.empty:
        lines.append("- Mixed outcomes; inspect scatter plots for separation patterns.")

    (out_dir / "score_experiment_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Rank scorer-loss experiments and generate diagnostics")
    parser.add_argument("--logs_root", type=str, required=True, help="Root containing camera_e2e_*/version_0")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <logs_root>/score_experiments)")
    parser.add_argument("--latest_n", type=int, default=0, help="Only process latest N runs (0 = all)")
    args = parser.parse_args()

    logs_root = Path(args.logs_root)
    out_dir = Path(args.out_dir) if args.out_dir else logs_root / "score_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_run_table(logs_root, latest_n=args.latest_n)
    if df.empty:
        raise RuntimeError(f"No runs with metrics found under {logs_root}")

    df["approach"] = df.apply(_approach_label, axis=1)
    df["diagnosis"] = df.apply(_diagnose_row, axis=1)
    df.to_csv(out_dir / "score_experiment_summary.csv", index=False)

    _plot_rank_bar(df, out_dir)
    _plot_oracle_vs_pred(df, out_dir)
    _plot_regret_vs_top1(df, out_dir)
    _write_report(df, out_dir)

    print(f"Wrote summary CSV and plots to: {out_dir}")
    print(f"Top run by val_ade_pred: {df.sort_values('val_ade_pred').iloc[0]['run_name']}")


if __name__ == "__main__":
    main()

