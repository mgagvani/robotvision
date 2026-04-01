"""
Generate debug diagnostic plots from training logs.

Reads metrics.csv produced by PyTorch Lightning's CSVLogger and renders:
  1. Gradient norms per module group (scene_encoder, proposal_init, refinement, scorer)
  2. Gradient norms for refinement internals (cross_attn, mlp, traj_residual per block)
  3. Gradient dominance ratio over time
  4. Activation statistics (mean, std, dead fraction) per module
  5. Proposal diagnostics (spread, diversity, score distribution)
  6. Loss component breakdown (stacked area)

Usage:
    python debug_viz.py --log_dir /path/to/logs/camera_e2e_YYYYMMDD_HHMM/version_0
    python debug_viz.py --log_dir /path/to/logs  # auto-picks newest run
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def find_metrics_csv(log_dir: Path) -> Path:
    if (log_dir / "metrics.csv").exists():
        return log_dir / "metrics.csv"
    candidates = sorted(log_dir.rglob("metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv found under {log_dir}")
    return candidates[-1]


def load_and_merge(csv_path: Path) -> pd.DataFrame:
    """
    Lightning CSVLogger writes one row per log call, with NaN in columns
    not logged in that call. Forward-fill within each step so we can
    correlate gradient norms with losses at the same step.
    """
    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise KeyError("metrics.csv must contain a 'step' column")
    return df


def _get_cols(df: pd.DataFrame, prefix: str) -> list:
    return sorted([c for c in df.columns if c.startswith(prefix)])


def _plot_series(ax, df, columns, title, ylabel, legend_strip="", logy=False, ewm_span=20):
    for col in columns:
        series = df.set_index("step")[col].dropna()
        if series.empty:
            continue
        label = col.replace(legend_strip, "").strip("/")
        smoothed = series.ewm(span=ewm_span, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed.values, label=label, linewidth=1.2)
        ax.fill_between(series.index, series.values, alpha=0.08)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Step")
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


# ======================================================================
# Individual plot functions
# ======================================================================

def plot_gradient_norms_by_module(df: pd.DataFrame, out_dir: Path):
    """Bar-style time series of gradient norms for the four main modules."""
    cols = [c for c in _get_cols(df, "grad_norm/") if c in (
        "grad_norm/scene_encoder", "grad_norm/proposal_init",
        "grad_norm/scorer", "grad_norm/total",
    ) or c.startswith("grad_norm/refine_block_")]
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    _plot_series(ax, df, cols, "Gradient Norms by Module", "L2 Norm", "grad_norm/", logy=True)
    fig.tight_layout()
    fig.savefig(out_dir / "grad_norms_modules.png", dpi=180)
    plt.close(fig)
    print(f"  -> {out_dir / 'grad_norms_modules.png'}")


def plot_gradient_norms_refinement(df: pd.DataFrame, out_dir: Path):
    """Detailed gradient norms inside refinement blocks."""
    cols = [c for c in _get_cols(df, "grad_norm/refine_") if "block" not in c]
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    _plot_series(ax, df, cols, "Refinement Internals — Gradient Norms", "L2 Norm", "grad_norm/", logy=True)
    fig.tight_layout()
    fig.savefig(out_dir / "grad_norms_refinement_detail.png", dpi=180)
    plt.close(fig)
    print(f"  -> {out_dir / 'grad_norms_refinement_detail.png'}")


def plot_gradient_norms_scorer_propinit(df: pd.DataFrame, out_dir: Path):
    """Scorer and ProposalInit sub-module gradient norms."""
    cols = _get_cols(df, "grad_norm/scorer_") + _get_cols(df, "grad_norm/propinit_")
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    _plot_series(ax, df, cols, "Scorer & ProposalInit Internals — Gradient Norms", "L2 Norm", "grad_norm/", logy=True)
    fig.tight_layout()
    fig.savefig(out_dir / "grad_norms_scorer_propinit.png", dpi=180)
    plt.close(fig)
    print(f"  -> {out_dir / 'grad_norms_scorer_propinit.png'}")


def plot_gradient_dominance(df: pd.DataFrame, out_dir: Path):
    """Which module dominates the gradient signal?"""
    col = "grad_norm/dominant_ratio"
    if col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    series = df.set_index("step")[col].dropna()
    smoothed = series.ewm(span=20, min_periods=1).mean()
    ax.plot(smoothed.index, smoothed.values, color="crimson", linewidth=1.5)
    ax.fill_between(series.index, series.values, alpha=0.15, color="crimson")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% dominance")
    ax.set_title("Gradient Dominance Ratio (max module / total)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Step")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "grad_dominance.png", dpi=180)
    plt.close(fig)
    print(f"  -> {out_dir / 'grad_dominance.png'}")


def plot_activation_stats(df: pd.DataFrame, out_dir: Path):
    """Activation mean/std/dead fraction for each hooked module output."""
    mean_cols = [c for c in _get_cols(df, "act/") if c.endswith("_mean")]
    std_cols = [c for c in _get_cols(df, "act/") if c.endswith("_std")]
    dead_cols = [c for c in _get_cols(df, "act/") if c.endswith("_frac_dead")]
    sat_cols = [c for c in _get_cols(df, "act/") if c.endswith("_frac_saturated")]

    if not mean_cols:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    _plot_series(axes[0, 0], df, mean_cols, "Activation Mean", "Mean", "act/")
    _plot_series(axes[0, 1], df, std_cols, "Activation Std", "Std", "act/")
    _plot_series(axes[1, 0], df, dead_cols, "Dead Neuron Fraction (|x| < 1e-6)", "Fraction", "act/")
    _plot_series(axes[1, 1], df, sat_cols, "Saturated Fraction (|x| > 10)", "Fraction", "act/")

    fig.suptitle("Activation Statistics by Module Output", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "activation_stats.png", dpi=180)
    plt.close(fig)
    print(f"  -> {out_dir / 'activation_stats.png'}")


def plot_proposal_diagnostics(df: pd.DataFrame, out_dir: Path):
    """Proposal spread, diversity, trajectory lengths, and score stats."""
    prop_cols = _get_cols(df, "proposals/")
    if not prop_cols:
        return

    spread_cols = [c for c in prop_cols if "std" in c or "dist" in c]
    length_cols = [c for c in prop_cols if "length" in c]
    score_cols = [c for c in prop_cols if "score" in c]

    n_panels = sum(1 for g in [spread_cols, length_cols, score_cols] if g)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    idx = 0
    if spread_cols:
        _plot_series(axes[idx], df, spread_cols, "Proposal Spread & Diversity", "Value", "proposals/")
        idx += 1
    if length_cols:
        _plot_series(axes[idx], df, length_cols, "Trajectory Lengths", "Length (m)", "proposals/")
        idx += 1
    if score_cols:
        _plot_series(axes[idx], df, score_cols, "Score Distribution", "Score", "proposals/")
        idx += 1

    fig.suptitle("Proposal Diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "proposal_diagnostics.png", dpi=180)
    plt.close(fig)
    print(f"  -> {out_dir / 'proposal_diagnostics.png'}")


def plot_loss_breakdown(df: pd.DataFrame, out_dir: Path):
    """Stacked area chart of all loss components."""
    loss_cols = [c for c in df.columns if c.startswith("train_loss_") and c != "train_loss"]
    if not loss_cols:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Individual loss curves
    _plot_series(axes[0], df, loss_cols, "Training Loss Components", "Loss", "train_loss_", logy=True)

    # Stacked area (absolute contribution)
    sub = df[["step"] + loss_cols].dropna(subset=loss_cols, how="all").copy()
    if sub.empty:
        plt.close(fig)
        return
    sub = sub.set_index("step").interpolate().fillna(0).clip(lower=0)
    sub_smooth = sub.ewm(span=30, min_periods=1).mean()
    axes[1].stackplot(sub_smooth.index, *[sub_smooth[c].values for c in loss_cols],
                      labels=[c.replace("train_loss_", "") for c in loss_cols], alpha=0.7)
    axes[1].set_title("Loss Component Breakdown (stacked)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Step")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "loss_breakdown.png", dpi=180)
    plt.close(fig)
    print(f"  -> {out_dir / 'loss_breakdown.png'}")


def plot_ade_regret(df: pd.DataFrame, out_dir: Path):
    """Oracle ADE vs predicted-best ADE, and regret over time."""
    cols = ["train_ade_pred", "train_ade_oracle", "train_ade_regret"]
    present = [c for c in cols if c in df.columns]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    _plot_series(ax, df, present, "ADE: Oracle vs Predicted-Best & Regret", "ADE (m)", "train_")
    fig.tight_layout()
    fig.savefig(out_dir / "ade_regret.png", dpi=180)
    plt.close(fig)
    print(f"  -> {out_dir / 'ade_regret.png'}")


def plot_param_norms(df: pd.DataFrame, out_dir: Path):
    """Parameter norms to detect weight explosion / vanishing."""
    cols = _get_cols(df, "param_norm/")
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    _plot_series(ax, df, cols, "Parameter Norms by Module", "L2 Norm", "param_norm/", logy=True)
    fig.tight_layout()
    fig.savefig(out_dir / "param_norms.png", dpi=180)
    plt.close(fig)
    print(f"  -> {out_dir / 'param_norms.png'}")


# ======================================================================
# Snapshot: single-image summary for quick glance
# ======================================================================

def plot_summary_dashboard(df: pd.DataFrame, out_dir: Path):
    """Single 2x3 dashboard with the most important signals."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Module gradient norms
    ax1 = fig.add_subplot(gs[0, 0])
    cols = [c for c in _get_cols(df, "grad_norm/") if c in (
        "grad_norm/scene_encoder", "grad_norm/proposal_init",
        "grad_norm/scorer", "grad_norm/total",
    ) or c.startswith("grad_norm/refine_block_")]
    if cols:
        _plot_series(ax1, df, cols, "Gradient Norms (modules)", "L2 Norm", "grad_norm/", logy=True, ewm_span=30)
    else:
        ax1.text(0.5, 0.5, "No grad_norm data", ha="center", va="center", transform=ax1.transAxes)

    # 2. Loss breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    loss_cols = [c for c in df.columns if c.startswith("train_loss_") and c != "train_loss"]
    if loss_cols:
        _plot_series(ax2, df, loss_cols, "Loss Components", "Loss", "train_loss_", logy=True, ewm_span=30)
    else:
        ax2.text(0.5, 0.5, "No loss data", ha="center", va="center", transform=ax2.transAxes)

    # 3. ADE regret
    ax3 = fig.add_subplot(gs[0, 2])
    ade_cols = [c for c in ("train_ade_pred", "train_ade_oracle", "train_ade_regret") if c in df.columns]
    if ade_cols:
        _plot_series(ax3, df, ade_cols, "ADE: Oracle vs Predicted", "ADE (m)", "train_", ewm_span=30)
    else:
        ax3.text(0.5, 0.5, "No ADE data", ha="center", va="center", transform=ax3.transAxes)

    # 4. Activation dead fraction
    ax4 = fig.add_subplot(gs[1, 0])
    dead_cols = [c for c in _get_cols(df, "act/") if c.endswith("_frac_dead")]
    if dead_cols:
        _plot_series(ax4, df, dead_cols, "Dead Neuron Fraction", "Fraction", "act/", ewm_span=30)
    else:
        ax4.text(0.5, 0.5, "No activation data", ha="center", va="center", transform=ax4.transAxes)

    # 5. Proposal diversity
    ax5 = fig.add_subplot(gs[1, 1])
    prop_cols = [c for c in _get_cols(df, "proposals/") if "dist" in c or "std" in c]
    if prop_cols:
        _plot_series(ax5, df, prop_cols, "Proposal Diversity", "Value", "proposals/", ewm_span=30)
    else:
        ax5.text(0.5, 0.5, "No proposal data", ha="center", va="center", transform=ax5.transAxes)

    # 6. Gradient dominance
    ax6 = fig.add_subplot(gs[1, 2])
    dom_col = "grad_norm/dominant_ratio"
    if dom_col in df.columns:
        series = df.set_index("step")[dom_col].dropna()
        if not series.empty:
            smoothed = series.ewm(span=30, min_periods=1).mean()
            ax6.plot(smoothed.index, smoothed.values, color="crimson", linewidth=1.5)
            ax6.fill_between(series.index, series.values, alpha=0.12, color="crimson")
            ax6.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax6.set_title("Gradient Dominance", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Ratio (max module / total)")
    ax6.set_xlabel("Step")
    ax6.set_ylim(0, 1.05)
    ax6.grid(True, alpha=0.3)

    fig.suptitle("Training Debug Dashboard", fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(out_dir / "debug_dashboard.png", dpi=200)
    plt.close(fig)
    print(f"  -> {out_dir / 'debug_dashboard.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate debug diagnostic plots from training logs")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to log directory (version_0/) or parent (auto-picks newest)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for plots (default: <log_dir>/debug_plots/)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    csv_path = find_metrics_csv(log_dir)
    print(f"Reading {csv_path}")
    df = load_and_merge(csv_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "debug_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    plot_gradient_norms_by_module(df, out_dir)
    plot_gradient_norms_refinement(df, out_dir)
    plot_gradient_norms_scorer_propinit(df, out_dir)
    plot_gradient_dominance(df, out_dir)
    plot_activation_stats(df, out_dir)
    plot_proposal_diagnostics(df, out_dir)
    plot_loss_breakdown(df, out_dir)
    plot_ade_regret(df, out_dir)
    plot_param_norms(df, out_dir)
    plot_summary_dashboard(df, out_dir)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} plots saved to {out_dir}")


if __name__ == "__main__":
    main()
