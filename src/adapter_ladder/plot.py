"""Ladder figure (spec section 8): relative improvement by condition.

Form: the data's job is magnitude comparison across a small set of named conditions,
so a bar chart with seed error bars. Colour encodes the *representation family*
(dense / learned-sparse / random-sparse) rather than the condition index, because the
finding is that the families separate -- identity is also carried by the axis labels and
a direct value label on every bar, so colour is never load-bearing alone.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the reference palette; validated with
# `validate_palette.js --pairs all --mode light`: all checks PASS (worst CVD dE 9.2,
# worst normal-vision dE 24.0). The aqua contrast WARN is relieved by direct labels.
DENSE = "#2a78d6"
LEARNED_SPARSE = "#eb6834"
RANDOM_SPARSE = "#1baf7a"
BASELINE = "#8a8985"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"

FAMILY = {
    "C0": ("base scorer, no adapter", BASELINE, "baseline"),
    "C1": ("dense raw  $h$", DENSE, "dense"),
    "C2": ("dense normalised  prep($h$)", DENSE, "dense"),
    "C5_h512": ("dense raw, hidden 512", DENSE, "dense"),
    "C5_h1024": ("dense raw, hidden 1024", DENSE, "dense"),
    "C4": ("SAE latent  $z$", LEARNED_SPARSE, "learned sparse"),
    "C3": ("random sparse  $z_{rand}$", RANDOM_SPARSE, "random sparse"),
}
ORDER = ["C0", "C1", "C2", "C5_h512", "C5_h1024", "C4", "C3"]
PAPER_CLAIM = 3.44


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.input_dir)

    rows = {r["condition"]: r for r in csv.DictReader((d / "aggregate_table.csv").open())}
    conds = [c for c in ORDER if c in rows]
    means = [float(rows[c]["rel_improvement_mean"]) for c in conds]
    stds = [float(rows[c]["rel_improvement_std"]) for c in conds]
    ns = [int(rows[c]["n_runs"]) for c in conds]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(conds))

    for i, c in enumerate(conds):
        label, color, _fam = FAMILY[c]
        ax.barh(i, means[i], height=0.62, color=color,
                edgecolor=SURFACE, linewidth=2, zorder=3)      # 2px surface gap between bars
        if stds[i] > 0:
            ax.errorbar(means[i], i, xerr=stds[i], fmt="none", ecolor=INK_MUTED,
                        elinewidth=1.4, capsize=3, zorder=4)
        txt = f"{means[i]:+.2f}%" + (f"  (n={ns[i]})" if ns[i] else "")
        ax.text(means[i] + stds[i] + 0.12, i, txt, va="center", ha="left",
                fontsize=9, color=INK, zorder=5)

    ax.axvline(PAPER_CLAIM, ls="--", lw=1.4, color=INK_MUTED, zorder=2)
    # Annotate at the top of the line; the legend occupies the lower right.
    ax.text(PAPER_CLAIM, -0.72, f"paper's claim {PAPER_CLAIM}%  ",
            fontsize=8.5, color=INK_MUTED, va="center", ha="right")

    ax.set_yticks(list(y))
    ax.set_yticklabels([FAMILY[c][0] for c in conds], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("relative ADE@5s improvement over base scorer (%)", fontsize=9.5, color=INK_MUTED)
    ax.set_xlim(0, max(m + s for m, s in zip(means, stds)) + 1.15)
    ax.tick_params(axis="x", labelsize=9, colors=INK_MUTED)
    ax.grid(axis="x", color="#e3e2de", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d5d4cf")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DENSE),
        plt.Rectangle((0, 0), 1, 1, color=LEARNED_SPARSE),
        plt.Rectangle((0, 0), 1, 1, color=RANDOM_SPARSE),
    ]
    ax.legend(handles, ["dense", "learned sparse (SAE)", "random sparse"],
              loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_MUTED)
    ax.set_title("Adapter capacity-control ladder: what actually drives the gain",
                 fontsize=11.5, color=INK, pad=12, loc="left")

    fig.tight_layout()
    out = Path(args.out) if args.out else d / "ladder_relative_improvement.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
