import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "best_class":
                    parsed[key] = value
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def infer_class_slugs(fieldnames: list[str]) -> list[str]:
    slugs = []
    for name in fieldnames:
        if name.startswith("r_"):
            slug = name[len("r_") :]
            if f"mean_{slug}" in fieldnames and f"active_rate_{slug}" in fieldnames:
                slugs.append(slug)
    return slugs


def pretty_name(slug: str) -> str:
    return slug.replace("_", " ").title()


def save_top_eta_plot(rows: list[dict], class_names: list[str], title: str, output_path: Path, top_k: int) -> None:
    top_rows = rows[:top_k]
    feature_labels = [f"f{int(row['feature_idx'])}" for row in reversed(top_rows)]
    eta_vals = [row["eta_sq"] for row in reversed(top_rows)]
    best_classes = [row["best_class"] for row in reversed(top_rows)]

    palette = plt.get_cmap("tab10")
    class_color = {name: palette(i % 10) for i, name in enumerate(class_names)}
    bar_colors = [class_color.get(best_class, "#666666") for best_class in best_classes]

    fig_h = max(4.5, 0.45 * len(top_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    bars = ax.barh(feature_labels, eta_vals, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax.set_title(f"{title}: Top {len(top_rows)} Features by eta^2")
    ax.set_xlabel("eta^2")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    for bar, best_class in zip(bars, best_classes):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x + max(eta_vals) * 0.015, y, best_class, va="center", fontsize=9)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, markerfacecolor=color, markeredgecolor="black", label=name)
        for name, color in class_color.items()
    ]
    ax.legend(handles=legend_handles, title="Best class", loc="lower right")
    ax.set_xlim(0, max(eta_vals) * 1.28 if eta_vals else 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(
    rows: list[dict],
    class_slugs: list[str],
    value_prefix: str,
    title: str,
    colorbar_label: str,
    output_path: Path,
    top_k: int,
    cmap: str,
    center: Optional[float] = None,
) -> None:
    top_rows = rows[:top_k]
    matrix = [[row[f"{value_prefix}_{slug}"] for slug in class_slugs] for row in top_rows]
    feature_labels = [f"f{int(row['feature_idx'])}" for row in top_rows]
    class_labels = [pretty_name(slug) for slug in class_slugs]

    if not matrix:
        return

    fig_w = max(6, 1.2 * len(class_slugs) + 3)
    fig_h = max(4.5, 0.45 * len(top_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmin = min(min(row) for row in matrix)
    vmax = max(max(row) for row in matrix)
    if center is not None:
        bound = max(abs(vmin), abs(vmax))
        norm = colors.TwoSlopeNorm(vmin=-bound, vcenter=center, vmax=bound)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels)

    for i in range(len(feature_labels)):
        for j in range(len(class_labels)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_scatter(rows: list[dict], title: str, output_path: Path, annotate_k: int) -> None:
    x = [row["active_rate_all"] for row in rows]
    y = [row["eta_sq"] for row in rows]
    c = [row["best_abs_r"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    scatter = ax.scatter(x, y, c=c, cmap="viridis", s=42, alpha=0.85, edgecolors="black", linewidths=0.3)
    ax.set_title(f"{title}: eta^2 vs Active Rate")
    ax.set_xlabel("active_all")
    ax.set_ylabel("eta^2")
    ax.grid(alpha=0.25, linestyle="--")

    for row in rows[:annotate_k]:
        ax.annotate(
            f"f{int(row['feature_idx'])}",
            (row["active_rate_all"], row["eta_sq"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("best |r|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_counts(rows: list[dict], class_slugs: list[str]) -> list[str]:
    lines = []
    for slug in class_slugs:
        key = f"active_rate_{slug}"
        avg_active = sum(row[key] for row in rows) / len(rows)
        lines.append(f"{pretty_name(slug)} avg active rate across features: {avg_active:.3f}")
    return lines


def make_plots(csv_path: Path, output_dir: Path, top_k: int, annotate_k: int) -> None:
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    with csv_path.open("r", newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames or []

    class_slugs = infer_class_slugs(fieldnames)
    class_names = [pretty_name(slug).upper() for slug in class_slugs]
    title = csv_path.stem.replace("sae_control_", "").replace("_", " ").title()

    output_dir.mkdir(parents=True, exist_ok=True)

    save_top_eta_plot(rows, class_names, title, output_dir / f"{csv_path.stem}_top_eta.png", top_k)
    save_heatmap(
        rows,
        class_slugs,
        "r",
        f"{title}: Top {min(top_k, len(rows))} Features by eta^2, class r",
        "r",
        output_dir / f"{csv_path.stem}_r_heatmap.png",
        top_k,
        cmap="RdBu_r",
        center=0.0,
    )
    save_heatmap(
        rows,
        class_slugs,
        "active_rate",
        f"{title}: Top {min(top_k, len(rows))} Features by eta^2, class active rate",
        "active rate",
        output_dir / f"{csv_path.stem}_active_heatmap.png",
        top_k,
        cmap="YlGnBu",
    )
    save_scatter(rows, title, output_dir / f"{csv_path.stem}_eta_vs_active.png", annotate_k)

    summary_path = output_dir / f"{csv_path.stem}_plot_summary.txt"
    summary_lines = [
        f"Source CSV: {csv_path}",
        f"Rows: {len(rows)}",
        f"Top eta^2 feature: f{int(rows[0]['feature_idx'])} ({rows[0]['eta_sq']:.5f})",
    ]
    summary_lines.extend(summarize_counts(rows, class_slugs))
    summary_path.write_text("\n".join(summary_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SAE control analysis CSV outputs.")
    parser.add_argument(
        "--input_csvs",
        nargs="+",
        required=True,
        help="One or more analysis CSV files produced by analyze_sae_control.py",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for generated figures")
    parser.add_argument("--top_k", type=int, default=15, help="Number of top eta^2 features to visualize")
    parser.add_argument("--annotate_k", type=int, default=12, help="Number of top eta^2 features to label in scatter plot")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for input_csv in args.input_csvs:
        make_plots(Path(input_csv), output_dir, args.top_k, args.annotate_k)


if __name__ == "__main__":
    main()
