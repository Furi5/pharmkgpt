#!/usr/bin/env python3
"""Render each exact-source Recall@K domain as an individual figure."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXPERIMENT_DIR / "results" / "topk_hit_rates.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "figures"
DOMAINS = ("Overall", "Clinical", "Genetic", "Cell", "Pathway", "Metabolism")
MODELS = ("LightRAG", "HippoRAG2", "SemanticRAG", "PharmkGPT")
STYLES = {
    "LightRAG": {"color": "#FFAA4A", "marker": "o", "linestyle": "-"},
    "HippoRAG2": {"color": "#02A9B7", "marker": "s", "linestyle": "--"},
    "SemanticRAG": {"color": "#84BD00", "marker": "^", "linestyle": "-."},
    "PharmkGPT": {"color": "#F45B73", "marker": "D", "linestyle": "-"},
}


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_results(path: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["domain"], row["model"])].append(
                {
                    "top_k": int(row["top_k"]),
                    "mean": float(row["mean_exact_source_recall_percent"]),
                    "sd": float(row["sd_exact_source_recall_percent"]),
                }
            )
    for key, values in grouped.items():
        values.sort(key=lambda item: item["top_k"])
        if [item["top_k"] for item in values] != list(range(1, 11)):
            raise ValueError(f"Incomplete Top-K series: {key}")
    expected_keys = {(domain, model) for domain in DOMAINS for model in MODELS}
    if set(grouped) != expected_keys:
        raise ValueError("Missing or unexpected domain/model series")
    return grouped


def plot_domain(
    domain: str,
    grouped: dict[tuple[str, str], list[dict[str, float]]],
    output_dir: Path,
    *,
    font_scale: float = 1.0,
    axis_linewidth: float = 1.15,
    sparse_ticks: bool = False,
    bold_text: bool = False,
    filename_suffix: str = "",
) -> tuple[Path, Path]:
    if font_scale <= 0 or axis_linewidth <= 0:
        raise ValueError("font_scale and axis_linewidth must be positive")
    size_factor = 1.0 + max(font_scale - 1.0, 0.0) * 0.30
    title_size = (11 if bold_text else 15) * font_scale
    axis_label_size = (9 if bold_text else 13) * font_scale
    tick_size = 8 * font_scale if bold_text else 10 * font_scale
    legend_size = 8 * font_scale if bold_text else 9 * font_scale
    fig, ax = plt.subplots(figsize=(5.15 * size_factor, 4.35 * size_factor))
    for model in MODELS:
        rows = grouped[(domain, model)]
        x = [int(row["top_k"]) for row in rows]
        means = [row["mean"] for row in rows]
        errors = [row["sd"] for row in rows]
        ax.errorbar(
            x,
            means,
            yerr=errors,
            label=model,
            linewidth=2.0 * min(font_scale, 1.6),
            markersize=5.0 * min(font_scale, 1.6),
            markeredgewidth=0,
            capsize=2.0,
            elinewidth=1.1,
            zorder=3,
            **STYLES[model],
        )

    ax.set_title(domain, fontsize=title_size, fontweight="bold", pad=8)
    ax.set_xlabel("Top-K", fontsize=axis_label_size, fontweight="bold")
    ax.set_ylabel(
        "Exact-source Recall (%)", fontsize=axis_label_size, fontweight="bold"
    )
    ax.set_xlim(0.55, 10.45)
    ax.set_ylim(40, 100)
    ax.set_xticks((1, 3, 5, 7, 10) if sparse_ticks else range(1, 11))
    ax.set_yticks((40, 60, 80, 100) if sparse_ticks else range(40, 101, 10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)
    ax.tick_params(
        axis="both", width=axis_linewidth, length=5, labelsize=tick_size
    )
    if bold_text:
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontweight("bold")
    legend_prop = {
        "size": legend_size,
        "weight": "bold" if bold_text else "normal",
    }
    ax.legend(
        loc="lower right", frameon=False, handlelength=2.5, prop=legend_prop
    )
    fig.tight_layout(pad=0.7)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{domain.lower()}_topk_hit_rate{filename_suffix}"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--axis-linewidth", type=float, default=1.15)
    parser.add_argument("--sparse-ticks", action="store_true")
    parser.add_argument("--bold-text", action="store_true")
    parser.add_argument("--filename-suffix", default="")
    args = parser.parse_args()
    grouped = read_results(args.input)
    for domain in DOMAINS:
        png_path, pdf_path = plot_domain(
            domain,
            grouped,
            args.output_dir,
            font_scale=args.font_scale,
            axis_linewidth=args.axis_linewidth,
            sparse_ticks=args.sparse_ticks,
            bold_text=args.bold_text,
            filename_suffix=args.filename_suffix,
        )
        print(f"Wrote {png_path}")
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
