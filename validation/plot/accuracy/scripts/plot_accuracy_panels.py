#!/usr/bin/env python3
"""Render the six accuracy panels as independent publication figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SOURCE = EXPERIMENT_DIR / "results" / "domain_accuracy_summary.csv"
SIGNIFICANCE_SOURCE = EXPERIMENT_DIR / "results" / "holm_significance_summary.csv"
FIGURE_DIR = EXPERIMENT_DIR / "figures"
DOMAINS = ("Overall", "Clinical", "Genetic", "Cell", "Pathway", "Metabolism")

COLORS = {
    "Qwen3-32B": "#d0d0d0",
    "DeepSeek-R1-32B": "#fac681",
    "LLaMA2-13B": "#ffae4a",
    "LLaMA3-8B": "#b9d36e",
    "Gemma3-27B": "#96c11f",
    "LightRAG": "#83cbcd",
    "SemanticRAG": "#08a7b4",
    "HippoRAG 2": "#ed8997",
    "PharmkGPT": "#f25570",
}


def load_data() -> tuple[dict[str, list[dict[str, float | str]]], dict[str, dict[str, str]]]:
    data: dict[str, list[dict[str, float | str]]] = {domain: [] for domain in DOMAINS}
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            data[row["domain"]].append(
                {
                    "method": row["method"],
                    "mean": float(row["mean_accuracy_percent"]),
                    "sd": float(row["sd_accuracy_percent"]),
                }
            )
    for domain in DOMAINS:
        # Python's stable sort preserves the reference method order for exact ties.
        data[domain].sort(key=lambda item: float(item["mean"]))

    markers: dict[str, dict[str, str]] = {domain: {} for domain in DOMAINS}
    with SIGNIFICANCE_SOURCE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            markers[row["domain"]][row["comparator"]] = row["marker"]
    return data, markers


def draw_reference_brackets(
    ax: plt.Axes,
    markers: list[str],
    last_x: int,
    *,
    font_scale: float = 1.0,
    axis_linewidth: float = 1.0,
    reference_sizing: bool = False,
) -> None:
    levels = (126.0, 119.0, 112.0, 105.0)
    for index, marker in enumerate(markers):
        left = index
        y = levels[index // 2]
        ax.plot(
            [left, left, last_x, last_x],
            [y - 1.7, y, y, y - 1.7],
            color="#333333",
            linewidth=axis_linewidth,
            solid_capstyle="butt",
            zorder=1,
        )
        marker_size = (7 if reference_sizing else 9) * font_scale
        ax.text(left + 0.30, y + 0.55, marker, ha="center", va="bottom", fontsize=marker_size, fontweight="bold")


def plot_domain(
    domain: str,
    records: list[dict[str, float | str]],
    markers: list[str],
    *,
    output_dir: Path = FIGURE_DIR,
    font_scale: float = 1.0,
    axis_linewidth: float = 1.25,
    sparse_ticks: bool = False,
    filename_suffix: str = "",
) -> None:
    if font_scale <= 0 or axis_linewidth <= 0:
        raise ValueError("font_scale and axis_linewidth must be positive")
    methods = [str(row["method"]) for row in records]
    means = np.array([float(row["mean"]) for row in records])
    sds = np.array([float(row["sd"]) for row in records])
    x = np.arange(len(records))

    size_factor = 1.0 + max(font_scale - 1.0, 0.0) * 0.34
    reference_sizing = sparse_ticks or font_scale != 1.0
    title_size = (11 if reference_sizing else 17) * font_scale
    axis_label_size = (9 if reference_sizing else 14) * font_scale
    tick_size = (8 if reference_sizing else 11) * font_scale
    value_size = (7 if reference_sizing else 9) * font_scale
    fig, ax = plt.subplots(figsize=(7.4 * size_factor, 5.3 * size_factor))
    bars = ax.bar(
        x,
        means,
        width=0.70,
        color=[COLORS[method] for method in methods],
        edgecolor="#111111",
        linewidth=axis_linewidth,
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=sds,
        fmt="none",
        ecolor="#555555",
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.9,
        zorder=3,
    )
    for bar, mean, sd in zip(bars, means, sds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + sd + 1.25,
            f"{mean:.2f}%",
            ha="center",
            va="bottom",
            fontsize=value_size,
            fontweight="bold",
        )

    draw_reference_brackets(
        ax,
        markers,
        len(records) - 1,
        font_scale=font_scale,
        axis_linewidth=axis_linewidth,
        reference_sizing=reference_sizing,
    )
    ax.set_title(domain, fontsize=title_size, fontweight="bold", pad=6)
    ax.set_ylabel("Accuracy (%)", fontsize=axis_label_size, fontweight="bold")
    ax.set_xticks(x, labels=methods, rotation=90)
    ax.set_ylim(0, 130)
    ax.set_yticks((0, 50, 100) if sparse_ticks else np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.tick_params(axis="x", length=0, labelsize=tick_size, pad=4)
    ax.tick_params(axis="y", length=5, width=axis_linewidth, labelsize=tick_size)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)
    ax.grid(False)

    fig.subplots_adjust(left=0.12, right=0.99, top=0.91, bottom=0.27)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{domain.lower()}_accuracy_comparison{filename_suffix}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.png')}")
    print(f"Wrote {stem.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--axis-linewidth", type=float, default=1.25)
    parser.add_argument("--sparse-ticks", action="store_true")
    parser.add_argument("--filename-suffix", default="")
    args = parser.parse_args()
    data, markers = load_data()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        method_order = [str(row["method"]) for row in data[domain]]
        domain_markers = [markers[domain][method] for method in method_order[:-1]]
        plot_domain(
            domain,
            data[domain],
            domain_markers,
            output_dir=args.output_dir,
            font_scale=args.font_scale,
            axis_linewidth=args.axis_linewidth,
            sparse_ticks=args.sparse_ticks,
            filename_suffix=args.filename_suffix,
        )


if __name__ == "__main__":
    main()
