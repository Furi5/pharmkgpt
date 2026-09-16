#!/usr/bin/env python3
"""Plot paired accuracy differences from completed results."""
import argparse
import os
import tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pharmkgpt-matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / "effect_size/results/canonical_effect_sizes_overall.csv"

def plot_overall_effects(effect_summary: pd.DataFrame, output_path: Path) -> None:
    overall = effect_summary[effect_summary["Category"] == "overall"].sort_values(
        "Accuracy_Difference", ascending=False
    )
    estimates = overall["Accuracy_Difference"].to_numpy() * 100
    lower = overall["Acc_Diff_95CI_Lower"].to_numpy() * 100
    upper = overall["Acc_Diff_95CI_Upper"].to_numpy() * 100
    y = np.arange(len(overall))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="o",
        color="#1f4e78",
        ecolor="#5b9bd5",
        capsize=4,
        linewidth=1.5,
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y, overall["Comparator"])
    ax.set_xlabel("PharmkGPT − comparator accuracy (percentage points, 95% CI)")
    ax.set_title("Paired effect sizes on the 1,045-question benchmark")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "figures/fig_paired_accuracy_differences.png")
    args = parser.parse_args()
    plot_overall_effects(pd.read_csv(args.input), args.output)
