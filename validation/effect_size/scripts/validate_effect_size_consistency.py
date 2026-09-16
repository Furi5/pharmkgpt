#!/usr/bin/env python3
"""Validate the canonical effect-size outputs and reviewer-facing claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


EXP_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = EXP_DIR / "data" / "all_answer.csv"
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR.parent / "plot/effect_size/figures"
REPORT_PATH = EXP_DIR / "consistency_report.json"

EXPECTED_INPUT_SHA256 = "7acb70260cfc2a90540b006269eca7a1794a64a8b82244688ac679e95705794d"
EXPECTED_N = 1045
EXPECTED_PHARM_ACCURACY = 0.9561722488038278
EXPECTED_CLOSEST_COMPARATOR = "HippoRAG 2"
EXPECTED_CLOSEST_ACCURACY = 0.8838277511961723
EXPECTED_CLOSEST_DIFFERENCE = 0.0723444976076556


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def near(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return abs(actual - expected) <= tolerance


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    overall = pd.read_csv(RESULTS_DIR / "canonical_effect_sizes_overall.csv")
    domains = pd.read_csv(RESULTS_DIR / "canonical_effect_sizes_by_domain.csv")
    accuracies = pd.read_csv(RESULTS_DIR / "method_accuracy_summary.csv")
    runs = pd.read_csv(RESULTS_DIR / "run_accuracy_summary.csv")
    metadata = json.loads((RESULTS_DIR / "analysis_metadata.json").read_text(encoding="utf-8"))

    closest = overall.sort_values("Comparator_Accuracy", ascending=False).iloc[0]
    pharm = accuracies[
        (accuracies["Category"] == "overall") & (accuracies["Method"] == "PharmkGPT")
    ].iloc[0]
    expected_domains = {"clinical", "metabolism", "cell", "pathway", "genetic"}

    checks = {
        "input_sha256": sha256(DATA_PATH) == EXPECTED_INPUT_SHA256,
        "input_rows": len(data) == EXPECTED_N,
        "unique_question_ids": data["question_id"].nunique() == EXPECTED_N,
        "overall_comparisons": len(overall) == 8,
        "domain_comparisons": len(domains) == 40,
        "domain_names": set(domains["Category"]) == expected_domains,
        "method_accuracy_rows": len(accuracies) == 54,
        "run_rows": len(runs) == 45,
        "pharm_accuracy_matches_manuscript_estimand": near(
            float(pharm["Accuracy"]), EXPECTED_PHARM_ACCURACY
        ),
        "closest_comparator": closest["Comparator"] == EXPECTED_CLOSEST_COMPARATOR,
        "closest_accuracy": near(
            float(closest["Comparator_Accuracy"]), EXPECTED_CLOSEST_ACCURACY
        ),
        "closest_difference": near(
            float(closest["Accuracy_Difference"]), EXPECTED_CLOSEST_DIFFERENCE
        ),
        "paired_ci_contains_point": float(closest["Acc_Diff_95CI_Lower"])
        < float(closest["Accuracy_Difference"])
        < float(closest["Acc_Diff_95CI_Upper"]),
        "paired_ci_above_zero": float(closest["Acc_Diff_95CI_Lower"]) > 0,
        "holm_adjusted": overall["Holm_Adjusted_p_value"].notna().all(),
        "all_overall_comparisons_significant": overall["Holm_Significant_0_05"].all(),
        "metadata_pairing_unit": metadata.get("pairing_unit") == "Question",
        "metadata_bootstrap_replicates": int(metadata.get("bootstrap_replicates", 0)) >= 10000,
        "canonical_figure": (FIGURES_DIR / "fig_paired_accuracy_differences.png").exists(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "experiment": "effect_size",
        "status": "complete" if all(checks.values()) else "failed",
        "canonical_estimand": "mean correctness across five runs",
        "actual": {
            "n_questions": len(data),
            "pharmkgpt_accuracy": float(pharm["Accuracy"]),
            "closest_comparator": closest["Comparator"],
            "closest_comparator_accuracy": float(closest["Comparator_Accuracy"]),
            "accuracy_difference": float(closest["Accuracy_Difference"]),
            "accuracy_difference_95ci": [
                float(closest["Acc_Diff_95CI_Lower"]),
                float(closest["Acc_Diff_95CI_Upper"]),
            ],
            "relative_error_reduction": float(closest["Relative_Error_Reduction"]),
            "relative_error_reduction_95ci": [
                float(closest["Relative_Error_Reduction_95CI_Lower"]),
                float(closest["Relative_Error_Reduction_95CI_Upper"]),
            ],
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
