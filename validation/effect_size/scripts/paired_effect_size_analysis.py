#!/usr/bin/env python3
"""Canonical paired effect-size analysis for the 1,045-question benchmark.

The manuscript's 95.62% PharmkGPT accuracy is the mean correctness over five
runs, not a majority-correct endpoint. This script keeps that estimand and uses
the question as the paired resampling unit. The primary effect size is the
absolute paired accuracy difference; relative error reduction is secondary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


EXP_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = EXP_DIR / "data" / "all_answer.csv"
RESULTS_DIR = EXP_DIR / "results"

SEED = 20260803
BOOTSTRAP_REPLICATES = 20_000
VALID_OPTIONS = {"A", "B", "C", "D"}

METHODS = OrderedDict(
    [
        ("PharmkGPT", [f"pharmkgpt-{i}" for i in range(1, 6)]),
        ("HippoRAG 2", [f"HippoRAG2-{i}" for i in range(1, 6)]),
        ("SemanticRAG", [f"SemanticRAG-{i}" for i in range(1, 6)]),
        ("LightRAG", [f"lightrag-{i}" for i in range(1, 6)]),
        ("Gemma3-27B", [f"Gemma3-27b-{i}" for i in range(1, 6)]),
        ("LLaMA3-8B", [f"llama3-8b-{i}" for i in range(1, 6)]),
        ("LLaMA2-13B", [f"llama2-13b-{i}" for i in range(1, 6)]),
        ("DeepSeek-32B", [f"DeepSeek-32b-{i}" for i in range(1, 6)]),
        ("Qwen-32B", [f"Qwen-32b-{i}" for i in range(1, 6)]),
    ]
)
CATEGORIES = OrderedDict(
    [
        ("overall", None),
        ("clinical", "clinical"),
        ("metabolism", "metabolism"),
        ("cell", "cell"),
        ("pathway", "pathway"),
        ("genetic", "genetic"),
    ]
)


def normalize(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def stable_rng(*parts: object) -> np.random.Generator:
    key = ":".join(str(part) for part in (SEED, *parts))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"question_id", "correct_option"}
    for columns in METHODS.values():
        required.update(columns)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input lacks required columns: {missing}")
    if data["question_id"].duplicated().any():
        duplicates = data.loc[data["question_id"].duplicated(), "question_id"].tolist()
        raise ValueError(f"Duplicate question IDs: {duplicates[:10]}")

    gold = normalize(data["correct_option"])
    invalid_gold = sorted(set(gold.dropna()) - VALID_OPTIONS)
    if gold.isna().any() or invalid_gold:
        raise ValueError(f"Invalid gold labels: missing={gold.isna().sum()}, values={invalid_gold}")

    data = data.copy()
    data["category"] = data["question_id"].astype(str).str.split("_").str[0].str.lower()
    unknown = sorted(set(data["category"]) - set(list(CATEGORIES)[1:]))
    if unknown:
        raise ValueError(f"Unknown question categories: {unknown}")
    return data


def correctness_matrices(data: pd.DataFrame):
    gold = normalize(data["correct_option"])
    matrices: dict[str, pd.DataFrame] = {}
    run_rows: list[dict] = []
    quality: dict[str, dict] = {}

    for method, columns in METHODS.items():
        predictions = data[columns].apply(normalize)
        correct = predictions.eq(gold, axis=0).fillna(False).astype(float)
        matrices[method] = correct

        method_missing = 0
        method_invalid_nonmissing = 0
        for run_number, column in enumerate(columns, 1):
            values = predictions[column]
            missing = int(values.isna().sum())
            invalid_nonmissing = int((values.notna() & ~values.isin(VALID_OPTIONS)).sum())
            method_missing += missing
            method_invalid_nonmissing += invalid_nonmissing
            run_rows.append(
                {
                    "Method": method,
                    "Run": run_number,
                    "Column": column,
                    "Accuracy": float(correct[column].mean()),
                    "Correct": int(correct[column].sum()),
                    "Total": len(data),
                    "Missing_Response": missing,
                    "Invalid_Nonmissing_Response": invalid_nonmissing,
                }
            )
        quality[method] = {
            "missing_responses": method_missing,
            "invalid_nonmissing_responses": method_invalid_nonmissing,
            "scoring_rule": "Missing and invalid responses are counted as incorrect.",
        }
    return matrices, pd.DataFrame(run_rows), quality


def bootstrap_mean_ci(values: np.ndarray, key: str, replicates: int) -> tuple[float, float]:
    rng = stable_rng("mean", key, replicates)
    n = len(values)
    estimates = np.empty(replicates, dtype=float)
    chunk = 2_000
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, n, size=(stop - start, n))
        estimates[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def bootstrap_pair_ci(
    reference: np.ndarray,
    comparator: np.ndarray,
    key: str,
    replicates: int,
) -> dict[str, float]:
    rng = stable_rng("pair", key, replicates)
    n = len(reference)
    differences = np.empty(replicates, dtype=float)
    error_reductions = np.empty(replicates, dtype=float)
    chunk = 2_000
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, n, size=(stop - start, n))
        ref_boot = reference[indices].mean(axis=1)
        comp_boot = comparator[indices].mean(axis=1)
        differences[start:stop] = ref_boot - comp_boot
        denominator = 1.0 - comp_boot
        error_reductions[start:stop] = np.divide(
            ref_boot - comp_boot,
            denominator,
            out=np.full_like(denominator, np.nan),
            where=denominator > 0,
        )
    diff_lower, diff_upper = np.quantile(differences, [0.025, 0.975])
    valid_error_reductions = error_reductions[np.isfinite(error_reductions)]
    error_lower, error_upper = np.quantile(valid_error_reductions, [0.025, 0.975])
    return {
        "diff_lower": float(diff_lower),
        "diff_upper": float(diff_upper),
        "error_reduction_lower": float(error_lower),
        "error_reduction_upper": float(error_upper),
    }


def holm_adjust(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def analyze(data: pd.DataFrame, replicates: int):
    matrices, run_summary, quality = correctness_matrices(data)
    per_question = {
        method: matrix.mean(axis=1).to_numpy(dtype=float)
        for method, matrix in matrices.items()
    }
    accuracy_rows: list[dict] = []
    effect_rows: list[dict] = []

    for category, category_value in CATEGORIES.items():
        mask = np.ones(len(data), dtype=bool) if category_value is None else data["category"].eq(category_value).to_numpy()
        n_questions = int(mask.sum())

        for method in METHODS:
            values = per_question[method][mask]
            lower, upper = bootstrap_mean_ci(values, f"{category}:{method}", replicates)
            accuracy_rows.append(
                {
                    "Category": category,
                    "Method": method,
                    "Accuracy": float(values.mean()),
                    "Accuracy_95CI_Lower": lower,
                    "Accuracy_95CI_Upper": upper,
                    "N_Questions": n_questions,
                    "Runs_Per_Method": len(METHODS[method]),
                }
            )

        reference = per_question["PharmkGPT"][mask]
        reference_accuracy = float(reference.mean())
        category_effects: list[dict] = []
        for comparator in list(METHODS)[1:]:
            comparison = per_question[comparator][mask]
            comparator_accuracy = float(comparison.mean())
            difference = reference - comparison
            point_difference = float(difference.mean())
            paired_ci = bootstrap_pair_ci(
                reference,
                comparison,
                f"{category}:PharmkGPT:{comparator}",
                replicates,
            )
            comparator_error = 1.0 - comparator_accuracy
            relative_error_reduction = point_difference / comparator_error if comparator_error > 0 else np.nan
            p_value = float(stats.ttest_rel(reference, comparison).pvalue)
            category_effects.append(
                {
                    "Category": category,
                    "Reference": "PharmkGPT",
                    "Comparator": comparator,
                    "PharmkGPT_Accuracy": reference_accuracy,
                    "Comparator_Accuracy": comparator_accuracy,
                    "Accuracy_Difference": point_difference,
                    "Acc_Diff_95CI_Lower": paired_ci["diff_lower"],
                    "Acc_Diff_95CI_Upper": paired_ci["diff_upper"],
                    "Relative_Error_Reduction": float(relative_error_reduction),
                    "Relative_Error_Reduction_95CI_Lower": paired_ci["error_reduction_lower"],
                    "Relative_Error_Reduction_95CI_Upper": paired_ci["error_reduction_upper"],
                    "Paired_t_p_value": p_value,
                    "N_Questions": n_questions,
                    "Runs_Per_Method": len(METHODS[comparator]),
                }
            )
        category_frame = pd.DataFrame(category_effects)
        category_frame["Holm_Adjusted_p_value"] = holm_adjust(category_frame["Paired_t_p_value"])
        category_frame["Holm_Significant_0_05"] = category_frame["Holm_Adjusted_p_value"] < 0.05
        effect_rows.extend(category_frame.to_dict("records"))

    accuracy_summary = pd.DataFrame(accuracy_rows).sort_values(
        ["Category", "Accuracy"], ascending=[True, False]
    )
    effect_summary = pd.DataFrame(effect_rows).sort_values(
        ["Category", "Comparator_Accuracy"], ascending=[True, False]
    )
    return accuracy_summary, effect_summary, run_summary, quality






def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("Use at least 1,000 bootstrap replicates for reviewer-facing results.")

    data = load_and_validate(args.input)
    accuracy_summary, effect_summary, run_summary, quality = analyze(
        data, args.bootstrap_replicates
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    accuracy_summary.to_csv(RESULTS_DIR / "method_accuracy_summary.csv", index=False)
    run_summary.to_csv(RESULTS_DIR / "run_accuracy_summary.csv", index=False)
    effect_summary[effect_summary["Category"] == "overall"].to_csv(
        RESULTS_DIR / "canonical_effect_sizes_overall.csv", index=False
    )
    effect_summary[effect_summary["Category"] != "overall"].to_csv(
        RESULTS_DIR / "canonical_effect_sizes_by_domain.csv", index=False
    )

    closest = effect_summary[effect_summary["Category"] == "overall"].sort_values(
        "Comparator_Accuracy", ascending=False
    ).head(1)

    input_quality = {
        "input_rows": len(data),
        "duplicate_question_ids": int(data["question_id"].duplicated().sum()),
        "category_counts": data["category"].value_counts().sort_index().to_dict(),
        "method_response_quality": quality,
    }
    (RESULTS_DIR / "input_quality_report.json").write_text(
        json.dumps(input_quality, indent=2), encoding="utf-8"
    )
    metadata = {
        "input": "data/all_answer.csv",
        "input_sha256": sha256(args.input),
        "n_questions": len(data),
        "runs_per_method": 5,
        "accuracy_estimand": "Mean correctness across five runs and all questions.",
        "pairing_unit": "Question",
        "primary_effect_size": "Paired absolute accuracy difference",
        "secondary_effect_size": "Relative error reduction",
        "confidence_interval": "Nonparametric percentile bootstrap clustered by question",
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": SEED,
        "multiplicity": "Holm adjustment across eight PharmkGPT comparisons within each category",
        "missing_response_policy": "Counted as incorrect",
    }
    (RESULTS_DIR / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


    closest_row = closest.iloc[0]
    print(f"PharmkGPT mean accuracy: {closest_row['PharmkGPT_Accuracy']:.6f}")
    print(
        f"Closest comparator: {closest_row['Comparator']} "
        f"({closest_row['Comparator_Accuracy']:.6f})"
    )
    print(
        f"Paired difference: {closest_row['Accuracy_Difference']:.6f} "
        f"[{closest_row['Acc_Diff_95CI_Lower']:.6f}, "
        f"{closest_row['Acc_Diff_95CI_Upper']:.6f}]"
    )


if __name__ == "__main__":
    main()
