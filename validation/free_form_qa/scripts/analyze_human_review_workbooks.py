#!/usr/bin/env python3
"""Analyze the completed two-reviewer free-form QA score tables.

The script keeps question IDs as the paired/bootstrap unit and uses the hidden
answer key only after scoring to map anonymized answer A/B back to each system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_ROOT = (
    EXPERIMENT_ROOT / "human_review"
)
FULL_MODEL = "PharmKGPT Full KG"
VANILLA_MODEL = "PharmKGPT Vanilla RAG"
MODELS = (FULL_MODEL, VANILLA_MODEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--reviewer-1", default="reviewer1.csv")
    parser.add_argument("--reviewer-2", default="reviewer2.csv")
    parser.add_argument("--answer-key", default="human_review_answer_key.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--permutation-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    review_root = args.review_root.resolve()
    key = pd.read_csv(review_root / args.answer_key, encoding="utf-8-sig")
    required_key = {
        "question_id",
        "answer_a_model",
        "answer_a_variant",
        "answer_a_gold_rank",
        "answer_a_cited_gold",
        "answer_b_model",
        "answer_b_variant",
        "answer_b_gold_rank",
        "answer_b_cited_gold",
    }
    missing_key = required_key - set(key.columns)
    if missing_key:
        raise ValueError(f"Answer key is missing columns: {sorted(missing_key)}")

    long_frames: list[pd.DataFrame] = []
    preference_frames: list[pd.DataFrame] = []
    reviewer_files = (
        ("Reviewer 1", args.reviewer_1),
        ("Reviewer 2", args.reviewer_2),
    )
    score_columns = (
        "answer_a_correctness_0_2",
        "answer_b_correctness_0_2",
        "answer_a_evidence_support_0_2",
        "answer_b_evidence_support_0_2",
        "preferred_answer_a_b_tie",
    )

    for reviewer, filename in reviewer_files:
        review = pd.read_csv(review_root / filename)
        review.columns = [str(column).strip() for column in review.columns]
        missing = {"question_id", *score_columns} - set(review.columns)
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
        if review["question_id"].duplicated().any():
            raise ValueError(f"{filename} contains duplicate question IDs")
        if review[list(score_columns)].isna().any().any():
            raise ValueError(f"{filename} contains incomplete scores")

        merged = review.merge(key, on="question_id", validate="one_to_one")
        if len(merged) != len(review) or len(merged) != len(key):
            raise ValueError(f"{filename} and the answer key do not contain identical IDs")

        for side in ("a", "b"):
            frame = pd.DataFrame(
                {
                    "reviewer": reviewer,
                    "question_id": merged["question_id"],
                    "model": merged[f"answer_{side}_model"],
                    "variant": merged[f"answer_{side}_variant"],
                    "correctness": merged[f"answer_{side}_correctness_0_2"].astype(int),
                    "evidence_support": merged[
                        f"answer_{side}_evidence_support_0_2"
                    ].astype(int),
                    "gold_rank": merged[f"answer_{side}_gold_rank"],
                    "cited_gold": merged[f"answer_{side}_cited_gold"].astype(bool),
                }
            )
            for column in ("correctness", "evidence_support"):
                invalid = ~frame[column].isin((0, 1, 2))
                if invalid.any():
                    raise ValueError(f"{filename} contains an invalid {column} score")
            long_frames.append(frame)

        preferences: list[str] = []
        for _, row in merged.iterrows():
            value = str(row["preferred_answer_a_b_tie"]).strip().lower()
            if value == "tie":
                preferences.append("Tie")
            elif value in {"a", "b"}:
                preferences.append(str(row[f"answer_{value}_model"]))
            else:
                raise ValueError(f"{filename} contains invalid preference: {value!r}")
        preference_frames.append(
            pd.DataFrame(
                {
                    "reviewer": reviewer,
                    "question_id": merged["question_id"],
                    "preference": preferences,
                }
            )
        )

    long = pd.concat(long_frames, ignore_index=True)
    preferences = pd.concat(preference_frames, ignore_index=True)
    if set(long["model"].unique()) != set(MODELS):
        raise ValueError(f"Unexpected systems in answer key: {sorted(long['model'].unique())}")
    return long, preferences


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    draws: list[np.ndarray] = []
    batch_size = 5_000
    for start in range(0, samples, batch_size):
        batch = min(batch_size, samples - start)
        indexes = rng.integers(0, n, size=(batch, n))
        draws.append(values[indexes].mean(axis=1))
    lower, upper = np.quantile(np.concatenate(draws), (0.025, 0.975))
    return float(lower), float(upper)


def sign_flip_pvalue(
    values: np.ndarray,
    rng: np.random.Generator,
    simulation_samples: int,
) -> tuple[float, str, int]:
    values = np.asarray(values, dtype=float)
    nonzero = np.abs(values[values != 0])
    observed = abs(float(values.sum()))
    if len(nonzero) <= 22:
        total = 1 << len(nonzero)
        extreme = 0
        batch_size = 200_000
        bit_positions = np.arange(len(nonzero), dtype=np.uint64)
        for start in range(0, total, batch_size):
            identifiers = np.arange(
                start, min(start + batch_size, total), dtype=np.uint64
            )[:, None]
            bits = ((identifiers >> bit_positions) & 1).astype(np.int8)
            sums = ((2 * bits - 1) * nonzero).sum(axis=1)
            extreme += int(np.count_nonzero(np.abs(sums) >= observed - 1e-12))
        return extreme / total, "exact", total

    signs = rng.choice((-1, 1), size=(simulation_samples, len(nonzero)))
    sums = (signs * nonzero).sum(axis=1)
    extreme = int(np.count_nonzero(np.abs(sums) >= observed - 1e-12))
    return (
        (extreme + 1) / (simulation_samples + 1),
        "monte_carlo",
        simulation_samples,
    )


def paired_metric(
    long: pd.DataFrame,
    score_column: str,
    transform: Callable[[pd.Series], pd.Series],
    rng: np.random.Generator,
    bootstrap_samples: int,
    permutation_samples: int,
) -> dict[str, float | int | str | list[float]]:
    data = long.assign(value=transform(long[score_column]).astype(float))
    paired = data.pivot(
        index=["question_id", "reviewer"], columns="model", values="value"
    )
    question_means = paired.groupby("question_id")[[*MODELS]].mean()
    differences = question_means[FULL_MODEL] - question_means[VANILLA_MODEL]
    ci = bootstrap_mean_ci(differences.to_numpy(), rng, bootstrap_samples)
    p_value, p_method, p_samples = sign_flip_pvalue(
        differences.to_numpy(), rng, permutation_samples
    )
    return {
        "full": float(question_means[FULL_MODEL].mean()),
        "vanilla": float(question_means[VANILLA_MODEL].mean()),
        "delta_full_minus_vanilla": float(differences.mean()),
        "paired_bootstrap_95ci": [ci[0], ci[1]],
        "sign_flip_p": float(p_value),
        "sign_flip_method": p_method,
        "sign_flip_samples": int(p_samples),
        "full_better_questions": int((differences > 0).sum()),
        "same_questions": int((differences == 0).sum()),
        "vanilla_better_questions": int((differences < 0).sum()),
    }


def model_metrics(long: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for model in MODELS:
        group = long[long["model"] == model]
        results[model] = {
            "reviewer_ratings": int(len(group)),
            "mean_correctness_0_2": float(group["correctness"].mean()),
            "fully_correct_count": int((group["correctness"] == 2).sum()),
            "fully_correct_rate": float((group["correctness"] == 2).mean()),
            "at_least_partly_correct_count": int((group["correctness"] >= 1).sum()),
            "at_least_partly_correct_rate": float(
                (group["correctness"] >= 1).mean()
            ),
            "mean_evidence_support_0_2": float(group["evidence_support"].mean()),
            "fully_supported_count": int((group["evidence_support"] == 2).sum()),
            "fully_supported_rate": float((group["evidence_support"] == 2).mean()),
            "at_least_partly_supported_count": int(
                (group["evidence_support"] >= 1).sum()
            ),
            "at_least_partly_supported_rate": float(
                (group["evidence_support"] >= 1).mean()
            ),
        }
    return results


def reviewer_metrics(long: pd.DataFrame, preferences: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for reviewer in sorted(long["reviewer"].unique()):
        reviewer_pref = preferences[preferences["reviewer"] == reviewer]
        for model in MODELS:
            group = long[(long["reviewer"] == reviewer) & (long["model"] == model)]
            rows.append(
                {
                    "reviewer": reviewer,
                    "model": model,
                    "questions": int(len(group)),
                    "mean_correctness_0_2": float(group["correctness"].mean()),
                    "fully_correct_rate": float((group["correctness"] == 2).mean()),
                    "mean_evidence_support_0_2": float(
                        group["evidence_support"].mean()
                    ),
                    "fully_supported_rate": float(
                        (group["evidence_support"] == 2).mean()
                    ),
                    "preference_wins": int((reviewer_pref["preference"] == model).sum()),
                    "ties": int((reviewer_pref["preference"] == "Tie").sum()),
                }
            )
    return rows


def retrieval_metrics(long: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    # Retrieval/citation metadata are duplicated across reviewers; deduplicate
    # before computing system-level automatic rates.
    deduplicated = long.drop_duplicates(["question_id", "model"])
    output: dict[str, dict[str, float | int]] = {}
    for model in MODELS:
        group = deduplicated[deduplicated["model"] == model]
        rank = group["gold_rank"]
        output[model] = {
            "questions": int(len(group)),
            "exact_source_recall_at_1": float((rank <= 1).mean()),
            "exact_source_recall_at_2": float((rank <= 2).mean()),
            "exact_source_recall_at_5": float((rank <= 5).mean()),
            "exact_source_recall_at_10": float((rank <= 10).mean()),
            "answer_cited_gold_pmid_rate": float(group["cited_gold"].mean()),
        }
    return output


def preference_metrics(
    preferences: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_samples: int,
    permutation_samples: int,
) -> dict:
    encoded = preferences.copy()
    for label in (*MODELS, "Tie"):
        encoded[label] = (encoded["preference"] == label).astype(float)
    question_rates = encoded.groupby("question_id")[[*MODELS, "Tie"]].mean()
    difference = question_rates[FULL_MODEL] - question_rates[VANILLA_MODEL]
    delta_ci = bootstrap_mean_ci(difference.to_numpy(), rng, bootstrap_samples)
    p_value, p_method, p_samples = sign_flip_pvalue(
        difference.to_numpy(), rng, permutation_samples
    )

    full_wins = int((preferences["preference"] == FULL_MODEL).sum())
    vanilla_wins = int((preferences["preference"] == VANILLA_MODEL).sum())
    ties = int((preferences["preference"] == "Tie").sum())

    paired_rates = question_rates[[FULL_MODEL, VANILLA_MODEL]].to_numpy()
    n = len(paired_rates)
    shares: list[np.ndarray] = []
    batch_size = 5_000
    for start in range(0, bootstrap_samples, batch_size):
        batch = min(batch_size, bootstrap_samples - start)
        indexes = rng.integers(0, n, size=(batch, n))
        totals = paired_rates[indexes].sum(axis=1)
        shares.append(totals[:, 0] / totals.sum(axis=1))
    share_ci = np.quantile(np.concatenate(shares), (0.025, 0.975))

    return {
        "reviewer_preferences": int(len(preferences)),
        "full_wins": full_wins,
        "vanilla_wins": vanilla_wins,
        "ties": ties,
        "full_win_rate_all_ratings": full_wins / len(preferences),
        "vanilla_win_rate_all_ratings": vanilla_wins / len(preferences),
        "tie_rate": ties / len(preferences),
        "full_share_among_non_ties": full_wins / (full_wins + vanilla_wins),
        "full_share_among_non_ties_bootstrap_95ci": [
            float(share_ci[0]),
            float(share_ci[1]),
        ],
        "full_minus_vanilla_preference_rate": float(difference.mean()),
        "full_minus_vanilla_bootstrap_95ci": [delta_ci[0], delta_ci[1]],
        "sign_flip_p": float(p_value),
        "sign_flip_method": p_method,
        "sign_flip_samples": int(p_samples),
    }


def agreement_metrics(long: pd.DataFrame, preferences: pd.DataFrame) -> dict:
    answer_agreement: dict[str, dict[str, float | int]] = {}
    for score in ("correctness", "evidence_support"):
        paired = long.pivot(
            index=["question_id", "model"], columns="reviewer", values=score
        ).sort_index()
        first = paired["Reviewer 1"].to_numpy()
        second = paired["Reviewer 2"].to_numpy()
        answer_agreement[score] = {
            "answer_pairs": int(len(first)),
            "exact_agreement": float((first == second).mean()),
            "unweighted_cohen_kappa": float(cohen_kappa_score(first, second)),
            "quadratic_weighted_cohen_kappa": float(
                cohen_kappa_score(first, second, weights="quadratic")
            ),
        }

    paired_preference = preferences.pivot(
        index="question_id", columns="reviewer", values="preference"
    ).sort_index()
    first_pref = paired_preference["Reviewer 1"]
    second_pref = paired_preference["Reviewer 2"]
    answer_agreement["preference"] = {
        "question_pairs": int(len(first_pref)),
        "exact_agreement": float((first_pref == second_pref).mean()),
        "unweighted_cohen_kappa": float(cohen_kappa_score(first_pref, second_pref)),
    }
    return answer_agreement


def write_outputs(
    output_dir: Path,
    summary: dict,
    paired_rows: list[dict],
    reviewer_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "human_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(paired_rows).to_csv(
        output_dir / "human_evaluation_paired_metrics.csv", index=False
    )
    pd.DataFrame(reviewer_rows).to_csv(
        output_dir / "human_evaluation_reviewer_metrics.csv", index=False
    )


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (args.review_root.resolve() / "results")
    )
    long, preferences = load_inputs(args)
    rng = np.random.default_rng(args.seed)

    paired_definitions = (
        ("mean_correctness_0_2", "correctness", lambda value: value),
        ("fully_correct_rate", "correctness", lambda value: value == 2),
        (
            "at_least_partly_correct_rate",
            "correctness",
            lambda value: value >= 1,
        ),
        ("mean_evidence_support_0_2", "evidence_support", lambda value: value),
        ("fully_supported_rate", "evidence_support", lambda value: value == 2),
        (
            "at_least_partly_supported_rate",
            "evidence_support",
            lambda value: value >= 1,
        ),
    )
    paired: dict[str, dict] = {}
    paired_rows: list[dict] = []
    for name, column, transform in paired_definitions:
        result = paired_metric(
            long,
            column,
            transform,
            rng,
            args.bootstrap_samples,
            args.permutation_samples,
        )
        paired[name] = result
        paired_rows.append({"metric": name, **result})

    reviewer_rows = reviewer_metrics(long, preferences)
    summary = {
        "status": "complete_for_scored_fields",
        "questions": int(long["question_id"].nunique()),
        "reviewers": int(long["reviewer"].nunique()),
        "systems": list(MODELS),
        "answer_level_ratings": int(len(long)),
        "generator_model_type": "DeepSeek-R1-API",
        "effective_api_model_default": "deepseek-chat",
        "bootstrap": {
            "unit": "question_id",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "model_metrics": model_metrics(long),
        "paired_metrics": paired,
        "preference": preference_metrics(
            preferences,
            rng,
            args.bootstrap_samples,
            args.permutation_samples,
        ),
        "inter_rater_agreement": agreement_metrics(long, preferences),
        "automatic_retrieval_and_citation": retrieval_metrics(long),
        "unscored_fields": [
            "human_citation_validity_0_1",
            "human_hallucination_yes_no",
        ],
        "interpretation_limits": [
            "The Vanilla variant disables KG retrieval, graph filtering, and reranking on the mixed entity/document index; it is not a clean document-only semantic RAG baseline.",
            "Answer-cited-gold-PMID rate is automatic citation presence, not human citation validity.",
            "The score tables contain no adjudicated consensus labels.",
        ],
    }
    write_outputs(output_dir, summary, paired_rows, reviewer_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
