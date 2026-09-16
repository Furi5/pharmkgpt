#!/usr/bin/env python3
"""Evaluate closed-model results with coverage, confidence intervals, and McNemar."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from common import DEFAULT_BENCHMARK, EXPERIMENT_ROOT, load_benchmark, wilson_interval, write_json


def load_answers(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    answers: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return answers
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["category"]), str(row["question_id"]))
            if key in answers:
                raise ValueError(f"Duplicate result key in {path}: {key}")
            answers[key] = row
    return answers


def exact_mcnemar_p_value(first_only: int, second_only: int) -> float | None:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def model_metrics(
    benchmark: list[dict[str, str]], answers: dict[tuple[str, str], dict[str, Any]], model: str
) -> dict[str, Any]:
    completed = 0
    parsed = 0
    correct = 0
    for row in benchmark:
        answer = answers.get((row["category"], row["question_id"]))
        if answer is None:
            continue
        completed += 1
        prediction = answer.get("predicted_option")
        if prediction:
            parsed += 1
        correct += int(str(prediction or "").upper() == row["correct_option"])
    total = len(benchmark)
    return {
        "model": model,
        "status": "complete" if completed == total else "partial",
        "benchmark_questions": total,
        "completed_questions": completed,
        "missing_questions": total - completed,
        "parsed_answers": parsed,
        "answer_parse_coverage_completed": parsed / completed if completed else None,
        "correct_answers": correct,
        "accuracy_completed": correct / completed if completed else None,
        "accuracy_completed_95ci": wilson_interval(correct, completed),
        "accuracy_full_benchmark_conservative": correct / total if total else None,
    }


def compare(
    benchmark: list[dict[str, str]],
    first: dict[tuple[str, str], dict[str, Any]],
    second: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    both_correct = first_only = second_only = both_wrong = 0
    paired = 0
    for row in benchmark:
        key = (row["category"], row["question_id"])
        if key not in first or key not in second:
            continue
        paired += 1
        gold = row["correct_option"]
        first_correct = str(first[key].get("predicted_option") or "").upper() == gold
        second_correct = str(second[key].get("predicted_option") or "").upper() == gold
        if first_correct and second_correct:
            both_correct += 1
        elif first_correct:
            first_only += 1
        elif second_correct:
            second_only += 1
        else:
            both_wrong += 1
    return {
        "comparison": "GPT-5.2 vs Gemini 3.1 Pro Preview",
        "paired_questions": paired,
        "both_correct": both_correct,
        "gpt_5_2_only_correct": first_only,
        "gemini_3_1_only_correct": second_only,
        "both_wrong": both_wrong,
        "discordant_pairs": first_only + second_only,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p_value(first_only, second_only) if paired else None,
        "valid_for_full_benchmark_claim": paired == len(benchmark),
    }


def write_per_question(
    path: Path,
    benchmark: list[dict[str, str]],
    openai: dict[tuple[str, str], dict[str, Any]],
    gemini: dict[tuple[str, str], dict[str, Any]],
) -> None:
    fields = [
        "category", "question_id", "correct_option", "gpt_5_2_prediction", "gpt_5_2_correct",
        "gemini_3_1_prediction", "gemini_3_1_correct",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in benchmark:
            key = (row["category"], row["question_id"])
            openai_prediction = openai.get(key, {}).get("predicted_option")
            gemini_prediction = gemini.get(key, {}).get("predicted_option")
            writer.writerow(
                {
                    "category": row["category"],
                    "question_id": row["question_id"],
                    "correct_option": row["correct_option"],
                    "gpt_5_2_prediction": openai_prediction,
                    "gpt_5_2_correct": int(str(openai_prediction or "").upper() == row["correct_option"]) if key in openai else "",
                    "gemini_3_1_prediction": gemini_prediction,
                    "gemini_3_1_correct": int(str(gemini_prediction or "").upper() == row["correct_option"]) if key in gemini else "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--openai", type=Path, default=EXPERIMENT_ROOT / "results" / "openai_answers.jsonl")
    parser.add_argument("--gemini", type=Path, default=EXPERIMENT_ROOT / "results" / "gemini_answers.jsonl")
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "results" / "closed_model_metrics.json")
    parser.add_argument("--per-question", type=Path, default=EXPERIMENT_ROOT / "results" / "closed_model_per_question.csv")
    args = parser.parse_args()
    benchmark = load_benchmark(args.benchmark)
    openai = load_answers(args.openai)
    gemini = load_answers(args.gemini)
    payload = {
        "models": [
            model_metrics(benchmark, openai, "gpt-5.2-2025-12-11"),
            model_metrics(benchmark, gemini, "gemini-3.1-pro-preview"),
        ],
        "paired_comparison": compare(benchmark, openai, gemini),
    }
    write_json(args.output, payload)
    write_per_question(args.per_question, benchmark, openai, gemini)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
