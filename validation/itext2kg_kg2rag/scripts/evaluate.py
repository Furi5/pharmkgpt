#!/usr/bin/env python3
"""Evaluate formal Generic iText2KG + KG2RAG retrieval and answer outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import CONFIG_PATH, EXPERIMENT_ROOT, atomic_write_json, iter_jsonl, read_json


DEFAULT_ROOT = EXPERIMENT_ROOT / "results" / "formal"


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if not total:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - margin, center + margin]


def retrieval_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_question = []
    for row in rows:
        pmids = list(row.get("retrieved_pmids") or [])
        try:
            rank = pmids.index(row["gold_pmid"]) + 1
        except ValueError:
            rank = None
        per_question.append(
            {
                "category": row["category"],
                "qid": row["qid"],
                "gold_pmid": row["gold_pmid"],
                "gold_rank": rank,
                "recall_at_1": int(rank is not None and rank <= 1),
                "recall_at_5": int(rank is not None and rank <= 5),
                "recall_at_10": int(rank is not None and rank <= 10),
                "reciprocal_rank": 1 / rank if rank else 0.0,
                "retrieved_count": len(pmids),
            }
        )
    total = len(per_question)
    metrics: dict[str, Any] = {"evaluation_denominator": total}
    for cutoff in (1, 5, 10):
        hits = sum(row[f"recall_at_{cutoff}"] for row in per_question)
        metrics[f"recall_at_{cutoff}_hits"] = hits
        metrics[f"recall_at_{cutoff}"] = hits / total if total else None
        metrics[f"recall_at_{cutoff}_95ci"] = wilson(hits, total)
    metrics["mrr"] = (
        statistics.mean(row["reciprocal_rank"] for row in per_question)
        if per_question
        else None
    )
    observed = sorted(row["gold_rank"] for row in per_question if row["gold_rank"] is not None)
    metrics["median_observed_rank"] = statistics.median(observed) if observed else None
    metrics["questions_with_fewer_than_10_contexts"] = sum(
        row["retrieved_count"] < 10 for row in per_question
    )
    return metrics, per_question


def answer_metrics(rows: list[dict[str, Any]], expected_questions: int) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["run"])].append(row)
    run_metrics = []
    for run, values in sorted(grouped.items()):
        parsed = sum(row.get("predicted_option") in {"A", "B", "C", "D"} for row in values)
        correct = sum(
            str(row.get("predicted_option") or "").upper()
            == str(row.get("correct_option") or "").upper()
            for row in values
        )
        run_metrics.append(
            {
                "run": run,
                "rows": len(values),
                "complete": len(values) == expected_questions,
                "parsed": parsed,
                "parse_coverage": parsed / len(values) if values else None,
                "correct": correct,
                "accuracy": correct / len(values) if values else None,
                "accuracy_95ci": wilson(correct, len(values)),
            }
        )
    accuracies = [row["accuracy"] for row in run_metrics if row["accuracy"] is not None]
    return {
        "completed_run_count": sum(row["complete"] for row in run_metrics),
        "stored_rows": sum(row["rows"] for row in run_metrics),
        "accuracy_mean": statistics.mean(accuracies) if accuracies else None,
        "accuracy_sample_sd": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
        "runs": run_metrics,
    }


def percentage(value: float | None) -> str:
    return "pending" if value is None else f"{100 * value:.2f}%"




def main() -> None:
    config = read_json(CONFIG_PATH)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    retrieval_path = args.root / "kg2rag" / "retrieval.jsonl"
    answers_path = args.root / "answers" / "answers.jsonl"
    construction_path = args.root / "official_itext2kg" / "construction_summary.json"
    retrieval_rows = list(iter_jsonl(retrieval_path)) if retrieval_path.exists() else []
    answer_rows = list(iter_jsonl(answers_path)) if answers_path.exists() else []
    construction = read_json(construction_path) if construction_path.exists() else {}
    retrieval, per_question = retrieval_metrics(retrieval_rows)
    answers = answer_metrics(answer_rows, int(config["expected_questions"]))
    valid = (
        construction.get("status") == "complete"
        and construction.get("documents_accounted_for") == config["expected_documents"]
        and retrieval["evaluation_denominator"] == config["expected_questions"]
        and answers["completed_run_count"] == config["answer"]["runs"]
        and all(row["parse_coverage"] == 1.0 for row in answers["runs"])
    )
    metrics = {
        "system": "Official iText2KG v0.0.7 + pinned KG2RAG PubMed adapter",
        "valid_for_reviewer_claim": valid,
        "expected_answer_runs": config["answer"]["runs"],
        "retrieval": retrieval,
        "answers": answers,
        "pubtator_annotations_used": False,
        "custom_biomedical_prompts_used": False,
    }
    output_dir = args.root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "metrics.json", metrics)
    if per_question:
        with (output_dir / "retrieval_per_question.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_question[0]))
            writer.writeheader()
            writer.writerows(per_question)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
