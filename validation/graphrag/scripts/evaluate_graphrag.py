#!/usr/bin/env python3
"""Evaluate GraphRAG context-source retrieval and multiple-choice answers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from common import EXPERIMENT_ROOT, rank_of, write_json


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def load_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[int, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                key = (
                    int(row.get("run", 1)),
                    str(row.get("category", "")),
                    str(row.get("question_id", "")),
                )
                latest[key] = row
    return list(latest.values())


def compute_metrics(
    rows: list[dict[str, Any]], system: str = "Microsoft GraphRAG 3.1.1 native local search (DeepSeek v4 Flash)"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    unique_retrieval_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("category", "")), str(row.get("question_id", "")))
        unique_retrieval_rows.setdefault(key, row)
    ranks: list[int] = []
    for row in unique_retrieval_rows.values():
        retrieved = [str(item) for item in row.get("retrieved_pmids", [])]
        rank = rank_of(str(row.get("gold_pmid", "")), retrieved)
        if rank is not None:
            ranks.append(rank)

    by_run: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(int(row.get("run", 1)), []).append(row)
    run_metrics = []
    for run_number in sorted(by_run):
        run_rows = by_run[run_number]
        answer_correct = 0
        parsed_answers = 0
        for row in run_rows:
            predicted = row.get("predicted_option")
            correct = str(row.get("correct_option", "")).upper()
            if predicted:
                parsed_answers += 1
            answer_correct += int(bool(predicted and str(predicted).upper() == correct))
        run_total = len(run_rows)
        run_metrics.append(
            {
                "run": run_number,
                "evaluation_denominator": run_total,
                "parsed_answers": parsed_answers,
                "answer_parse_coverage": parsed_answers / run_total if run_total else None,
                "correct_answers": answer_correct,
                "answer_accuracy": answer_correct / run_total if run_total else None,
                "answer_accuracy_95ci": wilson_interval(answer_correct, run_total),
            }
        )

    for row in rows:
        retrieved = [str(item) for item in row.get("retrieved_pmids", [])]
        rank = rank_of(str(row.get("gold_pmid", "")), retrieved)
        predicted = row.get("predicted_option")
        correct = str(row.get("correct_option", "")).upper()
        is_correct = bool(predicted and str(predicted).upper() == correct)
        details.append(
            {
                "run": int(row.get("run", 1)),
                "category": row.get("category"),
                "question_id": row.get("question_id"),
                "gold_pmid": row.get("gold_pmid"),
                "rank": rank,
                "hit_at_1": int(rank is not None and rank <= 1),
                "hit_at_5": int(rank is not None and rank <= 5),
                "hit_at_10": int(rank is not None and rank <= 10),
                "correct_option": correct,
                "predicted_option": predicted,
                "answer_correct": int(is_correct),
                "retrieved_pmids": retrieved,
            }
        )
    total = len(unique_retrieval_rows)

    def hit_count(k: int) -> int:
        return sum(rank <= k for rank in ranks)

    accuracies = [
        item["answer_accuracy"] for item in run_metrics if item["answer_accuracy"] is not None
    ]
    parse_coverages = [
        item["answer_parse_coverage"]
        for item in run_metrics
        if item["answer_parse_coverage"] is not None
    ]
    metrics: dict[str, Any] = {
        "system": system,
        "retrieval_evaluation_denominator": total,
        "retrieval_hit_questions": len(ranks),
        "recall_at_1": hit_count(1) / total if total else None,
        "recall_at_1_95ci": wilson_interval(hit_count(1), total),
        "recall_at_5": hit_count(5) / total if total else None,
        "recall_at_5_95ci": wilson_interval(hit_count(5), total),
        "recall_at_10": hit_count(10) / total if total else None,
        "recall_at_10_95ci": wilson_interval(hit_count(10), total),
        "mrr": sum(1 / rank for rank in ranks) / total if total else None,
        "median_observed_rank": statistics.median(ranks) if ranks else None,
        "answer_runs": run_metrics,
        "completed_answer_run_count": len(run_metrics),
        "answer_parse_coverage_mean": statistics.mean(parse_coverages) if parse_coverages else None,
        "answer_accuracy_mean": statistics.mean(accuracies) if accuracies else None,
        "answer_accuracy_sample_sd": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
        "index_constructed_independently_from_itext2kg": "Microsoft GraphRAG" in system,
        "pubtator_annotations_used": False,
    }
    if len(run_metrics) == 1:
        metrics["evaluation_denominator"] = run_metrics[0]["evaluation_denominator"]
        metrics["answer_parse_coverage"] = run_metrics[0]["answer_parse_coverage"]
        metrics["answer_accuracy"] = run_metrics[0]["answer_accuracy"]
        metrics["answer_accuracy_95ci"] = run_metrics[0]["answer_accuracy_95ci"]
    return metrics, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run", "category", "question_id", "gold_pmid", "rank", "hit_at_1", "hit_at_5",
        "hit_at_10", "correct_option", "predicted_option", "answer_correct", "retrieved_pmids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["retrieved_pmids"] = ";".join(row["retrieved_pmids"])
            writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=EXPERIMENT_ROOT / "results/v4_flash/graphrag_results.jsonl")
    parser.add_argument("--metrics", type=Path, default=EXPERIMENT_ROOT / "results/v4_flash/graphrag_metrics.json")
    parser.add_argument("--per-question", type=Path, default=EXPERIMENT_ROOT / "results/v4_flash/graphrag_per_question.csv")
    parser.add_argument("--system", default="Microsoft GraphRAG 3.1.1 native local search (DeepSeek v4 Flash)")
    args = parser.parse_args()
    metrics, details = compute_metrics(load_rows(args.input), system=args.system)
    write_json(args.metrics, metrics)
    write_csv(args.per_question, details)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
