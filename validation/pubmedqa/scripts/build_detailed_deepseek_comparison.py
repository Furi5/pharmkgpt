#!/usr/bin/env python3
"""Build the detailed three-arm DeepSeek/PharmKGPT comparison table."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = EXPERIMENT_DIR / "results" / "deepseek_controlled_comparison"
SUMMARY_PATH = RESULT_DIR / "comparison_summary.json"
RESPONSES_PATH = RESULT_DIR / "responses.jsonl"
RETRIEVAL_PATH = (
    EXPERIMENT_DIR / "data" / "pharmkgpt_benchmark" / "frozen_retrieval_top10.json"
)
INVENTORY = EXPERIMENT_DIR / "data" / "construction_inventory.csv"
DATA_PATH = EXPERIMENT_DIR / "data" / "pubmedqa" / "ori_pqal.json"

CONDITIONS = (
    ("closed_book", "DeepSeek (closed-book)"),
    ("pharmkgpt_retrieval", "DeepSeek + PharmKGPT"),
    ("gold", "DeepSeek + Gold"),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def build() -> dict[str, Any]:
    summary = load_json(SUMMARY_PATH)
    retrieval = load_json(RETRIEVAL_PATH)
    responses = load_jsonl(RESPONSES_PATH)
    benchmark = load_json(DATA_PATH)

    questions = len(benchmark)
    if questions != 1000 or summary.get("questions") != questions:
        raise ValueError("Expected the complete 1,000-question PQA-L benchmark")

    response_counts = Counter(row["condition"] for row in responses)
    unknown_counts = Counter(
        row["condition"] for row in responses if row.get("prediction") == "unknown"
    )
    error_counts = Counter(
        row["condition"] for row in responses if row.get("error") is not None
    )
    for condition, _ in CONDITIONS:
        if response_counts[condition] != questions:
            raise ValueError(f"Incomplete responses for {condition}")

    queries = retrieval["queries"]
    if list(queries) != list(benchmark):
        raise ValueError("Frozen retrieval questions do not match PQA-L order")
    ranks = []
    for gold_pmid, retrieved_pmids in queries.items():
        if len(retrieved_pmids) != 10:
            raise ValueError(f"Expected Top-10 retrieval for PMID {gold_pmid}")
        if gold_pmid in retrieved_pmids:
            ranks.append(retrieved_pmids.index(gold_pmid) + 1)
    recall = {
        str(k): sum(rank <= k for rank in ranks) / questions for k in (1, 2, 5, 10)
    }
    mrr = sum(1 / rank for rank in ranks) / questions

    recorded_recall = retrieval["exact_source_recall_at_k"]
    for k, value in recall.items():
        if abs(value - float(recorded_recall[k])) > 1e-12:
            raise ValueError(f"Recall@{k} differs from frozen manifest")

    with INVENTORY.open() as handle:
        kg_pmids = {row["pmid"] for row in csv.DictReader(handle) if row["saved_graph"] == "True"}
    expected_pmids = set(benchmark)
    unexpected_kgs = kg_pmids - expected_pmids
    if unexpected_kgs:
        raise ValueError(f"Unexpected KG PMIDs: {sorted(unexpected_kgs)}")
    missing_kg_pmids = sorted(expected_pmids - kg_pmids, key=int)

    arms: dict[str, dict[str, Any]] = {}
    for condition, display_name in CONDITIONS:
        metrics = summary["conditions"][condition]
        arm = {
            "display_name": display_name,
            "evidence_setting": {
                "closed_book": "No external evidence",
                "pharmkgpt_retrieval": "Frozen PharmKGPT Top-10 retrieval",
                "gold": "Oracle PubMedQA gold context",
            }[condition],
            "questions_evaluated": metrics["total"],
            "correct_answers": metrics["correct"],
            "answer_accuracy": metrics["accuracy"],
            "answer_accuracy_95ci_wilson": metrics["accuracy_95ci_wilson"],
            "macro_f1": metrics["macro_f1"],
            "unknown_outputs": unknown_counts[condition],
            "api_error_outputs": error_counts[condition],
        }
        if condition == "pharmkgpt_retrieval":
            arm.update(
                {
                    "source_articles_successfully_indexed_as_kgs": len(kg_pmids),
                    "kg_construction_or_parsing_failures": len(missing_kg_pmids),
                    "exact_source_recall_at_k": recall,
                    "mrr": mrr,
                    "retrieval_metric_type": "observed",
                }
            )
        elif condition == "gold":
            arm.update(
                {
                    "source_articles_successfully_indexed_as_kgs": None,
                    "kg_construction_or_parsing_failures": None,
                    "exact_source_recall_at_k": {str(k): 1.0 for k in (1, 2, 5, 10)},
                    "mrr": 1.0,
                    "retrieval_metric_type": "oracle_by_definition_not_observed_retrieval",
                }
            )
        else:
            arm.update(
                {
                    "source_articles_successfully_indexed_as_kgs": None,
                    "kg_construction_or_parsing_failures": None,
                    "exact_source_recall_at_k": None,
                    "mrr": None,
                    "retrieval_metric_type": "not_applicable_no_retrieval",
                }
            )
        arms[condition] = arm

    return {
        "status": "complete",
        "evaluation": "PubMedQA PQA-L expert-labeled subset",
        "model_control": "deepseek-r1:32b for all three arms",
        "questions": questions,
        "arms": arms,
        "pharmkgpt_missing_kg_pmids": missing_kg_pmids,
        "notes": [
            "Only the supplied evidence differs across arms; model and inference settings are fixed.",
            "Closed-book has no retrieval, so Recall@K and MRR are not applicable.",
            "Gold context is supplied directly; its 100% Recall@K and MRR=1 are oracle definitions, not measured retriever performance.",
            "All 1,000 questions remain in the answer denominator, including the five source articles without a parsed KG.",
        ],
    }


def write_csv(payload: dict[str, Any], path: Path) -> None:
    arms = payload["arms"]
    columns = [name for _, name in CONDITIONS]
    lookup = {name: arms[key] for key, name in CONDITIONS}
    rows = [
        ("Evaluation", *(payload["evaluation"] for _ in columns)),
        ("Evidence setting", *(lookup[name]["evidence_setting"] for name in columns)),
        ("Questions evaluated", *(lookup[name]["questions_evaluated"] for name in columns)),
        (
            "Source articles successfully indexed as KGs",
            "N/A",
            arms["pharmkgpt_retrieval"]["source_articles_successfully_indexed_as_kgs"],
            "N/A",
        ),
        (
            "KG construction/parsing failures",
            "N/A",
            arms["pharmkgpt_retrieval"]["kg_construction_or_parsing_failures"],
            "N/A",
        ),
        ("Correct answers", *(lookup[name]["correct_answers"] for name in columns)),
        ("Answer accuracy", *(pct(lookup[name]["answer_accuracy"]) for name in columns)),
        (
            "Answer accuracy 95% CI",
            *(
                f"{100 * lookup[name]['answer_accuracy_95ci_wilson'][0]:.2f}–"
                f"{100 * lookup[name]['answer_accuracy_95ci_wilson'][1]:.2f}%"
                for name in columns
            ),
        ),
        ("Macro-F1", *(f"{100 * lookup[name]['macro_f1']:.2f}%" for name in columns)),
        ("Unknown outputs", *(lookup[name]["unknown_outputs"] for name in columns)),
    ]
    for k in (1, 2, 5, 10):
        rows.append(
            (
                f"Exact-source Recall@{k}",
                "N/A",
                pct(arms["pharmkgpt_retrieval"]["exact_source_recall_at_k"][str(k)]),
                "100.0% (oracle-supplied)",
            )
        )
    rows.append(
        (
            "MRR",
            "N/A",
            f"{arms['pharmkgpt_retrieval']['mrr']:.4f}",
            "1.0000 (oracle-supplied)",
        )
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", *columns])
        writer.writerows(rows)


def main() -> None:
    payload = build()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULT_DIR / "detailed_system_comparison.json"
    csv_path = RESULT_DIR / "detailed_system_comparison.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(payload, csv_path)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
