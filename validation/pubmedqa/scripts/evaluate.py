#!/usr/bin/env python3
"""Recompute the three-condition PubMedQA results from saved predictions."""
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]


LABELS = ("yes", "no", "maybe")


def f1_metrics(truth: list[str], predictions: list[str]) -> dict[str, Any]:
    per_label = {}
    f1_values = []
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(truth, predictions))
        fp = sum(t != label and p == label for t, p in zip(truth, predictions))
        fn = sum(t == label and p != label for t, p in zip(truth, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(t == label for t in truth),
        }
        f1_values.append(f1)
    return {"macro_f1": sum(f1_values) / len(f1_values), "per_label": per_label}


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - margin, center + margin]


def summarize_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    truth = [record["ground_truth"] for record in records]
    predictions = [record["prediction"] for record in records]
    correct = sum(t == p for t, p in zip(truth, predictions))
    confusion = {
        true_label: {
            predicted_label: sum(
                t == true_label and p == predicted_label
                for t, p in zip(truth, predictions)
            )
            for predicted_label in (*LABELS, "unknown")
        }
        for true_label in LABELS
    }
    result = {
        "total": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else 0.0,
        "accuracy_95ci_wilson": wilson_interval(correct, len(records)),
        "ground_truth_distribution": dict(Counter(truth)),
        "prediction_distribution": dict(Counter(predictions)),
        "confusion_matrix": confusion,
    }
    result.update(f1_metrics(truth, predictions))
    return result


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def build_summary(
    data: dict[str, Any],
    retrieval: dict[str, Any],
    completed: dict[tuple[str, str], dict[str, Any]],
    conditions: list[str],
) -> dict[str, Any]:
    pmids = list(data)
    by_condition: dict[str, list[dict[str, Any]]] = {}
    condition_summaries = {}
    for condition in conditions:
        records = [
            completed[(pmid, condition)]
            for pmid in pmids
            if (pmid, condition) in completed
        ]
        by_condition[condition] = records
        condition_summaries[condition] = summarize_condition(records)

    pairwise = {}
    for index, left in enumerate(conditions):
        for right in conditions[index + 1 :]:
            common_pmids = [
                pmid
                for pmid in pmids
                if (pmid, left) in completed and (pmid, right) in completed
            ]
            left_correct = [
                completed[(pmid, left)]["correct"] for pmid in common_pmids
            ]
            right_correct = [
                completed[(pmid, right)]["correct"] for pmid in common_pmids
            ]
            b = sum(l and not r for l, r in zip(left_correct, right_correct))
            c = sum(not l and r for l, r in zip(left_correct, right_correct))
            left_accuracy = sum(left_correct) / len(common_pmids) if common_pmids else 0.0
            right_accuracy = sum(right_correct) / len(common_pmids) if common_pmids else 0.0
            pairwise[f"{left}_vs_{right}"] = {
                "paired_samples": len(common_pmids),
                "left_accuracy": left_accuracy,
                "right_accuracy": right_accuracy,
                "left_minus_right": left_accuracy - right_accuracy,
                "left_correct_right_wrong": b,
                "left_wrong_right_correct": c,
                "mcnemar_exact_two_sided_p": exact_mcnemar_p(b, c),
            }

    retrieval_strata = {}
    retrieval_rank_strata = {}
    if "pharmkgpt_retrieval" in conditions:
        retrieval_records = by_condition["pharmkgpt_retrieval"]
        for stratum, should_hit in (("gold_source_hit", True), ("gold_source_miss", False)):
            records = [
                record
                for record in retrieval_records
                if record["gold_source_retrieved"] is should_hit
            ]
            retrieval_strata[stratum] = summarize_condition(records)

        rank_buckets = {
            "rank_1": lambda rank: rank == 1,
            "rank_2": lambda rank: rank == 2,
            "rank_3_5": lambda rank: rank is not None and 3 <= rank <= 5,
            "rank_6_10": lambda rank: rank is not None and 6 <= rank <= 10,
            "miss": lambda rank: rank is None,
        }
        ranked_records = []
        for record in retrieval_records:
            evidence_pmids = record["evidence_pmids"]
            rank = (
                evidence_pmids.index(record["pmid"]) + 1
                if record["pmid"] in evidence_pmids
                else None
            )
            ranked_records.append((record, rank))
        for stratum, includes_rank in rank_buckets.items():
            records = [
                record for record, rank in ranked_records if includes_rank(rank)
            ]
            metrics = summarize_condition(records)
            stratum_pmids = [record["pmid"] for record in records]
            gold_correct = [
                completed[(pmid, "gold")]["correct"]
                for pmid in stratum_pmids
                if (pmid, "gold") in completed
            ]
            closed_correct = [
                completed[(pmid, "closed_book")]["correct"]
                for pmid in stratum_pmids
                if (pmid, "closed_book") in completed
            ]
            retrieval_correct = [record["correct"] for record in records]
            paired_total = min(len(gold_correct), len(retrieval_correct))
            gold_accuracy = (
                sum(gold_correct) / len(gold_correct) if gold_correct else 0.0
            )
            closed_accuracy = (
                sum(closed_correct) / len(closed_correct) if closed_correct else 0.0
            )
            gold_only = sum(
                gold and not retrieved
                for gold, retrieved in zip(gold_correct, retrieval_correct)
            )
            retrieval_only = sum(
                not gold and retrieved
                for gold, retrieved in zip(gold_correct, retrieval_correct)
            )
            metrics["gold_accuracy_on_same_questions"] = gold_accuracy
            metrics["closed_book_accuracy_on_same_questions"] = closed_accuracy
            metrics["gold_minus_retrieval"] = gold_accuracy - metrics["accuracy"]
            metrics["paired_gold_vs_retrieval"] = {
                "paired_samples": paired_total,
                "gold_correct_retrieval_wrong": gold_only,
                "gold_wrong_retrieval_correct": retrieval_only,
                "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                    gold_only, retrieval_only
                ),
            }
            retrieval_rank_strata[stratum] = metrics

    return {
        "status": "complete"
        if all(condition_summaries[c]["total"] == len(data) for c in conditions)
        else "partial",
        "questions": len(data),
        "conditions": condition_summaries,
        "pairwise": pairwise,
        "retrieval_input_metrics": {
            "exact_source_recall_at_k": retrieval["exact_source_recall_at_k"],
        },
        "pharmkgpt_retrieval_strata": retrieval_strata,
        "pharmkgpt_retrieval_rank_strata": retrieval_rank_strata,
    }


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "correct",
                "total",
                "accuracy",
                "accuracy_ci_low",
                "accuracy_ci_high",
                "macro_f1",
            ]
        )
        for condition, metrics in summary["conditions"].items():
            writer.writerow(
                [
                    condition,
                    metrics["correct"],
                    metrics["total"],
                    metrics["accuracy"],
                    *metrics["accuracy_95ci_wilson"],
                    metrics["macro_f1"],
                ]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/deepseek_controlled_comparison')
    args = parser.parse_args()
    data = json.loads((ROOT/'data/pubmedqa/ori_pqal.json').read_text())
    retrieval = json.loads((ROOT/'data/pharmkgpt_benchmark/frozen_retrieval_top10.json').read_text())
    records = [json.loads(line) for line in (ROOT/'results/deepseek_controlled_comparison/responses.jsonl').read_text().splitlines() if line.strip()]
    completed = {(row['pmid'],row['condition']):row for row in records}
    conditions = ['gold','closed_book','pharmkgpt_retrieval']
    if len(completed)!=len(records) or set(completed)!={(pmid,c) for pmid in data for c in conditions}:
        raise ValueError('Incomplete or duplicate PubMedQA predictions')
    if set(retrieval['queries'])!=set(data):
        raise ValueError('Retrieval questions do not match the benchmark')
    for row in records:
        gold = data[row['pmid']]['final_decision']
        if row['ground_truth'] != gold or row['correct'] != (row['prediction']==gold):
            raise ValueError('Prediction labels do not match the gold benchmark')
        if row['condition']=='pharmkgpt_retrieval' and row['evidence_pmids']!=retrieval['queries'][row['pmid']]:
            raise ValueError('Answer evidence differs from frozen retrieval')
    for k in (1,2,5,10):
        value = sum(pmid in hits[:k] for pmid,hits in retrieval['queries'].items())/len(data)
        if not math.isclose(value,retrieval['exact_source_recall_at_k'][str(k)],abs_tol=1e-12):
            raise ValueError('Stored recall differs from the ranked PMIDs')
    summary = build_summary(data,retrieval,completed,conditions)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv(args.output_dir/'comparison_table.csv',summary)
    print('PubMedQA: 3,000 predictions evaluated')

if __name__ == '__main__':
    main()
