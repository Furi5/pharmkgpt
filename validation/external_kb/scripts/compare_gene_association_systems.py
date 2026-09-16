#!/usr/bin/env python3
"""Compare paired benchmark correctness with exact McNemar tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

from evaluate_gene_association_predictions import (
    extract_prediction,
    index_unique,
    load_jsonl,
)


def parse_prediction_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--prediction must be NAME=PATH")
    name, path = raw.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError("--prediction must be NAME=PATH")
    return name.strip(), Path(path)


def exact_mcnemar_pvalue(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2 * probability)


def correctness_by_id(
    gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, bool]:
    return {
        item_id: extract_prediction(predictions[item_id]) == row.get("answer")
        if item_id in predictions
        else False
        for item_id, row in gold.items()
    }


def compare_pair(
    first_name: str,
    second_name: str,
    first: dict[str, bool],
    second: dict[str, bool],
) -> dict[str, Any]:
    ids = sorted(first)
    both_correct = sum(first[item_id] and second[item_id] for item_id in ids)
    first_only = sum(first[item_id] and not second[item_id] for item_id in ids)
    second_only = sum(not first[item_id] and second[item_id] for item_id in ids)
    both_wrong = sum(not first[item_id] and not second[item_id] for item_id in ids)
    total = len(ids)
    first_accuracy = sum(first.values()) / total if total else None
    second_accuracy = sum(second.values()) / total if total else None
    # Haldane-Anscombe correction keeps the paired odds ratio finite.
    paired_odds_ratio = (first_only + 0.5) / (second_only + 0.5)
    return {
        "system_1": first_name,
        "system_2": second_name,
        "total": total,
        "system_1_accuracy": first_accuracy,
        "system_2_accuracy": second_accuracy,
        "accuracy_difference": (
            first_accuracy - second_accuracy
            if first_accuracy is not None and second_accuracy is not None
            else None
        ),
        "both_correct": both_correct,
        "system_1_only_correct": first_only,
        "system_2_only_correct": second_only,
        "both_wrong": both_wrong,
        "paired_odds_ratio_haldane": paired_odds_ratio,
        "mcnemar_exact_two_sided_p": exact_mcnemar_pvalue(first_only, second_only),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-key", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    gold = index_unique(load_jsonl(args.gold_key), args.gold_key)
    systems: dict[str, dict[str, bool]] = {}
    input_manifest: dict[str, str] = {}
    for raw in args.prediction:
        name, path = parse_prediction_spec(raw)
        if name in systems:
            raise ValueError(f"Duplicate system name: {name}")
        predictions = index_unique(load_jsonl(path), path)
        systems[name] = correctness_by_id(gold, predictions)
        input_manifest[name] = "model_predictions/" + path.name

    comparisons = [
        compare_pair(first, second, systems[first], systems[second])
        for first, second in combinations(systems, 2)
    ]
    payload = {
        "gold_key": args.gold_key.name,
        "prediction_files": input_manifest,
        "comparisons": comparisons,
        "multiple_testing_note": "Report raw exact p-values and apply Holm correction if many systems are compared.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pairwise_comparisons.csv", comparisons)
    (args.output_dir / "pairwise_comparisons.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
