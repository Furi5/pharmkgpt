#!/usr/bin/env python3
"""Compare corrected Full with the no-PubTator-supplementation retrieval arm."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = EXPERIMENT_DIR.parent
DEFAULT_BASELINE = (
    VALIDATION_ROOT
    / "kg_ablation/results/corrected_retrieval/retrieval_only_ablation_results_corrected.json"
)
DEFAULT_ABLATED = EXPERIMENT_DIR / "results/without_all_pubtator_linked_entities.json"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results/all_pubtator_linked_comparison"
BASELINE_NAME = "Full Model"
ABLATED_NAME = "Full Model w/o all PubTator-linked entities"


def load_config(path: Path, name: str = "Full Model") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("configs", {}).get(name)
    if not config:
        raise KeyError(f"Configuration {name!r} is missing from {path}")
    details = config.get("details", [])
    metrics = config.get("metrics", {})
    if metrics.get("errors") != 0 or len(details) != metrics.get("total"):
        raise RuntimeError(
            f"Incomplete retrieval result in {path}: total={metrics.get('total')}, "
            f"details={len(details)}, errors={metrics.get('errors')}"
        )
    return metrics, details


def metrics_from_details(details: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(details)
    ranks = [row.get("gold_rank") for row in details]
    result: dict[str, float | int] = {"total": total}
    for k in (1, 5, 10):
        result[f"recall@{k}"] = sum(
            rank is not None and int(rank) <= k for rank in ranks
        ) / total
    result["mrr@10"] = sum(
        0.0 if rank is None or int(rank) > 10 else 1.0 / int(rank)
        for rank in ranks
    ) / total
    return result


def exact_mcnemar_p(losses: int, gains: int) -> float:
    discordant = losses + gains
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, i) for i in range(min(losses, gains) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty list")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap_ci(
    deltas: list[float], samples: int = 10_000, seed: int = 0
) -> list[float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        statistics.fmean(deltas[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    return [percentile(means, 0.025), percentile(means, 0.975)]


def compare(
    baseline_path: Path,
    ablated_path: Path,
    output_dir: Path,
    bootstrap_samples: int,
    ablated_name: str = ABLATED_NAME,
) -> dict[str, Any]:
    _, baseline_details = load_config(baseline_path)
    _, ablated_details = load_config(ablated_path)
    baseline = {row["id"]: row for row in baseline_details}
    ablated = {row["id"]: row for row in ablated_details}
    if set(baseline) != set(ablated):
        raise RuntimeError(
            f"Question sets differ: baseline_only={len(set(baseline) - set(ablated))}, "
            f"ablated_only={len(set(ablated) - set(baseline))}"
        )

    rows: list[dict[str, Any]] = []
    for question_id in baseline:
        full = baseline[question_id]
        without = ablated[question_id]
        if full.get("gold_pmid") != without.get("gold_pmid"):
            raise RuntimeError(f"Gold PMID mismatch for {question_id}")
        full_rank = full.get("gold_rank")
        without_rank = without.get("gold_rank")
        rows.append(
            {
                "id": question_id,
                "category": full.get("category"),
                "gold_pmid": full.get("gold_pmid"),
                "full_rank": full_rank,
                "without_pubtator_rank": without_rank,
                "full_rr": 0.0 if full_rank is None else 1.0 / int(full_rank),
                "without_pubtator_rr": (
                    0.0 if without_rank is None else 1.0 / int(without_rank)
                ),
            }
        )

    summary_rows = []
    configs = (
        (BASELINE_NAME, baseline_details),
        (ablated_name, ablated_details),
    )
    computed = {name: metrics_from_details(details) for name, details in configs}
    for name, _ in configs:
        metrics = computed[name]
        summary_rows.append({"Configuration": name, **metrics})

    paired: dict[str, Any] = {}
    for k in (1, 5, 10):
        losses = sum(
            row["full_rank"] is not None
            and int(row["full_rank"]) <= k
            and (row["without_pubtator_rank"] is None or int(row["without_pubtator_rank"]) > k)
            for row in rows
        )
        gains = sum(
            row["without_pubtator_rank"] is not None
            and int(row["without_pubtator_rank"]) <= k
            and (row["full_rank"] is None or int(row["full_rank"]) > k)
            for row in rows
        )
        paired[f"recall@{k}"] = {
            "full_minus_without": computed[BASELINE_NAME][f"recall@{k}"]
            - computed[ablated_name][f"recall@{k}"],
            "full_hit_without_miss": losses,
            "full_miss_without_hit": gains,
            "exact_mcnemar_p": exact_mcnemar_p(losses, gains),
        }

    rr_deltas = [row["full_rr"] - row["without_pubtator_rr"] for row in rows]
    paired["mrr@10"] = {
        "full_minus_without": statistics.fmean(rr_deltas),
        "paired_bootstrap_95_ci": paired_bootstrap_ci(
            rr_deltas, samples=bootstrap_samples
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": 0,
    }
    if "all PubTator-linked" in ablated_name:
        interpretation_guardrail = (
            "The treatment removes both PubTator-only and shared "
            "LLM-PubTator entity occurrences. It is a strong deletion stress "
            "test, not a clean estimate of PubTator's unique contribution. "
            "No generated-answer accuracy is used because generation adds "
            "unrelated stochastic variance."
        )
    else:
        interpretation_guardrail = (
            "The treatment removes PubTator-only supplemental entity "
            "occurrences; shared PubTator-normalized entities are retained. "
            "No generated-answer accuracy is used because generation adds "
            "unrelated stochastic variance."
        )
    result = {
        "evaluation": "retrieval-only exact gold PMID ranking",
        "configurations": computed,
        "paired_analysis": paired,
        "interpretation_guardrail": interpretation_guardrail,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pubtator_ablation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("Configuration", "total", "recall@1", "recall@5", "recall@10", "mrr@10"),
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "per_question_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "paired_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--ablated", type=Path, default=DEFAULT_ABLATED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ablated-name", default=ABLATED_NAME)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = compare(
        args.baseline,
        args.ablated,
        args.output_dir,
        args.bootstrap_samples,
        args.ablated_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
