#!/usr/bin/env python3
"""Rebuild five-run accuracy means and SDs from the authoritative answer matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE = (
    EXPERIMENT_DIR.parents[1]
    / "effect_size"
    / "data"
    / "all_answer.csv"
)
EFFECT_SOURCE_OVERALL = (
    EXPERIMENT_DIR.parents[1]
    / "effect_size"
    / "results"
    / "canonical_effect_sizes_overall.csv"
)
EFFECT_SOURCE_DOMAINS = (
    EXPERIMENT_DIR.parents[1]
    / "effect_size"
    / "results"
    / "canonical_effect_sizes_by_domain.csv"
)
OUTPUT = EXPERIMENT_DIR / "results" / "domain_accuracy_summary.csv"
SIGNIFICANCE_OUTPUT = EXPERIMENT_DIR / "results" / "holm_significance_summary.csv"
MANIFEST = EXPERIMENT_DIR / "results" / "analysis_manifest.json"

METHODS = OrderedDict(
    [
        ("Qwen3-32B", [f"Qwen-32b-{i}" for i in range(1, 6)]),
        ("DeepSeek-R1-32B", [f"DeepSeek-32b-{i}" for i in range(1, 6)]),
        ("LLaMA2-13B", [f"llama2-13b-{i}" for i in range(1, 6)]),
        ("LLaMA3-8B", [f"llama3-8b-{i}" for i in range(1, 6)]),
        ("Gemma3-27B", [f"Gemma3-27b-{i}" for i in range(1, 6)]),
        ("LightRAG", [f"lightrag-{i}" for i in range(1, 6)]),
        ("SemanticRAG", [f"SemanticRAG-{i}" for i in range(1, 6)]),
        ("HippoRAG 2", [f"HippoRAG2-{i}" for i in range(1, 6)]),
        ("PharmkGPT", [f"pharmkgpt-{i}" for i in range(1, 6)]),
    ]
)
DOMAINS = ("Overall", "Clinical", "Genetic", "Cell", "Pathway", "Metabolism")
VALID_OPTIONS = {"A", "B", "C", "D"}


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(VALIDATION_ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: str | None) -> str:
    return str(value or "").strip().upper()


def category(question_id: str) -> str:
    return question_id.split("_", 1)[0].strip().capitalize()


def calculate_summary(source: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1045:
        raise ValueError(f"Expected 1,045 questions, found {len(rows):,}")

    required = {"question_id", "correct_option"}
    for columns in METHODS.values():
        required.update(columns)
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    domain_counts = {domain: 0 for domain in DOMAINS}
    domain_counts["Overall"] = len(rows)
    for row in rows:
        row_domain = category(row["question_id"])
        if row_domain not in domain_counts or row_domain == "Overall":
            raise ValueError(f"Unknown domain: {row_domain}")
        domain_counts[row_domain] += 1

    output_rows: list[dict[str, str]] = []
    for domain in DOMAINS:
        selected = rows if domain == "Overall" else [row for row in rows if category(row["question_id"]) == domain]
        for method, columns in METHODS.items():
            run_accuracies: list[float] = []
            for column in columns:
                correct = 0
                for row in selected:
                    gold = normalize(row["correct_option"])
                    prediction = normalize(row[column])
                    if gold not in VALID_OPTIONS:
                        raise ValueError(f"Invalid gold answer for {row['question_id']}")
                    correct += int(prediction in VALID_OPTIONS and prediction == gold)
                run_accuracies.append(100.0 * correct / len(selected))
            output_rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "n_questions": str(len(selected)),
                    "n_runs": str(len(columns)),
                    "mean_accuracy_percent": f"{statistics.mean(run_accuracies):.8f}",
                    "sd_accuracy_percent": f"{statistics.stdev(run_accuracies):.8f}",
                    **{f"run_{i}_accuracy_percent": f"{value:.8f}" for i, value in enumerate(run_accuracies, 1)},
                }
            )
    return output_rows, domain_counts


def build_significance_rows(paths: tuple[Path, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    method_aliases = {
        "DeepSeek-32B": "DeepSeek-R1-32B",
        "Qwen-32B": "Qwen3-32B",
    }
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                adjusted_p = float(row["Holm_Adjusted_p_value"])
                domain = row["Category"].capitalize()
                comparator = method_aliases.get(row["Comparator"], row["Comparator"])
                rows.append(
                    {
                        "domain": domain,
                        "comparator": comparator,
                        "holm_adjusted_p_value": f"{adjusted_p:.16g}",
                        "marker": "**" if adjusted_p < 0.01 else "ns",
                    }
                )
    expected = {(domain, method) for domain in DOMAINS for method in METHODS if method != "PharmkGPT"}
    observed = {(row["domain"], row["comparator"]) for row in rows}
    if observed != expected:
        raise ValueError(f"Effect-size coverage mismatch: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")
    return rows


def main() -> None:
    rows, domain_counts = calculate_summary(SOURCE)
    significance_rows = build_significance_rows((EFFECT_SOURCE_OVERALL, EFFECT_SOURCE_DOMAINS))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with SIGNIFICANCE_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=significance_rows[0].keys())
        writer.writeheader()
        writer.writerows(significance_rows)

    manifest = {
        "source": {"path": portable(SOURCE), "sha256": sha256_file(SOURCE)},
        "significance_sources": [
            {"path": portable(path), "sha256": sha256_file(path)}
            for path in (EFFECT_SOURCE_OVERALL, EFFECT_SOURCE_DOMAINS)
        ],
        "n_questions": len({row["question_id"] for row in csv.DictReader(SOURCE.open(encoding="utf-8-sig"))}),
        "domain_counts": domain_counts,
        "methods": list(METHODS),
        "runs_per_method": 5,
        "accuracy_rule": "missing and invalid responses counted as incorrect",
        "error_bar": "sample SD of five run-level accuracies",
        "significance_rule": "** if Holm-adjusted paired p < 0.01; otherwise ns",
        "output": {"path": portable(OUTPUT), "sha256": sha256_file(OUTPUT)},
        "significance_output": {
            "path": portable(SIGNIFICANCE_OUTPUT),
            "sha256": sha256_file(SIGNIFICANCE_OUTPUT),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {SIGNIFICANCE_OUTPUT}")
    print(f"Wrote {MANIFEST}")
    print(f"Rows: {len(rows)}; questions: {sum(1 for _ in csv.DictReader(SOURCE.open(encoding='utf-8-sig'))) }")


if __name__ == "__main__":
    main()
