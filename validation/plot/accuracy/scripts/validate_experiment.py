#!/usr/bin/env python3
"""Validate Plot data and rendered artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RESULTS = EXPERIMENT_DIR / "results" / "domain_accuracy_summary.csv"
MANIFEST = EXPERIMENT_DIR / "results" / "analysis_manifest.json"
SIGNIFICANCE = EXPERIMENT_DIR / "results" / "holm_significance_summary.csv"
LATEST_TABLE = (
    EXPERIMENT_DIR.parent.parent
    / "effect_size"
    / "docs"
    / "SUPPLEMENTARY_TABLE_1_CORRECTED.md"
)
FIGURES = EXPERIMENT_DIR / "figures"
DOMAINS = ("Overall", "Clinical", "Genetic", "Cell", "Pathway", "Metabolism")
EXPECTED_PHARMKGPT = {
    "Overall": 95.62,
    "Clinical": 98.03,
    "Genetic": 89.31,
    "Cell": 98.12,
    "Pathway": 97.22,
    "Metabolism": 96.39,
}
TABLE_METHODS = (
    "PharmkGPT",
    "HippoRAG 2",
    "SemanticRAG",
    "LightRAG",
    "Gemma3-27B",
    "LLaMA3-8B",
    "LLaMA2-13B",
    "DeepSeek-R1-32B",
    "Qwen3-32B",
)


def parse_latest_table() -> dict[tuple[str, str], tuple[float, float]]:
    section = LATEST_TABLE.read_text(encoding="utf-8").split("## A. Corrected accuracy results", 1)[1].split("## B.", 1)[0]
    expected: dict[tuple[str, str], tuple[float, float]] = {}
    for line in section.splitlines():
        if not any(line.startswith(f"| {domain} |") for domain in DOMAINS):
            continue
        cells = [cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")]
        domain = cells[0]
        values = cells[2:]
        assert len(values) == len(TABLE_METHODS)
        for method, value in zip(TABLE_METHODS, values):
            mean_text, sd_text = (part.strip() for part in value.split("±"))
            expected[(domain, method)] = (float(mean_text), float(sd_text))
    return expected


def main() -> None:
    with RESULTS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 54
    observed = {(row["domain"], row["method"]): row for row in rows}
    expected_from_table = parse_latest_table()
    assert set(observed) == set(expected_from_table)
    for key, (expected_mean, expected_sd) in expected_from_table.items():
        assert round(float(observed[key]["mean_accuracy_percent"]), 2) == expected_mean, key
        assert round(float(observed[key]["sd_accuracy_percent"]), 2) == expected_sd, key
    for domain, expected in EXPECTED_PHARMKGPT.items():
        row = next(row for row in rows if row["domain"] == domain and row["method"] == "PharmkGPT")
        assert round(float(row["mean_accuracy_percent"]), 2) == expected
        assert int(row["n_runs"]) == 5

    with SIGNIFICANCE.open(encoding="utf-8", newline="") as handle:
        significance_rows = list(csv.DictReader(handle))
    assert len(significance_rows) == 48
    assert {row["marker"] for row in significance_rows} <= {"**", "ns"}
    nonsignificant = {
        (row["domain"], row["comparator"])
        for row in significance_rows
        if row["marker"] == "ns"
    }
    assert nonsignificant == {
        ("Genetic", "HippoRAG 2"),
        ("Genetic", "SemanticRAG"),
        ("Cell", "Gemma3-27B"),
        ("Cell", "LLaMA3-8B"),
    }

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["n_questions"] == 1045
    for domain in DOMAINS:
        for suffix in (".png", ".pdf"):
            path = FIGURES / f"{domain.lower()}_accuracy_comparison{suffix}"
            assert path.is_file() and path.stat().st_size > 1000, path
    print("Plot validation passed: latest table 54/54, Holm annotations 48/48, figures 12/12")


if __name__ == "__main__":
    main()
