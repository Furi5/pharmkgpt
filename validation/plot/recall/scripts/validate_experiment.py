#!/usr/bin/env python3
"""Validate Plot exact-source Recall@K statistics and figures."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RESULTS = EXPERIMENT_DIR / "results"
FIGURES = EXPERIMENT_DIR / "figures"
DOMAINS = ("Overall", "Clinical", "Genetic", "Cell", "Pathway", "Metabolism")
MODELS = ("LightRAG", "HippoRAG2", "SemanticRAG", "PharmkGPT")
EXPECTED_N = {
    "Overall": 1045,
    "Clinical": 233,
    "Genetic": 247,
    "Cell": 213,
    "Pathway": 230,
    "Metabolism": 122,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    csv_path = RESULTS / "topk_hit_rates.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 240

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["model"])].append(row)
        assert int(row["n_questions"]) == EXPECTED_N[row["domain"]]
        expected_runs = 1 if row["model"] == "PharmkGPT" else 5
        assert int(row["n_runs"]) == expected_runs
        assert 40 <= float(row["mean_exact_source_recall_percent"]) <= 100
        assert float(row["sd_exact_source_recall_percent"]) >= 0

    assert set(grouped) == {(domain, model) for domain in DOMAINS for model in MODELS}
    for series in grouped.values():
        series.sort(key=lambda row: int(row["top_k"]))
        means = [float(row["mean_exact_source_recall_percent"]) for row in series]
        assert [int(row["top_k"]) for row in series] == list(range(1, 11))
        assert means == sorted(means)

    overall_pharm = grouped[("Overall", "PharmkGPT")]
    assert round(float(overall_pharm[0]["mean_exact_source_recall_percent"]), 2) == 90.14
    assert round(float(overall_pharm[4]["mean_exact_source_recall_percent"]), 2) == 96.56
    assert round(float(overall_pharm[-1]["mean_exact_source_recall_percent"]), 2) == 97.13
    assert all(
        float(row["sd_exact_source_recall_percent"]) == 0.0
        for row in overall_pharm
    )

    manifest = json.loads((RESULTS / "analysis_manifest.json").read_text())
    assert manifest["question_counts"] == EXPECTED_N
    assert len(manifest["input_sha256"]) == 17
    assert manifest["corrected_full_configuration"] == "Full Model"
    assert manifest["endpoint"].startswith("exact-source Recall@K")
    assert manifest["output_csv_sha256"] == sha256_file(csv_path)

    for domain in DOMAINS:
        stem = domain.lower() + "_topk_hit_rate"
        for suffix in (".png", ".pdf"):
            path = FIGURES / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 10_000, path

    print("Plot validation passed: 240 points, 12 figure files")


if __name__ == "__main__":
    main()
