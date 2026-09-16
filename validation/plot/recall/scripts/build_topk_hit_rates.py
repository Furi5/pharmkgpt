#!/usr/bin/env python3
"""Recalculate domain-specific exact-source Recall@K curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_SOURCE = EXPERIMENT_DIR / "data"
DEFAULT_CORRECTED_FULL_SOURCE = (
    VALIDATION_ROOT
    / "kg_ablation/results/corrected_retrieval"
    / "retrieval_only_ablation_results_corrected.json"
)
DEFAULT_OUTPUT = EXPERIMENT_DIR / "results" / "topk_hit_rates.csv"
DEFAULT_MANIFEST = EXPERIMENT_DIR / "results" / "analysis_manifest.json"

DOMAINS = ("Overall", "Clinical", "Genetic", "Cell", "Pathway", "Metabolism")
BASELINE_MODELS = ("LightRAG", "HippoRAG2", "SemanticRAG")
MODELS = (*BASELINE_MODELS, "PharmkGPT")
MODEL_FILES = {
    "LightRAG": ("lightrag_{run}.json", "pmid"),
    "HippoRAG2": ("hipporag2_{run}.json", "retrieved_docs"),
    "SemanticRAG": ("semanticrag_{run}.json", "pmid"),
}


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


def normalize_pmid(value: Any) -> str:
    return str(value or "").strip().lower()


def calculate_run_hit_rates(
    benchmark: dict[str, dict[str, dict[str, Any]]],
    retrieval: dict[str, dict[str, dict[str, Any]]],
    categories: list[str],
    retrieval_field: str,
) -> tuple[list[float], int]:
    hits = [0] * 10
    total = 0
    for category in categories:
        if category not in benchmark:
            raise KeyError(f"Benchmark category is missing: {category}")
        for question_id, question in benchmark[category].items():
            total += 1
            record = retrieval.get(category, {}).get(question_id)
            if record is None:
                continue
            gold_pmid = normalize_pmid(question.get("pmid"))
            retrieved_pmids = [
                normalize_pmid(value) for value in (record.get(retrieval_field) or [])
            ]
            if not gold_pmid:
                continue
            try:
                rank = retrieved_pmids.index(gold_pmid) + 1
            except ValueError:
                continue
            for top_k in range(rank, 11):
                hits[top_k - 1] += 1
    return [100.0 * count / total for count in hits], total


def calculate_corrected_full_recall(
    details: list[dict[str, Any]], categories: list[str]
) -> tuple[list[float], int]:
    """Calculate Recall@1–10 directly from corrected per-question gold ranks."""
    selected = [row for row in details if str(row.get("category")) in categories]
    hits = [0] * 10
    for row in selected:
        rank = row.get("gold_rank")
        if rank is None:
            continue
        rank = int(rank)
        if not 1 <= rank <= 10:
            raise ValueError(f"Invalid corrected gold rank: {rank}")
        for top_k in range(rank, 11):
            hits[top_k - 1] += 1
    total = len(selected)
    if total == 0:
        raise ValueError(f"No corrected Full Model rows for categories {categories}")
    return [100.0 * count / total for count in hits], total


def load_corrected_full(
    path: Path, benchmark: dict[str, dict[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    full = payload.get("configs", {}).get("Full Model")
    if not full:
        raise KeyError(f"Corrected Full Model is missing from {path}")
    metrics = full.get("metrics", {})
    details = full.get("details", [])
    if metrics.get("errors") != 0 or len(details) != metrics.get("total"):
        raise ValueError(
            "Corrected Full Model retrieval is incomplete: "
            f"errors={metrics.get('errors')}, details={len(details)}, "
            f"total={metrics.get('total')}"
        )

    expected = {
        (category, question_id): normalize_pmid(question.get("pmid"))
        for category, questions in benchmark.items()
        for question_id, question in questions.items()
    }
    observed = {
        (str(row.get("category")), str(row.get("id"))): normalize_pmid(
            row.get("gold_pmid")
        )
        for row in details
    }
    if observed != expected:
        raise ValueError("Corrected Full Model questions or gold PMIDs do not match QA.json")
    return details


def build_rows(
    source_dir: Path, corrected_full_source: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    benchmark_path = VALIDATION_ROOT / "data/benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    input_hashes = {portable(benchmark_path): sha256_file(benchmark_path)}
    loaded_runs: dict[tuple[str, int], dict[str, Any]] = {}

    for model in BASELINE_MODELS:
        pattern, _ = MODEL_FILES[model]
        for run in range(1, 6):
            path = source_dir / pattern.format(run=run)
            loaded_runs[(model, run)] = json.loads(path.read_text(encoding="utf-8"))
            input_hashes[portable(path)] = sha256_file(path)
    corrected_details = load_corrected_full(corrected_full_source, benchmark)
    input_hashes[portable(corrected_full_source)] = sha256_file(
        corrected_full_source
    )

    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        categories = list(benchmark) if domain == "Overall" else [domain.lower()]
        for model in BASELINE_MODELS:
            _, retrieval_field = MODEL_FILES[model]
            run_rates: list[list[float]] = []
            sample_size = 0
            for run in range(1, 6):
                rates, run_sample_size = calculate_run_hit_rates(
                    benchmark,
                    loaded_runs[(model, run)],
                    categories,
                    retrieval_field,
                )
                if sample_size and run_sample_size != sample_size:
                    raise ValueError("Sample size changed between runs")
                sample_size = run_sample_size
                run_rates.append(rates)

            for index, top_k in enumerate(range(1, 11)):
                values = [run[index] for run in run_rates]
                rows.append(
                    {
                        "domain": domain,
                        "model": model,
                        "top_k": top_k,
                        "mean_exact_source_recall_percent": f"{statistics.fmean(values):.8f}",
                        "sd_exact_source_recall_percent": f"{statistics.pstdev(values):.8f}",
                        "n_questions": sample_size,
                        "n_runs": len(values),
                    }
                )

        corrected_rates, sample_size = calculate_corrected_full_recall(
            corrected_details, categories
        )
        for top_k, value in enumerate(corrected_rates, 1):
            rows.append(
                {
                    "domain": domain,
                    "model": "PharmkGPT",
                    "top_k": top_k,
                    "mean_exact_source_recall_percent": f"{value:.8f}",
                    "sd_exact_source_recall_percent": f"{0.0:.8f}",
                    "n_questions": sample_size,
                    "n_runs": 1,
                }
            )
    return rows, input_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--corrected-full-source", type=Path, default=DEFAULT_CORRECTED_FULL_SOURCE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    rows, input_hashes = build_rows(args.source_dir, args.corrected_full_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "model",
        "top_k",
        "mean_exact_source_recall_percent",
        "sd_exact_source_recall_percent",
        "n_questions",
        "n_runs",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "baseline_source_directory": portable(args.source_dir),
        "corrected_full_source": portable(args.corrected_full_source),
        "corrected_full_configuration": "Full Model",
        "endpoint": "exact-source Recall@K against the designated gold PMID",
        "domains": list(DOMAINS),
        "models": list(MODELS),
        "top_k_range": [1, 10],
        "run_count_per_model": {
            "LightRAG": 5,
            "HippoRAG2": 5,
            "SemanticRAG": 5,
            "PharmkGPT": 1,
        },
        "question_counts": {
            "Overall": 1045,
            "Clinical": 233,
            "Genetic": 247,
            "Cell": 213,
            "Pathway": 230,
            "Metabolism": 122,
        },
        "input_sha256": input_hashes,
        "output_csv": portable(args.output),
        "output_csv_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.manifest}")
    print(f"Rows: {len(rows)}; source files: {len(input_hashes)}")


if __name__ == "__main__":
    main()
