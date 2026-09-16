#!/usr/bin/env python3
"""Recompute component perturbation and exact-source retrieval summaries."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np
from scipy.stats import binomtest
ROOT = Path(__file__).resolve().parents[1]


FULL = "Full Model"


NO_KG = "w/o KG Expansion"


SEMANTIC = "Semantic-only + Reranker"


NO_GRAPH = "w/o Graph Filtering"


MATCHED_SEMANTIC = "Semantic-only + Reranker (Matched Context Count)"


COMPARISONS = (
    (
        FULL,
        NO_GRAPH,
        "Graph filtering effect with KG expansion enabled",
    ),
    (
        NO_GRAPH,
        SEMANTIC,
        "Removing KG expansion with graph filtering disabled and fixed top-10",
    ),
    (
        FULL,
        MATCHED_SEMANTIC,
        "Complete KG-guided pathway effect at matched context count",
    ),
    (
        NO_KG,
        SEMANTIC,
        "Graph filtering effect without KG expansion",
    ),
    (
        SEMANTIC,
        MATCHED_SEMANTIC,
        "Context-count control within semantic-only retrieval",
    ),
)


OPERATIONAL_COMPARISONS = (
    (FULL, NO_GRAPH, "Graph Filtering", "component ablation"),
    (FULL, "w/o MST Filtering", "MST Filtering", "component ablation"),
    (
        FULL,
        "w/o N-gram Overlap Edges",
        "N-gram Overlap Edges",
        "component ablation",
    ),
    (FULL, "w/o Entity Weighting", "Entity Weighting", "component ablation"),
    (FULL, NO_KG, "KG Expansion", "component ablation"),
    (FULL, "w/o Reranker", "Reranker", "component ablation"),
    (
        FULL,
        MATCHED_SEMANTIC,
        "Matched Semantic Control",
        "context-matched control",
    ),
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _detail_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return str(row.get("category", "")), str(row["id"])


def _ordered_details(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    details = result.get("details")
    if not isinstance(details, list):
        raise ValueError("Configuration is missing a details list")
    return details


def _validate_and_merge(
    final_results: Dict[str, Any],
    extension_results: Dict[str, Any],
    expected_total: int,
) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    required_final = (FULL, NO_KG, SEMANTIC)
    required_extension = (NO_GRAPH, MATCHED_SEMANTIC)
    missing_final = [name for name in required_final if name not in final_results]
    missing_extension = [name for name in required_extension if name not in extension_results]
    if missing_final or missing_extension:
        raise ValueError(
            f"Missing configurations; final={missing_final}, extension={missing_extension}"
        )

    merged = dict(final_results)
    overlap = set(merged).intersection(extension_results)
    if overlap:
        raise ValueError(f"Duplicate configuration names: {sorted(overlap)}")
    merged.update(extension_results)

    full_details = _ordered_details(merged[FULL])
    full_keys = [_detail_key(row) for row in full_details]
    if len(full_keys) != expected_total or len(set(full_keys)) != expected_total:
        raise ValueError(
            f"Full Model must contain {expected_total} unique questions; got {len(full_keys)}"
        )

    for name, result in merged.items():
        details = _ordered_details(result)
        keys = [_detail_key(row) for row in details]
        if len(keys) != expected_total or set(keys) != set(full_keys):
            raise ValueError(
                f"{name!r} is incomplete or not paired: {len(keys)}/{expected_total} rows"
            )

    full_by_key = {_detail_key(row): row for row in full_details}
    matched_by_key = {
        _detail_key(row): row for row in _ordered_details(merged[MATCHED_SEMANTIC])
    }
    failed_matched = [
        key
        for key in full_keys
        if not bool(matched_by_key[key].get("query_success", True))
    ]
    mismatches = [
        key
        for key in full_keys
        if key not in failed_matched
        if len(full_by_key[key].get("retrieved_pmids", []))
        != len(matched_by_key[key].get("retrieved_pmids", []))
    ]
    if mismatches:
        raise ValueError(
            "Successful matched semantic context counts differ from Full Model for "
            f"{len(mismatches)} questions; first={mismatches[0]}"
        )
    return merged, full_keys


def _index_details(
    result: Mapping[str, Any], keys: Sequence[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    by_key = {_detail_key(row): row for row in _ordered_details(result)}
    return [by_key[key] for key in keys]


def _bool_values(details: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    if field == "parseable":
        return np.asarray(
            [
                bool(row.get("query_success", True))
                and not bool(row.get("parse_failure", False))
                for row in details
            ],
            dtype=np.int8,
        )
    return np.asarray([bool(row.get(field, False)) for row in details], dtype=np.int8)


def _bootstrap_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> Tuple[float, float]:
    values: List[np.ndarray] = []
    remaining = samples
    while remaining:
        batch = min(1000, remaining)
        indices = rng.integers(0, len(differences), size=(batch, len(differences)))
        values.append(differences[indices].mean(axis=1) * 100)
        remaining -= batch
    return tuple(float(value) for value in np.percentile(np.concatenate(values), [2.5, 97.5]))


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def _paired_rows(
    results: Mapping[str, Any],
    keys: Sequence[Tuple[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    metric_fields = (
        ("Answer Accuracy", "answer_correct"),
        ("Recall@10", "recall@10"),
        ("Parseable Rate", "parseable"),
    )
    for reference, variant, interpretation in COMPARISONS:
        reference_details = _index_details(results[reference], keys)
        variant_details = _index_details(results[variant], keys)
        for metric, field in metric_fields:
            reference_values = _bool_values(reference_details, field)
            variant_values = _bool_values(variant_details, field)
            differences = variant_values - reference_values
            reference_only = int(
                np.sum((reference_values == 1) & (variant_values == 0))
            )
            variant_only = int(
                np.sum((reference_values == 0) & (variant_values == 1))
            )
            discordant = reference_only + variant_only
            p_value = (
                float(binomtest(reference_only, discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            ci_low, ci_high = _bootstrap_ci(differences, rng, bootstrap_samples)
            rows.append(
                {
                    "Reference": reference,
                    "Variant": variant,
                    "Interpretation": interpretation,
                    "Metric": metric,
                    "N": len(keys),
                    "Reference (%)": round(float(reference_values.mean() * 100), 4),
                    "Variant (%)": round(float(variant_values.mean() * 100), 4),
                    "Delta Variant-Reference (pp)": round(
                        float(differences.mean() * 100), 4
                    ),
                    "CI95 Low (pp)": round(ci_low, 4),
                    "CI95 High (pp)": round(ci_high, 4),
                    "Reference-only Positive": reference_only,
                    "Variant-only Positive": variant_only,
                    "Exact McNemar p": p_value,
                }
            )

    for metric, _ in metric_fields:
        positions = [index for index, row in enumerate(rows) if row["Metric"] == metric]
        adjusted = _holm_adjust([float(rows[index]["Exact McNemar p"]) for index in positions])
        for index, value in zip(positions, adjusted):
            rows[index]["Holm-adjusted p"] = value
    return rows


def _normalise_pmid(value: Any) -> str:
    text = str(value).strip()
    return text[4:] if text.lower().startswith("pmid") else text


def _retrieval_change_rows(
    results: Mapping[str, Any], keys: Sequence[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for reference, variant, interpretation in COMPARISONS:
        reference_details = _index_details(results[reference], keys)
        variant_details = _index_details(results[variant], keys)
        ordered_changed = 0
        set_changed = 0
        count_changed = 0
        overlap_sum = 0.0
        for reference_row, variant_row in zip(reference_details, variant_details):
            left = [str(value) for value in reference_row.get("retrieved_pmids", [])]
            right = [str(value) for value in variant_row.get("retrieved_pmids", [])]
            left_set, right_set = set(left), set(right)
            ordered_changed += left != right
            set_changed += left_set != right_set
            count_changed += len(left) != len(right)
            union = left_set | right_set
            overlap_sum += len(left_set & right_set) / len(union) if union else 1.0
        rows.append(
            {
                "Reference": reference,
                "Variant": variant,
                "Interpretation": interpretation,
                "N": len(keys),
                "Ordered List Changed": ordered_changed,
                "PMID Set Changed": set_changed,
                "Context Count Changed": count_changed,
                "Mean Jaccard": round(overlap_sum / len(keys), 6),
            }
        )
    return rows


def _operational_impact_rows(
    results: Mapping[str, Any], keys: Sequence[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """Count how often each switch changes retrieval and paired outcomes."""
    rows: List[Dict[str, Any]] = []
    for reference, variant, component, analysis_type in OPERATIONAL_COMPARISONS:
        reference_details = _index_details(results[reference], keys)
        variant_details = _index_details(results[variant], keys)
        counts = Counter(
            {
                "ordered": 0,
                "set": 0,
                "context_count": 0,
                "top1": 0,
                "answer_flip": 0,
                "reference_answer_win": 0,
                "variant_answer_win": 0,
                "recall_flip": 0,
                "reference_recall_win": 0,
                "variant_recall_win": 0,
            }
        )
        for reference_row, variant_row in zip(reference_details, variant_details):
            left = [str(value) for value in reference_row.get("retrieved_pmids", [])]
            right = [str(value) for value in variant_row.get("retrieved_pmids", [])]
            counts["ordered"] += left != right
            counts["set"] += set(left) != set(right)
            counts["context_count"] += len(left) != len(right)
            counts["top1"] += left[:1] != right[:1]

            reference_answer = bool(reference_row.get("answer_correct", False))
            variant_answer = bool(variant_row.get("answer_correct", False))
            counts["answer_flip"] += reference_answer != variant_answer
            counts["reference_answer_win"] += reference_answer and not variant_answer
            counts["variant_answer_win"] += variant_answer and not reference_answer

            reference_recall = bool(reference_row.get("recall@10", False))
            variant_recall = bool(variant_row.get("recall@10", False))
            counts["recall_flip"] += reference_recall != variant_recall
            counts["reference_recall_win"] += reference_recall and not variant_recall
            counts["variant_recall_win"] += variant_recall and not reference_recall

        total = len(keys)
        rows.append(
            {
                "Component or Control": component,
                "Analysis Type": analysis_type,
                "Reference": reference,
                "Variant": variant,
                "N": total,
                "Ordered List Changed": counts["ordered"],
                "Ordered List Changed (%)": round(counts["ordered"] / total * 100, 2),
                "PMID Set Changed": counts["set"],
                "PMID Set Changed (%)": round(counts["set"] / total * 100, 2),
                "Context Count Changed": counts["context_count"],
                "Context Count Changed (%)": round(
                    counts["context_count"] / total * 100, 2
                ),
                "Top-1 Changed": counts["top1"],
                "Answer Correctness Flipped": counts["answer_flip"],
                "Reference-only Correct": counts["reference_answer_win"],
                "Variant-only Correct": counts["variant_answer_win"],
                "Net Correct for Reference": (
                    counts["reference_answer_win"] - counts["variant_answer_win"]
                ),
                "Recall@10 Flipped": counts["recall_flip"],
                "Reference-only Recall@10": counts["reference_recall_win"],
                "Variant-only Recall@10": counts["variant_recall_win"],
                "Net Recall@10 for Reference": (
                    counts["reference_recall_win"] - counts["variant_recall_win"]
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--bootstrap-samples',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20260805)
    args=parser.parse_args()
    results,keys=_validate_and_merge(_load_json(ROOT/'results/answer_perturbation/base/ablation_results.json'),_load_json(ROOT/'results/answer_perturbation/extensions/ablation_results.json'),1045)
    out=args.output_dir/'answer_perturbation/extensions'
    out.mkdir(parents=True,exist_ok=True)
    _write_csv(out/'minimal_attribution_paired_statistics.csv',_paired_rows(results,keys,args.bootstrap_samples,args.seed))
    _write_csv(out/'minimal_attribution_retrieval_changes.csv',_retrieval_change_rows(results,keys))
    out=args.output_dir/'summary'
    out.mkdir(parents=True,exist_ok=True)
    _write_csv(out/'CORE_COMPONENT_IMPACT.csv',_operational_impact_rows(results,keys))
    corrected=_load_json(ROOT/'results/corrected_retrieval/retrieval_only_ablation_results_corrected.json')['configs']
    pubtator=_load_json(ROOT.parent/'pubtator_ablation/results/without_all_pubtator_linked_entities.json')['configs']['Full Model']
    ordered=list(corrected.items())
    ordered.insert(4,('w/o PubTator-linked entities',pubtator))
    table=[]
    for name,config in ordered:
        details=config['details']
        ranks=[r['retrieved_pmids'].index(r['gold_pmid'])+1 if r['gold_pmid'] in r['retrieved_pmids'] else None for r in details]
        settings=config['config']
        row={'Configuration':name,'KG Retrieve':int(settings['use_kg_retrieve']),'Graph Filter':int(settings['use_graph_filter']),'Reranker':int(settings['use_reranker']),'PubTator-linked entities':0 if name=='w/o PubTator-linked entities' else 'NA' if name.startswith('Semantic-only') else 1}
        for k in (1,5,10): row[f'Recall@{k} (%)']=round(100*sum(rank is not None and rank<=k for rank in ranks)/len(ranks),2)
        row['MRR@10 (%)']=round(100*sum(1/rank if rank else 0 for rank in ranks)/len(ranks),2)
        table.append(row)
    _write_csv(out/'GOLD_RETRIEVAL_BENCHMARK.csv',table)
    print('Ablation: paired statistics and retrieval summaries recomputed')

if __name__=='__main__': main()
