#!/usr/bin/env python3
"""Evaluate complete, structured yes/no predictions against the private gold key."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


NO_PATTERNS = [
    re.compile(r"^\s*(?:answer|prediction)?\s*[:=]?\s*no\b", re.I),
    re.compile(r"\bnot associated\b", re.I),
    re.compile(r"\bnot reported as (?:a )?phenotype\b", re.I),
    re.compile(r"\bno (?:reliable|supporting|curated) (?:evidence|association)\b", re.I),
]
YES_PATTERNS = [
    re.compile(r"^\s*(?:answer|prediction)?\s*[:=]?\s*yes\b", re.I),
    re.compile(r"\bis associated\b", re.I),
    re.compile(r"\bis reported as (?:a )?phenotype\b", re.I),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            rows.append(row)
    return rows


def index_unique(rows: Iterable[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("id") or row.get("question_id") or row.get("benchmark_id") or "")
        if not item_id:
            raise ValueError(f"Row without id in {path}")
        if item_id in indexed:
            raise ValueError(f"Duplicate id {item_id!r} in {path}")
        indexed[item_id] = row
    return indexed


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL).strip()


def parse_label(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        for key in ("prediction", "predicted_answer", "label", "answer"):
            if key in value:
                parsed = parse_label(value[key])
                if parsed != "unknown":
                    return parsed
        return "unknown"

    normalized = remove_think_blocks(str(value or ""))
    lowered = normalized.lower().strip().strip(".!,;: ")
    if lowered in {"yes", "y", "true", "1", "associated", "supported"}:
        return "yes"
    if lowered in {"no", "n", "false", "0", "not associated", "unsupported"}:
        return "no"
    if lowered in {"unknown", "uncertain", "insufficient evidence", "abstain"}:
        return "unknown"

    if normalized.startswith("{"):
        try:
            parsed = json.loads(normalized)
            label = parse_label(parsed)
            if label != "unknown":
                return label
        except json.JSONDecodeError:
            pass

    # Negative language is evaluated first so "not associated" is never
    # overridden by the substring "associated".
    if any(pattern.search(normalized) for pattern in NO_PATTERNS):
        return "no"
    if any(pattern.search(normalized) for pattern in YES_PATTERNS):
        return "yes"
    return "unknown"


def answer_text(row: dict[str, Any]) -> str:
    parsed_answer = row.get("parsed_answer")
    if isinstance(parsed_answer, dict):
        for key in ("answer", "prediction"):
            if parsed_answer.get(key):
                return str(parsed_answer[key])
    for key in ("model_answer", "raw_answer", "response", "output", "answer"):
        if row.get(key):
            return str(row[key])
    return ""


def prediction_text(row: dict[str, Any]) -> str:
    """Return answer text for label parsing (backward-compatible public helper)."""
    return answer_text(row)


def extract_prediction(row: dict[str, Any]) -> str:
    for key in ("prediction", "predicted_answer", "label"):
        if key in row:
            parsed = parse_label(row[key])
            if parsed != "unknown" or str(row[key]).strip():
                return parsed
    if isinstance(row.get("parsed_answer"), dict):
        parsed = parse_label(row["parsed_answer"])
        if parsed != "unknown":
            return parsed
    return parse_label(prediction_text(row))


def answer_citation_values(row: dict[str, Any]) -> list[str]:
    """Return citations explicitly emitted in the answer, excluding retrieved references."""
    values: list[str] = []
    for container in (row, row.get("parsed_answer") if isinstance(row.get("parsed_answer"), dict) else {}):
        for key in ("citations", "citation"):
            value = container.get(key)
            if isinstance(value, list):
                values.extend(str(item) for item in value if str(item).strip())
            elif value:
                values.append(str(value))
    if not values:
        combined = " ".join((answer_text(row), evidence_text(row)))
        values.extend(re.findall(r"\bPMID\s*[:#]?\s*\d+\b", combined, re.I))
    return list(dict.fromkeys(values))


def citation_values(row: dict[str, Any]) -> list[str]:
    """Backward-compatible alias for answer citations only."""
    return answer_citation_values(row)


def retrieved_reference_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for container in (
        row,
    ):
        value = container.get("retrieved_pmids")
        if isinstance(value, list):
            values.extend(str(item) for item in value if str(item).strip())
        elif value:
            values.append(str(value))
    return list(dict.fromkeys(values))


def evidence_text(row: dict[str, Any]) -> str:
    for container in (row, row.get("parsed_answer") if isinstance(row.get("parsed_answer"), dict) else {}):
        for key in ("evidence", "evidence_sentence", "rationale"):
            if container.get(key):
                return str(container[key])
    return ""


def mentions_target_gene(gene_symbol: str, text: str) -> bool:
    """Check the requested symbol without allowing alphanumeric substring matches."""
    symbol = str(gene_symbol or "").strip()
    if not symbol:
        return False
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return bool(pattern.search(str(text or "")))


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def compute_metrics(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored_rows)
    positives = [row for row in scored_rows if row["gold"] == "yes"]
    negatives = [row for row in scored_rows if row["gold"] == "no"]
    tp = sum(row["predicted"] == "yes" for row in positives)
    tn = sum(row["predicted"] == "no" for row in negatives)
    fp = sum(row["predicted"] == "yes" for row in negatives)
    fn = sum(row["predicted"] == "no" for row in positives)
    unknown_positive = sum(row["predicted"] == "unknown" for row in positives)
    unknown_negative = sum(row["predicted"] == "unknown" for row in negatives)
    unknown = unknown_positive + unknown_negative
    correct = tp + tn
    sensitivity = safe_div(tp, len(positives))
    specificity = safe_div(tn, len(negatives))
    precision = safe_div(tp, tp + fp)
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision is not None
        and sensitivity is not None
        and precision + sensitivity > 0
        else None
    )
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    citation_present = sum(bool(row.get("citation_present")) for row in scored_rows)
    retrieved_reference_present = sum(
        bool(row.get("retrieved_reference_present")) for row in scored_rows
    )
    evidence_present = sum(bool(row.get("evidence_present")) for row in scored_rows)
    target_gene_mentioned = sum(bool(row.get("target_gene_mentioned")) for row in scored_rows)
    anchored_true_positives = sum(
        bool(row.get("positive_evidence_anchor_pass")) for row in positives
    )
    evidence_anchored_correct = sum(
        bool(row.get("evidence_anchored_correct")) for row in scored_rows
    )
    parse_errors = sum(bool(row.get("parse_error")) for row in scored_rows)
    execution_errors = sum(bool(row.get("execution_error")) for row in scored_rows)
    return {
        "total": total,
        "positive_total": len(positives),
        "negative_total": len(negatives),
        "correct": correct,
        "accuracy": safe_div(correct, total),
        "accuracy_wilson_95ci": wilson_interval(correct, total),
        "balanced_accuracy": balanced_accuracy,
        "sensitivity_recall_positive": sensitivity,
        "sensitivity_wilson_95ci": wilson_interval(tp, len(positives)),
        "specificity_negative": specificity,
        "specificity_wilson_95ci": wilson_interval(tn, len(negatives)),
        "precision_positive": precision,
        "f1_positive": f1,
        "prediction_coverage": safe_div(total - unknown, total),
        "citation_presence_rate": safe_div(citation_present, total),
        "retrieved_reference_presence_rate": safe_div(retrieved_reference_present, total),
        "evidence_presence_rate": safe_div(evidence_present, total),
        "target_gene_mention_rate": safe_div(target_gene_mentioned, total),
        "evidence_anchored_positive_recovery_count": anchored_true_positives,
        "evidence_anchored_positive_recall": safe_div(anchored_true_positives, len(positives)),
        "evidence_anchored_positive_recall_wilson_95ci": wilson_interval(
            anchored_true_positives, len(positives)
        ),
        "evidence_anchored_correct": evidence_anchored_correct,
        "evidence_anchored_accuracy": safe_div(evidence_anchored_correct, total),
        "evidence_anchor_definition": (
            "For gold-positive items: predicted yes, target gene symbol appears in the "
            "answer/evidence, non-empty evidence is emitted, and the answer explicitly "
            "emits at least one citation. This is an automatic sanity check, not manual "
            "citation validity. Gold-negative items retain ordinary label correctness."
        ),
        "parse_error_count": parse_errors,
        "parse_error_rate": safe_div(parse_errors, total),
        "execution_error_count": execution_errors,
        "execution_error_rate": safe_div(execution_errors, total),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "unknown": unknown,
        "unknown_positive": unknown_positive,
        "unknown_negative": unknown_negative,
        "missing_predictions": sum(bool(row.get("missing_prediction")) for row in scored_rows),
    }


def stratified_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field) or "missing")].append(row)
    return {key: compute_metrics(value) for key, value in sorted(buckets.items())}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score_predictions(
    gold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]], prediction_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold = index_unique(gold_rows, Path("gold_key"))
    predictions = index_unique(prediction_rows, prediction_path)
    scored: list[dict[str, Any]] = []
    for item_id, gold_row in gold.items():
        pred_row = predictions.get(item_id)
        predicted = extract_prediction(pred_row) if pred_row else "unknown"
        text = prediction_text(pred_row or {})
        citations = answer_citation_values(pred_row or {})
        retrieved_references = retrieved_reference_values(pred_row or {})
        evidence = evidence_text(pred_row or {})
        target_gene_mentioned = mentions_target_gene(
            str(gold_row.get("gene_symbol", "")), " ".join((text, evidence))
        )
        positive_evidence_anchor_pass = (
            gold_row.get("answer") == "yes"
            and predicted == "yes"
            and target_gene_mentioned
            and bool(evidence.strip())
            and bool(citations)
        )
        label_correct = predicted == gold_row.get("answer")
        evidence_anchored_correct = label_correct and (
            gold_row.get("answer") != "yes" or positive_evidence_anchor_pass
        )
        scored.append(
            {
                "id": item_id,
                "benchmark_order": gold_row.get("benchmark_order", ""),
                "gene_symbol": gold_row.get("gene_symbol", ""),
                "disease_id": gold_row.get("disease_id", ""),
                "disease_name": gold_row.get("disease_name", ""),
                "task_subtype": gold_row.get("task_subtype", ""),
                "label_strength": gold_row.get("label_strength", ""),
                "question": gold_row.get("question", ""),
                "gold": gold_row.get("answer", ""),
                "predicted": predicted,
                "correct": int(label_correct),
                "evidence_anchored_correct": int(evidence_anchored_correct),
                "positive_evidence_anchor_pass": int(positive_evidence_anchor_pass),
                "missing_prediction": int(pred_row is None),
                "parse_error": int(bool(pred_row and pred_row.get("parse_error"))),
                "execution_error": int(bool(pred_row and pred_row.get("error"))),
                "prediction_text": text,
                "citations": ";".join(citations),
                "citation_present": int(bool(citations)),
                "retrieved_pmids": ";".join(retrieved_references),
                "retrieved_reference_present": int(bool(retrieved_references)),
                "evidence": evidence,
                "evidence_present": int(bool(evidence.strip())),
                "target_gene_mentioned": int(target_gene_mentioned),
                "label_source": gold_row.get("label_source", ""),
            }
        )

    metrics = compute_metrics(scored)
    metrics["prediction_rows_read"] = len(prediction_rows)
    metrics["extra_prediction_ids"] = sorted(set(predictions) - set(gold))
    metrics["predicted_label_counts"] = dict(sorted(Counter(row["predicted"] for row in scored).items()))
    metrics["by_disease"] = stratified_metrics(scored, "disease_id")
    metrics["by_task_subtype"] = stratified_metrics(scored, "task_subtype")
    metrics["by_label_strength"] = stratified_metrics(scored, "label_strength")
    return scored, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    gold_group = parser.add_mutually_exclusive_group(required=True)
    gold_group.add_argument("--gold-key", type=Path)
    gold_group.add_argument("--benchmark", type=Path, help="Backward-compatible alias for --gold-key")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    gold_path = args.gold_key or args.benchmark
    scored, metrics = score_predictions(
        load_jsonl(gold_path), load_jsonl(args.predictions), args.predictions
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "prediction_scores.csv", scored)
    (args.output_dir / "prediction_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
