#!/usr/bin/env python3
"""Verify saved experimental results using only package-local inputs."""

import csv
import json
import math
import runpy
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def experiment(name):
    return ROOT / name


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(rows, fields, expected):
    require(len(rows) == expected, f"Expected {expected} records, found {len(rows)}")
    require(len({tuple(str(row[key]) for key in fields) for row in rows}) == expected,
            f"Duplicate keys: {fields}")


def retrieval(config, allow_empty=False):
    rows = config["details"]
    unique(rows, ("category", "id"), 1045)
    for row in rows:
        pmids = row["retrieved_pmids"]
        require(not row.get("error"), "Failed retrieval")
        require(bool(pmids) or allow_empty, "Empty retrieval")
        require(len(pmids) == len(set(pmids)) and len(pmids) <= 10, "Invalid PMID list")
        rank = pmids.index(row["gold_pmid"]) + 1 if row["gold_pmid"] in pmids else None
        require(rank == row["gold_rank"], "Rank differs from returned PMID list")
    metrics = {f"recall@{k}": sum(row["gold_rank"] is not None and row["gold_rank"] <= k
                                for row in rows) / len(rows) for k in (1, 5, 10)}
    metrics["mrr@10"] = sum(1 / row["gold_rank"] if row["gold_rank"] else 0
                            for row in rows) / len(rows)
    for key, value in metrics.items():
        require(math.isclose(value, config["metrics"][key], abs_tol=1e-12), key)
    return metrics


def verify():
    checks = {}
    require(not any(p.is_symlink() for p in ROOT.rglob("*")), "External resource link")
    benchmark = read(ROOT / "data/benchmark.json")
    gold = {(category, qid): row for category, questions in benchmark.items() for qid, row in questions.items()}
    require(len(gold) == 1045, "Shared benchmark denominator")
    checks["inputs"] = "1,045 questions with options, gold labels and source PMIDs; no external links"

    root = experiment("pubmedqa") / "results/deepseek_controlled_comparison"
    responses = jsonl(root / "responses.jsonl")
    unique(responses, ("condition", "pmid"), 3000)
    for condition, expected in {"gold": 644, "closed_book": 144, "pharmkgpt_retrieval": 540}.items():
        rows = [row for row in responses if row["condition"] == condition]
        unique(rows, ("pmid",), 1000)
        require(all(not row.get("error") and row["prediction"] in {"yes", "no", "maybe"}
                    for row in rows), "PubMedQA failed/invalid answers")
        require(sum(row["prediction"] == row["ground_truth"] for row in rows) == expected, condition)
    checks["pubmedqa"] = "3,000 unique valid responses; correct=644/144/540"

    corrected = read(experiment("kg_ablation") / "results/corrected_retrieval/retrieval_only_ablation_results_corrected.json")
    for name, config in corrected["configs"].items():
        retrieval(config, allow_empty=True)
    gold_table = table(experiment("kg_ablation") / "results/summary/GOLD_RETRIEVAL_BENCHMARK.csv")
    for row in gold_table:
        if row["Configuration"] not in corrected["configs"]:
            continue
        metrics = retrieval(corrected["configs"][row["Configuration"]], allow_empty=True)
        for key in ("recall@1", "recall@5", "recall@10", "mrr@10"):
            column = key.replace("recall", "Recall").replace("mrr", "MRR") + " (%)"
            require(round(metrics[key] * 100, 2) == float(row[column]), column)
    impact = table(experiment("kg_ablation") / "results/summary/CORE_COMPONENT_IMPACT.csv")
    graph = next(row for row in impact if row["Component or Control"] == "Graph Filtering")
    require([int(graph[k]) for k in ("N", "PMID Set Changed", "Reference-only Correct", "Variant-only Correct")]
            == [1045, 775, 101, 59], "Answer perturbation table")
    checks["ablation"] = "6 x 1,045 corrected retrieval rows; one empty list in each of three semantic/no-KG arms retained as a miss; separate answer-perturbation table"

    matrix = table(experiment("effect_size") / "data/all_answer.csv")
    require(len(matrix) == 1045, "Answer matrix denominator")
    for row in matrix:
        expected = gold[(row["question_id"].split("_")[0], row["question_id"])]
        require(row["correct_option"] == expected["correct_option"], "Matrix gold labels")
    require(sum(row[f"pharmkgpt-{run}"].strip().upper() == row["correct_option"] for row in matrix for run in range(1, 6)) == 4996, "Five-run predictions")
    accuracies = table(experiment("effect_size") / "results/method_accuracy_summary.csv")
    pharm = next(row for row in accuracies if row["Category"] == "overall" and row["Method"] == "PharmkGPT")
    require(math.isclose(float(pharm["Accuracy"]), 4996 / 5225), "Historical five-run accuracy")
    checks["effect_sizes"] = "1,045-question matrix; PharmkGPT 4,996/5,225"

    human = read(experiment("free_form_qa") / "human_review/results/human_evaluation_summary.json")
    require((human["questions"], human["reviewers"], human["answer_level_ratings"]) == (100, 2, 400), "Human scores")
    key = table(experiment("free_form_qa") / "human_review/human_review_answer_key.csv")
    ids = {row["question_id"] for row in key}
    for reviewer in (1, 2):
        ratings = table(experiment("free_form_qa") / f"human_review/reviewer{reviewer}.csv")
        unique(ratings, ("question_id",), 100)
        require({row["question_id"] for row in ratings} == ids, "Rating IDs")
        require(all(row[f"answer_{side}_{metric}_0_2"] in {"0", "1", "2"} for row in ratings for side in ("a", "b") for metric in ("correctness", "evidence_support")), "Rating values")
    checks["human_qa"] = "100 questions; 400 anonymous answer ratings"

    paired = experiment("corpus_retrieval") / "results/paired"
    for arm in ("curated", "full"):
        retrieval(read(paired / arm / "retrieval.json")["configs"]["Full Model"])
    checks["corpus_scale"] = "2 x 1,045 valid retrieval lists; ranks and metrics recomputed"

    case_root = experiment("gene_case_study") / "case_study"
    require([len(table(case_root / name)) for name in
             ("group_delirium.csv", "group_dementia.csv", "group_delirium_DmP.csv", "group_delirium_DmN.csv")]
            == [20, 20, 20, 19], "Case-study groups")
    require(len(read(case_root / "answers.json")) == 150, "Case-study outputs")
    checks["case_study"] = "79 qualitative group rows; 150 workflow outputs; not 150 validated hypotheses"

    formal = experiment("itext2kg_kg2rag") / "results/formal"
    construction = read(formal / "official_itext2kg/construction_summary.json")
    require(construction["status"] == "complete" and construction["documents_accounted_for"] == 5059, "Construction")
    require(construction["successful_documents"] + construction["failed_documents"] == 5059, "Construction accounting")
    inventory = table(formal / "official_itext2kg/document_inventory.csv")
    unique(inventory, ("pmid",), 5059)
    require(sum(r["status"] == "success" for r in inventory) == construction["successful_documents"], "Construction successes")
    for column, total in (("entities", "total_entities"), ("relations", "total_relations")):
        require(sum(int(r[column]) for r in inventory) == construction[total], "Construction counts")
    answers = jsonl(formal / "answers/answers.jsonl")
    unique(answers, ("run", "category", "qid"), 5225)
    require(Counter(row["run"] for row in answers) == {i: 1045 for i in range(1, 6)}, "Five complete runs")
    require(all(row["predicted_option"] in {"A", "B", "C", "D"} for row in answers), "Answer parsing")
    require(read(formal / "evaluation/metrics.json")["valid_for_reviewer_claim"], "Formal completion gate")
    checks["itext2kg_kg2rag"] = "5,059 documents accounted for; 210 construction failures disclosed; 5,225 answers"

    root = experiment("external_kb") / "results"
    gold = {row["id"]: row["answer"] for row in jsonl(root / "benchmark_gold_key.jsonl")}
    require(len(gold) == 30, "External benchmark")
    for stem, expected in (("pharmkgpt_full_kg", 16), ("pharmkgpt_vanilla_rag", 11), ("no_retrieval_llm", 11)):
        rows = jsonl(root / "model_predictions" / f"{stem}_predictions.jsonl")
        unique(rows, ("id",), 30)
        require({row["id"] for row in rows} == set(gold), "External ID coverage")
        require(sum(row["prediction"] == gold[row["id"]] for row in rows) == expected, stem)
    checks["external_kb"] = "30 x 3 predictions; correct=16/11/11; manual citation scoring excluded"

    root = experiment("graphrag") / "results/v4_flash"
    answers = jsonl(root / "graphrag_results.jsonl")
    unique(answers, ("run", "category", "question_id"), 5225)
    for run, expected in enumerate((824, 824, 826, 823, 824), start=1):
        rows = [row for row in answers if row["run"] == run]
        unique(rows, ("category", "question_id"), 1045)
        require(sum(row["predicted_option"] == row["correct_option"] for row in rows) == expected, "GraphRAG accuracy")
    checks["graphrag"] = "Microsoft GraphRAG; 5 x 1,045 answer rows"

    for provider, expected in (("openai", 864), ("gemini", 897)):
        rows = jsonl(experiment("closed_models") / "results" / f"{provider}_answers.jsonl")
        unique(rows, ("category", "question_id"), 1045)
        require(sum(row["predicted_option"] == row["correct_option"] for row in rows) == expected, provider)
    checks["closed_models"] = "2 x 1,045 answers; correct=864/897"

    root = experiment("pubtator_provenance") / "results"
    provenance = read(root / "provenance_summary.json")
    require(provenance["coverage"]["complete"] and provenance["coverage"]["documents_successful"] == 5059, "Provenance coverage")
    require(provenance["configuration"]["analysis_mode"] == "historical_prompt_rerun", "Provisional provenance")
    rows = table(root / "entity_provenance.csv")
    counts = Counter(row["provenance"] for row in rows if row["stage"] == "final_connected")
    require(dict(counts) == provenance["final_connected_provenance"], "Per-occurrence provenance totals")
    require(sum(counts.values()) == 50605, "Final entity total")
    checks["provenance"] = "5,059 documents; 50,605 final occurrences; formal reconstruction"

    root = experiment("pubtator_ablation")
    build = read(root / "results/deletion_summary.json")
    require(build["formal_ready"] and build["counts"]["removed_entity_occurrences"] == 47188, "Ablation readiness")
    deletion = retrieval(read(root / "results/without_all_pubtator_linked_entities.json")["configs"]["Full Model"])
    require(round(deletion["recall@10"] * 100, 2) == 95.69, "Deletion Recall@10")
    checks["pubtator_ablation"] = "1,045 formal retrieval rows; 47,188 linked occurrences removed"

    require(len(table(experiment("plot/recall") / "results/topk_hit_rates.csv")) == 240, "Recall figure table")
    require(len(table(experiment("plot/accuracy") / "results/domain_accuracy_summary.csv")) == 54, "Accuracy figure table")
    recall = runpy.run_path(str(ROOT / "plot/recall/scripts/build_topk_hit_rates.py"))
    rebuilt, _ = recall["build_rows"](recall["DEFAULT_SOURCE"], recall["DEFAULT_CORRECTED_FULL_SOURCE"])
    stored = table(ROOT / "plot/recall/results/topk_hit_rates.csv")
    require([{k: str(v) for k, v in row.items()} for row in rebuilt] == stored, "Recomputed Recall panels")
    accuracy = runpy.run_path(str(ROOT / "plot/accuracy/scripts/build_accuracy_summary.py"))
    rebuilt, _ = accuracy["calculate_summary"](accuracy["SOURCE"])
    require([{k: str(v) for k, v in row.items()} for row in rebuilt] == table(ROOT / "plot/accuracy/results/domain_accuracy_summary.csv"), "Recomputed accuracy panels")
    checks["figures"] = "240 Recall@K points and 54 accuracy summaries recomputed from package inputs"
    return {"status": "passed", "mode": "offline_saved_artifact_verification", "checks": checks}


if __name__ == "__main__":
    report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
