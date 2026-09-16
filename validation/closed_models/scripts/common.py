from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = EXPERIMENT_ROOT.parent/'data/benchmark.json'


def load_benchmark(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for category, questions in data.items():
        if not isinstance(questions, dict):
            continue
        for question_id, item in questions.items():
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            correct = str(item.get("correct_option", "")).strip().upper()
            if question and correct:
                rows.append(
                    {
                        "category": str(category),
                        "question_id": str(question_id),
                        "question": question,
                        "options": str(item.get("options", "")).strip(),
                        "correct_option": correct,
                    }
                )
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


