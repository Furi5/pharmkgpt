from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def normalize_pmid(value: Any) -> str:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return f"pmid{digits}" if digits else ""


def rank_of(gold_pmid: str, retrieved_pmids: list[str]) -> int | None:
    try:
        return retrieved_pmids.index(gold_pmid) + 1
    except ValueError:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


