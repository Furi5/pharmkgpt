#!/usr/bin/env python3
"""Recompute provenance counts from saved document-entity occurrences."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
ROOT = Path(__file__).resolve().parents[1]


PROVENANCE_ORDER = (
    "shared_llm_pubtator",
    "pubtator_only",
    "llm_only",
    "unresolved",
)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_table(
    rows: list[dict[str, Any]], stage: str, denominator_label: str
) -> list[dict[str, Any]]:
    counts = Counter(row["provenance"] for row in rows if row["stage"] == stage)
    total = sum(counts.values())
    output: list[dict[str, Any]] = []
    for provenance in PROVENANCE_ORDER:
        count = counts.get(provenance, 0)
        if provenance == "unresolved" and stage == "candidate" and count == 0:
            continue
        output.append(
            {
                "stage": stage,
                "entity_provenance": provenance,
                "n": count,
                "percent": round(100.0 * count / total, 4) if total else 0.0,
                "denominator": denominator_label,
            }
        )
    output.append(
        {
            "stage": stage,
            "entity_provenance": "total",
            "n": total,
            "percent": 100.0 if total else 0.0,
            "denominator": denominator_label,
        }
    )
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    with (ROOT/'results/entity_provenance.csv').open() as f: rows=list(csv.DictReader(f))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    counts={}
    for stage,name,denominator in [('candidate','candidate_provenance','document-entity candidate occurrences'),('final_connected','final_connected_provenance','final connected document-entity occurrences')]:
        table=count_table(rows,stage,denominator)
        write_csv(args.output_dir/(name+'.csv'),['stage','entity_provenance','n','percent','denominator'],table)
        counts[name]={r['entity_provenance']:r['n'] for r in table if r['entity_provenance']!='total'}
    by_type=Counter((r['entity_type'],r['provenance']) for r in rows if r['stage']=='final_connected')
    write_csv(args.output_dir/'provenance_by_type.csv',['entity_type','entity_provenance','n'],[{'entity_type':t,'entity_provenance':p,'n':n} for (t,p),n in sorted(by_type.items())])
    summary=json.loads((ROOT/'results/provenance_summary.json').read_text())
    for name,value in counts.items():
        if summary[name]!=value: raise ValueError('Stored provenance totals differ from occurrence rows')
    for p in ('shared_llm_pubtator','pubtator_only','llm_only'):
        c=counts['candidate_provenance'].get(p,0); f=counts['final_connected_provenance'].get(p,0)
        actual={'candidate_n':c,'final_n':f,'retention_percent':round(100*f/c,4) if c else None}
        if summary['retention'][p]!=actual: raise ValueError('Stored retention differs from occurrence rows')
    print('Provenance: candidate, final and retention totals agree')

if __name__=='__main__': main()
