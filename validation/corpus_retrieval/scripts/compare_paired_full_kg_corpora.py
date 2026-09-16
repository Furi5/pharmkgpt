#!/usr/bin/env python3
"""Compare the saved paired retrieval rankings for both corpus sizes."""
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]


def exact_mcnemar_p(system_1_only: int, system_2_only: int) -> float:
    discordant = system_1_only + system_2_only
    if discordant == 0:
        return 1.0
    tail = min(system_1_only, system_2_only)
    probability = sum(math.comb(discordant, i) for i in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_mean_ci(
    differences: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(differences)
    estimates = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def load_arm(path):
    data=json.loads(path.read_text())['configs']['Full Model']['details']
    gold=json.loads((ROOT.parent/'data/benchmark.json').read_text())
    rows={}
    for row in data:
        key=(row['category'],str(row['id']))
        pmids=row['retrieved_pmids']
        if key in rows or row.get('error') or not pmids or len(pmids)>10 or len(pmids)!=len(set(pmids)):
            raise ValueError('Invalid or duplicate retrieval')
        if row['gold_pmid'] != gold[key[0]][key[1]]['pmid']:
            raise ValueError('Gold PMID differs from benchmark')
        rank=pmids.index(row['gold_pmid'])+1 if row['gold_pmid'] in pmids else None
        if rank!=row['gold_rank']: raise ValueError('Rank differs from PMID list')
        rows[key]={'rank':rank,'reciprocal_rank':1/rank if rank else 0,'gold':row['gold_pmid']}
    if len(rows)!=1045: raise ValueError('Incomplete retrieval')
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/paired/comparison')
    args=parser.parse_args()
    source=ROOT/'results/paired'
    curated,full=(load_arm(source/arm/'retrieval.json') for arm in ('curated','full'))
    if set(curated)!=set(full): raise ValueError('Paired question sets differ')
    keys=sorted(curated); total=len(keys); rows=[]
    for k in (1,5,10):
        left=[curated[q]['rank'] is not None and curated[q]['rank']<=k for q in keys]
        right=[full[q]['rank'] is not None and full[q]['rank']<=k for q in keys]
        c_only=sum(c and not f for c,f in zip(left,right)); f_only=sum(f and not c for c,f in zip(left,right))
        rows.append(dict(metric=f'Recall@{k}',curated=sum(left)/total,full=sum(right)/total,full_minus_curated=(sum(right)-sum(left))/total,curated_only_hit=c_only,full_only_hit=f_only,both_hit=sum(c and f for c,f in zip(left,right)),both_miss=sum(not c and not f for c,f in zip(left,right)),p=exact_mcnemar_p(c_only,f_only)))
    differences=[full[q]['reciprocal_rank']-curated[q]['reciprocal_rank'] for q in keys]
    ci=paired_bootstrap_mean_ci(differences,samples=10000,seed=20260716)
    rows.append(dict(metric='MRR@10',curated=sum(r['reciprocal_rank'] for r in curated.values())/total,full=sum(r['reciprocal_rank'] for r in full.values())/total,full_minus_curated=sum(differences)/total,ci95_low=ci[0],ci95_high=ci[1]))
    payload=dict(status='complete',questions=total,protocol=json.loads((source/'protocol.json').read_text()),metrics=rows,bootstrap_samples=10000,seed=20260716)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'paired_comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (args.output_dir/'paired_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row))); w.writeheader(); w.writerows(rows)
    with (args.output_dir/'per_question_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['category','qid','gold_pmid','curated_rank','full_rank','rr_difference'])
        w.writerows([*q,curated[q]['gold'],curated[q]['rank'],full[q]['rank'],d] for q,d in zip(keys,differences))
    print('Corpus comparison: 1,045 paired queries evaluated')

if __name__=='__main__': main()
