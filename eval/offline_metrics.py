#!/usr/bin/env python3
"""Offline metrics: the exact composite the challenge scores on (spec sec 4).

    composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10

Used to calibrate against a gold set (eval/gold_labels.csv) without a leaderboard
and to run ablations (turn a scoring component on/off and watch these move).

Relevance is graded 0-5 per the JD rubric; P@10 counts tier >= 3 as "relevant"
(spec: P@10 = fraction of top-10 that are relevant, tier 3+).
"""

from __future__ import annotations

import argparse
import csv
import math
from typing import Dict, List, Sequence

REL_THRESHOLD = 3  # "relevant" = tier 3+


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_rels: Sequence[float], all_rels: Sequence[float], k: int) -> float:
    ideal = sorted(all_rels, reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(list(ranked_rels)[:k]) / idcg


def average_precision(ranked_rels: Sequence[float], n_relevant_total: int) -> float:
    if n_relevant_total == 0:
        return 0.0
    hits, score = 0, 0.0
    for i, r in enumerate(ranked_rels, start=1):
        if r >= REL_THRESHOLD:
            hits += 1
            score += hits / i
    return score / n_relevant_total


def precision_at_k(ranked_rels: Sequence[float], k: int) -> float:
    top = list(ranked_rels)[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r >= REL_THRESHOLD) / k


def composite(ranked_rels: Sequence[float], all_rels: Sequence[float],
              n_relevant_total: int) -> Dict[str, float]:
    m = {
        "NDCG@10": ndcg_at_k(ranked_rels, all_rels, 10),
        "NDCG@50": ndcg_at_k(ranked_rels, all_rels, 50),
        "MAP": average_precision(ranked_rels, n_relevant_total),
        "P@10": precision_at_k(ranked_rels, 10),
    }
    m["composite"] = (0.50 * m["NDCG@10"] + 0.30 * m["NDCG@50"]
                      + 0.15 * m["MAP"] + 0.05 * m["P@10"])
    return m


def evaluate(submission_csv: str, gold_csv: str) -> Dict[str, float]:
    """Score a submission CSV against a gold_labels.csv (candidate_id,relevance)."""
    gold: Dict[str, float] = {}
    with open(gold_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold[row["candidate_id"]] = float(row["relevance"])

    ranked: List[tuple] = []
    with open(submission_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ranked.append((int(row["rank"]), row["candidate_id"]))
    ranked.sort()
    # Candidates in the submission but absent from gold are treated as tier 0.
    ranked_rels = [gold.get(cid, 0.0) for _, cid in ranked]

    all_rels = list(gold.values())
    n_relevant_total = sum(1 for r in all_rels if r >= REL_THRESHOLD)
    return composite(ranked_rels, all_rels, n_relevant_total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--gold", required=True)
    args = ap.parse_args()
    m = evaluate(args.submission, args.gold)
    for k, v in m.items():
        print(f"{k:12s} {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
