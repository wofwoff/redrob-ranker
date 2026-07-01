#!/usr/bin/env python3
"""Redrob ranking step: candidates.jsonl -> top-100 submission.csv.

Pure NumPy/SciPy. No GPU, no network, no hosted-LLM calls. Runs well inside the
5 min / 16 GB / CPU-only budget on the full 100K pool.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Optional booster: if a precomputed sentence-transformer artifact is passed via
--embeddings, the semantic component uses it; otherwise the self-contained
TF-IDF archetype classifier is used (fully reproducible, zero external deps).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

from redrob import scorer
from redrob.features import semantic
from redrob.loader import stream_candidates
from redrob.reasoning import compose

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_semantic_from_embeddings(path: str, cands, ideal_texts, anti_texts):
    """Optional path: score semantics from a precomputed embeddings artifact.

    Expects an .npz with arrays: ids (N,), emb (N,d, L2-normalized),
    ideal (d,), anti (d,). Aligns to candidate order by id.
    """
    z = np.load(path, allow_pickle=True)
    ids = list(z["ids"])
    emb, ideal, anti = z["emb"], z["ideal"], z["anti"]
    row = {cid: i for i, cid in enumerate(ids)}
    order = np.array([row[c["candidate_id"]] for c in cands])
    e = emb[order]
    diff = e.dot(ideal) - e.dot(anti)
    lo, hi = np.percentile(diff, 5), np.percentile(diff, 95)
    return np.full_like(diff, 0.5) if hi - lo < 1e-9 else np.clip((diff - lo) / (hi - lo), 0, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Redrob top-100 candidate ranker")
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl[.gz]")
    ap.add_argument("--out", required=True, help="output submission CSV path")
    ap.add_argument("--archetypes-dir", default=os.path.join(_HERE, "archetypes"))
    ap.add_argument("--embeddings", default=None,
                    help="optional precomputed embeddings .npz (else TF-IDF)")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args(argv)

    t0 = time.time()
    ideal_texts, anti_texts = semantic.load_archetypes(
        os.path.join(args.archetypes_dir, "ideal.txt"),
        os.path.join(args.archetypes_dir, "anti_pattern.txt"),
    )

    print(f"[rank] loading candidates from {args.candidates} ...", file=sys.stderr)
    cands = list(stream_candidates(args.candidates))
    print(f"[rank] loaded {len(cands):,} candidates in {time.time()-t0:.1f}s",
          file=sys.stderr)

    sem_scores = None
    if args.embeddings and os.path.exists(args.embeddings):
        print(f"[rank] using precomputed embeddings: {args.embeddings}", file=sys.stderr)
        sem_scores = _load_semantic_from_embeddings(
            args.embeddings, cands, ideal_texts, anti_texts)

    recs = scorer.score_pool(cands, ideal_texts, anti_texts, sem_scores)
    top = scorer.rank(recs, top_n=args.top)
    print(f"[rank] scored + ranked in {time.time()-t0:.1f}s", file=sys.stderr)

    # Emit strictly-decreasing scores: guarantees the validator's non-increasing
    # rule and makes the id-ascending tie rule vacuously true (no equal scores).
    rows = []
    prev = None
    for i, rec in enumerate(top, start=1):
        s = round(rec["final_score"], 6)
        if prev is not None and s >= prev:
            s = round(prev - 1e-6, 6)
        prev = s
        rows.append((rec["candidate_id"], i, f"{s:.6f}", compose(rec, i)))

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows)

    print(f"[rank] wrote {len(rows)} rows to {args.out} "
          f"(total {time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
