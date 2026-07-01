#!/usr/bin/env python3
"""OPTIONAL offline booster: precompute sentence-transformer embeddings.

This step is allowed to exceed the 5-minute window and to use the network (to
download the model once) -- it is NOT the ranking step. It writes an .npz that
rank.py consumes via --embeddings; rank.py itself never loads a model or touches
the network. If you never run this, the ranker falls back to the self-contained
TF-IDF archetype classifier and produces a valid submission regardless.

    pip install sentence-transformers torch
    python precompute_embeddings.py --candidates ./candidates.jsonl \
        --out artifacts/embeddings.npz

The artifact (~154 MB float32 for 100K x 384) exceeds GitHub's 100 MB limit --
keep it out of git (see .gitignore) and regenerate, or store via Git LFS.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from redrob.features import semantic                        # noqa: E402
from redrob.loader import profile_text, stream_candidates   # noqa: E402

MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, small, CPU-friendly


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="artifacts/embeddings.npz")
    ap.add_argument("--archetypes-dir", default="archetypes")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed; this is the OPTIONAL booster.\n"
              "  pip install sentence-transformers torch", file=sys.stderr)
        return 1

    model = SentenceTransformer(args.model)

    ids, texts = [], []
    for c in stream_candidates(args.candidates):
        ids.append(c["candidate_id"])
        texts.append(profile_text(c))

    emb = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                       show_progress_bar=True).astype(np.float32)

    ideal_texts, anti_texts = semantic.load_archetypes(
        os.path.join(args.archetypes_dir, "ideal.txt"),
        os.path.join(args.archetypes_dir, "anti_pattern.txt"))
    ideal = _unit(model.encode(ideal_texts, normalize_embeddings=True).mean(axis=0))
    anti = _unit(model.encode(anti_texts, normalize_embeddings=True).mean(axis=0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, ids=np.array(ids), emb=emb, ideal=ideal, anti=anti)
    print(f"[precompute] wrote {emb.shape} embeddings + archetypes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
