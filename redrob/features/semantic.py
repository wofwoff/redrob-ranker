"""Zero-label semantic classifier (design sec 3f).

No training labels exist (ground truth is hidden), so we build a label-free
classifier from archetype paragraphs: a handful of "ideal candidate" paragraphs
and a handful of "anti-pattern" paragraphs (keyword stuffer, pure-research
academic, LangChain-tutorial enthusiast, consultancy-only). We embed everything
in the same TF-IDF space, average each group into an ideal_vector / anti_vector,
and score each candidate as sim(candidate, ideal) - sim(candidate, anti).

This is computed once over the whole pool (vectorized), then min-max scaled to
[0,1] using pool percentiles so the component is stable and comparable.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy import sparse

from ..text import TfidfVectorizer


def _mean_unit(vecs: sparse.csr_matrix) -> np.ndarray:
    v = np.asarray(vecs.mean(axis=0)).ravel()
    n = np.linalg.norm(v)
    return v / n if n else v


def archetype_scores(matrix: sparse.csr_matrix, vec: TfidfVectorizer,
                     ideal_texts: Sequence[str], anti_texts: Sequence[str]) -> np.ndarray:
    """Return an (N,) array in [0,1]: higher = closer to ideal, further from anti."""
    ideal = _mean_unit(vec.transform(list(ideal_texts)))
    anti = _mean_unit(vec.transform(list(anti_texts)))

    sim_ideal = matrix.dot(ideal)          # rows are L2-normalized -> cosine
    sim_anti = matrix.dot(anti)
    diff = np.asarray(sim_ideal - sim_anti).ravel()

    lo, hi = np.percentile(diff, 5), np.percentile(diff, 95)
    if hi - lo < 1e-9:
        return np.full_like(diff, 0.5)
    return np.clip((diff - lo) / (hi - lo), 0.0, 1.0)


def load_archetypes(ideal_path: str, anti_path: str) -> tuple[List[str], List[str]]:
    """Read archetype paragraphs (blank-line separated) from the two txt files."""
    def _read(p: str) -> List[str]:
        with open(p, "r", encoding="utf-8") as f:
            blocks = [b.strip() for b in f.read().split("\n\n")]
        return [b for b in blocks if b and not b.startswith("#")]
    return _read(ideal_path), _read(anti_path)
