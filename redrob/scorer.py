"""Combine features into base_fit and the multiplicative final score.

    final_score = base_fit
                x disqualifier_mult
                x geo_fit
                x availability
                x consistency_factor

Multiplication means a candidate must clear *every* gate: a stuffer with great
skills but a Marketing title is zeroed by the title-driven base_fit; an
impossible profile is zeroed by the consistency killswitch; a ghost is capped by
availability; a non-relocating overseas candidate is capped by geo_fit.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import text as _text
from .features import (availability, consistency, disqualifiers, education,
                       experience, geo_fit, must_have, role_match, semantic)
from .loader import parse_date, profile_text

# base_fit blend (design sec 3a). Weights sum to 1.0; title does the heavy lifting.
BASE_WEIGHTS = {
    "role": 0.38,
    "semantic": 0.25,
    "must_have": 0.17,
    "experience": 0.13,
    "education": 0.07,
}


def reference_date(cands: Sequence[Dict[str, Any]]) -> _dt.date:
    """Deterministic 'now' = latest last_active_date in the pool (no wall-clock)."""
    best: Optional[_dt.date] = None
    for c in cands:
        d = parse_date((c.get("redrob_signals") or {}).get("last_active_date"))
        if d and (best is None or d > best):
            best = d
    return best or _dt.date(2025, 1, 1)


def score_pool(cands: List[Dict[str, Any]],
               ideal_texts: Sequence[str], anti_texts: Sequence[str],
               semantic_scores: Optional[np.ndarray] = None,
               disable: Optional[set] = None) -> List[Dict[str, Any]]:
    """Score every candidate. Returns one record dict per candidate.

    ``disable`` (for ablation) may contain base components ('role', 'semantic',
    'must_have', 'experience', 'education' -> weight zeroed) and/or multipliers
    ('disqualifier', 'geo', 'availability', 'consistency' -> forced to 1.0).
    """
    disable = disable or set()
    ref = reference_date(cands)

    # Per-candidate feature pass (interpretable parts kept for reasoning/ablation).
    recs: List[Dict[str, Any]] = []
    corpus: List[str] = []
    for c in cands:
        r = role_match.score(c)
        mh = must_have.score(c)
        ex = experience.score(c)
        ed = education.score(c)
        dq = disqualifiers.score(c)
        gf = geo_fit.score(c)
        av = availability.score(c, ref)
        cons = consistency.score(c)
        recs.append({
            "candidate_id": c["candidate_id"],
            "_raw": c,
            **r, **mh, **ex, **ed, **dq, **gf, **av, **cons,
        })
        corpus.append(profile_text(c))

    # Semantic component: use precomputed embeddings if provided, else TF-IDF.
    if semantic_scores is None:
        matrix, vec = _text.build_tfidf(corpus)
        sem = semantic.archetype_scores(matrix, vec, ideal_texts, anti_texts)
    else:
        sem = np.asarray(semantic_scores, dtype=float)

    def _w(name, key):
        return 0.0 if name in disable else BASE_WEIGHTS[key]

    def _m(name, value):
        return 1.0 if name in disable else value

    for i, rec in enumerate(recs):
        rec["semantic_score"] = round(float(sem[i]), 4)
        base = (_w("role", "role") * rec["role_score"]
                + _w("semantic", "semantic") * rec["semantic_score"]
                + _w("must_have", "must_have") * rec["must_have_score"]
                + _w("experience", "experience") * rec["experience_score"]
                + _w("education", "education") * rec["education_score"])
        rec["base_fit"] = round(base, 5)
        final = (base
                 * _m("disqualifier", rec["disqualifier_mult"])
                 * _m("geo", rec["geo_fit"])
                 * _m("availability", rec["availability"])
                 * _m("consistency", rec["consistency_factor"]))
        rec["final_score"] = round(final, 6)
    return recs


def rank(recs: List[Dict[str, Any]], top_n: int = 100) -> List[Dict[str, Any]]:
    """Sort by score desc, candidate_id asc (matches the validator tie rule)."""
    ordered = sorted(recs, key=lambda r: (-r["final_score"], r["candidate_id"]))
    return ordered[:top_n]
