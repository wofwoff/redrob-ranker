"""Education + validation -- a minor base_fit component (design sec 3a, ~0.07).

Institution tier and field relevance are weak JD signals; certifications and a
*high* github_activity_score act as external validation. Per design sec 3b,
GitHub is a positive-only signal: -1 ("no GitHub linked", ~65% of the pool) and
low scores never penalize here.
"""

from __future__ import annotations

from typing import Any, Dict

from .. import lexicons as lex
from ..loader import education, signals

_TIER_W = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.45,
           "unknown": 0.5}


def _field_relevance(c: Dict[str, Any]) -> float:
    best = 0.0
    for e in education(c):
        f = e.get("field_of_study", "")
        if f in lex.FIELD_HIGHLY_RELEVANT:
            best = max(best, 1.0)
        elif f in lex.FIELD_RELEVANT:
            best = max(best, 0.7)
        else:
            best = max(best, 0.3)
    return best


def _tier(c: Dict[str, Any]) -> float:
    best = 0.5
    for e in education(c):
        best = max(best, _TIER_W.get(e.get("tier", "unknown"), 0.5))
    return best


def score(c: Dict[str, Any]) -> Dict[str, Any]:
    sig = signals(c)
    field = _field_relevance(c)
    tier = _tier(c)

    n_certs = len(c.get("certifications", []) or [])
    cert_bonus = min(0.15, 0.05 * n_certs)

    gh = sig.get("github_activity_score", -1)
    gh_bonus = 0.15 * (gh / 100.0) if gh and gh > 0 else 0.0  # positive-only

    edu = 0.55 * field + 0.30 * tier + cert_bonus + gh_bonus
    return {
        "education_score": round(max(0.0, min(1.0, edu)), 4),
        "field_relevance": round(field, 2),
        "n_certifications": n_certs,
        "github_activity_score": gh,
    }
