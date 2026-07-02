"""Real applied-ML years, derived from career_history (design sec 3a).

The JD wants ~4-5 years in applied ML/AI at product companies, ~6-8 total -- and
is explicit that the *real* number comes from career history, not the raw
``years_of_experience`` field (which a stuffer can inflate). We weight each
role's months by how ML-relevant the title is and whether it was at a product
company, so 3 years as an "ML Engineer at Swiggy" counts and 8 years as an
"Accountant" does not.
"""

from __future__ import annotations

from typing import Any, Dict

from . import role_match
from ..description_tiers import tier_for
from ..loader import career, profile
from ..text import word_matcher

# ML-relevance weight per title category (fallback for off-vocabulary roles).
_ML_WEIGHT = {
    "strong_ml": 1.00,
    "research": 0.70,
    "cv_speech": 0.60,
    "adjacent": 0.45,   # data/SWE roles: partial, boosted when prose shows ML
    "weak_tech": 0.15,
    "non_tech": 0.02,
    "unknown": 0.10,
}
# ML-relevance weight per description tier (primary): a role's templated
# description measures how much of the work was actually applied ML, far more
# precisely than the title (a 'Data Scientist' on a tier-2 churn template counts
# less than one on a tier-5 ranking template).
_TIER_WEIGHT = {5: 1.00, 4: 1.00, 3: 0.90, 2: 0.60, 1: 0.25, 0: 0.02}
_ML_PROSE = word_matcher([
    "machine learning", "ml", "model", "models", "embedding", "embeddings",
    "retrieval", "ranking", "recommendation", "recommender", "nlp",
    "deep learning", "llm", "llms",
])


def _role_ml_weight(r: Dict[str, Any]) -> float:
    """How much of this role's months count as applied ML. Primary: description
    tier; fallback: title category (+ prose boost) for off-vocabulary text."""
    t = tier_for(r.get("description", "") or "")
    if t is not None:
        return _TIER_WEIGHT[t]
    cat = role_match.title_category(r.get("title", ""))
    w = _ML_WEIGHT[cat]
    if cat in ("adjacent", "weak_tech", "unknown"):
        if _ML_PROSE.search((r.get("description", "") or "").lower()):
            w = max(w, 0.60)
    return w


def _applied_months(c: Dict[str, Any]) -> float:
    total = 0.0
    for r in career(c):
        w = _role_ml_weight(r)
        pf = role_match.company_factor(
            r.get("company", ""), r.get("company_size", ""), r.get("industry", ""))
        total += (r.get("duration_months", 0) or 0) * w * (0.6 + 0.4 * pf)
    return total


def _curve(applied_years: float) -> float:
    if applied_years < 1:
        return 0.20
    if applied_years < 4:
        return 0.20 + 0.70 * (applied_years - 1) / 3.0   # 0.20 -> 0.90
    if applied_years <= 9:
        return 1.00                                       # JD sweet spot
    if applied_years <= 12:
        return 0.90
    return 0.80


def score(c: Dict[str, Any]) -> Dict[str, Any]:
    applied_months = _applied_months(c)
    applied_years = applied_months / 12.0
    total_years = profile(c).get("years_of_experience", 0) or 0

    exp = _curve(applied_years)
    if 5 <= total_years <= 9:          # JD's stated band, mild positive
        exp = min(1.0, exp + 0.05)
    return {
        "experience_score": round(exp, 4),
        "applied_ml_years": round(applied_years, 2),
        "total_years": round(float(total_years), 1),
    }
