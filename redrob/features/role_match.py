"""Role/title relevance -- the single most discriminative base_fit component.

This is what crushes the "Marketing Manager with 9 AI skills" stuffer: the JD is
explicit that title + real product-company context, not the skills list, decides
fit. We score the best of the current title and recent history (a strong ML past
at a product company still counts if the person is currently elsewhere), then
modulate by whether that role was at a product vs. services vs. non-tech shop.
"""

from __future__ import annotations

from typing import Any, Dict

from .. import lexicons as lex
from ..loader import career, profile

# Title -> base relevance for a Senior AI Engineer (retrieval/ranking) role.
_TITLE_BASE = {
    "strong_ml": 1.00,
    "research": 0.62,      # applied bits ok; pure-research knockdown handled in disqualifiers
    "cv_speech": 0.45,     # down-weighted specialty; resolved against NLP/IR evidence
    "adjacent": 0.55,      # SWE / data eng: transferable, prose decides how close
    "weak_tech": 0.30,
    "non_tech": 0.05,      # the stuffer trap surface
    "unknown": 0.20,
}


def title_category(title: str) -> str:
    t = (title or "").strip()
    if t in lex.TITLE_STRONG_ML:
        return "strong_ml"
    if t in lex.TITLE_RESEARCH:
        return "research"
    if t in lex.TITLE_CV_SPEECH:
        return "cv_speech"
    if t in lex.TITLE_ADJACENT_TECH:
        return "adjacent"
    if t in lex.TITLE_WEAK_TECH:
        return "weak_tech"
    if t in lex.TITLE_NON_TECH:
        return "non_tech"
    return "unknown"


def company_factor(company: str, company_size: str, industry: str) -> float:
    """Product-vs-services multiplier for a single role's employer.

    Product context is what the JD wants ("not pure services"). Named/close
    consultancies and IT-services industries are damped; genuine product and
    AI-product employers are full credit.
    """
    company = (company or "").strip()
    industry = (industry or "").strip()

    if company in lex.AI_PRODUCT_COMPANIES:
        return 1.00
    if company in lex.PRODUCT_COMPANIES:
        return 1.00
    if company in lex.CONSULTANCY_COMPANIES:
        return 0.72
    # Unlisted / generic employer: fall back to industry as the services tell.
    if industry in lex.SERVICES_INDUSTRIES:
        return 0.75
    if industry in lex.NON_TECH_INDUSTRIES:
        return 0.70
    return 0.92  # generic placeholder employer in a product-ish industry


def _role_value(title: str, company: str, size: str, industry: str) -> float:
    base = _TITLE_BASE[title_category(title)]
    if "junior" in (title or "").lower():   # senior role: junior titles temper
        base *= 0.85
    return base * company_factor(company, size, industry)


def score(c: Dict[str, Any]) -> Dict[str, Any]:
    p = profile(c)
    cur = _role_value(
        p.get("current_title", ""), p.get("current_company", ""),
        p.get("current_company_size", ""), p.get("current_industry", ""),
    )

    # Best historical role, lightly recency-discounted, so real product-ML
    # experience is rewarded even for someone currently at a services firm.
    best_hist = 0.0
    for r in career(c):
        best_hist = max(best_hist, _role_value(
            r.get("title", ""), r.get("company", ""),
            r.get("company_size", ""), r.get("industry", ""),
        ))

    role_score = max(cur, 0.90 * best_hist)
    return {
        "role_score": round(role_score, 4),
        "title_category": title_category(p.get("current_title", "")),
        "current_title": p.get("current_title", ""),
        "current_company": p.get("current_company", ""),
    }
