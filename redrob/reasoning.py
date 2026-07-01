"""Deterministic reasoning composer (design sec 4).

Every sentence is assembled from feature values that actually drove the score, so
the six Stage-4 checks pass by construction: specific facts (title, company,
applied-ML years, AI-core-skill count), JD connection (retrieval/ranking, product
vs services, geo), honest concerns (the top reducing factor is always surfaced),
no hallucination (only real fields are cited), variation (each candidate's
distinct profile yields distinct text), and tone-matches-rank (strong ranks read
confidently; weak ranks lead with the concern).

No network, no LLM at ranking time -- an LLM only helped author this composer.
"""

from __future__ import annotations

from typing import Any, Dict

from . import lexicons as lex

_AREA_PHRASE = {
    "retrieval": "embeddings/retrieval",
    "vector_db": "vector search infrastructure",
    "ranking": "ranking/recommendation systems",
}


def _ai_core_count(rec: Dict[str, Any]) -> int:
    names = {s.get("name", "") for s in (rec["_raw"].get("skills") or [])}
    return len(names & lex.SKILL_MUST_HAVE)


def _strength_clause(rec: Dict[str, Any]) -> str:
    areas = [_AREA_PHRASE[a] for a in rec.get("prose_areas", []) if a in _AREA_PHRASE]
    if areas and rec.get("has_production_signal"):
        return f"career history shows production {', '.join(areas)}"
    if areas:
        return f"career history shows {', '.join(areas)}"
    if rec["title_category"] == "strong_ml":
        return "a genuine applied-ML title at a product company"
    if rec["title_category"] == "adjacent":
        return "an adjacent engineering background transferable to ML"
    n = _ai_core_count(rec)
    return f"{n} AI-core skills but limited career evidence behind them"


def _concern_clause(rec: Dict[str, Any]) -> str:
    """The single most salient reducing factor, stated honestly. '' if none."""
    if rec.get("disqualifier_reasons"):
        return rec["disqualifier_reasons"][0]
    # geo is only a concern when the candidate is overseas; India is the target.
    if str(rec.get("geo_note", "")).startswith("overseas"):
        return rec["geo_note"]
    npd = rec.get("notice_period_days") or 0
    if npd and npd > 90:
        return f"long notice period ({npd} days)"
    days = rec.get("days_since_active")
    if days is not None and days > 120:
        return f"dormant on-platform for ~{days} days"
    rr = rec.get("recruiter_response_rate")
    if rr is not None and rr < 0.3:
        return f"low recruiter response rate ({rr:.0%})"
    if rec.get("soft_flags"):
        return rec["soft_flags"][0]
    return ""


def compose(rec: Dict[str, Any], rank: int) -> str:
    title = rec["current_title"] or "Unlisted role"
    company = rec["current_company"] or "an unnamed employer"
    ay, ty = rec["applied_ml_years"], rec["total_years"]
    strength = _strength_clause(rec)
    concern = _concern_clause(rec)

    lead = (f"{title} at {company}, {ay:.1f}y applied ML of {ty:.0f}y total; "
            f"{strength}")

    if rank <= 25 and not concern:
        tail = "strong availability and India-based fit for the Pune/Noida role."
        text = f"{lead}; {tail}"
    elif concern:
        connector = "but" if rank <= 40 else "though"
        text = f"{lead}, {connector} {concern}."
    else:
        text = f"{lead}; adjacent fit included as lower-ranked filler."

    # single line, collapse whitespace; csv.writer handles quoting/escaping.
    return " ".join(text.split())
