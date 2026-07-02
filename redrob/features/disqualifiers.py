"""Multiplicative disqualifier knockdowns from the JD's "we will not move
forward" / "we explicitly do NOT want" lists (design sec 3b).

Each check is a computable feature *and* a citable sentence for the reasoning
column. They are applied as a product of per-flag multipliers so a single soft
flag dampens rather than annihilates -- except pure-research, which the JD is
explicit about ("we will not move forward"), where a near-zero is appropriate.

github_activity_score is deliberately NOT used as evidence for the
closed-source flag: "no GitHub linked" is a profile fact (~65% of the pool),
not a career fact. A high score may help in education.py; it never penalizes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .. import lexicons as lex
from . import role_match
from ..loader import career, education, profile, signals
from ..text import word_matcher

_CODING_PROSE = word_matcher([
    "built", "developed", "implemented", "coded", "python", "shipped", "wrote",
    "prototype", "prototyped", "model", "models", "pipeline", "pipelines",
    "deployed",
])
_PROD_PROSE = word_matcher([
    "production", "shipped", "real users", "at scale", "deployed", "serving",
    "customers",
])


def _companies(c: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    p = profile(c)
    out = [(p.get("current_company", ""), p.get("current_industry", ""),
            p.get("current_company_size", ""))]
    out += [(r.get("company", ""), r.get("industry", ""), r.get("company_size", ""))
            for r in career(c)]
    return [(a, b, s) for a, b, s in out if a]


def _has_product(c: Dict[str, Any]) -> bool:
    for comp, ind, _ in _companies(c):
        if comp in lex.PRODUCT_COMPANIES:
            return True
        if comp in lex.GENERIC_COMPANIES and ind not in lex.SERVICES_INDUSTRIES \
                and ind not in lex.NON_TECH_INDUSTRIES:
            return True
    return False


def _all_text(c: Dict[str, Any]) -> str:
    p = profile(c)
    return "\n".join([p.get("summary", "")]
                     + [r.get("description", "") for r in career(c)]).lower()


# --- individual checks: each returns (multiplier, reason|None) ------------- #

def _consultancy_only(c: Dict[str, Any]):
    has_consultancy = any(
        comp in lex.CONSULTANCY_COMPANIES or ind in lex.SERVICES_INDUSTRIES
        for comp, ind, _ in _companies(c))
    if has_consultancy and not _has_product(c):
        return 0.35, "career entirely at services/consulting firms with no product-company stint"
    return 1.0, None


def _pure_research(c: Dict[str, Any]):
    cur = profile(c).get("current_title", "")
    has_phd = any(e.get("degree", "") == "Ph.D" for e in education(c))
    research_title = cur in lex.TITLE_RESEARCH
    production = bool(_PROD_PROSE.search(_all_text(c)))
    if (research_title or has_phd) and not production and not _has_product(c):
        return 0.20, "pure-research profile with no evidence of production deployment"
    return 1.0, None


def _langchain_only(c: Dict[str, Any]):
    text = _all_text(c)
    skill_names = {s.get("name", "") for s in c.get("skills", []) or []}
    llm_wrapper = {"LangChain", "LlamaIndex", "Prompt Engineering", "RAG", "LLMs"}
    classic_ml = {"Machine Learning", "Deep Learning", "PyTorch", "TensorFlow",
                  "scikit-learn", "Statistical Modeling", "Feature Engineering",
                  "Learning to Rank", "Recommendation Systems"}
    has_wrapper = bool(skill_names & llm_wrapper) or "langchain" in text
    has_classic = bool(skill_names & classic_ml)
    total_years = profile(c).get("years_of_experience", 0) or 0
    if has_wrapper and not has_classic and total_years < 3:
        return 0.50, "recent LangChain/LLM-wrapper work without substantial pre-LLM ML"
    return 1.0, None


def _seniority(title: str) -> int:
    t = (title or "").lower()
    if "junior" in t:
        return 0
    if any(k in t for k in ("staff", "principal", "lead", "head")):
        return 3
    if "senior" in t:
        return 2
    return 1


def _title_chaser(c: Dict[str, Any]):
    roles = career(c)
    if len(roles) < 3:
        return 1.0, None
    # career_history newest-first is typical; use average tenure + ascending titles.
    durs = [r.get("duration_months", 0) or 0 for r in roles]
    avg_tenure = sum(durs) / len(durs)
    sen = [_seniority(r.get("title", "")) for r in roles]
    ascending = sen[0] > sen[-1]  # newest more senior than oldest
    if avg_tenure < 20 and ascending and sen[0] >= 2:
        return 0.55, f"title-chaser pattern: ~{avg_tenure/12:.1f}y average tenure with ascending titles"
    return 1.0, None


def _stopped_coding(c: Dict[str, Any]):
    for r in career(c):
        if not r.get("is_current"):
            continue
        # Hands-on ML titles (Lead AI Engineer, Staff MLE, ...) are exempt: the
        # JD's disqualifier targets people who LEFT engineering for
        # architecture/tech-lead roles, not IC-leadership ML roles. The audit
        # found a referee-tier-5 "Lead AI Engineer @ Sarvam AI" (owned search
        # end-to-end) buried by this flag purely because his description used
        # outcome language instead of coding verbs.
        if role_match.title_category(r.get("title", "")) == "strong_ml":
            continue
        title = (r.get("title", "") or "").lower()
        is_leadership = any(k in title for k in ("lead", "staff", "principal",
                                                 "architect", "manager", "head"))
        months = r.get("duration_months", 0) or 0
        desc = (r.get("description", "") or "").lower()
        codes = bool(_CODING_PROSE.search(desc))
        if is_leadership and months >= 18 and not codes:
            return 0.55, f"in a non-coding leadership role for ~{months} months"
    return 1.0, None


def _cv_speech_primary(c: Dict[str, Any]):
    cur = profile(c).get("current_title", "")
    skill_names = {s.get("name", "") for s in c.get("skills", []) or []}
    cv_count = len(skill_names & lex.SKILL_CV_SPEECH)
    ir_count = len(skill_names & (lex.SKILL_RETRIEVAL_EMBEDDINGS
                                  | lex.SKILL_VECTOR_DB | lex.SKILL_RANKING_IR))
    text = _all_text(c)
    ir_prose = any(k in text for k in ("retrieval", "ranking", "nlp", "search",
                                       "recommendation", "embedding"))
    cv_primary = cur in lex.TITLE_CV_SPEECH or (cv_count >= 3 and cv_count > ir_count)
    if cv_primary and ir_count == 0 and not ir_prose:
        return 0.45, "primary expertise is computer-vision/speech without NLP/IR exposure"
    return 1.0, None


def _closed_source(c: Dict[str, Any]):
    total_years = profile(c).get("years_of_experience", 0) or 0
    n_certs = len(c.get("certifications", []) or [])
    text = _all_text(c)
    open_signal = "open source" in text or "open-source" in text or n_certs > 0
    # Mild, evidence-thin flag; certifications/open-source offset it entirely.
    if total_years >= 5 and not open_signal and not _has_product(c):
        return 0.85, "5+ years entirely on proprietary systems without external validation"
    return 1.0, None


_CHECKS = (_consultancy_only, _pure_research, _langchain_only, _title_chaser,
           _stopped_coding, _cv_speech_primary, _closed_source)


def score(c: Dict[str, Any]) -> Dict[str, Any]:
    mult = 1.0
    reasons: List[str] = []
    for check in _CHECKS:
        m, reason = check(c)
        if reason:
            mult *= m
            reasons.append(reason)
    mult = max(0.05, mult)  # never a true zero here; consistency owns the killswitch
    return {"disqualifier_mult": round(mult, 4), "disqualifier_reasons": reasons}
