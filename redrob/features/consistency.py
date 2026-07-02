"""Two-tier consistency (design sec 3e).

Tier A -- hard-impossible contradictions -> killswitch (factor 0). Logically
un-fakeable; a single one zeroes the candidate. Two sources:

* *Internal* contradictions (dates, durations, proficiency-vs-usage) -- these
  fire on ~43/100K candidates.
* *World-knowledge* contradictions: a role starting years before the employer
  was founded (lexicons.COMPANY_FOUNDED_YEAR). This is the spec's first
  honeypot archetype ("8 years at a company founded 3 years ago") -- internally
  coherent, so only checkable against public founding years. The generator's
  noise band extends ~1-2y pre-founding, so >= 3 years is the hard threshold
  (audit: the honeypots found in our own top-100 sat at 5y and 3y violations,
  wrapped in suspiciously perfect profiles); exactly 2y is a Tier-B soft
  anomaly; 1y is ignored as noise/stealth-mode slack.

See eval/honeypot_selftest.py, which asserts all of these actually fire.

Tier B -- soft anomalies -> penalty multiplier, never fatal. Split into
generator-noise (very mild) and quality-relevant (heavier) bands.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .. import lexicons as lex
from ..description_tiers import tier_for
from ..loader import career, parse_date, profile, signals

# Margins chosen from the data audit: hard flags fire on ~45/100K candidates.
_SPAN_TOLERANCE_MONTHS = 3.0      # duration_months over date-span slack
_SUM_TOLERANCE_MONTHS = 18.0      # summed tenure over yoe*12 slack (allows overlap)
_FOUNDING_HARD_YEARS = 3          # start >= 3y before founding -> killswitch
_FOUNDING_SOFT_YEARS = 2          # exactly 2y before founding -> soft anomaly


def _founding_violation_years(c: Dict[str, Any]) -> int:
    """Worst 'started N years before the company existed' across all roles."""
    worst = 0
    for r in career(c):
        fy = lex.COMPANY_FOUNDED_YEAR.get(r.get("company", ""))
        sd = parse_date(r.get("start_date"))
        if fy and sd and sd.year < fy:
            worst = max(worst, fy - sd.year)
    return worst


def _hard_flags(c: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    yoe = profile(c).get("years_of_experience", 0) or 0
    summed = 0.0
    for r in career(c):
        dm = r.get("duration_months", 0) or 0
        summed += dm
        sd, ed = parse_date(r.get("start_date")), parse_date(r.get("end_date"))
        if sd and ed:
            if sd > ed:
                flags.append("start_date after end_date")
            span = (ed - sd).days / 30.44
            if dm - span > _SPAN_TOLERANCE_MONTHS:
                flags.append("claimed tenure exceeds the role's own date span")
    if summed - (yoe * 12) > (yoe * 12) * 0.5 + _SUM_TOLERANCE_MONTHS:
        flags.append("summed career tenure far exceeds total years of experience")
    for s in c.get("skills", []) or []:
        if s.get("proficiency") in ("advanced", "expert") \
                and (s.get("duration_months", None) == 0):
            flags.append(f"'{s.get('name')}' claimed {s.get('proficiency')} with 0 months used")
    fv = _founding_violation_years(c)
    if fv >= _FOUNDING_HARD_YEARS:
        flags.append(f"role starts {fv} years before the employer was founded")
    # de-dup while preserving order
    seen, out = set(), []
    for f in flags:
        if f not in seen:
            seen.add(f); out.append(f)
    return out


def _soft(c: Dict[str, Any]):
    sig = signals(c)
    yoe = profile(c).get("years_of_experience", 0) or 0
    mult = 1.0
    soft: List[str] = []

    # generator-noise band (~0.97): pure artifacts, no quality meaning.
    sal = sig.get("expected_salary_range_inr_lpa", {}) or {}
    if sal.get("min", 0) > sal.get("max", float("inf")):
        mult *= 0.97; soft.append("salary min > max")
    su, la = parse_date(sig.get("signup_date")), parse_date(sig.get("last_active_date"))
    if su and la and su > la:
        mult *= 0.97; soft.append("signup after last-active")

    # quality-relevant band (~0.85 each, product floored at 0.6).
    #
    # Exception: candidates whose best role carries a TIER-5 description
    # (28 in the whole pool) get these flags waived. The flags are noisy
    # generator artifacts (the skill-duration one hits 9.2% of the pool and
    # disproportionately hits deep experts, whose long skill durations are what
    # trips it), while a tier-5 description is near-ground-truth work evidence
    # -- noise must not override it. Tier-A hard impossibilities and the
    # generator-noise band above still apply to everyone. A/B vs the
    # independent referee: +5 elite captures in top-100, NDCG@50 +0.013,
    # nothing regresses.
    elite = any((tier_for(r.get("description", "") or "") or 0) >= 5
                for r in career(c))
    q = 1.0
    fv = _founding_violation_years(c)
    if _FOUNDING_SOFT_YEARS <= fv < _FOUNDING_HARD_YEARS:
        q *= 0.85
        soft.append(f"role starts {fv} years before the employer was founded")
    assess = sig.get("skill_assessment_scores", {}) or {}
    if not elite:
        for s in c.get("skills", []) or []:
            dur = s.get("duration_months", 0) or 0
            if dur > (yoe * 12) + 12:
                q *= 0.85
                soft.append(f"'{s.get('name')}' used longer than the whole career")
                break
        for s in c.get("skills", []) or []:
            name = s.get("name", "")
            if s.get("proficiency") in ("advanced", "expert") and name in assess \
                    and assess[name] < 40:
                q *= 0.85
                soft.append(f"'{name}' claimed {s.get('proficiency')} but assessed {int(assess[name])}")
                break
    mult *= max(0.6, q)
    return mult, soft


def score(c: Dict[str, Any]) -> Dict[str, Any]:
    hard = _hard_flags(c)
    soft_mult, soft = _soft(c)
    if hard:
        return {"consistency_factor": 0.0, "hard_flags": hard, "soft_flags": soft}
    return {"consistency_factor": round(soft_mult, 4),
            "hard_flags": [], "soft_flags": soft}
