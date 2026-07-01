"""Two-tier consistency (design sec 3e).

Tier A -- hard-impossible contradictions -> killswitch (factor 0). Logically
un-fakeable; a single one zeroes the candidate. These are the honeypot markers
we *can* detect internally. (The pool has no company founding dates -- 63
company names reused ~31K times each -- so the "8 years at a company founded 3
years ago" honeypot archetype has no internal footprint and is handled by the
fit model pushing keyword-stuffers down, not by a check here. See
eval/honeypot_selftest.py, which asserts these checks actually fire.)

Tier B -- soft anomalies -> penalty multiplier, never fatal. Split into
generator-noise (very mild) and quality-relevant (heavier) bands.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..loader import career, parse_date, profile, signals

# Margins chosen from the data audit: hard flags fire on ~45/100K candidates.
_SPAN_TOLERANCE_MONTHS = 3.0      # duration_months over date-span slack
_SUM_TOLERANCE_MONTHS = 18.0      # summed tenure over yoe*12 slack (allows overlap)


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
    q = 1.0
    assess = sig.get("skill_assessment_scores", {}) or {}
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
