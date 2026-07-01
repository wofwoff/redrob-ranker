"""Availability multiplier (design sec 3d).

A perfect-on-paper candidate gone for months with a 5% response rate is, for
hiring, unavailable. We fold recency, responsiveness, open-to-work, interview
reliability, active job-seeking, and notice period into a single 0.3-1.0 soft
cap. It is a modifier, not a disqualifier -- we cap hard but never zero.

Recency is measured against a reference date computed from the data itself
(max last_active_date in the pool), so the result is deterministic and does not
depend on wall-clock time.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict

from ..loader import parse_date, signals

_W = {"recency": 0.28, "response": 0.24, "resp_time": 0.12,
      "open": 0.10, "interview": 0.12, "notice": 0.14}


def _recency_factor(days: float | None) -> float:
    if days is None:
        return 0.7
    if days <= 30:
        return 1.0
    if days <= 90:
        return 1.0 - 0.2 * (days - 30) / 60      # 1.0 -> 0.8
    if days <= 180:
        return 0.8 - 0.3 * (days - 90) / 90       # 0.8 -> 0.5
    if days <= 365:
        return 0.5 - 0.1 * (days - 180) / 185     # 0.5 -> 0.4
    return 0.3


def _resp_time_factor(hrs: float) -> float:
    if hrs <= 24:
        return 1.0
    if hrs <= 72:
        return 0.9
    if hrs <= 168:
        return 0.75
    return 0.6


def _notice_factor(npd: int) -> float:
    # JD: sub-30 loved (buyout covers 30); the bar rises from 30 onward.
    if npd <= 30:
        return 1.0
    return max(0.70, 1.0 - (npd - 30) / 500.0)


def score(c: Dict[str, Any], ref_date: _dt.date | None = None) -> Dict[str, Any]:
    sig = signals(c)

    last_active = parse_date(sig.get("last_active_date"))
    days = (ref_date - last_active).days if (ref_date and last_active) else None

    f_recency = _recency_factor(days)
    rr = sig.get("recruiter_response_rate", 0.0) or 0.0
    f_response = 0.4 + 0.6 * max(0.0, min(1.0, rr))
    f_resp_time = _resp_time_factor(sig.get("avg_response_time_hours", 0.0) or 0.0)
    f_open = 1.0 if sig.get("open_to_work_flag") else 0.85
    icr = sig.get("interview_completion_rate", 0.7) or 0.0
    f_interview = 0.7 + 0.3 * max(0.0, min(1.0, icr))
    f_notice = _notice_factor(sig.get("notice_period_days", 0) or 0)

    raw = (_W["recency"] * f_recency + _W["response"] * f_response
           + _W["resp_time"] * f_resp_time + _W["open"] * f_open
           + _W["interview"] * f_interview + _W["notice"] * f_notice)

    # Active job-seeking is the JD's "clear signal of being in the job market".
    if (sig.get("applications_submitted_30d", 0) or 0) > 0:
        raw = min(1.0, raw + 0.03)

    avail = max(0.30, min(1.0, raw))
    return {
        "availability": round(avail, 4),
        "days_since_active": days,
        "recruiter_response_rate": rr,
        "notice_period_days": sig.get("notice_period_days", 0),
    }
