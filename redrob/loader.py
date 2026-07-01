"""Streaming loader + safe accessors for candidate records.

The pool is ~465 MB uncompressed JSONL. We stream it line-by-line so peak
memory stays well under the 16 GB budget, and we never hold the raw text of the
whole file at once.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
from typing import Any, Dict, Iterator


def open_maybe_gzip(path: str):
    """Open plain ``.jsonl`` or gzipped ``.jsonl.gz`` transparently."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_candidates(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one candidate dict per non-blank line."""
    with open_maybe_gzip(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_date(s: Any) -> _dt.date | None:
    """Parse an ISO ``YYYY-MM-DD`` string; return None on missing/garbage."""
    if not s or not isinstance(s, str):
        return None
    try:
        return _dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def months_between(start: _dt.date | None, end: _dt.date | None) -> float | None:
    """Approximate months between two dates (30.44-day month)."""
    if start is None or end is None:
        return None
    return (end - start).days / 30.44


# Convenience getters -------------------------------------------------------- #

def profile(c: Dict[str, Any]) -> Dict[str, Any]:
    return c.get("profile") or {}


def signals(c: Dict[str, Any]) -> Dict[str, Any]:
    return c.get("redrob_signals") or {}


def career(c: Dict[str, Any]) -> list:
    return c.get("career_history") or []


def education(c: Dict[str, Any]) -> list:
    return c.get("education") or []


def skills(c: Dict[str, Any]) -> list:
    return c.get("skills") or []


def profile_text(c: Dict[str, Any]) -> str:
    """Concatenated free-text used for semantic matching: headline + summary +
    every career-history description. This is where prose-buried fit lives."""
    p = profile(c)
    parts = [p.get("headline", ""), p.get("summary", "")]
    parts += [r.get("description", "") for r in career(c)]
    parts += [r.get("title", "") for r in career(c)]
    return "\n".join(x for x in parts if x)
