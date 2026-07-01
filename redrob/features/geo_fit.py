"""Geo-fit cap (design sec 3c).

JD: Pune/Noida hybrid, Tier-1 Indian cities welcome, "Outside India:
case-by-case, we don't sponsor work visas." ~25% of the pool is non-India, so
geography is a real discriminator. Soft cap (never a hard zero) so a strong
relocatable candidate stays in play, but a non-relocating overseas candidate is
effectively removed.
"""

from __future__ import annotations

from typing import Any, Dict

from .. import lexicons as lex
from ..loader import profile, signals


def score(c: Dict[str, Any]) -> Dict[str, Any]:
    p = profile(c)
    sig = signals(c)
    country = p.get("country", "")
    location = (p.get("location", "") or "").lower()
    relocate = bool(sig.get("willing_to_relocate"))
    mode = sig.get("preferred_work_mode", "")

    if lex.is_india(country):
        preferred = any(city in location for city in lex.PREFERRED_INDIA_CITIES)
        geo = 1.0 if preferred else 0.95   # any Indian city is in scope
        note = "India" + (" (preferred metro)" if preferred else "")
    elif relocate:
        geo = 0.85                          # overseas but willing to relocate
        note = "overseas, willing to relocate"
    else:
        geo = 0.10                          # overseas + no relocation ~ out of contention
        note = "overseas, not willing to relocate (no visa sponsorship)"

    # Work-mode soft modifier: remote-only against a hybrid role nudges down.
    if mode == "remote" and geo > 0.15:
        geo *= 0.95

    return {"geo_fit": round(geo, 4), "geo_note": note,
            "country": country, "willing_to_relocate": relocate}
