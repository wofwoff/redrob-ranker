#!/usr/bin/env python3
"""Programmatic weak-gold labeler for calibration (design sec 6).

There is no leaderboard, so we validate methodologically against a gold set of
graded relevance (0-5) per the JD's "read between the lines" rubric. Hand-labeling
is ideal; this produces a transparent *weak* gold to bootstrap ablations, meant to
be spot-corrected by hand (edit eval/gold_labels.csv).

Design note: this rubric is deliberately INDEPENDENT of the ranker's TF-IDF
semantic and must-have-prose machinery -- it uses only raw fields (title class,
product company, applied years, geo, availability, hard-consistency). That
independence is what lets an ablation of the semantic/must-have components
actually move the offline metric instead of scoring against itself.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redrob import lexicons as lex                        # noqa: E402
from redrob.features import role_match                    # noqa: E402
from redrob.features.consistency import _hard_flags       # noqa: E402
from redrob.loader import career, profile, signals, stream_candidates  # noqa: E402


def _has_product(c) -> bool:
    p = profile(c)
    comps = [(p.get("current_company", ""), p.get("current_industry", ""))]
    comps += [(r.get("company", ""), r.get("industry", "")) for r in career(c)]
    for comp, ind in comps:
        if comp in lex.PRODUCT_COMPANIES:
            return True
        if comp in lex.GENERIC_COMPANIES and ind not in lex.SERVICES_INDUSTRIES:
            return True
    return False


def _applied_years(c) -> float:
    total = 0.0
    for r in career(c):
        cat = role_match.title_category(r.get("title", ""))
        w = {"strong_ml": 1.0, "research": 0.6, "cv_speech": 0.5, "adjacent": 0.4}.get(cat, 0.05)
        total += (r.get("duration_months", 0) or 0) * w
    return total / 12.0


def label(c) -> int:
    if _hard_flags(c):
        return 0
    cat = role_match.title_category(profile(c).get("current_title", ""))
    tier = {"strong_ml": 4, "research": 2, "cv_speech": 2, "adjacent": 2,
            "weak_tech": 1, "non_tech": 0, "unknown": 1}[cat]
    if _has_product(c):
        tier += 1
    ay = _applied_years(c)
    if 4 <= ay <= 9:
        tier += 1
    elif ay < 1:
        tier -= 1
    tier = max(0, min(5, tier))

    sig, p = signals(c), profile(c)
    if not lex.is_india(p.get("country", "")):
        tier = min(tier, 4 if sig.get("willing_to_relocate") else 1)
    if (sig.get("recruiter_response_rate", 1.0) or 0) < 0.1:
        tier = max(0, tier - 1)
    return tier


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "gold_labels.csv"))
    ap.add_argument("--stride", type=int, default=200,
                    help="label every Nth candidate (deterministic sample)")
    ap.add_argument("--also", default=None,
                    help="submission CSV whose candidates should also be labeled")
    args = ap.parse_args()

    also = set()
    if args.also and os.path.exists(args.also):
        with open(args.also, newline="") as f:
            also = {r["candidate_id"] for r in csv.DictReader(f)}

    rows = []
    for i, c in enumerate(stream_candidates(args.candidates)):
        if i % args.stride == 0 or c["candidate_id"] in also:
            rows.append((c["candidate_id"], label(c)))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "relevance"])
        w.writerows(rows)
    print(f"[make_gold] wrote {len(rows)} weak-gold labels to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
