#!/usr/bin/env python3
"""Honeypot self-test (design sec 3e; mitigates the Stage-3 DQ gate).

Honeypot rate > 10% in the top 100 is a hard Stage-3 disqualification. This test
does NOT just check that our top-100 looks clean -- it proves our Tier-A
hard-impossible checks actually *fire* on impossible profiles, and it documents
the one honeypot archetype we cannot detect internally and why that is safe.

Run standalone:
    python eval/honeypot_selftest.py --candidates <pool> [--submission data/submission.csv]
Or via pytest (the assert_* functions are picked up as tests).
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redrob.features.consistency import _hard_flags  # noqa: E402


def _profile(**career_and_skills):
    base = {
        "candidate_id": "CAND_9999999",
        "profile": {"years_of_experience": 6},
        "career_history": [],
        "skills": [],
        "redrob_signals": {},
    }
    base.update(career_and_skills)
    return base


def assert_checks_fire_on_constructed_honeypots():
    """Each Tier-A check must catch its own impossible construction."""
    # 1) start_date after end_date
    c = _profile(career_history=[{
        "start_date": "2022-01-01", "end_date": "2020-01-01",
        "duration_months": 12}])
    assert any("start_date after end_date" in f for f in _hard_flags(c)), "start>end not caught"

    # 2) duration exceeds the role's own date span
    c = _profile(career_history=[{
        "start_date": "2021-01-01", "end_date": "2021-06-01",
        "duration_months": 60}])
    assert any("date span" in f for f in _hard_flags(c)), "duration>span not caught"

    # 3) summed tenure wildly exceeds total years of experience
    c = _profile(profile={"years_of_experience": 4}, career_history=[
        {"start_date": "2010-01-01", "end_date": "2018-01-01", "duration_months": 96},
        {"start_date": "2018-01-01", "end_date": "2024-01-01", "duration_months": 72}])
    assert any("summed career tenure" in f for f in _hard_flags(c)), "sum>yoe not caught"

    # 4) expert/advanced skill with 0 months used
    c = _profile(skills=[{"name": "FAISS", "proficiency": "expert", "duration_months": 0}])
    assert any("0 months used" in f for f in _hard_flags(c)), "expert+0mo not caught"

    # 5) role starting years before the employer was founded (the spec's FIRST
    #    honeypot example: "8 years at a company founded 3 years ago").
    #    Sarvam AI was founded ~2023; a 2018 start is a 5-year impossibility.
    c = _profile(career_history=[{
        "company": "Sarvam AI", "start_date": "2018-09-01",
        "end_date": "2024-01-01", "duration_months": 64}])
    assert any("before the employer was founded" in f for f in _hard_flags(c)), \
        "founding-year violation not caught"

    # negative controls: clean profiles fire nothing
    clean = _profile(
        career_history=[{"start_date": "2019-01-01", "end_date": "2024-01-01",
                         "duration_months": 60}],
        skills=[{"name": "PyTorch", "proficiency": "advanced", "duration_months": 40}])
    assert _hard_flags(clean) == [], "clean profile falsely flagged"
    # 1y pre-founding is generator noise / stealth-mode slack -- must NOT hard-flag
    mild = _profile(career_history=[{
        "company": "Krutrim", "start_date": "2022-06-01",
        "end_date": "2024-01-01", "duration_months": 19}])
    assert _hard_flags(mild) == [], "1y founding slack falsely hard-flagged"
    print("[selftest] Tier-A checks fire on all 5 constructed honeypots; "
          "clean + mild-slack profiles pass.")


def assert_no_company_founding_field(sample_path: str):
    """The pool carries no founding-date field, which is exactly why the
    founding-year check uses WORLD KNOWLEDGE (lexicons.COMPANY_FOUNDED_YEAR for
    the real companies in the pool) rather than a data field. If the data ever
    grows such a field, prefer it over the lexicon."""
    with open(sample_path) as f:
        c = json.loads(next(l for l in f if l.strip()))
    forbidden = {"founded", "founding_date", "company_founded", "company_age"}
    present = set()
    for role in c.get("career_history", []):
        present |= (set(role.keys()) & forbidden)
    present |= (set(c.get("profile", {}).keys()) & forbidden)
    assert not present, f"founding field(s) now in schema: {present} -- use them!"
    print("[selftest] confirmed: no company-founding field in the schema "
          "(check is grounded in public founding years via lexicons).")


def report_pool_and_submission(candidates_path: str, submission_path: str | None):
    """Report the pool-wide hard-flag rate and (if given) the top-100 rate."""
    top_ids = set()
    if submission_path and os.path.exists(submission_path):
        import csv
        with open(submission_path, newline="") as f:
            top_ids = {r["candidate_id"] for r in csv.DictReader(f)}

    pool_hits = top_hits = 0
    with open(candidates_path) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if _hard_flags(c):
                pool_hits += 1
                if c["candidate_id"] in top_ids:
                    top_hits += 1
    print(f"[selftest] pool hard-impossible candidates: {pool_hits}")
    if top_ids:
        print(f"[selftest] hard-impossible in submitted top-100: {top_hits} "
              f"(Stage-3 gate: must be < 10)")
        assert top_hits < 10, "honeypot rate >= 10% in top 100 -- Stage-3 DQ risk!"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", default=None)
    args = ap.parse_args()

    assert_checks_fire_on_constructed_honeypots()
    assert_no_company_founding_field(args.candidates)
    report_pool_and_submission(args.candidates, args.submission)
    print("[selftest] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
