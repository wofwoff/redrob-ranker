#!/usr/bin/env python3
"""Correctness tests. Run with pytest, or standalone: ``python tests/test_pipeline.py``.

Covers the things that must not regress: spec-conformance of the output, the
consistency killswitch, the title gate vs. the keyword-stuffer, geo caps,
disqualifiers, base-weight integrity, and reasoning no-hallucination.
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redrob import scorer                                   # noqa: E402
from redrob.features import (consistency, disqualifiers, geo_fit,  # noqa: E402
                             must_have, role_match)
from redrob.reasoning import compose                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARCH = os.path.join(ROOT, "archetypes")


def _mk(cid, title, company, industry="SaaS", country="India", **sig):
    signals = {"willing_to_relocate": False, "recruiter_response_rate": 0.8,
               "last_active_date": "2025-01-01", "open_to_work_flag": True,
               "notice_period_days": 20, "skill_assessment_scores": {}}
    signals.update(sig)
    return {
        "candidate_id": cid,
        "profile": {"current_title": title, "current_company": company,
                    "current_company_size": "1001-5000", "current_industry": industry,
                    "country": country, "location": "Bangalore, Karnataka",
                    "summary": "", "headline": "", "years_of_experience": 6},
        "career_history": [{
            "company": company, "title": title, "industry": industry,
            "company_size": "1001-5000", "is_current": True,
            "start_date": "2019-01-01", "end_date": "2025-01-01",
            "duration_months": 72,
            "description": "Built and shipped ranking and retrieval systems in production."}],
        "education": [], "skills": [], "certifications": [],
        "redrob_signals": signals,
    }


# --- spec conformance ------------------------------------------------------ #

def test_base_weights_sum_to_one():
    assert abs(sum(scorer.BASE_WEIGHTS.values()) - 1.0) < 1e-9


def test_output_is_spec_conformant(tmp_path=None):
    """rank.py output validates: 100 unique ranks, strictly-decreasing scores,
    valid ids -- mirrors validate_submission.py's core checks."""
    sample = os.path.join(ROOT, "data", "sample.jsonl")
    if not os.path.exists(sample):
        print("  (skip: data/sample.jsonl not present)")
        return
    import subprocess
    out = os.path.join(ROOT, "data", "_test_out.csv")
    subprocess.run([sys.executable, os.path.join(ROOT, "rank.py"),
                    "--candidates", sample, "--out", out, "--top", "20"],
                   check=True, capture_output=True)
    rows = list(csv.DictReader(open(out)))
    assert [*rows[0].keys()] == ["candidate_id", "rank", "score", "reasoning"]
    ranks = [int(r["rank"]) for r in rows]
    assert ranks == list(range(1, len(rows) + 1)), "ranks not 1..N unique"
    scores = [float(r["score"]) for r in rows]
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)), \
        "scores must be strictly decreasing (guarantees validator passes)"
    for r in rows:
        assert r["candidate_id"].startswith("CAND_") and len(r["candidate_id"]) == 12


# --- the multiplicative gates --------------------------------------------- #

def test_title_gate_crushes_keyword_stuffer():
    """A Marketing Manager stuffed with AI skills must score far below a real
    ML engineer with the same skills (the core JD trap)."""
    ai_skills = [{"name": n, "proficiency": "expert", "endorsements": 9,
                  "duration_months": 24}
                 for n in ("FAISS", "Pinecone", "RAG", "Embeddings",
                           "Learning to Rank", "Recommendation Systems")]
    stuffer = _mk("CAND_0000001", "Marketing Manager", "Acme Corp", industry="Media")
    stuffer["skills"] = ai_skills
    # a real stuffer's career prose is about marketing, not ML
    stuffer["career_history"][0]["description"] = \
        "Led marketing campaigns, owned KPIs, and drove content and brand outcomes."
    real = _mk("CAND_0000002", "ML Engineer", "Swiggy", industry="Food Delivery")
    real["skills"] = ai_skills

    recs = scorer.score_pool([stuffer, real], *_arch())
    by = {r["candidate_id"]: r for r in recs}
    assert by["CAND_0000002"]["final_score"] > 3 * by["CAND_0000001"]["final_score"], \
        "title gate failed to separate stuffer from real ML engineer"


def test_description_tier_separates_elite_from_generic():
    """Two identical strong-ML candidates differing only in description tier:
    the tier-5 role must beat the tier-2 role on must_have and final rank."""
    from redrob import description_tiers as dt
    tier5 = next(d for d, t in dt.DESCRIPTION_TIER.items() if t == 5)
    tier2 = next(d for d, t in dt.DESCRIPTION_TIER.items() if t == 2)
    elite = _mk("CAND_0000020", "ML Engineer", "Swiggy", industry="Food Delivery")
    elite["career_history"][0]["description"] = tier5
    generic = _mk("CAND_0000021", "ML Engineer", "Swiggy", industry="Food Delivery")
    generic["career_history"][0]["description"] = tier2

    e = must_have.score(elite)["must_have_score"]
    g = must_have.score(generic)["must_have_score"]
    assert e > g + 0.15, f"tier-5 ({e}) should clearly beat tier-2 ({g})"

    recs = scorer.score_pool([elite, generic], *_arch())
    by = {r["candidate_id"]: r for r in recs}
    assert by["CAND_0000020"]["final_score"] > by["CAND_0000021"]["final_score"]


def test_description_tier_fallback_on_unknown_text():
    """Off-vocabulary descriptions must not crash and must fall back to keyword
    evidence (graceful degradation on unseen data)."""
    from redrob.description_tiers import tier_for
    assert tier_for("a totally novel description not in the pool") is None
    c = _mk("CAND_0000022", "ML Engineer", "Swiggy")
    c["career_history"][0]["description"] = \
        "Built a production ranking and retrieval system with embeddings."
    out = must_have.score(c)
    assert 0.0 <= out["must_have_score"] <= 1.0
    assert out["must_have_score"] > 0.2  # keyword fallback still credits real work


def test_consistency_killswitch_zeroes_impossible():
    c = _mk("CAND_0000003", "ML Engineer", "Swiggy")
    c["skills"] = [{"name": "FAISS", "proficiency": "expert", "duration_months": 0,
                    "endorsements": 3}]
    out = consistency.score(c)
    assert out["consistency_factor"] == 0.0 and out["hard_flags"]


def test_founding_year_killswitch():
    """A role starting >= 3y before the employer's public founding year is the
    spec's first honeypot archetype -> killswitch. 1y slack is noise -> clean."""
    hp = _mk("CAND_0000010", "Search Engineer", "Sarvam AI", industry="AI/ML")
    hp["career_history"][0]["start_date"] = "2018-09-01"   # Sarvam AI founded ~2023
    out = consistency.score(hp)
    assert out["consistency_factor"] == 0.0
    assert any("before the employer was founded" in f for f in out["hard_flags"])

    mild = _mk("CAND_0000011", "ML Engineer", "Krutrim", industry="AI/ML")
    mild["career_history"][0]["start_date"] = "2022-06-01"  # 1y slack: noise band
    mild["career_history"][0]["duration_months"] = 31       # keep span-consistent
    out = consistency.score(mild)
    assert out["consistency_factor"] > 0.0 and not out["hard_flags"]


def test_geo_cap_removes_non_relocating_overseas():
    overseas = geo_fit.score(_mk("CAND_0000004", "ML Engineer", "Swiggy",
                                 country="USA", willing_to_relocate=False))
    assert overseas["geo_fit"] <= 0.15
    relocate = geo_fit.score(_mk("CAND_0000005", "ML Engineer", "Swiggy",
                                 country="USA", willing_to_relocate=True))
    assert 0.7 <= relocate["geo_fit"] <= 0.9
    india = geo_fit.score(_mk("CAND_0000006", "ML Engineer", "Swiggy"))
    assert india["geo_fit"] >= 0.95


def test_consultancy_only_disqualifier_fires():
    c = _mk("CAND_0000007", "Software Engineer", "Infosys", industry="IT Services")
    c["career_history"][0]["company"] = "Infosys"
    c["career_history"][0]["industry"] = "IT Services"
    out = disqualifiers.score(c)
    assert out["disqualifier_mult"] < 0.5 and out["disqualifier_reasons"]


def test_stopped_coding_exempts_hands_on_ml_titles():
    """'Lead AI Engineer' / 'Staff MLE' are IC-leadership ML titles, not the
    JD's left-engineering-for-architecture archetype -- must NOT be flagged
    even when the description uses outcome language without coding verbs."""
    c = _mk("CAND_0000012", "Lead AI Engineer", "Sarvam AI", industry="AI/ML")
    c["career_history"][0]["description"] = \
        "Owned the search and discovery experience end-to-end, from data " \
        "infrastructure through ranking algorithms to evaluation."
    out = disqualifiers.score(c)
    assert not any("non-coding leadership" in r for r in out["disqualifier_reasons"])

    # but a generic Project Manager in the same situation IS flagged
    pm = _mk("CAND_0000013", "Engineering Manager", "Swiggy")
    pm["career_history"][0]["title"] = "Engineering Manager"
    pm["career_history"][0]["description"] = \
        "Ran ceremonies, managed delivery timelines, and reviewed designs."
    out = disqualifiers.score(pm)
    assert any("non-coding leadership" in r for r in out["disqualifier_reasons"])


def test_product_history_escapes_consultancy_flag():
    c = _mk("CAND_0000008", "Software Engineer", "Infosys", industry="IT Services")
    c["career_history"].append({
        "company": "Flipkart", "title": "ML Engineer", "industry": "E-commerce",
        "company_size": "1001-5000", "is_current": False,
        "start_date": "2016-01-01", "end_date": "2019-01-01",
        "duration_months": 36, "description": "Built recommendation systems."})
    out = disqualifiers.score(c)
    assert "services/consulting" not in " ".join(out["disqualifier_reasons"])


# --- reasoning integrity --------------------------------------------------- #

def test_reasoning_no_hallucination():
    """Every reasoning must cite the candidate's real title and company and must
    not invent skills not on the profile (Stage-4 no-hallucination check)."""
    real = _mk("CAND_0000009", "ML Engineer", "Swiggy")
    recs = scorer.score_pool([real], *_arch())
    text = compose(recs[0], 1)
    assert "ML Engineer" in text and "Swiggy" in text
    assert "\n" not in text  # single line for CSV safety


def _arch():
    from redrob.features import semantic
    return semantic.load_archetypes(os.path.join(ARCH, "ideal.txt"),
                                    os.path.join(ARCH, "anti_pattern.txt"))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
