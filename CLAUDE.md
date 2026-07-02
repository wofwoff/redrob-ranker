# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a submission project for the **Redrob Hackathon — Intelligent Candidate Discovery & Ranking
Challenge**. The task: rank the top 100 candidates from a 100,000-candidate pool against a Senior AI
Engineer job description, output as a CSV with per-candidate reasoning.

**Current state: implemented and validated.** The full ranker is built (`rank.py`, `redrob/`
package, `eval/`, `tests/`) and produces a spec-valid top-100 submission. On the full 100K pool it
runs in ~90 s / ~1.4 GB RAM, CPU-only, no network; honeypot rate in top 100 is 0. See
[README.md](README.md) for the operational guide.

**Data lives outside the repo** (it is git-ignored). The bundle is at
`/Users/ahmedwafi/Downloads/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/`
(`candidates.jsonl` 100K rows, `sample_candidates.json`, `candidate_schema.json`,
`validate_submission.py`, `sample_submission.csv`). Reference it via `--candidates <path>`; never
copy the 465 MB pool into the repo.

## Commands

```bash
# Produce the submission (the ≤5-min ranking step)
python rank.py --candidates <bundle>/candidates.jsonl --out data/submission.csv

# Validate with the challenge's own validator
python "<bundle>/validate_submission.py" data/submission.csv

# Honeypot self-test (asserts Tier-A checks fire; top-100 gate)
python eval/honeypot_selftest.py --candidates <bundle>/candidates.jsonl --submission data/submission.csv

# Weak-gold + offline composite (0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10)
python eval/make_gold.py --candidates <bundle>/candidates.jsonl --also data/submission.csv
python eval/offline_metrics.py --submission data/submission.csv --gold eval/gold_labels.csv

# Ablation study
python eval/ablation.py --candidates <bundle>/candidates.jsonl --gold eval/gold_labels.csv

# Tests (no pytest required; also pytest-collectable)
python tests/test_pipeline.py
```

For a quick smoke test, `data/sample.jsonl` is the 50-candidate sample converted to JSONL.

## Key documents (read in this order before writing any code)

1. `job_description.docx` — the JD candidates are ranked against. Read the "between the lines"
   hackathon-participant section closely — it defines the disqualifiers and trap patterns.
2. `submission_spec.docx` — submission format, compute constraints, and the 5-stage evaluation
   pipeline. Non-conforming submissions are auto-rejected without scoring.
3. `redrob_signals_doc.docx` — reference for the 23 behavioral signals in `redrob_signals`.
4. `README.docx` — participant bundle overview and setup steps (unpack `candidates.jsonl.gz`,
   run `validate_submission.py`, submit via portal).
5. **[Redrob_Solution_Design_v3.md](Redrob_Solution_Design_v3.md)** — the actual engineering plan
   for this repo. Everything below is derived from it; treat it as the source of truth once code
   exists, and update it if the approach changes.

## Hard constraints (ranking step only — precompute may exceed these)

- ≤ 5 min wall-clock, ≤ 16 GB RAM, CPU-only, **no network / hosted-LLM calls**, ≤ 5 GB disk.
- Score: `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` against a hidden ground truth.
- Precomputing embeddings offline (before the timed ranking step) is allowed to exceed the 5-min
  budget; only `rank.py` itself is timed.

## Planned architecture (per solution design §1, §7)

A hybrid retrieval → ranking → re-ranking pipeline, not a single model:

```
final_score = base_fit
            × disqualifier_penalty
            × geo_fit
            × availability_multiplier
            × consistency_factor
```

- **`base_fit`**: weighted blend of role/title relevance, career semantic match (embeddings vs.
  zero-label archetype classifier), must-have evidence (retrieval/vector-DB/ranking-eval
  experience read from career prose, corroborated by skills + assessment scores), real
  applied-ML years, education/validation.
- **`disqualifier_penalty`**: multiplicative knockdowns for the JD's explicit "won't move forward"
  list (consultancy-only, pure research, LangChain-tutorial-only, title-chasing, stopped coding,
  closed-source-only without validation).
- **`geo_fit`**: India-based → 1.0; non-India + willing to relocate → mild cap (~0.85); non-India
  + not willing → near-disqualifying cap (~0.1), per the JD's no-visa-sponsorship stance.
- **`availability_multiplier`**: recruiter response rate, recency, notice period, open-to-work
  signals — capped, never zeroed (a real person, just currently hard to reach).
- **`consistency_factor`**: two tiers — Tier A hard-impossible contradictions (e.g.
  `start_date > end_date`, `expert` skill with `duration_months == 0`) are a killswitch (score ≈
  0); Tier B soft anomalies are mild penalty multipliers, never fatal. Run Tier A checks on the
  top ~200–500 after fit scoring, not the whole pool.

Embedding similarity is **one signal, never the sole ranker** — pure cosine similarity ranks
keyword-stuffers highest. Popularity/passive-impression fields
(`profile_views_received_30d`, `search_appearance_30d`) are deliberately ignored as trap surfaces;
`saved_by_recruiters_30d` is used only as a light tiebreaker.

## Repo layout

Actual layout differs slightly from the design doc: shared logic lives in a proper `redrob/`
package (with `redrob/features/`) rather than a top-level `features/`, so imports are clean when
`rank.py` runs from the repo root. Full tree and per-module notes are in [README.md](README.md).
Key entry points: `rank.py` (ranking step), `redrob/scorer.py` (multiplicative combination),
`redrob/lexicons.py` (data-grounded taxonomy), `eval/` (metrics, gold, ablation, honeypot test).

Single reproduce command: `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`

## Working conventions for this project

- **Reasoning column is deterministic, not LLM-generated at ranking time** — there's no network
  access during the ranking step anyway. An LLM may be used offline, in the dev loop, to help
  write the composer logic, not to generate the output strings themselves.
- **Git history matters for Stage 4 review**: commit incrementally with visible iteration: a flat
  single-dump history is explicitly flagged by the evaluators. Commit as features/stages land, not
  as one final dump.
- **Keep [SOLUTION_ARCHITECTURE.md](SOLUTION_ARCHITECTURE.md) current**: it is the plain-language
  living doc. Every architectural change appends a dated entry to its Iteration log (what changed,
  why, effect) and refreshes the results snapshot. It complements the dense design doc and README,
  not replaces them.
- **Validate before every submission**: run `validate_submission.py` on the output CSV, and
  separately confirm every `candidate_id` exists in `candidates.jsonl` (the bundled validator
  checks ID *format* but not pool *membership*).
- **A sandbox link is required** (HuggingFace Spaces / Streamlit Cloud / Colab / Replit / Docker /
  Binder) running the ranker on a ≤100-candidate sample — set this up early, not at the end.
- Weights throughout the solution design are starting points, tuned against a hand-labeled gold
  set (§6 of the design doc) — don't treat any numeric weight in the design doc as final.
