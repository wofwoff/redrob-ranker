# Redrob Hackathon — Intelligent Candidate Discovery & Ranking
## Solution Design

**Task:** Rank the top 100 candidates from a 100,000-candidate pool against a Senior AI
Engineer job description.
**Scored on:** `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` against a hidden ground truth.
**Hard constraints (ranking step):** ≤5 min wall-clock, ≤16 GB RAM, CPU-only, no network /
hosted-LLM calls, ≤5 GB disk.

---

## 0. The insight that drives every decision

The scoring weights tell you where to spend effort:

- **80% of the score lives in the top 50** (0.50·NDCG@10 + 0.30·NDCG@50).
- NDCG discounts **logarithmically**, so within the top 50 the top ~10 dominate, and within
  the top 10 the top 3–5 dominate.
- P@10 is only 0.05; MAP (0.15) is the only term rewarding the full 100, and even MAP is
  dominated by early precision.

**Consequence:** ranks 51–100 are nearly cosmetic. The winning strategy is a *decent global
ranker* plus *disproportionate effort on the top ~50* — especially the top 10: read those
profiles hard, confirm internal consistency, and confirm the candidate is actually available.
Fill 51–100 with clearly-relevant, clearly-available, consistency-clean candidates and don't
agonize over their internal order.

**Calibration reference:** the provided `sample_submission.csv` is a deliberate anti-example.
Its top 10 is "HR Manager / Content Writer / Graphic Designer, each with 9 AI core skills."
That is the keyword-stuffer trap. Any ranking that resembles it is the failure mode.

---

## 1. Architecture: hybrid retrieval → ranking → re-ranking

A production-style pipeline:

1. **Embedding similarity** is kept as **one signal**, not the ranker. Pure cosine similarity
   ranks keyword stuffers and inconsistent profiles highest (they contain all the right words),
   so it must never be shipped alone. It is, however, the only good tool for plain-language
   candidates whose fit is buried in prose.
2. **Rule-based / feature-engineered scoring** is the interpretable backbone: fast, CPU-trivial,
   defensible, and it produces explicit feature values for the reasoning column.
3. **Optional learning-to-rank re-rank** sharpens the order of the top slots only.

Meta-advantage worth stating at the interview: the JD describes the job as owning "ranking,
retrieval, and matching systems," so building the solution *as* a retrieval+ranking pipeline is
itself a demonstration of the competency being hired for.

---

## 2. Data-coverage audit — use everything that carries signal

The dataset gives 8 top-level objects per candidate: `profile`, `career_history`, `education`,
`skills`, `certifications`, `languages`, `redrob_signals` (23 fields), plus `candidate_id`.
Every field is either used, deliberately ignored, or treated as a trap surface.

### 2.1 Fields that carry signal — and how each is used

| Field | Where it feeds | Why |
|---|---|---|
| `profile.current_title`, `career_history[].title` | role match; title gate | Crushes the "Marketing Manager with 9 AI skills" stuffer; JD says this explicitly |
| `profile.current_company`, `career_history[].company` | role match; consultancy disqualifier | Product-vs-services distinction; named-consultancy knockdown |
| `profile.current_company_size`, `career_history[].company_size` | role match | Catches services-shaped firms not on the named list |
| `profile.current_industry`, `career_history[].industry` | role match; services penalty | "IT Services" is a services tell even off the named list |
| `career_history[].description` | semantic match **and** must-have evidence | Prose-buried fit lives here: "built a recommendation system at a product company" with no buzzwords |
| `career_history[].{start_date,end_date,duration_months,is_current}` | applied-ML years, cadence, consistency checks, "stopped coding" | Real applied-ML years ≠ raw `years_of_experience` |
| `profile.summary`, `profile.headline` | semantic match (embeddings) | Prose-buried fit and intent signals |
| `profile.years_of_experience` | consistency cross-check; coarse band | JD wants 5–9 total, 4–5 applied; the real number comes from career history |
| `profile.location`, `profile.country` | **geo-fit** | JD: Pune/Noida hybrid, no visa sponsorship, outside-India case-by-case |
| `education[].{degree,field_of_study}` | research disqualifier; field relevance | The JD's "pure research" knockout |
| `education[].tier` | base_fit (minor) | Institution prestige |
| `skills[].{name,proficiency,endorsements,duration_months}` | must-have corroboration; consistency | Endorsement/duration trust-weighting; `expert` + 0 months is impossible |
| `redrob_signals.skill_assessment_scores` | competence validation; soft consistency | A high assessment is real external validation of a claimed skill |
| `redrob_signals.last_active_date`, `signup_date` | availability recency; soft consistency | A long-dormant profile is unavailable |
| `redrob_signals.recruiter_response_rate`, `avg_response_time_hours` | availability | 5% response / multi-day reply = not reachable |
| `redrob_signals.open_to_work_flag`, `applications_submitted_30d` | availability | JD: "clear signal of being in the job market" |
| `redrob_signals.notice_period_days` | availability / fit | JD grades this directly: ≤30 loved, 30+ raises the bar |
| `redrob_signals.preferred_work_mode`, `willing_to_relocate` | geo-fit | Relocation willingness is the key geo modifier |
| `redrob_signals.interview_completion_rate` | availability | Flake risk |
| `redrob_signals.github_activity_score` | **external-validation positive only** | A *high* score is a small positive; it is **never** a penalty (see §3b) |
| `certifications[]`, `verified_email/phone`, `linkedin_connected` | legitimacy / validation nudge | Weak positive + trust gate |

### 2.2 Fields ignored on purpose
`anonymized_name` (anonymized), `languages` (near-zero fit signal here), `connection_count` and
`endorsements_received` (aggregate, redundant with per-skill), `expected_salary_range` (no budget
anchor to compare against; only useful as a soft `min>max` anomaly), `offer_acceptance_rate` and
`profile_completeness_score` (weak; tiebreak at most).

### 2.3 Trap surfaces and the popularity cluster
Of the three popularity signals, **only `saved_by_recruiters_30d` is used** — as a *light
positive tiebreaker* among already-validated candidates, never as a major fit signal. A
recruiter *saving* a profile is a deliberate, intentional action, and it is the least circular
of the three: in the pool it separates real engineers from keyword-stuffers (~9.9 vs ~5.9 saves).

`profile_views_received_30d` and `search_appearance_30d` are **ignored.** They are passive,
keyword-driven surface impressions that mostly re-encode the skill/title signals already scored
directly, so they add little beyond noise. Even `saved_by_recruiters_30d` is held at tiebreaker
weight to avoid double-counting fit that `base_fit` already measures.

---

## 3. The scoring model

Score multiplicatively so a candidate must clear **every** gate. Multiplication kills the traps:
a stuffer with great skills but a Marketing title is zeroed by the title gate; an impossible
profile is zeroed by the consistency killswitch; a ghost is capped by availability; a
non-relocating overseas candidate is capped by geo-fit.

```
final_score = base_fit
            × disqualifier_penalty
            × geo_fit
            × availability_multiplier
            × consistency_factor
```

Keep every factor in [0,1] (`geo_fit`, `availability`, and the soft part of `consistency` are
soft caps, not zeros; the hard part of `consistency` is a true killswitch). Tune all weights
against the hand-labeled gold set (§6) — the numbers below are starting points.

### 3a. `base_fit` (0–1) — weighted blend, title does the heavy lifting

| Component | Weight (start) | What it computes | Why |
|---|---|---|---|
| Role/title relevance | ~0.38 | Is the current/recent title a real ML/AI/SWE/data-eng role at a **product** company? Uses title + company + company_size + industry. | Crushes the Marketing-Manager-with-9-AI-skills stuffer |
| Career semantic match | ~0.25 | Embedding similarity of career descriptions + summary vs a JD-derived ideal (zero-label classifier, §3f) | Catches prose-buried fits without buzzwords |
| Must-have evidence | ~0.17 | Retrieval / vector-DB / ranking-eval experience, scored from **career_history descriptions first**, corroborated by `skills` (endorsements × duration) and `skill_assessment_scores` | The JD's absolute needs are exactly the keywords a stuffer lists; read them from prose and corroborate with assessments |
| Real applied-ML years | ~0.13 | Years actually in ML/product roles, derived from `career_history` (not raw `years_of_experience`) | JD wants 4–5 applied-ML years at a product company |
| Education + validation | ~0.07 | Institution `tier`, `field_of_study` relevance; certifications and a *high* `github_activity_score` as external validation | Minor JD signals; validation is JD-explicit and used as a **positive only** |

The must-have component is the heart of the model: read the JD's absolute needs (embeddings
retrieval in production, vector DB, NDCG/MRR evaluation) from the **prose of real roles** and
corroborate with assessment scores, so a plain-language candidate who actually built retrieval
at a product company outscores a stuffer with the right skill names and nothing behind them.

### 3b. `disqualifier_penalty` — multiplicative knockdowns from the JD's "we will not move forward" list

Each is a computable feature and a citable sentence for the reasoning column:

- **Career-long consultancy-only** (TCS / Infosys / Wipro / Accenture / Cognizant / Capgemini)
  with no product-company stint — extend with `company_size`/`industry` to catch unlisted
  services firms.
- **Pure research / no production deployment** — corroborate with `education.degree` (PhD) +
  research-only titles + absent product roles.
- **Recent LangChain-only** AI experience (<12 months) without substantial pre-LLM ML.
- **Title-chaser:** job-hop cadence ~1.5 yr with ascending titles — computable from
  `career_history` durations + title sequence.
- **Stopped coding** (architecture/tech-lead only) for 18+ months — use `is_current` title +
  recent descriptions. **Hands-on ML titles (Lead AI Engineer, Staff MLE) are exempt**: the JD
  targets people who *left* engineering for architecture roles, and the audit found a
  referee-tier-5 "Lead AI Engineer @ Sarvam AI" buried by this flag purely because his
  description used outcome language ("owned the search experience end-to-end") instead of
  coding verbs.
- **CV / speech / robotics** primary expertise without NLP/IR.
- **Closed-source-only 5+ years without external validation** — fire only on explicit career
  evidence (long tenure entirely at proprietary employers), with certifications as a positive
  offset. **`github_activity_score` is not used here**: a missing/low score is not evidence of
  the disqualifier, since "no GitHub linked" is a profile fact (true of ~65% of the pool), not a
  career fact, and the JD's other validation channels (papers, talks) are unobservable in this
  data. A *high* GitHub score may act as a small positive in §3a; it never penalizes.

Apply as a product of per-flag multipliers (e.g., 0.2–0.6 each) rather than a hard 0, so a
single soft flag dampens rather than annihilates — except where the JD is explicit ("we will
not move forward"), where a near-zero is appropriate.

### 3c. `geo_fit` (0.3–1.0)

The JD is explicit: Pune/Noida hybrid, Tier-1 Indian cities welcome, "Outside India:
case-by-case, we don't sponsor work visas." Geography is a real discriminator — ~25% of the
pool is non-India.

- `country == India` (esp. Pune/Noida/Hyderabad/Mumbai/Delhi-NCR) → **1.0**.
- Non-India but `willing_to_relocate == true` → **mild cap (~0.85)**; remains in contention.
- Non-India and not willing to relocate → **very hard cap (~0.1)** — visa + onsite-cadence
  reality effectively removes them from contention. Even an overwhelming `base_fit` cannot
  recover from this multiplier; it is kept just above zero only because the JD says
  "case-by-case."
- `preferred_work_mode` as a soft modifier: remote-only against a hybrid role nudges down;
  flexible/hybrid is neutral-to-positive.

The non-relocating overseas cap is deliberately near-disqualifying; the willing-to-relocate
case stays a genuine soft cap so strong relocatable candidates remain in play.

### 3d. `availability_multiplier` (0.3–1.0)

A perfect-on-paper candidate gone months with a 5% response rate is, for hiring, unavailable.
Combine into a single 0.3–1.0 multiplier:

- `last_active_date` recency, `recruiter_response_rate`, `avg_response_time_hours`,
  `open_to_work_flag`, `interview_completion_rate`.
- `applications_submitted_30d` — active job-seeking is the JD's "clear signal of being in the
  job market."
- `notice_period_days` — JD grades it directly: "We'd love sub-30-day notice… 30+ day notice
  candidates are still in scope but the bar gets higher." So the penalty ramps from **30**, not
  90: ≤30 ideal (1.0), and a monotonic soft cap above 30 that keeps deepening (buyout covers only
  the first 30 days). Do **not** treat 30–90 as flat/free.

Cap hard, don't zero — availability is a modifier, not a disqualifier.

> **Calibration (validated by A/B against an independent referee rubric):** the raw blend is
> **sqrt-compressed** before use. The JD's mandate targets extremes ("hasn't logged in for 6
> months, 5% response rate"), not micro-differences among reachable candidates — uncompressed,
> a 0.85-vs-1.0 availability gap outweighed a full fit-tier difference at the top-10 margin.
> Compression halved the normal-band spread (P@10 against the referee rubric went 0.90 → 1.00)
> while leaving true ghosts fatally capped. A parallel A/B on softening Tier-B consistency
> (0.85 → 0.93) showed **no gain** on either pseudo-truth and was rejected — the 0.85 stands.

### 3e. `consistency_factor` — two tiers

Profiles are checked for internal consistency. There are two distinct kinds of inconsistency,
and they are treated very differently. (The honeypot population is ~80 / 0.08%; ranking any in
the top 10 signals a keyword-embedding system, and >10% in the top 100 is a Stage-3
disqualification.)

**Tier A — Hard-impossible contradictions → killswitch (score ≈ 0).** Logically un-fakeable;
each is rare. **A single one is sufficient** to zero the candidate. Run these precisely on the
**top ~200–500 after fit scoring**, where the top-100 honeypot rate is decided.
- role `start_date > end_date`
- `duration_months` exceeding the actual `start_date → end_date` span by a clear margin (catches
  "more tenure claimed than the dates allow," with no company founding date required)
- summed `career_history` durations wildly exceeding `years_of_experience × 12`
- a skill at `advanced`/`expert` proficiency with `duration_months == 0`

**Tier B — Soft anomalies → penalty multiplier (never fatal).** These appear in large fractions
of the pool and are not honeypot markers, so they dampen a score but never zero it. They split
into two weight bands:

*Generator-noise flags — very mild (~0.97 each):* pure data artifacts with no quality meaning.
- salary `min > max`
- `signup_date > last_active_date`

*Quality-relevant flags — heavier (~0.85 each, product floored ~0.6):* not logically impossible,
but real credibility/competence dents.
- single skill `duration_months > years_of_experience × 12` — not impossible (skill use can
  predate a professional career: hobby, school, open-source), but a meaningful credibility dent
  when present.
- `advanced`/`expert` skill with `skill_assessment_scores` < 40 — a competence-doubt signal. It
  *also* applies as a steeper per-skill discount inside §3a (must-have evidence), so a claimed
  expert skill the candidate scores poorly on counts for little toward fit.

> **Company-age check — via world knowledge, not the data.** The pool carries no founding-date
> field, and founding dates cannot be *inferred* from the data (only 63 distinct company names,
> each reused thousands of times). An earlier revision of this doc concluded the spec's *first*
> honeypot example — "8 years of experience at a company founded 3 years ago" — was therefore
> undetectable. **The full-pool audit disproved that:** 55 of the 63 companies are real companies
> with *public* founding years (Sarvam AI ≈ 2023, Krutrim ≈ 2023, CRED 2018, …). Encoding those
> in `lexicons.COMPANY_FOUNDED_YEAR` makes the archetype checkable: a role cannot start years
> before the employer existed, even though the profile is internally coherent.
>
> Empirics that set the thresholds: 93 candidates have roles starting ≥2 yrs pre-founding; the
> generator's noise band extends ~1–2 yrs pre-founding (23/64 Krutrim roles start before 2022),
> and the extreme violators found in our own pre-fix top-100 (ranks 5–6, at **5 yrs** each) were
> wrapped in suspiciously perfect profiles — `saved_by_recruiters` 43 and 27 vs pool mean 7.7.
> So: **≥3 yrs pre-founding → Tier-A killswitch; exactly 2 yrs → Tier-B soft anomaly; 1 yr →
> ignored (noise/stealth-mode slack).** The demotion is positive-expected-value even if a flagged
> profile were innocent noise: replacements at the top-100 boundary are near-identical in quality,
> while a honeypot at rank 5–6 costs ≈0.08 composite (NDCG@10 1.00 → 0.836).
>
> The **honeypot self-test** (`eval/honeypot_selftest.py`) asserts all Tier-A checks — internal
> *and* founding-year — actually **fire on** constructed honeypots (not merely that the top-100
> looks clean), keeps negative controls for the noise band, and confirms the schema still has no
> founding field (if one ever appears, prefer it over the lexicon).

### 3f. Zero-label semantic classifier (powers 3a's "career semantic match")

No training labels exist (ground truth is hidden). Build a label-free classifier:

1. Write 3–5 short **"ideal candidate"** archetype paragraphs (e.g., "Built and shipped a hybrid
   retrieval + ranking system serving recruiters at a Series-B product company; owns NDCG/A-B
   eval; 6 yrs, 4 in applied ML").
2. Write 3–5 **"anti-pattern"** paragraphs (keyword stuffer with wrong title; pure-research
   academic; LangChain-tutorial enthusiast; consultancy-only).
3. Embed all of them with the local model; average each group into an `ideal_vector` and an
   `anti_vector`.
4. Score each candidate's profile embedding as `sim(candidate, ideal) − sim(candidate, anti)`.

This catches prose-buried fits and pushes down trap archetypes without a single human label.

> **Precompute the archetype vectors offline, not at ranking time.** `ideal_vector` and
> `anti_vector` are fixed (they don't depend on the candidate pool), so embed them during the
> offline precompute pass and commit them alongside the candidate embeddings as small `.npy`
> files. This lets `rank.py` be **pure NumPy with no model load** — see §7. If `rank.py` instead
> instantiates `sentence-transformers` at runtime, it will try to fetch/cache model weights, which
> is a network+disk dependency that will not exist in the clean Stage-3 sandbox and will fail
> reproduction.

---

## 4. Reasoning column (Stage 4)

Stage 4 samples 10 reasonings and checks: **specific facts, JD connection, honest concerns, no
hallucination, variation, tone-matches-rank.**

**Approach: a deterministic reasoning composer**, built from the actual feature values that drove
the score. For each candidate, pull: real applied-ML years, current/recent title, the single
strongest career signal, the top concern flag (geo, notice period, response latency,
consistency), and compose 1–2 sentences from those fields.

- **No hallucination** — every sentence is sourced from a real field.
- **Variation** — driven by each candidate's distinct feature profile.
- **Honest concerns** — the concern flag is included by construction.
- **Tone matches rank** — high ranks carry high component scores; low ranks surface concerns.

Use an LLM offline, in the dev loop, to help *write the composer* — not to generate the strings
(the no-hallucination check is strict and there is no network at ranking time regardless).

---

## 5. Optional: learning-to-rank refinement (top-50 only)

No labels are provided, so supervised LTR is a *refinement*, not the backbone:

1. Hand-label ~50–100 sample candidates against the JD rubric (or label programmatically from
   the rule scorer + manual correction).
2. Train a small **XGBoost ranker** on the engineered features.
3. Apply it **only to the top ~150–200** from the heuristic scorer to sharpen the order of the
   top slots, where NDCG concentrates.

Keep the interpretable scorer as the core: it is defensible at Stage 5, produces the reasoning,
and doesn't depend on label quality. A badly-labeled LTR model loses to a good heuristic.

> **The emitted `score` column must be monotonic with the *final* rank, or the validator
> auto-rejects the file** (spec §3, §6: "score is non-increasing with rank"). Because the LTR
> re-rank *reorders* the top slots after the heuristic assigned scores, emitting the original
> `final_score` will produce rows where a better rank has a lower score → instant rejection. When
> a re-rank (LTR, diversity promotion, or honeypot removal + backfill) changes the order, emit a
> score consistent with the order actually written: use the LTR score itself, or re-map scores to
> be non-increasing down the final ranking. This applies to **any** stage that reshuffles rows
> after scoring — sort by the exact score you emit.

---

## 6. Calibrating without a leaderboard

There is no live feedback, so validate methodologically:

1. **Hand-label a gold set** — read 60–100 sample candidates carefully, assign relevance tiers
   0–5 per the JD's "between the lines" section.
2. **Compute your own NDCG@10/@50, MAP, P@10** on that gold set as you tune.
3. **Ablate** — turn each scoring component on/off and watch your offline metrics move; this
   both tunes weights and produces Stage-5 talking points.
4. **Consistency audit** — confirm 0 hard-impossible profiles in your top 100 on the sample, and
   run the honeypot self-test (§3e) so the checks are proven to *fire* on seeded honeypots.
5. **Confirm the load-bearing data stats before trusting the tuned multipliers.** Several weights
   in this doc are anchored to empirical claims that must be verified on `sample_candidates.json` /
   `candidates.jsonl`, not assumed: ~25% non-India (§3c), ~65% of pool with no GitHub linked
   (§3b), `saved_by_recruiters` ≈ 9.9 vs 5.9 for real-vs-stuffer (§2.3), the 63-distinct-company
   fact (§3e), and the `sample_submission.csv` "HR Manager / Content Writer" anti-example (§0). If
   any is off, re-tune the affected factor.
6. Spend your 3 submissions deliberately; the last valid one counts.

---

## 7. Compute & engineering plan

- **Load:** stream the 100K JSONL (~465 MB uncompressed) — fits in 16 GB easily.
- **Precompute (offline, allowed to exceed 5 min):** embeddings via a small local model
  (`BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2`); save the 100K candidate matrix **and** the
  fixed `ideal_vector`/`anti_vector` (§3f) as numpy `.npy` / memmap.
  - **Artifact size / GitHub limit:** the candidate matrix is 100K × 384 × 4 B ≈ **154 MB**, which
    **exceeds GitHub's 100 MB per-file hard limit.** Do **not** `git add` it raw. Either (a) ship
    only `precompute_embeddings.py` and regenerate the artifact (cleanest, keeps history light and
    proves the pipeline), or (b) store it via **Git LFS**. Prefer (a); the spec explicitly allows
    "a script that produces them" (spec §10.3). Consider float16 (≈77 MB) if an artifact must be
    committed.
- **Ranking step (≤5 min CPU) — pure NumPy, no model, no network:** load precomputed embeddings +
  archetype vectors + raw candidates → vectorized feature computation (numpy/pandas) → matrix
  cosine (100K × 384 is trivial) → multiplicative score → apply hard-impossible killswitch to the
  top band → sort by the emitted score → write CSV. Finishes well under a minute. `rank.py` must
  **not** import `sentence-transformers` or load model weights (that would need network/cache
  absent in the Stage-3 sandbox); all embedding work happens in the offline precompute step.
- **Determinism:** fixed random seeds; deterministic tie-break (secondary signal, else
  `candidate_id` ascending — matches the validator).
- **Validate:** run `validate_submission.py` on the output before every submission, and also
  verify every `candidate_id` exists in `candidates.jsonl` (the bundled validator checks ID
  *format* but not pool *membership*).

### Suggested repo layout
```
redrob-ranker/
├── README.md                      # setup + single reproduce command
├── requirements.txt
├── submission_metadata.yaml       # mirrors portal metadata
├── precompute_embeddings.py       # offline, may exceed 5 min
├── rank.py                        # the ≤5-min ranking step → submission.csv
├── features/
│   ├── role_match.py
│   ├── disqualifiers.py
│   ├── geo_fit.py
│   ├── availability.py
│   ├── consistency.py             # two-tier: hard killswitch + soft penalty
│   └── reasoning.py               # deterministic composer
├── archetypes/
│   ├── ideal.txt
│   └── anti_pattern.txt
├── artifacts/
│   ├── embeddings.npy             # precomputed, ~154 MB → regenerate via script or Git LFS (not raw)
│   └── archetype_vectors.npy      # precomputed ideal_vector + anti_vector (small, safe to commit)
└── eval/
    ├── gold_labels.csv            # hand-labeled calibration set
    ├── honeypot_selftest.py       # asserts Tier-A checks fire on seeded honeypots (§3e)
    └── offline_metrics.py         # NDCG/MAP/P@k on the gold set
```

**Single reproduce command** (for Stage 3 + metadata):
```
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

---

## 8. Surviving Stages 3–5

- **Stage 3 (reproduction):** code must run in the sandbox within constraints; honeypot rate
  ≤10% in top 100. Test locally on a 16 GB CPU machine first.
- **Stage 4 (manual review):** real **git history with visible iteration** (flat single-dump
  history is flagged); pass the 6 reasoning checks; codebase must be more than LLM API calls.
- **Stage 5 (defend-your-work):** be able to explain every weight, every disqualifier rule, and
  the retrieval→ranking→re-ranking design. Strong talking points: the multiplicative model is
  easy to narrate; the zero-label archetype classifier; the **full data-coverage audit**,
  including which fields were deliberately ignored as trap surfaces and **why each consistency
  check is hard vs soft, grounded in its frequency in the pool.**
- **Sandbox link (required):** HuggingFace Spaces / Streamlit Cloud / Colab / Replit / Docker /
  Binder running the ranker on a ≤100-candidate sample. Set this up early.
- **AI-tools declaration:** honest. Declared use isn't penalized; a declaration that contradicts
  your interview is.

---

## 9. Build order

1. Load sample, eyeball 10–15 profiles, inspect for hard-impossible patterns.
2. Hand-label a 60–100 gold set + write `eval/offline_metrics.py`.
3. Build the deterministic feature scorer (role match → disqualifiers → geo-fit → availability →
   consistency). Measure offline.
4. Add embeddings + zero-label archetype classifier; wire must-have evidence from descriptions +
   assessment scores. Re-measure.
5. Write the reasoning composer.
6. (Optional) XGBoost top-50 re-rank.
7. Precompute on full pool, run `rank.py` (pure NumPy, no model/network), validate: format +
   pool membership + **emitted score non-increasing with final rank** (§5) + honeypot self-test
   (§3e). Stand up sandbox.
8. Ablate components for the methodology summary and interview prep.

---

*Tune all weights against your own gold set; nothing here is final until your offline metrics
say so.*
