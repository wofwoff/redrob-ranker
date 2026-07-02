# Solution Architecture — in plain words

This is the plain-language story of how the ranker works and how it got here. For the
dense technical rationale see [Redrob_Solution_Design_v3.md](Redrob_Solution_Design_v3.md);
for how to run it see [README.md](README.md). **This file is a living document — every
architectural change adds a dated entry to the [Iteration log](#iteration-log) below.**

---

## 1. The problem

We're given 100,000 candidate profiles and one job description (a "Senior AI Engineer" for a
startup that builds recruiter search/ranking tools). We must output the **top 100** candidates,
best first, each with a short reason. We're scored against a hidden answer key using metrics that
care overwhelmingly about the **top 10** (half the score) and top 50 (another 30%).

Three things make it hard:
- **Traps.** The dataset deliberately plants keyword-stuffers (e.g. an "HR Manager" who lists 9
  AI skills) and ~80 "honeypots" (subtly impossible profiles). Ranking these highly is penalised.
- **No answer key while building.** We can't peek at a leaderboard, so we validate by reasoning,
  independent cross-checks, and manual reading — never by fitting to a score.
- **Hard limits.** The ranking step must finish in **5 minutes**, on **CPU only**, using
  **≤16 GB RAM**, with **no internet**. So no calling GPT-4 per candidate; it's a fast, local,
  self-contained program.

## 2. The big idea — a series of checkpoints

We give each candidate a score by **multiplying** several factors together:

```
final score = base fit  ×  disqualifier  ×  geography  ×  availability  ×  consistency
```

Multiplying (instead of adding) means a candidate must clear **every** checkpoint. Any single
factor near zero vetoes them — exactly what we want:
- A keyword-stuffer with a "Marketing Manager" title has near-zero **base fit** → gone.
- An impossible/honeypot profile gets a zero **consistency** → gone.
- Someone overseas who won't relocate gets a tiny **geography** cap → gone.
- A ghost who never replies gets a low **availability** cap → pushed down.

**Base fit** is itself a weighted blend of five ingredients (weights in parentheses):
- **Role/title (0.38)** — is their actual job title a real ML/AI role at a *product* company?
- **Semantic match (0.25)** — does the free-text of their career read like our "ideal candidate"?
- **Must-have evidence (0.17)** — did they actually build retrieval / vector-search / ranking
  systems, read from their real job descriptions?
- **Experience (0.13)** — real years of *applied ML*, computed from their career history.
- **Education (0.07)** — field, institution tier, certifications (minor).

## 3. Each piece, simply

- **Role/title.** We classified all 48 job titles in the pool into buckets (real ML, adjacent
  tech, non-technical, …) and all 63 companies into product vs. consulting. A great title at a
  product company scores high; "Business Analyst" scores ~0. This is the main stuffer-killer.
- **Semantic match.** We wrote a handful of "ideal candidate" paragraphs and "anti-pattern"
  paragraphs, turned everything into simple word-frequency vectors (no downloaded model, so it
  runs offline), and score each candidate by how much they lean ideal vs. anti. This catches
  people whose fit is buried in plain prose.
- **Must-have evidence.** The JD's non-negotiables are embeddings-retrieval, vector databases,
  and ranking evaluation. We read these from the candidate's **real job descriptions** (see the
  data insight in §4 — this is now a near-ground-truth signal), corroborated by their skills list
  (weighted by endorsements, duration, and assessment scores so a bare keyword counts for little).
- **Experience.** We don't trust the self-reported "years of experience" number. We add up the
  months in each role, weighted by how ML-relevant that role actually was, and reward the JD's
  sweet spot (~4–5 applied-ML years).
- **Disqualifiers.** The JD lists people it "won't move forward" with: consulting-only careers,
  pure research, LangChain-tutorial-only, title-chasers, people who stopped coding, CV/speech
  specialists. Each is a computed penalty.
- **Geography.** India → full marks; overseas-but-willing-to-relocate → mild cap; overseas and
  unwilling → near-zero (the company doesn't sponsor visas).
- **Availability.** Recruiter response rate, last-active date, notice period, open-to-work. A
  perfect-on-paper person who's been offline for months is, for hiring, unavailable — capped, but
  never zeroed (they're still a real person).
- **Consistency.** Two tiers. Impossible profiles (dates that don't add up, "expert in a skill
  used 0 months", a job starting before the company existed) get a **zero** — this is the
  honeypot killswitch. Milder oddities get a small penalty.

Finally, we sort by score, take the top 100, and write a one-line **reason** for each, assembled
only from real facts about that candidate (so there's nothing made up).

## 4. What we reverse-engineered about the data

The dataset is a **closed world** — a fixed, finite vocabulary — which lets us classify things
exactly instead of guessing:
- **63 companies**, **48 job titles**, **24 industries**, **133 skills** — all enumerated and
  hand-classified.
- **44 career-description templates.** Every one of the 300,171 job descriptions in the pool is
  one of just 44 fixed strings. The author hand-tiered them: common ones are generic
  ("recommendation features, lighter than FAANG"), rare ones are elite and read straight from the
  JD ("RAG ranking pipeline, 50M queries/month, NDCG/MRR/recall@K"). So the description a role
  carries tells us, almost exactly, how good that role was — see the [latest iteration](#iteration-log).
- **Founding-year honeypots.** Some "impossible" profiles only reveal themselves against
  real-world knowledge (a role starting years before that company was founded). We encoded public
  founding years to catch them.

This closed-world approach is the backbone of the whole solution and the main thing to defend at
interview: we didn't guess with fuzzy matching, we *read the data's own structure*.

## 5. Current results snapshot

_Updated 2026-07-02 (description-tier change):_
- Runtime **~85 s** on the full 100K pool · peak RAM **~1.4 GB** · CPU-only, no network.
- Official validator: **valid**. Honeypot rate in top 100: **0**. Top 100: **100% India**.
- Top 10: uniformly genuine ML/AI engineers at product companies, ~6–8 years applied ML, all on
  tier-≥4 career descriptions (production-scale ranking/retrieval).

---

## Iteration log

Newest first. Each entry: **what changed · why · effect.**

### 2026-07-02 — Description-template tier as the primary must-have signal
- **What.** Discovered the pool's 300K job descriptions are only 44 fixed templates, hand-tiered
  by the author. Added `redrob/description_tiers.py` (exact-match tier 0–5 for all 44) and made it
  the *primary* signal in `must_have.py` and `experience.py`, with keyword matching kept as a
  fallback for any unknown text.
- **Why.** The old keyword matcher treated "recommendation features, lighter than FAANG" (tier 2)
  and "RAG ranking, 50M queries/mo, NDCG/MRR/recall@K" (tier 5) as the same "ranking hit". That
  blindness is what let generic candidates outrank genuinely elite ones in the last audit.
- **Effect.** In the top 100, 11 generic tier-2 candidates were replaced by production-scale
  tier-4 ones (tier-4 count 40→51), and the top 10 became uniformly tier-4/5. Measured against the
  independent referee rubric: composite **0.63→0.75**, **NDCG@10 0.69→0.88** (the top-10, where
  half the score lives), P@10 held at 1.00, no regression. Verified safe: tier ≥3 descriptions
  occur only under real ML titles, so no stuffer can exploit it.

### 2026-07-02 — Referee-driven calibration
- **What.** Built an independent "referee" scorer (strict JD rubric, zero shared code) to audit
  the ranker non-circularly. It surfaced two bugs: (1) the "stopped coding" penalty was burying a
  genuinely elite *Lead AI Engineer* just because his description used outcome words instead of
  coding verbs — now hands-on ML leadership titles are exempt; (2) small availability differences
  among reachable people were outweighing real fit differences — availability is now
  square-root-compressed so only true ghosts get hit hard.
- **Why.** Grading our own output with our own logic is circular; a separately-built referee gives
  informative disagreements.
- **Effect.** Referee P@10 went 0.90 → 1.00 with no cost elsewhere. A third tested tweak (softening
  the mild-consistency penalty) showed no gain and was rejected.

### 2026-07-02 — Founding-year honeypot check (world knowledge)
- **What.** Added public company founding years; a role starting ≥3 years before its employer
  existed is now an impossible-profile killswitch (2 years = soft penalty, 1 year = ignored noise).
- **Why.** The audit found the spec's headline honeypot archetype ("N years at a company founded
  M<N years ago") sitting at ranks 5–6 of our own output — internally coherent, so only catchable
  against real-world founding dates.
- **Effect.** Those honeypots dropped out; estimated recovery ~0.08 composite. Thresholds were set
  from the data's own noise band so we don't flag innocent profiles.

### 2026-07-01 — Design-review hardening
- **What.** Made emitted scores strictly decreasing (so the validator's ordering rules can't
  fail); guaranteed the ranking step loads no model and makes no network call; kept the big
  embedding artifact out of git; made the notice-period penalty ramp from 30 days per the JD.
- **Why.** These were latent submission-rejection and Stage-3-reproduction risks found while
  reviewing the design against the spec.
- **Effect.** Bulletproofed spec-conformance and clean-sandbox reproduction.

### 2026-07-01 — Initial build
- **What.** Built the whole multiplicative pipeline: data-grounded lexicons, five base-fit
  components, four gates, the zero-label TF-IDF archetype classifier, deterministic reasoning
  composer, and the eval suite (metrics, weak-gold, ablation, honeypot self-test).
- **Why.** Implement the approved solution design end to end.
- **Effect.** First spec-valid top-100: ~90 s, 0 honeypots, no keyword-stuffers in the top 100.
