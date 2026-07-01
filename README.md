# Redrob Ranker — Intelligent Candidate Discovery & Ranking

Ranks the top 100 candidates from the 100K-candidate pool against the Senior AI
Engineer job description, with a 1–2 sentence reasoning per candidate.

Full rationale for every design decision is in
[Redrob_Solution_Design_v3.md](Redrob_Solution_Design_v3.md). This README is the
operational guide.

## Reproduce the submission

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv        # from the challenge bundle
```

On the full 100K pool this runs in **~90 seconds** on a CPU laptop using
**~1.4 GB RAM** — well inside the 5-minute / 16 GB / CPU-only / no-network budget.
The ranking step is pure NumPy/SciPy: no GPU, no network, no hosted-LLM calls.

## How it works

A hybrid retrieval → feature-scoring → re-ranking pipeline. The score is
**multiplicative**, so a candidate must clear *every* gate:

```
final_score = base_fit                     # weighted blend, title does the heavy lifting
            × disqualifier_mult            # JD "we will not move forward" knockdowns
            × geo_fit                       # Pune/Noida hybrid, no visa sponsorship
            × availability                  # responsiveness, recency, notice period
            × consistency_factor            # 0 for hard-impossible (honeypot killswitch)
```

`base_fit = 0.38·role + 0.25·semantic + 0.17·must_have + 0.13·experience + 0.07·education`

- **role** — title × product-vs-services company classification. Crushes the
  "Marketing Manager with 9 AI skills" keyword-stuffer trap.
- **semantic** — a **zero-label** TF-IDF archetype classifier:
  `sim(candidate, ideal) − sim(candidate, anti)` over `archetypes/*.txt`. Catches
  prose-buried fit ("built a recommendation system") without training labels.
- **must_have** — the JD's non-negotiables (embeddings retrieval, vector DB,
  ranking eval) read from **real career-role descriptions first**, corroborated by
  the skills list (endorsement/duration trust-weighting) and Redrob assessment scores.
- **experience** — real applied-ML years derived from career history, not the raw
  `years_of_experience` field.
- **consistency** — two tiers: hard-impossible contradictions are a killswitch
  (score 0); soft anomalies are mild penalties. Killswitch checks are both
  *internal* (dates, durations, proficiency-vs-usage) and *world-knowledge*: a
  role starting ≥3 years before the employer's public founding year
  (`lexicons.COMPANY_FOUNDED_YEAR`) — the spec's "8 years at a company founded
  3 years ago" honeypot archetype, which is internally coherent and only
  catchable against real-world founding dates.

All lexicons in `redrob/lexicons.py` are **data-grounded**: the pool is a closed
world (63 companies, 48 titles, 24 industries, 133 skills), enumerated and
classified explicitly rather than fuzzy-matched.

## Layout

```
rank.py                    # the ≤5-min ranking step → submission.csv
precompute_embeddings.py   # OPTIONAL offline booster (transformer embeddings)
redrob/
  lexicons.py              # data-grounded company/title/industry/skill taxonomy
  loader.py                # streaming JSONL loader + safe accessors
  text.py                  # self-contained TF-IDF (no sklearn/network)
  scorer.py                # multiplicative combination + ranking + ablation hooks
  reasoning.py             # deterministic reasoning composer
  features/                # role_match, must_have, experience, education,
                           # disqualifiers, geo_fit, availability, consistency, semantic
archetypes/                # ideal.txt / anti_pattern.txt (zero-label classifier)
eval/
  offline_metrics.py       # NDCG@10/@50, MAP, P@10 = the exact composite
  make_gold.py             # programmatic weak-gold labeler (hand-correct me)
  ablation.py              # component on/off study
  honeypot_selftest.py     # asserts Tier-A checks fire; top-100 honeypot gate
tests/test_pipeline.py     # spec-conformance, title gate, killswitch, geo, reasoning
```

## Validate the approach (no leaderboard)

```bash
# 1. Prove the honeypot killswitch fires and the top-100 is clean (Stage-3 gate)
python eval/honeypot_selftest.py --candidates ./candidates.jsonl --submission submission.csv

# 2. Build a weak-gold set and score the exact composite against it
python eval/make_gold.py   --candidates ./candidates.jsonl --also submission.csv
python eval/offline_metrics.py --submission submission.csv --gold eval/gold_labels.csv

# 3. Ablate each component
python eval/ablation.py --candidates ./candidates.jsonl --gold eval/gold_labels.csv

# 4. Correctness tests (runs under pytest or standalone)
python tests/test_pipeline.py
```

## Optional: transformer-embedding booster

The default semantic component is TF-IDF (zero external deps, fully reproducible).
To use `BAAI/bge-small-en-v1.5` embeddings instead — an **offline** step, allowed
to exceed the 5-min window and to download the model once:

```bash
pip install sentence-transformers torch
python precompute_embeddings.py --candidates ./candidates.jsonl --out artifacts/embeddings.npz
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --embeddings artifacts/embeddings.npz
```

`rank.py` still never touches the network; it only *reads* the precomputed
artifact. The artifact (~154 MB) is git-ignored — regenerate it or use Git LFS.

## Constraints compliance

| Constraint | Limit | This ranker |
|---|---|---|
| Runtime | ≤ 5 min | ~90 s on full 100K |
| Memory | ≤ 16 GB | ~1.4 GB peak |
| Compute | CPU only | pure NumPy/SciPy |
| Network | off | no calls; no model load at rank time |
| Disk | ≤ 5 GB | none (TF-IDF path) |
