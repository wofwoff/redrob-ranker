# Redrob Ranker — Intelligent Candidate Discovery & Ranking

Ranks the top 100 candidates from the 100K-candidate pool against the Senior AI
Engineer job description, with a 1–2 sentence reasoning per candidate.

This README is the operational guide.

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
  ranking eval), scored **primarily from the description-template tier**
  (`redrob/description_tiers.py`): the pool's career descriptions are 44 hand-tiered
  templates (0 = non-technical … 5 = elite retrieval/ranking at scale), so a role's
  description is a near-ground-truth JD-relevance signal. Corroborated by the skills
  list (endorsement/duration trust-weighting) and assessment scores; keyword matching
  is the fallback for off-vocabulary text.
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

## Google Colab Sandbox

A pre-configured Jupyter notebook sandbox is available at [redrob_ranker_sandbox.ipynb](redrob_ranker_sandbox.ipynb). 

Once pushed to your GitHub repository, you can open and execute this sandbox directly on Google Colab using the following link:
`https://colab.research.google.com/github/wofwoff/redrob-ranker/blob/main/redrob_ranker_sandbox.ipynb`

This sandbox environment:
1. Clones the repository.
2. Installs the light dependencies (`numpy` and `scipy`).
3. Runs the ranking system end-to-end on the 50-candidate sample dataset (`data/sample.jsonl`).
4. Renders the top-ranked candidates with their reasoning.
5. Runs the validation test suite.

## Constraints compliance

| Constraint | Limit | This ranker |
|---|---|---|
| Runtime | ≤ 5 min | ~90 s on full 100K |
| Memory | ≤ 16 GB | ~1.4 GB peak |
| Compute | CPU only | pure NumPy/SciPy |
| Network | off | no calls; no model load at rank time |
| Disk | ≤ 5 GB | none (TF-IDF path) |
