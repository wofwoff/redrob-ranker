"""Must-have evidence -- the heart of the model (design sec 3a).

The JD's absolute needs are: (1) embeddings-based retrieval in production,
(2) vector DB / hybrid search infra, (3) ranking-eval frameworks (NDCG/MRR/MAP).
These are exactly the keywords a stuffer lists, so we read them from the *prose*
of real roles first and only *corroborate* with the skills list, endorsement /
duration trust-weighting, and Redrob assessment scores. A plain-language
candidate who actually built retrieval at a product company thus outscores a
stuffer with the right skill names and nothing behind them.
"""

from __future__ import annotations

from typing import Any, Dict

from .. import lexicons as lex
from ..loader import career, profile, signals
from ..text import word_matcher as _matcher

# Prose evidence: phrases grouped by must-have area. Read from career
# descriptions + summary, where buzzword-free fit actually lives.
_PROSE = {
    "retrieval": [
        "embedding", "retrieval", "semantic search", "sentence transformer",
        "dense retrieval", "nearest neighbor", "ann index", "encoder",
        "retrieval-augmented", "rag pipeline", "rag system",
    ],
    "vector_db": [
        "vector database", "vector db", "vector store", "faiss", "pinecone",
        "weaviate", "qdrant", "milvus", "pgvector", "elasticsearch",
        "opensearch", "hybrid search", "bm25", "inverted index",
        "search infrastructure", "search index",
    ],
    "ranking": [
        "ranking", "learning to rank", "ndcg", "mrr", "recommendation",
        "recommender", "relevance", "re-rank", "rerank", "click-through",
        "ctr", "precision@", "recall@", "matching system",
    ],
}
# Production / product-scale signals: the JD's "shipped to real users", not research.
_PROD_SIGNALS = [
    "in production", "production", "shipped", "real users", "at scale",
    "serving", "deployed", "latency", "throughput", "a/b test", "experiment",
]

_PROF_W = {"beginner": 0.45, "intermediate": 0.75, "advanced": 1.0, "expert": 1.1}

_PROSE_RE = {b: _matcher(kws) for b, kws in _PROSE.items()}
_PROD_RE = _matcher(_PROD_SIGNALS)


def _prose_bucket_hits(text: str) -> Dict[str, bool]:
    t = text.lower()
    return {b: bool(rx.search(t)) for b, rx in _PROSE_RE.items()}


def _skill_trust(skill: Dict[str, Any], assess: Dict[str, float]) -> float:
    """Endorsement/duration/proficiency trust weight for one skill, discounted
    hard when the candidate scored poorly on its Redrob assessment."""
    prof = _PROF_W.get(skill.get("proficiency", "intermediate"), 0.75)
    end = skill.get("endorsements", 0) or 0
    dur = skill.get("duration_months", 0) or 0
    end_f = 0.5 + 0.5 * min(1.0, end / 15.0)
    dur_f = 0.5 + 0.5 * min(1.0, dur / 24.0)
    w = prof * end_f * dur_f

    name = skill.get("name", "")
    if name in assess:
        s = assess[name]
        if s < 40:
            w *= 0.30          # claimed skill the candidate scores poorly on
        elif s >= 70:
            w *= 1.15          # external validation of the claim
    return w


def _skill_bucket_scores(c: Dict[str, Any]) -> Dict[str, float]:
    assess = signals(c).get("skill_assessment_scores", {}) or {}
    buckets = {
        "retrieval": (lex.SKILL_RETRIEVAL_EMBEDDINGS, 0.0),
        "vector_db": (lex.SKILL_VECTOR_DB, 0.0),
        "ranking": (lex.SKILL_RANKING_IR, 0.0),
        "core_ml": (lex.SKILL_CORE_ML, 0.0),
    }
    acc = {k: 0.0 for k in buckets}
    for sk in c.get("skills", []) or []:
        name = sk.get("name", "")
        w = _skill_trust(sk, assess)
        for b, (names, _) in buckets.items():
            if name in names:
                acc[b] += w
    # Saturate: ~1.5 trust units in a bucket == full coverage of that bucket.
    return {b: min(1.0, v / 1.5) for b, v in acc.items()}


def score(c: Dict[str, Any]) -> Dict[str, Any]:
    p = profile(c)
    # Evidence from REAL ROLES (career descriptions) counts full; the
    # self-authored summary counts at 0.4 (buzzwords / hobby projects live there:
    # e.g. "built a small RAG side project, not in a professional capacity").
    desc_text = "\n".join(r.get("description", "") for r in career(c))
    summary_text = "\n".join([p.get("summary", ""), p.get("headline", "")])
    desc_hits = _prose_bucket_hits(desc_text)
    summ_hits = _prose_bucket_hits(summary_text)

    credit = {b: (1.0 if desc_hits[b] else (0.4 if summ_hits[b] else 0.0))
              for b in _PROSE}
    prose_cov = sum(credit.values()) / len(credit)          # 0..1 across 3 areas
    desc_areas = [b for b, v in desc_hits.items() if v]     # real-role evidence only
    # Production credit requires a real ML-role prose hit, not a generic
    # "shipped production features" line from an unrelated engineering role.
    prod = bool(_PROD_RE.search(desc_text.lower())) and bool(desc_areas)
    prod_bonus = 0.10 if prod else 0.0

    sk = _skill_bucket_scores(c)
    must_skill = 0.5 * (sk["retrieval"] + sk["vector_db"] + sk["ranking"]) / 1.5 \
        + 0.5 * min(1.0, (sk["retrieval"] + sk["vector_db"] + sk["ranking"]) / 2.0)
    must_skill = min(1.0, must_skill)
    core_floor = sk["core_ml"]

    # Prose is primary (read from real roles first); skills corroborate.
    raw = 0.55 * prose_cov + 0.35 * must_skill + 0.10 * core_floor + prod_bonus
    must = max(0.0, min(1.0, raw))

    return {
        "must_have_score": round(must, 4),
        "prose_areas": desc_areas,   # only claim "career history shows X" for real roles
        "has_production_signal": prod,
        "skill_buckets": {k: round(v, 3) for k, v in sk.items()},
    }
