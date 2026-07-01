"""Self-contained TF-IDF vectorizer (pure NumPy/SciPy).

Deliberately dependency-light: no scikit-learn, no transformer, no network. This
is what makes the semantic component reproducible inside the Stage-3 sandbox with
zero external calls. A transformer backend is available as an optional booster
(see precompute_embeddings.py); when its artifact is absent we fall back to this.

Unigrams + bigrams, sublinear TF, smoothed IDF, L2-normalized rows -- enough to
catch prose-buried fit ("built a recommendation system", "vector search") that a
raw keyword list would miss.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np
from scipy import sparse

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+.#-]*")


def word_matcher(keywords):
    """Alnum-boundary matcher: 'rag' won't fire inside 'storage', 'serving'
    won't fire inside 'observing'. Handles multi-word phrases and trailing
    symbols. Returns a compiled regex; call ``.search(text.lower())``."""
    pat = "|".join(re.escape(k) for k in sorted(set(keywords), key=len, reverse=True))
    return re.compile(r"(?<![a-z0-9])(?:" + pat + r")(?![a-z0-9])")


def _tokenize(text: str) -> List[str]:
    toks = [t for t in _TOKEN.findall(text.lower()) if len(t) >= 2]
    bigrams = [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return toks + bigrams


class TfidfVectorizer:
    def __init__(self, min_df: int = 10, max_df: float = 0.4,
                 max_features: int = 30000):
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.vocab_: dict = {}
        self.idf_: np.ndarray | None = None

    def _build_vocab(self, docs: Sequence[str]) -> None:
        n = len(docs)
        df: Counter = Counter()
        for d in docs:
            df.update(set(_tokenize(d)))
        max_count = self.max_df * n
        terms = [(t, c) for t, c in df.items() if c >= self.min_df and c <= max_count]
        terms.sort(key=lambda x: (-x[1], x[0]))
        terms = terms[: self.max_features]
        self.vocab_ = {t: i for i, (t, _) in enumerate(terms)}
        idf = np.empty(len(self.vocab_), dtype=np.float32)
        for t, i in self.vocab_.items():
            idf[i] = np.log((1.0 + n) / (1.0 + df[t])) + 1.0
        self.idf_ = idf

    def _to_matrix(self, docs: Sequence[str]) -> sparse.csr_matrix:
        vocab = self.vocab_
        indptr = [0]
        indices: List[int] = []
        data: List[float] = []
        for d in docs:
            counts: Counter = Counter()
            for tok in _tokenize(d):
                j = vocab.get(tok)
                if j is not None:
                    counts[j] += 1
            for j, c in counts.items():
                indices.append(j)
                data.append(1.0 + np.log(c))          # sublinear TF
            indptr.append(len(indices))
        m = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), indices, indptr),
            shape=(len(docs), len(vocab)),
        )
        m = m.multiply(self.idf_)                     # apply IDF
        m = sparse.csr_matrix(m)
        _l2_normalize_rows(m)
        return m

    def fit_transform(self, docs: Sequence[str]) -> sparse.csr_matrix:
        self._build_vocab(docs)
        return self._to_matrix(docs)

    def transform(self, docs: Sequence[str]) -> sparse.csr_matrix:
        return self._to_matrix(docs)


def _l2_normalize_rows(m: sparse.csr_matrix) -> None:
    norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    inv = 1.0 / norms
    m.data *= np.repeat(inv, np.diff(m.indptr))


def build_tfidf(corpus: Sequence[str]) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
    vec = TfidfVectorizer()
    matrix = vec.fit_transform(corpus)
    return matrix, vec
