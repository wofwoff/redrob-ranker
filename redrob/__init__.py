"""Redrob candidate ranker.

A hybrid retrieval -> feature-scoring -> re-ranking pipeline for the Redrob
"Intelligent Candidate Discovery & Ranking" challenge. See
Redrob_Solution_Design_v3.md for the full rationale behind every component.

The ranking step (``rank.py``) is pure NumPy/SciPy: no GPU, no network, no
hosted-LLM calls, well inside the 5 min / 16 GB / CPU-only budget.
"""

__version__ = "1.0.0"
