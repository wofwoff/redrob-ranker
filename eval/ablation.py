#!/usr/bin/env python3
"""Ablation study (design sec 6): turn each scoring component off and watch the
offline composite move. Produces the Stage-5 talking points ("what does the geo
cap actually buy us?") and guards against a component that silently does nothing.

Runs on the weak-gold-labeled subset (fast), scoring it with each component
disabled and reporting the composite delta vs. the full model.

    python eval/ablation.py --candidates <pool> --gold eval/gold_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redrob import scorer                                   # noqa: E402
from redrob.features import semantic                        # noqa: E402
from redrob.loader import stream_candidates                 # noqa: E402
from offline_metrics import composite                       # noqa: E402

COMPONENTS = ["role", "semantic", "must_have", "experience", "education",
              "disqualifier", "geo", "availability", "consistency"]


def _composite_for(recs, gold):
    ordered = sorted(recs, key=lambda r: (-r["final_score"], r["candidate_id"]))
    ranked_rels = [gold.get(r["candidate_id"], 0.0) for r in ordered]
    all_rels = list(gold.values())
    n_rel = sum(1 for r in all_rels if r >= 3)
    return composite(ranked_rels, all_rels, n_rel)["composite"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--gold", default=os.path.join(os.path.dirname(__file__), "gold_labels.csv"))
    args = ap.parse_args()

    gold = {}
    with open(args.gold, newline="") as f:
        for row in csv.DictReader(f):
            gold[row["candidate_id"]] = float(row["relevance"])

    subset = [c for c in stream_candidates(args.candidates)
              if c["candidate_id"] in gold]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arch = semantic.load_archetypes(
        os.path.join(root, "archetypes", "ideal.txt"),
        os.path.join(root, "archetypes", "anti_pattern.txt"))

    base = _composite_for(scorer.score_pool(subset, *arch), gold)
    print(f"{'config':16s} {'composite':>10s} {'delta':>9s}")
    print(f"{'FULL MODEL':16s} {base:10.4f} {'':>9s}")
    for comp in COMPONENTS:
        val = _composite_for(scorer.score_pool(subset, *arch, disable={comp}), gold)
        print(f"{'-' + comp:16s} {val:10.4f} {val - base:>+9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
