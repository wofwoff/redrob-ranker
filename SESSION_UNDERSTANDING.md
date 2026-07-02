# Session Understanding Checklist

Working through this until you can explain and defend the ranker unprompted —
this doubles as your Stage-5 "defend your work" interview prep.

## 1. The Problem
- [ ] Can state what the task was and how it's scored (the metric weights and what they imply)
- [ ] Can explain the central trap in the dataset and *why* a naive system falls for it
- [ ] Can explain what honeypots are and why they're a hard elimination gate
- [ ] Knows why "just use embedding cosine similarity" is the wrong backbone

## 2. The Solution
- [ ] Can explain why the score is *multiplicative* rather than a weighted sum
- [ ] Can explain what base_fit is and why `role` carries the most weight
- [ ] Can explain the zero-label archetype classifier (why no training labels exist, how it works)
- [ ] Can explain the "read from career descriptions, not the summary" decision (the RAG-side-project fix)
- [ ] Can explain the two-tier consistency check (hard killswitch vs soft penalty) and which honeypots it can't catch
- [ ] Can explain why the ranking step is pure TF-IDF/NumPy instead of a transformer
- [ ] Can explain the strictly-decreasing-score trick and what it prevents

## 3. Broader Context
- [ ] Understands the 5-stage evaluation gauntlet and which design choice defends each stage
- [ ] Knows the three changes that would cause instant elimination if you got them wrong
- [ ] Can articulate the honest weakness in our own validation (the weak-gold caveat)
- [ ] Can explain what would break if the compute constraints were removed
