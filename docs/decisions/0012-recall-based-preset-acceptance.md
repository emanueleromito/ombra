# 0012. Recall-based acceptance rule for presets

- **Status:** accepted
- **Date:** 2026-11-09

## Context

Presets have been judged by a maximum score error of 1e-6, which was a guess. It rejected `m0_compact_n8192` (T0.11), yet T0.8 and T1.6 show CKKS noise has no measurable effect on recall at `m0_depth2`. What users care about is whether encrypted search returns the same results as plaintext search. T1.6 built a check for that, in `core/bench/recall.py` and `core/bench/recall_check.py`.

## Options considered

- **A. Keep the 1e-6 score-error bound.** Simple and quick, but not tied to what users see, and it can reject a preset that returns the same results.
- **B. Recall-based rule.** A preset passes if, on at least 2 public embedding datasets with at least 1,000 queries each, recall@10 under encryption is within 0.005 of plaintext recall at the same cluster settings, for every setting. Measures the outcome directly, at the cost of a real-data run.
- **C. Require both.** Keeps the guess as a gate, so it would still reject presets that return the same results.

## Decision

B. Score error stays a reported safety margin, not a pass/fail rule. Kernel correctness is separate and unaffected: hard rule 3's exact OpenFHE oracles remain required.

## Consequences

`m0_depth2` passes on 3 datasets with a worst gap of 0.0000 (`docs/plan/results/2026-11-09-t1.6-recall.md`). Presets are judged on recall, so a lower-precision preset such as `m0_compact_n8192` becomes eligible (T1.7). A pass means encrypted recall matches plaintext recall at the same cluster settings; it says nothing about how good clustering itself is. The rule's datasets are English text embeddings at 100,000 vectors, so other data may behave differently. Tests that assert 1e-6 on a preset would need revisiting if this is accepted.
