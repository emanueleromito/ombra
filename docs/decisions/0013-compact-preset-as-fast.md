# 0013. Propose `m0_compact_n8192` as the "fast" product preset

- **Status:** proposed
- **Date:** 2026-11-09

## Context

`m0_compact_n8192` (ring 8192, depth 2, 40-bit scale, log2(QP) = 195, 128-bit classic) expands queries faster and has smaller keys than `m0_depth2`, but failed the old 1e-6 score-error bound (T0.11). T1.7 ran the recall check from T1.6 on it: it passes on all 3 datasets (d = 384, 768, 1536), 1,000 queries each, with a largest recall gap of 0.0004 against a tolerance of 0.005 (`docs/plan/results/2026-11-09-t1.7-compact-recall.md`). This decision depends on decision 0012, which is still proposed: without the recall-based rule the preset stays rejected.

## Options considered

- **A. Adopt it as "fast".** Expansion is 2.1x to 2.5x faster in T0.11's controlled comparison, and expansion is the largest step at d = 1536 on 100,000 vectors at 5% searched (63%, T1.3). Evaluation keys are about 40% smaller (T0.11: 18.9 MB against 31.5 MB at d = 1536). Costs: score error up to 1.74e-5 against 2.8e-8, and each ciphertext holds half as many vectors, so a block scan yields twice as many result ciphertexts to decrypt.
- **B. Keep `m0_depth2` only.** Simplest, and the highest score precision, at the price of slower expansion and larger keys.
- **C. Wait for T1.5's composite preset.** It targets search throughput rather than expansion, so the two do not exclude each other; T1.8 compares them.

## Decision

A, if decision 0012 is accepted: offer `m0_compact_n8192` as "fast". `m0_depth2` stays the baseline and is not changed.

## Consequences

Storage per vector does not improve (both presets have 3 towers at the same bytes per slot), so "fast" means faster server expansion and smaller keys, not cheaper storage. In the recall driver, client encryption took 1.21x to 1.23x as long and decryption 1.39x to 1.49x, because that driver scans every block; a real query decrypts only the probed clusters, and T1.3 has not been repeated for this preset, so its end-to-end latency is unknown. Decryption was 46% of the query in T1.3's largest case, so end-to-end speedup needs measuring before the label is final. Users who rely on score values rather than ranking get about 3 orders of magnitude less precision. The tests asserting 1e-6 on this preset need revisiting. Datasets are English text embeddings at 100,000 vectors.
