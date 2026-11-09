# 0013. Offer `m0_compact_n8192` as the "fast" product preset

- **Status:** accepted
- **Date:** 2026-11-09

## Context

`m0_compact_n8192` (ring 8192, depth 2, 40-bit scale, log2(QP) = 195, 128-bit classic) expands queries faster and has smaller keys than `m0_depth2`, but failed the old 1e-6 score-error bound (T0.11). T1.7 ran the recall check from T1.6 on it: it passes on all 3 datasets (d = 384, 768, 1536), 1,000 queries each, with a largest recall gap of 0.0004 against a tolerance of 0.005 (`docs/plan/results/2026-11-09-t1.7-compact-recall.md`). This decision depends on decision 0012, which is now accepted. T2.2 then measured the end-to-end latency (below).

## Options considered

- **A. Adopt it as "fast".** Expansion is 2.1x to 2.5x faster in T0.11's controlled comparison, and expansion is the largest step at d = 1536 on 100,000 vectors at 5% searched (63%, T1.3). Evaluation keys are about 40% smaller (T0.11: 18.9 MB against 31.5 MB at d = 1536). Costs: score error up to 1.74e-5 against 2.8e-8, and each ciphertext holds half as many vectors, so a block scan yields twice as many result ciphertexts to decrypt.
- **B. Keep `m0_depth2` only.** Simplest, and the highest score precision, at the price of slower expansion and larger keys.
- **C. Wait for T1.5's composite preset.** It targets search throughput rather than expansion, so the two do not exclude each other; T1.8 compares them.

## Decision

A: offer `m0_compact_n8192` as "fast". Decision 0012 is accepted, and T2.2 (`docs/plan/results/2026-11-09-t2.2-compact-latency.md`, A100, same run as `m0_depth2`) measured end-to-end latency at 0.52x to 0.65x of `m0_depth2` in all 12 cases (d = 384, 768, 1536; 100k and 1M vectors; 5% and 20% searched), e.g. 69.9 ms against 130.7 ms at d = 1536, 100k, 5%, and 268.5 ms against 413.8 ms at d = 1536, 1M, 20%. `m0_depth2` stays the baseline and is not changed.

## Consequences

Storage per vector does not improve (both presets have 3 towers at the same bytes per slot), so "fast" means faster server expansion and smaller keys, not cheaper storage. In the recall driver, client encryption took 1.21x to 1.23x as long and decryption 1.39x to 1.49x, because that driver scans every block; a real query decrypts only the probed clusters, and the T2.2 measurement settled it: client decrypt and top-k is 0.66x to 0.68x of `m0_depth2`'s even with about twice as many result ciphertexts, expansion 0.47x to 0.57x, and the query upload is half the size (394,207 against 787,423 bytes). Users who rely on score values rather than ranking get about 3 orders of magnitude less precision. Score error in T2.2 was 3.7e-06 to 9.0e-06. The tests asserting 1e-6 on this preset need revisiting. The T2.2 host CPU model is hidden by Modal (17 cores visible), and absolute times were 1.1x to 1.7x those of the T1.3 file, so the ratios, not the milliseconds, are what the decision rests on. Datasets are English text embeddings at 100,000 vectors.
