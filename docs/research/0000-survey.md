# 0000. Survey: state of the art and the candidate list

- **Date:** 2026-11-13
- **Purpose:** where each experiment in the index comes from, and why some ideas were not tried.

## Where the time goes today (baseline)

From `docs/plan/results/2026-11-09-m1-refresh-a100.md` (A100, `m0_depth2`, 100k vectors, 5%
searched) and experiments 0001, 0004, 0005:

| Step | d = 384 | d = 1536 | Kind |
| --- | --- | --- | --- |
| Client encode and encrypt | 9.3 ms | 9.4 ms | client CPU |
| Server: read query | 8.4 ms | 8.3 ms | server CPU (deserialization, 0005) |
| Expansion | 23.1 ms | 72.9 ms | GPU, 77% NTT and 23% key-switch inner product (T0.5b) |
| Leaf handoff | 3.6 ms | 12.2 ms | GPU copy of every leaf |
| Stage 1 and stage 2 kernels | 2.3 ms | 5.3 ms | GPU |
| Client stage 1 and stage 2 decrypt | 20.9 ms | 20.9 ms | client CPU, about 10 ms per block (0001) |

At 1M vectors, 20% searched, client decrypt is 257 ms of 329 ms. In local mode (CPU), expansion
is 70% to 90% of a search (0004).

Memory: 96 B per vector per column on disk and on the GPU; d expanded leaves (1.2 GB at d = 1536)
exist twice during the handoff; four twiddle tables and one key per tree level stay resident.

## Sources

- **GPU CKKS libraries.** FIDESlib (ISPASS 2025, arXiv 2507.04775) is OpenFHE-compatible and
  reports beating Phantom; Cheddar, HEonGPU, Over100x and TensorFHE are the other current
  libraries. Their common techniques: batched NTT with few passes, Shoup or Montgomery butterflies,
  fused basis conversion, hoisting, and 32-bit RNS words (Cheddar). Ombra's kernels already use
  the first three (T0.5b). 32-bit words need other moduli, so a preset change.
- **Query expansion in PIR.** The tree is the oblivious expansion of SealPIR, used by OnionPIR,
  Spiral and Respire. Published improvements trade keys for automorphisms (fewer keys, more work)
  or switch the modulus after expansion; none removes key switches for a given leaf count.
- **Encrypted vector search.** HEVEC (arXiv 2506.17336) and the FIDESlib-based similarity search
  (arXiv 2604.00546) both find the encrypted scoring memory-bound and keep top-k on the client, as
  Ombra does. Plaintext-database variants avoid ciphertext x ciphertext (Ombra: decision 0018).
- **Lazy reduction.** Summing unreduced 128-bit products and reducing once is standard in RNS
  inner products (HEXL, SEAL's dot products, Lattigo's key switch). Experiments 0003 and 0006.
- **Own profile.** Experiments 0001 and 0005: the largest CPU costs are OpenFHE's multi-tower
  decode and its per-ciphertext context rebuild, neither of which is arithmetic.

## Candidates

| Candidate | Expected effect | Experiment | Why (not) |
| --- | --- | --- | --- |
| Client drops to one tower before decrypting | block decrypt 1.9x to 2.2x | 0001 | adopted |
| Fused, lazy, Karatsuba score on the CPU | local-mode score 5x | 0003 | adopted |
| Monomial in evaluation form on the CPU | local-mode expansion 1.07x to 1.25x | 0004 | adopted |
| Template reader for ciphertexts | minus 3.3 ms per query and per block | 0005, 0009 | proposed |
| Lazy Karatsuba search kernel on the GPU | none: 2% to 10% slower | 0006 | rejected: the kernel is memory-bound |
| Expansion writes leaves in place | server 1.09x to 1.14x, minus 1.2 GB at d = 1536 | 0007 | adopted |
| GPU families beyond Ampere | T4, L4, L40S, H200 all exact | 0008 | adopted as an option; decision 0024 proposed |
| Hand-written CPU key switch | local-mode expansion 2x or more | 0010 | not run: large; proposed as a task |
| Lazy inner product in the key switch | at most 5% of expansion | none | not run: the inner product is 23% of a split and mostly gathers, not arithmetic |
| Harvey lazy butterflies in the NTT | under 10% of the NTT | none | not run: on a GPU the 64-bit high multiply dominates a butterfly, not the corrections |
| Single-kernel NTT (whole polynomial in shared memory) | fewer global passes | none | not run: needs 128 KB of shared memory per block at N = 16384, more than an A10 has (99 KB) |
| Staying in the extended basis across tree levels (lazy ModDown) | fewer NTTs per split | none | rejected without a run: the leaves would no longer equal OpenFHE's word for word |
| Server drops the result to one tower | download halved, block 11.7 to 9.0 ms | 0001 | needs approval: changes what the server returns and uses the spare level |
| Depth-1 preset (two Q towers) | about one third less memory, storage and compute | none | needs approval: preset and level budget (0019 leaves the second level unused) |
| Seeded ciphertexts | stored columns 0.50x | 0011 | adopted after the user approved it (decision 0023, proposed) |
| 32-bit RNS words | NTT about 2x | none | needs approval: other moduli, so new presets |
