# Performance research, GPU families and seeded storage (2026-11-13)

Not a plan task: the experiments behind PR "performance, memory, GPU families, seeded storage". Every number below has its own folder in `docs/research/` with the setup, the single command, the raw output and the script that makes the table. This file is the summary the results rule asks for.

- **Commits:** kernels and in-place expansion measured at `c854ce3` and `3c36b14`; families and seeded storage at `3c36b14`.
- **GPUs (Modal):** A10 (tests), A100-SXM4-40GB (benchmarks), T4, L4, L40S, H200. **CUDA:** 12.6.3 image, one build for sm_75, 80, 86, 89, 90. **OpenFHE:** 1.6.1.
- **CPU measurements:** AMD Ryzen 5 5600H (WSL2), threads as stated in each experiment.
- **Parameter sets:** `m0_depth2` and `m0_compact_n8192`, unchanged.

| Change | Result | Evidence |
| --- | --- | --- |
| Client drops a result to one tower before decrypting | One block: 21.9 to 11.7 ms (`m0_depth2`), 11.2 to 5.1 ms (`m0_compact_n8192`); same error | research 0001 |
| Local mode: fused score, monomial in evaluation form | Search at 10k vectors, d = 384: 2.08 to 1.42 s (implementer's run); score alone 4.8x to 5.8x, expansion 1.07x to 1.25x | research 0003, 0004 |
| Expansion writes its leaves in place | Handoff 3.1 / 6.1 / 12.6 ms to 0 (d = 384 / 768 / 1536); server 1.07x to 1.19x; 1.2 GB less device memory at d = 1536 | research 0007 |
| Lazy Karatsuba search kernel | Rejected: 0.90x to 0.98x; the kernel reads 820 to 970 GB/s and is memory-bound | research 0006 |
| Seeded stored ciphertexts (decision 0023, proposed) | Stored column 0.4995x; collection data 938 to 559 MB; same top-k | research 0011 |
| Turing, Ada, Hopper | 146 GPU tests pass on T4, L4, L40S, H200; expansion d = 768: 177 / 114 / 38 / 19 ms against 74 (A10) and 36 (A100) | research 0008, decision 0024 (proposed) |

Modal spend for all of it: about USD 3 of the 10 allowed.
