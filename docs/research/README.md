# Research log

One folder per experiment, kept whether it worked or not, so that anyone can rerun it and nobody
repeats a dead end. Benchmark results of plan tasks stay in `docs/plan/results/`; this folder is for
experiments that compare candidates.

## Rules

- **One folder per experiment:** `NNNN-short-name/` with
  - `README.md`: what was tried and why, the exact setup, the single command that reruns it, the
    results and their reading (copy [template.md](template.md));
  - `run.py` or `run.sh`: that single command's script;
  - `raw/`: the raw output, unedited, one JSON file per run, each starting with the environment
    record from [tools/envinfo.py](tools/envinfo.py) (commit, GPU, CPU, versions);
  - `analyze.py`: turns the raw output into the tables and figures of the README.
- **Failed experiments stay,** with the outcome "rejected" and the reason.
- **Exact or rejected.** Every candidate is compared with the OpenFHE oracle. A candidate that
  changes an RNS word, or recall, is rejected whatever its speed.
- **Same-run comparisons.** Before and after are measured in one run on one machine. Numbers from
  different machines are never compared directly (the M1 refresh showed 26% run-to-run variation).
- **Synthetic data only** unless the README names a public dataset. No keys, plaintext or scores in
  `raw/`.
- **Hard rules still apply** (`CLAUDE.md`): 128-bit floor, parameters from presets only. A
  candidate that needs a preset, level-budget, security-level or format change is recorded as
  "needs approval" and is not built into the product without the user's decision.

## Index

| # | Experiment | Area | Outcome |
| --- | --- | --- | --- |
| [0000](0000-survey.md) | Survey: baseline, sources, candidate list | all | reference |
| [0001](0001-cpu-client-profile/README.md) | Client CPU profile; drop to one tower before decrypting | client CPU | adopted: block decrypt 1.9x to 2.2x |
| 0002 | GPU baseline on the current commit | GPU | the "before" files of 0006 and 0007 (same run as the candidates) |
| [0003](0003-cpu-fused-score/README.md) | Local mode: fused lazy score | client CPU | adopted: score 4.8x to 5.8x |
| [0004](0004-cpu-expansion-monomial/README.md) | Local mode: monomial in evaluation form | client CPU | adopted: expansion 1.07x to 1.25x |
| [0005](0005-ciphertext-deserialize/README.md) | Cost of deserializing a ciphertext | server and client CPU | baseline; 3.3 ms of fixed cost per ciphertext |
| [0006](0006-gpu-lazy-search-kernel/README.md) | GPU search kernel: lazy accumulation, Karatsuba cross term | GPU kernel | rejected: 2% to 10% slower; the kernel is memory-bound |
| [0007](0007-expansion-leaf-sink/README.md) | Expansion writes its leaves in place (no handoff) | GPU host, memory | adopted: server 1.09x to 1.14x, 1.2 GB less at d = 1536 |
| [0008](0008-gpu-families/README.md) | GPU families: T4, L4, L40S, H200 against A10 and A100 | GPU families | adopted as an option: exact on all; default list proposed in decision 0024 |
| 0009 | Template reader for ciphertexts | server and client CPU | proposed, not built |
| 0010 | Hand-written CPU key switch for local mode | client CPU | not run: proposed as a task |
| [0011](0011-seeded-stored-ciphertexts/README.md) | Seeded stored ciphertexts (decision 0023) | disk, upload | adopted pending 0023: stored columns 0.50x, collection data 0.60x |

Needs the user's decision before any work (see the last rows of [0000](0000-survey.md)): the
server dropping results to one tower, a depth-1 preset, 32-bit RNS words. Seeded ciphertexts were
approved on 2026-11-13 and are experiment 0011.
