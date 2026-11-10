# Decision records

One short file per design decision, so the reasoning survives. New decisions copy [template.md](template.md) and take the next number.

| # | Decision | Status |
| --- | --- | --- |
| [0001](0001-source-available-license.md) | Source-available license: PolyForm Noncommercial + commercial license | accepted |
| [0002](0002-hybrid-openfhe-cuda-ckks.md) | CKKS with OpenFHE for security-critical code, own CUDA kernels for the hot path | accepted |
| [0003](0003-column-packing-query-expansion.md) | Column packing with server-side query expansion | accepted; amended by 0010 |
| [0004](0004-two-stage-cluster-search.md) | Two-stage cluster search with Fast / Decoy / Strict security levels | accepted |
| [0005](0005-homomorphic-filtering.md) | Metadata filters evaluated under FHE as encrypted masks | accepted; amended by 0019 |
| [0006](0006-cpp-cuda-server-python-client.md) | C++/CUDA server and core, Python client, one repository | accepted |
| [0007](0007-sharded-server-cluster-unit.md) | Router plus sharded nodes, clusters as the unit of distribution | accepted; premise amended by 0011 |
| [0008](0008-ampere-first.md) | Target NVIDIA Ampere only for the first version | accepted |
| [0009](0009-m0-ckks-parameters.md) | CKKS parameter preset for the M0 spike: ring dimension 16384, depth 2 | accepted |
| [0010](0010-query-expansion-via-automorphisms.md) | Query expansion via automorphisms, using no multiplication level | accepted |
| [0011](0011-batching-is-not-the-main-lever.md) | Batching is not the main throughput lever | accepted |
| [0012](0012-recall-based-preset-acceptance.md) | Recall-based acceptance rule for presets | accepted |
| [0013](0013-compact-preset-as-fast.md) | Offer `m0_compact_n8192` as the "fast" product preset | accepted |
| [0014](0014-composite-scaling-preset.md) | Composite scaling preset with 32-bit limbs: not adopted, keep `m0_depth2`, code removed | rejected |
| [0015](0015-gpu-encoder-tolerance.md) | GPU encoder: tolerance only for the FFT and rounding, integer steps exact | accepted |
| [0016](0016-ciphertext-file-envelope.md) | Ciphertext and key file envelope (magic, version, kind, preset, OpenFHE version); drivers keep raw files | accepted |
| [0017](0017-decryption-noise-position.md) | Position on decryption noise: keep the default, pin it, do not claim IND-CPA-D | accepted |
| [0018](0018-plaintext-db-disclosure.md) | Plaintext-DB mode: only for corpora any client may read | accepted |
| [0019](0019-filter-evaluation-within-depth-2.md) | Filter evaluation within depth 2: additive penalty folded into the query | accepted |

Smaller decisions (key management, threat model, accuracy, deletes, migrations, key rotation, parameter presets, format versioning, storage, API, client packaging) are recorded directly in [product-requirements.md](../product-requirements.md) and [architecture.md](../architecture.md).
