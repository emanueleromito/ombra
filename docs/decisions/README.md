# Decision records

One short file per design decision, so the reasoning survives. New decisions copy [template.md](template.md) and take the next number.

| # | Decision | Status |
| --- | --- | --- |
| [0001](0001-source-available-license.md) | Source-available license: PolyForm Noncommercial + commercial license | accepted |
| [0002](0002-hybrid-openfhe-cuda-ckks.md) | CKKS with OpenFHE for security-critical code, own CUDA kernels for the hot path | accepted |
| [0003](0003-column-packing-query-expansion.md) | Column packing with server-side query expansion | accepted |
| [0004](0004-two-stage-cluster-search.md) | Two-stage cluster search with Fast / Decoy / Strict security levels | accepted |
| [0005](0005-homomorphic-filtering.md) | Metadata filters evaluated under FHE as encrypted masks | accepted |
| [0006](0006-cpp-cuda-server-python-client.md) | C++/CUDA server and core, Python client, one repository | accepted |
| [0007](0007-sharded-server-cluster-unit.md) | Router plus sharded nodes, clusters as the unit of distribution | accepted |
| [0008](0008-ampere-first.md) | Target NVIDIA Ampere only for the first version | accepted |
| [0009](0009-m0-ckks-parameters.md) | CKKS parameter preset for the M0 spike: ring dimension 16384, depth 2 | accepted |

Smaller decisions (key management, threat model, accuracy, deletes, migrations, key rotation, parameter presets, format versioning, storage, API, client packaging) are recorded directly in [product-requirements.md](../product-requirements.md) and [architecture.md](../architecture.md).
