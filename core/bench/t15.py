"""T1.5 benchmark: composite scaling candidate against m0_depth2 (called from core/bench/run.py,
`--task t1.5`).

Same harness for both presets, d = 384, 768 and 1536 (the `--dims` default):

  * search throughput: the T0.4 column-packed search benchmark (`ombra_bench_column_search`), via
    run.py's `run_t04` with `preset=...`. The candidate runs the 32-bit-limb kernel
    (column_search32.cu) because every one of its Q primes is below 2^31; m0_depth2 runs the 64-bit
    kernel. A sweep of the 32-bit kernel's __launch_bounds__ minimum blocks per SM (4, 6, 8) runs at
    the first d only, through the environment variable OMBRA_SEARCH32_MIN_BLOCKS, and its result is
    reported; the headline row uses the default (4).
  * expansion time: the T0.5 GPU expansion benchmark (`ombra_bench_expand`), via `run_t05`.
    Expansion, key switching, relinearize, rescale and mask stay on 64-bit words for the candidate.
  * storage per vector: bytes of one column-packed ciphertext (components x towers x N x word size)
    as held on the device, and as serialized by OpenFHE (BINARY), divided by the slots and times d.
  * score error: not here; the exact GPU-vs-OpenFHE pipeline test in
    tests/test_gpu_relin_rescale_mask.py reports it (`make test-gpu`), because the result is
    bit-identical to OpenFHE's and does not depend on the GPU model.

Output: `<out>/t1.5-a100.json`, with the per-preset records of the sub-benchmarks inlined. Only the
context and the public key go to a temporary directory; the secret key never leaves the process.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

PRESETS = ("m0_depth2", "m1_composite_n16384")
OUT_NAME = "t1.5-a100.json"
SWEEP = (4, 6, 8)


def storage_record(preset: str, dims: tuple[int, ...], word_bytes: int) -> dict[str, Any]:
    """Bytes per column ciphertext and per vector from the real OpenFHE objects (no data kept)."""
    import openfhe as ofhe
    from ombra import params as ombra_params

    _, cc = ombra_params.make_context(preset)
    keys = cc.KeyGen()
    ring = cc.GetRingDimension()
    slots = ombra_params.get_preset(preset)["batch_size"]
    cd = ombra_params.get_preset(preset).get("composite_degree", 1)
    towers = (ombra_params.get_preset(preset)["multiplicative_depth"] + 1) * cd
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([0.5] * slots))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ct.bin")
        if not ofhe.SerializeToFile(path, ct, ofhe.BINARY):
            raise SystemExit("cannot serialize ciphertext")
        serialized = os.path.getsize(path)
    raw_words = 2 * towers * ring
    rec: dict[str, Any] = {
        "preset": preset,
        "ring_dim": ring,
        "slots": slots,
        "towers": towers,
        "raw_words_per_ciphertext": raw_words,
        "device_word_bytes": word_bytes,
        "device_bytes_per_ciphertext": raw_words * word_bytes,
        "device_bytes_per_ciphertext_if_64bit_words": raw_words * 8,
        "serialized_bytes_per_ciphertext": serialized,
        "per_d": {},
    }
    for d in dims:
        rec["per_d"][str(d)] = {
            "device_bytes_per_vector": d * raw_words * word_bytes / slots,
            "device_bytes_per_vector_if_64bit_words": d * raw_words * 8 / slots,
            "serialized_bytes_per_vector": d * serialized / slots,
            "float32_bytes_per_vector": 4 * d,
        }
    return rec


def run_t15(
    out_dir: Path,
    dims: tuple[int, ...],
    warmup: int,
    iters: int,
    run_t04: Callable[..., Path],
    run_t05: Callable[..., Path],
    openfhe_version: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "task": "t1.5",
        "format_version": 1,
        "openfhe_version": openfhe_version,
        "dims": list(dims),
        "presets": {},
        "note": (
            "same harness for both presets; the candidate's search kernel uses 32-bit stored and "
            "query words, everything else (expansion, key switching, relinearize, rescale, mask) "
            "uses 64-bit words holding the same residues"
        ),
    }
    for preset in PRESETS:
        entry: dict[str, Any] = {}
        search = run_t04(
            out_dir, dims, None, warmup, iters, f"t1.5-search-{preset}.json", preset=preset
        )
        entry["search"] = json.loads(search.read_text())
        expand = run_t05(out_dir, dims, warmup, iters, f"t1.5-expand-{preset}.json", preset=preset)
        entry["expansion"] = json.loads(expand.read_text())
        search32 = entry["search"]["results"][0].get("stored_word_bytes", 8)
        entry["storage"] = storage_record(preset, dims, int(search32))
        record["presets"][preset] = entry

    # Launch-bounds sweep of the 32-bit kernel at the first d only.
    sweep: dict[str, Any] = {}
    previous = os.environ.get("OMBRA_SEARCH32_MIN_BLOCKS")
    try:
        for mb in SWEEP:
            os.environ["OMBRA_SEARCH32_MIN_BLOCKS"] = str(mb)
            path = run_t04(
                out_dir,
                dims[:1],
                None,
                warmup,
                iters,
                f"t1.5-search-sweep{mb}.json",
                preset="m1_composite_n16384",
            )
            sweep[str(mb)] = json.loads(path.read_text())["results"][0]
    finally:
        if previous is None:
            os.environ.pop("OMBRA_SEARCH32_MIN_BLOCKS", None)
        else:
            os.environ["OMBRA_SEARCH32_MIN_BLOCKS"] = previous
    record["search32_launch_bounds_sweep"] = sweep

    out_path = out_dir / OUT_NAME
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out_path}")
    return out_path
