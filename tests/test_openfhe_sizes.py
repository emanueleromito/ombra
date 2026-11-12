"""Object sizes at preset m0_depth2 for d = 384, 768, 1536 (sizes only, no data printed).

Measures with real OpenFHE objects (seeded synthetic data):
  * "raw" = RNS words held in memory/GPU: components x towers x 8 B x N (what Ombra stores on the
    device). It is computed from the structure and cross-checked against the serialized size.
  * "serialized" = bytes of OpenFHE's BINARY serialization written to a temp file.
Objects: column-packed stored ciphertext (encrypted-DB mode), column-packed plaintext (plaintext-DB
mode), query ciphertext, score download (level 2 after multiply-add, relinearize, rescale, mask,
rescale; and level 1 before the mask), automorphism keys and the relinearization key. From these
it derives storage per million vectors, bytes per dimension per vector, the ratio to a float32
vector DB and how many vectors fit in an A100 40 GB. Writes /data/results/sizes.json when
/data/results exists (Modal volume), else a temp file. CPU only.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pytest
from ombra import params as ombra_params
from ombra import query as oq

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.openfhe

PRESET = "m0_depth2"
SEED = 20260109
DIMS = [384, 768, 1536]
WORD = 8  # bytes per RNS word (64-bit)
A100_BYTES = 42_429_251_584  # A100 40 GB as reported by CUDA
SCRATCH_BYTES = 1 << 30  # assumed fixed scratch (kernel temporaries, CUDA context): 1 GiB
VECTORS = 1_000_000
FLOAT32_BYTES_PER_DIM = 4


def _file_size(write) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "obj.bin")
        ok = write(path)
        assert ok, "serialization failed"
        return os.path.getsize(path)


def test_sizes() -> None:
    import openfhe as ofhe

    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    ring = cc.GetRingDimension()
    slots = ring // 2
    preset = ombra_params.get_preset(PRESET)
    towers0 = preset["multiplicative_depth"] + 1
    assert (ring, slots, towers0) == (16384, 8192, 3)
    rng = np.random.default_rng(SEED)

    def raw_ct(level: int, comps: int = 2) -> int:
        return comps * (towers0 - level) * WORD * ring

    def ser(obj) -> int:
        return _file_size(lambda p: ofhe.SerializeToFile(p, obj, ofhe.BINARY))

    # Level 0 stored column (encrypted DB) and plaintext column (plaintext DB).
    col_vals = rng.standard_normal(slots).tolist()
    pt = cc.MakeCKKSPackedPlaintext(col_vals, 1, 0)
    col = cc.Encrypt(keys.publicKey, pt)
    assert col.GetLevel() == 0
    ct0_ser = ser(col)
    pt_raw = 1 * towers0 * WORD * ring
    try:
        pt_ser = ser(pt)
        pt_ser_note = "serialized plaintext measured"
    except Exception as exc:  # plaintext may not be serializable in the bindings
        pt_ser = None
        pt_ser_note = f"OpenFHE cannot serialize a Plaintext here ({type(exc).__name__})"

    # Query ciphertext and the real score path: multiply-add, relinearize, rescale, mask, rescale.
    q = rng.standard_normal(384)
    q /= np.linalg.norm(q)
    qct = oq.encode_query(cc, keys.publicKey, q)
    q_ser = ser(qct)
    term = cc.EvalMultNoRelin(col, qct)  # stand-in for column x expanded leaf: same size
    acc = cc.EvalAdd(term, cc.EvalMultNoRelin(col, qct))  # a sum of terms keeps the size
    s1 = cc.Rescale(cc.Relinearize(acc))
    assert s1.GetLevel() == 1
    mask_ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([1.0] * slots, 1, 1))
    s2 = cc.Rescale(cc.EvalMult(s1, mask_ct))
    assert s2.GetLevel() == 2
    s1_ser, s2_ser = ser(s1), ser(s2)

    # Invariants: raw = 2 x towers x 8 x N; level 2 has 1 tower; serialized >= raw and close to it.
    assert raw_ct(0) == 2 * 3 * 8 * ring == 786_432
    assert raw_ct(2) == 2 * 1 * 8 * ring
    checks = (
        (raw_ct(0), ct0_ser),
        (raw_ct(0), q_ser),
        (raw_ct(1), s1_ser),
        (raw_ct(2), s2_ser),
    )
    for raw, s in checks:
        assert s >= raw, "serialized smaller than raw"
        assert s <= raw * 1.05, "serialized unexpectedly larger than raw (tower count wrong?)"
    if pt_ser is not None:
        assert pt_ser >= pt_raw

    # Evaluation keys: relinearization key now; automorphism keys per d below.
    relin_ser = _file_size(
        lambda p: ofhe.CryptoContext.SerializeEvalMultKey(
            p, ofhe.BINARY, keys.publicKey.GetKeyTag()
        )
    )
    assert relin_ser > 0
    res_raw = raw_ct(1)  # per-block score ciphertext kept on device (level 1, conservative)

    result: dict = {
        "preset": PRESET,
        "ring_dim": ring,
        "slots": slots,
        "towers_level0": towers0,
        "a100_bytes": A100_BYTES,
        "scratch_bytes": SCRATCH_BYTES,
        "plaintext_serialized_note": pt_ser_note,
        "ciphertext_level0": {"raw": raw_ct(0), "serialized": ct0_ser},
        "plaintext_level0": {"raw": pt_raw, "serialized": pt_ser},
        "query_upload": {"raw": raw_ct(0), "serialized": q_ser},
        "score_download_level2_masked": {"raw": raw_ct(2), "serialized": s2_ser},
        "score_download_level1_unmasked": {"raw": raw_ct(1), "serialized": s1_ser},
        "relin_key_serialized": relin_ser,
        "dims": {},
    }
    print(
        f"ct_level0 raw={raw_ct(0)} serialized={ct0_ser} "
        f"B_per_dim_per_vec raw={raw_ct(0) / slots:.2f} serialized={ct0_ser / slots:.2f}"
    )
    print(
        f"pt_level0 raw={pt_raw} serialized={pt_ser} ({pt_ser_note}) "
        f"B_per_dim_per_vec raw={pt_raw / slots:.2f}"
    )
    print(f"query_upload raw={raw_ct(0)} serialized={q_ser}")
    print(
        f"score_level2_masked raw={raw_ct(2)} serialized={s2_ser} "
        f"B_per_score raw={raw_ct(2) / slots:.2f} serialized={s2_ser / slots:.2f}; "
        f"level1_unmasked raw={raw_ct(1)} serialized={s1_ser}"
    )
    print(f"relin_key serialized={relin_ser}")

    blocks_1m = math.ceil(VECTORS / slots)  # whole blocks of 8192 vectors, rounded up
    for d in DIMS:
        cc.ClearEvalAutomorphismKeys()
        indices = oq.automorphism_indices(ring, d)
        cc.EvalAutomorphismKeyGen(keys.secretKey, indices[:1])
        one_key = _file_size(lambda p: cc.SerializeEvalAutomorphismKey(p, ofhe.BINARY))
        cc.ClearEvalAutomorphismKeys()
        cc.EvalAutomorphismKeyGen(keys.secretKey, indices)
        auto_total = _file_size(lambda p: cc.SerializeEvalAutomorphismKey(p, ofhe.BINARY))
        assert auto_total > one_key > 0

        entry: dict = {
            "automorphism_keys": len(indices),
            "automorphism_key_one_serialized": one_key,
            "automorphism_total_serialized": auto_total,
            "automorphism_avg_per_key_serialized": auto_total / len(indices),
            "eval_keys_total_serialized": auto_total + relin_ser,
            "modes": {},
        }
        for mode, sizes in (
            ("encrypted_db", {"raw": raw_ct(0), "serialized": ct0_ser}),
            ("plaintext_db", {"raw": pt_raw, "serialized": pt_ser}),
        ):
            m: dict = {}
            for kind, per_col in sizes.items():
                if per_col is None:
                    continue
                block = d * per_col  # d columns per block of 8192 vectors
                per_dim_per_vec = per_col / slots
                total_1m = blocks_1m * block
                ratio = per_dim_per_vec / FLOAT32_BYTES_PER_DIM
                # A100 fit: fixed = scratch + eval keys + one expanded query (d level-0 cts);
                # each block also needs its score ciphertext (level 1, conservative).
                fixed = SCRATCH_BYTES + auto_total + relin_ser + d * raw_ct(0)
                n_blocks = max(0, (A100_BYTES - fixed) // (block + res_raw))
                m[kind] = {
                    "bytes_per_dim_per_vector": per_dim_per_vec,
                    "blocks_per_million": blocks_1m,
                    "storage_per_million_bytes": total_1m,
                    "ratio_to_float32": ratio,
                    "float32_per_million_bytes": VECTORS * d * FLOAT32_BYTES_PER_DIM,
                    "a100_fixed_reserved_bytes": fixed,
                    "a100_vectors_fit": int(n_blocks) * slots,
                }
                print(
                    f"d={d} mode={mode} {kind} B/dim/vec={per_dim_per_vec:.2f} "
                    f"storage_1M={total_1m / 1e9:.2f} GB ratio_f32={ratio:.1f}x "
                    f"a100_fit={int(n_blocks) * slots:,}"
                )
            entry["modes"][mode] = m
        print(
            f"d={d} auto_keys={len(indices)} one={one_key} total={auto_total} "
            f"avg={auto_total / len(indices):.0f} relin={relin_ser}"
        )
        # Encrypted DB raw: 2 polys x 3 towers x 8 B x N (16384) / 8192 slots = 96 B per dim.
        assert entry["modes"]["encrypted_db"]["raw"]["bytes_per_dim_per_vector"] == 96.0
        result["dims"][str(d)] = entry

    out_dir = Path("/data/results")
    base = out_dir if out_dir.is_dir() else Path(tempfile.gettempdir())
    out = base / "sizes.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
