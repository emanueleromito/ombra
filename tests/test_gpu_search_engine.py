"""T2.7a: the search engine (expansion, stage 1, stage 2 with filter columns, tombstones) is exact.

Python builds a small collection with synthetic data for both presets: d = 16, two filter fields
(cardinalities 3 and 4, so D = 23 columns per data block), three clusters (one with two blocks) and
a block of cluster centers, then deletes some ids. Only public material is written to disk (context,
EvalMult and automorphism keys, ciphertexts); the secret key never leaves this process.

The driver `ombra_search_engine_check` inserts everything into a real Store, opens a SearchEngine
with a GPU cache budget below the collection size (so blocks are paged and evicted), and for a
filtered and an unfiltered query runs stage 1 and stage 2 through the real cache. It compares EVERY
output ciphertext word for word with the OpenFHE CPU chain on the same inputs (CPU expansion,
EvalMult leaves x columns summed, relinearize, rescale, EvalAdd of the tombstone penalty plaintext),
and checks the refusals: wrong request key tag, query made under another key, key files that do not
belong to the collection's key tag, search before the keys are uploaded, wrong filter_leaves.

This test then decrypts the engine's output with the secret key (held only here, never passed to the
engine) and checks slot = score + 4 * c against numpy (c = number of matching filter fields,
decision 0019), that the band decode recovers c, and that deleted vectors fall below every live
slot. The tolerances below are sanity bounds; exactness is the driver's job. Data is synthetic; only
aggregates are printed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params
from ombra import query as oq
from ombra.file_format import Kind, read_envelope

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.gpu

PRESETS = ["m0_depth2", "m0_compact_n8192"]
SEED = 20261111
D_VEC = 16
CARDS = [3, 4]
LAMBDA = 4.0
N_DATA_CLUSTERS = 3
# (cluster, block index, number of vectors); cluster N_DATA_CLUSTERS holds the centers.
BLOCKS = [(0, 0, 7), (0, 1, 4), (1, 0, 6), (2, 0, 5), (N_DATA_CLUSTERS, 0, 3)]
DELETED = ["0.0.1", "0.0.4", "1.0.2", "2.0.0"]
WANT = [1, 2]  # filtered query: field 0 == 1 and field 1 == 2
VERSION = "test-openfhe"
# Sanity bounds on |decrypted - expected| (slots reach 1 + 4F = 9); the exact check is the driver's.
TOLERANCE = {"m0_depth2": 1e-3, "m0_compact_n8192": 1e-2}

REPO_ROOT = Path(__file__).resolve().parents[1]


def _driver() -> Path:
    candidates = [
        os.environ.get("OMBRA_SEARCH_ENGINE_CHECK_BIN", ""),
        "/root/ombra-build/server/search/ombra_search_engine_check",
        str(REPO_ROOT / "build" / "server" / "search" / "ombra_search_engine_check"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail("ombra_search_engine_check not found; build with `make build` or `make test-gpu`")


def _unit(rng, n: int, rows: int):
    v = rng.standard_normal((rows, n))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.mark.parametrize("preset", PRESETS)
def test_search_engine_exact_and_correct(tmp_path: Path, preset: str) -> None:
    import openfhe as ofhe

    n_leaves = D_VEC + sum(CARDS)
    _, cc = ombra_params.make_context(preset)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    slots = ombra_params.get_preset(preset)["batch_size"]
    ring = cc.GetRingDimension()
    cc.ClearEvalAutomorphismKeys()  # the key store is static: hold exactly this D's indices
    cc.EvalAutomorphismKeyGen(keys.secretKey, oq.automorphism_indices(ring, n_leaves))
    other = cc.KeyGen()  # a second key pair: its ciphertexts must be refused
    key_tag = keys.publicKey.GetKeyTag()
    assert other.publicKey.GetKeyTag() != key_tag

    def write_ct(name: str, values, pk=keys.publicKey) -> None:
        ct = cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext(list(map(float, values))))
        assert ofhe.SerializeToFile(str(tmp_path / name), ct, ofhe.BINARY)

    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    assert cc.SerializeEvalAutomorphismKey(str(tmp_path / "autokeys.bin"), ofhe.BINARY)
    evalmult = str(tmp_path / "evalmult.bin")
    assert ofhe.CryptoContext.SerializeEvalMultKey(evalmult, ofhe.BINARY, key_tag)

    rng = np.random.default_rng(SEED)
    vectors: dict[tuple[int, int], np.ndarray] = {}
    fields: dict[tuple[int, int], np.ndarray] = {}
    for cluster, idx, n_ids in BLOCKS:
        vec = np.zeros((slots, D_VEC))
        vec[:n_ids] = _unit(rng, D_VEC, n_ids)
        fv = np.zeros((slots, len(CARDS)), dtype=int)
        for f, card in enumerate(CARDS):
            fv[:n_ids, f] = rng.integers(0, card, size=n_ids)
        vectors[(cluster, idx)], fields[(cluster, idx)] = vec, fv
    # Some vectors match the filter, including deleted ones (so the penalty matters).
    for cl, idx, slot in [(0, 0, 0), (0, 0, 1), (0, 1, 2), (1, 0, 2), (2, 0, 0), (2, 0, 3)]:
        fields[(cl, idx)][slot] = WANT

    for cluster, idx, n_ids in BLOCKS:
        vec, fv = vectors[(cluster, idx)], fields[(cluster, idx)]
        cols = [vec[:, j] for j in range(D_VEC)]
        if cluster != N_DATA_CLUSTERS:  # data blocks: one-hot columns, field-major
            for f, card in enumerate(CARDS):
                for k in range(card):
                    onehot = (fv[:, f] == k).astype(float)
                    onehot[n_ids:] = 0.0
                    cols.append(onehot)
            assert len(cols) == n_leaves
        for j, col in enumerate(cols):
            write_ct(f"col_{cluster}_{idx}_{j}.bin", col)

    q = _unit(rng, D_VEC, 1)[0]

    def ext_query(filtered: bool) -> np.ndarray:
        parts = [q]
        for f, card in enumerate(CARDS):
            if filtered:
                oh = np.zeros(card)
                oh[WANT[f]] = LAMBDA
            else:
                oh = np.full(card, LAMBDA)  # all-ones for an unfiltered field (decision 0019)
            parts.append(oh)
        return np.concatenate(parts)

    def write_query(name: str, vec: np.ndarray, pk) -> None:
        slots_v = oq.symmetric_slots(oq.prescale_query(vec), ring)
        ct = cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext(slots_v.tolist()))
        assert ofhe.SerializeToFile(str(tmp_path / name), ct, ofhe.BINARY)

    write_query("query_nofilter.bin", ext_query(False), keys.publicKey)
    write_query("query_filter.bin", ext_query(True), keys.publicKey)
    write_query("query_wrongkey.bin", ext_query(True), other.publicKey)

    lines = [
        f"preset {preset}",
        f"version {VERSION}",
        "collection docs",
        f"key_tag {key_tag}",
        f"d {D_VEC}",
        "card " + " ".join(map(str, CARDS)),
        f"clusters {N_DATA_CLUSTERS}",
        *[f"block {c} {i} {n}" for c, i, n in BLOCKS],
        *[f"delete {x}" for x in DELETED],
        "query nofilter",
        "query filter",
    ]
    (tmp_path / "manifest.txt").write_text("\n".join(lines) + "\n")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    proc = subprocess.run(  # noqa: S603
        [str(_driver()), str(tmp_path), str(out_dir), "3", "4"],
        capture_output=True,
        text=True,
        timeout=3000,
    )
    msg = f"driver failed (exit {proc.returncode}): {proc.stdout}{proc.stderr}"
    assert proc.returncode == 0, msg
    assert "FAIL" not in proc.stdout
    assert "search engine exact match with OpenFHE: PASS" in proc.stdout
    for line in proc.stdout.splitlines():
        print(line)  # verdicts, shapes and counters only

    # ---- Decrypt with the secret key (held only in this process) and compare with numpy --------
    def decrypt(name: str) -> np.ndarray:
        raw = (out_dir / f"{name}.bin").read_bytes()
        env = read_envelope(raw, Kind.CIPHERTEXT, preset, VERSION)
        path = tmp_path / f"{name}.payload"
        path.write_bytes(env.payload)
        ct, ok = ofhe.DeserializeCiphertext(str(path), ofhe.BINARY)
        assert ok
        assert ct.GetLevel() == 1
        dec = cc.Decrypt(ct, keys.secretKey)
        dec.SetLength(slots)
        return np.array(dec.GetRealPackedValue())

    tol = TOLERANCE[preset]
    n_live_matches = 0
    worst = {"score": 0.0, "stage1": 0.0}
    for name_line in (out_dir / "index.txt").read_text().splitlines():
        name, cluster_s, idx_s, penalty_s = name_line.split()
        cluster, idx, penalty = int(cluster_s), int(idx_s), bool(int(penalty_s))
        qname, stage = name.split("_")[0], name.split("_")[1]
        n_ids = next(n for c, i, n in BLOCKS if (c, i) == (cluster, idx))
        got = decrypt(name)[:n_ids]
        if stage == "stage1":
            assert not penalty
            want = vectors[(cluster, idx)][:n_ids] @ q
            worst["stage1"] = max(worst["stage1"], float(np.max(np.abs(got - want))))
            assert np.max(np.abs(got - want)) < tol, name
            continue
        filtered = qname == "filter"
        score = vectors[(cluster, idx)][:n_ids] @ q
        match = fields[(cluster, idx)][:n_ids]
        c_true = (match == np.array(WANT)).sum(axis=1) if filtered else np.full(n_ids, len(CARDS))
        gone = np.array([f"{cluster}.{idx}.{s}" in DELETED for s in range(n_ids)])
        assert penalty == bool(gone.any()), name  # only blocks with a tombstone got the plaintext
        want = score + LAMBDA * c_true - gone * LAMBDA * (len(CARDS) + 1)
        err = float(np.max(np.abs(got - want)))
        worst["score"] = max(worst["score"], err)
        assert err < tol, (name, err)
        live = ~gone
        # Band decode (decision 0019): c = floor((slot + 2) / lambda) recovers the match count.
        assert np.array_equal(np.floor((got[live] + 2.0) / LAMBDA).astype(int), c_true[live]), name
        assert np.all(got[gone] < -2.0), name  # deleted vectors fall below every live slot
        if filtered:
            n_live_matches += int((live & (c_true == len(CARDS))).sum())
    assert n_live_matches > 0  # the filter matched some live vectors
    print(
        f"preset={preset} max_err_stage2={worst['score']:.3e} max_err_stage1={worst['stage1']:.3e}"
    )
