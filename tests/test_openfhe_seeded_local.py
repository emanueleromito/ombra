# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""SEED-1, client and local mode: seeded stored columns (docs/decisions/0023) end to end.

CPU only, synthetic data, d = 384 on the "fast" preset (m0_compact_n8192). Checks: (a) a collection
created seeded answers like the same data created with OMBRA_SEEDED=0 (plain, filtered, after a
delete, after a repack and after a recluster) and the stored column bytes are about half;
(b) fresh seeds: two encryptions of the same values differ and the seed source raises on failure;
(c) a client without the compiled module writes kind 2 and refuses kind 4 naming ombra._native;
(d) queries stay ordinary ciphertexts (kind 2); a truncated seeded column is refused at insert.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("openfhe")
pytest.importorskip("grpc_tools")
pytest.importorskip("faiss")
native = pytest.importorskip("ombra._native", reason="the compiled extension is not built")

from ombra import _seeding  # noqa: E402
from ombra._local import LocalTransport  # noqa: E402
from ombra._proto import insert_requests  # noqa: E402
from ombra.collection import Collection, KeyKit  # noqa: E402
from ombra.file_format import Kind, read_column_envelope  # noqa: E402
from ombra.seeded_format import read_payload  # noqa: E402

pytestmark = pytest.mark.openfhe

D = 384
CARDS = (3,)
K = 5
TOL = 1e-2  # compact preset noise


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _stored_bytes(col: Collection) -> tuple[int, set[int]]:
    """Bytes of the stored data columns of every block, and the envelope kinds found."""
    total, kinds = 0, set()
    for b in col.block_stats():
        header, cols = col._get_block(b.segment)
        total += sum(len(c) for c in cols)
        kinds |= {
            int(read_column_envelope(c, col.derived.preset, col.kit.openfhe_version).kind)
            for c in cols
        }
    return total, kinds


def _brute(x, fields, live, q, want=None):
    rows = [i for i in range(len(x)) if live[i] and (want is None or fields[i, 0] == want)]
    rows.sort(key=lambda i: (-float(x[i] @ q), i))
    return rows[:K], {i: float(x[i] @ q) for i in rows}


def _check_hits(hits, brute, true):
    got = [int(h.id[1:]) for h in hits]
    kth = true[brute[-1]]
    assert len(got) == len(brute)
    for i in set(got) ^ set(brute):
        assert abs(true[i] - kth) <= 2 * TOL
    for h in hits:
        assert abs(h.score - true[int(h.id[1:])]) < TOL


def _run(tmp_path, monkeypatch, seeded: bool):
    """Create, search (plain and filtered), delete, repack, recluster; return what to compare."""
    monkeypatch.setenv("OMBRA_SEEDED", "1" if seeded else "0")
    transport = LocalTransport(tmp_path / ("s" if seeded else "p"))
    pb, stub = transport.pb, transport.stub
    out: dict = {}
    try:
        rng = np.random.default_rng(11)
        n = 60
        x = _unit(rng.standard_normal((n + 30, D)))
        fields = rng.integers(0, CARDS[0], size=(n + 30, 1))
        ids = [b"v%04d" % i for i in range(n + 30)]
        live = np.ones(n + 30, dtype=bool)
        qs = [_unit(rng.standard_normal((1, D)))[0] for _ in range(2)]
        col = Collection.create(
            pb,
            stub,
            "c",
            x[:n],
            ids[:n],
            preset="fast",
            fields=fields[:n],
            cards=CARDS,
            n_clusters=2,
            openfhe_version=transport.openfhe_version,
        )
        total, kinds = _stored_bytes(col)
        out["bytes"], out["kinds"] = total, kinds
        for q in qs:
            for want in (None, 1):
                res = col.search(q, K, filter=None if want is None else {0: want})
                b, true = _brute(x[:n], fields[:n], live[:n], q, want)
                _check_hits(res.hits, b, true)
        col.insert_vectors(x[n:], ids[n:], fields[n:])
        victims = [i for i in range(n + 30) if i % 5 in (0, 1)]
        assert col.delete([ids[i] for i in victims]) == len(victims)
        live[victims] = False
        for q in qs:
            b, true = _brute(x, fields, live, q)
            _check_hits(col.search(q, K).hits, b, true)
        rep = col.repack(0.3)
        assert rep.groups > 0
        for q in qs:
            b, true = _brute(x, fields, live, q)
            _check_hits(col.search(q, K).hits, b, true)
            b, true = _brute(x, fields, live, q, 2)
            _check_hits(col.search(q, K, filter={0: 2}).hits, b, true)
        new = col.recluster(n_clusters=2)
        col.retire()
        out["kinds_after"] = _stored_bytes(new)[1]
        for q in qs:
            b, true = _brute(x, fields, live, q)
            _check_hits(new.search(q, K).hits, b, true)
        out["top"] = [[h.id for h in new.search(q, K).hits] for q in qs]
        new.close()
    finally:
        transport.close()
    return out


def test_seeded_collection_equals_plain_and_halves_the_bytes(tmp_path, monkeypatch) -> None:
    plain = _run(tmp_path, monkeypatch, seeded=False)
    seeded = _run(tmp_path, monkeypatch, seeded=True)
    assert plain["kinds"] == {int(Kind.CIPHERTEXT)} == plain["kinds_after"]
    assert seeded["kinds"] == {int(Kind.SEEDED_CIPHERTEXT)} == seeded["kinds_after"]
    assert plain["top"] == seeded["top"]
    ratio = seeded["bytes"] / plain["bytes"]
    assert 0.45 <= ratio <= 0.60, ratio


def test_seeds_are_fresh_and_the_seed_source_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OMBRA_SEEDED", "1")
    kit = KeyKit(__import__("ombra.presets", fromlist=["derive"]).derive("fast", D), "1.6.1", D)
    vals = np.linspace(-1, 1, 16)
    a, b = kit.encrypt_columns([vals, vals])
    pa = read_payload(read_column_envelope(a, kit.derived.preset, "1.6.1").payload)
    pb_ = read_payload(read_column_envelope(b, kit.derived.preset, "1.6.1").payload)
    assert len(pa.seed) == 32 and pa.seed != pb_.seed and pa.c0 != pb_.c0
    assert pa.key_tag == kit.key_tag
    assert np.allclose(kit.decrypt_column(a)[:16], vals, atol=1e-3)
    assert np.allclose(kit.decrypt_column(b)[:16], vals, atol=1e-3)
    # fresh_seed: never the same twice, and a failing source raises (no partial seed is returned)
    assert len({native.fresh_seed() for _ in range(64)}) == 64
    with pytest.raises(RuntimeError):
        native._seed_from_file("/dev/null")
    with pytest.raises(RuntimeError):
        native._seed_from_file(str(tmp_path / "missing"))
    short = tmp_path / "short"
    short.write_bytes(b"x" * 31)
    with pytest.raises(RuntimeError):
        native._seed_from_file(str(short))


def test_no_native_module_writes_kind_2_and_refuses_kind_4(tmp_path, monkeypatch) -> None:
    from ombra import presets

    derived = presets.derive("fast", D)
    kit = KeyKit(derived, "1.6.1", D)
    monkeypatch.setenv("OMBRA_SEEDED", "1")
    seeded = kit.encrypt_column(np.ones(4) * 0.5)
    assert read_column_envelope(seeded, derived.preset, "1.6.1").kind == Kind.SEEDED_CIPHERTEXT
    monkeypatch.setattr(_seeding, "native_installed", lambda: False)  # the import is hidden
    assert not _seeding.available()
    plain = kit.encrypt_column(np.ones(4) * 0.5)
    assert read_column_envelope(plain, derived.preset, "1.6.1").kind == Kind.CIPHERTEXT
    assert np.allclose(kit.decrypt_column(plain)[:4], 0.5, atol=1e-3)
    with pytest.raises(_seeding.SeededUnavailableError, match=r"ombra\._native"):
        kit.decrypt_column(seeded)
    monkeypatch.setattr(_seeding, "native_installed", lambda: True)
    monkeypatch.setenv("OMBRA_SEEDED", "0")  # the opt-out
    assert not _seeding.available()


def test_query_stays_kind_2_and_truncated_seeded_column_is_refused(tmp_path, monkeypatch) -> None:
    from ombra import presets
    from ombra.file_format import EnvelopeError, read_envelope

    monkeypatch.setenv("OMBRA_SEEDED", "1")
    derived = presets.derive("fast", D)
    transport = LocalTransport(tmp_path / "t")
    pb, stub = transport.pb, transport.stub
    try:
        other = KeyKit(derived, transport.openfhe_version, D)  # first: the key store is static
        mine = KeyKit(derived, transport.openfhe_version, D)
        q = mine.encrypt_query(np.ones(D) / np.sqrt(D))
        assert read_envelope(q, Kind.CIPHERTEXT, derived.preset, transport.openfhe_version)
        with pytest.raises(EnvelopeError):  # a query is never seeded
            read_envelope(q, Kind.SEEDED_CIPHERTEXT, derived.preset, transport.openfhe_version)
        spec = pb.CollectionSpec(
            api_version=1,
            name="c",
            preset=derived.preset,
            dimension=D,
            key_tag=mine.key_tag,
            cluster_count=1,
        )
        assert stub.CreateCollection(pb.CreateCollectionRequest(spec=spec)).WhichOneof(
            "result"
        ) == ("spec")
        chunks = []
        for kind, data in mine.key_files(pb).items():
            for off in range(0, len(data), 4_000_000):
                chunks.append(
                    pb.UploadKeysChunk(
                        api_version=1,
                        collection="c" if not chunks else "",
                        kind=kind,
                        offset=off,
                        data=data[off : off + 4_000_000],
                        last=off + 4_000_000 >= len(data),
                    )
                )
        assert not stub.UploadKeys(iter(chunks)).error.code
        # The insert accepts well-formed seeded columns and refuses a truncated one. (Another
        # key's columns are refused at search time by the key-tag check of Engine.load_columns.)
        cols = other.encrypt_columns([np.zeros(4)] * D)
        blk = pb.InsertBlock(
            api_version=1, collection="c", cluster_id=1, ids=[b"center0"], columns=cols,
            is_center=True,
        )  # fmt: skip
        assert not stub.Insert(insert_requests(pb, [blk])).error.code
        truncated = [*cols[:-1], cols[-1][:-8]]
        blk2 = pb.InsertBlock(
            api_version=1, collection="c", cluster_id=0, ids=[b"x"], columns=truncated
        )
        assert stub.Insert(insert_requests(pb, [blk2])).error.code != 0
    finally:
        transport.close()
