"""T2.7b: end-to-end through the gRPC server `ombra_server`, both presets.

The test starts the server on a free localhost port in a temp directory and drives it with
tests/e2e_client.py: the client creates the keys (the secret key stays in this process), creates a
collection (d = 16, two filter fields with 3 and 4 values), uploads the public keys, inserts 6
clusters of encrypted vectors plus one block of cluster centers, searches, decrypts, decodes the
bands (decision 0019) and takes the top-k.

Cases: (a) no filter: top-10 equals the CPU OpenFHE oracle (the client evaluates the same sum on its
own copies of the ciphertexts) and numpy brute force; (b) filter on one field and on both (AND):
top-10 equals brute force over the matching vectors; (d) a selective filter with fewer than k
matches in the first chosen clusters: a second SearchChosen on the same stream reaches k results,
and a filter with fewer than k matches overall ends `incomplete`; (c) deleted vectors never appear,
cluster centers are never tombstoned; (e) refusals through the API.

CKKS scores carry noise (about 1e-4 at m0_depth2, up to 1e-2 at the compact preset), so two
vectors whose true scores differ by less than the noise may swap at the k-th place. The comparison
is exact on ids; a difference is accepted only if the vectors involved lie within 2 * TOL of the
k-th best true score, and that is counted and printed. Data is synthetic; only aggregates print.
"""

from __future__ import annotations

import threading
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")
pytest.importorskip("openfhe")

import e2e_client as ec  # noqa: E402

pytestmark = pytest.mark.gpu

PRESETS = ["m0_depth2", "m0_compact_n8192"]
VERSION = "test-openfhe"
D = 16
CARDS = [3, 4]
N = 300
CLUSTERS = 6
K = 10
SEED = 20261112
TOL = {"m0_depth2": 1e-3, "m0_compact_n8192": 1e-2}
TOO_MANY_FIELDS = {"m0_depth2": 15, "m0_compact_n8192": 17}  # above 14 / 16 (headroom, proto limit)
SELECTIVE = (1, 2)  # absent from clusters 0 and 1, present elsewhere
RARE = (2, 3)  # only four vectors
RARE_ROWS = (5, 77, 150, 222)


def _ident(i: int) -> bytes:
    return b"v%04d" % i


class Deployment:
    """Server process, client, plaintext data (kept for the oracles) and encrypted copies."""

    def __init__(self, tmp, preset: str, extra_args: tuple[str, ...] = ()) -> None:
        binary = ec.find_server_binary()
        if binary is None:
            pytest.fail("ombra_server not found; build with `make build` or `make test-gpu`")
        self.preset = preset
        self.pb, pbg = ec.load_stubs(tmp / "stubs")
        self.data_dir = tmp / "data"
        self.server = ec.ServerProcess(
            binary, self.data_dir, VERSION, tmp / "server.log", extra_args=extra_args
        )
        self.client = ec.OmbraClient(self.pb, pbg, self.server.address, preset, VERSION)
        self.tol = TOL[preset]
        self.deleted: set[int] = set()
        self._build_data()
        self._upload()

    # ---- data ---------------------------------------------------------------------------------
    def _build_data(self) -> None:
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal((N, D))
        self.x = x / np.linalg.norm(x, axis=1, keepdims=True)
        self.centers = self.x[rng.choice(N, CLUSTERS, replace=False)]
        self.assign = np.argmax(self.x @ self.centers.T, axis=1)
        f = np.stack([rng.integers(0, c, size=N) for c in CARDS], axis=1)
        sel = np.all(f == SELECTIVE, axis=1)
        f[sel & (self.assign < 2), 1] = 3  # no selective match in clusters 0 and 1
        f[np.all(f == RARE, axis=1), 1] = 2  # then plant exactly four rare matches
        for r in RARE_ROWS:
            f[r] = RARE
        later = [i for i in range(N) if self.assign[i] >= 2 and i not in RARE_ROWS]
        for i in later[:14]:
            f[i] = SELECTIVE
        self.fields = f
        self.queries = [self._unit(rng) for _ in range(3)]

    @staticmethod
    def _unit(rng) -> np.ndarray:
        v = rng.standard_normal(D)
        return v / np.linalg.norm(v)

    def _columns(self, vecs: np.ndarray, fields: np.ndarray | None) -> list[np.ndarray]:
        slots = self.client.slots
        cols = []
        for j in range(D):
            col = np.zeros(slots)
            col[: len(vecs)] = vecs[:, j]
            cols.append(col)
        if fields is not None:  # one-hot columns, field by field
            for fi, card in enumerate(CARDS):
                for v in range(card):
                    col = np.zeros(slots)
                    col[: len(vecs)] = (fields[:, fi] == v).astype(float)
                    cols.append(col)
        return cols

    def _upload(self) -> None:
        cl, pb = self.client, self.pb
        cl.make_keys(D + sum(CARDS))
        cl.create_collection(cl.make_spec("docs", D, CARDS, CLUSTERS))
        # Insert before the keys are uploaded is refused (checked in the refusal test, too).
        with pytest.raises(ec.ApiError) as ei:
            cl.insert([pb.InsertBlock(api_version=1, collection="docs", cluster_id=0, ids=[b"x"])])
        assert ei.value.code == pb.ERROR_KEYS_MISSING
        assert not cl.get_collection("docs").keys_uploaded
        cl.check(cl.upload_keys("docs", cl.key_files()).error)
        assert cl.get_collection("docs").keys_uploaded

        self.cols: dict[int, list] = {}  # the client's own ciphertexts, for the CPU oracle
        blocks = []
        for c in range(CLUSTERS):
            rows = np.nonzero(self.assign == c)[0]
            assert len(rows) > 0
            cts = [cl.encrypt(col) for col in self._columns(self.x[rows], self.fields[rows])]
            self.cols[c] = cts
            blocks.append(
                pb.InsertBlock(
                    api_version=1,
                    collection="docs",
                    cluster_id=c,
                    ids=[_ident(int(i)) for i in rows],
                    columns=[cl.ct_envelope(ct) for ct in cts],
                )
            )
        center_cts = [cl.encrypt(col) for col in self._columns(self.centers, None)]
        blocks.append(
            pb.InsertBlock(
                api_version=1,
                collection="docs",
                cluster_id=CLUSTERS,
                ids=[b"center%d" % c for c in range(CLUSTERS)],
                columns=[cl.ct_envelope(ct) for ct in center_cts],
                is_center=True,
            )
        )
        assert cl.insert(blocks) == N + CLUSTERS
        info = cl.get_collection("docs")
        assert info.vector_count == N and info.deleted_count == 0  # centers are not vectors

    # ---- oracles ------------------------------------------------------------------------------
    def brute_force(self, q, wants, k=K, clusters=None) -> tuple[list[bytes], dict[bytes, float]]:
        """numpy: top-k over live vectors that match the filter (ties by index)."""
        rows = [
            i
            for i in range(N)
            if i not in self.deleted
            and (clusters is None or int(self.assign[i]) in clusters)
            and all(w is None or self.fields[i, fi] == w for fi, w in enumerate(wants))
        ]
        true = {_ident(i): float(self.x[i] @ q) for i in rows}
        rows.sort(key=lambda i: (-float(self.x[i] @ q), i))
        return [_ident(i) for i in rows[:k]], true

    def cpu_oracle(self, q, wants, k=K) -> list[bytes]:
        """OpenFHE on the CPU: the client evaluates the same sum on its own ciphertext copies."""
        cl = self.client
        ext = cl.ext_query(q, CARDS, wants)
        cand = []
        for c in range(CLUSTERS):
            rows = np.nonzero(self.assign == c)[0]
            c_band, sc = cl.decode_band(cl.oracle_scores(self.cols[c], ext))
            for j, i in enumerate(rows):
                if int(i) not in self.deleted and c_band[j] == len(CARDS):
                    cand.append((float(sc[j]), int(i)))
        cand.sort(key=lambda t: (-t[0], t[1]))
        return [_ident(i) for _, i in cand[:k]]

    def assert_same_topk(self, got, want, true, label: str) -> int:
        """Exact ids; a swap is accepted only inside the noise band around the k-th score."""
        assert len(got) == len(want), (label, len(got), len(want))
        if got == want:
            return 0
        kth = true[want[-1]]
        diff = set(got) ^ set(want)
        for i in diff:
            assert abs(true[i] - kth) <= 2 * self.tol, (label, "difference outside the noise band")
        for a, b in zip(got, got[1:], strict=False):
            assert true[a] >= true[b] - 2 * self.tol, (label, "order outside the noise band")
        return len(diff)

    def close(self) -> None:
        self.client.close()
        self.server.stop()


@pytest.fixture(scope="module", params=PRESETS)
def dep(request, tmp_path_factory):
    d = Deployment(tmp_path_factory.mktemp(request.param), request.param)
    yield d
    d.close()


@pytest.fixture(scope="module")
def dep_c(tmp_path_factory):
    """A second deployment whose server compacts every 200 ms (compact preset only, faster)."""
    d = Deployment(
        tmp_path_factory.mktemp("compact"), "m0_compact_n8192", ("--compact-interval-ms", "200")
    )
    yield d
    d.close()


def _search_all(dep: Deployment, q, wants):
    return dep.client.search("docs", q, CARDS, wants, K)


def test_1_no_filter_equals_oracle_and_numpy(dep: Deployment) -> None:
    swaps = 0
    for n, q in enumerate(dep.queries):
        wants = [None, None]
        res = _search_all(dep, q, wants)
        assert res.rounds == 1 and not res.incomplete
        want, true = dep.brute_force(q, wants)
        swaps += dep.assert_same_topk(res.ids, want, true, f"numpy q{n}")
        swaps += dep.assert_same_topk(res.ids, dep.cpu_oracle(q, wants), true, f"oracle q{n}")
        for ident, s in zip(res.ids, res.scores, strict=True):
            assert abs(s - true[ident]) < dep.tol
    print(f"preset={dep.preset} no-filter swaps_in_noise_band={swaps}")


def test_2_filters_equal_brute_force(dep: Deployment) -> None:
    q = dep.queries[0]
    for wants in ([1, None], [None, 2], [1, 2], [0, 3]):
        res = _search_all(dep, q, wants)
        want, true = dep.brute_force(q, wants)
        assert len(want) == K and not res.incomplete, wants
        dep.assert_same_topk(res.ids, want, true, f"filter {wants}")
        dep.assert_same_topk(res.ids, dep.cpu_oracle(q, wants), true, f"oracle {wants}")
        for ident in res.ids:  # every returned vector matches the filter
            i = int(ident[1:])
            assert all(w is None or dep.fields[i, fi] == w for fi, w in enumerate(wants))


def test_3_selective_filter_needs_a_second_round(dep: Deployment) -> None:
    q = dep.queries[1]
    # Scan clusters two at a time, 0 and 1 first: they hold no vector with the selective values.
    res = dep.client.search(
        "docs", q, CARDS, list(SELECTIVE), K, order=list(range(CLUSTERS)), batch=2
    )
    assert res.first_round_matches == 0 and res.rounds >= 2
    assert len(res.ids) == K and not res.incomplete
    want, true = dep.brute_force(q, list(SELECTIVE), clusters=set(res.scanned))
    dep.assert_same_topk(res.ids, want, true, "second round")

    # Four matches overall: all clusters are scanned and the answer is incomplete.
    res = dep.client.search("docs", q, CARDS, list(RARE), K, order=list(range(CLUSTERS)), batch=2)
    want, true = dep.brute_force(q, list(RARE))
    assert res.incomplete and len(res.ids) == 4 and res.rounds == CLUSTERS // 2
    dep.assert_same_topk(res.ids, want, true, "incomplete")


def test_4_deleted_vectors_never_appear(dep: Deployment) -> None:
    cl, q = dep.client, dep.queries[0]
    top, _ = dep.brute_force(q, [None, None])
    victims = list(dict.fromkeys([int(i[1:]) for i in top[:3]] + [7, 100, 250]))  # unique ids
    ask = [_ident(i) for i in victims] + [b"center0", b"center3", b"unknown-id"]
    assert cl.delete("docs", ask) == len(victims)  # centers and unknown ids are ignored
    assert cl.delete("docs", ask) == 0  # already deleted
    dep.deleted.update(victims)
    info = cl.get_collection("docs")
    assert info.vector_count == N and info.deleted_count == len(victims)

    for wants in ([None, None], [1, None]):
        res = _search_all(dep, q, wants)
        assert not {_ident(i) for i in victims} & set(res.ids)
        want, true = dep.brute_force(q, wants)
        dep.assert_same_topk(res.ids, want, true, f"after delete {wants}")
        dep.assert_same_topk(res.ids, dep.cpu_oracle(q, wants), true, f"oracle {wants}")
    # The centers are untouched: stage 1 still returns all of them.
    stream, centers = cl.open_search(
        "docs", cl.encrypt_query(cl.ext_query(q, CARDS, [None, None])), sum(CARDS)
    )
    try:
        assert centers.centers.center_count == CLUSTERS
        got = cl.decrypt_envelope(centers.centers.score_blocks[0])[:CLUSTERS]
        assert np.max(np.abs(got - dep.centers @ q)) < dep.tol
    finally:
        stream.close()


def test_5_refusals(dep: Deployment) -> None:
    cl, pb = dep.client, dep.pb
    q = dep.queries[0]

    def code(fn) -> int:
        with pytest.raises(ec.ApiError) as ei:
            fn()
        return ei.value.code

    # Unknown api_version on every stored message.
    assert code(lambda: cl.create_collection(cl.make_spec("v2", D, CARDS, 2, api_version=2))) == (
        pb.ERROR_UNSUPPORTED_VERSION
    )
    assert (
        code(lambda: cl.delete("docs", [b"v0001"], api_version=2)) == pb.ERROR_UNSUPPORTED_VERSION
    )
    resp = cl.upload_keys("docs", cl.key_files(), api_version=2)
    assert resp.error.code == pb.ERROR_UNSUPPORTED_VERSION
    blk = pb.InsertBlock(api_version=2, collection="docs", cluster_id=0, ids=[b"x"])
    assert code(lambda: cl.insert([blk])) == pb.ERROR_UNSUPPORTED_VERSION

    # Wrong key tag in SearchStart, and a query made under another key.
    ext = cl.ext_query(q, CARDS, [None, None])
    query = cl.encrypt_query(ext)
    assert code(lambda: cl.open_search("docs", query, sum(CARDS), key_tag="not-the-tag")) == (
        pb.ERROR_KEY_TAG_MISMATCH
    )
    other = cl.encrypt_query(ext, public_key=cl.other.publicKey)
    assert code(lambda: cl.open_search("docs", other, sum(CARDS))) == pb.ERROR_KEY_TAG_MISMATCH
    # filter_leaves that does not match the collection.
    assert code(lambda: cl.open_search("docs", query, sum(CARDS) + 1)) == (
        pb.ERROR_INVALID_FILTER_LAYOUT
    )

    # Search (and insert) before the evaluation keys were uploaded.
    cl.create_collection(cl.make_spec("nokeys", D, CARDS, 2))
    assert code(lambda: cl.open_search("nokeys", query, sum(CARDS))) == pb.ERROR_KEYS_MISSING

    # More filter fields than the preset allows.
    many = cl.make_spec("many", D, [2] * TOO_MANY_FIELDS[dep.preset], 2)
    assert code(lambda: cl.create_collection(many)) == pb.ERROR_INVALID_FILTER_LAYOUT
    # Cardinality above the limit of 64.
    big = cl.make_spec("big", D, [65], 2)
    assert code(lambda: cl.create_collection(big)) == pb.ERROR_INVALID_FILTER_LAYOUT

    # Unknown and existing collections.
    assert code(lambda: cl.get_collection("nope")) == pb.ERROR_COLLECTION_NOT_FOUND
    assert code(lambda: cl.drop_collection("nope")) == pb.ERROR_COLLECTION_NOT_FOUND
    assert code(lambda: cl.delete("nope", [b"a"])) == pb.ERROR_COLLECTION_NOT_FOUND
    assert code(lambda: cl.open_search("nope", query, sum(CARDS))) == pb.ERROR_COLLECTION_NOT_FOUND
    assert code(lambda: cl.create_collection(cl.make_spec("docs", D, CARDS, 2))) == (
        pb.ERROR_COLLECTION_EXISTS
    )

    # Bad stream order: SearchChosen first; a key chunk with a wrong offset.
    stream = ec.SearchStream(cl.stub, pb.SearchRequest(chosen=pb.SearchChosen(cluster_ids=[0])))
    with pytest.raises(ec.ApiError) as ei:
        stream.next()
    stream.abort()
    assert ei.value.code == pb.ERROR_BAD_SEQUENCE
    bad = pb.UploadKeysChunk(
        api_version=1, collection="nokeys", kind=pb.KEY_KIND_CONTEXT, offset=5, data=b"x", last=True
    )
    assert cl.stub.UploadKeys(iter([bad])).error.code == pb.ERROR_BAD_SEQUENCE

    # Drop and list.
    cl.drop_collection("nokeys")
    assert code(lambda: cl.get_collection("nokeys")) == pb.ERROR_COLLECTION_NOT_FOUND
    names = [c.spec.name for c in cl.stub.ListCollections(pb.ListCollectionsRequest()).collections]
    assert names == ["docs"]


def _segment_files(dep: Deployment) -> set[str]:
    return {p.name for p in (dep.data_dir / "segments").iterdir()}


def _stored_bytes(dep: Deployment) -> int:
    return sum(p.stat().st_size for p in (dep.data_dir / "segments").iterdir())


def _block_of(dep: Deployment, cluster: int):
    (b,) = [b for b in dep.client.list_blocks("docs") if b.cluster_id == cluster]
    return b


def test_6_bad_key_upload_leaves_nothing(dep: Deployment) -> None:
    """The key tag is checked before anything is stored: no key file, no catalog row."""
    import sqlite3

    cl, pb = dep.client, dep.pb
    keys_dir = dep.data_dir / "keys"
    before = sorted(p.name for p in keys_dir.iterdir())
    # The collection claims the other key pair's tag; the uploaded keys belong to the first one.
    other_tag = cl.other.publicKey.GetKeyTag()
    cl.create_collection(cl.make_spec("badkeys", D, CARDS, 2, key_tag=other_tag))
    resp = cl.upload_keys("badkeys", cl.key_files())
    assert resp.error.code == pb.ERROR_KEY_TAG_MISMATCH
    assert sorted(p.name for p in keys_dir.iterdir()) == before
    assert not cl.get_collection("badkeys").keys_uploaded
    con = sqlite3.connect(f"file:{dep.data_dir / 'catalog.db'}?mode=ro", uri=True)
    try:
        n = con.execute("SELECT COUNT(*) FROM key_files WHERE collection='badkeys'").fetchone()[0]
    finally:
        con.close()
    assert n == 0
    # An upload with fewer than three kinds is refused the same way.
    part = {k: v for k, v in cl.key_files().items() if k != pb.KEY_KIND_AUTOMORPHISM}
    assert cl.upload_keys("badkeys", part).error.code == pb.ERROR_INVALID_ARGUMENT
    assert sorted(p.name for p in keys_dir.iterdir()) == before
    cl.drop_collection("badkeys")
    # The loaded keys of the working collection are untouched.
    res = _search_all(dep, dep.queries[0], [None, None])
    want, true = dep.brute_force(dep.queries[0], [None, None])
    dep.assert_same_topk(res.ids, want, true, "after bad upload")


def test_7_dead_block_is_dropped_and_search_stays_correct(dep_c: Deployment) -> None:
    cl, q = dep_c.client, dep_c.queries[0]
    victim_cluster = 2
    rows = [int(i) for i in np.nonzero(dep_c.assign == victim_cluster)[0]]
    blk = _block_of(dep_c, victim_cluster)
    assert blk.n_vectors == len(rows) and blk.n_deleted == 0
    assert blk.segment in _segment_files(dep_c)
    want, true = dep_c.brute_force(q, [None, None])
    dep_c.assert_same_topk(_search_all(dep_c, q, [None, None]).ids, want, true, "before")

    assert cl.delete("docs", [_ident(i) for i in rows]) == len(rows)
    dep_c.deleted.update(rows)
    want, true = dep_c.brute_force(q, [None, None])
    assert not {_ident(i) for i in rows} & set(want)

    errors: list[BaseException] = []
    stop = threading.Event()

    def searcher() -> None:
        try:
            while not stop.is_set():
                got = _search_all(dep_c, q, [None, None])
                dep_c.assert_same_topk(got.ids, want, true, "during compaction")
                assert not {_ident(i) for i in rows} & set(got.ids)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=searcher)
    t.start()
    try:
        deadline = time.monotonic() + 30
        while blk.segment in _segment_files(dep_c) and time.monotonic() < deadline:
            time.sleep(0.1)
        time.sleep(1.0)  # keep searching a little after the drop
    finally:
        stop.set()
        t.join()
    assert not errors, errors
    assert blk.segment not in _segment_files(dep_c), "the dead block was not dropped"
    assert all(b.cluster_id != victim_cluster for b in cl.list_blocks("docs"))
    info = cl.get_collection("docs")
    assert info.vector_count == N - len(rows) and info.deleted_count == 0
    dep_c.assert_same_topk(_search_all(dep_c, q, [None, None]).ids, want, true, "after")
    for wants in ([1, None], [None, 2]):
        w, tr = dep_c.brute_force(q, wants)
        dep_c.assert_same_topk(_search_all(dep_c, q, wants).ids, w, tr, f"after {wants}")


def test_8_client_repack_with_get_and_replace_block(dep_c: Deployment) -> None:
    cl, pb, q = dep_c.client, dep_c.pb, dep_c.queries[1]
    cluster = 0
    rows = [int(i) for i in np.nonzero(dep_c.assign == cluster)[0]]
    assert len(rows) >= 4
    victims = rows[::2]  # scattered deletes: every other slot
    assert cl.delete("docs", [_ident(i) for i in victims]) == len(victims)
    dep_c.deleted.update(victims)
    blk = _block_of(dep_c, cluster)
    assert blk.n_vectors == len(rows) and blk.n_deleted == len(victims)  # counts only
    bytes_before = _stored_bytes(dep_c)

    # The client repacks: decrypt every column, drop the deleted slots, re-encrypt.
    header, cols = cl.get_block("docs", blk.segment)
    ids = [bytes(i) for i in header.ids]
    dead = list(header.deleted)
    assert len(ids) == len(dead) == len(rows) and sum(dead) == len(victims)
    live = [j for j, d in enumerate(dead) if not d][::-1]  # the client picks the slot order
    new_cols = []
    for env in cols:
        vals = cl.decrypt_envelope(env)
        packed = np.zeros(cl.slots)
        packed[: len(live)] = vals[live]
        new_cols.append(cl.ct_envelope(cl.encrypt(packed)))

    live_ids = [ids[j] for j in live]
    # Refusals first: a dead id kept, a live id missing; the block stays as it was.
    with pytest.raises(ec.ApiError) as ei:
        cl.replace_block("docs", blk.segment, ids, new_cols)
    assert ei.value.code == pb.ERROR_BLOCK_MISMATCH
    with pytest.raises(ec.ApiError) as ei:
        cl.replace_block("docs", blk.segment, live_ids[1:], new_cols)
    assert ei.value.code == pb.ERROR_BLOCK_MISMATCH
    with pytest.raises(ec.ApiError) as ei:
        cl.replace_block("docs", "s_missing.oseg", live_ids, new_cols)
    assert ei.value.code == pb.ERROR_BLOCK_NOT_FOUND
    assert _block_of(dep_c, cluster).segment == blk.segment

    new_segment = cl.replace_block("docs", blk.segment, live_ids, new_cols)
    assert new_segment != blk.segment
    nb = _block_of(dep_c, cluster)
    assert nb.segment == new_segment and nb.n_vectors == len(live_ids) and nb.n_deleted == 0
    deadline = time.monotonic() + 10
    while blk.segment in _segment_files(dep_c) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert blk.segment not in _segment_files(dep_c)
    # A column-packed block always holds full-size ciphertexts, so 1 -> 1 does not shrink the
    # stored bytes; merging N blocks into one does (test 9).
    assert _stored_bytes(dep_c) <= bytes_before
    info = cl.get_collection("docs")
    data_blocks = [b for b in cl.list_blocks("docs") if b.cluster_id != CLUSTERS]
    assert info.deleted_count == sum(b.n_deleted for b in data_blocks) == 0
    assert info.vector_count == sum(b.n_vectors for b in data_blocks)

    for wants in ([None, None], [1, None], [None, 2]):
        got = _search_all(dep_c, q, wants)
        w, tr = dep_c.brute_force(q, wants)
        dep_c.assert_same_topk(got.ids, w, tr, f"after repack {wants}")
        assert not {_ident(i) for i in victims} & set(got.ids)


def _file_size(dep: Deployment, segment: str) -> int:
    return (dep.data_dir / "segments" / segment).stat().st_size


def test_9_merge_two_half_deleted_blocks_into_one(dep_c: Deployment) -> None:
    """ReplaceBlock N -> 1: the stored bytes fall by one block; results stay correct."""
    cl, pb = dep_c.client, dep_c.pb
    cluster = 1
    rng = np.random.default_rng(SEED + 9)
    rows = [int(i) for i in np.nonzero(dep_c.assign == cluster)[0]]
    assert len(rows) >= 4
    # A second block in the same cluster: 12 new vectors, four of them close to the query.
    q = dep_c.queries[2]
    n_extra = 12
    ex = rng.standard_normal((n_extra, D))
    for i in range(4):
        ex[i] = q + 0.05 * rng.standard_normal(D)
    ex = ex / np.linalg.norm(ex, axis=1, keepdims=True)
    ef = np.stack([rng.integers(0, c, size=n_extra) for c in CARDS], axis=1)
    eids = [b"e%04d" % i for i in range(n_extra)]
    cols_e = [cl.ct_envelope(cl.encrypt(c)) for c in dep_c._columns(ex, ef)]
    blk = pb.InsertBlock(
        api_version=1, collection="docs", cluster_id=cluster, ids=eids, columns=cols_e
    )
    assert cl.insert([blk]) == n_extra
    old = [b for b in cl.list_blocks("docs") if b.cluster_id == cluster]
    assert len(old) == 2

    # Scattered deletes in both blocks (extra vectors 0 and 2 are close to q and get deleted).
    victims = rows[::2]
    evict = list(range(0, n_extra, 2))
    assert cl.delete("docs", [_ident(i) for i in victims] + [eids[i] for i in evict]) == (
        len(victims) + len(evict)
    )
    dep_c.deleted.update(victims)
    old = [b for b in cl.list_blocks("docs") if b.cluster_id == cluster]  # fresh counts
    assert len(old) == 2

    # The client reads both blocks and merges the live slots into one (its own slot order).
    live_ids: list[bytes] = []
    live_vals: list[list[np.ndarray]] = []  # per block: per column, the live slot values
    old_files = []
    for b in old:
        header, cols = cl.get_block("docs", b.segment)
        ids = [bytes(i) for i in header.ids]
        live = [j for j, d in enumerate(header.deleted) if not d]
        assert len(ids) == b.n_vectors and len(ids) - len(live) == b.n_deleted
        live_ids += [ids[j] for j in live]
        live_vals.append([cl.decrypt_envelope(c)[live] for c in cols])
        old_files.append((b.segment, ids, cols))
    n_cols = len(live_vals[0])
    assert len(live_ids) <= cl.slots
    new_cols = []
    for j in range(n_cols):
        packed = np.zeros(cl.slots)
        merged = np.concatenate([v[j] for v in live_vals])
        packed[: len(merged)] = merged
        new_cols.append(cl.ct_envelope(cl.encrypt(packed)))

    def code(fn) -> int:
        with pytest.raises(ec.ApiError) as ei:
            fn()
        return ei.value.code

    segs = [b.segment for b in old]
    assert code(lambda: cl.replace_block("docs", segs, live_ids[1:], new_cols)) == (
        pb.ERROR_BLOCK_MISMATCH
    )
    assert code(lambda: cl.replace_block("docs", segs + segs[:1], live_ids, new_cols)) == (
        pb.ERROR_BLOCK_MISMATCH
    )
    other = _block_of(dep_c, 3).segment  # a block of another cluster
    assert code(lambda: cl.replace_block("docs", [segs[0], other], live_ids, new_cols)) == (
        pb.ERROR_BLOCK_MISMATCH
    )
    assert [b.segment for b in cl.list_blocks("docs") if b.cluster_id == cluster] == segs

    before = _stored_bytes(dep_c)
    sizes_old = [_file_size(dep_c, sg) for sg in segs]
    assert before >= sum(sizes_old)
    new_segment = cl.replace_block("docs", segs, live_ids, new_cols)
    deadline = time.monotonic() + 10
    while any(sg in _segment_files(dep_c) for sg in segs) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(sg in _segment_files(dep_c) for sg in segs)
    now = [b for b in cl.list_blocks("docs") if b.cluster_id == cluster]
    assert [b.segment for b in now] == [new_segment]
    assert now[0].n_vectors == len(live_ids) and now[0].n_deleted == 0

    # Exact byte accounting. Segment file = 22 bytes of header and counts, per id 4 + len(id),
    # per column 8 + len(envelope). All blocks have the same columns, so the difference between
    # the old blocks and the new one is exactly one block of columns plus one header and the ids
    # that were dropped.
    def file_size(ids, cols) -> int:
        return 22 + sum(4 + len(i) for i in ids) + sum(8 + len(c) for c in cols)

    for (_sg, ids, cols), size in zip(old_files, sizes_old, strict=True):
        assert size == file_size(ids, cols)
    assert _file_size(dep_c, new_segment) == file_size(live_ids, new_cols)
    one_block_columns = sum(8 + len(c) for c in old_files[0][2])
    assert sum(len(c) for c in new_cols) == sum(len(c) for c in old_files[0][2])
    delta = before - _stored_bytes(dep_c)
    dropped_ids = sum(4 + len(i) for _, ids, _ in old_files for i in ids) - sum(
        4 + len(i) for i in live_ids
    )
    assert delta == one_block_columns + 22 + dropped_ids
    assert delta > 0.9 * one_block_columns  # about one block of ciphertexts

    # Search equals the numpy oracle over the originals plus the extra vectors.
    top_all, true = dep_c.brute_force(q, [None, None], k=N)
    for i in range(n_extra):
        if i not in evict:
            true[eids[i]] = float(ex[i] @ q)
    want = sorted(true, key=lambda i: (-true[i], i))[:K]
    got = _search_all(dep_c, q, [None, None])
    dep_c.assert_same_topk(got.ids, want, true, "after merge")
    assert eids[1] in want and not {eids[i] for i in evict} & set(got.ids)
    assert not {_ident(i) for i in victims} & set(got.ids)
