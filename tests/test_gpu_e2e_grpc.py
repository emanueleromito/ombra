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

    def __init__(self, tmp, preset: str) -> None:
        binary = ec.find_server_binary()
        if binary is None:
            pytest.fail("ombra_server not found; build with `make build` or `make test-gpu`")
        self.preset = preset
        self.pb, pbg = ec.load_stubs(tmp / "stubs")
        self.server = ec.ServerProcess(binary, tmp / "data", VERSION, tmp / "server.log")
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
    victims = [int(i[1:]) for i in top[:3]] + [7, 100, 250]
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
