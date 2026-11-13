# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""SEED-1: seeded stored ciphertexts (file-formats.md) against OpenFHE, CPU only.

Runs the C++ test `ombra_seeded_exact` (client/native/seeded_test.cpp) once per preset of
core/params/ckks_presets.json, parameters read through ombra.params:
  - the phase c0 + c1*s equals c0' + u*s word for word in every tower, metadata are equal, the
    expanded ciphertext decrypts like OpenFHE's own (1e-6 m0_depth2, 1e-5 m0_compact_n8192);
  - generator known-answer test: the C++ generator against an INDEPENDENT Python implementation
    of the specification (BLAKE2xb built from hashlib.blake2b following the BLAKE2X paper) and
    against the pinned fixture tests/fixtures/seeded_kat.json.
Synthetic data only; the executable prints no secrets.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params

pytestmark = pytest.mark.openfhe

REPO_ROOT = Path(__file__).resolve().parents[1]
KAT = REPO_ROOT / "tests" / "fixtures" / "seeded_kat.json"
TOL = {"m0_depth2": "1e-6", "m0_compact_n8192": "1e-5"}


def _binary() -> Path:
    candidates = [
        os.environ.get("OMBRA_SEEDED_EXACT_BIN", ""),
        "/root/ombra-build/client/ombra_seeded_exact",
        str(REPO_ROOT / "build-native" / "ombra_seeded_exact"),
        str(REPO_ROOT / "build-openfhe" / "client" / "ombra_seeded_exact"),
    ]
    for c in candidates:
        if c and os.access(c, os.X_OK) and Path(c).is_file():
            return Path(c)
    pytest.fail("ombra_seeded_exact not found; build client/ (see client/CMakeLists.txt)")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [str(_binary()), *args], capture_output=True, text=True, timeout=1500, check=False
    )


_IV = (
    0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
    0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179,
)  # fmt: skip
_SIGMA = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    (14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3),
    (11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4),
    (7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8),
    (9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13),
    (2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9),
    (12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11),
    (13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10),
    (6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5),
    (10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0),
)  # fmt: skip
_M64 = (1 << 64) - 1


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (64 - n))) & _M64


def _compress(h: list[int], block: bytes, t: int, last: bool) -> None:
    m = [int.from_bytes(block[8 * i : 8 * i + 8], "little") for i in range(16)]
    v = h[:] + list(_IV)
    v[12] ^= t & _M64
    if last:
        v[14] ^= _M64

    def g(a: int, b: int, c: int, d: int, x: int, y: int) -> None:
        v[a] = (v[a] + v[b] + x) & _M64
        v[d] = _rotr(v[d] ^ v[a], 32)
        v[c] = (v[c] + v[d]) & _M64
        v[b] = _rotr(v[b] ^ v[c], 24)
        v[a] = (v[a] + v[b] + y) & _M64
        v[d] = _rotr(v[d] ^ v[a], 16)
        v[c] = (v[c] + v[d]) & _M64
        v[b] = _rotr(v[b] ^ v[c], 63)

    for r in range(12):
        s = _SIGMA[r % 10]
        g(0, 4, 8, 12, m[s[0]], m[s[1]])
        g(1, 5, 9, 13, m[s[2]], m[s[3]])
        g(2, 6, 10, 14, m[s[4]], m[s[5]])
        g(3, 7, 11, 15, m[s[6]], m[s[7]])
        g(0, 5, 10, 15, m[s[8]], m[s[9]])
        g(1, 6, 11, 12, m[s[10]], m[s[11]])
        g(2, 7, 8, 13, m[s[12]], m[s[13]])
        g(3, 4, 9, 14, m[s[14]], m[s[15]])
    for i in range(8):
        h[i] ^= v[i] ^ v[i + 8]


def _blake2b_params(data: bytes, digest: int, fanout: int, depth: int, leaf: int,
                    node_offset: int, node_depth: int, inner: int) -> bytes:  # fmt: skip
    """BLAKE2b with a full parameter block (no key, salt or personalization); written out because
    hashlib cannot set depth 0 or the 32-bit XOF length that BLAKE2X needs."""
    pb = bytearray(64)
    pb[0], pb[1], pb[2], pb[3] = digest, 0, fanout, depth
    pb[4:8] = leaf.to_bytes(4, "little")
    pb[8:16] = node_offset.to_bytes(8, "little")
    pb[16], pb[17] = node_depth, inner
    h = [_IV[i] ^ int.from_bytes(pb[8 * i : 8 * i + 8], "little") for i in range(8)]
    pos = 0
    while len(data) - pos > 128:
        pos += 128
        _compress(h, data[pos - 128 : pos], pos, False)
    tail = data[pos:]
    _compress(h, tail.ljust(128, b"\0"), len(data), True)
    return b"".join(x.to_bytes(8, "little") for x in h)[:digest]


def blake2xb(data: bytes, outlen: int) -> bytes:
    """BLAKE2xb per the BLAKE2X specification: root hash H0 with xof_length = outlen (high 32 bits
    of the node offset field), then output block i = BLAKE2b(H0) with fanout 0, depth 0, leaf
    length 64, node offset i, inner length 64."""
    xof = outlen << 32
    h0 = _blake2b_params(data, 64, 1, 1, 0, xof, 0, 0)
    return b"".join(
        _blake2b_params(h0, min(64, outlen - 64 * i), 0, 0, 64, i | xof, 0, 64)
        for i in range((outlen + 63) // 64)
    )


def test_python_blake2b_matches_hashlib() -> None:
    """The hand-written compression is checked against hashlib on plain BLAKE2b."""
    for n in (0, 1, 127, 128, 129, 300):
        data = bytes(range(256))[:n] if n <= 256 else bytes(range(256)) + bytes(range(n - 256))
        assert _blake2b_params(data, 64, 1, 1, 0, 0, 0, 0) == hashlib.blake2b(data).digest()


def py_expand(seed: bytes, tower: int, q: int, n: int) -> list[int]:
    """The specification of decision 0023, written independently of the C++ code."""
    mask = (1 << q.bit_length()) - 1
    out: list[int] = []
    c = 0
    while len(out) < n:
        block = blake2xb(
            b"ombra-seeded-v1" + seed + tower.to_bytes(4, "little") + c.to_bytes(8, "little"), 4096
        )
        for o in range(0, 4096, 8):
            w = int.from_bytes(block[o : o + 8], "little") & mask
            if w < q:
                out.append(w)
        c += 1
    return out[:n]


def _cpp_expand(seed: bytes, tower: int, q: int, n: int) -> list[int]:
    done = _run(["expand", seed.hex(), str(tower), str(q), str(n)])
    assert done.returncode == 0, done.stderr
    return [int(x) for x in done.stdout.split()]


def _preset_args(name: str) -> list[str]:
    p = ombra_params.get_preset(name)
    ombra_params.validate_preset(p)
    assert p["scaling_technique"] == "FIXEDMANUAL" and p["key_switch_technique"] == "HYBRID"
    return [
        str(p["ring_dim"]),
        str(p["multiplicative_depth"]),
        str(p["scaling_mod_size"]),
        str(p["first_mod_size"]),
        str(p["batch_size"]),
        str(p.get("num_large_digits", 0)),
        TOL[name],
    ]


def _parse_kat(stdout: str) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for line in stdout.splitlines():
        if not line.startswith("kat "):
            continue
        head, tail = line.split(":", 1)
        f = dict(x.split("=") for x in head.split()[1:])
        first, last = tail.split("...")
        out[f["tower"]] = {
            "q": [int(f["q"])],
            "n": [int(f["n"])],
            "first8": [int(x) for x in first.split()],
            "last2": [int(x) for x in last.split()],
        }
    return out


@pytest.fixture(scope="module", params=sorted(TOL))
def exact_run(request: pytest.FixtureRequest) -> tuple[str, subprocess.CompletedProcess[str]]:
    assert sorted(ombra_params.load_presets()) == sorted(TOL)
    name = request.param
    return name, _run(_preset_args(name))


def test_exact_phase_metadata_and_decrypt(exact_run) -> None:
    _, done = exact_run
    assert done.returncode == 0, done.stdout + done.stderr
    assert "FAIL" not in done.stdout
    # 4 rounds x 9 checks + 1 payload size + 2 refusals, plus the seed-reuse checks
    assert done.stdout.count("ok  ") >= 40
    for what in ("phase", "metadata", "u != c1", "seeded_words", "Decrypt(expanded) == plaintext"):
        assert what in done.stdout


def test_generator_known_answer(exact_run) -> None:
    name, done = exact_run
    got = _parse_kat(done.stdout)
    assert set(got) == {"0", "1"}
    seed = bytes((i * 7 + 1) % 256 for i in range(32))
    for t, rec in got.items():
        q, n = rec["q"][0], rec["n"][0]
        words = py_expand(seed, int(t), q, n)  # independent implementation
        assert rec["first8"] == words[:8] and rec["last2"] == words[-2:]
        assert all(w < q for w in words)
    pinned = json.loads(KAT.read_text())
    assert pinned[name] == got, "generator output changed: the fixture pins it byte for byte"


@pytest.mark.parametrize("n,q_bits", [(100, 40), (1000, 40), (5000, 59), (9000, 60)])
def test_generator_properties(n: int, q_bits: int) -> None:
    q = (1 << q_bits) - 1 - 2 * 37  # odd, below 2^q_bits; primality is not needed by the spec
    seed = bytes(range(32))
    a = _cpp_expand(seed, 0, q, n)
    assert len(a) == n and all(w < q for w in a)
    assert a == py_expand(seed, 0, q, n)  # block boundary: 4096 B = 512 words < n for n > 512
    assert _cpp_expand(bytes(range(1, 33)), 0, q, n) != a  # another seed
    assert _cpp_expand(seed, 1, q, n) != a  # another tower


def test_generator_refuses_bad_modulus() -> None:
    for q in (0, 1, 100, 1 << 60, (1 << 60) + 1):
        assert _run(["expand", "00" * 32, "0", str(q), "8"]).returncode != 0
