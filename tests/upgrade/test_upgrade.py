# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Upgrade safety: client and server one version apart, and old data dirs, on the CPU build.

The previous release is the highest `v*` tag below HEAD (`OMBRA_PREV_REF` names another ref, see
prev_release.py). Without a tag the cross-version tests skip. With an untagged override ref, a
failure in the old/new pairing is reported as an expected incompatibility (xfail, with the
error); with a real release tag, any failure is a test failure.

  a. old client -> new server   b. new client -> old server
  c. a data dir written by the old server opens with the new server
  d. every frozen fixture tests/fixtures/datadir-<version>/, and one written by the current code

Select them with `pytest -m upgrade`. Needs `make build-cpu`, openfhe-python and git.
Search cannot run on the CPU build: the test checks that the call fails with the no-GPU error.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")
pytest.importorskip("openfhe")

import common  # noqa: E402
import prev_release  # noqa: E402
from common import REPO_ROOT, DriverFailure, run_driver  # noqa: E402
from server_harness import ServerProcess, find_server_binary  # noqa: E402

pytestmark = [pytest.mark.openfhe, pytest.mark.upgrade]

NEW_BIN = find_server_binary()
if NEW_BIN is None or not common.VERIFY.is_file():
    pytest.skip(
        "ombra_server or ombra_verify (CPU build) not found: run `make build-cpu`",
        allow_module_level=True,
    )
NEW_CLIENT = REPO_ROOT / "client"


# ---- the previous release ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def prev() -> prev_release.PrevRelease:
    try:
        p = prev_release.resolve()
    except prev_release.NoPreviousRelease as e:
        pytest.skip(str(e))
    try:
        return prev_release.build(p)
    except prev_release.BuildFailed as e:
        if p.tagged:
            raise
        pytest.skip(f"{p.ref} ({p.sha[:10]}) has no buildable CPU server: {e}")


def _incompatible(prev: prev_release.PrevRelease, what: str, err: DriverFailure) -> None:
    """A real release must be compatible. An untagged ref may differ on purpose (xfail)."""
    msg = f"{what} with {prev.ref} ({prev.sha[:10]}): {err}"
    if prev.tagged:
        pytest.fail("incompatible " + msg, pytrace=False)
    pytest.xfail("expected incompatibility, ref has no release tag: " + msg)


# ---- a. old client -> new server --------------------------------------------------------------
def test_old_client_new_server(prev, tmp_path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    with ServerProcess(tmp_path, binary=NEW_BIN) as server:
        try:
            run_driver(prev.client_path, "write", server.address, work)
            # a second session: reopen with keys from the key file and read everything back
            run_driver(prev.client_path, "read", server.address, work)
        except DriverFailure as e:
            _incompatible(prev, "old client, new server", e)
    assert common.verify(server.data_dir).returncode == 0


# ---- b and c. old server: new client, then the data dir with the new server -------------------
@dataclass
class OldWorld:
    root: Path
    work: Path
    error: DriverFailure | None


@pytest.fixture(scope="module")
def old_world(prev, tmp_path_factory) -> OldWorld:
    """The previous server's data dir, written and read by the new client, then stopped."""
    root = tmp_path_factory.mktemp("old_world")
    work = root / "work"
    work.mkdir()
    error = None
    server = ServerProcess(root, binary=prev.server_bin)
    with server:
        try:
            run_driver(NEW_CLIENT, "write", server.address, work)
            run_driver(NEW_CLIENT, "read", server.address, work)  # (b) read back on the old server
        except DriverFailure as e:
            error = e
        server.stop()
    return OldWorld(root, work, error)


def test_new_client_old_server(prev, old_world) -> None:
    if old_world.error:
        _incompatible(prev, "new client, old server", old_world.error)


def test_old_data_dir_new_server(prev, old_world) -> None:
    if old_world.error:
        _incompatible(prev, "new client, old server (writing the data dir)", old_world.error)
    # No copy: the data dir is a few hundred MB, and the driver keeps state.json in step with it.
    assert (old_world.work / "state.json").is_file()
    _open_and_check(old_world.root, old_world.work)


def _open_and_check(root: Path, work: Path) -> None:
    """The new server opens `root/data`: verify clean, client reads and writes, stats sane."""
    server = ServerProcess(root, binary=NEW_BIN)
    before = common.verify(server.data_dir)
    assert before.returncode == 0, before.stdout + before.stderr
    offset = common.log_size(server.log_path)
    with server:
        stats = common.recovery_after(server.log_path, offset)
        assert stats is not None, "no recovery line in the server log"
        records, torn = stats
        assert torn == 0, f"torn bytes after a clean stop: {torn}"
        assert records < 1000, f"implausible number of replayed records: {records}"
        out = run_driver(NEW_CLIENT, "read", server.address, work)
        assert out["info"][0] >= 1
        server.stop()
    after = common.verify(server.data_dir)
    assert after.returncode == 0, after.stdout + after.stderr


# ---- d. data dirs written by older (and the current) code -------------------------------------
def _frozen() -> list[Path]:
    return sorted(p for p in common.FIXTURES.glob("datadir-*") if (p / "data").is_dir())


def _tree_hash() -> str:
    """HEAD plus any uncommitted change: the key of the cached fixture written by this code."""
    import hashlib

    def git(*a: str) -> bytes:
        return subprocess.run(  # noqa: S603, S607
            ["git", "-C", str(REPO_ROOT), *a], capture_output=True, check=True
        ).stdout

    h = hashlib.sha256(git("rev-parse", "HEAD"))
    h.update(
        git(
            "diff",
            "HEAD",
            "--",
            "client",
            "server",
            "core",
            "proto",
            "CMakeLists.txt",
            "tests/upgrade",
        )
    )
    return h.hexdigest()[:16]


@pytest.fixture(scope="session")
def generated_fixture() -> Path:
    """A data dir written by the current code, generated once and cached (it is too big for git:
    keys and ciphertext columns at d = 384 are tens of MB). Cached by commit and working tree."""
    import make_fixture

    cache = prev_release.cache_root() / f"fixture-{_tree_hash()}"
    if not (cache / "VERSION").is_file():
        tmp = cache.with_name(cache.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        make_fixture.make(tmp)
        tmp.rename(cache)
    return cache


@pytest.fixture(params=["generated"] + [p.name for p in _frozen()])
def fixture_dir(request, generated_fixture, tmp_path) -> Path:
    src = generated_fixture if request.param == "generated" else common.FIXTURES / request.param
    dst = tmp_path / "fx"
    shutil.copytree(src, dst)
    return dst


def test_new_server_opens_every_fixture(fixture_dir) -> None:
    assert (fixture_dir / "key.bin").is_file() and (fixture_dir / "data").is_dir()
    # the harness keeps its data dir at <root>/data: root = the fixture dir itself
    _open_and_check(fixture_dir, fixture_dir)


def test_cache_dir_override_is_honoured(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OMBRA_UPGRADE_CACHE", str(tmp_path))
    assert prev_release.cache_root() == tmp_path
    assert "OMBRA_UPGRADE_CACHE" in os.environ
