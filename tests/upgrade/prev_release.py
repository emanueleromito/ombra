# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Finds the previous release and builds its CPU server (the upgrade tests use both).

The previous release is the highest `v*` tag that is an ancestor of HEAD and not HEAD itself.
`OMBRA_PREV_REF` names another ref (a commit, branch or tag) for trying the tests out before a tag
exists. The ref's sources are exported with `git archive` into
`$OMBRA_UPGRADE_CACHE/<sha>/src` (default `~/.cache/ombra-upgrade`), built into `<sha>/build` with
`OMBRA_ENABLE_CUDA=OFF`, and kept there, so a second run (or a CI cache) does not build again. The
old client is that source tree's `client/` directory, put on PYTHONPATH of a subprocess.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class NoPreviousRelease(Exception):
    """There is no tag yet (the first release is v0.1.0)."""


class BuildFailed(Exception):
    pass


@dataclass(frozen=True)
class PrevRelease:
    ref: str
    sha: str
    tags: tuple[str, ...]  # the v* tags on this commit; empty for an untagged ref (override only)
    src: Path

    @property
    def tagged(self) -> bool:
        return bool(self.tags)

    @property
    def client_path(self) -> Path:
        return self.src / "client"

    @property
    def server_bin(self) -> Path:
        return self.src.parent / "build" / "server" / "grpc" / "ombra_server"


def _git(*args: str) -> str:
    out = subprocess.run(  # noqa: S603, S607 - fixed git commands
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.strip()


def cache_root() -> Path:
    return Path(os.environ.get("OMBRA_UPGRADE_CACHE") or Path.home() / ".cache" / "ombra-upgrade")


def _tags_at(sha: str) -> tuple[str, ...]:
    return tuple(t for t in _git("tag", "--points-at", sha).split() if t.startswith("v"))


def resolve() -> PrevRelease:
    """The ref to test against. Raises NoPreviousRelease when there is none."""
    head = _git("rev-parse", "HEAD")
    override = os.environ.get("OMBRA_PREV_REF")
    if override:
        sha = _git("rev-parse", "--verify", f"{override}^{{commit}}")
        ref = override
    else:
        sha, ref = "", ""
        for tag in _git("tag", "--list", "v*", "--sort=-v:refname").split():
            tag_sha = _git("rev-parse", f"{tag}^{{commit}}")
            if tag_sha == head:
                continue
            anc = subprocess.run(  # noqa: S603, S607
                ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", tag_sha, head],
                check=False,
            )
            if anc.returncode == 0:
                sha, ref = tag_sha, tag
                break
        if not sha:
            raise NoPreviousRelease(
                "no previous release tag (no v* tag is an ancestor of HEAD); the first release "
                "will be v0.1.0. Set OMBRA_PREV_REF to try the tests against another commit"
            )
    return PrevRelease(ref, sha, _tags_at(sha), cache_root() / sha / "src")


def build(prev: PrevRelease) -> PrevRelease:
    """Export and build the previous release's CPU server (cached by commit). Returns `prev`."""
    top = prev.src.parent
    if prev.server_bin.is_file() and (top / "build.ok").is_file():
        return prev
    top.mkdir(parents=True, exist_ok=True)
    if prev.src.exists():
        shutil.rmtree(prev.src)
    prev.src.mkdir()
    tar = top / "src.tar"
    subprocess.run(  # noqa: S603, S607
        ["git", "-C", str(REPO_ROOT), "archive", "--format=tar", "-o", str(tar), prev.sha],
        check=True,
    )
    with tarfile.open(tar) as t:
        t.extractall(prev.src, filter="data")
    tar.unlink()
    log = top / "build.log"
    extra = shlex.split(os.environ.get("CMAKE_ARGS", ""))
    build_dir = top / "build"
    jobs = str(os.cpu_count() or 2)
    steps = [
        [
            "cmake",
            "-S",
            str(prev.src),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DOMBRA_ENABLE_CUDA=OFF",
            *extra,
        ],
        ["cmake", "--build", str(build_dir), "-j", jobs, "--target", "ombra_server"],
    ]
    with open(log, "wb") as f:
        for step in steps:
            if subprocess.run(step, stdout=f, stderr=subprocess.STDOUT, check=False).returncode:  # noqa: S603
                tail = log.read_text(errors="replace").splitlines()[-15:]
                raise BuildFailed(
                    f"building the CPU server of {prev.ref} ({prev.sha[:10]}) failed "
                    f"(full log {log}):\n" + "\n".join(tail)
                )
    if not prev.server_bin.is_file():
        raise BuildFailed(f"{prev.ref} built, but {prev.server_bin} does not exist")
    (top / "build.ok").write_text(prev.sha + "\n")
    return prev
