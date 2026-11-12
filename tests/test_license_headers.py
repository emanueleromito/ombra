# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The dual license (PolyForm Noncommercial or commercial) is declared everywhere."""

import importlib.util
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPRESSION = "PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial"

_spec = importlib.util.spec_from_file_location("add_spdx", ROOT / "ci" / "add_spdx.py")
add_spdx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(add_spdx)


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],  # noqa: S607
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return out.splitlines()


def test_source_files_have_spdx_line():
    files = add_spdx.tracked_source_files()
    assert len(files) > 100, "file discovery looks broken"
    missing = [
        path
        for path, _ in files
        if f"SPDX-License-Identifier: {EXPRESSION}"
        not in "\n".join((ROOT / path).read_text(encoding="utf-8").splitlines()[:5])
    ]
    assert not missing, f"missing SPDX line (run python ci/add_spdx.py): {missing}"


def test_pyprojects_declare_the_dual_license():
    checked = 0
    for path in _tracked():
        if not path.endswith("pyproject.toml") or path.startswith("packaging/ombra-openfhe/"):
            continue
        data = tomllib.loads((ROOT / path).read_text(encoding="utf-8"))
        project = data.get("project")
        if project is None:  # tool settings only, nothing is published from it
            continue
        checked += 1
        assert project.get("license") == EXPRESSION, path
        assert not any(c.startswith("License ::") for c in project.get("classifiers", [])), path
        folder = (ROOT / path).parent
        for name in project.get("license-files", []):
            assert (folder / name).is_file(), f"{path}: license file {name} missing"
    assert checked >= 2


def test_packaged_license_files_match_the_root_ones():
    for path in _tracked():
        if not path.endswith("pyproject.toml") or path.startswith("packaging/ombra-openfhe/"):
            continue
        folder = (ROOT / path).parent
        if folder == ROOT:
            continue
        for name in ("LICENSE", "LICENSING.md"):
            copy = folder / name
            if copy.exists():
                assert copy.read_bytes() == (ROOT / name).read_bytes(), f"{copy} differs from root"


def test_images_carry_the_source_and_license_labels():
    dockerfiles = [ROOT / "ci" / "Dockerfile", *sorted((ROOT / "docker").glob("Dockerfile*"))]
    assert len(dockerfiles) >= 3, "ci image and the two release images are expected"
    for path in dockerfiles:
        text = path.read_text(encoding="utf-8")
        assert f'LABEL org.opencontainers.image.licenses="{EXPRESSION}"' in text, path
        assert (
            'LABEL org.opencontainers.image.source="https://github.com/erailabs/ombra"' in text
        ), path
