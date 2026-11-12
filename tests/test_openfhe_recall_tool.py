"""T3.9: the recall tool end to end in local mode on a small synthetic corpus."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("openfhe")
pytest.importorskip("grpc_tools")
pytest.importorskip("faiss")
pytest.importorskip("ombra._native", reason="the compiled extension is not built")

from ombra import recall  # noqa: E402

pytestmark = pytest.mark.openfhe


def test_tool_prints_recalls_gap_and_verdict(tmp_path, capsys) -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2000, 384))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    np.save(tmp_path / "v.npy", x)
    code = recall.main([str(tmp_path / "v.npy"), "--n-queries", "20"])
    out = capsys.readouterr().out
    assert code == 0, out
    for needle in ("plaintext recall@10", "encrypted recall@10", "gap", "0012 rule"):
        assert needle in out
    assert "NOT ELIGIBLE" in out  # 20 queries are fewer than the rule's 1,000
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v.npy"]  # nothing written out
