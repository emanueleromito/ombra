# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The server's default receive limit must fit one block insert of a supported collection.

An Insert carries every stored column of one block in one message. A stored column of the
Balanced preset is 393,336 bytes when seeded and 787,423 bytes when not (docs/site/performance/
storage.md). The client seeds when its compiled extension is present, as in the wheels.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDED_COLUMN = 393_336
UNSEEDED_COLUMN = 787_423
HEADROOM = 1.05  # ids, filter columns and framing


def _default_mb(name: str) -> int:
    src = (ROOT / "server/grpc/src/ombra_server.cpp").read_text(encoding="utf-8")
    m = re.search(rf"\b{name} = (\d+)", src)
    assert m, f"no default for {name} in ombra_server.cpp"
    return int(m.group(1))


def test_default_receive_limit_fits_one_block_insert():
    limit = _default_mb("max_message_mb") << 20
    assert limit >= 1536 * SEEDED_COLUMN * HEADROOM, "seeded Balanced block at d=1536"
    assert limit >= 768 * UNSEEDED_COLUMN * HEADROOM, "unseeded Balanced block at d=768"


def test_memory_quota_holds_two_largest_messages():
    assert _default_mb("memory_quota_mb") >= 2 * _default_mb("max_message_mb")
