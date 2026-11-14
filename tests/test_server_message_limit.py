# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The server's default receive limit must fit every message a client sends.

Insert and ReplaceBlock send one column per message, PutBlob and UploadKeys chunks of at most
LIMIT_MAX_CHUNK_BYTES, so the largest message is a chunk (4 MiB) or the ids of one block. The limit
is 16 MiB by default. A stored column of the Balanced preset is 393,336 bytes when seeded and
787,423 bytes when not (docs/site/performance/storage.md).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNSEEDED_COLUMN = 787_423
CHUNK = 4_194_304
MAX_ID = 256
MAX_BATCH = 8192  # slots of the largest preset (core/params/ckks_presets.json)
FRAMING = 64 * 1024  # field tags, collection name, api_version: generous


def _default_mb(name: str) -> int:
    src = (ROOT / "server/grpc/src/ombra_server.cpp").read_text(encoding="utf-8")
    m = re.search(rf"\b{name} = (\d+)", src)
    assert m, f"no default for {name} in ombra_server.cpp"
    return int(m.group(1))


def test_default_receive_limit_is_16_mib():
    assert _default_mb("max_message_mb") == 16


def test_every_message_kind_fits():
    limit = _default_mb("max_message_mb") << 20
    assert limit >= 4 * UNSEEDED_COLUMN, "one unseeded column, with large margin"
    assert limit >= CHUNK + FRAMING, "a key or blob chunk"
    assert limit >= MAX_BATCH * (MAX_ID + 4) + FRAMING, (
        "the ids of one block, all at maximum length"
    )
    assert limit >= 100_000 * (128 + 3), "a delete of the maximum id count with 128 byte ids"


def test_memory_quota_holds_many_messages():
    assert _default_mb("memory_quota_mb") >= 32 * _default_mb("max_message_mb")


def test_client_send_limit_equals_server_receive_limit():
    src = (ROOT / "client/ombra/_transport.py").read_text(encoding="utf-8")
    m = re.search(r"MAX_SEND_MESSAGE = (\d+) << 20", src)
    assert m and int(m.group(1)) == _default_mb("max_message_mb")
