# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Ombra client package."""

from ombra._core import CollectionInfo, SearchHit
from ombra.client import AsyncClient, Client
from ombra.errors import OmbraError
from ombra.retry import RetryPolicy

__version__ = "0.1.0"
API_VERSION = "ombra.v1"  # matches the proto package in proto/ombra/v1

__all__ = [
    "API_VERSION",
    "AsyncClient",
    "Client",
    "CollectionInfo",
    "OmbraError",
    "RetryPolicy",
    "SearchHit",
    "__version__",
]
