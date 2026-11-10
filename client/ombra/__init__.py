"""Ombra client package."""

from ombra.client import AsyncClient, Client

__version__ = "0.1.0"
API_VERSION = "ombra.v1"  # matches the proto package in proto/ombra/v1

__all__ = ["API_VERSION", "AsyncClient", "Client", "__version__"]
