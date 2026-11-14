# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Exceptions of the Ombra client, with a flag that says whether a repeat can help.

Every error that comes from talking to a server derives from `OmbraError`:

* `retryable`: True when repeating the same call may succeed (server not reachable, deadline
  passed, server overloaded). The client already retries the calls that are safe to repeat; see
  `ombra.retry.RetryPolicy`.
* `grpc_status`: the name of the gRPC status code ("UNAVAILABLE", ...), or None when the server
  answered with an ombra.v1 error inside a normal response.
* `error_code`: the ombra.v1 `ErrorCode` number, or None when the failure was at transport level.

Messages never contain ids, vectors, scores or ciphertext bytes: only the call name and codes.
This module needs neither grpc nor the generated stubs.
"""

from __future__ import annotations


class OmbraError(Exception):
    """Base class of every error raised for a failed call to an Ombra server."""

    retryable: bool = False
    grpc_status: str | None = None
    error_code: int | None = None

    def __init__(
        self,
        message: str = "",
        *,
        retryable: bool | None = None,
        grpc_status: str | None = None,
        error_code: int | None = None,
    ) -> None:
        super().__init__(message)
        if retryable is not None:
            self.retryable = retryable
        if grpc_status is not None:
            self.grpc_status = grpc_status
        if error_code is not None:
            self.error_code = error_code


# ---- transport level (gRPC status codes) -------------------------------------------------------
class ServerUnavailableError(OmbraError):
    """UNAVAILABLE: the server could not be reached or dropped the connection."""

    retryable = True


class DeadlineExceededError(OmbraError):
    """DEADLINE_EXCEEDED: the call did not finish within its deadline."""

    retryable = True


class ResourceExhaustedError(OmbraError):
    """RESOURCE_EXHAUSTED: the server is overloaded or a quota is used up."""

    retryable = True


class NotFoundError(OmbraError):
    """NOT_FOUND, or an ombra.v1 error about something that does not exist."""


class InvalidArgumentError(OmbraError):
    """INVALID_ARGUMENT, or ERROR_INVALID_ARGUMENT."""


class PermissionDeniedError(OmbraError):
    """PERMISSION_DENIED: the API key is not allowed to do this."""


class UnauthenticatedError(OmbraError):
    """UNAUTHENTICATED: the API key is missing or wrong."""


class AlreadyExistsError(OmbraError):
    """ALREADY_EXISTS, or ERROR_COLLECTION_EXISTS."""


class FailedPreconditionError(OmbraError):
    """FAILED_PRECONDITION: the server is not in a state that allows the call."""


class UnimplementedError(OmbraError):
    """UNIMPLEMENTED: the server does not know this call."""


class CancelledError(OmbraError):
    """CANCELLED: the call was cancelled."""


class ServerInternalError(OmbraError):
    """INTERNAL / UNKNOWN / ERROR_INTERNAL: a server-side failure. Not retried."""


# ---- ombra.v1 ErrorCode values ----------------------------------------------------------------
class UnsupportedVersionError(OmbraError):
    """ERROR_UNSUPPORTED_VERSION."""


class CollectionNotFoundError(NotFoundError):
    """ERROR_COLLECTION_NOT_FOUND."""


class CollectionExistsError(AlreadyExistsError):
    """ERROR_COLLECTION_EXISTS."""


class SizeLimitError(OmbraError):
    """ERROR_SIZE_LIMIT: a server limit was exceeded."""


class KeyTagMismatchError(OmbraError):
    """ERROR_KEY_TAG_MISMATCH: ciphertext or keys belong to another key than the collection's."""


class PresetMismatchError(OmbraError):
    """ERROR_PRESET_MISMATCH."""


class InvalidFilterLayoutError(InvalidArgumentError):
    """ERROR_INVALID_FILTER_LAYOUT."""


class KeysMissingError(FailedPreconditionError):
    """ERROR_KEYS_MISSING: search or insert before the evaluation keys were uploaded."""


class BadSequenceError(OmbraError):
    """ERROR_BAD_SEQUENCE: stream messages out of order."""


class BlockNotFoundError(NotFoundError):
    """ERROR_BLOCK_NOT_FOUND."""


class BlockMismatchError(OmbraError):
    """ERROR_BLOCK_MISMATCH."""


class BlobNotFoundError(NotFoundError):
    """ERROR_BLOB_NOT_FOUND."""


class StorageFullError(ResourceExhaustedError):
    """ERROR_STORAGE_FULL: the server's disk or quota is full. Nothing was stored; a repeat may
    work once space is freed, so it is retryable (the server also sends RESOURCE_EXHAUSTED)."""


# ---- classes kept for existing callers --------------------------------------------------------
class CollectionError(OmbraError):
    """The server answered with an ombra.v1.Error (or a collection check failed). `code` is the
    ombra.v1 ErrorCode (0 when the client itself found the problem)."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"error code {code}: {message}", error_code=code or None)
        self.code = code
        self.message = message


class BlobStoreError(OmbraError, RuntimeError):
    """A blob store operation failed. The message carries no ids or contents."""


_STATUS_CLASSES: dict[str, type[OmbraError]] = {
    "UNAVAILABLE": ServerUnavailableError,
    "DEADLINE_EXCEEDED": DeadlineExceededError,
    "RESOURCE_EXHAUSTED": ResourceExhaustedError,
    "NOT_FOUND": NotFoundError,
    "INVALID_ARGUMENT": InvalidArgumentError,
    "PERMISSION_DENIED": PermissionDeniedError,
    "UNAUTHENTICATED": UnauthenticatedError,
    "ALREADY_EXISTS": AlreadyExistsError,
    "FAILED_PRECONDITION": FailedPreconditionError,
    "UNIMPLEMENTED": UnimplementedError,
    "CANCELLED": CancelledError,
    "INTERNAL": ServerInternalError,
    "UNKNOWN": ServerInternalError,
}

# ombra.v1 ErrorCode number -> class (see proto/ombra/v1/ombra.proto)
_PROTO_CLASSES: dict[int, type[OmbraError]] = {
    1: UnsupportedVersionError,
    2: CollectionNotFoundError,
    3: CollectionExistsError,
    4: InvalidArgumentError,
    5: SizeLimitError,
    6: KeyTagMismatchError,
    7: PresetMismatchError,
    8: InvalidFilterLayoutError,
    9: KeysMissingError,
    10: BadSequenceError,
    11: ServerInternalError,
    12: BlockNotFoundError,
    13: BlockMismatchError,
    14: BlobNotFoundError,
    15: StorageFullError,
}

_combos: dict[tuple[type, type], type[OmbraError]] = {}


def _combo(kind: type[OmbraError], base: type[OmbraError]) -> type[OmbraError]:
    """A class that is both `kind` and `base`, so `except base` and `except kind` both match."""
    if issubclass(kind, base):
        return kind
    key = (kind, base)
    if key not in _combos:
        _combos[key] = type(kind.__name__, (kind, base), {"__module__": __name__})
    return _combos[key]


def from_proto_error(
    code: int, message: str, base: type[OmbraError] = CollectionError, text: str | None = None
) -> OmbraError:
    """The exception for an ombra.v1.Error answer. It is an instance of `base` (what callers
    caught before this module existed) and of the class that names the ErrorCode."""
    kind = _PROTO_CLASSES.get(code)
    cls = _combo(kind, base) if kind is not None else base
    err = cls.__new__(cls)
    OmbraError.__init__(
        err,
        text if text is not None else f"error code {code}: {message}",
        error_code=code or None,
    )
    err.code = code  # type: ignore[attr-defined]
    err.message = message  # type: ignore[attr-defined]
    return err


def collection_error(code: int, message: str) -> CollectionError:
    """A CollectionError for an ombra.v1.Error that is also the ErrorCode's class."""
    err = from_proto_error(code, message, CollectionError)
    assert isinstance(err, CollectionError)
    return err


def status_name(exc: BaseException) -> str | None:
    """The gRPC status name of a grpc.RpcError (found by its `code()` method), else None."""
    code = getattr(exc, "code", None)
    if not callable(code):
        return None
    try:
        name = getattr(code(), "name", None)
    except Exception:
        return None
    return name if isinstance(name, str) and name.isupper() else None


def from_rpc_error(exc: BaseException, rpc: str) -> OmbraError | None:
    """Map a grpc.RpcError to an OmbraError, or None if `exc` is not one. Only the call name and
    the status code go into the message, never the server's detail text."""
    status = status_name(exc)
    if status is None:
        return None
    cls = _STATUS_CLASSES.get(status, OmbraError)
    return cls(f"{rpc} failed with status {status}", grpc_status=status)
