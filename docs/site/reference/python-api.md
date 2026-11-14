# Python API

Every public name of the `ombra` package, generated from the docstrings in `client/ombra/`. Install with `pip install ombra`. The package imports numpy, so the build of this page reads the source without importing it.

The package exports these names: `Client`, `AsyncClient`, `CollectionInfo`, `SearchHit`, `OmbraError`, `RetryPolicy`, `API_VERSION` and `__version__`. `API_VERSION` is the gRPC package the client speaks, `ombra.v1`. The sections after them cover the modules you may touch directly: `ombra.errors`, `ombra.retry`, `ombra.collection`, `ombra.presets` and `ombra.testing`. Everything else under `ombra.` (names starting with an underscore, `search`, `topk`, `clustering` and so on) is internal and can change between versions.

`AsyncClient` has the same methods as `Client`, as coroutines. The work still runs in the calling thread, so awaiting a call blocks the event loop until it returns.

For how to choose between `Client("host:port")` and `Client.local(path)`, see [Local mode or a server](../getting-started/local-or-server.md).

## Clients

::: ombra.client.Client
    options:
      show_if_no_docstring: true

::: ombra.client.AsyncClient
    options:
      show_if_no_docstring: true

## Results

::: ombra.CollectionInfo
    options:
      show_if_no_docstring: true

::: ombra.SearchHit
    options:
      show_if_no_docstring: true

## Errors, retries and deadlines

These apply to `Client("host:port")`. Local mode has no network and makes one attempt per call.

**Deadlines.** Every call to a server has a deadline, and the deadline bounds the whole call. For a stream, such as a search or an upload, it bounds the entire stream. The defaults are 30 s for small reads (`GetCollection`, `ListCollections`, `ListBlocks`), 120 s for other calls with one request and one answer, 300 s for one blob, and 1800 s for `Insert`, `UploadKeys`, `ReplaceBlock`, `GetBlock` and `Search`. Change them in the constructor: `timeout` replaces the 120 s value, `stream_timeout` replaces the 1800 s value, and `deadlines={"Search": 600}` sets single calls by their gRPC name.

**Errors.** Every failed server call raises a subclass of `ombra.OmbraError`. It has three attributes:

- `retryable`: `True` if repeating the call may work.
- `grpc_status`: the gRPC status name, such as `"UNAVAILABLE"`, or `None` when the server answered with an Ombra error code.
- `error_code`: the `ombra.v1` `ErrorCode` number, or `None` for a transport failure.

`ServerUnavailableError` (UNAVAILABLE), `DeadlineExceededError` and `ResourceExhaustedError` are retryable. Everything else is not, for example `NotFoundError`, `InvalidArgumentError`, `PermissionDeniedError`, `UnauthenticatedError`, `KeyTagMismatchError` and `PresetMismatchError`. The full list is in `ombra.errors`. `CollectionError` and `BlobStoreError` still exist; the errors that name an Ombra error code are also instances of them, so existing `except` clauses keep working. Messages hold the call name and the codes, never ids, vectors, scores or ciphertext bytes.

**Retries.** The client retries a failed call when its error is retryable and a repeat is safe: `GetCollection`, `ListCollections`, `ListBlocks`, `GetBlock`, `GetBlob`, `Search`, `Delete`, `PutBlob`, `DeleteBlobs` and `UploadKeys`. `CreateCollection` is repeated too; if the repeat finds the collection already there, it counts as success only when the stored settings are identical. `Insert`, `ReplaceBlock` and `DropCollection` are never retried, so a lost answer shows up as an error you handle yourself. Before each retry the client waits a random time between zero and `min(max_delay, base_delay * 2**(attempt - 1))`.

A search whose stream breaks opens a new stream and sends the same query and the same cluster choices again, in the same order. It never draws new decoys, because two decoy sets for one query would show the server which clusters are real.

Set the policy with `Client(..., retry=RetryPolicy(...))`:

```python
from ombra import Client, RetryPolicy

client = Client("db.example.com:443", retry=RetryPolicy(max_attempts=6, total_budget=60))
client = Client("db.example.com:443", retry=RetryPolicy.never())  # no retries
```

The defaults are 4 attempts, a 0.25 s base delay, an 8 s cap and a 120 s total budget. No new attempt starts if it would pass the budget. Retries are logged at debug level on the `ombra.retry` logger with the call name, attempt number and status code only.

::: ombra.errors
    options:
      show_if_no_docstring: true

::: ombra.retry
    options:
      show_if_no_docstring: true

## Collections and key files

::: ombra.collection
    options:
      show_if_no_docstring: true

## Presets

The preset values themselves are listed on the [Presets](presets.md) page.

::: ombra.presets
    options:
      show_if_no_docstring: true

## Testing

`FakeClient` is an in-memory stand-in for unit tests of your own code. It does not encrypt anything, so never use it to try Ombra itself.

::: ombra.testing
    options:
      show_if_no_docstring: true
