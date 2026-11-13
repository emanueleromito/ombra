# Python API

Every public name of the `ombra` package, generated from the docstrings in `client/ombra/`. Install with `pip install ombra`. The package imports numpy, so the build of this page reads the source without importing it.

The package exports these names: `Client`, `AsyncClient`, `CollectionInfo`, `SearchHit`, `API_VERSION` and `__version__`. `API_VERSION` is the gRPC package the client speaks, `ombra.v1`. The sections after them cover the modules you may touch directly: `ombra.collection`, `ombra.presets` and `ombra.testing`. Everything else under `ombra.` (names starting with an underscore, `search`, `topk`, `clustering` and so on) is internal and can change between versions.

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
