# Fuzzing

libFuzzer targets for the parsers of untrusted bytes that build on the CPU (no CUDA, no gRPC). They are built with clang and `-fsanitize=fuzzer,address,undefined`, only when CMake gets `-DOMBRA_FUZZ=ON`.

| Target | Parser under test |
| --- | --- |
| `envelope` | file envelope reader, `core/include/ombra/core/file_envelope.hpp` |
| `seeded` | seeded ciphertext payload, `core/include/ombra/core/seeded_ciphertext.hpp` |
| `segment` | segment file, column envelopes and the insert-time column check, `server/storage/src/segment.cpp` |
| `blob` | blob file, `server/storage/src/blob.cpp` |
| `wal` | write-ahead log records and torn-tail repair, `server/storage/src/wal.cpp` |
| `recovery` | crash recovery replay of a fuzzed `wal.log` over a small data dir, `Store::recover` |
| `api_keys` | API key file, `ApiKeyStore::load` and `reload` |
| `http` | REST request parser and the `Authorization` check on its output |
| `auth_header` | `authorize()` on raw header values, and the flag parsers of `limits.cpp` |

Not covered here: gRPC messages and the OpenFHE context and key deserialization. The SQLite catalog is SQLite's own file format and is not fuzzed.

## Run

```
make fuzz FUZZ_SECONDS=60
```

This builds `build-fuzz` and runs every target for `FUZZ_SECONDS` on a copy of its corpus. It fails if a target crashes, leaks, times out or trips a sanitizer. The input that did it is left in `build-fuzz/fuzz-artifacts/`. To run one target for longer:

```
make fuzz FUZZ_SECONDS=600  # all targets
fuzz/run_fuzzers.sh build-fuzz 600 wal
```

Reproduce a finding with `build-fuzz/fuzz/ombra_fuzz_<target> <artifact file>`. Minimise it with `-minimize_crash=1`. Every finding gets a regression test in the normal C++ test suite, so `make test` catches it without fuzzing.

## How the targets are built

- The first byte of the `segment`, `blob`, `wal` and `recovery` inputs is a mode byte. With bit 0 set the harness repairs magic, version and checksums before parsing, so the fuzzer can reach the code behind the checksum. The `blob` target, bit 1, wraps the input in a valid blob of the smallest bucket size.
- A target that writes files uses one private directory under the temp dir and removes it at exit and on a crash. Set `TMPDIR` to choose where. `make fuzz` uses `/dev/shm` when it exists.
- A parser that accepts an input must produce the same bytes when it writes the parsed value back (envelope, seeded payload, segment, blob). The harness aborts if not.

## Corpus and dictionaries

`fuzz/corpus/<target>/` holds a few small valid inputs per target, built from the project's own writers with synthetic data only. `make fuzz-corpus` regenerates them. `fuzz/dict/<target>.dict` lists magic strings and format versions.
