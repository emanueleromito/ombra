#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# Runs every fuzz target for SECONDS each. Usage: fuzz/run_fuzzers.sh BUILD_DIR SECONDS [TARGET...]
# Each target starts from a copy of its committed corpus (fuzz/corpus/<target>), so the committed
# files are never modified; finds are kept in BUILD_DIR/fuzz-work. A crash, leak, timeout or sanitizer
# report fails the script and leaves the input in BUILD_DIR/fuzz-artifacts (see fuzz/README.md).
set -u

build="${1:?usage: run_fuzzers.sh BUILD_DIR SECONDS [TARGET...]}"
secs="${2:?usage: run_fuzzers.sh BUILD_DIR SECONDS [TARGET...]}"
shift 2
root="$(cd "$(dirname "$0")/.." && pwd)"
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  targets=(envelope seeded segment blob wal recovery api_keys http auth_header)
fi

# The wal and recovery targets fsync; a memory-backed temp dir keeps them fast.
if [ -z "${TMPDIR:-}" ] && [ -d /dev/shm ] && [ -w /dev/shm ]; then export TMPDIR=/dev/shm; fi
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:abort_on_error=0:handle_abort=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1}"

work="$build/fuzz-work"
art="$build/fuzz-artifacts"
mkdir -p "$work" "$art"
failed=0
for t in "${targets[@]}"; do
  bin="$build/fuzz/ombra_fuzz_$t"
  [ -x "$bin" ] || { echo "missing $bin (build the ombra_fuzzers target first)"; exit 2; }
  rm -rf "$work/$t"
  mkdir -p "$work/$t"
  cp -r "$root/fuzz/corpus/$t/." "$work/$t/" 2>/dev/null
  args=(-max_total_time="$secs" -timeout=20 -rss_limit_mb=2048 -max_len=65536
        -artifact_prefix="$art/$t-" -print_final_stats=1)
  [ -f "$root/fuzz/dict/$t.dict" ] && args+=(-dict="$root/fuzz/dict/$t.dict")
  echo "== fuzz $t for ${secs}s"
  if ! "$bin" "${args[@]}" "$work/$t" 2>"$work/$t.log"; then
    echo "FAILED: $t (log: $work/$t.log, input in $art)"
    tail -n 40 "$work/$t.log"
    failed=1
  else
    grep -E "^(stat::number_of_executed_units|stat::average_exec_per_sec|stat::new_units_added)" "$work/$t.log" | tr '\n' ' '
    echo
  fi
done
exit $failed
