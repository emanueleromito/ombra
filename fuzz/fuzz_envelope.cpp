// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Fuzz target: the file envelope reader (core/include/ombra/core/file_envelope.hpp). Runs every
// reader entry point with the preset and OpenFHE version the corpus carries. Whatever a reader
// accepts must write back to the same bytes (the format has one encoding).
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>

#include "fuzz_util.hpp"
#include "ombra/core/file_envelope.hpp"

namespace core = ombra::core;

static void check_roundtrip(const core::Envelope& e, const std::uint8_t* d, std::size_t n) {
  if (e.payload.size() > n) std::abort();
  const auto out = core::write_envelope(e);
  // Version 2 may carry kinds 1-3 on read, but the writer emits version 1 for them.
  const bool canonical = e.kind == core::EnvelopeKind::SeededCiphertext || n < 6 || d[4] == 1;
  if (canonical && (out.size() != n || !std::equal(out.begin(), out.end(), d))) std::abort();
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* d, std::size_t n) {
  const std::string preset = ombra::fuzz::kPreset, ofhe = ombra::fuzz::kOfhe;
  for (auto kind : {core::EnvelopeKind::Context, core::EnvelopeKind::Ciphertext,
                    core::EnvelopeKind::PublicEvalKey, core::EnvelopeKind::SeededCiphertext}) {
    try {
      check_roundtrip(core::read_envelope(d, n, kind, preset, ofhe), d, n);
    } catch (const std::exception&) {
    }
  }
  try {
    check_roundtrip(core::read_column_envelope(d, n, preset, ofhe), d, n);
  } catch (const std::exception&) {
  }
  return 0;
}
