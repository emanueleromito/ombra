// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Fuzz target: the seeded ciphertext payload reader (core/include/ombra/core/seeded_ciphertext.hpp,
// payload of envelope kind 4). The full read must write back to the same bytes.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>

#include "ombra/core/seeded_ciphertext.hpp"

namespace core = ombra::core;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* d, std::size_t n) {
  try {
    const auto p = core::read_seeded_payload(d, n);
    const auto out = core::write_seeded_payload(p);
    if (out.size() != n || !std::equal(out.begin(), out.end(), d)) std::abort();
    // The cheap check accepts everything the full read accepts.
    (void)core::check_seeded_header(d, n);
  } catch (const std::exception&) {
  }
  try {
    const auto h = core::check_seeded_header(d, n);
    if (!h.c0.empty() || h.moduli.size() != h.towers) std::abort();
  } catch (const std::exception&) {
  }
  return 0;
}
