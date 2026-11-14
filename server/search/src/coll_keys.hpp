// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Private to server/search. The OpenFHE half of a loaded collection and the checks that make it,
// shared by the GPU engine (session.cpp) and the CPU-only engine (session_cpu.cpp), so that a key
// upload runs the same checks in both builds. Compiles without CUDA.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ombra/search/layout.hpp"
#include "ombra/search/session.hpp"
#include "ombra/storage/store.hpp"
#include "openfhe.h"

namespace ombra::search::detail {

struct CollKeys {
  storage::CollectionSpec spec;
  QueryLayout layout;
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
  std::size_t ring = 0, batch = 0;
  double column_scale = 0;  // scaling factor of a fresh level-0 ciphertext
  lbcrypto::EvalKey<lbcrypto::DCRTPoly> relin;
  // Same type as core::GpuExpansion::KeyMap.
  std::map<std::uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> akeys;
};

// Reads the context and the EvalMult and automorphism key files held in memory and checks them:
// envelope kind and preset, security floor, the context against the collection's preset (when
// cfg.preset_check is set), key tags, and the layout limits. Fills `c`. Throws what
// SearchEngine::prepare_keys documents. Touches no engine state; OpenFHE's process-wide key stores
// are left as they were except for the collection's own tag.
void prepare_collection_keys(CollKeys& c, storage::Store& store, const EngineConfig& cfg,
                             const std::string& name, const std::vector<std::uint8_t>& ctx_b,
                             const std::vector<std::uint8_t>& mult_b,
                             const std::vector<std::uint8_t>& auto_b);

}  // namespace ombra::search::detail
