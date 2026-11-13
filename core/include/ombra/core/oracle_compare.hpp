// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// The one exact ciphertext comparator the oracle drivers share.
//
// Compares everything a GPU result must carry to equal OpenFHE's: level, noise-scale degree,
// scaling factor, key tag, encoding type, slots, number of components, format, moduli and every
// RNS word. Key tag and encoding type are included so an oracle notices a GPU path that labels
// its result with the wrong key. Header-only, so drivers do not each keep a private copy.
#pragma once

#include <cstddef>

#include "openfhe.h"

namespace ombra::core {

inline bool identical_ciphertexts(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& x,
                                  const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& y) {
  using namespace lbcrypto;
  if (x->GetLevel() != y->GetLevel() || x->GetNoiseScaleDeg() != y->GetNoiseScaleDeg() ||
      x->GetScalingFactor() != y->GetScalingFactor() || x->GetKeyTag() != y->GetKeyTag() ||
      x->GetEncodingType() != y->GetEncodingType() || x->GetSlots() != y->GetSlots() ||
      x->GetElements().size() != y->GetElements().size()) {
    return false;
  }
  for (std::size_t c = 0; c < x->GetElements().size(); ++c) {
    const DCRTPoly& px = x->GetElements()[c];
    const DCRTPoly& py = y->GetElements()[c];
    if (px.GetFormat() != py.GetFormat() || px.GetNumOfElements() != py.GetNumOfElements()) {
      return false;
    }
    for (std::size_t t = 0; t < px.GetNumOfElements(); ++t) {
      const auto& tx = px.GetElementAtIndex(t);
      const auto& ty = py.GetElementAtIndex(t);
      if (tx.GetModulus() != ty.GetModulus() || !(tx.GetValues() == ty.GetValues())) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace ombra::core
