// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// CPU reference operations for the expansion tree on OpenFHE (shared by ombra_expand and
// ombra_expand_gpu): EvalAutomorphism (base-leveledshe.cpp:382), EvalAdd / EvalSub and an exact
// multiplication by X^(-k) on the coefficients of every RNS tower. No keys other than the public
// automorphism keys.
#pragma once

#include <map>
#include <utility>
#include <vector>

#include "openfhe.h"

namespace ombra::core::cpu_ref {

using namespace lbcrypto;

using Ct = Ciphertext<DCRTPoly>;

// Multiply every element of the ciphertext by X^(-k) in Z[X]/(X^N+1), exactly: coefficient i moves
// to i - k, negated when it wraps around. No level is used and the scaling factor is unchanged.
inline Ct mul_by_neg_monomial(const Ct& ct, std::size_t k) {
  Ct out = ct->Clone();
  std::vector<DCRTPoly> elems = ct->GetElements();
  for (DCRTPoly& poly : elems) {
    const Format original = poly.GetFormat();
    poly.SetFormat(Format::COEFFICIENT);
    for (std::size_t t = 0; t < poly.GetNumOfElements(); ++t) {
      NativePoly tower = poly.GetElementAtIndex(t);
      const NativeInteger q = tower.GetModulus();
      const std::size_t n = tower.GetLength();
      const NativeVector& src = tower.GetValues();
      NativeVector dst(n, q);
      for (std::size_t i = 0; i < n; ++i) {
        if (i >= k) {
          dst[i - k] = src[i];
        } else {
          dst[i - k + n] = (src[i] == NativeInteger(0)) ? NativeInteger(0) : q - src[i];
        }
      }
      tower.SetValues(std::move(dst), Format::COEFFICIENT);
      poly.SetElementAtIndex(t, std::move(tower));
    }
    poly.SetFormat(original);
  }
  out->SetElements(std::move(elems));
  return out;
}

class OpenFheOps {
 public:
  OpenFheOps(CryptoContext<DCRTPoly> cc, const std::map<uint32_t, EvalKey<DCRTPoly>>& keys)
      : cc_(std::move(cc)), keys_(keys) {}

  std::pair<Ct, Ct> split(const Ct& c, std::size_t /*level*/, std::uint32_t galois,
                          std::size_t shift, bool want_odd) {
    Ct sub = cc_->EvalAutomorphism(c, galois, keys_);
    Ct even = cc_->EvalAdd(c, sub);
    if (!want_odd) {
      return {std::move(even), Ct()};
    }
    return {std::move(even), mul_by_neg_monomial(cc_->EvalSub(c, sub), shift)};
  }

 private:
  CryptoContext<DCRTPoly> cc_;
  const std::map<uint32_t, EvalKey<DCRTPoly>>& keys_;
};

}  // namespace ombra::core::cpu_ref
