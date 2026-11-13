// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// OpenFHE part of seeded stored ciphertexts (docs/decisions/0023). Lives in client/native, NOT in
// core/: make_seeded takes the secret key, and core/ never touches it (core/CLAUDE.md). The pure
// format and the generator are in core/include/ombra/core/seeded_ciphertext*.hpp.
//
//   make_seeded: (c0, c1) -> (c0' = c0 + (c1 - u) * s, seed) with u = the seeded uniform
//     polynomial, s the secret key. Then c0' + u * s == c0 + c1 * s word for word, so the pair
//     (c0', u) decrypts exactly like OpenFHE's ciphertext. The original c1 is dropped.
//   expand_seeded: payload -> OpenFHE ciphertext (c0', u). Needs no secret key.
//
// The secret key is read in exactly one place: make_seeded (GetPrivateElement). Nothing here
// prints or stores it.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ombra/core/seeded_ciphertext.hpp"
#include "ombra/core/seeded_ciphertext_prg.hpp"
#include "openfhe.h"

namespace ombra::native {

using lbcrypto::Ciphertext;
using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;
using lbcrypto::NativeInteger;
using lbcrypto::NativeVector;
using lbcrypto::PrivateKey;

namespace seeded_impl {

inline void set_tower_words(DCRTPoly& poly, std::size_t t, std::uint64_t q,
                            const std::uint64_t* words) {
  const std::size_t n = poly.GetRingDimension();
  NativeVector v(n, NativeInteger(q));
  for (std::size_t i = 0; i < n; ++i) v[i] = NativeInteger(words[i]);
  poly.GetAllElements()[t].SetValues(std::move(v), Format::EVALUATION);
}

inline std::uint64_t tower_modulus(const DCRTPoly& poly, std::size_t t) {
  return poly.GetAllElements()[t].GetModulus().ConvertToInt<std::uint64_t>();
}

inline std::vector<std::uint64_t> tower_words(const DCRTPoly& poly, std::size_t t) {
  const auto& vals = poly.GetAllElements()[t].GetValues();
  std::vector<std::uint64_t> out(poly.GetRingDimension());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = vals[i].ConvertToInt<std::uint64_t>();
  return out;
}

}  // namespace seeded_impl

// ct must have exactly 2 components in EVALUATION format. Returns the kind-4 payload (not the
// envelope). `seed` must be fresh from the OS generator and never used twice.
inline std::vector<std::uint8_t> make_seeded(const CryptoContext<DCRTPoly>& cc,
                                             const PrivateKey<DCRTPoly>& sk,
                                             const Ciphertext<DCRTPoly>& ct,
                                             const std::array<std::uint8_t, 32>& seed) {
  using namespace seeded_impl;
  (void)cc;
  const auto& els = ct->GetElements();
  if (els.size() != 2) throw std::invalid_argument("make_seeded: needs a 2-component ciphertext");
  if (els[0].GetFormat() != Format::EVALUATION || els[1].GetFormat() != Format::EVALUATION)
    throw std::invalid_argument("make_seeded: ciphertext must be in EVALUATION format");
  const DCRTPoly& c0 = els[0];
  const DCRTPoly& c1 = els[1];
  const std::size_t towers = c0.GetNumOfElements(), n = c0.GetRingDimension();
  if (towers == 0 || towers > 65535 || c1.GetNumOfElements() != towers)
    throw std::invalid_argument("make_seeded: bad tower count");

  ombra::core::SeededPayload p;
  p.seed = seed;
  p.ring_dim = static_cast<std::uint32_t>(n);
  p.towers = static_cast<std::uint16_t>(towers);
  p.level = static_cast<std::uint32_t>(ct->GetLevel());
  p.noise_scale_deg = static_cast<std::uint32_t>(ct->GetNoiseScaleDeg());
  p.slots = ct->GetSlots();
  p.scaling_factor = ct->GetScalingFactor();
  p.key_tag = ct->GetKeyTag();
  p.moduli.resize(towers);

  DCRTPoly u = c1;  // shape and parameters only; every word is overwritten
  for (std::size_t t = 0; t < towers; ++t) {
    p.moduli[t] = tower_modulus(c0, t);
    const auto w =
        ombra::core::expand_seed_tower(seed, static_cast<std::uint32_t>(t), p.moduli[t], n);
    set_tower_words(u, t, p.moduli[t], w.data());
  }

  // The secret key s, aligned to the ciphertext's towers the way OpenFHE's own decryption does
  // (ckksrns-pke.cpp DecryptCore): copy, then drop the last (sizeQ - sizeQl) towers.
  DCRTPoly s = sk->GetPrivateElement();
  s.SetFormat(Format::EVALUATION);
  if (s.GetNumOfElements() < towers)
    throw std::invalid_argument("make_seeded: key has fewer towers than the ciphertext");
  s.DropLastElements(s.GetNumOfElements() - towers);
  for (std::size_t t = 0; t < towers; ++t)
    if (tower_modulus(s, t) != p.moduli[t])
      throw std::invalid_argument("make_seeded: key and ciphertext moduli differ");

  const DCRTPoly c0p = c0 + (c1 - u) * s;
  p.c0.reserve(towers * n);
  for (std::size_t t = 0; t < towers; ++t) {
    const auto w = tower_words(c0p, t);
    p.c0.insert(p.c0.end(), w.begin(), w.end());
  }
  return ombra::core::write_seeded_payload(p);
}

// Rebuilds the OpenFHE ciphertext (c0', u) from a parsed payload. Refuses a payload whose ring
// dimension or moduli differ from the context's. No secret key.
inline Ciphertext<DCRTPoly> expand_seeded(const CryptoContext<DCRTPoly>& cc,
                                          const ombra::core::SeededPayload& p) {
  using namespace seeded_impl;
  if (p.ring_dim != cc->GetRingDimension())
    throw std::invalid_argument("expand_seeded: ring dimension differs from the context's");
  auto full = cc->GetCryptoParameters()->GetElementParams();
  const std::size_t full_towers = full->GetParams().size();
  if (p.towers > full_towers) throw std::invalid_argument("expand_seeded: too many towers");
  DCRTPoly c0(full, Format::EVALUATION, true);
  c0.DropLastElements(full_towers - p.towers);
  for (std::size_t t = 0; t < p.towers; ++t)
    if (tower_modulus(c0, t) != p.moduli[t])
      throw std::invalid_argument("expand_seeded: moduli differ from the context's");
  DCRTPoly u = c0;
  for (std::size_t t = 0; t < p.towers; ++t) {
    set_tower_words(c0, t, p.moduli[t], p.c0.data() + t * p.ring_dim);
    const auto w = ombra::core::expand_seed_tower(p.seed, static_cast<std::uint32_t>(t),
                                                  p.moduli[t], p.ring_dim);
    set_tower_words(u, t, p.moduli[t], w.data());
  }
  auto ct = std::make_shared<lbcrypto::CiphertextImpl<DCRTPoly>>(cc, p.key_tag,
                                                                 lbcrypto::CKKS_PACKED_ENCODING);
  ct->SetElements({std::move(c0), std::move(u)});
  // Degree before level: SetLevel checks the tower count against the current degree.
  ct->SetNoiseScaleDeg(p.noise_scale_deg);
  ct->SetLevel(p.level);
  ct->SetScalingFactor(p.scaling_factor);
  ct->SetSlots(p.slots);
  return ct;
}

}  // namespace ombra::native
