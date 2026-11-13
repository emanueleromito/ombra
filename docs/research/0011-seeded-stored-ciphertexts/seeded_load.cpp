// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Experiment 0011: what the server pays per stored column when it loads a block.
//
//   plain  : today's path for an OpenFHE column: Serial::Deserialize, then copy the words out.
//   seeded : read_seeded_payload (with the word-range check) plus the BLAKE2xb expansion of the
//            random half (core/include/ombra/core/seeded_ciphertext_prg.hpp), i.e. what
//            EngineCore::load_seeded_column does.
// The seeded payload here is built from the plain ciphertext's own c0 words and a fixed seed: the
// load cost does not depend on the values, and no secret key is needed for this measurement.
//
// Usage: seeded_load <ring_dim> <depth> <scaling_bits> <first_bits> <batch> <digits or 0> <reps>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "ombra/core/seeded_ciphertext.hpp"
#include "ombra/core/seeded_ciphertext_prg.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;

namespace {
template <class F>
double median_us(F&& f, int reps) {
  std::vector<double> s;
  for (int r = 0; r < reps + 1; ++r) {
    const auto t0 = Clock::now();
    f();
    if (r > 0) s.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
  }
  std::sort(s.begin(), s.end());
  return s[s.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr, "usage: seeded_load N depth scale first batch digits reps\n");
    return 2;
  }
  const std::uint64_t ring = std::stoull(argv[1]);
  CCParams<CryptoContextCKKSRNS> p;
  p.SetSecurityLevel(HEStd_128_classic);
  p.SetRingDim(ring);
  p.SetMultiplicativeDepth(std::stoul(argv[2]));
  p.SetScalingModSize(std::stoul(argv[3]));
  p.SetFirstModSize(std::stoul(argv[4]));
  p.SetBatchSize(std::stoul(argv[5]));
  p.SetScalingTechnique(FIXEDMANUAL);
  p.SetKeySwitchTechnique(HYBRID);
  if (std::stoul(argv[6]) != 0) p.SetNumLargeDigits(std::stoul(argv[6]));
  const int reps = std::stoi(argv[7]);
  auto cc = GenCryptoContext(p);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  auto keys = cc->KeyGen();
  std::vector<double> v(ring / 2, 0.25);
  const auto ct = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v));

  std::ostringstream os(std::ios::binary);
  Serial::Serialize(ct, os, SerType::BINARY);
  const std::string plain = os.str();

  ombra::core::SeededPayload sp;
  for (std::size_t i = 0; i < sp.seed.size(); ++i) sp.seed[i] = static_cast<std::uint8_t>(i + 1);
  const auto& c0 = ct->GetElements()[0];
  sp.ring_dim = static_cast<std::uint32_t>(ring);
  sp.towers = static_cast<std::uint16_t>(c0.GetNumOfElements());
  sp.level = 0;
  sp.noise_scale_deg = 1;
  sp.slots = static_cast<std::uint32_t>(ring / 2);
  sp.scaling_factor = ct->GetScalingFactor();
  sp.key_tag = ct->GetKeyTag();
  for (std::size_t t = 0; t < sp.towers; ++t) {
    sp.moduli.push_back(c0.GetAllElements()[t].GetModulus().ConvertToInt<std::uint64_t>());
    for (std::size_t i = 0; i < ring; ++i)
      sp.c0.push_back(c0.GetAllElements()[t].GetValues()[i].ConvertToInt<std::uint64_t>());
  }
  const std::vector<std::uint8_t> seeded = ombra::core::write_seeded_payload(sp);

  std::vector<std::uint64_t> dst(2 * sp.towers * ring);
  const double t_plain = median_us(
      [&] {
        std::istringstream in(plain, std::ios::binary);
        Ciphertext<DCRTPoly> c;
        Serial::Deserialize(c, in, SerType::BINARY);
        std::size_t k = 0;
        for (const auto& e : c->GetElements())
          for (std::size_t t = 0; t < e.GetNumOfElements(); ++t)
            for (std::size_t i = 0; i < ring; ++i)
              dst[k++] = e.GetAllElements()[t].GetValues()[i].ConvertToInt<std::uint64_t>();
      },
      reps);
  const double t_parse = median_us(
      [&] { (void)ombra::core::read_seeded_payload(seeded.data(), seeded.size()); }, reps);
  const double t_seeded = median_us(
      [&] {
        const auto q = ombra::core::read_seeded_payload(seeded.data(), seeded.size());
        std::copy(q.c0.begin(), q.c0.end(), dst.begin());
        for (std::uint32_t t = 0; t < q.towers; ++t) {
          const auto w = ombra::core::expand_seed_tower(q.seed, t, q.moduli[t], q.ring_dim);
          std::copy(w.begin(), w.end(), dst.begin() + q.c0.size() + t * ring);
        }
      },
      reps);
  std::printf(
      "{\"ring_dim\": %llu, \"towers\": %u, \"plain_bytes\": %zu, \"seeded_bytes\": %zu, "
      "\"plain_load_us\": %.1f, \"seeded_parse_us\": %.1f, \"seeded_load_us\": %.1f}\n",
      (unsigned long long)ring, (unsigned)sp.towers, plain.size(), seeded.size(), t_plain, t_parse,
      t_seeded);
  return 0;
}
