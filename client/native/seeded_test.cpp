// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// ombra_seeded_exact: exact test of seeded stored ciphertexts (docs/decisions/0023) against
// OpenFHE. Synthetic data; keys are made in this process and never written; the only output is
// pass/fail lines and, in `kat` mode, generator words of public seeds (no secrets).
//
// Usage:
//   ombra_seeded_exact <ring_dim> <depth> <scaling_bits> <first_bits> <batch> <digits or 0> <tol>
//     a. for several random public-key ciphertexts: c0 + c1*s == c0' + u*s word for word in every
//        tower, metadata equal, u != c1, component 1 == expand_seed_tower, seeded_words == words
//        of the expanded ciphertext, payload parse round trip;
//     b. Decrypt(expanded) == Decrypt(original) == known plaintext within <tol>;
//     c. expand_seeded refuses a payload with other moduli or ring dimension;
//     d. prints "kat" lines: moduli and the first 8 and last 2 generator words of towers 0 and 1.
//   ombra_seeded_exact expand <seed hex> <tower> <q> <N>   prints the N words, one per line.
// Exit code: 0 all pass, 1 a failure, 2 usage or error.
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "seeded_ciphertext_openfhe.hpp"

using namespace lbcrypto;
using namespace ombra::native;
using Ct = Ciphertext<DCRTPoly>;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++failures;
}

std::array<std::uint8_t, 32> seed_from(std::mt19937_64& rng) {
  std::array<std::uint8_t, 32> s{};
  for (auto& b : s) b = static_cast<std::uint8_t>(rng());
  return s;
}

std::vector<std::uint64_t> words_of(const DCRTPoly& p) {
  std::vector<std::uint64_t> out;
  for (std::size_t t = 0; t < p.GetNumOfElements(); ++t) {
    auto w = seeded_impl::tower_words(p, t);
    out.insert(out.end(), w.begin(), w.end());
  }
  return out;
}

std::vector<double> decrypt(const CryptoContext<DCRTPoly>& cc, const PrivateKey<DCRTPoly>& sk,
                            const Ct& ct, std::size_t slots) {
  Plaintext pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(slots);
  return pt->GetRealPackedValue();
}

void print_kat(const CryptoContext<DCRTPoly>& cc) {
  std::array<std::uint8_t, 32> seed{};
  for (std::size_t i = 0; i < 32; ++i) seed[i] = static_cast<std::uint8_t>(i * 7 + 1);
  const auto full = cc->GetCryptoParameters()->GetElementParams()->GetParams();
  const std::size_t n = cc->GetRingDimension();
  for (std::uint32_t t = 0; t < 2 && t < full.size(); ++t) {
    const std::uint64_t q = full[t]->GetModulus().ConvertToInt<std::uint64_t>();
    const auto w = ombra::core::expand_seed_tower(seed, t, q, n);
    std::printf("kat tower=%u q=%llu n=%zu :", t, static_cast<unsigned long long>(q), n);
    for (std::size_t i = 0; i < 8; ++i) std::printf(" %llu", (unsigned long long)w[i]);
    std::printf(" ...");
    for (std::size_t i = n - 2; i < n; ++i) std::printf(" %llu", (unsigned long long)w[i]);
    std::printf("\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 6 && std::string(argv[1]) == "expand") {
      std::array<std::uint8_t, 32> seed{};
      const std::string hex = argv[2];
      if (hex.size() != 64) return 2;
      for (std::size_t i = 0; i < 32; ++i)
        seed[i] = static_cast<std::uint8_t>(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
      const auto w = ombra::core::expand_seed_tower(seed, std::stoul(argv[3]), std::stoull(argv[4]),
                                                    std::stoull(argv[5]));
      for (auto x : w) std::printf("%llu\n", static_cast<unsigned long long>(x));
      return 0;
    }
    if (argc != 8) {
      std::fprintf(stderr, "usage: ombra_seeded_exact ring depth scale first batch digits tol\n");
      return 2;
    }
    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecurityLevel(HEStd_128_classic);
    params.SetRingDim(std::stoull(argv[1]));
    params.SetMultiplicativeDepth(std::stoul(argv[2]));
    params.SetScalingModSize(std::stoul(argv[3]));
    params.SetFirstModSize(std::stoul(argv[4]));
    params.SetBatchSize(std::stoul(argv[5]));
    params.SetScalingTechnique(FIXEDMANUAL);
    params.SetKeySwitchTechnique(HYBRID);
    if (std::stoul(argv[6]) != 0) params.SetNumLargeDigits(std::stoul(argv[6]));
    const double tol = std::stod(argv[7]);
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    auto keys = cc->KeyGen();
    const std::size_t slots = cc->GetRingDimension() / 2;
    std::mt19937_64 rng{20261113};
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::array<std::uint8_t, 32>> seen;

    DCRTPoly s = keys.secretKey->GetPrivateElement();
    s.SetFormat(Format::EVALUATION);
    for (int round = 0; round < 4; ++round) {
      std::vector<double> v(slots);
      for (auto& x : v) x = dist(rng);
      Ct ct = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v));
      const auto seed = seed_from(rng);
      const auto bytes = make_seeded(cc, keys.secretKey, ct, seed);
      const auto payload = ombra::core::read_seeded_payload(bytes);
      Ct ct2 = expand_seeded(cc, payload);
      const std::string r = " round=" + std::to_string(round);

      DCRTPoly sl = s;
      sl.DropLastElements(s.GetNumOfElements() - ct->GetElements()[0].GetNumOfElements());
      const DCRTPoly ph1 = ct->GetElements()[0] + ct->GetElements()[1] * sl;
      const DCRTPoly ph2 = ct2->GetElements()[0] + ct2->GetElements()[1] * sl;
      check(words_of(ph1) == words_of(ph2), "phase c0 + c1*s equal word for word" + r);

      const bool meta =
          ct->GetLevel() == ct2->GetLevel() && ct->GetNoiseScaleDeg() == ct2->GetNoiseScaleDeg() &&
          ct->GetScalingFactor() == ct2->GetScalingFactor() && ct->GetSlots() == ct2->GetSlots() &&
          ct->GetKeyTag() == ct2->GetKeyTag() && ct->GetEncodingType() == ct2->GetEncodingType() &&
          ct2->GetElements().size() == 2 &&
          ct2->GetElements()[0].GetFormat() == Format::EVALUATION &&
          ct2->GetElements()[1].GetFormat() == Format::EVALUATION && !ct->GetKeyTag().empty();
      check(meta, "metadata equal (level, degree, scale, slots, key tag, encoding)" + r);
      check(words_of(ct->GetElements()[1]) != words_of(ct2->GetElements()[1]), "u != c1" + r);
      std::vector<std::uint64_t> u;
      for (std::uint32_t t = 0; t < payload.towers; ++t) {
        auto w = ombra::core::expand_seed_tower(seed, t, payload.moduli[t], payload.ring_dim);
        u.insert(u.end(), w.begin(), w.end());
      }
      check(words_of(ct2->GetElements()[1]) == u, "component 1 == expand_seed_tower" + r);
      auto sw = ombra::core::seeded_words(payload);
      auto want = words_of(ct2->GetElements()[0]);
      const auto w1 = words_of(ct2->GetElements()[1]);
      want.insert(want.end(), w1.begin(), w1.end());
      check(sw == want, "seeded_words == words of the expanded ciphertext" + r);
      check(bytes.size() <
                8 * 2 * payload.towers * payload.ring_dim * 6 / 10 + 8 * payload.towers + 200,
            "payload is about half of the ciphertext" + r);
      for (const auto& o : seen) check(o != seed, "seed not reused" + r);
      seen.push_back(seed);

      const auto d1 = decrypt(cc, keys.secretKey, ct, slots);
      const auto d2 = decrypt(cc, keys.secretKey, ct2, slots);
      double e12 = 0, e2v = 0;
      for (std::size_t i = 0; i < slots; ++i) {
        e12 = std::max(e12, std::abs(d1[i] - d2[i]));
        e2v = std::max(e2v, std::abs(d2[i] - v[i]));
      }
      check(e12 < tol, "Decrypt(expanded) == Decrypt(original)" + r);
      check(e2v < tol, "Decrypt(expanded) == plaintext" + r);

      if (round == 0) {
        // refusals of expand_seeded
        auto other = payload;
        other.moduli[0] += 2;
        bool threw = false;
        try {
          expand_seeded(cc, other);
        } catch (const std::invalid_argument&) {
          threw = true;
        }
        check(threw, "expand_seeded refuses other moduli");
        other = payload;
        other.ring_dim /= 2;
        other.c0.resize(other.towers * other.ring_dim);
        threw = false;
        try {
          expand_seeded(cc, other);
        } catch (const std::invalid_argument&) {
          threw = true;
        }
        check(threw, "expand_seeded refuses another ring dimension");
      }
    }
    print_kat(cc);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
