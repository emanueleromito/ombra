// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Experiment 0005: what one ciphertext deserialization costs, and where.
//
// Times Serial::Deserialize of a fresh ciphertext (the query the server reads) and of a
// multiplied and rescaled one (the block the client reads), from memory, and for comparison:
//   memcpy      : copying the same number of bytes (the floor);
//   words_only  : building the ciphertext from the raw words into a clone of a sample
//                 ciphertext (what core::make_ciphertext_like does), i.e. the cost if the words
//                 were read without OpenFHE's archive.
// Run under `valgrind --tool=callgrind` with reps = 3 to see where the archive spends its time.
//
// Usage: deser <ring_dim> <depth> <scaling_bits> <first_bits> <batch> <digits or 0> <reps>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using Ct = Ciphertext<DCRTPoly>;
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
    std::fprintf(stderr, "usage: deser N depth scale first batch digits reps\n");
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
  ombra::core::require_security_floor(cc);  // hard rule 1: never below the 128-bit floor
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  auto keys = cc->KeyGen();
  cc->EvalMultKeyGen(keys.secretKey);
  std::vector<double> v(ring / 2, 0.25);
  const Ct fresh = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v));
  const Ct served = cc->Rescale(cc->EvalMult(fresh, fresh));

  std::printf("{\"ring_dim\": %llu, \"rows\": [", (unsigned long long)ring);
  bool first = true;
  for (const auto& [name, ct] : {std::pair<const char*, Ct>{"fresh", fresh}, {"served", served}}) {
    std::ostringstream os(std::ios::binary);
    Serial::Serialize(ct, os, SerType::BINARY);
    const std::string bytes = os.str();
    const double ser = median_us(
        [&] {
          std::ostringstream o(std::ios::binary);
          Serial::Serialize(ct, o, SerType::BINARY);
        },
        reps);
    const double de = median_us(
        [&] {
          std::istringstream in(bytes, std::ios::binary);
          Ct out;
          Serial::Deserialize(out, in, SerType::BINARY);
        },
        reps);
    std::vector<char> sink(bytes.size());
    const double cp =
        median_us([&] { std::memcpy(sink.data(), bytes.data(), bytes.size()); }, reps);
    // words_only: clone a sample and overwrite its towers from a flat word array.
    std::vector<std::uint64_t> flat;
    for (const auto& e : ct->GetElements())
      for (std::size_t t = 0; t < e.GetNumOfElements(); ++t)
        for (std::size_t i = 0; i < ring; ++i)
          flat.push_back(e.GetAllElements()[t].GetValues()[i].ConvertToInt<std::uint64_t>());
    const double wo = median_us(
        [&] {
          Ct out = ct->Clone();
          auto& els = out->GetElements();
          const std::uint64_t* w = flat.data();
          for (auto& e : els) {
            auto& towers = e.GetAllElements();
            for (auto& tw : towers) {
              NativeVector nv(ring, tw.GetModulus());
              std::memcpy(&nv[0], w, ring * 8);
              w += ring;
              tw.SetValues(std::move(nv), Format::EVALUATION);
            }
          }
        },
        reps);
    std::printf(
        "%s{\"ciphertext\": \"%s\", \"bytes\": %zu, \"serialize_us\": %.1f, "
        "\"deserialize_us\": %.1f, \"memcpy_us\": %.1f, \"words_only_us\": %.1f}",
        first ? "" : ", ", name, bytes.size(), ser, de, cp, wo);
    first = false;
  }
  std::printf("]}\n");
  return 0;
}
