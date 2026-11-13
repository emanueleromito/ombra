// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Experiment 0004: the CPU expansion tree (local mode), two ways.
//
//   baseline : core/src/openfhe_expand_ops.hpp as used by client/native/ombra_native.cpp: per
//              split EvalAutomorphism, EvalAdd, EvalSub, then X^-k applied in COEFFICIENT form
//              (an inverse and a forward NTT of every tower of both components, plus clones).
//   eval_mono: the same split, with X^-k applied in EVALUATION form: one pointwise product with
//              the NTT of the monomial, computed once per tree level. This is what the GPU path
//              already does (the `mono` table of core/kernels/ntt.cu).
// Both are exact ring operations, so the leaves must be identical: every RNS word of every leaf
// is compared and the process exits non-zero on any difference.
//
// Usage: expand_cpu <ring_dim> <depth> <scaling_bits> <first_bits> <batch> <digits or 0> <d>
//                   <reps> <seed>
// Synthetic query; keys are made in-process and never written.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "ombra/core/expansion_tree.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "openfhe_expand_ops.hpp"

using namespace lbcrypto;
using Ct = Ciphertext<DCRTPoly>;
using Clock = std::chrono::steady_clock;
using KeyMap = std::map<uint32_t, EvalKey<DCRTPoly>>;

namespace {

template <class Split>
struct LevelOps {
  Split one;
  std::pair<Ct, Ct> split(const Ct& c, std::size_t l, std::uint32_t g, std::size_t s, bool odd) {
    return one(c, l, g, s, odd);
  }
  std::vector<std::pair<Ct, Ct>> split_level(const std::vector<Ct>& nodes, std::size_t l,
                                             std::uint32_t g, std::size_t s,
                                             const std::vector<char>& odd) {
    std::vector<std::pair<Ct, Ct>> out(nodes.size());
    const long n = (long)nodes.size();
#pragma omp parallel for schedule(dynamic)
    for (long i = 0; i < n; ++i) out[i] = one(nodes[i], l, g, s, odd[i] != 0);
    return out;
  }
};

// NTT of X^(-k) = -X^(N-k) over the towers of `like`.
DCRTPoly monomial(const DCRTPoly& like, std::size_t k) {
  DCRTPoly m(like.GetParams(), Format::COEFFICIENT, true);
  const std::size_t n = like.GetRingDimension();
  for (std::size_t t = 0; t < m.GetNumOfElements(); ++t) {
    NativePoly tower = m.GetElementAtIndex(t);
    NativeVector v(n, tower.GetModulus());
    v[n - k] = tower.GetModulus() - NativeInteger(1);
    tower.SetValues(std::move(v), Format::COEFFICIENT);
    m.SetElementAtIndex(t, std::move(tower));
  }
  m.SetFormat(Format::EVALUATION);
  return m;
}

std::vector<std::uint64_t> all_words(const std::vector<Ct>& leaves) {
  std::vector<std::uint64_t> out;
  for (const Ct& c : leaves)
    for (const auto& e : c->GetElements())
      for (std::size_t t = 0; t < e.GetNumOfElements(); ++t)
        for (std::size_t i = 0; i < e.GetRingDimension(); ++i)
          out.push_back(e.GetAllElements()[t].GetValues()[i].ConvertToInt<std::uint64_t>());
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 10) {
    std::fprintf(stderr, "usage: expand_cpu N depth scale first batch digits d reps seed\n");
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
  const std::size_t d = std::stoul(argv[7]);
  const int reps = std::stoi(argv[8]);
  auto cc = GenCryptoContext(p);
  ombra::core::require_security_floor(cc);  // hard rule 1: never below the 128-bit floor
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  auto keys = cc->KeyGen();
  const std::size_t depth = ombra::core::expansion_depth(d);
  std::vector<uint32_t> galois;
  for (std::size_t l = 0; l < depth; ++l) galois.push_back((uint32_t)(ring / (1u << l) + 1));
  const auto km = cc->EvalAutomorphismKeyGen(keys.secretKey, galois);
  const KeyMap& kmap = *km;

  std::mt19937_64 rng(std::stoull(argv[9]));
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> v(ring / 2);
  for (auto& x : v) x = dist(rng) * 1e-3;
  const Ct root = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v));

  ombra::core::cpu_ref::OpenFheOps ref(cc, kmap);
  const auto base_split = [&](const Ct& c, std::size_t l, std::uint32_t g, std::size_t s,
                              bool odd) { return ref.split(c, l, g, s, odd); };
  std::vector<DCRTPoly> monos;
  for (std::size_t l = 0; l < depth; ++l)
    monos.push_back(monomial(root->GetElements()[0], std::size_t{1} << l));
  const auto mono_split = [&](const Ct& c, std::size_t l, std::uint32_t g, std::size_t,
                              bool odd) -> std::pair<Ct, Ct> {
    Ct sub = cc->EvalAutomorphism(c, g, kmap);
    Ct even = cc->EvalAdd(c, sub);
    if (!odd) return {std::move(even), Ct()};
    // sub = (c - sub) * mono, in place on the automorphism's own buffers.
    auto& se = sub->GetElements();
    const auto& ce = c->GetElements();
    for (std::size_t r = 0; r < se.size(); ++r) {
      se[r] = ce[r] - se[r];
      se[r] *= monos[l];
    }
    return {std::move(even), std::move(sub)};
  };

  LevelOps<decltype(base_split)> a{base_split};
  LevelOps<decltype(mono_split)> b{mono_split};
  const auto la = ombra::core::expand_tree(root, d, ring, a);
  const auto lb = ombra::core::expand_tree(root, d, ring, b);
  const bool exact = all_words(la) == all_words(lb);
  bool meta = la.size() == lb.size();
  for (std::size_t j = 0; meta && j < la.size(); ++j) {
    meta = la[j]->GetLevel() == lb[j]->GetLevel() &&
           la[j]->GetNoiseScaleDeg() == lb[j]->GetNoiseScaleDeg() &&
           la[j]->GetScalingFactor() == lb[j]->GetScalingFactor() &&
           la[j]->GetKeyTag() == lb[j]->GetKeyTag() && la[j]->GetSlots() == lb[j]->GetSlots();
  }
  const auto time = [&](auto& ops) {
    std::vector<double> s;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = Clock::now();
      ombra::core::expand_tree(root, d, ring, ops);
      s.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    std::sort(s.begin(), s.end());
    return s[s.size() / 2];
  };
  const double ta = time(a), tb = time(b);
  std::printf(
      "{\"ring_dim\": %llu, \"d\": %zu, \"threads\": %d, \"leaves\": %zu, \"exact\": %s, "
      "\"metadata_equal\": %s, \"baseline_ms\": %.2f, \"eval_mono_ms\": %.2f}\n",
      (unsigned long long)ring, d, omp_get_max_threads(), la.size(), exact ? "true" : "false",
      meta ? "true" : "false", ta, tb);
  return exact && meta ? 0 : 1;
}
