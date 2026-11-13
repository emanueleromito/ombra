// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// ombra_native_exact: exactness test of the fast CPU operations of local mode (PERF-B,
// fast_cpu_ops.hpp) against the reference code. Synthetic data; keys are made in this process and
// never written; the only output is pass/fail lines (no values).
//
// Usage: ombra_native_exact <ring_dim> <depth> <scaling_bits> <first_bits> <batch> <digits or 0>
//                           <large 0|1>
// The parameters are the fields of one preset of core/params/ckks_presets.json, passed by
// tests/test_openfhe_native_exact.py, which reads them through ombra.params.
//
//  a. expansion: every RNS word and the metadata of every leaf of the evaluation-form split equal
//     the leaves of the reference cpu_ref::OpenFheOps tree, d = 13 and 16 (and 384 with large = 1).
//  b. score: every RNS word and the metadata of the fused 3-component sum equal the
//     EvalMultNoRelin / EvalAdd loop, for 1, 5, 17 and 400 columns and 1 and 4 OpenMP threads;
//     the serialized result of score_finish decrypts to the expected dot products.
// Exit code: 0 all equal, 1 a difference, 2 usage or error.

#include <omp.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "fast_cpu_ops.hpp"
#include "key/key-ser.h"
#include "ombra/core/expansion_tree.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using ombra::native::Ct;
using KeyMap = std::map<uint32_t, EvalKey<DCRTPoly>>;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++failures;
}

std::vector<std::uint64_t> all_words(const Ct& c) {
  std::vector<std::uint64_t> out;
  for (const auto& e : c->GetElements())
    for (std::size_t t = 0; t < e.GetNumOfElements(); ++t)
      for (std::size_t i = 0; i < e.GetRingDimension(); ++i)
        out.push_back(e.GetAllElements()[t].GetValues()[i].ConvertToInt<std::uint64_t>());
  return out;
}

bool same_meta(const Ct& a, const Ct& b) {
  return a->GetLevel() == b->GetLevel() && a->GetNoiseScaleDeg() == b->GetNoiseScaleDeg() &&
         a->GetScalingFactor() == b->GetScalingFactor() && a->GetKeyTag() == b->GetKeyTag() &&
         a->GetSlots() == b->GetSlots() && a->GetEncodingType() == b->GetEncodingType() &&
         a->GetElements().size() == b->GetElements().size();
}

bool same(const Ct& a, const Ct& b) { return same_meta(a, b) && all_words(a) == all_words(b); }

struct Plain {
  CryptoContext<DCRTPoly> cc;
  KeyPair<DCRTPoly> keys;
  std::mt19937_64 rng{20261113};

  Ct encrypt_random(double scale) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> v(cc->GetRingDimension() / 2);
    for (auto& x : v) x = dist(rng) * scale;
    return cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v));
  }
};

void test_expansion(Plain& p, const KeyMap& kmap, std::size_t d) {
  const std::size_t ring = p.cc->GetRingDimension();
  const Ct root = p.encrypt_random(1e-3);
  ombra::core::cpu_ref::OpenFheOps ref(p.cc, kmap);
  ombra::native::MonomialCache cache;
  ombra::native::FastSplitOps fast(p.cc, kmap, cache);
  const auto want = ombra::core::expand_tree(root, d, ring, ref);
  const auto got = ombra::core::expand_tree(root, d, ring, fast);
  bool ok = want.size() == got.size();
  for (std::size_t j = 0; ok && j < want.size(); ++j) ok = same(want[j], got[j]);
  check(ok, "expansion d=" + std::to_string(d));
}

void test_score(Plain& p, std::size_t n_cols) {
  constexpr std::size_t kPool = 20;
  std::vector<Ct> pool_c, pool_l;
  for (std::size_t i = 0; i < std::min(n_cols, kPool); ++i) {
    pool_c.push_back(p.encrypt_random(1.0));
    pool_l.push_back(p.encrypt_random(1.0));
  }
  std::vector<Ct> cols(n_cols), leaves(n_cols);
  for (std::size_t j = 0; j < n_cols; ++j) {
    cols[j] = pool_c[j % pool_c.size()];
    leaves[j] = pool_l[(j * 7 + 3) % pool_l.size()];
    if (j >= kPool) leaves[j] = pool_l[(j * 3 + 1) % pool_l.size()];
  }
  check(ombra::native::detail::fused_applicable(leaves, cols, n_cols),
        "fused pass applicable, columns=" + std::to_string(n_cols));
  for (int threads : {1, 4}) {
    omp_set_num_threads(threads);
    const Ct want = ombra::native::score_sum_loop(p.cc, leaves, cols, n_cols);
    const Ct got = ombra::native::score_sum(p.cc, leaves, cols, n_cols);
    check(want->GetElements().size() == 3 && same(want, got),
          "score sum columns=" + std::to_string(n_cols) + " threads=" + std::to_string(threads));
  }
}

void test_decrypt(Plain& p) {
  const std::size_t slots = p.cc->GetRingDimension() / 2;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const std::size_t n = 5;
  std::vector<Ct> cols, leaves;
  std::vector<double> want(slots, 0.0), penalty(slots);
  for (std::size_t j = 0; j < n; ++j) {
    std::vector<double> v(slots);
    for (auto& x : v) x = dist(p.rng);
    const double c = dist(p.rng);
    cols.push_back(p.cc->Encrypt(p.keys.publicKey, p.cc->MakeCKKSPackedPlaintext(v)));
    leaves.push_back(p.cc->Encrypt(p.keys.publicKey,
                                   p.cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, c))));
    for (std::size_t s = 0; s < slots; ++s) want[s] += c * v[s];
  }
  for (auto& x : penalty) x = dist(p.rng);
  for (std::size_t s = 0; s < slots; ++s) want[s] += penalty[s];
  for (int threads : {1, 4}) {
    omp_set_num_threads(threads);
    const Ct sum = ombra::native::score_sum(p.cc, leaves, cols, n);
    const Ct res = ombra::native::score_finish(p.cc, sum, penalty);
    std::ostringstream os(std::ios::out | std::ios::binary);
    Serial::Serialize(res, os, SerType::BINARY);
    std::istringstream is(os.str(), std::ios::in | std::ios::binary);
    Ct back;
    Serial::Deserialize(back, is, SerType::BINARY);
    Plaintext pt;
    p.cc->Decrypt(p.keys.secretKey, back, &pt);
    pt->SetLength(slots);
    const auto& got = pt->GetRealPackedValue();
    double err = 0;
    for (std::size_t s = 0; s < slots; ++s) err = std::max(err, std::abs(got[s] - want[s]));
    check(err < 1e-6, "score decrypts to the dot products, threads=" + std::to_string(threads));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr, "usage: ombra_native_exact ring depth scale first batch digits large\n");
    return 2;
  }
  try {
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
    const bool large = std::string(argv[7]) == "1";
    Plain p;
    p.cc = GenCryptoContext(params);
    p.cc->Enable(PKE);
    p.cc->Enable(KEYSWITCH);
    p.cc->Enable(LEVELEDSHE);
    ombra::core::require_security_floor(p.cc);  // hard rule 1
    p.keys = p.cc->KeyGen();
    p.cc->EvalMultKeyGen(p.keys.secretKey);
    const std::size_t ring = p.cc->GetRingDimension();
    std::vector<std::size_t> dims = {13, 16};
    if (large) dims.push_back(384);
    std::size_t max_depth = 0;
    for (std::size_t d : dims) max_depth = std::max(max_depth, ombra::core::expansion_depth(d));
    std::vector<uint32_t> galois;
    for (std::size_t l = 0; l < max_depth; ++l)
      galois.push_back(static_cast<uint32_t>(ring / (std::size_t{1} << l) + 1));
    const auto km = p.cc->EvalAutomorphismKeyGen(p.keys.secretKey, galois);
    for (std::size_t d : dims) test_expansion(p, *km, d);
    for (std::size_t n : {1, 5, 17, 400}) test_score(p, n);
    test_decrypt(p);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
