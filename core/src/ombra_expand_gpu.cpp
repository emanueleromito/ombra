// ombra_expand_gpu: GPU query expansion checked exactly against OpenFHE (T0.5, decision 0010).
//
// Usage: ombra_expand_gpu <context.bin> <query.bin> <autokeys.bin> <d> [out_dir]
//
// Inputs are OpenFHE BINARY serializations of public material only (context, one query
// ciphertext, automorphism keys); no secret key is read or needed. Stages, each compared EXACTLY
// (every RNS word) with OpenFHE on the same inputs; any mismatch exits with 1:
//   1. NTT / inverse NTT per tower, Q and special-prime (P) towers  vs  DCRTPoly::SetFormat
//   2. key-switch digits vs EvalFastRotationPrecompute; inner product + c0*P + automorphism
//      permutation vs EvalFastKeySwitchCoreExt + AutomorphismTransform; ModDown vs ApproxModDown
//   3. one full key-switched automorphism per Galois element vs EvalAutomorphism
//   4. the whole expansion tree (expand_tree with the CPU and the GPU Ops, the GPU one batched per
//      level, in 3-node chunks, and node by node): every leaf's RNS words, moduli, level,
//      noise-scale degree and scaling factor
// Stage 1 and 2 use synthetic pseudo-random words for the special-prime towers. Prints stage
// verdicts, indices of mismatches and timings only: never data values. If out_dir is given, writes
// cpu_<j>.bin and gpu_<j>.bin for the leaves. Exit code: 0 all exact, 1 mismatch, 2 error.

#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/column_search.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/expansion_tree.hpp"
#include "openfhe.h"
#include "openfhe_expand_ops.hpp"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;
using u64 = std::uint64_t;

namespace {

using ombra::core::GpuCt;
using ombra::core::GpuExpansion;
using ombra::core::cpu_ref::OpenFheOps;
using Ct = Ciphertext<DCRTPoly>;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

int g_failures = 0;

// Compares two word arrays laid out as rows of `row` words; reports PASS or the first mismatch
// position and the mismatch count (never the values).
void check(const std::string& stage, const std::vector<u64>& got, const std::vector<u64>& want,
           std::size_t row) {
  if (got.size() != want.size()) {
    std::printf("%s: FAIL (size %zu vs %zu)\n", stage.c_str(), got.size(), want.size());
    ++g_failures;
    return;
  }
  std::size_t bad = 0, first = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      if (bad++ == 0) first = i;
    }
  }
  if (bad == 0) {
    std::printf("%s: PASS\n", stage.c_str());
  } else {
    std::printf("%s: FAIL (%zu of %zu words differ; first at row %zu, index %zu)\n", stage.c_str(),
                bad, got.size(), first / row, first % row);
    ++g_failures;
  }
}

// [tower][coeff] words of a polynomial in whatever format it is in.
std::vector<u64> poly_words(const DCRTPoly& p) {
  std::vector<u64> w;
  for (std::size_t t = 0; t < p.GetNumOfElements(); ++t) {
    const auto& v = p.GetElementAtIndex(t).GetValues();
    for (std::size_t i = 0; i < v.GetLength(); ++i) w.push_back(v[i].ConvertToInt<u64>());
  }
  return w;
}

void append(std::vector<u64>& dst, const std::vector<u64>& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

std::vector<u64> ct_words(const Ct& ct) { return ombra::core::flatten_ciphertexts({ct}).words; }

void stage1(GpuExpansion& gx, const Ct& query, const DCRTPoly& c1) {
  const auto& t = gx.tables();
  const std::size_t n = t.ring_dim, nq = t.num_q, qp = nq + t.num_p;
  ombra::core::ExpansionDevice& d = gx.device();

  // (a) the Q towers of the real second component.
  DCRTPoly coef = c1;
  coef.SetFormat(Format::COEFFICIENT);
  std::vector<std::uint32_t> q_towers;
  for (std::size_t i = 0; i < nq; ++i) q_towers.push_back(static_cast<std::uint32_t>(i));
  const auto eval_w = poly_words(c1);
  const auto coef_w = poly_words(coef);
  check("stage 1a inverse NTT, Q towers of c1", d.test_ntt(false, eval_w, q_towers), coef_w, n);
  check("stage 1a forward NTT, Q towers of c1", d.test_ntt(true, coef_w, q_towers), eval_w, n);

  // (b) all towers of the extended basis Q * P on pseudo-random words (synthetic, seeded).
  const auto params_qp =
      c1.GetExtendedCRTBasis(std::dynamic_pointer_cast<CryptoParametersRNS>(
                                 query->GetCryptoContext()->GetCryptoParameters())
                                 ->GetParamsP());
  DCRTPoly ext(params_qp, Format::COEFFICIENT, true);
  u64 state = 0x9E3779B97F4A7C15ull;
  for (std::size_t x = 0; x < qp; ++x) {
    NativePoly tower = ext.GetElementAtIndex(x);
    const NativeInteger q = tower.GetModulus();
    NativeVector v(n, q);
    for (std::size_t i = 0; i < n; ++i) {
      state = state * 6364136223846793005ull + 1442695040888963407ull;
      v[i] = NativeInteger(state % t.moduli[x]);
    }
    tower.SetValues(std::move(v), Format::COEFFICIENT);
    ext.SetElementAtIndex(x, std::move(tower));
  }
  const auto ext_coef = poly_words(ext);
  ext.SetFormat(Format::EVALUATION);
  const auto ext_eval = poly_words(ext);
  std::vector<std::uint32_t> all;
  for (std::size_t x = 0; x < qp; ++x) all.push_back(static_cast<std::uint32_t>(x));
  check("stage 1b forward NTT, Q and P towers", d.test_ntt(true, ext_coef, all), ext_eval, n);
  check("stage 1b inverse NTT, Q and P towers", d.test_ntt(false, ext_eval, all), ext_coef, n);
}

void stage2and3(GpuExpansion& gx, const CryptoContext<DCRTPoly>& cc, const Ct& query,
                const std::map<std::uint32_t, EvalKey<DCRTPoly>>& keys, std::size_t d) {
  const auto& t = gx.tables();
  const std::size_t n = t.ring_dim, nq = t.num_q, qp = nq + t.num_p, nd = t.digit_first.size();
  const auto cp = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
  const std::vector<u64> query_w = ct_words(query);
  const DCRTPoly& cv0 = query->GetElements()[0];

  const auto cpu_digits = cc->EvalFastRotationPrecompute(query);
  if (cpu_digits->size() != nd) {
    std::printf("stage 2 digits: FAIL (digit count %zu vs %zu)\n", cpu_digits->size(), nd);
    ++g_failures;
    return;
  }
  std::vector<u64> cpu_d;
  for (const DCRTPoly& p : *cpu_digits) append(cpu_d, poly_words(p));
  check("stage 2 digits (ModUp)", gx.device().test_digits(query_w), cpu_d, n);

  const PlaintextModulus tt = 0;  // CKKS: noise scale 1, as in EvalAutomorphismCore
  for (std::size_t l = 0; l < ombra::core::expansion_depth(d); ++l) {
    const auto galois = static_cast<std::uint32_t>(n / (std::size_t{1} << l) + 1);
    const std::string tag = " (galois " + std::to_string(galois) + ")";
    std::vector<std::uint32_t> map(n);
    PrecomputeAutoMap(static_cast<std::uint32_t>(n), galois, &map);

    auto ba =
        cc->GetScheme()->EvalFastKeySwitchCoreExt(cpu_digits, keys.at(galois), cv0.GetParams());
    for (std::size_t i = 0; i < nq; ++i) {
      ba[0].GetAllElements()[i].MultAccEqNoCheck(cv0.GetAllElements()[i], cp->GetPModq()[i]);
    }
    ba[0] = ba[0].AutomorphismTransform(galois, map);
    ba[1] = ba[1].AutomorphismTransform(galois, map);
    std::vector<u64> cpu_ext = poly_words(ba[0]);
    append(cpu_ext, poly_words(ba[1]));
    check("stage 2 inner product + c0*P + permutation" + tag, gx.device().test_ext(query_w, galois),
          cpu_ext, n);

    std::vector<u64> cpu_down;
    for (int r = 0; r < 2; ++r) {
      const DCRTPoly down =
          ba[r].ApproxModDown(cv0.GetParams(), cp->GetParamsP(), cp->GetPInvModq(),
                              cp->GetPInvModqPrecon(), cp->GetPHatInvModp(),
                              cp->GetPHatInvModpPrecon(), cp->GetPHatModq(), cp->GetModqBarrettMu(),
                              cp->GettInvModp(), cp->GettInvModpPrecon(), tt, cp->GettModqPrecon());
      append(cpu_down, poly_words(down));
    }
    check("stage 2 ModDown" + tag, gx.device().test_mod_down(cpu_ext), cpu_down, n);

    const Ct cpu_auto = cc->EvalAutomorphism(query, galois, keys);
    check("stage 3 key-switched automorphism" + tag, gx.device().test_automorphism(query_w, galois),
          ct_words(cpu_auto), n);
  }
  (void)qp;
}

bool leaf_equal(const Ct& cpu, const Ct& gpu, std::size_t j) {
  const auto a = ombra::core::flatten_ciphertexts({cpu});
  const auto b = ombra::core::flatten_ciphertexts({gpu});
  bool ok = true;
  if (a.moduli != b.moduli || a.num_towers != b.num_towers || a.ring_dim != b.ring_dim) {
    std::printf("leaf %zu: moduli or shape differ\n", j);
    ok = false;
  }
  if (a.level != b.level || a.noise_scale_deg != b.noise_scale_deg ||
      a.scaling_factor != b.scaling_factor) {
    std::printf("leaf %zu: level, noise degree or scaling factor differ\n", j);
    ok = false;
  }
  if (a.words != b.words) {
    std::size_t bad = 0, first = 0;
    for (std::size_t i = 0; i < a.words.size() && i < b.words.size(); ++i) {
      if (a.words[i] != b.words[i] && bad++ == 0) first = i;
    }
    std::printf("leaf %zu: %zu words differ (first at component %zu, tower %zu, index %zu)\n", j,
                bad, first / (a.num_towers * a.ring_dim), (first / a.ring_dim) % a.num_towers,
                first % a.ring_dim);
    ok = false;
  }
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    std::fprintf(stderr, "usage: ombra_expand_gpu <context> <query> <autokeys> <d> [out_dir]\n");
    return 2;
  }
  try {
    const std::size_t d = std::stoul(argv[4]);
    const std::string out_dir = argc == 6 ? argv[5] : "";

    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    std::ifstream keyfile(argv[3], std::ios::in | std::ios::binary);
    if (!keyfile.is_open() ||
        !CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(keyfile, SerType::BINARY)) {
      throw std::runtime_error("cannot read automorphism keys");
    }
    Ct query;
    if (!Serial::DeserializeFromFile(argv[2], query, SerType::BINARY)) {
      throw std::runtime_error("cannot read query ciphertext");
    }
    const auto& keys = CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(query->GetKeyTag());
    const std::size_t ring = cc->GetRingDimension();

    GpuExpansion gx(cc, query, keys, d);
    std::printf("device: %s; towers Q=%zu P=%zu digits=%zu keys=%zu setup_seconds=%.3f\n",
                ombra::core::cuda_device_name().c_str(), gx.tables().num_q, gx.tables().num_p,
                gx.tables().digit_first.size(), gx.key_count(), gx.setup_seconds());

    stage1(gx, query, query->GetElements()[1]);
    stage2and3(gx, cc, query, keys, d);

    // Stage 4: the whole tree with both Ops.
    OpenFheOps cpu_ops(cc, keys);
    ombra::core::ExpansionStats stats;
    auto t0 = Clock::now();
    std::vector<Ct> cpu_leaves = ombra::core::expand_tree(query, d, ring, cpu_ops, &stats);
    const double cpu_seconds = seconds_since(t0);

    t0 = Clock::now();
    const GpuCt root = gx.upload(query);
    const double h2d_seconds = seconds_since(t0);
    {  // warm-up run: fills the buffer pool and loads the kernels
      auto warm = ombra::core::expand_tree(root, d, ring, gx);
      gx.synchronize();
    }
    t0 = Clock::now();
    std::vector<GpuCt> gpu_dev = ombra::core::expand_tree(root, d, ring, gx);
    gx.synchronize();
    const double gpu_seconds = seconds_since(t0);

    // The same tree again with a tiny chunk size (3 nodes per batch, so most levels are split into
    // several chunks and the last chunk is ragged) and with the per-node path (split of one node at
    // a time): all must give the very same words as the batched run and as OpenFHE.
    const std::size_t default_batch = gx.device().max_batch();
    gx.device().set_max_batch(3);
    std::vector<GpuCt> gpu_chunked = ombra::core::expand_tree(root, d, ring, gx);
    gx.device().set_max_batch(default_batch);
    struct PerNode {  // hides split_level so expand_tree takes the per-node path
      GpuExpansion& g;
      std::pair<GpuCt, GpuCt> split(const GpuCt& c, std::size_t level, std::uint32_t galois,
                                    std::size_t shift, bool odd) {
        return g.split(c, level, galois, shift, odd);
      }
    } per_node{gx};
    std::vector<GpuCt> gpu_single = ombra::core::expand_tree(root, d, ring, per_node);
    gx.synchronize();

    std::size_t bad_leaves = 0;
    for (std::size_t j = 0; j < d; ++j) {
      const Ct gpu = gx.download(gpu_dev[j]);
      if (!leaf_equal(cpu_leaves[j], gpu, j)) ++bad_leaves;
      if (!leaf_equal(cpu_leaves[j], gx.download(gpu_chunked[j]), j)) {
        std::printf("leaf %zu: chunked batches differ from OpenFHE\n", j);
        ++bad_leaves;
      }
      if (!leaf_equal(cpu_leaves[j], gx.download(gpu_single[j]), j)) {
        std::printf("leaf %zu: per-node splits differ from OpenFHE\n", j);
        ++bad_leaves;
      }
      if (!out_dir.empty()) {
        if (!Serial::SerializeToFile(out_dir + "/cpu_" + std::to_string(j) + ".bin", cpu_leaves[j],
                                     SerType::BINARY) ||
            !Serial::SerializeToFile(out_dir + "/gpu_" + std::to_string(j) + ".bin", gpu,
                                     SerType::BINARY)) {
          throw std::runtime_error("cannot write leaf files");
        }
      }
    }
    if (bad_leaves != 0) {
      std::printf("stage 4 expansion tree: FAIL (%zu of %zu leaves differ)\n", bad_leaves, d);
      ++g_failures;
    } else {
      std::printf("stage 4 expansion tree: PASS (%zu leaves)\n", d);
    }

    std::printf(
        "d=%zu depth=%zu leaves=%zu nodes=%zu automorphisms=%zu cpu_seconds=%.3f "
        "gpu_seconds=%.4f h2d_seconds=%.4f\n",
        d, stats.depth, gpu_dev.size(), stats.nodes, stats.automorphisms, cpu_seconds, gpu_seconds,
        h2d_seconds);
    if (g_failures != 0) {
      std::printf("GPU expansion exact RNS match with OpenFHE: FAIL (%d stages)\n", g_failures);
      return 1;
    }
    std::printf("GPU expansion exact RNS match with OpenFHE: PASS\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
