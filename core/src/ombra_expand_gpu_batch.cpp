// ombra_expand_gpu_batch: batched GPU query expansion checked exactly against OpenFHE (T0.7).
//
// Usage: ombra_expand_gpu_batch <context.bin> <autokeys.bin> <d> <queries> <dir>
//
// Inputs are OpenFHE BINARY serializations of public material only: the context, the automorphism
// keys, and in <dir> the query ciphertexts query_<q>.bin (q < queries). No secret key is read.
// The trees of all queries run together on the GPU (expand_trees: every level is one batched
// split_level over the nodes of all queries), once with the default chunk size and once with a
// tiny one (3 nodes per chunk, so chunks mix nodes of different queries and the last is ragged).
// Each query's tree also runs alone on the OpenFHE CPU path. Every leaf of every query must be
// identical in every RNS word, modulus, level, noise-scale degree and scaling factor. Writes
// cpu_<q>_<j>.bin and gpu_<q>_<j>.bin to <dir> for the decryption check in the Python test.
// Prints PASS/FAIL lines and timings only, never data values. Exit code: 0 exact, 1 mismatch,
// 2 error.

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
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "openfhe_expand_ops.hpp"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;
using Ct = Ciphertext<DCRTPoly>;

namespace {

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

bool leaf_equal(const Ct& cpu, const Ct& gpu) {
  const auto a = ombra::core::flatten_ciphertexts({cpu});
  const auto b = ombra::core::flatten_ciphertexts({gpu});
  return a.moduli == b.moduli && a.num_towers == b.num_towers && a.ring_dim == b.ring_dim &&
         a.level == b.level && a.noise_scale_deg == b.noise_scale_deg &&
         a.scaling_factor == b.scaling_factor && a.words == b.words;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr,
                 "usage: ombra_expand_gpu_batch <context> <autokeys> <d> <queries> <dir>\n");
    return 2;
  }
  try {
    const std::size_t d = std::stoul(argv[3]);
    const std::size_t nq = std::stoul(argv[4]);
    const std::string dir = argv[5];
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    ombra::core::require_security_floor(cc);
    std::ifstream keyfile(argv[2], std::ios::in | std::ios::binary);
    if (!keyfile.is_open() ||
        !CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(keyfile, SerType::BINARY)) {
      throw std::runtime_error("cannot read automorphism keys");
    }
    std::vector<Ct> queries(nq);
    for (std::size_t q = 0; q < nq; ++q) {
      if (!Serial::DeserializeFromFile(dir + "/query_" + std::to_string(q) + ".bin", queries[q],
                                       SerType::BINARY)) {
        throw std::runtime_error("cannot read query ciphertext");
      }
    }
    const auto& keys =
        CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(queries[0]->GetKeyTag());
    const std::size_t ring = cc->GetRingDimension();

    ombra::core::GpuExpansion gx(cc, queries[0], keys, d);
    std::vector<ombra::core::GpuCt> roots;
    for (const Ct& c : queries) roots.push_back(gx.upload(c));

    ombra::core::cpu_ref::OpenFheOps cpu_ops(cc, keys);
    std::vector<std::vector<Ct>> cpu(nq);
    auto t0 = Clock::now();
    for (std::size_t q = 0; q < nq; ++q) {
      cpu[q] = ombra::core::expand_tree(queries[q], d, ring, cpu_ops);
    }
    const double cpu_seconds = seconds_since(t0);

    ombra::core::ExpansionStats stats;
    t0 = Clock::now();
    auto gpu = ombra::core::expand_trees(roots, d, ring, gx, &stats);
    gx.synchronize();
    const double gpu_seconds = seconds_since(t0);

    const std::size_t default_batch = gx.device().max_batch();
    gx.device().set_max_batch(3);
    auto gpu_chunked = ombra::core::expand_trees(roots, d, ring, gx);
    gx.device().set_max_batch(default_batch);
    gx.synchronize();

    std::size_t bad = 0;
    if (gpu.size() != nq || gpu_chunked.size() != nq) ++bad;
    for (std::size_t q = 0; q < nq && bad == 0; ++q) {
      if (gpu[q].size() != d || gpu_chunked[q].size() != d) {
        ++bad;
        break;
      }
      for (std::size_t j = 0; j < d; ++j) {
        const Ct g = gx.download(gpu[q][j]);
        if (!leaf_equal(cpu[q][j], g)) {
          std::printf("query %zu leaf %zu: batched GPU differs from OpenFHE\n", q, j);
          ++bad;
        }
        if (!leaf_equal(cpu[q][j], gx.download(gpu_chunked[q][j]))) {
          std::printf("query %zu leaf %zu: chunked batched GPU differs from OpenFHE\n", q, j);
          ++bad;
        }
        const std::string tag = std::to_string(q) + "_" + std::to_string(j) + ".bin";
        if (!Serial::SerializeToFile(dir + "/cpu_" + tag, cpu[q][j], SerType::BINARY) ||
            !Serial::SerializeToFile(dir + "/gpu_" + tag, g, SerType::BINARY)) {
          throw std::runtime_error("cannot write leaf files");
        }
      }
    }
    std::printf("d=%zu queries=%zu depth=%zu automorphisms=%zu cpu_seconds=%.3f gpu_seconds=%.4f\n",
                d, nq, stats.depth, stats.automorphisms, cpu_seconds, gpu_seconds);
    if (bad != 0) {
      std::printf("batched GPU expansion exact RNS match with OpenFHE: FAIL (%zu)\n", bad);
      return 1;
    }
    std::printf(
        "batched GPU expansion exact RNS match with OpenFHE: PASS (%zu queries x %zu "
        "leaves)\n",
        nq, d);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
