// ombra_bench_expand: GPU query expansion timing (T0.5).
//
// Usage: ombra_bench_expand <context.bin> <public_key.bin> <autokeys.bin> <d> <warmup> <iters>
//                           <out.json>
//
// Inputs are OpenFHE BINARY files written by core/bench/run.py: the CryptoContext (a preset from
// core/params), the PUBLIC key and the automorphism keys. No secret key is read. A seeded synthetic
// query (random slot values; the cost does not depend on the values) is encrypted with the public
// key. The timed work is the whole expansion tree on the GPU, d leaves, from the uploaded root to
// the last leaf, waiting for completion; context, key and monomial upload and the root upload are
// excluded and reported apart, as are the warm-up runs (which also fill the buffer pool). Correctness
// of the same code path is established by ombra_expand_gpu, not here. Writes one JSON object with
// sizes and timings only.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/column_search.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/expansion_tree.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;

namespace {
double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr,
                 "usage: ombra_bench_expand <context> <public_key> <autokeys> <d> <warmup> <iters> "
                 "<out.json>\n");
    return 2;
  }
  try {
    const std::size_t d = std::stoul(argv[4]);
    const int warmup = std::stoi(argv[5]);
    const int iters = std::stoi(argv[6]);
    if (d == 0 || warmup < 0 || iters < 1) {
      throw std::invalid_argument("d and iters must be positive");
    }
    CryptoContext<DCRTPoly> cc;
    PublicKey<DCRTPoly> pk;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    if (!Serial::DeserializeFromFile(argv[2], pk, SerType::BINARY)) {
      throw std::runtime_error("cannot read public key");
    }
    const std::size_t key_bytes = std::filesystem::file_size(argv[3]);
    std::ifstream keyfile(argv[3], std::ios::in | std::ios::binary);
    if (!keyfile.is_open() ||
        !CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(keyfile, SerType::BINARY)) {
      throw std::runtime_error("cannot read automorphism keys");
    }

    const std::size_t slots = cc->GetEncodingParams()->GetBatchSize();
    std::mt19937_64 rng(5);
    std::normal_distribution<double> normal(0.0, 1.0 / std::sqrt(static_cast<double>(d)));
    std::vector<double> values(slots);
    for (auto& v : values) v = normal(rng);
    const Ciphertext<DCRTPoly> query = cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(values));
    const auto& keys = CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(query->GetKeyTag());
    const std::size_t ring = cc->GetRingDimension();

    ombra::core::GpuExpansion gx(cc, query, keys, d);
    auto t0 = Clock::now();
    const ombra::core::GpuCt root = gx.upload(query);
    gx.synchronize();
    const double root_h2d = seconds_since(t0);

    ombra::core::ExpansionStats stats;
    for (int i = 0; i < warmup; ++i) {
      auto out = ombra::core::expand_tree(root, d, ring, gx);
      gx.synchronize();
    }
    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
      t0 = Clock::now();
      auto out = ombra::core::expand_tree(root, d, ring, gx, &stats);
      gx.synchronize();
      times.push_back(seconds_since(t0));
    }
    std::sort(times.begin(), times.end());
    const double median = (times.size() % 2 == 1)
                              ? times[times.size() / 2]
                              : 0.5 * (times[times.size() / 2 - 1] + times[times.size() / 2]);

    std::ofstream out(argv[7]);
    out.precision(9);
    out << "{\n"
        << "  \"d\": " << d << ",\n"
        << "  \"L\": " << stats.depth << ",\n"
        << "  \"nodes\": " << stats.nodes << ",\n"
        << "  \"automorphisms\": " << stats.automorphisms << ",\n"
        << "  \"keys\": " << gx.key_count() << ",\n"
        << "  \"key_bytes\": " << key_bytes << ",\n"
        << "  \"gpu_seconds\": " << median << ",\n"
        << "  \"gpu_seconds_min\": " << times.front() << ",\n"
        << "  \"gpu_seconds_max\": " << times.back() << ",\n"
        << "  \"iterations\": " << iters << ",\n"
        << "  \"warmup_iterations\": " << warmup << ",\n"
        << "  \"ring_dim\": " << ring << ",\n"
        << "  \"num_q_towers\": " << gx.tables().num_q << ",\n"
        << "  \"num_p_towers\": " << gx.tables().num_p << ",\n"
        << "  \"setup_seconds_tables_and_key_upload\": " << gx.setup_seconds() << ",\n"
        << "  \"copy_h2d_seconds_root\": " << root_h2d << ",\n"
        << "  \"gpu_name\": \"" << ombra::core::cuda_device_name() << "\",\n"
        << "  \"note\": \"whole expansion tree on the GPU, d leaves resident on the device; "
           "excludes key/root upload and warm-up; median of timed iterations\"\n"
        << "}\n";
    if (!out) throw std::runtime_error("cannot write output file");
    std::printf("expansion bench done: d=%zu gpu_seconds=%.4f\n", d, median);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
