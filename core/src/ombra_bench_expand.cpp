// ombra_bench_expand: GPU query expansion timing.
//
// Usage: ombra_bench_expand <context.bin> <public_key.bin> <autokeys.bin> <d> <warmup> <iters>
//                           <out.json> [profile]
//
// Inputs are OpenFHE BINARY files written by core/bench/run.py: the CryptoContext (a preset from
// core/params), the PUBLIC key and the automorphism keys. No secret key is read. A seeded synthetic
// query (random slot values; the cost does not depend on the values) is encrypted with the public
// key. The timed work is the whole expansion tree on the GPU, d leaves, from the uploaded root to
// the last leaf, waiting for completion; context, key and monomial upload and the root upload are
// excluded and reported apart, as are the warm-up runs (which also fill the buffer pool).
// Correctness of the same code path is established by ombra_expand_gpu, not here. Writes one JSON
// object with sizes and timings only. With the optional last argument `profile`, also times the
// stages of one split (CUDA events between the stages; off the timed path) and adds a "profile"
// object.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/column_search.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/expansion_tree.hpp"
#include "ombra/core/security_floor_context.hpp"
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
  if (argc != 8 && argc != 9) {
    std::fprintf(stderr,
                 "usage: ombra_bench_expand <context> <public_key> <autokeys> <d> <warmup> <iters> "
                 "<out.json> [profile]\n");
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
    ombra::core::require_security_floor(cc);
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

    const bool profile = argc == 9 && std::string(argv[8]) == "profile";
    std::string profile_json;
    if (profile) {
      const auto galois = static_cast<std::uint32_t>(ring + 1);  // level 0
      const auto one_profile = [&](std::size_t nodes) {
        const ombra::core::SplitProfile pr =
            gx.device().profile_level(root.dev, galois, 0, nodes, 20);
        std::ostringstream o;
        o.precision(6);
        o << "{\"nodes_per_batch\": " << pr.nodes << ", \"reps\": 20, \"stage_us_per_batch\": {";
        double sum = 0;
        for (std::size_t i = 0; i < pr.stage.size(); ++i) {
          o << (i ? ", " : "") << "\"" << pr.stage[i] << "\": " << pr.stage_us[i];
          sum += pr.stage_us[i];
        }
        const double per = 1.0 / static_cast<double>(pr.nodes);
        o << "}, \"stage_sum_us_per_split\": " << sum * per
          << ", \"host_enqueue_us_per_split\": " << pr.host_enqueue_us * per
          << ", \"pipelined_us_per_split\": " << pr.pipelined_us * per
          << ", \"synced_us_per_split\": " << pr.synced_us * per
          << ", \"empty_kernel_launch_us\": " << pr.noop_launch_us << "}";
        return o.str();
      };
      // Per-level wall times of the real tree (synchronized after every level), median of 3.
      struct LevelTimer {
        ombra::core::GpuExpansion& g;
        std::vector<double> ms;
        std::vector<std::size_t> nodes;
        std::vector<std::pair<ombra::core::GpuCt, ombra::core::GpuCt>> split_level(
            const std::vector<ombra::core::GpuCt>& in, std::size_t level, std::uint32_t galois,
            std::size_t shift, const std::vector<char>& odd) {
          g.synchronize();
          const auto t = Clock::now();
          auto r = g.split_level(in, level, galois, shift, odd);
          g.synchronize();
          ms.push_back(1000.0 * seconds_since(t));
          nodes.push_back(in.size());
          return r;
        }
      };
      std::vector<std::vector<double>> level_runs;
      std::vector<std::size_t> level_nodes;
      for (int i = 0; i < 3; ++i) {
        LevelTimer lt{gx, {}, {}};
        auto leaves = ombra::core::expand_tree(root, d, ring, lt);
        level_runs.push_back(lt.ms);
        level_nodes = lt.nodes;
      }
      std::ostringstream lv;
      lv.precision(6);
      lv << "[";
      for (std::size_t l = 0; l < level_nodes.size(); ++l) {
        std::vector<double> v;
        for (const auto& run : level_runs) v.push_back(run[l]);
        std::sort(v.begin(), v.end());
        lv << (l ? ", " : "") << "{\"level\": " << l << ", \"nodes\": " << level_nodes[l]
           << ", \"ms\": " << v[v.size() / 2] << "}";
      }
      lv << "]";
      // Whole-tree time for several chunk sizes (smaller chunks keep the scratch in the L2 cache).
      std::ostringstream sw;
      sw.precision(6);
      sw << "[";
      const std::size_t default_batch = gx.device().max_batch();
      bool first_entry = true;
      for (std::size_t mb : {default_batch, std::size_t{128}, std::size_t{64}, std::size_t{32},
                             std::size_t{16}, std::size_t{8}, std::size_t{4}}) {
        if (mb > default_batch) continue;
        gx.device().set_max_batch(mb);
        std::vector<double> v;
        for (int i = 0; i < 3; ++i) {
          const auto ts = Clock::now();
          auto leaves = ombra::core::expand_tree(root, d, ring, gx);
          gx.synchronize();
          v.push_back(seconds_since(ts));
        }
        std::sort(v.begin(), v.end());
        sw << (first_entry ? "" : ", ") << "{\"max_batch\": " << mb << ", \"seconds\": " << v[1]
           << "}";
        first_entry = false;
      }
      sw << "]";
      gx.device().set_max_batch(default_batch);
      std::ostringstream o;
      const std::size_t wide = std::min<std::size_t>(d, gx.device().max_batch());
      o << "{\"single_split\": " << one_profile(1) << ", \"batched\": " << one_profile(wide)
        << ", \"max_batch\": " << gx.device().max_batch() << ", \"levels\": " << lv.str()
        << ", \"batch_sweep\": " << sw.str() << "}";
      profile_json = o.str();
    }

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
        << "  \"batched_levels\": true,\n"
        << "  \"max_batch_nodes\": " << gx.device().max_batch() << ",\n"
        << "  \"iterations\": " << iters << ",\n"
        << "  \"warmup_iterations\": " << warmup << ",\n"
        << "  \"ring_dim\": " << ring << ",\n"
        << "  \"num_q_towers\": " << gx.tables().num_q << ",\n"
        << "  \"num_p_towers\": " << gx.tables().num_p << ",\n"
        << "  \"setup_seconds_tables_and_key_upload\": " << gx.setup_seconds() << ",\n"
        << "  \"copy_h2d_seconds_root\": " << root_h2d << ",\n"
        << "  \"gpu_name\": \"" << ombra::core::cuda_device_name() << "\",\n"
        << (profile ? "  \"profile\": " + profile_json + ",\n" : std::string())
        << "  \"note\": \"whole expansion tree on the GPU (one batched pass per level), d leaves "
           "resident on the device; "
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
