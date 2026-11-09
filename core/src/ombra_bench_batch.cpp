// ombra_bench_batch: query batching benchmark for search and expansion (T0.7).
//
// Usage: ombra_bench_batch <context.bin> <public_key.bin> <autokeys.bin> <d> <blocks> <warmup>
//                          <iters> <mem_budget_bytes> <queries_csv> <out.json> [max_per_pass]
//
// Inputs are OpenFHE BINARY files written by core/bench/run.py: the context (a preset from
// core/params), the PUBLIC key and the automorphism keys. No secret key is read. Synthetic seeded
// data only (the cost of both workloads does not depend on the values): one stored block of d
// column ciphertexts replicated on the device to `blocks` blocks, one pre-expanded query replicated
// to as many query slots as a pass needs, and one expansion root per query of a pass.
//
// For every Q in <queries_csv> it measures
//   search:    Q queries scored against the stored blocks (batched kernel), and
//   expansion: the trees of Q queries (expand_trees), with the stored blocks resident.
// MEMORY PLAN. The device holds the stored blocks, the process baseline (CUDA context, keys,
// OpenFHE tables) and, per pass, Q_pass queries. One expanded query (d ciphertexts, search) or the
// nodes of one tree (expansion) is large, so the number of queries per pass is capped:
//   search:    per query = query words + result words;
//              cap = (budget - used - stored) / per query, where used = bytes in use at start.
//   expansion: per query = (expansion_peak_nodes(d) + 1 root) ciphertexts (a level's parents live
//              until the whole level is split); one chunk of scratch is reserved first.
// When Q exceeds the cap the work runs as consecutive passes of at most `cap` queries (results of
// a pass are dropped before the next, nothing accumulates) and the time counted is the total wall
// time for all Q queries. Peak device memory is read with cudaMemGetInfo after the first pass:
// buffers are never returned during a case (the pool keeps its peak, the stored blocks and scratch
// are static), so the reading is the peak of the case. It is recorded next to the budget.
// Timing: median of `iters` runs after `warmup` runs. Search: wall clock around all passes (each
// pass waits for completion) and the sum of the CUDA-event kernel times. Expansion: wall clock
// around all passes, synchronized after every pass.
// Writes one JSON object with sizes, memory and timings only; no data, no keys.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
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
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;
using ombra::core::BatchedColumnSearchDevice;
using ombra::core::ColumnSearchShape;
using ombra::core::GpuCt;

namespace {

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return (v.size() % 2 == 1) ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

std::size_t used_bytes() {
  const auto m = ombra::core::cuda_memory_info();
  return m.total_bytes - m.free_bytes;
}

ombra::core::FlatCiphertexts make_ciphertexts(const CryptoContext<DCRTPoly>& cc,
                                              const PublicKey<DCRTPoly>& pk, std::size_t count,
                                              std::size_t slots, std::size_t dims, bool constant,
                                              std::uint32_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0 / std::sqrt(static_cast<double>(dims)));
  std::vector<Ciphertext<DCRTPoly>> cts;
  cts.reserve(count);
  for (std::size_t j = 0; j < count; ++j) {
    std::vector<double> v(slots);
    if (constant) {
      std::fill(v.begin(), v.end(), normal(rng));
    } else {
      for (auto& x : v) x = normal(rng);
    }
    cts.push_back(cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(v)));
  }
  return ombra::core::flatten_ciphertexts(cts);
}

std::vector<std::size_t> parse_list(const std::string& s) {
  std::vector<std::size_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoul(item));
  if (out.empty()) throw std::invalid_argument("empty query list");
  return out;
}

std::string json_vec(const std::vector<double>& v) {
  std::ostringstream o;
  o.precision(9);
  o << "[";
  for (std::size_t i = 0; i < v.size(); ++i) o << (i ? ", " : "") << v[i];
  o << "]";
  return o.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 11 && argc != 12) {
    std::fprintf(stderr,
                 "usage: ombra_bench_batch <context> <public_key> <autokeys> <d> <blocks> <warmup> "
                 "<iters> <mem_budget_bytes> <queries_csv> <out.json> [max_per_pass]\n");
    return 2;
  }
  try {
    const std::size_t d = std::stoul(argv[4]);
    const std::size_t blocks = std::stoul(argv[5]);
    const int warmup = std::stoi(argv[6]);
    const int iters = std::stoi(argv[7]);
    const std::size_t budget = std::stoull(argv[8]);
    const std::vector<std::size_t> qlist = parse_list(argv[9]);
    // Optional extra cap on queries per pass (tests use it to exercise the multi-pass path).
    const std::size_t pass_arg = argc == 12 ? std::stoul(argv[11]) : 0;
    const std::size_t pass_limit = pass_arg == 0 ? SIZE_MAX : pass_arg;
    if (d == 0 || blocks == 0 || warmup < 0 || iters < 1) {
      throw std::invalid_argument("d, blocks and iters must be positive");
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
    std::ifstream keyfile(argv[3], std::ios::in | std::ios::binary);
    if (!keyfile.is_open() ||
        !CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(keyfile, SerType::BINARY)) {
      throw std::runtime_error("cannot read automorphism keys");
    }
    const std::size_t slots = cc->GetEncodingParams()->GetBatchSize();
    const std::size_t ring = cc->GetRingDimension();

    // Host data: one stored block, one pre-expanded query (search), one expansion root.
    const auto block = make_ciphertexts(cc, pk, d, slots, d, false, 1);
    auto expanded_query = make_ciphertexts(cc, pk, d, slots, d, true, 2);
    const ColumnSearchShape shape{d, blocks, block.num_towers, block.ring_dim};
    std::mt19937_64 rng(5);
    std::normal_distribution<double> normal(0.0, 1.0 / std::sqrt(static_cast<double>(d)));
    std::vector<double> values(slots);
    for (auto& v : values) v = normal(rng);
    const Ciphertext<DCRTPoly> root_ct = cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(values));
    const auto& keys = CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(root_ct->GetKeyTag());

    const std::size_t baseline_start = used_bytes();  // CUDA context only, nothing of ours yet
    const auto mem0 = ombra::core::cuda_memory_info();
    const std::size_t ct_bytes_host = 2 * block.num_towers * block.ring_dim * sizeof(std::uint64_t);
    const std::size_t stored_bytes = shape.stored_words() * sizeof(std::uint64_t);

    // ---------------------------------------------------------------- search
    std::ostringstream search_json;
    search_json << "[";
    for (std::size_t qi = 0; qi < qlist.size(); ++qi) {
      const std::size_t total_q = qlist[qi];
      const std::size_t used_before = used_bytes();
      const std::size_t per_query =
          (shape.query_words() + shape.out_words()) * sizeof(std::uint64_t);
      std::printf("phase search d=%zu Q=%zu: planning (used %.2f GiB, per query %.3f GiB)\n", d,
                  total_q, static_cast<double>(used_before) / 1073741824.0,
                  static_cast<double>(per_query) / 1073741824.0);
      std::fflush(stdout);
      if (budget <= used_before + stored_bytes + per_query) {
        throw std::runtime_error("memory budget too small for the stored blocks and one query");
      }
      const std::size_t cap =
          std::min(pass_limit, (budget - used_before - stored_bytes) / per_query);
      const std::size_t qpass = std::min(total_q, cap);
      const std::size_t passes = (total_q + qpass - 1) / qpass;
      const std::size_t last_pass = total_q - (passes - 1) * qpass;

      std::unique_ptr<BatchedColumnSearchDevice> dev;
      {
        dev = std::make_unique<BatchedColumnSearchDevice>(shape, block.moduli, qpass);
        dev->upload_block(0, block.words.data(), block.words.size());
        for (std::size_t b = 1; b < blocks; ++b) dev->copy_block(0, b);
        dev->upload_query(0, expanded_query.words.data(), expanded_query.words.size());
        for (std::size_t q = 1; q < qpass; ++q) dev->copy_query(0, q);
      }
      const auto run_all = [&](double& kernel_sum) {
        kernel_sum = 0;
        const auto t0 = Clock::now();
        for (std::size_t p = 0; p < passes; ++p) {
          kernel_sum += dev->run(p + 1 == passes ? last_pass : qpass);
        }
        return seconds_since(t0);
      };
      double ks = 0;
      for (int i = 0; i < warmup; ++i) run_all(ks);
      std::vector<double> wall, kern_sum, full_pass;
      for (int i = 0; i < iters; ++i) {
        wall.push_back(run_all(ks));
        kern_sum.push_back(ks);
        full_pass.push_back(dev->run(qpass));  // one full pass alone, for the bandwidth figures
      }
      const std::size_t peak_used = used_bytes();
      const double wall_med = median_of(wall);
      const double pass_med = median_of(full_pass);
      const auto tr = dev->traffic(qpass);
      const double loaded = static_cast<double>(tr.stored_load_bytes + tr.query_load_bytes);
      const double stored_wr = static_cast<double>(tr.store_bytes);
      const double mulmods = 4.0 * static_cast<double>(shape.poly_words()) *
                             static_cast<double>(d) * static_cast<double>(blocks) *
                             static_cast<double>(qpass);
      search_json.precision(9);
      search_json << (qi ? ",\n    " : "\n    ") << "{\"queries\": " << total_q
                  << ", \"queries_per_pass\": " << qpass << ", \"passes\": " << passes
                  << ", \"last_pass_queries\": " << last_pass
                  << ", \"mem_cap_queries_per_pass\": " << cap << ", \"stored_blocks\": " << blocks
                  << ", \"stored_vectors\": " << blocks * slots
                  << ", \"stored_bytes\": " << stored_bytes
                  << ", \"query_bytes_each\": " << shape.query_words() * sizeof(std::uint64_t)
                  << ", \"result_bytes_each\": " << shape.out_words() * sizeof(std::uint64_t)
                  << ", \"peak_device_bytes_used\": " << peak_used
                  << ", \"budget_bytes\": " << budget
                  << ", \"within_budget\": " << (peak_used <= budget ? "true" : "false")
                  << ", \"total_seconds_median\": " << wall_med
                  << ", \"total_seconds_min\": " << *std::min_element(wall.begin(), wall.end())
                  << ", \"total_seconds_max\": " << *std::max_element(wall.begin(), wall.end())
                  << ", \"kernel_seconds_sum_median\": " << median_of(kern_sum)
                  << ", \"queries_per_second\": " << static_cast<double>(total_q) / wall_med
                  << ", \"vectors_scored_per_second\": "
                  << static_cast<double>(total_q) * static_cast<double>(blocks * slots) / wall_med
                  << ", \"full_pass_kernel_seconds_median\": " << pass_med
                  << ", \"full_pass_all_samples_seconds\": " << json_vec(full_pass)
                  << ", \"pass_bytes_loaded\": " << loaded
                  << ", \"pass_bytes_loaded_stored_blocks\": " << tr.stored_load_bytes
                  << ", \"pass_bytes_loaded_queries\": " << tr.query_load_bytes
                  << ", \"pass_bytes_written\": " << stored_wr
                  << ", \"pass_bytes_compulsory\": " << tr.compulsory_bytes
                  << ", \"achieved_gb_per_s_access_pattern\": "
                  << (loaded + stored_wr) / pass_med / 1e9 << ", \"achieved_gb_per_s_compulsory\": "
                  << static_cast<double>(tr.compulsory_bytes) / pass_med / 1e9
                  << ", \"modmuls_per_second\": " << mulmods / pass_med << "}";
      dev.reset();
    }
    search_json << "\n  ]";

    // ------------------------------------------------------------- expansion
    // The stored blocks stay on the device (one query slot) while the trees run.
    std::vector<std::uint64_t>().swap(expanded_query.words);
    auto resident = std::make_unique<BatchedColumnSearchDevice>(shape, block.moduli, 1);
    resident->upload_block(0, block.words.data(), block.words.size());
    for (std::size_t b = 1; b < blocks; ++b) resident->copy_block(0, b);

    ombra::core::GpuExpansion gx(cc, root_ct, keys, d);
    const std::size_t ct_bytes = gx.device().ct_bytes();
    if (ct_bytes != ct_bytes_host) throw std::runtime_error("ciphertext size mismatch");
    const std::size_t peak_nodes = ombra::core::expansion_peak_nodes(d);
    const std::size_t scratch = gx.device().scratch_bytes(gx.device().max_batch());
    // Real footprint: ciphertexts live in slabs (one cudaMalloc each, rounded to the driver's
    // granularity), so a ciphertext costs slab_bytes / slab_slots, and the last slab may be nearly
    // empty, which is reserved as one more slab.
    const std::size_t slab_bytes = gx.device().slab_bytes();
    const std::size_t ct_footprint =
        (slab_bytes + gx.device().slab_slots() - 1) / gx.device().slab_slots();
    const std::size_t per_query_exp = (peak_nodes + 1) * ct_footprint;
    const std::size_t exp_fixed = scratch + slab_bytes;

    std::ostringstream exp_json;
    exp_json << "[";
    for (std::size_t qi = 0; qi < qlist.size(); ++qi) {
      const std::size_t total_q = qlist[qi];
      const std::size_t used_before = used_bytes();
      std::printf("phase expansion d=%zu Q=%zu: planning (used %.2f GiB, per query %.3f GiB)\n", d,
                  total_q, static_cast<double>(used_before) / 1073741824.0,
                  static_cast<double>(per_query_exp) / 1073741824.0);
      std::fflush(stdout);
      if (budget <= used_before + exp_fixed + per_query_exp) {
        throw std::runtime_error("memory budget too small for one expansion tree");
      }
      const std::size_t cap =
          std::min(pass_limit, (budget - used_before - exp_fixed) / per_query_exp);
      const std::size_t qpass = std::min(total_q, cap);
      const std::size_t passes = (total_q + qpass - 1) / qpass;
      const std::size_t last_pass = total_q - (passes - 1) * qpass;

      std::vector<GpuCt> roots;
      for (std::size_t q = 0; q < qpass; ++q) roots.push_back(gx.upload(root_ct));
      gx.synchronize();

      std::vector<double> pass_seconds;
      const auto run_all = [&]() {
        pass_seconds.clear();
        const auto t0 = Clock::now();
        for (std::size_t p = 0; p < passes; ++p) {
          const std::size_t n = p + 1 == passes ? last_pass : qpass;
          const auto tp = Clock::now();
          {
            const std::vector<GpuCt> in(roots.begin(), roots.begin() + static_cast<long>(n));
            auto leaves = ombra::core::expand_trees(in, d, ring, gx);
            gx.synchronize();  // leaves are dropped here, before the next pass
          }
          pass_seconds.push_back(seconds_since(tp));
        }
        return seconds_since(t0);
      };
      for (int i = 0; i < warmup; ++i) run_all();
      std::vector<double> wall, first_pass;
      for (int i = 0; i < iters; ++i) {
        wall.push_back(run_all());
        first_pass.push_back(pass_seconds.front());
      }
      // Pool and scratch keep their peak until released, so this reading is the peak of the case.
      const std::size_t peak_used = used_bytes();
      const std::size_t pool_at_peak = gx.device().pool_bytes();
      roots.clear();
      gx.release_pool();  // frees every slab: nothing carries over to the next Q
      const std::size_t pool_after_release = gx.device().pool_bytes();
      const std::size_t used_after_release = used_bytes();
      const double wall_med = median_of(wall);
      exp_json.precision(9);
      exp_json << (qi ? ",\n    " : "\n    ") << "{\"queries\": " << total_q
               << ", \"queries_per_pass\": " << qpass << ", \"passes\": " << passes
               << ", \"last_pass_queries\": " << last_pass
               << ", \"mem_cap_queries_per_pass\": " << cap << ", \"stored_blocks\": " << blocks
               << ", \"stored_vectors\": " << blocks * slots
               << ", \"stored_bytes\": " << stored_bytes
               << ", \"nodes_peak_per_query\": " << peak_nodes + 1
               << ", \"planned_bytes_per_query\": " << per_query_exp
               << ", \"ciphertext_nominal_bytes\": " << ct_bytes
               << ", \"ciphertext_footprint_bytes\": " << ct_footprint
               << ", \"slab_bytes\": " << slab_bytes << ", \"pool_bytes_at_peak\": " << pool_at_peak
               << ", \"pool_bytes_after_release\": " << pool_after_release
               << ", \"device_bytes_used_after_release\": " << used_after_release
               << ", \"scratch_bytes_reserved\": " << scratch
               << ", \"peak_device_bytes_used\": " << peak_used << ", \"budget_bytes\": " << budget
               << ", \"within_budget\": " << (peak_used <= budget ? "true" : "false")
               << ", \"total_seconds_median\": " << wall_med
               << ", \"total_seconds_min\": " << *std::min_element(wall.begin(), wall.end())
               << ", \"total_seconds_max\": " << *std::max_element(wall.begin(), wall.end())
               << ", \"queries_per_second\": " << static_cast<double>(total_q) / wall_med
               << ", \"latency_per_query_amortized_seconds\": "
               << wall_med / static_cast<double>(total_q)
               << ", \"first_pass_latency_seconds_median\": " << median_of(first_pass) << "}";
      std::printf("phase expansion d=%zu Q=%zu: done, %zu passes, pool %.2f -> %.2f GiB\n", d,
                  total_q, passes, static_cast<double>(pool_at_peak) / 1073741824.0,
                  static_cast<double>(pool_after_release) / 1073741824.0);
      std::fflush(stdout);
    }
    exp_json << "\n  ]";

    const auto mem_end = ombra::core::cuda_memory_info();
    std::ofstream out(argv[10]);
    out.precision(9);
    out << "{\n"
        << "  \"d\": " << d << ",\n"
        << "  \"L\": " << ombra::core::expansion_depth(d) << ",\n"
        << "  \"ring_dim\": " << ring << ",\n"
        << "  \"num_q_towers\": " << gx.tables().num_q << ",\n"
        << "  \"ciphertext_bytes\": " << ct_bytes << ",\n"
        << "  \"stored_blocks\": " << blocks << ",\n"
        << "  \"stored_vectors\": " << blocks * slots << ",\n"
        << "  \"warmup_iterations\": " << warmup << ",\n"
        << "  \"timed_iterations\": " << iters << ",\n"
        << "  \"device_total_bytes\": " << mem0.total_bytes << ",\n"
        << "  \"device_used_at_start_bytes\": " << baseline_start << ",\n"
        << "  \"device_used_at_end_bytes\": " << mem_end.total_bytes - mem_end.free_bytes << ",\n"
        << "  \"memory_budget_bytes\": " << budget << ",\n"
        << "  \"gpu_name\": \"" << ombra::core::cuda_device_name() << "\",\n"
        << "  \"search\": " << search_json.str() << ",\n"
        << "  \"expansion\": " << exp_json.str() << "\n"
        << "}\n";
    if (!out) throw std::runtime_error("cannot write output file");
    std::printf("batch bench done: d=%zu\n", d);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
