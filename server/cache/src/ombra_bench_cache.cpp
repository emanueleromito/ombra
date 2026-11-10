// ombra_bench_cache: benchmark of the GPU cluster cache (T2.6).
//
// Usage: ombra_bench_cache <context.bin> <public_key.bin> <dims> <budget_blocks> <accesses_per_block>
//                          <out.json>
//
// Inputs are OpenFHE BINARY files from core/bench/run.py (context from a preset in core/params and
// the PUBLIC key only; no secret key here). One synthetic ciphertext is encrypted only to learn the
// shape and moduli; the blocks themselves are seeded random words reduced modulo the tower primes
// (the kernel's speed does not depend on the values), so nothing real is stored or written.
//
// Measures, on the current GPU:
//   1. host-to-device bandwidth, pinned vs pageable memory, at several sizes (GB/s = 1e9 B/s)
//   2. search over a collection of 3 * budget_blocks blocks with a GPU budget of budget_blocks
//      blocks and every block in the pinned pool: serial (copy, then compute, per block) vs
//      overlapped (copy of block i+1 during compute of block i), against compute only
//   3. cache hit rate for uniform and Zipf(s = 1) block popularity under LRU
// Writes one JSON object (sizes, timings, counters; no data). Exit code 0 ok, 2 error.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/cache/gpu_cache.hpp"
#include "ombra/cache/search_harness.hpp"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/column_search.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using ombra::cache::BlockKey;
using ombra::cache::CacheConfig;
using ombra::cache::CacheMetrics;
using ombra::cache::DeviceArray;
using ombra::cache::GpuClusterCache;
using ombra::cache::measure_h2d_gbps;
using ombra::cache::run_through_cache;
using ombra::cache::Stream;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

BlockKey key_of(std::size_t b) { return BlockKey{"bench", static_cast<std::uint32_t>(b)}; }

std::string metrics_json(const CacheMetrics& m) {
    std::ostringstream o;
    o << "{\"hits\": " << m.hits << ", \"misses\": " << m.misses
      << ", \"evictions\": " << m.evictions << ", \"bytes_paged\": " << m.bytes_paged
      << ", \"hit_rate\": " << m.hit_rate() << "}";
    return o.str();
}

// One timed run of `order` through a cold GPU cache (pinned pool warm). Returns seconds.
double timed_run(GpuClusterCache& cache, const ombra::core::ColumnSearchLauncher& launcher,
                 const ombra::core::ColumnSearchShape& shape, const DeviceArray& query,
                 const std::vector<BlockKey>& order, Stream& compute, DeviceArray& out,
                 bool serial) {
    cache.clear_gpu();
    cache.reset_metrics();
    const auto t0 = Clock::now();
    run_through_cache(cache, launcher, shape, query, order, compute, out, serial);
    return seconds_since(t0);
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
                     "usage: ombra_bench_cache <context> <public_key> <dims> <budget_blocks> "
                     "<accesses_per_block> <out.json>\n");
        return 2;
    }
    try {
        const std::size_t dims = std::stoul(argv[3]);
        const std::size_t budget_blocks = std::stoul(argv[4]);
        const std::size_t per_block = std::stoul(argv[5]);
        if (dims == 0 || budget_blocks < 3 || per_block == 0) {
            throw std::invalid_argument("dims > 0, budget_blocks >= 3 and accesses_per_block > 0");
        }
        const std::size_t blocks = 3 * budget_blocks;

        CryptoContext<DCRTPoly> cc;
        PublicKey<DCRTPoly> pk;
        if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
            throw std::runtime_error("cannot read crypto context");
        }
        ombra::core::require_security_floor(cc);
        if (!Serial::DeserializeFromFile(argv[2], pk, SerType::BINARY)) {
            throw std::runtime_error("cannot read public key");
        }
        const std::size_t slots = cc->GetEncodingParams()->GetBatchSize();
        // One synthetic ciphertext only for the shape and the moduli.
        const auto sample = ombra::core::flatten_ciphertexts(
            {cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, 0.5)))});
        const std::vector<std::uint64_t> moduli = sample.moduli;
        const ombra::core::ColumnSearchShape shape{dims, 1, sample.num_towers, sample.ring_dim};
        const std::size_t block_words = shape.block_words();
        const std::size_t block_bytes = block_words * sizeof(std::uint64_t);
        const std::size_t out_words = 3 * shape.poly_words();

        // 1. Bandwidth, pinned vs pageable.
        std::ostringstream bw;
        const std::size_t mib = 1u << 20;
        bool first_bw = true;
        for (std::size_t size_mib : {1u, 16u, 64u, 256u}) {
            const double pinned = measure_h2d_gbps(size_mib * mib, true, 10);
            const double pageable = measure_h2d_gbps(size_mib * mib, false, 10);
            bw << (first_bw ? "" : ",\n    ") << "{\"mib\": " << size_mib
               << ", \"pinned_gbps\": " << pinned << ", \"pageable_gbps\": " << pageable << "}";
            first_bw = false;
            std::printf("h2d %zu MiB: pinned %.2f GB/s, pageable %.2f GB/s\n", size_mib, pinned,
                        pageable);
        }

        // Synthetic loader: words reduced modulo the tower primes ([poly][tower][coeff] layout).
        ombra::cache::BlockLoader loader;
        loader.size_words = [&](const BlockKey&) { return block_words; };
        loader.load = [&](const BlockKey& key, std::uint64_t* dst, std::size_t words) {
            std::mt19937_64 rng(0x6f6d62726163ULL + key.segment);
            for (std::size_t i = 0; i < words; ++i) {
                const std::size_t chunk = i / shape.ring_dim;
                dst[i] = rng() % moduli[chunk % shape.num_towers];
            }
        };

        ombra::core::ColumnSearchLauncher launcher(shape, moduli);
        Stream compute;
        DeviceArray query(shape.query_words());
        {
            std::vector<std::uint64_t> q(shape.query_words());
            std::mt19937_64 rng(7);
            for (std::size_t i = 0; i < q.size(); ++i) {
                q[i] = rng() % moduli[(i / shape.ring_dim) % shape.num_towers];
            }
            query.upload(q.data(), q.size());
        }

        CacheConfig cfg;
        cfg.gpu_budget_bytes = budget_blocks * block_bytes;
        cfg.host_budget_bytes = blocks * block_bytes;  // the whole collection stays in the pool
        cfg.lookahead = 1;
        GpuClusterCache cache(cfg, loader);

        // Warm the pinned pool: every block is loaded once, not timed.
        std::vector<BlockKey> all_blocks;
        for (std::size_t b = 0; b < blocks; ++b) all_blocks.push_back(key_of(b));
        DeviceArray warm_out(all_blocks.size() * out_words);
        run_through_cache(cache, launcher, shape, query, all_blocks, compute, warm_out);
        const CacheMetrics warm = cache.metrics();

        // 2. Serial vs overlapped, random order, per_block passes of the collection.
        std::mt19937_64 rng(12345);
        std::vector<BlockKey> order;
        for (std::size_t p = 0; p < per_block; ++p) {
            std::vector<std::size_t> perm(blocks);
            for (std::size_t b = 0; b < blocks; ++b) perm[b] = b;
            std::shuffle(perm.begin(), perm.end(), rng);
            for (std::size_t b : perm) order.push_back(key_of(b));
        }
        DeviceArray out(order.size() * out_words);
        const int reps = 5;
        std::vector<double> serial_s, overlap_s;
        CacheMetrics serial_m, overlap_m;
        for (int r = 0; r < reps; ++r) {
            serial_s.push_back(timed_run(cache, launcher, shape, query, order, compute, out, true));
            serial_m = cache.metrics();
            overlap_s.push_back(timed_run(cache, launcher, shape, query, order, compute, out, false));
            overlap_m = cache.metrics();
        }
        // Compute only: the same number of launches on one resident block.
        double compute_only_s = 0;
        {
            DeviceArray resident(block_words);
            std::vector<std::uint64_t> w(block_words);
            loader.load(key_of(0), w.data(), block_words);
            resident.upload(w.data(), w.size());
            DeviceArray o(out_words);
            std::vector<double> ts;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = Clock::now();
                for (std::size_t i = 0; i < order.size(); ++i) {
                    launcher.launch(resident.get(), query.get(), o.get(), compute.get());
                }
                compute.synchronize();
                ts.push_back(seconds_since(t0));
            }
            compute_only_s = median(ts);
        }
        const double serial_med = median(serial_s), overlap_med = median(overlap_s);
        std::printf("search %zu accesses: serial %.3f s, overlapped %.3f s, compute only %.3f s\n",
                    order.size(), serial_med, overlap_med, compute_only_s);

        // 3. Hit rate, uniform and Zipf(1) popularity under LRU (cold GPU, warm pool).
        const std::size_t accesses = per_block * blocks * 4;
        std::vector<double> weights(blocks);
        std::vector<std::size_t> rank_to_block(blocks);
        for (std::size_t b = 0; b < blocks; ++b) rank_to_block[b] = b;
        std::shuffle(rank_to_block.begin(), rank_to_block.end(), rng);
        for (std::size_t r = 0; r < blocks; ++r) weights[r] = 1.0 / static_cast<double>(r + 1);
        std::discrete_distribution<std::size_t> zipf(weights.begin(), weights.end());
        std::uniform_int_distribution<std::size_t> uniform(0, blocks - 1);
        std::vector<BlockKey> uni_order, zipf_order;
        for (std::size_t i = 0; i < accesses; ++i) {
            uni_order.push_back(key_of(uniform(rng)));
            zipf_order.push_back(key_of(rank_to_block[zipf(rng)]));
        }
        DeviceArray hr_out(accesses * out_words);
        const double uni_s = timed_run(cache, launcher, shape, query, uni_order, compute, hr_out, false);
        const CacheMetrics uni_m = cache.metrics();
        const double zipf_s = timed_run(cache, launcher, shape, query, zipf_order, compute, hr_out, false);
        const CacheMetrics zipf_m = cache.metrics();
        std::printf("hit rate: uniform %.3f, zipf %.3f\n", uni_m.hit_rate(), zipf_m.hit_rate());

        std::ofstream o(argv[6]);
        o.precision(9);
        const double paged_bytes_overlap = static_cast<double>(overlap_m.bytes_paged);
        o << "{\n"
          << "  \"dims\": " << dims << ",\n"
          << "  \"num_towers\": " << shape.num_towers << ",\n"
          << "  \"ring_dim\": " << shape.ring_dim << ",\n"
          << "  \"block_bytes\": " << block_bytes << ",\n"
          << "  \"blocks\": " << blocks << ",\n"
          << "  \"gpu_budget_blocks\": " << budget_blocks << ",\n"
          << "  \"collection_to_budget\": 3,\n"
          << "  \"gpu_budget_bytes\": " << cfg.gpu_budget_bytes << ",\n"
          << "  \"host_pool_bytes\": " << cfg.host_budget_bytes << ",\n"
          << "  \"lookahead\": " << cfg.lookahead << ",\n"
          << "  \"warmup_pool_loads\": " << warm.host_loads << ",\n"
          << "  \"h2d_bandwidth\": [\n    " << bw.str() << "\n  ],\n"
          << "  \"search\": {\n"
          << "    \"accesses\": " << order.size() << ",\n"
          << "    \"repetitions\": " << reps << ",\n"
          << "    \"serial_seconds_median\": " << serial_med << ",\n"
          << "    \"overlapped_seconds_median\": " << overlap_med << ",\n"
          << "    \"compute_only_seconds_median\": " << compute_only_s << ",\n"
          << "    \"overlap_speedup\": " << serial_med / overlap_med << ",\n"
          << "    \"paging_gbps_overlapped\": " << paged_bytes_overlap / overlap_med / 1e9 << ",\n"
          << "    \"paging_gbps_serial\": " << static_cast<double>(serial_m.bytes_paged) / serial_med / 1e9 << ",\n"
          << "    \"serial_metrics\": " << metrics_json(serial_m) << ",\n"
          << "    \"overlapped_metrics\": " << metrics_json(overlap_m) << "\n"
          << "  },\n"
          << "  \"hit_rate\": {\n"
          << "    \"accesses\": " << accesses << ",\n"
          << "    \"uniform\": " << metrics_json(uni_m) << ",\n"
          << "    \"uniform_seconds\": " << uni_s << ",\n"
          << "    \"zipf_s1\": " << metrics_json(zipf_m) << ",\n"
          << "    \"zipf_seconds\": " << zipf_s << "\n"
          << "  },\n"
          << "  \"gpu_name\": \"" << ombra::core::cuda_device_name() << "\",\n"
          << "  \"note\": \"synthetic blocks (seeded random words); search = column search kernel, "
             "3-component sums, no relinearization; gpu cache starts cold for every timed run, pinned "
             "pool warm; paging_gbps = bytes copied host to device / wall time of the whole search\"\n"
          << "}\n";
        if (!o) throw std::runtime_error("cannot write output file");
        std::printf("cache bench done: dims=%zu blocks=%zu\n", dims, blocks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
