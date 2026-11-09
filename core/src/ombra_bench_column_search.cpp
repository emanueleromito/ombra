// ombra_bench_column_search: throughput benchmark of the column-packed search kernel (T0.4).
//
// Usage: ombra_bench_column_search <context.bin> <public_key.bin> <dims> <blocks> <warmup> <iters>
//                                  <out.json>
//
// Inputs are OpenFHE BINARY files written by core/bench/run.py (or the tests): the CryptoContext
// (built from a preset in core/params) and the PUBLIC key only. This process never sees a secret
// key. It encrypts seeded synthetic data with the public key (normalized-scale random values; no
// real data), uploads ONE block of `dims` ciphertexts and the pre-expanded query (`dims`
// ciphertexts), replicates the block on the device to fill `blocks` blocks (the kernel's speed does
// not depend on the values), then times the kernel (CUDA events, waits for completion) for `iters`
// runs after `warmup` runs. Writes one JSON object to <out.json>; it contains sizes and timings
// only, no data or keys. No relinearization and no rescale are part of the measured work.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/column_search.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Encrypts `count` ciphertexts of `slots` seeded random values (scaled so a d-term dot product
// stays below 1) or, if `constant` is true, a constant value per ciphertext (a pre-expanded query).
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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
                     "usage: ombra_bench_column_search <context> <public_key> <dims> <blocks> "
                     "<warmup> <iters> <out.json>\n");
        return 2;
    }
    try {
        const std::size_t dims = std::stoul(argv[3]);
        const std::size_t blocks = std::stoul(argv[4]);
        const int warmup = std::stoi(argv[5]);
        const int iters = std::stoi(argv[6]);
        if (dims == 0 || blocks == 0 || warmup < 0 || iters < 1) {
            throw std::invalid_argument("dims, blocks and iters must be positive");
        }
        CryptoContext<DCRTPoly> cc;
        PublicKey<DCRTPoly> pk;
        if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
            throw std::runtime_error("cannot read crypto context");
        }
        if (!Serial::DeserializeFromFile(argv[2], pk, SerType::BINARY)) {
            throw std::runtime_error("cannot read public key");
        }
        const std::size_t slots = cc->GetEncodingParams()->GetBatchSize();

        ombra::core::ColumnSearchShape shape{};
        std::vector<std::uint64_t> moduli;
        double query_h2d = 0, block_h2d = 0;
        std::size_t block_bytes = 0;

        // Generate, flatten and upload one block, then free the host copy before the query.
        std::unique_ptr<ombra::core::ColumnSearchDevice> dev;
        {
            const auto block = make_ciphertexts(cc, pk, dims, slots, dims, false, 1);
            shape = {dims, blocks, block.num_towers, block.ring_dim};
            moduli = block.moduli;
            dev = std::make_unique<ombra::core::ColumnSearchDevice>(shape, moduli);
            const auto t0 = Clock::now();
            dev->upload_block(0, block.words.data(), block.words.size());
            block_h2d = seconds_since(t0);
            block_bytes = block.words.size() * sizeof(std::uint64_t);
        }
        {
            const auto query = make_ciphertexts(cc, pk, dims, slots, dims, true, 2);
            const auto t0 = Clock::now();
            dev->upload_query(query.words.data(), query.words.size());
            query_h2d = seconds_since(t0);
        }
        for (std::size_t b = 1; b < blocks; ++b) dev->copy_block(0, b);

        for (int i = 0; i < warmup; ++i) dev->run();
        std::vector<double> times;
        for (int i = 0; i < iters; ++i) times.push_back(dev->run());

        std::vector<std::uint64_t> result;
        const auto t0 = Clock::now();
        dev->download_result(result);
        const double d2h = seconds_since(t0);

        std::sort(times.begin(), times.end());
        const double median = (times.size() % 2 == 1)
                                  ? times[times.size() / 2]
                                  : 0.5 * (times[times.size() / 2 - 1] + times[times.size() / 2]);
        const std::size_t vectors = blocks * slots;
        const std::size_t device_bytes =
            (shape.stored_words() + shape.query_words() + shape.out_words()) * sizeof(std::uint64_t);

        std::ofstream out(argv[7]);
        out.precision(9);
        out << "{\n"
            << "  \"dims\": " << dims << ",\n"
            << "  \"blocks\": " << blocks << ",\n"
            << "  \"vectors_per_block\": " << slots << ",\n"
            << "  \"total_vectors_scored\": " << vectors << ",\n"
            << "  \"num_towers\": " << shape.num_towers << ",\n"
            << "  \"ring_dim\": " << shape.ring_dim << ",\n"
            << "  \"warmup_iterations\": " << warmup << ",\n"
            << "  \"timed_iterations\": " << iters << ",\n"
            << "  \"kernel_seconds_median\": " << median << ",\n"
            << "  \"kernel_seconds_min\": " << times.front() << ",\n"
            << "  \"kernel_seconds_max\": " << times.back() << ",\n"
            << "  \"vectors_per_second\": " << static_cast<double>(vectors) / median << ",\n"
            << "  \"copy_h2d_seconds_per_block\": " << block_h2d << ",\n"
            << "  \"copy_h2d_bytes_per_block\": " << block_bytes << ",\n"
            << "  \"copy_h2d_seconds_query\": " << query_h2d << ",\n"
            << "  \"copy_d2h_seconds_result\": " << d2h << ",\n"
            << "  \"copy_note\": \"h2d seconds include a host check that all words are reduced; "
               "blocks 1.. are filled by device-to-device copy of block 0 and are not timed\",\n"
            << "  \"device_memory_bytes\": " << device_bytes << ",\n"
            << "  \"gpu_name\": \"" << ombra::core::cuda_device_name() << "\",\n"
            << "  \"note\": \"relinearization and rescale are excluded; result has 3 components "
               "per block\"\n"
            << "}\n";
        if (!out) throw std::runtime_error("cannot write output file");
        std::printf("column search bench done: dims=%zu blocks=%zu\n", dims, blocks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
