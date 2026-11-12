// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// ombra_cache_search: paging test driver for the GPU cluster cache.
//
// Usage: ombra_cache_search <context.bin> <in_dir> <dims> <blocks> <seed>
//
// Inputs (OpenFHE BINARY files written by tests/test_gpu_cache_paging.py, public material only):
// the CryptoContext, <in_dir>/stored_<b>_<j>.bin (dimension j of block b) and
// <in_dir>/query_<j>.bin. A block is the `dims` stored ciphertexts of block b; the cache loader
// turns them into device words with the existing core code (flatten_ciphertexts). The driver
//   1. scores every block with everything resident on the GPU (reference),
//   2. scores all blocks twice in random orders through a cache whose GPU budget is one third of
//      the collection (and whose pinned pool holds two thirds, so the pool evicts too), and
//      compares every RNS word of every result with the reference (exact equality),
//   3. holds a block pinned while the others cycle through, then scores it and compares,
//   4. checks that a budget smaller than one block is refused with CacheError.
// The kernel is unchanged (its own OpenFHE oracle test is tests/test_gpu_column_search.py); this
// shows that paging does not corrupt data. Never reads a secret key. Prints counters, never data.
// Exit code: 0 all pass, 1 mismatch, 2 error.

#include <algorithm>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
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
using ombra::cache::DeviceArray;
using ombra::cache::GpuClusterCache;
using ombra::cache::Stream;

namespace {

const char* kCollection = "synthetic";

Ciphertext<DCRTPoly> load_ct(const std::string& path) {
  Ciphertext<DCRTPoly> ct;
  if (!Serial::DeserializeFromFile(path, ct, SerType::BINARY)) {
    throw std::runtime_error("cannot read ciphertext " + path);
  }
  return ct;
}

ombra::core::FlatCiphertexts load_block(const std::string& dir, std::size_t block,
                                        std::size_t dims) {
  std::vector<Ciphertext<DCRTPoly>> cts;
  for (std::size_t j = 0; j < dims; ++j) {
    cts.push_back(
        load_ct(dir + "/stored_" + std::to_string(block) + "_" + std::to_string(j) + ".bin"));
  }
  return ombra::core::flatten_ciphertexts(cts);
}

bool report(const char* what, bool ok) {
  std::printf("%s: %s\n", what, ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: ombra_cache_search <context> <in_dir> <dims> <blocks> <seed>\n");
    return 2;
  }
  try {
    const std::string dir = argv[2];
    const std::size_t dims = std::stoul(argv[3]);
    const std::size_t blocks = std::stoul(argv[4]);
    const std::uint64_t seed = std::stoull(argv[5]);
    if (blocks < 6) throw std::invalid_argument("need at least 6 blocks");

    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    ombra::core::require_security_floor(cc);

    // Reference data: flatten every block once on the host and the query.
    std::vector<Ciphertext<DCRTPoly>> query_cts;
    for (std::size_t j = 0; j < dims; ++j) {
      query_cts.push_back(load_ct(dir + "/query_" + std::to_string(j) + ".bin"));
    }
    const auto query = ombra::core::flatten_ciphertexts(query_cts);
    const auto first = load_block(dir, 0, dims);
    const ombra::core::ColumnSearchShape shape{dims, 1, first.num_towers, first.ring_dim};
    const std::vector<std::uint64_t>& moduli = first.moduli;
    const std::size_t block_words = shape.block_words();
    const std::size_t block_bytes = block_words * sizeof(std::uint64_t);
    const std::size_t out_words = 3 * shape.poly_words();
    if (first.words.size() != block_words || query.words.size() != shape.query_words()) {
      throw std::runtime_error("unexpected ciphertext shape");
    }

    std::vector<std::vector<std::uint64_t>> host_blocks(blocks);
    host_blocks[0] = first.words;
    for (std::size_t b = 1; b < blocks; ++b) {
      auto f = load_block(dir, b, dims);
      if (f.moduli != moduli || f.words.size() != block_words) {
        throw std::runtime_error("blocks differ in shape or moduli");
      }
      host_blocks[b] = std::move(f.words);
    }

    ombra::core::ColumnSearchLauncher launcher(shape, moduli);
    Stream compute;
    DeviceArray dev_query(shape.query_words());
    dev_query.upload(query.words.data(), query.words.size());

    // 1. Reference: everything resident (one device array for the whole collection).
    std::vector<std::uint64_t> reference(blocks * out_words);
    {
      DeviceArray all(blocks * block_words);
      DeviceArray out(blocks * out_words);
      for (std::size_t b = 0; b < blocks; ++b) {
        all.upload(host_blocks[b].data(), block_words, b * block_words);
      }
      for (std::size_t b = 0; b < blocks; ++b) {
        launcher.launch(all.get() + b * block_words, dev_query.get(), out.get() + b * out_words,
                        compute.get());
      }
      compute.synchronize();
      out.download(reference.data(), reference.size());
    }

    // The loader: the column files of the segment -> device words. Counts loads only.
    std::size_t loads = 0;
    ombra::cache::BlockLoader loader;
    loader.size_words = [&](const BlockKey&) { return block_words; };
    loader.load = [&](const BlockKey& key, std::uint64_t* dst, std::size_t words) {
      const auto f = load_block(dir, key.segment, dims);
      if (f.words.size() != words || f.moduli != moduli) {
        throw std::runtime_error("loader: block shape changed");
      }
      std::copy(f.words.begin(), f.words.end(), dst);
      ++loads;
    };

    bool ok = true;
    const auto key_of = [&](std::size_t b) {
      return BlockKey{kCollection, static_cast<std::uint32_t>(b)};
    };
    const auto equal_slot = [&](const std::vector<std::uint64_t>& got, std::size_t slot,
                                std::size_t block) {
      return std::equal(got.begin() + slot * out_words, got.begin() + (slot + 1) * out_words,
                        reference.begin() + block * out_words);
    };

    // 2. Two passes in random orders through a cache with 1/3 of the collection on the GPU.
    {
      CacheConfig cfg;
      cfg.gpu_budget_bytes = blocks * block_bytes / 3;
      cfg.host_budget_bytes = blocks * block_bytes * 2 / 3;
      cfg.lookahead = 1;
      GpuClusterCache cache(cfg, loader);

      std::mt19937_64 rng(seed);
      std::vector<std::size_t> p1(blocks), p2(blocks);
      for (std::size_t b = 0; b < blocks; ++b) p1[b] = p2[b] = b;
      std::shuffle(p1.begin(), p1.end(), rng);
      std::shuffle(p2.begin(), p2.end(), rng);
      // Start the second pass with the block the first pass ended on, so at least that
      // access is a hit under LRU (the other orders stay random).
      std::iter_swap(p2.begin(), std::find(p2.begin(), p2.end(), p1.back()));

      std::vector<std::size_t> idx(p1);
      idx.insert(idx.end(), p2.begin(), p2.end());
      std::vector<BlockKey> order;
      for (std::size_t b : idx) order.push_back(key_of(b));

      DeviceArray out(order.size() * out_words);
      ombra::cache::run_through_cache(cache, launcher, shape, dev_query, order, compute, out);
      std::vector<std::uint64_t> got(order.size() * out_words);
      out.download(got.data(), got.size());
      bool match = true;
      for (std::size_t i = 0; i < idx.size(); ++i) match = match && equal_slot(got, i, idx[i]);
      ok = report("cache paging exact match", match) && ok;

      const auto m = cache.metrics();
      std::printf(
          "cache metrics: hits=%llu misses=%llu evictions=%llu hit_rate=%.4f "
          "bytes_paged=%llu host_loads=%llu host_evictions=%llu\n",
          static_cast<unsigned long long>(m.hits), static_cast<unsigned long long>(m.misses),
          static_cast<unsigned long long>(m.evictions), m.hit_rate(),
          static_cast<unsigned long long>(m.bytes_paged),
          static_cast<unsigned long long>(m.host_loads),
          static_cast<unsigned long long>(m.host_evictions));
      std::printf("cache collection_bytes=%zu gpu_budget_bytes=%zu block_bytes=%zu\n",
                  blocks * block_bytes, cfg.gpu_budget_bytes, block_bytes);

      // 3. A block held by a lease is never evicted while the others cycle.
      cache.plan({key_of(0)});
      GpuClusterCache::Lease held = cache.next(compute);
      std::vector<std::size_t> rest;
      for (std::size_t b = 1; b < blocks; ++b) rest.push_back(b);
      std::shuffle(rest.begin(), rest.end(), rng);
      std::vector<BlockKey> rest_keys;
      for (std::size_t b : rest) rest_keys.push_back(key_of(b));
      DeviceArray out2(rest.size() * out_words);
      ombra::cache::run_through_cache(cache, launcher, shape, dev_query, rest_keys, compute, out2);
      DeviceArray out_held(out_words);
      launcher.launch(held.device_ptr(), dev_query.get(), out_held.get(), compute.get());
      held.release();
      compute.synchronize();
      std::vector<std::uint64_t> got2(rest.size() * out_words), got_held(out_words);
      out2.download(got2.data(), got2.size());
      out_held.download(got_held.data(), got_held.size());
      bool pin_ok = std::equal(got_held.begin(), got_held.end(), reference.begin());
      for (std::size_t i = 0; i < rest.size(); ++i) pin_ok = pin_ok && equal_slot(got2, i, rest[i]);
      ok = report("cache pinned block survives eviction pressure", pin_ok) && ok;

      const auto m2 = cache.metrics();
      const bool sane = m.evictions > 0 && m.hits > 0 && m.misses > 0 && m.hit_rate() > 0.0 &&
                        m.hit_rate() < 1.0 && m2.evictions >= m.evictions && loads > 0;
      ok = report("cache evictions happened and hit rate is in (0, 1)", sane) && ok;
    }

    // 4. A budget smaller than one block is refused with an error, not a crash.
    {
      CacheConfig cfg;
      cfg.gpu_budget_bytes = block_bytes / 2;
      cfg.host_budget_bytes = 2 * block_bytes;
      GpuClusterCache cache(cfg, loader);
      cache.plan({key_of(0)});
      bool refused = false;
      try {
        cache.next(compute);
      } catch (const ombra::cache::CacheError&) {
        refused = true;
      }
      ok = report("cache refuses a block larger than the GPU budget", refused) && ok;
    }

    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
