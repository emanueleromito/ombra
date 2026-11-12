// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Metrics registry: atomic counters only. It holds no collection name, id, score, vector or
// byte of data; the JSON output is built from these totals and from gauges the server computes
// (counts, GPU memory, cache counters).
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "ombra/server/auth.hpp"

namespace ombra::server {

class Metrics {
 public:
  struct RpcSnapshot {
    std::uint64_t requests = 0, errors = 0;
    std::uint64_t latency_count = 0, latency_sum_us = 0, latency_max_us = 0;
  };

  void record(Rpc rpc, bool error, double ms) {
    auto& r = rpcs_[static_cast<std::size_t>(rpc)];
    const auto us = static_cast<std::uint64_t>(ms < 0 ? 0 : ms * 1000.0);
    r.requests.fetch_add(1, std::memory_order_relaxed);
    if (error) r.errors.fetch_add(1, std::memory_order_relaxed);
    r.latency_count.fetch_add(1, std::memory_order_relaxed);
    r.latency_sum_us.fetch_add(us, std::memory_order_relaxed);
    std::uint64_t cur = r.latency_max_us.load(std::memory_order_relaxed);
    while (us > cur &&
           !r.latency_max_us.compare_exchange_weak(cur, us, std::memory_order_relaxed)) {
    }
  }
  void count_unauthenticated() { unauthenticated_.fetch_add(1, std::memory_order_relaxed); }
  void count_permission_denied() { denied_.fetch_add(1, std::memory_order_relaxed); }

  RpcSnapshot snapshot(Rpc rpc) const {
    const auto& r = rpcs_[static_cast<std::size_t>(rpc)];
    return {r.requests.load(), r.errors.load(), r.latency_count.load(), r.latency_sum_us.load(),
            r.latency_max_us.load()};
  }
  std::uint64_t unauthenticated() const { return unauthenticated_.load(); }
  std::uint64_t permission_denied() const { return denied_.load(); }

 private:
  struct PerRpc {
    std::atomic<std::uint64_t> requests{0}, errors{0}, latency_count{0}, latency_sum_us{0},
        latency_max_us{0};
  };
  PerRpc rpcs_[kRpcCount];
  std::atomic<std::uint64_t> unauthenticated_{0}, denied_{0};
};

// Values read at scrape time. Totals over all collections: no names.
struct Gauges {
  std::uint64_t collections = 0, vectors_live = 0, vectors_deleted = 0, blocks = 0;
  std::uint64_t gpu_used_bytes = 0, gpu_total_bytes = 0;
  std::uint64_t cache_hits = 0, cache_misses = 0, cache_evictions = 0;
};

// The body of GET /metrics. Keys: collections, vectors_live, vectors_deleted, blocks, rpc{<name>:
// {requests, errors, latency_ms{sum, count, max}}}, auth{unauthenticated, permission_denied},
// gpu_memory{used_bytes, total_bytes}, cache{hits, misses, evictions, hit_rate}.
std::string metrics_json(const Metrics& m, const Gauges& g);

}  // namespace ombra::server
