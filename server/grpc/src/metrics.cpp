#include "ombra/server/metrics.hpp"

#include <cstdio>

namespace ombra::server {

namespace {

std::string ms3(std::uint64_t us) {
  char buf[48];
  std::snprintf(buf, sizeof buf, "%.3f", static_cast<double>(us) / 1000.0);
  return buf;
}

}  // namespace

std::string metrics_json(const Metrics& m, const Gauges& g) {
  const auto u = [](std::uint64_t v) { return std::to_string(v); };
  std::string s = "{";
  s += "\"collections\":" + u(g.collections);
  s += ",\"vectors_live\":" + u(g.vectors_live);
  s += ",\"vectors_deleted\":" + u(g.vectors_deleted);
  s += ",\"blocks\":" + u(g.blocks);
  s += ",\"rpc\":{";
  for (std::size_t i = 0; i < kRpcCount; ++i) {
    const auto r = m.snapshot(static_cast<Rpc>(i));
    if (i > 0) s += ",";
    s += std::string("\"") + rpc_name(static_cast<Rpc>(i)) + "\":{\"requests\":" + u(r.requests) +
         ",\"errors\":" + u(r.errors) + ",\"latency_ms\":{\"sum\":" + ms3(r.latency_sum_us) +
         ",\"count\":" + u(r.latency_count) + ",\"max\":" + ms3(r.latency_max_us) + "}}";
  }
  s += "},\"auth\":{\"unauthenticated\":" + u(m.unauthenticated()) +
       ",\"permission_denied\":" + u(m.permission_denied()) + "}";
  s += ",\"gpu_memory\":{\"used_bytes\":" + u(g.gpu_used_bytes) +
       ",\"total_bytes\":" + u(g.gpu_total_bytes) + "}";
  const std::uint64_t req = g.cache_hits + g.cache_misses;
  char rate[48];
  std::snprintf(rate, sizeof rate, "%.4f",
                req == 0 ? 0.0 : static_cast<double>(g.cache_hits) / static_cast<double>(req));
  s += ",\"cache\":{\"hits\":" + u(g.cache_hits) + ",\"misses\":" + u(g.cache_misses) +
       ",\"evictions\":" + u(g.cache_evictions) + ",\"hit_rate\":" + rate + "}}";
  return s;
}

}  // namespace ombra::server
