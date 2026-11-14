// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// ombra_server --data-dir D --listen HOST:PORT [--openfhe-version V]
//                [--gpu-budget-mb N] [--host-budget-mb N] [--compact-interval-ms N]
//                [--tls-cert F --tls-key F] [--rest-port N [--rest-host H]]
//                [--insecure] [--no-auth]
//                [--max-message-mb N] [--memory-quota-mb N] [--max-threads N]
//                [--shutdown-grace-s N]
//                [--max-sessions N] [--max-sessions-per-key N] [--session-budget-mb N]
//                [--search-idle-s N] [--search-total-s N] [--max-rounds N]
//                [--max-collections N] [--max-cluster-count N] [--max-blobs N]
//                [--max-blob-mb N] [--max-key-upload-mb N] [--rest-workers N] [--rest-max-conns N]
// ombra_server --create-api-key --scope admin|read|write --data-dir D
// ombra_server --version
//
// The ombra.v1 gRPC server over the storage engine and the GPU search engine, with TLS,
// API keys and a REST admin port. A build with OMBRA_ENABLE_CUDA=OFF (tests and development only)
// has no GPU: every call works except Search, which answers with an error that says so, and
// /metrics reports gpu_available false.
//
// Safe defaults: the server refuses to start without TLS unless --insecure is given, and without an
// API key file (D/api_keys.v1, made with --create-api-key) unless --no-auth is given. Both flags
// log a loud warning and are limited to loopback addresses. An API key is an access token; the
// server never accepts a secret key and never logs a token.
//
// Limits (all flags, defaults in ombra/server/limits.hpp and below): a call is authenticated before
// its body is read (TLS), gRPC memory and threads are bounded by a ResourceQuota, idle connections
// are closed, Search streams are capped per key and globally and have idle and total deadlines.
// SIGHUP re-reads the API key file (revoke a key without a restart).
//
// Every --compact-interval-ms (default 60000, 0 disables) a background thread drops, in every
// collection, the blocks whose vectors are all deleted; it logs counts only.
//
// Prints "ombra_server listening on HOST:PORT" on stdout once the port is bound (PORT 0 picks a
// free port), then, with --rest-port, "ombra_server rest listening on HOST:PORT".
//
// Shutdown. On the first SIGTERM or SIGINT the server (1) reports not ready at once (GET /readyz
// answers 503; /healthz stays 200 while the process lives), (2) stops accepting new calls (they get
// UNAVAILABLE, which clients retry), (3) lets running calls finish for --shutdown-grace-s seconds
// (default 30), then cancels the rest (a cancelled call is not acknowledged and never stores the
// block it was still receiving), and (4) exits 0. A second SIGTERM or SIGINT skips the wait and
// exits 1 at once; every acknowledged write is already durable and recovery sweeps the rest.
#include <grpcpp/grpcpp.h>
#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "gpu_info.hpp"
#include "ombra/search/session.hpp"
#include "ombra/server/auth.hpp"
#include "ombra/server/log.hpp"
#include "ombra/server/limits.hpp"
#include "ombra/server/metrics.hpp"
#include "ombra/server/policy.hpp"
#include "ombra/storage/store.hpp"
#include "rest_server.hpp"
#include "service.hpp"

#ifndef OMBRA_VERSION
#define OMBRA_VERSION "unknown"
#endif

#ifndef OMBRA_OPENFHE_VERSION
#define OMBRA_OPENFHE_VERSION "unknown"
#endif

namespace {

std::atomic<bool> g_stop{false};
std::atomic<int> g_signals{0};  // SIGTERM/SIGINT received so far; the second one forces the exit
std::atomic<bool> g_reload{false};
extern "C" void on_signal(int) {
  g_signals.fetch_add(1);
  g_stop.store(true);
}
extern "C" void on_hup(int) { g_reload.store(true); }

using ombra::server::is_loopback_listen;

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Totals only, read at scrape time: no names, ids or data.
ombra::server::Gauges collect_gauges(ombra::storage::Store& store,
                                     ombra::search::SearchEngine& engine) {
  ombra::server::Gauges g;
  for (const auto& name : store.list_collections()) {
    ombra::storage::CollectionSpec spec;
    if (!store.get_collection(name, &spec)) continue;
    ++g.collections;
    for (const auto& b : store.block_stats(name)) {
      ++g.blocks;
      if (b.cluster == spec.cluster_count) continue;  // cluster-center block: not data vectors
      g.vectors_live += b.n_vectors - b.n_deleted;
      g.vectors_deleted += b.n_deleted;
    }
  }
  ombra::server::gpu_memory(&g.gpu_used_bytes, &g.gpu_total_bytes);
#ifdef OMBRA_NO_GPU
  g.gpu_available = false;
#endif
  const auto cm = engine.cache_metrics();
  g.cache_hits = cm.hits;
  g.cache_misses = cm.misses;
  g.cache_evictions = cm.evictions;
  return g;
}

// gRPC channel arguments by name (the macro names differ between gRPC releases).
constexpr const char* kArgMaxConcurrentStreams = "grpc.max_concurrent_streams";
constexpr const char* kArgMaxConnectionIdleMs = "grpc.max_connection_idle_ms";
constexpr const char* kArgMaxConnectionAgeMs = "grpc.max_connection_age_ms";
constexpr const char* kArgMaxConnectionAgeGraceMs = "grpc.max_connection_age_grace_ms";
constexpr const char* kArgKeepaliveTimeMs = "grpc.keepalive_time_ms";
constexpr const char* kArgKeepaliveTimeoutMs = "grpc.keepalive_timeout_ms";
constexpr const char* kArgKeepalivePermitWithoutCalls = "grpc.keepalive_permit_without_calls";
constexpr const char* kArgMinPingIntervalMs = "grpc.http2.min_ping_interval_without_data_ms";
constexpr const char* kArgMaxPingStrikes = "grpc.http2.max_ping_strikes";
constexpr const char* kArgHandshakeTimeoutMs = "grpc.server_handshake_timeout_ms";

}  // namespace

int main(int argc, char** argv) {
  std::string data_dir, listen, ofhe = OMBRA_OPENFHE_VERSION;
  std::string tls_cert, tls_key, scope_arg, rest_host, key_collection, log_format_arg;
  std::uint64_t gpu_mb = 4096, host_mb = 2048;
  std::uint64_t compact_ms = 60000;
  // The C++ API has one receive limit for the whole server. No message is larger than a column or
  // a 4 MiB chunk of a key or blob (Insert, ReplaceBlock, PutBlob and UploadKeys are streams of
  // parts), so 16 MiB holds the largest one with room: an unseeded column of the largest preset
  // is under 1 MiB, an id list of one block at most 2.2 MB. The limit is applied after
  // authentication, and the ResourceQuota bounds all buffered messages together.
  std::uint64_t max_message_mb = 16, memory_quota_mb = 2048, max_threads = 128;
  std::uint64_t shutdown_grace_s = 30;
  std::uint64_t session_budget_mb = 0, rest_workers = 4, rest_max_conns = 32;
  std::uint64_t max_blob_mb = 16384, max_key_upload_mb = 2048;
  std::uint64_t search_idle_s = 60, search_total_s = 600;
  ombra::server::ServiceLimits limits;
  int rest_port = -1;  // -1: no REST port
  bool insecure = false, no_auth = false, create_key = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    // Numeric flag: the whole value must be a decimal number in range, else the process exits with
    // a message (audit A-54). No exception can escape from here.
    const auto num = [&](std::uint64_t lo, std::uint64_t hi) -> std::uint64_t {
      const std::string v = next();
      std::uint64_t out = 0;
      if (!ombra::server::parse_uint(v, lo, hi, &out)) {
        std::fprintf(stderr, "invalid value for %s: expected a number from %llu to %llu\n",
                     a.c_str(), static_cast<unsigned long long>(lo),
                     static_cast<unsigned long long>(hi));
        std::exit(2);
      }
      return out;
    };
    if (a == "--version") {
      std::printf("ombra_server %s\n", OMBRA_VERSION);
      return 0;
    } else if (a == "--data-dir")
      data_dir = next();
    else if (a == "--listen")
      listen = next();
    else if (a == "--openfhe-version")
      ofhe = next();
    else if (a == "--gpu-budget-mb")
      gpu_mb = num(1, std::uint64_t{1} << 24);
    else if (a == "--host-budget-mb")
      host_mb = num(1, std::uint64_t{1} << 24);
    else if (a == "--compact-interval-ms")
      compact_ms = num(0, std::uint64_t{1} << 40);
    else if (a == "--max-message-mb")
      max_message_mb = num(8, 2000);  // at least a 4 MiB chunk plus its framing, with room
    else if (a == "--shutdown-grace-s")
      shutdown_grace_s = num(0, 86400);
    else if (a == "--memory-quota-mb")
      memory_quota_mb = num(64, std::uint64_t{1} << 24);
    else if (a == "--max-threads")
      max_threads = num(8, 100000);
    else if (a == "--session-budget-mb")
      session_budget_mb = num(0, std::uint64_t{1} << 24);
    else if (a == "--max-sessions")
      limits.max_sessions = num(1, 100000);
    else if (a == "--max-sessions-per-key")
      limits.max_sessions_per_key = num(1, 100000);
    else if (a == "--search-idle-s")
      search_idle_s = num(0, 86400);
    else if (a == "--search-total-s")
      search_total_s = num(0, 7 * 86400);
    else if (a == "--max-rounds")
      limits.max_rounds = num(1, 1000000);
    else if (a == "--max-collections")
      limits.max_collections = num(1, 10000000);
    else if (a == "--max-cluster-count")
      limits.max_cluster_count =
          static_cast<std::uint32_t>(num(1, ombra::server::kMaxClusterCount));
    else if (a == "--max-blobs")
      limits.max_blobs_per_collection = num(0, std::uint64_t{1} << 40);
    else if (a == "--max-blob-mb")
      max_blob_mb = num(0, std::uint64_t{1} << 30);
    else if (a == "--max-key-upload-mb")
      max_key_upload_mb = num(1, 4096);
    else if (a == "--rest-workers")
      rest_workers = num(1, 256);
    else if (a == "--rest-max-conns")
      rest_max_conns = num(1, 100000);
    else if (a == "--tls-cert")
      tls_cert = next();
    else if (a == "--tls-key")
      tls_key = next();
    else if (a == "--rest-port")
      rest_port = static_cast<int>(num(0, 65535));
    else if (a == "--rest-host")
      rest_host = next();
    else if (a == "--scope")
      scope_arg = next();
    else if (a == "--collection")
      key_collection = next();
    else if (a == "--log-format")
      log_format_arg = next();
    else if (a == "--create-api-key")
      create_key = true;
    else if (a == "--insecure")
      insecure = true;
    else if (a == "--no-auth")
      no_auth = true;
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }

  if (!log_format_arg.empty()) {
    ombra::server::log::Format lf;
    if (!ombra::server::log::parse_format(log_format_arg, &lf)) {
      std::fprintf(stderr, "invalid value for --log-format: expected text or json\n");
      return 2;
    }
    ombra::server::log::set_format(lf);
  }

  if (create_key) {
    ombra::server::Scope scope;
    if (data_dir.empty() || !ombra::server::parse_scope(scope_arg, &scope)) {
      std::fprintf(stderr,
                   "usage: ombra_server --create-api-key --scope admin|read|write --data-dir D "
                   "[--collection NAME (read or write only)]\n");
      return 2;
    }
    try {
      std::string id;
      const std::string token = ombra::server::create_api_key(data_dir, scope, &id, key_collection);
      // The only place a token is ever shown.
      std::printf("%s\n", token.c_str());
      std::fflush(stdout);
      std::fprintf(stderr,
                   "created %s API key %s%s (shown once; only its salted hash is stored)\n",
                   scope_arg.c_str(), id.c_str(),
                   key_collection.empty() ? "" : " for one collection");
    } catch (const std::exception& e) {
      std::fprintf(stderr, "fatal: %s\n", e.what());
      return 1;
    }
    return 0;
  }

  if (data_dir.empty() || listen.empty()) {
    std::fprintf(stderr,
                 "usage: ombra_server --data-dir D --listen HOST:PORT --tls-cert F --tls-key F\n");
    return 2;
  }
  if (tls_cert.empty() != tls_key.empty()) {
    std::fprintf(stderr, "--tls-cert and --tls-key go together\n");
    return 2;
  }
  const bool tls = !tls_cert.empty();
  if (!tls && !insecure) {
    std::fprintf(stderr,
                 "refusing to start without TLS: pass --tls-cert and --tls-key (or --insecure)\n");
    return 2;
  }
  std::unique_ptr<ombra::server::ApiKeyStore> keys;
  if (!no_auth) {
    try {
      keys = std::make_unique<ombra::server::ApiKeyStore>(
          ombra::server::ApiKeyStore::load(ombra::server::ApiKeyStore::path_in(data_dir)));
    } catch (const std::exception& e) {
      std::fprintf(stderr,
                   "no usable API key file (%s): create one with --create-api-key, or pass "
                   "--no-auth\n",
                   e.what());
      return 2;
    }
    if (keys->size() == 0) {
      std::fprintf(stderr, "the API key file is empty: create a key with --create-api-key\n");
      return 2;
    }
  }
#ifdef OMBRA_NO_GPU
  ombra::server::log::warn(
      "this server was built without GPU support (for tests and development): Search fails, "
      "every other call works");
#endif
  if (!tls) ombra::server::log::warn("--insecure: connections are NOT encrypted");
  if (no_auth) ombra::server::log::warn("--no-auth: anyone who can connect has full access");
  // The REST port is checked on the host it really binds (audit A-18), and defaults to loopback.
  const std::string rhost = rest_host.empty() ? "127.0.0.1" : rest_host;
  if ((!tls || no_auth) && !is_loopback_listen(listen)) {
    std::fprintf(stderr,
                 "refusing to listen on a non-loopback address with --insecure or --no-auth\n");
    return 2;
  }
  if ((!tls || no_auth) && rest_port >= 0 && !ombra::server::is_loopback_host(rhost)) {
    std::fprintf(stderr,
                 "refusing to serve the REST port on a non-loopback address with --insecure or "
                 "--no-auth\n");
    return 2;
  }
  std::size_t gpu_bytes = 0, host_bytes = 0, session_bytes = 0, quota_bytes = 0;
  std::size_t blob_bytes = 0, upload_bytes = 0;
  if (!ombra::server::mb_to_bytes(gpu_mb, &gpu_bytes) ||
      !ombra::server::mb_to_bytes(host_mb, &host_bytes) ||
      !ombra::server::mb_to_bytes(session_budget_mb, &session_bytes) ||
      !ombra::server::mb_to_bytes(memory_quota_mb, &quota_bytes) ||
      !ombra::server::mb_to_bytes(max_blob_mb, &blob_bytes) ||
      !ombra::server::mb_to_bytes(max_key_upload_mb, &upload_bytes)) {
    std::fprintf(stderr, "a memory size flag is too large\n");
    return 2;
  }
  limits.max_blob_bytes_per_collection = blob_bytes;
  limits.max_key_upload_bytes = upload_bytes;
  limits.search_idle_ms = search_idle_s * 1000;
  limits.search_total_ms = search_total_s * 1000;
  if (ofhe == "unknown") {
    std::fprintf(stderr, "OpenFHE version unknown at build time: pass --openfhe-version\n");
    return 2;
  }

  // A test hook of the security floor is never honoured by the server, whatever the build.
  ::unsetenv("OMBRA_TEST_LOG2QP_LIMIT");
  ::umask(077);  // the data directory is private: files 0600, directories 0700 (audit A-22)
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::signal(SIGHUP, on_hup);
  try {
    ombra::storage::Store store(data_dir, ofhe);
    ombra::search::EngineConfig cfg;
    cfg.gpu_budget_bytes = gpu_bytes;
    cfg.host_budget_bytes = host_bytes;
    cfg.session_budget_bytes = session_bytes;
    // Presets come from the generated table of core/params; the engine and the store use it to
    // refuse an uploaded context or stored column that is not the collection's preset (A-16, A-71).
    cfg.preset_check = [](const std::string& preset, std::size_t ring, std::size_t batch,
                          std::size_t depth, std::size_t first_bits, std::size_t scale_bits) {
      const auto* p = ombra::server::find_preset(preset);
      const auto close = [](std::size_t a, std::size_t b) { return a + 1 >= b && b + 1 >= a; };
      return p && p->ring_dim == ring && p->batch_size == batch && p->depth == depth &&
             close(p->first_mod_bits, first_bits) && close(p->scale_bits, scale_bits);
    };
    store.set_shape_provider(
        [](const std::string& preset, std::uint32_t* ring, std::uint32_t* slots) {
          const auto* p = ombra::server::find_preset(preset);
          if (!p) return false;
          *ring = static_cast<std::uint32_t>(p->ring_dim);
          *slots = static_cast<std::uint32_t>(p->batch_size);
          return true;
        });
    ombra::search::SearchEngine engine(store, cfg);
    ombra::server::Metrics metrics;
    ombra::server::OmbraService service(store, engine, keys.get(), metrics, limits);

    std::shared_ptr<grpc::ServerCredentials> creds;
    const auto auth_processor =
        std::make_shared<ombra::server::ApiKeyAuthProcessor>(keys.get(), metrics);
    if (tls) {
      grpc::SslServerCredentialsOptions opts;
      opts.pem_key_cert_pairs.push_back({read_file(tls_key), read_file(tls_cert)});
      creds = grpc::SslServerCredentials(opts);
      // Authenticate when the headers arrive, before gRPC reads a message body (audit A-03).
      if (keys) creds->SetAuthMetadataProcessor(auth_processor);
    } else {
      creds = grpc::InsecureServerCredentials();
    }
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort(listen, creds, &port);
    const int max_msg = static_cast<int>(max_message_mb << 20);
    builder.SetMaxReceiveMessageSize(max_msg);
    builder.SetMaxSendMessageSize(max_msg);
    // Memory and threads of the whole gRPC server (buffered messages of all calls count here).
    builder.SetResourceQuota(grpc::ResourceQuota("ombra")
                                 .Resize(quota_bytes)
                                 .SetMaxThreads(static_cast<int>(max_threads)));
    // Connection limits and keepalive policy. Idle connections are closed, connections are
    // recycled, and a client that pings too often is disconnected.
    builder.AddChannelArgument(kArgMaxConcurrentStreams, 32);
    builder.AddChannelArgument(kArgMaxConnectionIdleMs, 5 * 60 * 1000);
    builder.AddChannelArgument(kArgMaxConnectionAgeMs, 60 * 60 * 1000);
    builder.AddChannelArgument(kArgMaxConnectionAgeGraceMs, 5 * 60 * 1000);
    builder.AddChannelArgument(kArgKeepaliveTimeMs, 2 * 60 * 1000);
    builder.AddChannelArgument(kArgKeepaliveTimeoutMs, 20 * 1000);
    builder.AddChannelArgument(kArgKeepalivePermitWithoutCalls, 0);
    builder.AddChannelArgument(kArgMinPingIntervalMs, 60 * 1000);
    builder.AddChannelArgument(kArgMaxPingStrikes, 2);
    builder.AddChannelArgument(kArgHandshakeTimeoutMs, 10 * 1000);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server || port == 0) {
      std::fprintf(stderr, "cannot listen on %s\n", listen.c_str());
      return 1;
    }
    const std::string host = listen.substr(0, listen.rfind(':'));
    std::printf("ombra_server listening on %s:%d\n", host.c_str(), port);
    std::fflush(stdout);

    std::unique_ptr<ombra::server::RestServer> rest;
    if (rest_port >= 0) {
      rest = std::make_unique<ombra::server::RestServer>(
          rhost, rest_port, tls_cert, tls_key, keys.get(), metrics,
          [&store, &engine] { return collect_gauges(store, engine); },
          static_cast<std::size_t>(rest_workers), static_cast<std::size_t>(rest_max_conns));
      std::printf("ombra_server rest listening on %s:%d\n", rhost.c_str(), rest->port());
      std::fflush(stdout);
    }

    std::thread compactor;
    if (compact_ms > 0) {
      compactor = std::thread([&store, compact_ms] {
        auto next = std::chrono::steady_clock::now() + std::chrono::milliseconds(compact_ms);
        while (!g_stop.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          if (std::chrono::steady_clock::now() < next) continue;
          next = std::chrono::steady_clock::now() + std::chrono::milliseconds(compact_ms);
          try {
            std::size_t dropped = 0;
            for (const auto& c : store.list_collections()) dropped += store.drop_dead_blocks(c);
            if (dropped > 0)
              ombra::server::log::info("compaction", {{"dropped", static_cast<std::uint64_t>(dropped)}});
          } catch (const std::exception& e) {
            ombra::server::log::error("compaction failed", {{"error", e.what()}});
          }
        }
      });
    }
    while (!g_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (g_reload.exchange(false) && keys) {
        // SIGHUP: pick up revoked or new API keys. A broken file keeps the loaded keys.
        try {
          keys->reload(ombra::server::ApiKeyStore::path_in(data_dir));
          ombra::server::log::info("reloaded API keys",
                                   {{"count", static_cast<std::uint64_t>(keys->size())}});
        } catch (const std::exception& e) {
          ombra::server::log::error("API key reload failed, keeping the loaded keys",
                                    {{"error", e.what()}});
        }
      }
    }
    // Shutdown (see the header comment). Not ready first, so a load balancer stops sending.
    ombra::server::log::info("shutdown requested: not ready, draining",
                             {{"grace_s", static_cast<std::uint64_t>(shutdown_grace_s)}});
    if (rest) rest->begin_drain();
    service.begin_drain();
    if (compactor.joinable()) compactor.join();
    std::atomic<bool> drained{false};
    // gRPC: no new calls from now on; running calls have until the deadline, then they are
    // cancelled. Shutdown returns when every handler has returned.
    std::thread drain([&] {
      server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(shutdown_grace_s));
      drained.store(true);
    });
    while (!drained.load() && g_signals.load() < 2)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (!drained.load()) {
      ombra::server::log::warn("second signal: exiting now, running calls are cut");
      std::_Exit(1);  // acknowledged writes are durable; recovery sweeps unfinished ones
    }
    drain.join();
    server->Wait();
    if (rest) rest->stop();
    ombra::server::log::info("shutdown complete");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
