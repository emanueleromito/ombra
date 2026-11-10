// ombra_server --data-dir D --listen HOST:PORT [--openfhe-version V]
//                [--gpu-budget-mb N] [--host-budget-mb N] [--compact-interval-ms N]
//                [--tls-cert F --tls-key F] [--rest-port N [--rest-host H]]
//                [--insecure] [--no-auth]
// ombra_server --create-api-key --scope admin|read|write --data-dir D
//
// The ombra.v1 gRPC server over the storage engine and the GPU search engine (T2.7b), with TLS,
// API keys and a REST admin port (T2.9).
//
// Safe defaults: the server refuses to start without TLS unless --insecure is given, and without an
// API key file (D/api_keys.v1, made with --create-api-key) unless --no-auth is given. Both flags
// log a loud warning and are limited to loopback addresses. An API key is an access token; the
// server never accepts a secret key (hard rule 2) and never logs a token.
//
// Every --compact-interval-ms (default 60000, 0 disables) a background thread drops, in every
// collection, the blocks whose vectors are all deleted; it logs counts only.
//
// Prints "ombra_server listening on HOST:PORT" on stdout once the port is bound (PORT 0 picks a
// free port), then, with --rest-port, "ombra_server rest listening on HOST:PORT". Shuts down on
// SIGINT or SIGTERM.
#include <grpcpp/grpcpp.h>

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
#include "ombra/server/metrics.hpp"
#include "ombra/storage/store.hpp"
#include "rest_server.hpp"
#include "service.hpp"

#ifndef OMBRA_OPENFHE_VERSION
#define OMBRA_OPENFHE_VERSION "unknown"
#endif

namespace {

std::atomic<bool> g_stop{false};
extern "C" void on_signal(int) { g_stop.store(true); }

bool is_loopback(const std::string& addr) {
  return addr.rfind("127.0.0.1:", 0) == 0 || addr.rfind("localhost:", 0) == 0 ||
         addr.rfind("[::1]:", 0) == 0;
}

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
  const auto cm = engine.cache_metrics();
  g.cache_hits = cm.hits;
  g.cache_misses = cm.misses;
  g.cache_evictions = cm.evictions;
  return g;
}

// Messages carry whole blocks (tens of MB at the larger preset): lift gRPC's 4 MiB default.
constexpr int kMaxMessageBytes = 1 << 30;

}  // namespace

int main(int argc, char** argv) {
  std::string data_dir, listen, ofhe = OMBRA_OPENFHE_VERSION;
  std::string tls_cert, tls_key, scope_arg, rest_host;
  std::size_t gpu_mb = 4096, host_mb = 2048;
  std::size_t compact_ms = 60000;
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
    if (a == "--data-dir") data_dir = next();
    else if (a == "--listen") listen = next();
    else if (a == "--openfhe-version") ofhe = next();
    else if (a == "--gpu-budget-mb") gpu_mb = std::stoul(next());
    else if (a == "--host-budget-mb") host_mb = std::stoul(next());
    else if (a == "--compact-interval-ms") compact_ms = std::stoul(next());
    else if (a == "--tls-cert") tls_cert = next();
    else if (a == "--tls-key") tls_key = next();
    else if (a == "--rest-port") rest_port = std::stoi(next());
    else if (a == "--rest-host") rest_host = next();
    else if (a == "--scope") scope_arg = next();
    else if (a == "--create-api-key") create_key = true;
    else if (a == "--insecure") insecure = true;
    else if (a == "--no-auth") no_auth = true;
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }

  if (create_key) {
    ombra::server::Scope scope;
    if (data_dir.empty() || !ombra::server::parse_scope(scope_arg, &scope)) {
      std::fprintf(stderr,
                   "usage: ombra_server --create-api-key --scope admin|read|write --data-dir D\n");
      return 2;
    }
    try {
      std::string id;
      const std::string token = ombra::server::create_api_key(data_dir, scope, &id);
      // The only place a token is ever shown.
      std::printf("%s\n", token.c_str());
      std::fflush(stdout);
      std::fprintf(stderr, "created %s API key %s (shown once; only its salted hash is stored)\n",
                   scope_arg.c_str(), id.c_str());
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
  if (!tls) std::fprintf(stderr, "WARNING: --insecure: connections are NOT encrypted\n");
  if (no_auth) std::fprintf(stderr, "WARNING: --no-auth: anyone who can connect has full access\n");
  if ((!tls || no_auth) && !is_loopback(listen)) {
    std::fprintf(stderr,
                 "refusing to listen on a non-loopback address with --insecure or --no-auth\n");
    return 2;
  }
  if (ofhe == "unknown") {
    std::fprintf(stderr, "OpenFHE version unknown at build time: pass --openfhe-version\n");
    return 2;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  try {
    ombra::storage::Store store(data_dir, ofhe);
    ombra::search::EngineConfig cfg;
    cfg.gpu_budget_bytes = gpu_mb << 20;
    cfg.host_budget_bytes = host_mb << 20;
    ombra::search::SearchEngine engine(store, cfg);
    ombra::server::Metrics metrics;
    ombra::server::OmbraService service(store, engine, keys.get(), metrics);

    std::shared_ptr<grpc::ServerCredentials> creds;
    if (tls) {
      grpc::SslServerCredentialsOptions opts;
      opts.pem_key_cert_pairs.push_back({read_file(tls_key), read_file(tls_cert)});
      creds = grpc::SslServerCredentials(opts);
    } else {
      creds = grpc::InsecureServerCredentials();
    }
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort(listen, creds, &port);
    builder.SetMaxReceiveMessageSize(kMaxMessageBytes);
    builder.SetMaxSendMessageSize(kMaxMessageBytes);
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
      const std::string rhost = rest_host.empty() ? host : rest_host;
      rest = std::make_unique<ombra::server::RestServer>(
          rhost, rest_port, tls_cert, tls_key, keys.get(), metrics,
          [&store, &engine] { return collect_gauges(store, engine); });
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
            if (dropped > 0) std::fprintf(stderr, "[ombra.server] compaction dropped=%zu\n", dropped);
          } catch (const std::exception&) {
            std::fprintf(stderr, "[ombra.server] compaction failed\n");
          }
        }
      });
    }
    while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (compactor.joinable()) compactor.join();
    if (rest) rest->stop();
    server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(2));
    server->Wait();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
