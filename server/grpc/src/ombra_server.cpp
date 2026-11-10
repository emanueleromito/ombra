// ombra_server --data-dir D --listen 127.0.0.1:PORT [--openfhe-version V]
//                [--gpu-budget-mb N] [--host-budget-mb N] [--compact-interval-ms N]
//
// The ombra.v1 gRPC server over the storage engine and the GPU search engine (T2.7b).
//
// Every --compact-interval-ms (default 60000, 0 disables) a background thread drops, in every
// collection, the blocks whose vectors are all deleted; it logs counts only.
//
// PLAINTEXT, LOCALHOST ONLY. TLS and API keys are T2.9; until then the server refuses to listen on
// any address that is not loopback. It never accepts a secret key (hard rule 2).
//
// Prints "ombra_server listening on HOST:PORT" on stdout once the port is bound (PORT 0 picks a
// free port), and shuts down on SIGINT or SIGTERM.
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include "ombra/search/session.hpp"
#include "ombra/storage/store.hpp"
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

// Messages carry whole blocks (tens of MB at the larger preset): lift gRPC's 4 MiB default.
constexpr int kMaxMessageBytes = 1 << 30;

}  // namespace

int main(int argc, char** argv) {
  std::string data_dir, listen, ofhe = OMBRA_OPENFHE_VERSION;
  std::size_t gpu_mb = 4096, host_mb = 2048;
  std::size_t compact_ms = 60000;
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
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (data_dir.empty() || listen.empty()) {
    std::fprintf(stderr, "usage: ombra_server --data-dir D --listen 127.0.0.1:PORT\n");
    return 2;
  }
  if (!is_loopback(listen)) {
    std::fprintf(stderr, "refusing to listen on a non-loopback address: TLS and API keys come with T2.9\n");
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
    ombra::server::OmbraService service(store, engine);

    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort(listen, grpc::InsecureServerCredentials(), &port);
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
    server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(2));
    server->Wait();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
