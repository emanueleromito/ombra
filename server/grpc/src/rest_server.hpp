// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// REST admin port: GET /healthz (no auth) and GET /metrics (admin key). HTTP/1.1 over
// OpenSSL (TLS 1.2+) or, with --insecure, plain TCP. One request per connection, header cap 8 KiB,
// a total deadline per connection that also covers the TLS handshake, a small worker pool so that
// one slow client cannot block /healthz, and a cap on open connections (audit A-17). OpenSSL is
// already a dependency of gRPC.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "ombra/server/auth.hpp"
#include "ombra/server/metrics.hpp"

namespace ombra::server {

class RestServer {
 public:
  // tls_cert and tls_key both empty means plain TCP. host is a dotted IPv4 address or "localhost".
  // Throws std::runtime_error if the TLS files or the port are unusable. port 0 picks a free one.
  RestServer(const std::string& host, int port, const std::string& tls_cert,
             const std::string& tls_key, const ApiKeyStore* keys, Metrics& metrics,
             std::function<Gauges()> gauges, std::size_t workers = 4,
             std::size_t max_connections = 32);
  ~RestServer();
  RestServer(const RestServer&) = delete;
  RestServer& operator=(const RestServer&) = delete;

  int port() const { return port_; }
  void stop();

 private:
  struct Tls;  // SSL_CTX holder
  void run();
  void worker();
  void serve(int fd);

  const ApiKeyStore* keys_;
  Metrics& metrics_;
  std::function<Gauges()> gauges_;
  std::unique_ptr<Tls> tls_;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> stop_{false};
  std::thread thread_;  // accept loop
  std::vector<std::thread> workers_;
  std::size_t max_connections_;
  std::mutex qmu_;
  std::condition_variable qcv_;
  std::deque<int> queue_;
  std::size_t open_ = 0;  // accepted and not yet closed (queued or being served), guarded by qmu_
};

}  // namespace ombra::server
