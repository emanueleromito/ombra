#include "rest_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>

#include "ombra/server/http.hpp"

namespace ombra::server {

namespace {

constexpr int kReadTimeoutMs = 5000;  // total budget per connection for handshake and headers

struct SslDeleter {
  void operator()(SSL* s) const { SSL_free(s); }
};
struct CtxDeleter {
  void operator()(SSL_CTX* c) const { SSL_CTX_free(c); }
};

}  // namespace

struct RestServer::Tls {
  std::unique_ptr<SSL_CTX, CtxDeleter> ctx;
};

RestServer::RestServer(const std::string& host, int port, const std::string& tls_cert,
                       const std::string& tls_key, const ApiKeyStore* keys, Metrics& metrics,
                       std::function<Gauges()> gauges)
    : keys_(keys), metrics_(metrics), gauges_(std::move(gauges)) {
  if (!tls_cert.empty() || !tls_key.empty()) {
    tls_ = std::make_unique<Tls>();
    tls_->ctx.reset(SSL_CTX_new(TLS_server_method()));
    if (!tls_->ctx) throw std::runtime_error("cannot create the TLS context");
    SSL_CTX_set_min_proto_version(tls_->ctx.get(), TLS1_2_VERSION);
    if (SSL_CTX_use_certificate_chain_file(tls_->ctx.get(), tls_cert.c_str()) != 1 ||
        SSL_CTX_use_PrivateKey_file(tls_->ctx.get(), tls_key.c_str(), SSL_FILETYPE_PEM) != 1 ||
        SSL_CTX_check_private_key(tls_->ctx.get()) != 1)
      throw std::runtime_error("cannot load the TLS certificate or key for the REST port");
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<std::uint16_t>(port));
  const std::string h = host == "localhost" ? "127.0.0.1" : host;
  if (inet_pton(AF_INET, h.c_str(), &addr.sin_addr) != 1)
    throw std::runtime_error("the REST port needs an IPv4 address");
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listen_fd_ < 0) throw std::runtime_error("cannot create the REST socket");
  const int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 ||
      ::listen(listen_fd_, 16) != 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
    throw std::runtime_error("cannot listen on the REST port");
  }
  socklen_t len = sizeof addr;
  ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
  port_ = ntohs(addr.sin_port);
  thread_ = std::thread([this] { run(); });
}

RestServer::~RestServer() { stop(); }

void RestServer::stop() {
  stop_.store(true);
  if (thread_.joinable()) thread_.join();
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
}

void RestServer::run() {
  while (!stop_.load()) {
    pollfd p{listen_fd_, POLLIN, 0};
    if (::poll(&p, 1, 100) <= 0) continue;
    const int fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd < 0) continue;
    try {
      serve(fd);
    } catch (...) {
    }
    ::close(fd);
  }
}

void RestServer::serve(int fd) {
  using Clock = std::chrono::steady_clock;
  const auto deadline = Clock::now() + std::chrono::milliseconds(kReadTimeoutMs);
  const auto left_ms = [&] {
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count());
  };
  timeval tv{kReadTimeoutMs / 1000, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);

  std::unique_ptr<SSL, SslDeleter> ssl;
  if (tls_) {
    ssl.reset(SSL_new(tls_->ctx.get()));
    if (!ssl) return;
    SSL_set_fd(ssl.get(), fd);
    if (SSL_accept(ssl.get()) != 1) return;  // plain-text or broken client: dropped, no answer
  }
  const auto read_some = [&](char* buf, std::size_t n) -> long {
    if (ssl) {
      const int r = SSL_read(ssl.get(), buf, static_cast<int>(n));
      return r > 0 ? r : -1;
    }
    const ssize_t r = ::recv(fd, buf, n, 0);
    return r > 0 ? static_cast<long>(r) : -1;
  };
  const auto write_all = [&](const std::string& s) {
    std::size_t off = 0;
    while (off < s.size()) {
      long w;
      if (ssl) {
        w = SSL_write(ssl.get(), s.data() + off, static_cast<int>(s.size() - off));
      } else {
        w = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
      }
      if (w <= 0) return;
      off += static_cast<std::size_t>(w);
    }
  };

  std::string buf;
  HttpRequest req;
  HttpParse st = HttpParse::Incomplete;
  while (st == HttpParse::Incomplete) {
    if (left_ms() <= 0) {
      write_all(http_response(408, "{\"error\":\"timeout\"}"));
      return;
    }
    char chunk[1024];
    const long n = read_some(chunk, sizeof chunk);
    if (n <= 0) return;
    buf.append(chunk, static_cast<std::size_t>(n));
    st = parse_http_request(buf, &req);
  }
  if (st == HttpParse::TooLarge) {
    write_all(http_response(431, "{\"error\":\"headers too large\"}"));
    return;
  }
  if (st == HttpParse::Bad) {
    write_all(http_response(400, "{\"error\":\"bad request\"}"));
    return;
  }
  if (req.method != "GET") {
    write_all(http_response(405, "{\"error\":\"method not allowed\"}"));
    return;
  }
  if (req.path == "/healthz") {
    write_all(http_response(200, "{\"status\":\"ok\"}"));
    return;
  }
  if (req.path == "/metrics") {
    std::string key_id;
    switch (authorize(keys_, req.authorization, Scope::Admin, &key_id)) {
      case AuthResult::Unauthenticated:
        metrics_.count_unauthenticated();
        write_all(http_response(401, "{\"error\":\"unauthenticated\"}"));
        return;
      case AuthResult::PermissionDenied:
        metrics_.count_permission_denied();
        std::fprintf(stderr, "[ombra.server] rest denied path=/metrics key=%s\n", key_id.c_str());
        write_all(http_response(403, "{\"error\":\"permission denied\"}"));
        return;
      case AuthResult::Ok:
        break;
    }
    write_all(http_response(200, metrics_json(metrics_, gauges_())));
    return;
  }
  write_all(http_response(404, "{\"error\":\"not found\"}"));
}

}  // namespace ombra::server
