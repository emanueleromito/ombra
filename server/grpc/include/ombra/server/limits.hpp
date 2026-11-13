// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Resource limits of the server as plain C++ (no gRPC, no OpenFHE, no GPU), tested on the CPU:
// the cap on concurrent search sessions, a watchdog for stream deadlines, a log rate limiter and
// strict parsing of numeric flags. Nothing here sees ids, bytes or keys; the per-key bucket is the
// short key id that is already safe to log.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace ombra::server {

// ---- Concurrent search sessions: global cap and per-key cap (audit A-02) ----------------------
class SessionLimiter {
 public:
  SessionLimiter(std::size_t max_total, std::size_t max_per_key)
      : max_total_(max_total), max_per_key_(max_per_key) {}

  // RAII slot. valid() is false when a cap was hit.
  class Ticket {
   public:
    Ticket() = default;
    Ticket(Ticket&& o) noexcept : lim_(o.lim_), key_(std::move(o.key_)) { o.lim_ = nullptr; }
    Ticket& operator=(Ticket&& o) noexcept {
      if (this != &o) {
        release();
        lim_ = o.lim_;
        key_ = std::move(o.key_);
        o.lim_ = nullptr;
      }
      return *this;
    }
    Ticket(const Ticket&) = delete;
    Ticket& operator=(const Ticket&) = delete;
    ~Ticket() { release(); }
    bool valid() const { return lim_ != nullptr; }

   private:
    friend class SessionLimiter;
    Ticket(SessionLimiter* l, std::string k) : lim_(l), key_(std::move(k)) {}
    void release() {
      if (lim_) lim_->release(key_);
      lim_ = nullptr;
    }
    SessionLimiter* lim_ = nullptr;
    std::string key_;
  };

  Ticket try_acquire(const std::string& key_id);
  std::size_t active() const;

 private:
  void release(const std::string& key_id);
  mutable std::mutex mu_;
  std::size_t max_total_, max_per_key_;
  std::size_t total_ = 0;
  std::map<std::string, std::size_t> per_key_;
};

// ---- Stream deadlines (audit A-02) ------------------------------------------------------------
// One background thread calls the registered cancel function (ServerContext::TryCancel in the
// service) when a stream has been idle for too long or has lived too long. A Guard removes its
// entry under the same mutex the thread holds while cancelling, so a cancel never runs after the
// Guard is gone.
class Watchdog {
 public:
  using Clock = std::chrono::steady_clock;
  Watchdog();
  ~Watchdog();
  Watchdog(const Watchdog&) = delete;
  Watchdog& operator=(const Watchdog&) = delete;

  class Guard {
   public:
    ~Guard();
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    void touch();           // the client sent something: restart the idle clock
    void busy(bool on);     // while the server computes, the idle clock is not enforced
    bool timed_out() const;  // a deadline fired and cancel was called

   private:
    friend class Watchdog;
    Guard(Watchdog* w, std::uint64_t id) : w_(w), id_(id) {}
    Watchdog* w_;
    std::uint64_t id_;
  };

  // idle_ms or total_ms == 0 disables that deadline.
  std::unique_ptr<Guard> watch(std::function<void()> cancel, std::uint64_t idle_ms,
                               std::uint64_t total_ms);

 private:
  struct Entry {
    std::function<void()> cancel;
    Clock::time_point start, last;
    std::chrono::milliseconds idle{0}, total{0};
    bool busy = false, fired = false;
  };
  void loop();
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::map<std::uint64_t, Entry> entries_;
  std::uint64_t next_id_ = 1;
  bool stop_ = false;
  std::thread thread_;
};

// ---- Log rate limit (audit A-53) ---------------------------------------------------------------
// allow() is true at most once per interval; *suppressed receives how many calls were refused
// since the last allowed one.
class LogLimiter {
 public:
  explicit LogLimiter(std::chrono::milliseconds interval) : interval_(interval) {}
  bool allow(std::uint64_t* suppressed = nullptr);

 private:
  std::mutex mu_;
  std::chrono::milliseconds interval_;
  std::chrono::steady_clock::time_point last_{};
  bool any_ = false;
  std::uint64_t suppressed_ = 0;
};

// ---- Flags (audit A-18, A-54) ------------------------------------------------------------------
// Whole string, decimal digits only, within [min, max]. No exceptions, no sign, no spaces.
bool parse_uint(const std::string& s, std::uint64_t min, std::uint64_t max, std::uint64_t* out);
// mb * 2^20 without overflow of size_t; false if it would overflow.
bool mb_to_bytes(std::uint64_t mb, std::size_t* bytes);
// "127.0.0.1", "localhost", "::1" or "[::1]" (no port).
bool is_loopback_host(const std::string& host);
// "HOST:PORT" form of --listen.
bool is_loopback_listen(const std::string& addr);

}  // namespace ombra::server
