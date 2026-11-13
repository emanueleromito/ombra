// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include "ombra/server/limits.hpp"

#include <limits>

namespace ombra::server {

// ---- SessionLimiter ----------------------------------------------------------------------------

SessionLimiter::Ticket SessionLimiter::try_acquire(const std::string& key_id) {
  std::lock_guard lk(mu_);
  if (total_ >= max_total_) return {};
  auto& n = per_key_[key_id];
  if (n >= max_per_key_) {
    if (n == 0) per_key_.erase(key_id);
    return {};
  }
  ++n;
  ++total_;
  return Ticket(this, key_id);
}

void SessionLimiter::release(const std::string& key_id) {
  std::lock_guard lk(mu_);
  auto it = per_key_.find(key_id);
  if (it != per_key_.end()) {
    if (it->second > 0) --it->second;
    if (it->second == 0) per_key_.erase(it);
  }
  if (total_ > 0) --total_;
}

std::size_t SessionLimiter::active() const {
  std::lock_guard lk(mu_);
  return total_;
}

// ---- Watchdog ----------------------------------------------------------------------------------

Watchdog::Watchdog() : thread_([this] { loop(); }) {}

Watchdog::~Watchdog() {
  {
    std::lock_guard lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

std::unique_ptr<Watchdog::Guard> Watchdog::watch(std::function<void()> cancel,
                                                 std::uint64_t idle_ms, std::uint64_t total_ms) {
  std::lock_guard lk(mu_);
  const std::uint64_t id = next_id_++;
  Entry e;
  e.cancel = std::move(cancel);
  e.start = e.last = Clock::now();
  e.idle = std::chrono::milliseconds(idle_ms);
  e.total = std::chrono::milliseconds(total_ms);
  entries_.emplace(id, std::move(e));
  return std::unique_ptr<Guard>(new Guard(this, id));
}

Watchdog::Guard::~Guard() {
  std::lock_guard lk(w_->mu_);
  w_->entries_.erase(id_);
}

void Watchdog::Guard::touch() {
  std::lock_guard lk(w_->mu_);
  auto it = w_->entries_.find(id_);
  if (it != w_->entries_.end()) it->second.last = Clock::now();
}

void Watchdog::Guard::busy(bool on) {
  std::lock_guard lk(w_->mu_);
  auto it = w_->entries_.find(id_);
  if (it == w_->entries_.end()) return;
  it->second.busy = on;
  it->second.last = Clock::now();
}

bool Watchdog::Guard::timed_out() const {
  std::lock_guard lk(w_->mu_);
  auto it = w_->entries_.find(id_);
  return it != w_->entries_.end() && it->second.fired;
}

void Watchdog::loop() {
  std::unique_lock lk(mu_);
  while (!stop_) {
    cv_.wait_for(lk, std::chrono::milliseconds(20));
    if (stop_) break;
    const auto now = Clock::now();
    for (auto& [id, e] : entries_) {
      if (e.fired) continue;
      const bool total_hit = e.total.count() > 0 && now - e.start >= e.total;
      const bool idle_hit = e.idle.count() > 0 && !e.busy && now - e.last >= e.idle;
      if (total_hit || idle_hit) {
        e.fired = true;
        if (e.cancel) e.cancel();  // under mu_: the Guard cannot vanish meanwhile
      }
    }
  }
}

// ---- LogLimiter --------------------------------------------------------------------------------

bool LogLimiter::allow(std::uint64_t* suppressed) {
  std::lock_guard lk(mu_);
  const auto now = std::chrono::steady_clock::now();
  if (!any_ || now - last_ >= interval_) {
    any_ = true;
    last_ = now;
    if (suppressed) *suppressed = suppressed_;
    suppressed_ = 0;
    return true;
  }
  ++suppressed_;
  return false;
}

// ---- Flags -------------------------------------------------------------------------------------

bool parse_uint(const std::string& s, std::uint64_t min, std::uint64_t max, std::uint64_t* out) {
  if (s.empty() || s.size() > 20) return false;
  std::uint64_t v = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    const std::uint64_t d = static_cast<std::uint64_t>(c - '0');
    if (v > (std::numeric_limits<std::uint64_t>::max() - d) / 10) return false;
    v = v * 10 + d;
  }
  if (v < min || v > max) return false;
  *out = v;
  return true;
}

bool mb_to_bytes(std::uint64_t mb, std::size_t* bytes) {
  if (mb > (std::numeric_limits<std::size_t>::max() >> 20)) return false;
  *bytes = static_cast<std::size_t>(mb) << 20;
  return true;
}

bool is_loopback_host(const std::string& host) {
  return host == "127.0.0.1" || host == "localhost" || host == "::1" || host == "[::1]";
}

bool is_loopback_listen(const std::string& addr) {
  const auto pos = addr.rfind(':');
  if (pos == std::string::npos) return false;
  return is_loopback_host(addr.substr(0, pos));
}

}  // namespace ombra::server
