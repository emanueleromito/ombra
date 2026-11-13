// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// CPU unit tests of the resource limits: session caps, stream deadlines, log rate limit, strict
// flag parsing, loopback checks.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>

#include "ombra/server/limits.hpp"

#define CHECK(c)                                                                \
  do {                                                                          \
    if (!(c)) {                                                                 \
      std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

using namespace ombra::server;
using namespace std::chrono_literals;

static void test_sessions() {
  SessionLimiter lim(3, 2);
  auto a1 = lim.try_acquire("a");
  auto a2 = lim.try_acquire("a");
  CHECK(a1.valid() && a2.valid());
  CHECK(!lim.try_acquire("a").valid());  // per-key cap
  auto b1 = lim.try_acquire("b");
  CHECK(b1.valid() && lim.active() == 3);
  CHECK(!lim.try_acquire("c").valid());  // global cap
  a1 = SessionLimiter::Ticket();         // release
  CHECK(lim.active() == 2);
  auto c1 = lim.try_acquire("c");
  CHECK(c1.valid());
  SessionLimiter::Ticket moved = std::move(c1);
  CHECK(!c1.valid() && moved.valid() && lim.active() == 3);
  moved = SessionLimiter::Ticket();
  CHECK(lim.active() == 2);
}

static void test_watchdog() {
  Watchdog w;
  std::atomic<int> idle_fired{0}, total_fired{0}, busy_fired{0};
  auto idle = w.watch([&] { ++idle_fired; }, 100, 0);
  auto total = w.watch([&] { ++total_fired; }, 0, 150);
  auto busy = w.watch([&] { ++busy_fired; }, 100, 0);
  busy->busy(true);
  for (int i = 0; i < 3; ++i) {  // the idle stream is kept alive for 150 ms by touching it
    std::this_thread::sleep_for(50ms);
    idle->touch();
  }
  CHECK(idle_fired == 0);
  std::this_thread::sleep_for(300ms);  // touches stop: idle fires; total fired at 150 ms
  CHECK(total_fired == 1 && total->timed_out());
  CHECK(idle_fired == 1 && idle->timed_out());
  CHECK(busy_fired == 0 && !busy->timed_out());  // not enforced while the server computes
  busy->busy(false);
  std::this_thread::sleep_for(250ms);
  CHECK(busy_fired == 1);
  // A removed guard is never cancelled afterwards.
  std::atomic<int> gone{0};
  { auto g = w.watch([&] { ++gone; }, 50, 0); }
  std::this_thread::sleep_for(150ms);
  CHECK(gone == 0);
}

static void test_log_limiter() {
  LogLimiter l(200ms);
  std::uint64_t s = 99;
  CHECK(l.allow(&s) && s == 0);
  CHECK(!l.allow() && !l.allow());
  std::this_thread::sleep_for(250ms);
  CHECK(l.allow(&s) && s == 2);
}

static void test_flags() {
  std::uint64_t v = 0;
  CHECK(parse_uint("4096", 1, 1000000, &v) && v == 4096);
  CHECK(!parse_uint("", 0, 10, &v) && !parse_uint("-1", 0, 10, &v) && !parse_uint("1x", 0, 10, &v));
  CHECK(!parse_uint(" 1", 0, 10, &v) && !parse_uint("+1", 0, 10, &v));
  CHECK(!parse_uint("0", 1, 10, &v) && !parse_uint("11", 0, 10, &v));
  CHECK(!parse_uint("99999999999999999999999", 0, UINT64_MAX, &v));
  CHECK(!parse_uint("18446744073709551616", 0, UINT64_MAX, &v));  // 2^64
  CHECK(parse_uint("18446744073709551615", 0, UINT64_MAX, &v) && v == UINT64_MAX);
  std::size_t b = 0;
  CHECK(mb_to_bytes(4096, &b) && b == (std::size_t{4096} << 20));
  CHECK(!mb_to_bytes(std::numeric_limits<std::uint64_t>::max(), &b));
  CHECK(!mb_to_bytes((std::numeric_limits<std::size_t>::max() >> 20) + 1, &b));
  CHECK(is_loopback_host("127.0.0.1") && is_loopback_host("localhost") && is_loopback_host("::1"));
  CHECK(!is_loopback_host("0.0.0.0") && !is_loopback_host("127.0.0.1.evil.com") &&
        !is_loopback_host(""));
  CHECK(is_loopback_listen("127.0.0.1:50051") && is_loopback_listen("[::1]:1") &&
        is_loopback_listen("localhost:0"));
  CHECK(!is_loopback_listen("0.0.0.0:50051") && !is_loopback_listen("127.0.0.1") &&
        !is_loopback_listen("[::]:1"));
}

int main() {
  test_sessions();
  test_watchdog();
  test_log_limiter();
  test_flags();
  std::printf("limits_test: ok\n");
  return 0;
}
