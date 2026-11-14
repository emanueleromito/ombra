// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// The server's one logging helper. Text lines (default) or JSON lines (--log-format json) on
// stderr.
//
// Rules (server/CLAUDE.md): a log line carries counts, sizes, timings, the 8-hex API key id and the
// request id. Never a collection name, an id, ciphertext, a value or a token. Callers pass only
// fields that follow this rule; the helper does not look at the values.
//
// Inside an RPC, a CallScope (made by the service) puts request_id, rpc and key_id on every line
// logged by that thread until the scope ends. gRPC's synchronous API runs a call on one thread, so
// the handler and everything it calls share the scope.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace ombra::server::log {

enum class Format { Text, Json };
enum class Level { Info, Warn, Error };

bool parse_format(std::string_view s, Format* out);  // "text" or "json"
void set_format(Format f);
Format format();

// A name and a value; numbers stay numbers in JSON.
struct Field {
  std::string key;
  std::string value;
  bool numeric = false;
  Field(const char* k, std::string_view v) : key(k), value(v) {}
  Field(const char* k, const char* v) : key(k), value(v) {}
  Field(const char* k, const std::string& v) : key(k), value(v) {}
  Field(const char* k, double v);
  Field(const char* k, std::uint64_t v) : key(k), value(std::to_string(v)), numeric(true) {}
  Field(const char* k, std::int64_t v) : key(k), value(std::to_string(v)), numeric(true) {}
  Field(const char* k, int v) : key(k), value(std::to_string(v)), numeric(true) {}
  Field(const char* k, unsigned v) : key(k), value(std::to_string(v)), numeric(true) {}
};

// The line for `msg` and `fields`, without a trailing newline; `ts` is passed in so that tests can
// compare exact text. Used by write().
std::string format_line(Format f, Level lv, std::string_view ts, std::string_view msg,
                        const std::vector<Field>& fields);

// Writes one line to stderr (one write call, so lines of different threads do not mix).
void write(Level lv, std::string_view msg, std::initializer_list<Field> fields = {});
inline void info(std::string_view msg, std::initializer_list<Field> f = {}) {
  write(Level::Info, msg, f);
}
inline void warn(std::string_view msg, std::initializer_list<Field> f = {}) {
  write(Level::Warn, msg, f);
}
inline void error(std::string_view msg, std::initializer_list<Field> f = {}) {
  write(Level::Error, msg, f);
}

// ---- request ids --------------------------------------------------------------------------------
inline constexpr std::size_t kMaxRequestIdBytes = 64;
// 1 to 64 characters from [A-Za-z0-9._-].
bool valid_request_id(std::string_view s);
// 16 random hex characters.
std::string new_request_id();

// Identity of the call running on this thread. Strings are copied, so the scope may outlive the
// metadata they came from.
class CallScope {
 public:
  CallScope(std::string request_id, const char* rpc);
  ~CallScope();
  CallScope(const CallScope&) = delete;
  CallScope& operator=(const CallScope&) = delete;
  void set_key_id(std::string key_id);  // after authentication
  const std::string& request_id() const { return request_id_; }

 private:
  std::string request_id_;
  const char* rpc_;
  std::string key_id_;
  CallScope* prev_;
  friend void write(Level, std::string_view, std::initializer_list<Field>);
};

}  // namespace ombra::server::log
