// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include "ombra/server/log.hpp"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <random>

namespace ombra::server::log {

namespace {

std::atomic<int> g_format{static_cast<int>(Format::Text)};
thread_local CallScope* t_scope = nullptr;

const char* level_name(Level lv) {
  switch (lv) {
    case Level::Info:
      return "info";
    case Level::Warn:
      return "warn";
    case Level::Error:
      return "error";
  }
  return "info";
}

void json_string(std::string& out, std::string_view s) {
  out.push_back('"');
  for (const unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", c);
          out += b;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

std::string now_iso() {
  const auto now = std::chrono::system_clock::now();
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  const std::time_t t = static_cast<std::time_t>(ms / 1000);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char b[48];
  std::snprintf(b, sizeof b, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", tm.tm_year + 1900,
                tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,
                static_cast<int>(ms % 1000));
  return b;
}

}  // namespace

Field::Field(const char* k, double v) : key(k), numeric(true) {
  char b[40];
  std::snprintf(b, sizeof b, "%.2f", v);
  value = b;
}

bool parse_format(std::string_view s, Format* out) {
  if (s == "text")
    *out = Format::Text;
  else if (s == "json")
    *out = Format::Json;
  else
    return false;
  return true;
}
void set_format(Format f) { g_format.store(static_cast<int>(f)); }
Format format() { return static_cast<Format>(g_format.load()); }

std::string format_line(Format f, Level lv, std::string_view ts, std::string_view msg,
                        const std::vector<Field>& fields) {
  std::string out;
  if (f == Format::Json) {
    out = "{\"ts\":";
    json_string(out, ts);
    out += ",\"level\":\"";
    out += level_name(lv);
    out += "\",\"msg\":";
    json_string(out, msg);
    for (const Field& fl : fields) {
      out.push_back(',');
      json_string(out, fl.key);
      out.push_back(':');
      if (fl.numeric)
        out += fl.value;
      else
        json_string(out, fl.value);
    }
    out.push_back('}');
    return out;
  }
  out = "[ombra.server] ";
  if (lv == Level::Error) out += "error: ";
  if (lv == Level::Warn) out += "warning: ";
  out += msg;
  for (const Field& fl : fields) {
    out.push_back(' ');
    out += fl.key;
    out.push_back('=');
    const bool quote = !fl.numeric && fl.value.find_first_of(" \t\"\n\r") != std::string::npos;
    if (quote)
      json_string(out, fl.value);
    else
      out += fl.value;
  }
  return out;
}

void write(Level lv, std::string_view msg, std::initializer_list<Field> fields) {
  std::vector<Field> all(fields);
  if (const CallScope* c = t_scope) {
    all.emplace_back("request_id", c->request_id_);
    if (c->rpc_) all.emplace_back("rpc", c->rpc_);
    if (!c->key_id_.empty()) all.emplace_back("key_id", c->key_id_);
  }
  std::string line = format_line(format(), lv, now_iso(), msg, all);
  line.push_back('\n');
  // One write call, so lines of different threads do not mix.
  const ssize_t w = ::write(STDERR_FILENO, line.data(), line.size());
  (void)w;
}

bool valid_request_id(std::string_view s) {
  if (s.empty() || s.size() > kMaxRequestIdBytes) return false;
  for (const char c : s) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                    c == '.' || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

std::string new_request_id() {
  static thread_local std::random_device rd;
  static const char* kHex = "0123456789abcdef";
  std::string s;
  for (int i = 0; i < 4; ++i) {
    const std::uint32_t v = rd();
    for (int k = 0; k < 4; ++k) s.push_back(kHex[(v >> (4 * k)) & 15]);
  }
  return s;
}

CallScope::CallScope(std::string request_id, const char* rpc)
    : request_id_(std::move(request_id)), rpc_(rpc), prev_(t_scope) {
  t_scope = this;
}
CallScope::~CallScope() { t_scope = prev_; }
void CallScope::set_key_id(std::string key_id) { key_id_ = std::move(key_id); }

}  // namespace ombra::server::log
