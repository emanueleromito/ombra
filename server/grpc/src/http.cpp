// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include "ombra/server/http.hpp"

#include <cctype>
#include <utility>

namespace ombra::server {

namespace {

bool iequals(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(a[i])) !=
        std::tolower(static_cast<unsigned char>(b[i])))
      return false;
  return true;
}

std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

const char* reason(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 401:
      return "Unauthorized";
    case 403:
      return "Forbidden";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 408:
      return "Request Timeout";
    case 431:
      return "Request Header Fields Too Large";
    default:
      return "Error";
  }
}

}  // namespace

HttpParse parse_http_request(std::string_view buf, HttpRequest* out) {
  const std::size_t end = buf.find("\r\n\r\n");
  if (end == std::string_view::npos)
    return buf.size() >= kMaxHttpHeaderBytes ? HttpParse::TooLarge : HttpParse::Incomplete;
  if (end + 4 > kMaxHttpHeaderBytes) return HttpParse::TooLarge;

  std::string_view head = buf.substr(0, end);
  const std::size_t eol = head.find("\r\n");
  const std::string_view line = head.substr(0, eol);
  const std::size_t s1 = line.find(' ');
  const std::size_t s2 = s1 == std::string_view::npos ? s1 : line.find(' ', s1 + 1);
  if (s1 == std::string_view::npos || s2 == std::string_view::npos || s1 == 0 || s2 == s1 + 1)
    return HttpParse::Bad;
  const std::string_view version = line.substr(s2 + 1);
  if (version != "HTTP/1.1" && version != "HTTP/1.0") return HttpParse::Bad;
  HttpRequest req;
  req.method = std::string(line.substr(0, s1));
  req.path = std::string(line.substr(s1 + 1, s2 - s1 - 1));
  if (req.path.empty() || req.path[0] != '/') return HttpParse::Bad;

  std::string_view rest = eol == std::string_view::npos ? std::string_view{} : head.substr(eol + 2);
  while (!rest.empty()) {
    const std::size_t e = rest.find("\r\n");
    const std::string_view h = rest.substr(0, e);
    rest = e == std::string_view::npos ? std::string_view{} : rest.substr(e + 2);
    const std::size_t colon = h.find(':');
    if (colon == std::string_view::npos || colon == 0) return HttpParse::Bad;
    const std::string_view name = h.substr(0, colon);
    const std::string_view value = trim(h.substr(colon + 1));
    if (iequals(name, "authorization"))
      req.authorization = std::string(value);
    else if (iequals(name, "transfer-encoding"))
      return HttpParse::Bad;
    else if (iequals(name, "content-length") && value != "0")
      return HttpParse::Bad;
  }
  *out = std::move(req);
  return HttpParse::Ok;
}

std::string http_response(int status, std::string_view body, std::string_view content_type) {
  std::string s = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\n";
  s += "Content-Type: " + std::string(content_type) + "\r\n";
  s += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  s += "Cache-Control: no-store\r\nConnection: close\r\n";
  if (status == 401) s += "WWW-Authenticate: Bearer\r\n";
  s += "\r\n";
  s += body;
  return s;
}

}  // namespace ombra::server
