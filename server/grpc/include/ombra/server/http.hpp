// Minimal HTTP/1.1 request parsing for the REST admin port. Network-facing, so it is small
// and strict: the header block is capped, one request per connection, no request bodies.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace ombra::server {

inline constexpr std::size_t kMaxHttpHeaderBytes = 8 * 1024;

struct HttpRequest {
  std::string method, path;
  std::string authorization;  // value of the Authorization header, empty if absent
};

enum class HttpParse { Incomplete, Ok, TooLarge, Bad };

// `buf` holds the bytes received so far. Incomplete: the blank line ending the headers has not
// arrived and the cap is not reached. TooLarge: the header block exceeds kMaxHttpHeaderBytes.
// Bad: malformed request line or header, a body (Content-Length > 0, Transfer-Encoding) or a
// version other than HTTP/1.0 or 1.1.
HttpParse parse_http_request(std::string_view buf, HttpRequest* out);

// A complete response with Connection: close.
std::string http_response(int status, std::string_view body,
                          std::string_view content_type = "application/json");

}  // namespace ombra::server
