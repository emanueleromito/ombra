// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// CPU unit test of the request policy: limits, preset headroom derived from the preset JSON.
#include "ombra/server/policy.hpp"

#include <cstdio>
#include <cstdlib>

#define CHECK(c)                                                                \
  do {                                                                          \
    if (!(c)) {                                                                 \
      std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

using namespace ombra::server;
using ombra::storage::CollectionSpec;

static CollectionSpec base(const std::string& preset) {
  CollectionSpec s;
  s.name = "docs";
  s.preset = preset;
  s.dimension = 16;
  s.security_level = 1;
  s.filter_fields = {{0, 3}, {1, 4}};
  s.key_tag = "tag";
  s.cluster_count = 8;
  return s;
}

static CollectionSpec with_fields(const std::string& preset, std::size_t n) {
  auto s = base(preset);
  s.filter_fields.clear();
  for (std::uint32_t i = 0; i < n; ++i) s.filter_fields.push_back({i, 2});
  return s;
}

int main() {
  const auto* d2 = find_preset("m0_depth2");
  const auto* cp = find_preset("m0_compact_n8192");
  CHECK(d2 && cp && !find_preset("nope"));
  CHECK(d2->ring_dim == 16384 && d2->batch_size == 8192 && d2->first_mod_bits == 60 &&
        d2->scale_bits == 50);
  CHECK(cp->ring_dim == 8192 && cp->batch_size == 4096 && cp->first_mod_bits == 55 &&
        cp->scale_bits == 40);
  CHECK(max_filter_fields_for(*d2) == 14 && max_filter_fields_for(*cp) == 16);

  CHECK(validate_spec(base("m0_depth2")).ok() && validate_spec(base("m0_compact_n8192")).ok());
  CHECK(validate_spec(with_fields("m0_depth2", 14)).ok());
  CHECK(validate_spec(with_fields("m0_depth2", 15)).code == Err::InvalidFilterLayout);
  CHECK(validate_spec(with_fields("m0_compact_n8192", 16)).ok());
  CHECK(validate_spec(with_fields("m0_compact_n8192", 17)).code == Err::InvalidFilterLayout);

  auto s = base("m0_depth2");
  s.api_version = 2;
  CHECK(validate_spec(s).code == Err::UnsupportedVersion);
  s = base("m0_depth2");
  s.name.clear();
  CHECK(validate_spec(s).code == Err::InvalidArgument);
  s.name = std::string(129, 'x');
  CHECK(validate_spec(s).code == Err::SizeLimit);
  s = base("other");
  CHECK(validate_spec(s).code == Err::InvalidArgument);
  s = base("m0_depth2");
  s.filter_fields[1].cardinality = 65;
  CHECK(validate_spec(s).code == Err::InvalidFilterLayout);
  s = base("m0_depth2");
  s.filter_fields[1].field_id = 5;
  CHECK(validate_spec(s).code == Err::InvalidFilterLayout);
  s = base("m0_compact_n8192");
  s.dimension = 4096;  // D = 4096 + 7 > N/2
  CHECK(validate_spec(s).code == Err::InvalidFilterLayout);
  s = base("m0_depth2");
  s.key_tag.clear();
  CHECK(validate_spec(s).code == Err::InvalidArgument);
  s = base("m0_depth2");
  s.cluster_count = 0;
  CHECK(validate_spec(s).code == Err::InvalidArgument);
  return 0;
}
