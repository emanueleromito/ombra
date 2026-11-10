// CPU unit tests of the search layout and penalty logic. Synthetic data only.
#include <cstdio>
#include <cstdlib>

#include "ombra/search/layout.hpp"

#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                                \
    }                                                                              \
  } while (0)

#define CHECK_THROWS(expr)            \
  do {                                \
    bool threw_ = false;              \
    try {                             \
      expr;                           \
    } catch (const std::exception&) { \
      threw_ = true;                  \
    }                                 \
    CHECK(threw_);                    \
  } while (0)

using namespace ombra::search;
using ombra::storage::CollectionSpec;

static CollectionSpec spec(std::uint32_t d, std::vector<std::uint32_t> cards) {
  CollectionSpec s;
  s.dimension = d;
  for (std::uint32_t i = 0; i < cards.size(); ++i) s.filter_fields.push_back({i, cards[i]});
  return s;
}

int main() {
  // F_max from the headroom (decision 0019): m0_depth2 (60/50 bits) 14, compact (55/40 bits) 16 (tombstone penalty included).
  CHECK(max_filter_fields(60, 50) == 14);
  CHECK(max_filter_fields(55, 40) == 16);
  CHECK(max_filter_fields(52, 50) == 0);

  {
    auto l = make_layout(spec(16, {3, 4}), 8192, 16);
    CHECK(l.d == 16 && l.fields() == 2 && l.filter_leaves() == 7 && l.leaves() == 23);
    CHECK(l.center_columns() == 16 && l.data_columns() == 23);
    CHECK(l.tombstone_penalty() == -12.0);
  }
  {
    auto l = make_layout(spec(8, {}), 8192, 16);  // F = 0: penalty -4
    CHECK(l.leaves() == 8 && l.tombstone_penalty() == -4.0);
  }
  CHECK_THROWS(make_layout(spec(0, {}), 8192, 16));
  CHECK_THROWS(make_layout(spec(8, {3, 4, 5}), 8192, 2));                           // above F_max
  CHECK_THROWS(make_layout(spec(8, std::vector<std::uint32_t>(17, 2)), 8192, 99));  // above 16
  CHECK_THROWS(make_layout(spec(8, {0}), 8192, 16));                                // cardinality 0
  CHECK_THROWS(make_layout(spec(8, {65}), 8192, 16));                               // above 64
  CHECK_THROWS(make_layout(spec(4090, {7}), 8192, 16));  // D = 4097 > N/2 = 4096
  CHECK(make_layout(spec(4089, {7}), 8192, 16).leaves() == 4096);
  {
    auto s = spec(8, {3});
    s.filter_fields[0].field_id = 1;
    CHECK_THROWS(make_layout(s, 8192, 16));
  }

  {
    std::unordered_set<std::string> tomb{"b", "d", "zz"};
    CHECK(penalty_vector({"a", "c"}, tomb, 8, -8.0).empty());  // no tombstone: no plaintext needed
    auto v = penalty_vector({"a", "b", "c", "d"}, tomb, 8, -8.0);
    CHECK(v.size() == 8 && v[0] == 0 && v[1] == -8.0 && v[2] == 0 && v[3] == -8.0 && v[7] == 0);
    CHECK_THROWS(penalty_vector({"a", "b", "c"}, tomb, 2, -8.0));
  }
  std::puts("search layout tests passed");
  return 0;
}
