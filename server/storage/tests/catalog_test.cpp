#include <sqlite3.h>

#include "ombra/storage/catalog.hpp"
#include "test_util.hpp"

using namespace ombra::storage;
using namespace ombra::storage::testing;

int main() {
  TempDir dir;
  const std::string p = dir.path + "/catalog.db";
  {
    Catalog c(p);
    CHECK(c.next_segment_id() == 1 && c.next_segment_id() == 2);
    auto s = spec();
    c.create_collection(s);
    CHECK_THROWS(c.create_collection(s));  // duplicate name
    CollectionSpec back;
    CHECK(c.get_collection("docs", &back));
    CHECK(back.preset == s.preset && back.dimension == 8 && back.cluster_count == 4 &&
          back.key_tag == s.key_tag && back.security_level == 1 && back.api_version == 1);
    CHECK(back.filter_fields.size() == 2 && back.filter_fields[1].cardinality == 5);
    CHECK(!c.get_collection("nope", nullptr));

    c.add_segment("docs", {1, "s1", 2, 3, 99}, {"a", "b"});
    c.add_segment("docs", {1, "s1", 2, 3, 99}, {"a", "b"});  // idempotent
    CHECK(c.vector_ids("docs").size() == 2);
    CHECK_THROWS(c.add_segment("docs", {1, "s2", 2, 3, 99}, {"c", "a"}));  // dup id: rolled back
    CHECK(!c.has_segment("s2") && !c.id_exists("docs", "c"));
    CHECK_THROWS(c.add_segment("docs", {9, "s3", 1, 1, 1}, {"z"}));  // cluster FK

    c.tombstone("docs", {"a", "unknown"});
    c.tombstone("docs", {"a"});
    CHECK(c.tombstones("docs") == std::vector<std::string>{"a"});
    c.put_key_file("docs", "eval_mult", "k1.env");
    c.put_key_file("docs", "eval_mult", "k2.env");
    CHECK(c.key_files("docs").size() == 1 && c.key_files("docs")[0].second == "k2.env");
    sqlite3_exec(c.raw(), "UPDATE meta SET value=2 WHERE key='schema_version'", nullptr, nullptr,
                 nullptr);
  }
  CHECK_THROWS(Catalog c(p));
  {
    // Fresh catalog; tamper with the api_version instead.
    const std::string q = dir.path + "/c2.db";
    {
      Catalog c(q);
      sqlite3_exec(c.raw(), "UPDATE meta SET value=2 WHERE key='api_version'", nullptr, nullptr,
                   nullptr);
    }
    CHECK_THROWS(Catalog c(q));
  }
  return 0;
}
