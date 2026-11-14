// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Helpers for the verify, backup and disk-full tests: a store with every kind of file, and a
// canonical dump of its state. Synthetic data only.
#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ombra/storage/store.hpp"
#include "test_util.hpp"

namespace ombra::storage::testing {

inline std::vector<std::vector<std::uint8_t>> cols2(std::uint8_t f) {
  return {column(f), column(static_cast<std::uint8_t>(f + 1))};
}

inline std::vector<std::uint8_t> blob_bytes(std::uint8_t fill, std::size_t n = 16384) {
  return std::vector<std::uint8_t>(n, fill);
}

inline std::vector<std::uint8_t> key_envelope(bool context, std::uint8_t fill) {
  ombra::core::Envelope e{
      context ? ombra::core::EnvelopeKind::Context : ombra::core::EnvelopeKind::PublicEvalKey,
      kPreset, kOfhe, std::vector<std::uint8_t>(48, fill)};
  return ombra::core::write_envelope(e);
}

inline std::vector<std::pair<std::string, std::vector<std::uint8_t>>> key_set(std::uint8_t fill) {
  return {{kKeyContext, key_envelope(true, fill)},
          {kKeyEvalMult, key_envelope(false, static_cast<std::uint8_t>(fill + 1))},
          {kKeyAutomorphism, key_envelope(false, static_cast<std::uint8_t>(fill + 2))}};
}

// Ids are long and distinctive so a test can check that no tool output contains them.
inline const std::vector<std::string> kA{"vec-alpha-1", "vec-alpha-2", "vec-alpha-3"};
inline const std::vector<std::string> kB{"vec-beta-1", "vec-beta-2"};
inline const std::vector<std::pair<std::uint32_t, std::string>> kBlobIds{
    {1, "blob-rec-1"}, {2, "blob-doc-1"}, {1, "blob-rec-gone"}};

// Two collections, inserts into data and center clusters, deletes, a dead-block drop, a replace,
// blobs (one replaced, one deleted), key files, and at the end an insert plus a delete so that the
// write-ahead log holds records (the last operations are ones that do not checkpoint it).
inline void fill_store(const std::string& root) {
  Store s(root, kOfhe);
  s.create_collection(spec("docs"));
  s.create_collection(spec("other"));
  s.insert("docs", 0, kA, cols2(1));
  s.insert("docs", 1, kB, cols2(5));
  s.insert("docs", 4, {"vec-center-1"}, cols2(9));  // the center block (cluster_count is 4)
  s.insert("other", 0, {"vec-other-1", "vec-other-2"}, cols2(11));
  s.remove("docs", {"vec-alpha-2", "vec-beta-1", "vec-beta-2"});
  s.drop_dead_blocks("docs");  // block B is fully deleted
  s.insert("docs", 0, {"vec-alpha-4", "vec-alpha-5"}, cols2(3));
  s.remove("docs", {"vec-alpha-5"});
  std::vector<std::string> old;
  for (auto& r : s.catalog().segments("docs"))
    if (r.cluster == 0) old.push_back(r.file);
  s.replace_segments("docs", old,
                     SegmentData{0, {"vec-alpha-3", "vec-alpha-4", "vec-alpha-1"}, cols2(13)});
  s.put_blob("docs", 1, "blob-rec-1", blob_bytes(1));
  s.put_blob("docs", 1, "blob-rec-1", blob_bytes(2));  // replaced
  s.put_blob("docs", 2, "blob-doc-1", blob_bytes(3, 262144));
  s.put_blob("docs", 1, "blob-rec-gone", blob_bytes(4));
  s.delete_blobs("docs", 1, {"blob-rec-gone"});
  s.put_key_files("docs", key_set(20));
  s.put_key_files("docs", key_set(30));  // replaced
  s.insert("docs", 1, {"vec-beta-9"}, cols2(7));
  s.remove("docs", {"vec-beta-9"});
}

// Canonical text of everything a client could see in the store: per collection the blocks (with
// their names if with_names), ids, column bytes, tombstones, key files and blobs. Test-only.
inline std::string dump_state(Store& s, bool with_names) {
  std::ostringstream o;
  for (const auto& name : s.list_collections()) {
    o << "collection " << name << "\n";
    CollectionSpec spec;
    s.get_collection(name, &spec);
    o << " clusters " << spec.cluster_count << "\n";
    std::vector<std::string> blocks;
    for (const auto& r : s.segments(name)) {
      auto sd = s.read_segment(name, r.file);
      std::ostringstream b;
      if (with_names) b << r.file << " ";
      b << "cluster " << r.cluster << " ids";
      for (const auto& id : sd.ids) b << " " << id;
      for (const auto& c : sd.column_envelopes) {
        std::uint32_t h = 0;
        for (auto x : c) h = h * 131 + x;
        b << " col" << c.size() << ":" << h;
      }
      blocks.push_back(b.str());
    }
    std::sort(blocks.begin(), blocks.end());
    for (const auto& b : blocks) o << " block " << b << "\n";
    auto t = s.tombstones(name);
    std::sort(t.begin(), t.end());
    for (const auto& id : t) o << " tomb " << id << "\n";
    for (const char* k : {kKeyContext, kKeyEvalMult, kKeyAutomorphism}) {
      std::vector<std::uint8_t> env;
      if (s.get_key_file(name, k, &env)) {
        std::uint32_t h = 0;
        for (auto x : env) h = h * 131 + x;
        o << " key " << k << " " << env.size() << ":" << h << "\n";
      }
    }
    for (const auto& [kind, id] : kBlobIds) {
      std::vector<std::uint8_t> p;
      if (s.get_blob(name, kind, id, &p)) {
        std::uint32_t h = 0;
        for (auto x : p) h = h * 131 + x;
        o << " blob " << kind << " " << id << " " << p.size() << ":" << h << "\n";
      }
    }
  }
  return o.str();
}

}  // namespace ombra::storage::testing
