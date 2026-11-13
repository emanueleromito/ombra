// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Seeded stored columns (envelope kind 4, docs/decisions/0023) in the storage layer: accepted at
// insert, replace and segment read, structural refusals, kind 2 and kind 4 mixed. Synthetic data:
// the moduli are odd numbers below 2^60 used only as format inputs, not real RNS primes.
#include "ombra/core/seeded_ciphertext.hpp"
#include "ombra/storage/segment.hpp"
#include "ombra/storage/store.hpp"
#include "test_util.hpp"

using namespace ombra::storage;
using namespace ombra::storage::testing;
namespace core = ombra::core;

static core::SeededPayload payload(std::uint32_t n = 8, std::uint16_t t = 2) {
  core::SeededPayload p;
  p.seed.fill(7);
  p.ring_dim = n;
  p.towers = t;
  p.slots = 4;
  p.noise_scale_deg = 1;
  p.scaling_factor = 1099511627776.0;
  p.key_tag = "tag-synthetic";
  for (std::uint16_t i = 0; i < t; ++i) p.moduli.push_back(1000003ull + 2 * i);
  p.c0.assign(static_cast<std::size_t>(t) * n, 5);
  return p;
}

static std::vector<std::uint8_t> seeded_column(const std::vector<std::uint8_t>& payload_bytes,
                                               const std::string& preset = kPreset) {
  return core::write_envelope({core::EnvelopeKind::SeededCiphertext, preset, kOfhe, payload_bytes});
}

static std::vector<std::uint8_t> seeded_column() {
  return seeded_column(core::write_seeded_payload(payload()));
}

int main() {
  TempDir dir;
  const std::string path = dir.path + "/s.oseg";

  // A segment of kind-4 columns, and a mixed one, round-trip byte-identical.
  const auto big = seeded_column(core::write_seeded_payload(payload(16, 3)));
  SegmentData sd{1, {"a", "b"}, {seeded_column(), big}};
  write_segment(path, sd);
  CHECK(read_segment(path, kPreset, kOfhe).column_envelopes == sd.column_envelopes);
  SegmentData mixed{1, {"a"}, {column(1), seeded_column(), column(2, 10)}};
  write_segment(path, mixed);
  CHECK(read_segment(path, kPreset, kOfhe).column_envelopes == mixed.column_envelopes);
  CHECK_THROWS(read_segment(path, "other_preset", kOfhe));

  // Broken payloads inside a valid container (CRC correct) are refused on read and on insert.
  const auto good = core::write_seeded_payload(payload());
  std::vector<std::vector<std::uint8_t>> bad;
  auto b = good;
  b[0] = 9;  // payload version
  bad.push_back(seeded_column(b));
  b = good;
  b[2] = 9;  // generator
  bad.push_back(seeded_column(b));
  b = good;
  b.pop_back();  // size does not match N and T
  bad.push_back(seeded_column(b));
  b = good;
  b.push_back(0);  // trailing bytes
  bad.push_back(seeded_column(b));
  b.assign(good.begin(), good.begin() + 20);  // truncated header
  bad.push_back(seeded_column(b));
  auto v1 = seeded_column();
  v1[4] = 1;  // kind 4 in a version-1 envelope
  bad.push_back(v1);

  Store s(dir.path + "/store", kOfhe);
  s.create_collection(spec());
  for (const auto& c : bad) {
    SegmentData x{0, {"x"}, {c}};
    write_segment(path, x);
    CHECK_THROWS(read_segment(path, kPreset, kOfhe));
    CHECK_THROWS(s.insert("docs", 0, {"x"}, {c}));
  }
  CHECK_THROWS(s.insert("docs", 0, {"x"}, {seeded_column(good, "m0_depth2")}));

  // Header fields are compared with the collection at insert (audit A-71): key tag, ring
  // dimension and slots (when the preset shape is known), level, noise degree, word range.
  {
    s.set_shape_provider([](const std::string&, std::uint32_t* ring, std::uint32_t* slots) {
      *ring = 8;
      *slots = 4;
      return true;
    });
    auto mod = [&](auto f) {
      auto p = payload();
      f(p);
      return seeded_column(core::write_seeded_payload(p));
    };
    CHECK_THROWS(s.insert("docs", 0, {"x"}, {mod([](core::SeededPayload& p) { p.key_tag = "other"; })}));
    CHECK_THROWS(s.insert("docs", 0, {"x"}, {mod([](core::SeededPayload& p) { p.level = 1; })}));
    CHECK_THROWS(s.insert("docs", 0, {"x"}, {mod([](core::SeededPayload& p) { p.noise_scale_deg = 2; })}));
    CHECK_THROWS(s.insert("docs", 0, {"x"}, {mod([](core::SeededPayload& p) { p.slots = 2; })}));
    CHECK_THROWS(s.insert("docs", 0, {"x"}, {mod([](core::SeededPayload& p) { p.c0[0] = p.moduli[0]; })}));
    CHECK_THROWS(s.insert("docs", 0, {"x"}, {seeded_column(core::write_seeded_payload(payload(16, 3)))}));
    s.set_shape_provider(nullptr);
  }

  // Insert and replace accept kind 4 and kind 2 together; the stored bytes are unchanged.
  const std::string seg = s.insert("docs", 0, {"a", "b"}, {seeded_column(), column(1)});
  CHECK(s.read_segment("docs", seg).column_envelopes ==
        std::vector<std::vector<std::uint8_t>>({seeded_column(), column(1)}));
  CHECK_THROWS(s.replace_segments("docs", {seg}, {0, {"a", "b"}, {bad[0], column(1)}}));
  SegmentData neu{0, {"a", "b"}, {column(3), big}};
  const std::string seg2 = s.replace_segments("docs", {seg}, neu);
  CHECK(s.read_segment("docs", seg2).column_envelopes == neu.column_envelopes);
  return 0;
}
