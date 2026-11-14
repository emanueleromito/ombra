// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Writes the seed corpora of the fuzz targets: small valid inputs, synthetic data only, built with
// the project's own writers. Usage: ombra_fuzz_seedgen <out dir>   (writes <out dir>/<target>/*).
// The committed corpus under fuzz/corpus/ is its output (see fuzz/README.md). Deterministic.
#include <sys/stat.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "fuzz_util.hpp"
#include "ombra/core/file_envelope.hpp"
#include "ombra/core/seeded_ciphertext.hpp"
#include "ombra/server/auth.hpp"
#include "ombra/storage/blob.hpp"
#include "ombra/storage/fsutil.hpp"
#include "ombra/storage/segment.hpp"
#include "ombra/storage/wal.hpp"

namespace fs = std::filesystem;
namespace core = ombra::core;
namespace st = ombra::storage;
using Bytes = std::vector<std::uint8_t>;

static fs::path g_out;

static void emit(const std::string& target, const std::string& name, const Bytes& b) {
  fs::create_directories(g_out / target);
  std::ofstream f(g_out / target / name, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}
static void emit(const std::string& target, const std::string& name, const std::string& s) {
  emit(target, name, Bytes(s.begin(), s.end()));
}
// The recovery target replays the same logs.
static void emit_wal(const std::string& name, const Bytes& b) {
  emit("wal", name, b);
  emit("recovery", name, b);
}
static Bytes with_mode(std::uint8_t mode, const Bytes& b) {
  Bytes o{mode};
  o.insert(o.end(), b.begin(), b.end());
  return o;
}

static core::SeededPayload seeded(std::uint32_t n, std::uint16_t towers, const std::string& tag) {
  core::SeededPayload p;
  p.seed.fill(0x5A);
  p.ring_dim = n;
  p.towers = towers;
  p.level = 0;
  p.noise_scale_deg = 1;
  p.slots = n / 2;
  p.scaling_factor = 1099511627776.0;
  p.key_tag = tag;
  const std::uint64_t mods[] = {12289, 40961, 65537};
  for (std::uint16_t t = 0; t < towers; ++t) p.moduli.push_back(mods[t]);
  for (std::uint16_t t = 0; t < towers; ++t)
    for (std::uint32_t i = 0; i < n; ++i) p.c0.push_back((i * 7u + t) % mods[t]);
  return p;
}

static Bytes envelope(core::EnvelopeKind k, const Bytes& payload) {
  return core::write_envelope({k, ombra::fuzz::kPreset, ombra::fuzz::kOfhe, payload});
}

static Bytes read_all(const std::string& path) { return st::read_file(path); }

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <out dir>\n", argv[0]);
    return 2;
  }
  g_out = argv[1];
  const Bytes fill(24, 0x11);

  // envelope
  const Bytes sp = core::write_seeded_payload(seeded(8, 2, "tag-synthetic"));
  emit("envelope", "context", envelope(core::EnvelopeKind::Context, fill));
  emit("envelope", "ciphertext", envelope(core::EnvelopeKind::Ciphertext, fill));
  emit("envelope", "pubkey", envelope(core::EnvelopeKind::PublicEvalKey, fill));
  emit("envelope", "seeded", envelope(core::EnvelopeKind::SeededCiphertext, sp));
  emit("envelope", "empty_payload", envelope(core::EnvelopeKind::Ciphertext, {}));

  // seeded payload
  emit("seeded", "n8_t2", sp);
  emit("seeded", "n4_t1_notag", core::write_seeded_payload(seeded(4, 1, "")));
  emit("seeded", "n16_t3", core::write_seeded_payload(seeded(16, 3, "k")));

  // segment (mode byte 0: file as is, 1: header and CRC repaired by the harness)
  st::SegmentData sd;
  sd.cluster = 2;
  sd.ids = {"id-1", "id-2", ""};
  sd.column_envelopes = {envelope(core::EnvelopeKind::Ciphertext, fill),
                         envelope(core::EnvelopeKind::SeededCiphertext, sp)};
  const Bytes seg = st::encode_segment(sd);
  emit("segment", "two_columns_m0", with_mode(0, seg));
  emit("segment", "two_columns_m1", with_mode(1, seg));
  st::SegmentData empty;
  emit("segment", "empty_m1", with_mode(1, st::encode_segment(empty)));
  // A bare envelope, for the insert-time check that runs on the same bytes.
  emit("segment", "bare_seeded_m0",
       with_mode(0, envelope(core::EnvelopeKind::SeededCiphertext, sp)));

  // blob: mode 0/1 file bytes (a small payload is refused as not a bucket size); mode 2 and 6 build
  // a valid blob around the rest of the input.
  emit("blob", "small_m0", with_mode(0, st::encode_blob(st::kBlobKindRecord, Bytes(40, 0x33))));
  emit("blob", "small_m1", with_mode(1, st::encode_blob(st::kBlobKindDocument, Bytes(40, 0x33))));
  emit("blob", "wrap_record", with_mode(2, Bytes(32, 0x44)));
  emit("blob", "wrap_document", with_mode(6, Bytes(32, 0x44)));

  // wal: a real log written by the project's writer
  {
    ombra::fuzz::TempDir tmp;
    const std::string p = tmp.path() + "/wal.log";
    const std::string s1 = "s0000000000000001.oseg", s2 = "s0000000000000002.oseg";
    {
      st::Wal w(p);
      w.append({st::WalType::InsertIntent, "docs", 1, s2, {"b1", "b2"}});
      w.append({st::WalType::InsertCommit, "docs", 0, s2, {}});
      w.append({st::WalType::Delete, "docs", 0, "", {"a1"}});
    }
    const Bytes full = read_all(p);
    emit_wal("intent_commit_delete_m0", with_mode(0, full));
    emit_wal("intent_commit_delete_m1", with_mode(1, full));
    Bytes torn(full.begin(), full.end() - 3);
    emit_wal("torn_tail_m0", with_mode(0, torn));
    {
      st::Wal w(p);
      w.reset();
      w.append({st::WalType::InsertIntent, "docs", 1, s1, {"a1", "a2"}});
    }
    emit_wal("intent_known_segment_m1", with_mode(1, read_all(p)));
    {
      st::Wal w(p);
      w.reset();
    }
    emit_wal("header_only_m0", with_mode(0, read_all(p)));
  }

  // api key file: two entries, salts fixed, token "tok-synthetic"
  {
    std::string content = std::string(ombra::server::ApiKeyStore::kHeader) + "\n";
    int i = 0;
    for (const char* scope : {"read", "admin"}) {
      std::array<std::uint8_t, 16> salt{};
      salt.fill(static_cast<std::uint8_t>(0x10 + i++));
      std::string buf(reinterpret_cast<const char*>(salt.data()), salt.size());
      buf += "tok-synthetic";
      const auto h =
          ombra::server::sha256(reinterpret_cast<const std::uint8_t*>(buf.data()), buf.size());
      content += ombra::server::to_hex(salt.data(), salt.size()) + " " +
                 ombra::server::to_hex(h.data(), h.size()) + " " + scope + "\n";
    }
    emit("api_keys", "two_entries", content);
    emit("api_keys", "header_only", std::string(ombra::server::ApiKeyStore::kHeader) + "\n");
  }

  // http requests
  emit("http", "get_metrics",
       std::string("GET /metrics HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer "
                   "tok-synthetic\r\n\r\n"));
  emit("http", "get_health", std::string("GET /healthz HTTP/1.0\r\n\r\n"));
  emit("http", "content_length_zero",
       std::string(
           "GET /metrics HTTP/1.1\r\ncontent-length: 0\r\nAUTHORIZATION:   Bearer x \r\n\r\n"));
  emit("http", "post_body", std::string("POST /metrics HTTP/1.1\r\nContent-Length: 4\r\n\r\nabcd"));

  // authorization header values
  emit("auth_header", "bearer_valid", std::string("Bearer tok-synthetic"));
  emit("auth_header", "bearer_wrong", std::string("Bearer nope"));
  emit("auth_header", "basic", std::string("Basic dXNlcjpwdw=="));
  emit("auth_header", "number", std::string("1048576"));
  emit("auth_header", "listen", std::string("127.0.0.1:8080"));
  return 0;
}
