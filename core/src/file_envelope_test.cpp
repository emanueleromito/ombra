// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Unit test of the file envelope. No OpenFHE, no GPU. Synthetic bytes only.
// argv[1] = tests/fixtures/envelope_v1_ciphertext.bin, argv[2] = envelope_v2_seeded.bin, both
// written by the Python writer.
#include "ombra/core/file_envelope.hpp"

#include <fstream>
#include <iostream>
#include <iterator>

#include "ombra/core/seeded_ciphertext.hpp"

using namespace ombra::core;

static int fails = 0;
#define CHECK(c)                                             \
  do {                                                       \
    if (!(c)) {                                              \
      std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; \
      ++fails;                                               \
    }                                                        \
  } while (0)

template <class F>
static bool refuses(F f) {
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

using Bytes = std::vector<std::uint8_t>;

static void rd(const Bytes& x, EnvelopeKind k = EnvelopeKind::Ciphertext,
               const char* p = "m0_depth2", const char* v = "1.2.3") {
  read_envelope(x, k, p, v);
}

int main(int argc, char** argv) {
  Envelope e{EnvelopeKind::Ciphertext, "m0_depth2", "1.2.3", {0, 1, 2, 3, 250, 255}};
  Bytes b = write_envelope(e);
  auto r = read_envelope(b, EnvelopeKind::Ciphertext, "m0_depth2", "1.2.3");
  CHECK(r.payload == e.payload && r.preset == e.preset && r.openfhe_version == "1.2.3");
  Bytes bad = b;
  bad[0] = 'X';
  CHECK(refuses([&] { rd(bad); }));
  bad = b;
  bad[4] = 3;  // unknown version
  CHECK(refuses([&] { rd(bad); }));
  CHECK(refuses([&] { rd(b, EnvelopeKind::Context); }));
  CHECK(refuses([&] { rd(b, EnvelopeKind::Ciphertext, "other"); }));
  CHECK(refuses([&] { rd(b, EnvelopeKind::Ciphertext, "m0_depth2", "9.9"); }));
  bad = b;
  bad.pop_back();
  CHECK(refuses([&] { rd(bad); }));
  bad = b;
  bad.push_back(0);
  CHECK(refuses([&] { rd(bad); }));
  bad = b;  // huge declared length must be refused without allocating
  for (int i = 0; i < 8; ++i) bad[bad.size() - 6 - 8 + i] = 0xFF;
  CHECK(refuses([&] { rd(bad); }));
  CHECK(refuses([&] { rd({}); }));
  for (std::size_t n = 0; n < b.size(); ++n)
    CHECK(refuses([&] { rd(Bytes(b.begin(), b.begin() + n)); }));

  // Version 2 and kind 4.
  CHECK(b[4] == 1 && b[5] == 0);  // kind 2 is still written as version 1
  Envelope se{EnvelopeKind::SeededCiphertext, "m0_depth2", "1.2.3", {9, 8, 7}};
  Bytes sb = write_envelope(se);
  CHECK(sb[4] == 2 && sb[5] == 0 && sb[6] == 4);
  CHECK(read_envelope(sb, EnvelopeKind::SeededCiphertext, "m0_depth2", "1.2.3").payload ==
        se.payload);
  CHECK(read_column_envelope(sb, "m0_depth2", "1.2.3").kind == EnvelopeKind::SeededCiphertext);
  CHECK(read_column_envelope(b, "m0_depth2", "1.2.3").kind == EnvelopeKind::Ciphertext);
  CHECK(refuses([&] { rd(sb); }));  // plain reader never accepts kind 4 silently
  Bytes ctx = write_envelope({EnvelopeKind::Context, "m0_depth2", "1.2.3", {1}});
  CHECK(refuses([&] { read_column_envelope(ctx, "m0_depth2", "1.2.3"); }));
  bad = sb;
  bad[4] = 1;  // kind 4 inside a version 1 envelope
  CHECK(refuses([&] { read_column_envelope(bad, "m0_depth2", "1.2.3"); }));
  CHECK(refuses([&] { read_envelope(bad, EnvelopeKind::SeededCiphertext, "m0_depth2", "1.2.3"); }));
  bad = b;
  bad[4] = 2;  // version 2 may carry kind 2
  CHECK(read_envelope(bad, EnvelopeKind::Ciphertext, "m0_depth2", "1.2.3").payload == e.payload);
  for (std::size_t n = 0; n < sb.size(); ++n)
    CHECK(refuses(
        [&] { read_column_envelope(Bytes(sb.begin(), sb.begin() + n), "m0_depth2", "1.2.3"); }));

  if (argc > 2) {  // seeded fixture: envelope and payload
    std::ifstream f(argv[2], std::ios::binary);
    Bytes fx((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto env = read_column_envelope(fx, "fixture_preset", "0.0.0-fixture");
    CHECK(env.kind == EnvelopeKind::SeededCiphertext);
    CHECK(write_envelope(env) == fx);
    SeededPayload p = read_seeded_payload(env.payload);
    CHECK(p.ring_dim == 8 && p.towers == 2 && p.level == 0 && p.noise_scale_deg == 1 &&
          p.slots == 4 && p.scaling_factor == 1099511627776.0 && p.key_tag == "fixture-tag");
    CHECK(p.moduli == (std::vector<std::uint64_t>{97, 193}));
    CHECK(p.seed[0] == 0 && p.seed[31] == 31);
    bool words_ok = p.c0.size() == 16;
    for (std::size_t i = 0; words_ok && i < 16; ++i)
      words_ok = p.c0[i] == (i < 8 ? i : 100 + i - 8);
    CHECK(words_ok);
    CHECK(write_seeded_payload(p) == env.payload);
    CHECK(check_seeded_header(env.payload).towers == 2);
  }

  if (argc > 1) {  // cross-language: identical bytes
    std::ifstream f(argv[1], std::ios::binary);
    Bytes fx((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Envelope x{EnvelopeKind::Ciphertext,
               "fixture_preset",
               "0.0.0-fixture",
               {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
    CHECK(!fx.empty());
    CHECK(fx == write_envelope(x));
    CHECK(read_envelope(fx, EnvelopeKind::Ciphertext, "fixture_preset", "0.0.0-fixture").payload ==
          x.payload);
  }
  std::cout << (fails ? "FAILED" : "file_envelope ok") << "\n";
  return fails ? 1 : 0;
}
