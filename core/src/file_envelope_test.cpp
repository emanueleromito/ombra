// Unit test of the file envelope. No OpenFHE, no GPU. Synthetic bytes only.
// argv[1] = path of tests/fixtures/envelope_v1_ciphertext.bin, written by the Python writer.
#include "ombra/core/file_envelope.hpp"

#include <fstream>
#include <iostream>
#include <iterator>

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
  bad[4] = 2;
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
