#include <fstream>

#include "ombra/storage/fsutil.hpp"
#include "ombra/storage/segment.hpp"
#include "test_util.hpp"

using namespace ombra::storage;
using namespace ombra::storage::testing;

static void patch(const std::string& path, std::size_t off, std::uint8_t v) {
  auto b = read_file(path);
  b.at(off) = v;
  std::ofstream(path, std::ios::binary | std::ios::trunc)
      .write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}

int main() {
  TempDir dir;
  const std::string p = dir.path + "/a.oseg";
  SegmentData sd{2, {"id-a", std::string("id\0b", 4)}, {column(1), column(2, 10)}};
  auto info = write_segment(p, sd);
  CHECK(info.n_vectors == 2 && info.n_columns == 2);

  SegmentInfo ri;
  auto back = read_segment(p, kPreset, kOfhe, &ri);
  CHECK(back.cluster == 2 && back.ids == sd.ids && back.column_envelopes == sd.column_envelopes);
  CHECK(ri.crc == info.crc);

  CHECK_THROWS(read_segment(p, "other_preset", kOfhe));
  CHECK_THROWS(read_segment(p, kPreset, "9.9"));

  patch(p, 4, 2);  // unknown container version
  CHECK_THROWS(read_segment(p, kPreset, kOfhe));
  patch(p, 4, 1);
  read_segment(p, kPreset, kOfhe);

  patch(p, 40, static_cast<std::uint8_t>(read_file(p)[40] ^ 1));  // body bit flip -> CRC
  CHECK_THROWS(read_segment(p, kPreset, kOfhe));

  patch(p, 0, 'X');  // magic
  CHECK_THROWS(read_segment(p, "x", kOfhe));

  // Truncated file.
  write_segment(p, sd);
  auto b = read_file(p);
  std::ofstream(p, std::ios::binary | std::ios::trunc)
      .write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size() - 3));
  CHECK_THROWS(read_segment(p, kPreset, kOfhe));

  // A column with an unknown envelope version, inside a valid container (CRC correct).
  auto bad = column(3);
  bad[4] = 2;
  SegmentData sd2{0, {"x"}, {column(1), bad}};
  write_segment(p, sd2);
  CHECK_THROWS(read_segment(p, kPreset, kOfhe));

  // A column of the wrong preset.
  SegmentData sd3{0, {"x"}, {column(1, 8, "m0_depth2")}};
  write_segment(p, sd3);
  CHECK_THROWS(read_segment(p, kPreset, kOfhe));
  return 0;
}
