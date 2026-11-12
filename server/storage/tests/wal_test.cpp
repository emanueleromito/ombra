// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include "ombra/storage/wal.hpp"

#include <fcntl.h>
#include <unistd.h>

#include "ombra/storage/fsutil.hpp"
#include "test_util.hpp"

using namespace ombra::storage;
using namespace ombra::storage::testing;

int main() {
  TempDir dir;
  const std::string p = dir.path + "/wal.log";
  WalRecord a{WalType::InsertIntent, "c", 3, "seg1", {"i1", "i2"}};
  WalRecord b{WalType::InsertCommit, "c", 0, "seg1", {}};
  WalRecord c{WalType::Delete, "c", 0, "", {"i1"}};
  {
    Wal w(p);
    CHECK(w.recover().empty());
    w.append(a);
    w.append(b);
    w.append(c);
  }
  {
    Wal w(p);
    auto r = w.recover();
    CHECK(r.size() == 3);
    CHECK(r[0].type == WalType::InsertIntent && r[0].cluster == 3 && r[0].segment == "seg1" &&
          r[0].ids == a.ids);
    CHECK(r[1].type == WalType::InsertCommit && r[1].segment == "seg1");
    CHECK(r[2].type == WalType::Delete && r[2].ids == c.ids);
  }
  // Torn tail: cut the last record short; the first two survive and the file is truncated.
  auto size = read_file(p).size();
  CHECK(::truncate(p.c_str(), static_cast<off_t>(size - 2)) == 0);
  {
    Wal w(p);
    std::size_t dropped = 0;
    CHECK(w.recover(&dropped).size() == 2 && dropped > 0);
    CHECK(w.recover().size() == 2);  // stable after truncation
    w.append(c);
    CHECK(w.recover().size() == 3);
  }
  // Bad CRC in the last record's body.
  {
    auto bytes = read_file(p);
    int fd = ::open(p.c_str(), O_WRONLY);
    std::uint8_t x = static_cast<std::uint8_t>(bytes.back() ^ 0xFF);
    CHECK(::pwrite(fd, &x, 1, static_cast<off_t>(bytes.size() - 1)) == 1);
    ::close(fd);
    Wal w(p);
    CHECK(w.recover().size() == 2);
  }
  // Checkpoint.
  {
    Wal w(p);
    w.reset();
    CHECK(w.recover().empty());
  }
  // Unknown version is refused.
  {
    int fd = ::open(p.c_str(), O_WRONLY);
    std::uint8_t v = 2;
    CHECK(::pwrite(fd, &v, 1, 4) == 1);
    ::close(fd);
    CHECK_THROWS(Wal w(p));
  }
  return 0;
}
