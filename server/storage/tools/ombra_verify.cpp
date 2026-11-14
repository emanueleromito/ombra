// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// ombra_verify: read-only integrity check of a data directory. CPU only, no OpenFHE.
//   ombra_verify --data-dir DIR [--json] [--openfhe-version V]
// Exit 0 clean, 1 something corrupt, 2 usage error or unreadable directory. Output: counts and
// file names only, never contents or ids. Writes nothing, repairs nothing, takes no lock.
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include "ombra/storage/verify.hpp"

namespace {
void usage(FILE* f) {
  std::fputs(
      "usage: ombra_verify --data-dir DIR [--json] [--openfhe-version V]\n"
      "  Checks every segment, blob and key file, the catalog and the write-ahead log of a data\n"
      "  directory without changing anything. Exit 0: clean, 1: corrupt, 2: usage error.\n",
      f);
}
}  // namespace

int main(int argc, char** argv) {
  std::string dir;
  ombra::storage::VerifyOptions opt;
  bool json = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      usage(stdout);
      return 0;
    } else if (a == "--json") {
      json = true;
    } else if (a == "--data-dir" && i + 1 < argc) {
      dir = argv[++i];
    } else if (a == "--openfhe-version" && i + 1 < argc) {
      opt.expected_openfhe_version = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (dir.empty()) {
    usage(stderr);
    return 2;
  }
  try {
    ombra::storage::VerifyReport r;
    ombra::storage::verify_data_dir(dir, r, opt);
    const std::string out = json ? ombra::storage::report_json(r) : ombra::storage::report_text(r);
    std::fputs(out.c_str(), stdout);
    return r.clean() ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ombra_verify: %s\n", e.what());
    return 2;
  }
}
