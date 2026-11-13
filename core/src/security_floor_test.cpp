// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// CPU unit test for check_security_floor on plain numbers. No OpenFHE, no insecure context.
#include "ombra/core/security_floor.hpp"

#include <cstdio>
#include <stdexcept>

using namespace ombra::core;

static int failures = 0;

static void expect_ok(StdLevel l, std::size_t n, std::size_t q, std::size_t extra, const char* w) {
  try {
    check_security_floor(l, n, q, extra);
  } catch (const std::exception& e) {
    std::printf("FAIL (should accept) %s: %s\n", w, e.what());
    ++failures;
  }
}

static void expect_refused(StdLevel l, std::size_t n, std::size_t q, std::size_t extra,
                           const char* w) {
  try {
    check_security_floor(l, n, q, extra);
  } catch (const std::invalid_argument&) {
    return;
  }
  std::printf("FAIL (should refuse) %s\n", w);
  ++failures;
}

int main() {
  const auto c = StdLevel::kClassic128;
  expect_ok(c, 16384, 280, 0, "m0_depth2");
  expect_ok(c, 8192, 195, 0, "m0_compact_n8192");
  expect_ok(c, 16384, 438, 0, "boundary at limit");
  expect_refused(c, 16384, 439, 0, "boundary over limit");
  expect_ok(c, 8192, 218, 0, "boundary 8192");
  expect_refused(c, 8192, 219, 0, "boundary 8192 over");
  expect_refused(StdLevel::kNotSet, 16384, 280, 0, "NotSet");
  expect_refused(StdLevel::kNotSet, 16384, 1, 0, "NotSet small qp");
  expect_refused(c, 12288, 100, 0, "unknown ring dimension");
  expect_refused(c, 0, 0, 0, "zero ring dimension");
  expect_ok(StdLevel::kClassic256, 16384, 280, 0, "256 level accepted");
  // The extra limit can only lower the limit.
  expect_refused(c, 16384, 280, 1, "extra limit 1 refuses a valid preset");
  expect_refused(c, 16384, 280, 279, "extra limit just below");
  expect_ok(c, 16384, 280, 280, "extra limit equal");
  expect_refused(c, 16384, 500, 100000, "huge extra limit never accepts more");
  // Secret key distribution: ternary and Gaussian pass; sparse or unknown refuse.
  try {
    check_security_floor(c, 16384, 280, 0, SecretDist::kUniformTernary);
    check_security_floor(c, 16384, 280, 0, SecretDist::kGaussian);
  } catch (const std::exception& e) {
    std::printf("FAIL (should accept ternary and gaussian): %s\n", e.what());
    ++failures;
  }
  for (auto d : {SecretDist::kSparseTernary, SecretDist::kOther}) {
    try {
      check_security_floor(c, 16384, 280, 0, d);
      std::printf("FAIL (should refuse a weak secret distribution)\n");
      ++failures;
    } catch (const std::invalid_argument&) {
    }
  }
  if (failures) {
    std::printf("security_floor_test: %d failures\n", failures);
    return 1;
  }
  std::printf("security_floor_test: ok\n");
  return 0;
}
