// Minimal check macros for the cache unit tests.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>

#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                                \
    }                                                                              \
  } while (0)

#define CHECK_THROWS(expr)                                                          \
  do {                                                                              \
    bool threw_ = false;                                                            \
    try {                                                                           \
      expr;                                                                         \
    } catch (const std::exception&) {                                               \
      threw_ = true;                                                                \
    }                                                                               \
    if (!threw_) {                                                                  \
      std::fprintf(stderr, "expected throw %s:%d: %s\n", __FILE__, __LINE__, #expr); \
      std::exit(1);                                                                 \
    }                                                                               \
  } while (0)
