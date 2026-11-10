// Boundary helper: one CUDA call under the project macro, throwing core::CudaError on failure.
#pragma once

#include "ombra/core/cuda_check.hpp"

#define OMBRA_CUDA_THROW(call)                          \
  do {                                                  \
    const ::ombra::core::Status s_ = [&]() -> ::ombra::core::Status { \
      OMBRA_CUDA_CHECK(call);                           \
      return {};                                        \
    }();                                                \
    s_.throw_if_error();                                \
  } while (0)
