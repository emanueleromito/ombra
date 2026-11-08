// Project error macros for CUDA calls. Include only from CUDA translation units (.cu).
#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace ombra::core {

// Thrown at the boundary of the core library when a CUDA call fails.
class CudaError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

inline void throw_cuda_error(cudaError_t err, const char* what, const char* file, int line) {
    std::ostringstream msg;
    msg << "CUDA error at " << file << ":" << line << " in `" << what
        << "`: " << cudaGetErrorString(err);
    throw CudaError(msg.str());
}

}  // namespace ombra::core

// Check one CUDA runtime call. Wrap every call with it.
#define OMBRA_CUDA_CHECK(call)                                                       \
    do {                                                                             \
        const cudaError_t ombra_err_ = (call);                                       \
        if (ombra_err_ != cudaSuccess) {                                             \
            ::ombra::core::throw_cuda_error(ombra_err_, #call, __FILE__, __LINE__); \
        }                                                                            \
    } while (0)

// Check the result of a kernel launch. In debug builds (NDEBUG not defined) it also synchronizes,
// so asynchronous execution errors are reported at the launch that caused them.
#ifdef NDEBUG
#define OMBRA_CUDA_CHECK_LAUNCH() OMBRA_CUDA_CHECK(cudaGetLastError())
#else
#define OMBRA_CUDA_CHECK_LAUNCH()                  \
    do {                                           \
        OMBRA_CUDA_CHECK(cudaGetLastError());      \
        OMBRA_CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
#endif
