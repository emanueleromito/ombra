// Project error handling for CUDA calls. Include only from CUDA translation units (.cu).
//
// Convention (core/CLAUDE.md): host code inside the library returns a `Status`; exceptions are
// thrown only at the public boundary, where a public function calls `Status::throw_if_error()`.
//
//   Status do_work(...) {                         // internal: status values
//       OMBRA_CUDA_CHECK(cudaMalloc(&p, n));       // on failure: `return Status{...}`
//       return {};
//   }
//   void public_api(...) { do_work(...).throw_if_error(); }   // boundary: exception
#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace ombra::core {

// Thrown at the boundary of the core library when a CUDA call failed.
class CudaError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Result of an internal operation. Default-constructed means success. `what` and `file` must be
// string literals (they are, when created by OMBRA_CUDA_CHECK).
class Status {
  public:
    Status() = default;
    Status(cudaError_t code, const char* what, const char* file, int line)
        : code_(code), what_(what), file_(file), line_(line) {}

    bool ok() const { return code_ == cudaSuccess; }
    cudaError_t code() const { return code_; }

    // Boundary conversion: throws CudaError if this is not a success.
    void throw_if_error() const {
        if (ok()) return;
        std::ostringstream msg;
        msg << "CUDA error at " << file_ << ":" << line_ << " in `" << what_
            << "`: " << cudaGetErrorString(code_);
        throw CudaError(msg.str());
    }

  private:
    cudaError_t code_ = cudaSuccess;
    const char* what_ = "";
    const char* file_ = "";
    int line_ = 0;
};

}  // namespace ombra::core

// Check one CUDA runtime call inside a function that returns `Status`: on failure, returns the
// failing Status from the enclosing function. Wrap every CUDA call with it.
#define OMBRA_CUDA_CHECK(call)                                                   \
    do {                                                                         \
        const cudaError_t ombra_err_ = (call);                                   \
        if (ombra_err_ != cudaSuccess) {                                         \
            return ::ombra::core::Status(ombra_err_, #call, __FILE__, __LINE__); \
        }                                                                        \
    } while (0)

// Propagate a failed Status from a callee: `OMBRA_TRY(callee(...));`.
#define OMBRA_TRY(expr)                                     \
    do {                                                    \
        const ::ombra::core::Status ombra_status_ = (expr); \
        if (!ombra_status_.ok()) return ombra_status_;      \
    } while (0)

// Check the result of a kernel launch (inside a Status-returning function). In debug builds
// (NDEBUG not defined) it also synchronizes, so asynchronous execution errors are reported at
// the launch that caused them.
#ifdef NDEBUG
#define OMBRA_CUDA_CHECK_LAUNCH() OMBRA_CUDA_CHECK(cudaGetLastError())
#else
#define OMBRA_CUDA_CHECK_LAUNCH()                  \
    do {                                           \
        OMBRA_CUDA_CHECK(cudaGetLastError());      \
        OMBRA_CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
#endif
