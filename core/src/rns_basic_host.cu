// Host wrappers for the RNS add / multiply kernels (kernels: core/kernels/rns_basic.cu).
// Internal code returns Status; the public functions convert to exceptions at the boundary.

#include <algorithm>
#include <stdexcept>

#include "device_buffer.cuh"
#include "kernel_launch.hpp"
#include "ombra/core/cuda_check.hpp"
#include "ombra/core/rns_basic.hpp"

namespace ombra::core {
namespace {

using LaunchFn = cudaError_t (*)(const u64*, const u64*, u64*, const TowerConst*, u64, u64, u64,
                                 u64, cudaStream_t);

void check_inputs(const std::vector<u64>& a, std::size_t b_words, const RnsShape& s,
                  const std::vector<u64>& moduli) {
  if (s.num_polys == 0 || s.num_towers == 0 || s.ring_dim == 0) {
    throw std::invalid_argument("rns: empty shape");
  }
  if (moduli.size() != s.num_towers) {
    throw std::invalid_argument("rns: one modulus per tower is required");
  }
  if (a.size() != s.total_words() || b_words == 0) {
    throw std::invalid_argument("rns: operand size does not match shape");
  }
}

Status run_device(LaunchFn launch, const std::vector<u64>& a, const std::vector<u64>& b,
                  std::vector<u64>& out, const std::vector<TowerConst>& tc, const RnsShape& s,
                  std::size_t b_period) {
  const std::size_t total = s.total_words();
  DeviceBuffer<u64> d_a, d_b, d_out;
  DeviceBuffer<TowerConst> d_tc;
  OMBRA_TRY(d_a.allocate(total));
  OMBRA_TRY(d_b.allocate(b.size()));
  OMBRA_TRY(d_out.allocate(total));
  OMBRA_TRY(d_tc.allocate(tc.size()));
  OMBRA_CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), total * sizeof(u64), cudaMemcpyHostToDevice));
  OMBRA_CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), b.size() * sizeof(u64), cudaMemcpyHostToDevice));
  OMBRA_CUDA_CHECK(
      cudaMemcpy(d_tc.get(), tc.data(), tc.size() * sizeof(TowerConst), cudaMemcpyHostToDevice));

  OMBRA_CUDA_CHECK(launch(d_a.get(), d_b.get(), d_out.get(), d_tc.get(), total, s.ring_dim,
                          s.num_towers, b_period, nullptr));
  OMBRA_CUDA_CHECK_LAUNCH();

  out.assign(total, 0);
  OMBRA_CUDA_CHECK(
      cudaMemcpy(out.data(), d_out.get(), total * sizeof(u64), cudaMemcpyDeviceToHost));
  return {};
}

void run(LaunchFn launch, const std::vector<u64>& a, const std::vector<u64>& b,
         std::vector<u64>& out, const RnsShape& s, const std::vector<u64>& moduli,
         std::size_t b_period) {
  check_inputs(a, b.size(), s, moduli);
  if (b.size() != b_period) throw std::invalid_argument("rns: operand b size mismatch");
  const std::vector<TowerConst> tc = make_tower_consts(moduli);
  for (std::size_t i = 0; i < s.total_words(); ++i) {
    if (a[i] >= moduli[(i / s.ring_dim) % s.num_towers]) {
      throw std::invalid_argument("rns: operand a has an unreduced word");
    }
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    if (b[i] >= moduli[(i / s.ring_dim) % s.num_towers]) {
      throw std::invalid_argument("rns: operand b has an unreduced word");
    }
  }
  run_device(launch, a, b, out, tc, s, b_period).throw_if_error();  // boundary
}

}  // namespace

void rns_add(const std::vector<u64>& a, const std::vector<u64>& b, std::vector<u64>& out,
             const RnsShape& shape, const std::vector<u64>& moduli) {
  run(launch_rns_add, a, b, out, shape, moduli, shape.total_words());
}

void rns_mul_by_poly(const std::vector<u64>& a, const std::vector<u64>& b, std::vector<u64>& out,
                     const RnsShape& shape, const std::vector<u64>& moduli) {
  run(launch_rns_mul, a, b, out, shape, moduli, shape.poly_words());
}

}  // namespace ombra::core
