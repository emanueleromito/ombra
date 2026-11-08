// Kernel family: pointwise modular add / multiply on RNS towers in evaluation (NTT) form.
//
// Representation (OpenFHE v1.6.1, NATIVE_SIZE=64 build): one coefficient = one uint64_t word,
// every tower has its own modulus q < 2^60 taken from the ciphertext, never hard-coded here.
// In evaluation form, polynomial multiplication mod (X^N+1) is a pointwise product per tower.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ombra/core/cuda_check.hpp"
#include "ombra/core/rns_basic.hpp"

namespace ombra::core {
namespace {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

// Per-tower constants for Barrett reduction. For q with bit length n (2 <= q < 2^n, n <= 60):
// mu = floor(2^(2n) / q) < 2^(n+1) <= 2^61, so it fits a 64-bit word.
struct TowerConst {
    u64 q;
    u64 mu;
    u64 n;  // bit length of q
};

constexpr int kBlock = 256;

// Computes out[i] = (a[i] + b[i % b_period]) mod q(i) for i in [0, total).
// q(i) = tc[(i / ring_dim) % num_towers].q.  Inputs must be reduced (< q), so a + b < 2^61 and
// one conditional subtraction suffices; no overflow is possible for q < 2^60.
// Launch shape: 1-D grid of ceil(total / kBlock) blocks (capped, grid-stride loop), kBlock threads
// per block, no shared memory. Memory: a, b, out are device arrays of 64-bit words; tc has
// num_towers entries (read-only, global memory).
__global__ void rns_add_kernel(const u64* __restrict__ a, const u64* __restrict__ b,
                               u64* __restrict__ out, const TowerConst* __restrict__ tc,
                               u64 total, u64 ring_dim, u64 num_towers, u64 b_period) {
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < total;
         i += (u64)gridDim.x * blockDim.x) {
        const u64 q = tc[(i / ring_dim) % num_towers].q;
        u64 r = a[i] + b[i % b_period];
        if (r >= q) r -= q;
        out[i] = r;
    }
}

// Computes out[i] = (a[i] * b[i % b_period]) mod q(i) for i in [0, total), q(i) as above.
// The 128-bit product x < q^2 < 2^(2n) is reduced with Barrett reduction (HAC 14.42, b=2, k=n):
//   q1 = x >> (n-1); q3 = (q1 * mu) >> (n+1); r = x - q3*q  (0 <= r < 3q), then up to 2 subtracts.
// Correct for any q < 2^60; r < 3q < 2^62 so the low 64 bits of x - q3*q are exact.
// Launch shape: as rns_add_kernel. Memory: as rns_add_kernel.
__global__ void rns_mul_kernel(const u64* __restrict__ a, const u64* __restrict__ b,
                               u64* __restrict__ out, const TowerConst* __restrict__ tc,
                               u64 total, u64 ring_dim, u64 num_towers, u64 b_period) {
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < total;
         i += (u64)gridDim.x * blockDim.x) {
        const TowerConst c = tc[(i / ring_dim) % num_towers];
        const u128 x = (u128)a[i] * b[i % b_period];
        const u64 q1 = (u64)(x >> (c.n - 1));
        const u64 q3 = (u64)(((u128)q1 * c.mu) >> (c.n + 1));
        u64 r = (u64)x - q3 * c.q;
        if (r >= c.q) r -= c.q;
        if (r >= c.q) r -= c.q;
        out[i] = r;
    }
}

// Owning device buffer (RAII).
template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t count) {
        OMBRA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }  // destructor must not throw
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }

  private:
    T* ptr_ = nullptr;
};

std::vector<TowerConst> make_tower_consts(const std::vector<u64>& moduli) {
    std::vector<TowerConst> out;
    for (u64 q : moduli) {
        if (q < 2 || q >= (u64(1) << 60)) {
            throw std::invalid_argument("rns: modulus must satisfy 2 <= q < 2^60");
        }
        u64 n = 0;
        while ((u64(1) << n) <= q) ++n;  // q < 2^n
        const u64 mu = (u64)((u128(1) << (2 * n)) / q);
        out.push_back({q, mu, n});
    }
    return out;
}

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

template <typename Kernel>
void run(Kernel kernel, const std::vector<u64>& a, const std::vector<u64>& b, std::vector<u64>& out,
         const RnsShape& s, const std::vector<u64>& moduli, std::size_t b_period) {
    check_inputs(a, b.size(), s, moduli);
    if (b.size() != b_period) throw std::invalid_argument("rns: operand b size mismatch");
    const std::vector<TowerConst> tc = make_tower_consts(moduli);
    const std::size_t total = s.total_words();
    for (std::size_t i = 0; i < total; ++i) {
        if (a[i] >= moduli[(i / s.ring_dim) % s.num_towers]) {
            throw std::invalid_argument("rns: operand a has an unreduced word");
        }
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        if (b[i] >= moduli[(i / s.ring_dim) % s.num_towers]) {
            throw std::invalid_argument("rns: operand b has an unreduced word");
        }
    }
    out.assign(total, 0);

    DeviceBuffer<u64> d_a(total), d_b(b.size()), d_out(total);
    DeviceBuffer<TowerConst> d_tc(tc.size());
    OMBRA_CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), total * sizeof(u64), cudaMemcpyHostToDevice));
    OMBRA_CUDA_CHECK(
        cudaMemcpy(d_b.get(), b.data(), b.size() * sizeof(u64), cudaMemcpyHostToDevice));
    OMBRA_CUDA_CHECK(cudaMemcpy(d_tc.get(), tc.data(), tc.size() * sizeof(TowerConst),
                                cudaMemcpyHostToDevice));

    const u64 blocks = std::min<u64>((total + kBlock - 1) / kBlock, 65535);
    kernel<<<(unsigned)blocks, kBlock>>>(d_a.get(), d_b.get(), d_out.get(), d_tc.get(), total,
                                         s.ring_dim, s.num_towers, b_period);
    OMBRA_CUDA_CHECK_LAUNCH();

    OMBRA_CUDA_CHECK(
        cudaMemcpy(out.data(), d_out.get(), total * sizeof(u64), cudaMemcpyDeviceToHost));
}

}  // namespace

void rns_add(const std::vector<u64>& a, const std::vector<u64>& b, std::vector<u64>& out,
             const RnsShape& shape, const std::vector<u64>& moduli) {
    run(rns_add_kernel, a, b, out, shape, moduli, shape.total_words());
}

void rns_mul_by_poly(const std::vector<u64>& a, const std::vector<u64>& b, std::vector<u64>& out,
                     const RnsShape& shape, const std::vector<u64>& moduli) {
    run(rns_mul_kernel, a, b, out, shape, moduli, shape.poly_words());
}

}  // namespace ombra::core
