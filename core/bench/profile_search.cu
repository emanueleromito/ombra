// ombra_profile_search (profiling only): separates the arithmetic cost of the column-packed search
// kernel from its memory cost with bench-only micro kernels, and times the real kernel as the
// reference on the same run. Synthetic data only (hash-filled words reduced mod each tower's q);
// no OpenFHE, no keys. Not part of the product; it does not change any existing kernel.
//
// Usage: ombra_profile_search [--d D] [--blocks B] [--iters N] [--warmup W] [--ref-only]
//                             [--no-verify] --ring N --bits b0,b1,...
// Default: d = 384, 768, 1536 with blocks = 24576 / d (the column-search benchmark sizes: 19.3 GB
// of stored data), ring size and tower bit sizes are given by --ring and --bits (the Modal entry
// point reads them from core/params/ckks_presets.json, preset m0_depth2: bits 60,50,50, P = 3 *
// ring words). The moduli are fresh primes of those sizes, not the exact OpenFHE ones (timing is
// value-blind).
//
// Every variant uses the launch shape of column_search_kernel: block = 256 threads,
// grid = (ceil(P / (256 * W)), ceil(blocks / 4)), 4 stored blocks per thread, W words per thread.
// Variants (template <MODE, W, LOAD>):
//   MODE 0  Barrett search body (identical arithmetic to the column-search kernel)
//   MODE 1  Montgomery search body (query pre-multiplied by R mod q on the host)
//   MODE 2  lazy search body: 128-bit accumulators, one reduction per 64 dimensions
//   MODE 3  memory only: the search kernel's loads, xor-reduced (no modular arithmetic)
//   MODE 4  mul_mod only (Barrett), operands in registers: 16 independent mul_mod per iteration
//   MODE 5  add_mod only, operands in registers
//   MODE 6  Montgomery mul only, operands in registers
//   LOAD=false: operands come from registers (loaded once, kept alive with an empty asm), so no
//   memory traffic inside the loop; LOAD=true: loaded from global memory like the real kernel.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "kernel_launch.hpp"
#include "modarith.cuh"

using namespace ombra::core;

#define PCHECK(call)                                                                       \
  do {                                                                                     \
    cudaError_t e_ = (call);                                                               \
    if (e_ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,   \
                   __LINE__);                                                              \
      std::exit(2);                                                                        \
    }                                                                                      \
  } while (0)

namespace {

constexpr int kThreads = 256;
constexpr int kTile = 4;
constexpr int kFlush = 64;  // lazy variant: dimensions between reductions

struct ProfConst {
  TowerConst t;
  u64 qinv;  // q^-1 mod 2^64 (Montgomery)
  u64 r2;    // 2^128 mod q
};

__device__ __forceinline__ u64 mont_mul(u64 a, u64 b, u64 q, u64 qinv) {
  const u128 t = (u128)a * b;
  const u64 m = (u64)t * qinv;
  const u64 hi = (u64)(t >> 64);
  const u64 mq = (u64)(((u128)m * q) >> 64);
  return hi >= mq ? hi - mq : hi + q - mq;  // a*b*2^-64 mod q, exact for a, b < q
}

// REDC of T < q * 2^64: (T - m q) / 2^64 with m = lo(T) * q^-1, exact; result T * 2^-64 mod q.
__device__ __forceinline__ u64 redc(u128 T, u64 q, u64 qinv) {
  const u64 m = (u64)T * qinv;
  const u64 hi = (u64)(T >> 64);
  const u64 mq = (u64)(((u128)m * q) >> 64);
  return hi >= mq ? hi - mq : hi + q - mq;
}

__device__ __forceinline__ u64 reduce_lazy(u128 x, u64 r64, const TowerConst& c) {
  const u64 hi = (u64)(x >> 64);
  const u64 lo = (u64)x;
  return add_mod(mul_mod(barrett_reduce((u128)hi, c), r64, c), barrett_reduce((u128)lo, c), c.q);
}

template <int W>
__device__ __forceinline__ void load_w(const u64* p, u64 (&v)[W]) {
  if constexpr (W == 1) {
    v[0] = p[0];
  } else {
    const ulonglong2 x = *reinterpret_cast<const ulonglong2*>(p);
    v[0] = x.x;
    v[1] = x.y;
  }
}

template <int W>
__device__ __forceinline__ void store_w(u64* p, const u64 (&v)[W]) {
  if constexpr (W == 1) {
    p[0] = v[0];
  } else {
    *reinterpret_cast<ulonglong2*>(p) = make_ulonglong2(v[0], v[1]);
  }
}

template <int N>
__device__ __forceinline__ void keep(u64 (&v)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+l"(v[i]));
}

// Launch: see the header comment. stored/query/out layouts as column_search.cu.
template <int MODE, int W, bool LOAD, int TILE = 4>
__global__ void __launch_bounds__(kThreads)
    prof_kernel(const u64* __restrict__ stored, const u64* __restrict__ query,
                u64* __restrict__ out, const ProfConst* __restrict__ pcs, u64 dims, u64 blocks,
                u64 ring_dim, u64 P) {
  const u64 e = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * W;
  if (e >= P) return;
  const ProfConst pc = pcs[e / ring_dim];
  const TowerConst c = pc.t;
  const u64 b0 = (u64)blockIdx.y * TILE;
  const u64 nb = min((u64)TILE, blocks - b0);

  using Acc = std::conditional_t<(MODE == 2 || MODE == 7), u128, u64>;
  Acc a0[TILE][W], a1[TILE][W], a2[TILE][W];
#pragma unroll
  for (int k = 0; k < TILE; ++k)
#pragma unroll
    for (int w = 0; w < W; ++w) a0[k][w] = a1[k][w] = a2[k][w] = 0;

  u64 r0[TILE][W], r1[TILE][W], r2[TILE][W];
#pragma unroll
  for (int k = 0; k < TILE; ++k)
#pragma unroll
    for (int w = 0; w < W; ++w) r0[k][w] = r1[k][w] = r2[k][w] = 0;
  u64 r64 = 0;
  if constexpr (MODE == 2) r64 = barrett_reduce((u128)1 << 64, c);

  u64 rq0[W], rq1[W], rs0[TILE][W], rs1[TILE][W];
  if constexpr (!LOAD) {
    load_w<W>(&query[e], rq0);
    load_w<W>(&query[P + e], rq1);
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      const u64 base = (((b0 + k) * dims) * 2) * P + e;
      load_w<W>(&stored[base], rs0[k]);
      load_w<W>(&stored[base + P], rs1[k]);
    }
  }

  for (u64 j = 0; j < dims; ++j) {
    u64 q0[W], q1[W];
    if constexpr (LOAD) {
      load_w<W>(&query[(j * 2) * P + e], q0);
      load_w<W>(&query[(j * 2 + 1) * P + e], q1);
    } else {
#pragma unroll
      for (int w = 0; w < W; ++w) {
        q0[w] = rq0[w];
        q1[w] = rq1[w];
      }
      keep<W>(q0);
      keep<W>(q1);
    }
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      if ((u64)k < nb) {
        u64 s0[W], s1[W];
        if constexpr (LOAD) {
          const u64 base = (((b0 + k) * dims + j) * 2) * P + e;
          load_w<W>(&stored[base], s0);
          load_w<W>(&stored[base + P], s1);
        } else {
#pragma unroll
          for (int w = 0; w < W; ++w) {
            s0[w] = rs0[k][w];
            s1[w] = rs1[k][w];
          }
          keep<W>(s0);
          keep<W>(s1);
        }
#pragma unroll
        for (int w = 0; w < W; ++w) {
          if constexpr (MODE == 0) {
            a0[k][w] = add_mod(a0[k][w], mul_mod(s0[w], q0[w], c), c.q);
            const u64 cross = add_mod(mul_mod(s0[w], q1[w], c), mul_mod(s1[w], q0[w], c), c.q);
            a1[k][w] = add_mod(a1[k][w], cross, c.q);
            a2[k][w] = add_mod(a2[k][w], mul_mod(s1[w], q1[w], c), c.q);
          } else if constexpr (MODE == 1) {
            a0[k][w] = add_mod(a0[k][w], mont_mul(s0[w], q0[w], c.q, pc.qinv), c.q);
            const u64 cross = add_mod(mont_mul(s0[w], q1[w], c.q, pc.qinv),
                                      mont_mul(s1[w], q0[w], c.q, pc.qinv), c.q);
            a1[k][w] = add_mod(a1[k][w], cross, c.q);
            a2[k][w] = add_mod(a2[k][w], mont_mul(s1[w], q1[w], c.q, pc.qinv), c.q);
          } else if constexpr (MODE == 2) {
            a0[k][w] += (u128)s0[w] * q0[w];
            a1[k][w] += (u128)s0[w] * q1[w] + (u128)s1[w] * q0[w];
            a2[k][w] += (u128)s1[w] * q1[w];
          } else if constexpr (MODE == 7) {
            a0[k][w] += (u128)s0[w] * q0[w];
            a1[k][w] += (u128)s0[w] * q1[w] + (u128)s1[w] * q0[w];
            a2[k][w] += (u128)s1[w] * q1[w];
          } else if constexpr (MODE == 3) {
            a0[k][w] ^= s0[w] ^ q0[w];
            a1[k][w] ^= s1[w] ^ q1[w];
          } else if constexpr (MODE == 4) {
            a0[k][w] ^= mul_mod(s0[w], q0[w], c) ^ mul_mod(s0[w], q1[w], c);
            a1[k][w] ^= mul_mod(s1[w], q0[w], c) ^ mul_mod(s1[w], q1[w], c);
          } else if constexpr (MODE == 5) {
            a0[k][w] = add_mod(a0[k][w], s0[w], c.q);
            a1[k][w] = add_mod(a1[k][w], s1[w], c.q);
            a2[k][w] = add_mod(a2[k][w], q0[w], c.q);
            a2[k][w] = add_mod(a2[k][w], q1[w], c.q);
          } else {
            a0[k][w] ^= mont_mul(s0[w], q0[w], c.q, pc.qinv) ^ mont_mul(s0[w], q1[w], c.q, pc.qinv);
            a1[k][w] ^= mont_mul(s1[w], q0[w], c.q, pc.qinv) ^ mont_mul(s1[w], q1[w], c.q, pc.qinv);
          }
        }
      }
    }
    if constexpr (MODE == 7) {
      if (((j + 1) % 8) == 0) {
#pragma unroll
        for (int k = 0; k < TILE; ++k)
#pragma unroll
          for (int w = 0; w < W; ++w) {
            r0[k][w] = add_mod(r0[k][w], redc(a0[k][w], c.q, pc.qinv), c.q);
            r1[k][w] = add_mod(r1[k][w], redc(a1[k][w], c.q, pc.qinv), c.q);
            r2[k][w] = add_mod(r2[k][w], redc(a2[k][w], c.q, pc.qinv), c.q);
            a0[k][w] = a1[k][w] = a2[k][w] = 0;
          }
      }
    }
    if constexpr (MODE == 2) {
      if (((j + 1) % kFlush) == 0) {
#pragma unroll
        for (int k = 0; k < TILE; ++k)
#pragma unroll
          for (int w = 0; w < W; ++w) {
            a0[k][w] = reduce_lazy(a0[k][w], r64, c);
            a1[k][w] = reduce_lazy(a1[k][w], r64, c);
            a2[k][w] = reduce_lazy(a2[k][w], r64, c);
          }
      }
    }
  }
#pragma unroll
  for (int k = 0; k < TILE; ++k) {
    if ((u64)k < nb) {
      u64 v0[W], v1[W], v2[W];
#pragma unroll
      for (int w = 0; w < W; ++w) {
        if constexpr (MODE == 7) {
          // dims is a multiple of 8 in the benchmarks; running sums hold R^-1 * sum
          const u64 R2 = pc.r2;
          v0[w] = mont_mul(add_mod(r0[k][w], redc(a0[k][w], c.q, pc.qinv), c.q), R2, c.q,
                           pc.qinv);
          v1[w] = mont_mul(add_mod(r1[k][w], redc(a1[k][w], c.q, pc.qinv), c.q), R2, c.q,
                           pc.qinv);
          v2[w] = mont_mul(add_mod(r2[k][w], redc(a2[k][w], c.q, pc.qinv), c.q), R2, c.q,
                           pc.qinv);
        } else if constexpr (MODE == 2) {
          v0[w] = reduce_lazy(a0[k][w], r64, c);
          v1[w] = reduce_lazy(a1[k][w], r64, c);
          v2[w] = reduce_lazy(a2[k][w], r64, c);
        } else {
          v0[w] = a0[k][w];
          v1[w] = a1[k][w];
          v2[w] = a2[k][w];
        }
      }
      const u64 o = ((b0 + k) * 3) * P + e;
      store_w<W>(&out[o], v0);
      store_w<W>(&out[o + P], v1);
      store_w<W>(&out[o + 2 * P], v2);
    }
  }
}

// Fills n words: word i belongs to tower (i % P) / ring_dim; value = hash(i, seed) mod q.
__global__ void fill_kernel(u64* p, u64 n, u64 P, u64 ring_dim, const ProfConst* pcs, u64 seed) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) {
    u64 x = i + seed * 0x9E3779B97F4A7C15ull;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    p[i] = (x >> 3) % pcs[(i % P) / ring_dim].t.q;
  }
}

// Raw instruction-rate probes: 8 independent chains per thread, ITER iterations.
// KIND 0: mad.lo.u32 (IMAD), 1: mad.wide.u32 (IMAD.WIDE), 2: add.u32 + xor (IADD3/LOP3 mix).
template <int KIND>
__global__ void rate_kernel(u32* sink, long long* clk, u32 seed, int iters) {
  u32 x[8];
  u64 y[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    x[i] = seed + threadIdx.x * 8 + i;
    y[i] = x[i];
  }
  const u32 m = seed | 1;
  const long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (KIND == 0) {
        asm volatile("mad.lo.u32 %0, %0, %1, %0;" : "+r"(x[i]) : "r"(m));
      } else if constexpr (KIND == 1) {
        asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(y[i]) : "r"(x[i]), "r"(m));
      } else {
        asm volatile("add.u32 %0, %0, %1;" : "+r"(x[i]) : "r"(m));
      }
    }
  }
  const long long t1 = clock64();
  u32 acc = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) acc ^= x[i] ^ (u32)y[i];
  if (acc == 0x12345678u) sink[0] = acc;
  if (threadIdx.x == 0) atomicMax((unsigned long long*)clk, (unsigned long long)(t1 - t0));
}

// ---------- host helpers ----------
u64 mulmod_host(u64 a, u64 b, u64 m) { return (u64)((u128)a * b % m); }
u64 powmod_host(u64 a, u64 e, u64 m) {
  u64 r = 1;
  a %= m;
  while (e) {
    if (e & 1) r = mulmod_host(r, a, m);
    a = mulmod_host(a, a, m);
    e >>= 1;
  }
  return r;
}
bool is_prime(u64 n) {
  if (n < 2) return false;
  for (u64 p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
    if (n % p == 0) return n == p;
  }
  u64 d = n - 1;
  int s = 0;
  while ((d & 1) == 0) d >>= 1, ++s;
  for (u64 a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
    u64 x = powmod_host(a, d, n);
    if (x == 1 || x == n - 1) continue;
    bool comp = true;
    for (int r = 1; r < s; ++r) {
      x = mulmod_host(x, x, n);
      if (x == n - 1) {
        comp = false;
        break;
      }
    }
    if (comp) return false;
  }
  return true;
}
// Largest prime below 2^bits, congruent to 1 mod 2N, different from those in `used`.
u64 find_prime(int bits, u64 two_n, const std::vector<u64>& used) {
  u64 q = ((u64(1) << bits) / two_n) * two_n + 1;
  while (true) {
    q -= two_n;
    if (std::find(used.begin(), used.end(), q) == used.end() && is_prime(q)) return q;
  }
}

std::vector<ProfConst> make_consts(u64 ring_dim, const std::vector<int>& bits) {
  std::vector<u64> moduli;
  for (int b : bits) moduli.push_back(find_prime(b, 2 * ring_dim, moduli));
  const auto tcs = make_tower_consts(moduli);
  std::vector<ProfConst> out;
  for (const auto& t : tcs) {
    u64 inv = t.q;  // Newton: inverse mod 2^64
    for (int i = 0; i < 6; ++i) inv *= 2 - t.q * inv;
    const u64 r1m = (u64)(((u128)1 << 64) % t.q);
    out.push_back({t, inv, mulmod_host(r1m, r1m, t.q)});
  }
  return out;
}

struct Dev {
  u64* stored = nullptr;
  u64* query = nullptr;
  u64* query_mont = nullptr;
  u64* out = nullptr;
  TowerConst* tc = nullptr;
  ProfConst* pc = nullptr;
};

double median(std::vector<float>& v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <class F>
std::vector<float> time_runs(F&& launch, int warmup, int iters) {
  cudaEvent_t a, b;
  PCHECK(cudaEventCreate(&a));
  PCHECK(cudaEventCreate(&b));
  for (int i = 0; i < warmup; ++i) launch();
  PCHECK(cudaDeviceSynchronize());
  std::vector<float> ms;
  for (int i = 0; i < iters; ++i) {
    PCHECK(cudaEventRecord(a));
    launch();
    PCHECK(cudaEventRecord(b));
    PCHECK(cudaEventSynchronize(b));
    float t = 0;
    PCHECK(cudaEventElapsedTime(&t, a, b));
    ms.push_back(t);
  }
  PCHECK(cudaGetLastError());
  PCHECK(cudaEventDestroy(a));
  PCHECK(cudaEventDestroy(b));
  return ms;
}

struct Shape {
  u64 d, blocks, ring_dim, towers, P;
};

template <int MODE, int W, bool LOAD, int TILE = 4>
void launch_prof(const Dev& dv, const Shape& s, const u64* query) {
  const dim3 grid((unsigned)((s.P / W + kThreads - 1) / kThreads),
                  (unsigned)((s.blocks + TILE - 1) / TILE));
  prof_kernel<MODE, W, LOAD, TILE>
      <<<grid, kThreads>>>(dv.stored, query, dv.out, dv.pc, s.d, s.blocks, s.ring_dim, s.P);
}

template <int MODE, int W, bool LOAD, int TILE = 4>
void report(const char* name, const Dev& dv, const Shape& s, int warmup, int iters,
            double mul_per_iter_per_k, double load_words_per_k) {
  const u64 words = s.P / W * W;
  auto ms = time_runs([&] { launch_prof<MODE, W, LOAD, TILE>(dv, s, dv.query); }, warmup, iters);
  const double t = median(ms) * 1e-3;
  // per thread-word, per dimension, per stored block: mul_per_iter_per_k operations
  const double ops = (double)words * s.blocks * s.d * mul_per_iter_per_k;
  const double stored_bytes = (double)words * s.blocks * s.d * 2 * 8;
  cudaFuncAttributes fa;
  PCHECK(cudaFuncGetAttributes(&fa, prof_kernel<MODE, W, LOAD, TILE>));
  int occ = 0;
  PCHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, prof_kernel<MODE, W, LOAD, TILE>, kThreads,
                                                       0));
  std::printf(
      "RESULT d=%llu blocks=%llu variant=%s median_ms=%.3f min_ms=%.3f max_ms=%.3f "
      "ops_per_s=%.3e stored_GBps=%.1f regs=%d blocks_per_sm=%d\n",
      (unsigned long long)s.d, (unsigned long long)s.blocks, name, t * 1e3, ms.front(), ms.back(),
      ops / t, load_words_per_k > 0 ? stored_bytes / t / 1e9 : 0.0, fa.numRegs, occ);
  std::fflush(stdout);
}

bool verify(const Dev& dv, const Shape& s, const char* name, void (*run)(const Dev&, const Shape&),
            const std::vector<u64>& ref) {
  PCHECK(cudaMemset(dv.out, 0xFF, ref.size() * 8));
  run(dv, s);
  PCHECK(cudaDeviceSynchronize());
  std::vector<u64> got(ref.size());
  PCHECK(cudaMemcpy(got.data(), dv.out, got.size() * 8, cudaMemcpyDeviceToHost));
  const bool ok = got == ref;
  std::printf("VERIFY %s %s\n", name, ok ? "match" : "MISMATCH");
  return ok;
}

template <int MODE, int W, int TILE = 4>
void run_variant(const Dev& dv, const Shape& s) {
  launch_prof<MODE, W, true, TILE>(dv, s, MODE == 1 ? dv.query_mont : dv.query);
}

void rate_probe(int sms, int clock_khz) {
  u32* sink;
  long long* clk;
  PCHECK(cudaMalloc(&sink, 4));
  PCHECK(cudaMallocManaged(&clk, 8));
  const int iters = 20000;
  const int blocks = sms * 8;  // 8 blocks x 256 threads = 2048 threads per SM
  for (int kind = 0; kind < 3; ++kind) {
    *clk = 0;
    cudaEvent_t a, b;
    PCHECK(cudaEventCreate(&a));
    PCHECK(cudaEventCreate(&b));
    auto go = [&] {
      if (kind == 0) rate_kernel<0><<<blocks, 256>>>(sink, clk, 3, iters);
      else if (kind == 1) rate_kernel<1><<<blocks, 256>>>(sink, clk, 3, iters);
      else rate_kernel<2><<<blocks, 256>>>(sink, clk, 3, iters);
    };
    go();
    PCHECK(cudaDeviceSynchronize());
    PCHECK(cudaEventRecord(a));
    go();
    PCHECK(cudaEventRecord(b));
    PCHECK(cudaEventSynchronize(b));
    float ms = 0;
    PCHECK(cudaEventElapsedTime(&ms, a, b));
    const double instr = (double)blocks * 256 * iters * 8;  // thread-instructions
    const double sec = ms * 1e-3;
    // SM clock during the run, from the longest per-block clock64 span (cycles) over elapsed time
    const double clk_hz = (double)*clk / sec;
    std::printf(
        "RATE kind=%s thread_instr_per_s=%.3e per_sm_per_cycle_at_measured_clock=%.1f "
        "measured_sm_clock_MHz=%.0f (attribute clockRate %d MHz)\n",
        kind == 0 ? "IMAD.lo32" : kind == 1 ? "IMAD.WIDE.u32" : "IADD3", instr / sec,
        instr / sec / sms / clk_hz, clk_hz / 1e6, clock_khz / 1000);
    PCHECK(cudaEventDestroy(a));
    PCHECK(cudaEventDestroy(b));
  }
  PCHECK(cudaFree(sink));
  PCHECK(cudaFree(clk));
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<u64> dlist = {384, 768, 1536};
  u64 blocks_override = 0;
  int iters = 11, warmup = 3;
  bool ref_only = false, do_verify = true, do_rate = true;
  u64 ring_dim = 0;
  std::vector<int> bits;  // ring size and modulus bit sizes come from the preset (core/params)
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--d") dlist = {std::stoull(argv[++i])};
    else if (a == "--blocks") blocks_override = std::stoull(argv[++i]);
    else if (a == "--iters") iters = std::stoi(argv[++i]);
    else if (a == "--warmup") warmup = std::stoi(argv[++i]);
    else if (a == "--ref-only") ref_only = true, do_verify = false, do_rate = false;
    else if (a == "--no-verify") do_verify = false;
    else if (a == "--ring") ring_dim = std::stoull(argv[++i]);
    else if (a == "--bits") {
      std::string l = argv[++i];
      for (std::size_t p = 0; p < l.size();) {
        bits.push_back(std::stoi(l.substr(p)));
        p = l.find(',', p) == std::string::npos ? l.size() : l.find(',', p) + 1;
      }
    }
  }
  cudaDeviceProp prop;
  PCHECK(cudaGetDeviceProperties(&prop, 0));
  int clock_khz = 0;
  PCHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  int mem_khz = 0, bus = 0;
  PCHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0));
  PCHECK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0));
  std::printf("GPU %s sms=%d clockRate_MHz=%d mem_clock_MHz=%d bus_bits=%d\n", prop.name,
              prop.multiProcessorCount, clock_khz / 1000, mem_khz / 1000, bus);

  if (ring_dim == 0 || bits.empty()) {
    std::fprintf(stderr, "--ring N and --bits b0,b1,... are required (from core/params)\n");
    return 2;
  }
  const u64 towers = bits.size(), P = ring_dim * towers;
  const auto pcs = make_consts(ring_dim, bits);
  std::vector<TowerConst> tcs;
  for (const auto& p : pcs) tcs.push_back(p.t);
  for (std::size_t i = 0; i < pcs.size(); ++i)
    std::printf("TOWER %zu q=%llu bits=%llu\n", i, (unsigned long long)pcs[i].t.q,
                (unsigned long long)pcs[i].t.n);

  Dev dv;
  PCHECK(cudaMalloc(&dv.tc, tcs.size() * sizeof(TowerConst)));
  PCHECK(cudaMalloc(&dv.pc, pcs.size() * sizeof(ProfConst)));
  PCHECK(cudaMemcpy(dv.tc, tcs.data(), tcs.size() * sizeof(TowerConst), cudaMemcpyHostToDevice));
  PCHECK(cudaMemcpy(dv.pc, pcs.data(), pcs.size() * sizeof(ProfConst), cudaMemcpyHostToDevice));

  if (do_rate) rate_probe(prop.multiProcessorCount, clock_khz);

  // ---- correctness of the prototypes against the existing kernel (small shape) ----
  if (do_verify) {
    const Shape s{48, 6, ring_dim, towers, P};  // 6 blocks: the last tile has nb = 2
    const u64 sw = s.blocks * s.d * 2 * P, qw = s.d * 2 * P;
    PCHECK(cudaMalloc(&dv.stored, sw * 8));
    PCHECK(cudaMalloc(&dv.query, qw * 8));
    PCHECK(cudaMalloc(&dv.query_mont, qw * 8));
    PCHECK(cudaMalloc(&dv.out, s.blocks * 3 * P * 8));
    fill_kernel<<<4096, 256>>>(dv.stored, sw, P, ring_dim, dv.pc, 1);
    fill_kernel<<<4096, 256>>>(dv.query, qw, P, ring_dim, dv.pc, 2);
    PCHECK(cudaDeviceSynchronize());
    std::vector<u64> hq(qw);
    PCHECK(cudaMemcpy(hq.data(), dv.query, qw * 8, cudaMemcpyDeviceToHost));
    for (u64 i = 0; i < qw; ++i) {
      const auto& t = pcs[(i % P) / ring_dim].t;
      const u64 r = (u64)(((u128)1 << 64) % t.q);  // R mod q
      hq[i] = mulmod_host(hq[i], r, t.q);
    }
    PCHECK(cudaMemcpy(dv.query_mont, hq.data(), qw * 8, cudaMemcpyHostToDevice));
    ColumnSearchDims cd{s.d, s.blocks, towers, ring_dim};
    PCHECK(launch_column_search(dv.stored, dv.query, dv.out, dv.tc, cd, 0));
    PCHECK(cudaDeviceSynchronize());
    std::vector<u64> ref(s.blocks * 3 * P);
    PCHECK(cudaMemcpy(ref.data(), dv.out, ref.size() * 8, cudaMemcpyDeviceToHost));
    bool ok = true;
    ok &= verify(dv, s, "barrett_w1", run_variant<0, 1>, ref);
    ok &= verify(dv, s, "barrett_w2", run_variant<0, 2>, ref);
    ok &= verify(dv, s, "montgomery_w1", run_variant<1, 1>, ref);
    ok &= verify(dv, s, "montgomery_w2", run_variant<1, 2>, ref);
    ok &= verify(dv, s, "lazy_w1", run_variant<2, 1>, ref);
    ok &= verify(dv, s, "lazy_w2", run_variant<2, 2>, ref);
    ok &= verify(dv, s, "lazyredc_t4", run_variant<7, 1, 4>, ref);
    ok &= verify(dv, s, "lazyredc_t2", run_variant<7, 1, 2>, ref);
    ok &= verify(dv, s, "lazyredc_w2_t2", run_variant<7, 2, 2>, ref);
    ok &= verify(dv, s, "lazyredc_t1", run_variant<7, 1, 1>, ref);
    PCHECK(cudaFree(dv.stored));
    PCHECK(cudaFree(dv.query));
    PCHECK(cudaFree(dv.query_mont));
    PCHECK(cudaFree(dv.out));
    if (!ok) {
      std::fprintf(stderr, "prototype verification failed\n");
      return 1;
    }
  }

  for (u64 d : dlist) {
    const u64 blocks = blocks_override ? blocks_override : 24576 / d;
    const Shape s{d, blocks, ring_dim, towers, P};
    const u64 sw = blocks * d * 2 * P, qw = d * 2 * P;
    PCHECK(cudaMalloc(&dv.stored, sw * 8));
    PCHECK(cudaMalloc(&dv.query, qw * 8));
    PCHECK(cudaMalloc(&dv.query_mont, qw * 8));
    PCHECK(cudaMalloc(&dv.out, blocks * 3 * P * 8));
    fill_kernel<<<8192, 256>>>(dv.stored, sw, P, ring_dim, dv.pc, 1);
    fill_kernel<<<8192, 256>>>(dv.query, qw, P, ring_dim, dv.pc, 2);
    PCHECK(cudaMemcpy(dv.query_mont, dv.query, qw * 8, cudaMemcpyDeviceToDevice));
    PCHECK(cudaDeviceSynchronize());

    // reference: the existing column-search kernel
    {
      ColumnSearchDims cd{d, blocks, towers, ring_dim};
      auto ms = time_runs(
          [&] { PCHECK(launch_column_search(dv.stored, dv.query, dv.out, dv.tc, cd, 0)); }, warmup,
          iters);
      const double t = median(ms) * 1e-3;
      const double mm = (double)P * blocks * d * 4;
      std::printf(
          "RESULT d=%llu blocks=%llu variant=REFERENCE_T0.4 median_ms=%.3f min_ms=%.3f "
          "max_ms=%.3f ops_per_s=%.3e stored_GBps=%.1f\n",
          (unsigned long long)d, (unsigned long long)blocks, t * 1e3, ms.front(), ms.back(),
          mm / t, (double)P * blocks * d * 16 / t / 1e9);
      std::fflush(stdout);
    }
    // the other two search kernels, kernel-only (before/after the optimisation): batched Q = 4 (one
    // query tile; the query buffer holds 4 copies of the query) and ciphertext x plaintext
    {
      const u64 nq = 4;
      u64 *bq = nullptr, *bout = nullptr, *plain_out = nullptr;
      PCHECK(cudaMalloc(&bq, nq * qw * 8));
      for (u64 q = 0; q < nq; ++q)
        PCHECK(cudaMemcpy(bq + q * qw, dv.query, qw * 8, cudaMemcpyDeviceToDevice));
      PCHECK(cudaMalloc(&bout, nq * blocks * 3 * P * 8));
      PCHECK(cudaMalloc(&plain_out, blocks * 2 * P * 8));
      ColumnSearchDims cd{d, blocks, towers, ring_dim};
      auto mb = time_runs(
          [&] {
            PCHECK(launch_column_search_batched(dv.stored, bq, bout, dv.tc, cd, nq, 0));
          },
          warmup, iters);
      const double tb = median(mb) * 1e-3;
      std::printf(
          "RESULT d=%llu blocks=%llu variant=BATCHED_Q4_kernel median_ms=%.3f min_ms=%.3f "
          "max_ms=%.3f ops_per_s=%.3e\n",
          (unsigned long long)d, (unsigned long long)blocks, tb * 1e3, mb.front(), mb.back(),
          (double)P * blocks * d * 4 * nq / tb);
      // plain: [block][dim][P] words, the first half of the stored buffer is big enough
      auto mp = time_runs(
          [&] {
            PCHECK(launch_plain_search(dv.stored, dv.query, plain_out, dv.tc, d, blocks, towers,
                                       ring_dim, 0));
          },
          warmup, iters);
      const double tp = median(mp) * 1e-3;
      std::printf(
          "RESULT d=%llu blocks=%llu variant=PLAIN_kernel median_ms=%.3f min_ms=%.3f "
          "max_ms=%.3f ops_per_s=%.3e\n",
          (unsigned long long)d, (unsigned long long)blocks, tp * 1e3, mp.front(), mp.back(),
          (double)P * blocks * d * 2 / tp);
      std::fflush(stdout);
      PCHECK(cudaFree(bq));
      PCHECK(cudaFree(bout));
      PCHECK(cudaFree(plain_out));
    }
    if (!ref_only) {
      // ops: mul_mod count per (word, dim, block): search bodies 4; MODE 4/6: 4; MODE 5: 4 add_mod
      report<0, 1, true>("barrett_load_w1", dv, s, warmup, iters, 4, 1);
      report<0, 1, false>("barrett_regs_w1", dv, s, warmup, iters, 4, 0);
      report<3, 1, true>("memonly_w1", dv, s, warmup, iters, 4, 1);
      report<3, 2, true>("memonly_w2", dv, s, warmup, iters, 4, 1);
      report<0, 2, true>("barrett_load_w2", dv, s, warmup, iters, 4, 1);
      report<0, 2, false>("barrett_regs_w2", dv, s, warmup, iters, 4, 0);
      report<1, 1, true>("montgomery_load_w1", dv, s, warmup, iters, 4, 1);
      report<1, 1, false>("montgomery_regs_w1", dv, s, warmup, iters, 4, 0);
      report<1, 2, true>("montgomery_load_w2", dv, s, warmup, iters, 4, 1);
      report<2, 1, true>("lazy_load_w1", dv, s, warmup, iters, 4, 1);
      report<2, 1, false>("lazy_regs_w1", dv, s, warmup, iters, 4, 0);
      report<2, 2, true>("lazy_load_w2", dv, s, warmup, iters, 4, 1);
      report<7, 1, true, 4>("lazyredc_load_t4", dv, s, warmup, iters, 4, 1);
      report<7, 1, true, 2>("lazyredc_load_t2", dv, s, warmup, iters, 4, 1);
      report<7, 1, true, 1>("lazyredc_load_t1", dv, s, warmup, iters, 4, 1);
      report<7, 2, true, 2>("lazyredc_load_w2_t2", dv, s, warmup, iters, 4, 1);
      report<7, 1, false, 4>("lazyredc_regs_t4", dv, s, warmup, iters, 4, 0);
      report<1, 1, true, 2>("montgomery_load_t2", dv, s, warmup, iters, 4, 1);
      report<3, 1, true, 2>("memonly_t2", dv, s, warmup, iters, 4, 1);
      report<4, 1, false>("mulmod_only_barrett", dv, s, warmup, iters, 4, 0);
      report<6, 1, false>("mulmod_only_montgomery", dv, s, warmup, iters, 4, 0);
      report<5, 1, false>("addmod_only", dv, s, warmup, iters, 4, 0);
    }
    PCHECK(cudaFree(dv.stored));
    PCHECK(cudaFree(dv.query));
    PCHECK(cudaFree(dv.query_mont));
    PCHECK(cudaFree(dv.out));
  }
  return 0;
}
