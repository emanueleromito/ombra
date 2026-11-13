// CPU unit test of the lazy column-search arithmetic (core/kernels/column_search_lazy.cuh): the
// same functions the CUDA kernel column_search_lazy_kernel calls (lazy_mac, lazy_fold,
// lazy_finish, lazy_fold_interval, mont_redc) are run on the host and compared with a plain
// 128-bit `%` reference, for random reduced inputs and for the adversarial all-(q - 1) inputs, at
// dimension counts around the fold intervals. The loop below is the kernel's loop for one block of
// one word position (fold every K dimensions, once more at the end).
//
// The moduli are arbitrary ODD numbers of the stated bit lengths, used only as arithmetic inputs
// to exercise the overflow bounds: they are not FHE parameter sets (hard rule 1 concerns those).
// No OpenFHE and no CUDA are needed.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "column_search_lazy.cuh"

using namespace ombra::core;

namespace {

int failures = 0;

std::uint64_t rng_state = 0x9E3779B97F4A7C15ull;
std::uint64_t next_u64() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

// Odd modulus with exactly `bits` bits (2^(bits-1) <= q < 2^bits).
std::uint64_t odd_modulus(unsigned bits) {
    const std::uint64_t top = std::uint64_t{1} << (bits - 1);
    return top | (next_u64() & (top - 1)) | 1;
}

struct Out {
    u64 o0, o1, o2;
};

// One (block, word position): the dimension loop of column_search_lazy_kernel.
Out lazy_dot(const std::vector<u64>& s0, const std::vector<u64>& s1, const std::vector<u64>& q0,
             const std::vector<u64>& q1, const TowerConst& c, bool check_bound) {
    LazyAcc acc = lazy_zero();
    const u64 fold = lazy_fold_interval(c);
    const u128 limit = (u128)c.q << 64;  // REDC is exact only below q * 2^64
    u64 pending = 0;
    for (std::size_t j = 0; j < s0.size(); ++j) {
        lazy_mac(acc, s0[j], s1[j], q0[j], q1[j]);
        if (check_bound && !(acc.a0 < limit && acc.a1 < limit && acc.a2 < limit)) {
            std::printf("FAIL accumulator reached q * 2^64 before a fold (q bits %llu, K %llu)\n",
                        (unsigned long long)c.n, (unsigned long long)fold);
            ++failures;
        }
        if (++pending == fold) {
            lazy_fold(acc, c);
            pending = 0;
        }
    }
    if (pending != 0) lazy_fold(acc, c);
    Out o{};
    lazy_finish(acc, c, o.o0, o.o1, o.o2);
    return o;
}

Out reference(const std::vector<u64>& s0, const std::vector<u64>& s1, const std::vector<u64>& q0,
              const std::vector<u64>& q1, u64 q) {
    u128 r0 = 0, r1 = 0, r2 = 0;
    for (std::size_t j = 0; j < s0.size(); ++j) {
        r0 = (r0 + (u128)s0[j] * q0[j]) % q;
        r1 = (r1 + (u128)s0[j] * q1[j] + (u128)s1[j] * q0[j]) % q;
        r2 = (r2 + (u128)s1[j] * q1[j]) % q;
    }
    return {(u64)r0, (u64)r1, (u64)r2};
}

void run_case(u64 q, std::size_t dims, bool adversarial, const char* label) {
    const TowerConst c = make_tower_consts({q})[0];
    std::vector<u64> s0(dims), s1(dims), q0(dims), q1(dims);
    for (std::size_t j = 0; j < dims; ++j) {
        if (adversarial) {
            s0[j] = s1[j] = q0[j] = q1[j] = q - 1;
        } else {
            s0[j] = next_u64() % q;
            s1[j] = next_u64() % q;
            q0[j] = next_u64() % q;
            q1[j] = next_u64() % q;
        }
    }
    const Out got = lazy_dot(s0, s1, q0, q1, c, true);
    const Out want = reference(s0, s1, q0, q1, q);
    if (got.o0 != want.o0 || got.o1 != want.o1 || got.o2 != want.o2) {
        std::printf("FAIL %s: q bits %llu, dims %zu, K %llu\n", label, (unsigned long long)c.n, dims,
                    (unsigned long long)lazy_fold_interval(c));
        ++failures;
    }
}

}  // namespace

int main() {
    // The fold interval: K * 4 q^2 < q * 2^64 for the largest modulus of each bit length.
    for (unsigned bits : {2u, 20u, 40u, 50u, 55u, 59u, 60u}) {
        const u64 q = bits == 60 ? (std::uint64_t{1} << 60) - 1 : (std::uint64_t{1} << bits) - 1;
        const TowerConst c = make_tower_consts({q | 1})[0];
        const u64 k = lazy_fold_interval(c);
        if (k < 1 || k > 256 || (u128)k * 4 * c.q * c.q >= ((u128)c.q << 64)) {
            std::printf("FAIL fold interval %llu too large for %u-bit modulus\n",
                        (unsigned long long)k, bits);
            ++failures;
        }
    }
    {
        const TowerConst c60 = make_tower_consts({(std::uint64_t{1} << 60) - 1})[0];
        const TowerConst c50 = make_tower_consts({(std::uint64_t{1} << 50) - 1})[0];
        if (lazy_fold_interval(c60) != 4 || lazy_fold_interval(c50) != 256) {
            std::printf("FAIL fold interval of the 60-bit (4) or 50-bit (256) tower\n");
            ++failures;
        }
    }

    const std::size_t dim_counts[] = {1, 2, 3, 4, 5, 8, 255, 256, 257, 512, 1536};
    for (unsigned bits : {60u, 55u, 50u, 40u}) {
        for (int rep = 0; rep < 3; ++rep) {
            const u64 q = rep == 0 && bits == 60 ? (std::uint64_t{1} << 60) - 1 : odd_modulus(bits);
            for (std::size_t dims : dim_counts) {
                run_case(q, dims, false, "random");
                run_case(q, dims, true, "all q - 1");
            }
        }
    }

    if (failures != 0) {
        std::printf("column search lazy arithmetic: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("column search lazy arithmetic: PASS\n");
    return 0;
}
