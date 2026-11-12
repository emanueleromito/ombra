// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Pointwise modular arithmetic on RNS towers (OpenFHE evaluation/NTT form), executed on the GPU.
//
// This header has no OpenFHE or CUDA dependency. Data layout (matches OpenFHE's DCRTPoly):
// a "polynomial set" is `num_polys` RNS polynomials (e.g. the 2 components of a ciphertext), each
// made of `num_towers` towers of `ring_dim` 64-bit words, stored contiguously as
// [poly][tower][coefficient]. Tower t of every polynomial uses modulus `moduli[t]`.
// All words must already be reduced (< their modulus); every modulus must satisfy 2 <= q < 2^60.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ombra::core {

struct RnsShape {
  std::size_t num_polys;
  std::size_t num_towers;
  std::size_t ring_dim;

  std::size_t poly_words() const { return num_towers * ring_dim; }
  std::size_t total_words() const { return num_polys * poly_words(); }
};

// out[i] = (a[i] + b[i]) mod q(i) for all i; a, b, out are host arrays of shape.total_words().
// Copies to the GPU, runs the kernel, copies back. Throws on invalid input or CUDA failure.
void rns_add(const std::vector<std::uint64_t>& a, const std::vector<std::uint64_t>& b,
             std::vector<std::uint64_t>& out, const RnsShape& shape,
             const std::vector<std::uint64_t>& moduli);

// out[i] = (a[i] * b[i mod shape.poly_words()]) mod q(i): `a` has shape.total_words() words and
// `b` is ONE polynomial (poly_words() words, e.g. an encoded plaintext) applied to every poly of a.
void rns_mul_by_poly(const std::vector<std::uint64_t>& a, const std::vector<std::uint64_t>& b,
                     std::vector<std::uint64_t>& out, const RnsShape& shape,
                     const std::vector<std::uint64_t>& moduli);

}  // namespace ombra::core
