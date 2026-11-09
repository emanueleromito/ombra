// GPU query expansion on raw RNS words (T0.5). No OpenFHE or CUDA types in this header.
//
// The host glue (expansion_gpu.hpp) extracts every table from OpenFHE objects into
// ExpansionTables; kernels in core/kernels/{ntt,keyswitch}.cu use them. Public material only:
// ciphertext words and automorphism (key-switching) keys; no secret key can enter here.
// Public functions throw std::invalid_argument for bad input and CudaError for CUDA failures.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ombra::core {

// Everything the GPU needs about the context at one ciphertext level. "QP" is the extended basis:
// the num_q Q towers of the ciphertext, then the num_p special towers P.
struct ExpansionTables {
  std::size_t ring_dim = 0;  // N
  std::size_t num_q = 0;
  std::size_t num_p = 0;
  std::vector<std::uint64_t> moduli;       // [num_q + num_p]
  std::vector<std::uint64_t> psi_rev;      // [QP tower][N], psi^i at bitrev(i)
  std::vector<std::uint64_t> psi_inv_rev;  // [QP tower][N], psi^-i at bitrev(i)
  std::vector<std::uint64_t> n_inv;        // [QP tower], N^-1 mod modulus

  // Key-switch digits (hybrid): digit p covers Q towers [digit_first[p], +digit_count[p]).
  std::vector<std::size_t> digit_first;
  std::vector<std::size_t> digit_count;
  std::vector<std::uint64_t> part_hat_inv;  // [num_q] (QHat of the tower's digit)^-1 mod q_i
  // [digit] -> [i in digit][j in complement], complement = Q towers outside the digit (ascending)
  // then the P towers; entry = QHat_i mod complement modulus j.
  std::vector<std::vector<std::uint64_t>> part_hat_mod;

  std::vector<std::uint64_t> p_mod_q;          // [num_q] P mod q_i
  std::vector<std::uint64_t> p_inv_mod_q;      // [num_q] P^-1 mod q_i
  std::vector<std::uint64_t> p_hat_inv_mod_p;  // [num_p] (P/p_j)^-1 mod p_j
  std::vector<std::uint64_t> p_hat_mod_q;      // [num_p][num_q] (P/p_j) mod q_i

  // Rescale (T1.2, dropping the last Q tower): q_{num_q-1}^-1 mod q_i for i < num_q - 1
  // (CryptoParametersCKKSRNS::GetqlInvModq). Empty if num_q == 1 (nothing to drop).
  std::vector<std::uint64_t> ql_inv_mod_q;
};

// An owned device array of 64-bit words, for the raw-pointer operations of ExpansionDevice
// (relinearize, rescale, multiply) and the GPU search pipeline. Synchronous copies.
class DeviceWords {
 public:
  explicit DeviceWords(std::size_t count);
  ~DeviceWords();
  DeviceWords(DeviceWords&&) noexcept;
  DeviceWords& operator=(DeviceWords&&) noexcept;
  DeviceWords(const DeviceWords&) = delete;
  DeviceWords& operator=(const DeviceWords&) = delete;
  std::uint64_t* data() const;
  std::size_t size() const;
  void upload(const std::vector<std::uint64_t>& words, std::size_t offset = 0);
  std::vector<std::uint64_t> download(std::size_t offset, std::size_t count) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Timing of the pieces of one split (profiling only, see ExpansionDevice::profile_split).
struct SplitProfile {
  std::size_t nodes = 0;  // splits per batch
  std::vector<std::string> stage;
  std::vector<double> stage_us;  // per stage, GPU time between events (includes launch gaps)
  double host_enqueue_us = 0;    // host time to enqueue one batch of `nodes` splits
  double pipelined_us = 0;       // wall time per batch, batches enqueued back to back
  double synced_us = 0;          // wall time per batch, synchronized after each
  double noop_launch_us = 0;     // time per empty kernel launch, back to back
};

class DeviceCt;  // one ciphertext on the device: 2 * num_q * N words
using CtHandle = std::shared_ptr<DeviceCt>;

class ExpansionDevice {
 public:
  explicit ExpansionDevice(const ExpansionTables& tables);
  ~ExpansionDevice();
  ExpansionDevice(const ExpansionDevice&) = delete;
  ExpansionDevice& operator=(const ExpansionDevice&) = delete;

  // Key for one Galois element. auto_map[j] = source index of output j (OpenFHE PrecomputeAutoMap),
  // key_words = [digit][0 = b, 1 = a][QP tower][N] in evaluation form.
  void add_key(std::uint32_t galois, const std::vector<std::uint32_t>& auto_map,
               const std::vector<std::uint64_t>& key_words);
  // Evaluation form of X^-shift in the Q towers, [num_q][N], for tree level `level`.
  void add_monomial(std::size_t level, const std::vector<std::uint64_t>& words);

  CtHandle upload(const std::vector<std::uint64_t>& words);  // [comp][Q tower][N]
  std::vector<std::uint64_t> download(const CtHandle& ct);
  // T1.3: device pointer to the words of a ciphertext ([component][Q tower][N], 2 * num_q * N
  // words), valid while a handle to it is alive. For device-to-device handoff of expansion leaves.
  const std::uint64_t* device_ptr(const CtHandle& ct) const;
  void synchronize();

  // (c + tau_g(c), (c - tau_g(c)) * X^-shift); the second is null unless want_odd. Asynchronous.
  std::pair<CtHandle, CtHandle> split(const CtHandle& c, std::uint32_t galois, std::size_t level,
                                      bool want_odd);

  // All splits of one tree level at once: node k gives (c + tau_g(c), (c - tau_g(c)) * X^-shift)
  // with the second null unless want_odd[k]. The level has one Galois element and one monomial, so
  // the tables are shared and every kernel launch covers a whole chunk of nodes (chunks bound the
  // scratch memory, see max_batch). Asynchronous. Results equal `split` node by node.
  std::vector<std::pair<CtHandle, CtHandle>> split_level(const std::vector<CtHandle>& nodes,
                                                         std::uint32_t galois, std::size_t level,
                                                         const std::vector<char>& want_odd);
  // Largest chunk of nodes per batch (default from a ~1 GiB scratch budget); tests lower it to
  // exercise chunking. The scratch grows to the largest chunk used and is never shrunk.
  // Memory planning (T0.7). Bytes of one ciphertext on the device; bytes of scratch for a chunk of
  // `nodes` nodes (what the scratch holds once it has grown to that chunk size).
  std::size_t ct_bytes() const;
  std::size_t scratch_bytes(std::size_t nodes) const;
  // Ciphertexts live in slabs of slab_slots() slots (one cudaMalloc each, about 256 MiB). slab_bytes()
  // is the driver's footprint of one slab (rounded to 2 MiB), so a ciphertext costs
  // slab_bytes() / slab_slots() bytes plus at most one partly used slab. pool_bytes() is the
  // footprint of the slabs currently allocated.
  std::size_t slab_slots() const;
  std::size_t slab_bytes() const;
  std::size_t pool_bytes() const;
  // Frees every slab without a live handle (the pool otherwise keeps its peak forever and reuses
  // it). Waits for the device. Slabs that still hold live handles are kept.
  void release_pool();
  void set_max_batch(std::size_t nodes);
  std::size_t max_batch() const;

  // Profiling only (bench option, never on the hot path): times the stages of one batch of `nodes`
  // identical splits of `c` (nodes <= max_batch), `reps` times.
  SplitProfile profile_level(const CtHandle& c, std::uint32_t galois, std::size_t level,
                             std::size_t nodes, int reps);

  // --- T1.2: relinearize, rescale and ciphertext multiply on raw device pointers. A "ciphertext"
  // here is [component][Q tower][N] words with this device's num_q towers (rescale outputs have
  // num_q - 1). All pointers are device pointers (DeviceWords::data() + offset), asynchronous on
  // the default stream. Outputs must not alias inputs.
  // Public EvalMult key (the key-switching key of s^2), in the layout of add_key's key_words.
  void set_relin_key(const std::vector<std::uint64_t>& key_words);
  // 3 -> 2 components: KeySwitchCore of component 2 with the relin key, added to components 0, 1
  // (LeveledSHEBase::RelinearizeInPlace). in3[k] has 3 * num_q * N words, out2[k] 2 * num_q * N.
  void relinearize(const std::vector<const std::uint64_t*>& in3,
                   const std::vector<std::uint64_t*>& out2);
  // Drops the last Q tower and divides by it (ModReduceInternalInPlace, one level). in2[k] is
  // 2 * num_q * N words, out2[k] 2 * (num_q - 1) * N. Needs tables.ql_inv_mod_q.
  void rescale(const std::vector<const std::uint64_t*>& in2,
               const std::vector<std::uint64_t*>& out2);
  // Tensor product of 2-component ciphertexts (EvalMultCore): out3[k] has 3 * num_q * N words.
  void multiply(const std::vector<const std::uint64_t*>& a, const std::vector<const std::uint64_t*>& b,
                const std::vector<std::uint64_t*>& out3);

  // Synchronous stage entry points for the oracle tests (copy results to the host).
  // NTT of rows [rows][N]; tower[r] is the QP tower (modulus) of row r.
  std::vector<std::uint64_t> test_ntt(bool forward, const std::vector<std::uint64_t>& rows,
                                      const std::vector<std::uint32_t>& tower);
  // Digits of the second component of `ct`: [digit][QP tower][N] (OpenFHE
  // EvalKeySwitchPrecomputeCore).
  std::vector<std::uint64_t> test_digits(const std::vector<std::uint64_t>& ct);
  // Extended-basis result before ModDown: [2][QP tower][N] (inner product, + c0 * P, automorphism).
  std::vector<std::uint64_t> test_ext(const std::vector<std::uint64_t>& ct, std::uint32_t galois);
  // ApproxModDown of an extended-basis pair [2][QP][N] -> [2][num_q][N].
  std::vector<std::uint64_t> test_mod_down(const std::vector<std::uint64_t>& ext);
  // tau_g(ct): the key-switched automorphism alone (EvalAutomorphism), [2][num_q][N].
  std::vector<std::uint64_t> test_automorphism(const std::vector<std::uint64_t>& ct,
                                               std::uint32_t galois);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ombra::core
