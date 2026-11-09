// Host side of the GPU query expansion (kernels: core/kernels/ntt.cu and keyswitch.cu).
// Internal code returns Status; the public functions convert to exceptions at the boundary.
//
// One split of the expansion tree on a ciphertext c (2 components, evaluation form), following
// LeveledSHEBase::EvalAutomorphismCore HYBRID (base-leveledshe.cpp:400):
//   1. digits of c1: inverse NTT of the Q towers, per digit a basis conversion to the other Q
//   towers
//      and the P towers, forward NTT of the converted towers (EvalKeySwitchPrecomputeCore)
//   2. inner product with the key + c0 * P + automorphism permutation       -> ext (QP basis)
//   3. ModDown: inverse NTT of the P towers, P -> Q conversion, forward NTT of the result fused
//      with the combine: sub = (ext - pq) * P^-1 = tau(c); even = c + sub; odd = (c - sub) *
//      X^-shift.
// All splits of one tree level share the key and the monomial, so a level (in chunks that bound the
// scratch memory) runs as one batch: every kernel launch covers all nodes of the chunk. Scratch
// buffers are laid out [node][...]; ciphertext buffers come from a free-list pool.

#include <algorithm>
#include <chrono>
#include <map>
#include <stdexcept>
#include <string>

#include "device_buffer.cuh"
#include "kernel_launch.hpp"
#include "ombra/core/cuda_check.hpp"
#include "ombra/core/expansion_device.hpp"

namespace ombra::core {
namespace {

template <typename T>
Status to_device(DeviceBuffer<T>& buf, const std::vector<T>& host) {
  OMBRA_TRY(buf.allocate(host.empty() ? 1 : host.size()));
  if (!host.empty()) {
    OMBRA_CUDA_CHECK(
        cudaMemcpy(buf.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  return {};
}

std::vector<u64> shoup_of(const std::vector<u64>& w, const std::vector<u64>& q_of) {
  std::vector<u64> out(w.size());
  for (std::size_t i = 0; i < w.size(); ++i) out[i] = shoup_quotient(w[i], q_of[i]);
  return out;
}

struct Pool {
  std::vector<DeviceBuffer<u64>> free_list;
};

struct KeyDev {
  DeviceBuffer<std::uint32_t> map;
  DeviceBuffer<u64> words;
};

// Optional stage timer for the profiling entry point (off in normal use: one branch per stage).
struct Marks {
  bool on = false;
  std::vector<cudaEvent_t> events;
  Status mark() {
    if (!on) return {};
    cudaEvent_t e;
    OMBRA_CUDA_CHECK(cudaEventCreate(&e));
    events.push_back(e);
    OMBRA_CUDA_CHECK(cudaEventRecord(e, nullptr));
    return {};
  }
  void clear() {
    for (cudaEvent_t e : events) cudaEventDestroy(e);
    events.clear();
  }
  ~Marks() { clear(); }
};

void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("expansion: ") + what);
}

constexpr std::size_t kScratchBudgetBytes = std::size_t{1} << 30;  // default scratch cap per chunk
constexpr std::size_t kMaxBatchCeiling = 4096;

}  // namespace

// A ciphertext buffer; returns itself to the pool when the last handle goes away.
class DeviceCt {
 public:
  DeviceCt(std::shared_ptr<Pool> pool, DeviceBuffer<u64>&& buf)
      : pool_(std::move(pool)), buf_(std::move(buf)) {}
  ~DeviceCt() { pool_->free_list.push_back(std::move(buf_)); }
  DeviceCt(const DeviceCt&) = delete;
  DeviceCt& operator=(const DeviceCt&) = delete;
  u64* data() const { return buf_.get(); }

 private:
  std::shared_ptr<Pool> pool_;
  DeviceBuffer<u64> buf_;
};

struct ExpansionDevice::Impl {
  ExpansionTables t;
  u64 n = 0, nq = 0, np = 0, qp = 0, nd = 0;
  std::vector<TowerConst> tcs;
  std::shared_ptr<Pool> pool = std::make_shared<Pool>();

  DeviceBuffer<TowerConst> tc;
  DeviceBuffer<u64> psi, psi_shoup, psi_inv, psi_inv_shoup, p_mod_q, p_inv_mod_q, p_inv_shoup;
  DeviceBuffer<u64> mat, mat_shoup;
  DeviceBuffer<std::uint32_t> digit_of_tower;
  DeviceBuffer<ConvJob> jobs_up, jobs_down;
  DeviceBuffer<NttRow> rows_inv_q, rows_fwd_dig, rows_inv_p, rows_fwd_pq;
  u32 n_rows_inv_q = 0, n_rows_fwd_dig = 0, n_rows_inv_p = 0, n_rows_fwd_pq = 0;
  // Scratch for `cap` nodes: Y [nq][N], digits [nd][qp][N], ext [2][qp][N], pq [2][nq][N] per node,
  // and three device pointer arrays [in | even | odd], each `cap` entries.
  DeviceBuffer<u64> ybuf, dbuf, ext, pq;
  DeviceBuffer<u64*> ptrs;
  std::size_t cap = 0;
  std::size_t max_batch = 1;
  std::map<std::uint32_t, KeyDev> keys;
  std::map<std::size_t, DeviceBuffer<u64>> monos;
  Marks marks;

  NttTables ntt() const {
    return {psi.get(), psi_shoup.get(), psi_inv.get(), psi_inv_shoup.get(), tc.get(), n};
  }
  KsDims dims() const { return {n, nq, qp, nd}; }
  u64 ct_words() const { return 2 * nq * n; }
  u64 scratch_words_per_node() const { return nq * n + nd * qp * n + 2 * qp * n + 2 * nq * n; }
  const u64* const* in_ptrs() const { return reinterpret_cast<const u64* const*>(ptrs.get()); }
  u64* const* even_ptrs() const { return ptrs.get() + cap; }
  u64* const* odd_ptrs() const { return ptrs.get() + 2 * cap; }

  Status init() {
    tcs = make_tower_consts(t.moduli);
    // The conversion kernels rely on moduli of at least 31 bits only through the preset; the
    // Shoup arithmetic itself needs q < 2^63, which make_tower_consts (q < 2^60) guarantees.
    OMBRA_TRY(to_device(tc, tcs));
    std::vector<u64> q_of_row(t.psi_rev.size());
    for (u64 x = 0; x < qp; ++x) {
      for (u64 i = 0; i < n; ++i) q_of_row[x * n + i] = t.moduli[x];
    }
    OMBRA_TRY(to_device(psi, t.psi_rev));
    OMBRA_TRY(to_device(psi_shoup, shoup_of(t.psi_rev, q_of_row)));
    OMBRA_TRY(to_device(psi_inv, t.psi_inv_rev));
    OMBRA_TRY(to_device(psi_inv_shoup, shoup_of(t.psi_inv_rev, q_of_row)));
    OMBRA_TRY(to_device(p_mod_q, t.p_mod_q));
    OMBRA_TRY(to_device(p_inv_mod_q, t.p_inv_mod_q));
    std::vector<u64> pinv_q(nq);
    for (u64 i = 0; i < nq; ++i) pinv_q[i] = t.moduli[i];
    OMBRA_TRY(to_device(p_inv_shoup, shoup_of(t.p_inv_mod_q, pinv_q)));

    std::vector<std::uint32_t> dot(qp, 0xFFFFFFFFu);  // P towers belong to no digit
    for (u64 p = 0; p < nd; ++p) {
      for (u64 i = t.digit_first[p]; i < t.digit_first[p] + t.digit_count[p]; ++i) {
        dot[i] = (std::uint32_t)p;
      }
    }
    OMBRA_TRY(to_device(digit_of_tower, dot));

    // Basis-conversion jobs (one contiguous run of source-tower constants each) and the NTT rows
    // that consume them: forward-NTT row k of fwd_dig / fwd_pq transforms the output of job k.
    // Inverse scales fold in N^-1 and the next step's multiplier.
    const auto row = [&](u64 off, u64 scale, u64 tower, u32 job = 0) {
      return NttRow{off, scale, shoup_quotient(scale, t.moduli[tower]), (u32)tower, job};
    };
    std::vector<u64> m, m_q;
    std::vector<ConvJob> up, down;
    std::vector<NttRow> inv_q, fwd_dig, inv_p, fwd_pq;
    for (u64 i = 0; i < nq; ++i) {
      inv_q.push_back(row(i * n, mul_mod(t.n_inv[i], t.part_hat_inv[i], tcs[i]), i));
    }
    for (u64 p = 0; p < nd; ++p) {
      const u64 first = t.digit_first[p], count = t.digit_count[p], ncomp = qp - count;
      for (u64 jj = 0; jj < ncomp; ++jj) {
        const u64 dst =
            jj < first ? jj : jj + count;  // ascending Q towers outside the digit, then P
        fwd_dig.push_back(row((p * qp + dst) * n, 0, dst, (u32)up.size()));
        up.push_back({first * n, m.size(), (u32)count, 0});
        for (u64 i = 0; i < count; ++i) {
          m.push_back(t.part_hat_mod[p][i * ncomp + jj] % t.moduli[dst]);
          m_q.push_back(t.moduli[dst]);
        }
      }
    }
    for (u64 r = 0; r < 2; ++r) {
      for (u64 j = 0; j < np; ++j) {
        const u64 x = nq + j;
        inv_p.push_back(
            row((r * qp + x) * n, mul_mod(t.n_inv[x], t.p_hat_inv_mod_p[j], tcs[x]), x));
      }
      for (u64 i = 0; i < nq; ++i) {
        fwd_pq.push_back(row((r * nq + i) * n, 0, i, (u32)down.size()));
        down.push_back({(r * qp + nq) * n, m.size(), (u32)np, 0});
        for (u64 j = 0; j < np; ++j) {
          m.push_back(t.p_hat_mod_q[j * nq + i] % t.moduli[i]);
          m_q.push_back(t.moduli[i]);
        }
      }
    }
    OMBRA_TRY(to_device(mat, m));
    OMBRA_TRY(to_device(mat_shoup, shoup_of(m, m_q)));
    OMBRA_TRY(to_device(jobs_up, up));
    OMBRA_TRY(to_device(jobs_down, down));
    n_rows_inv_q = (u32)inv_q.size();
    n_rows_fwd_dig = (u32)fwd_dig.size();
    n_rows_inv_p = (u32)inv_p.size();
    n_rows_fwd_pq = (u32)fwd_pq.size();
    OMBRA_TRY(to_device(rows_inv_q, inv_q));
    OMBRA_TRY(to_device(rows_fwd_dig, fwd_dig));
    OMBRA_TRY(to_device(rows_inv_p, inv_p));
    OMBRA_TRY(to_device(rows_fwd_pq, fwd_pq));

    max_batch = std::clamp<std::size_t>(kScratchBudgetBytes / (scratch_words_per_node() * 8), 1,
                                        kMaxBatchCeiling);
    return ensure_scratch(1);
  }

  // Grows the scratch to hold `nodes` nodes (never shrinks).
  Status ensure_scratch(std::size_t nodes) {
    if (nodes <= cap) return {};
    OMBRA_CUDA_CHECK(cudaDeviceSynchronize());  // the old buffers may still be in use
    OMBRA_TRY(ybuf.allocate(nodes * nq * n));
    OMBRA_TRY(dbuf.allocate(nodes * nd * qp * n));
    OMBRA_TRY(ext.allocate(nodes * 2 * qp * n));
    OMBRA_TRY(pq.allocate(nodes * 2 * nq * n));
    OMBRA_TRY(ptrs.allocate(3 * nodes));
    cap = nodes;
    return {};
  }

  Status acquire(CtHandle& out) {
    DeviceBuffer<u64> buf;
    if (!pool->free_list.empty()) {
      buf = std::move(pool->free_list.back());
      pool->free_list.pop_back();
    } else {
      OMBRA_TRY(buf.allocate(ct_words()));
    }
    out = std::make_shared<DeviceCt>(pool, std::move(buf));
    return {};
  }

  // Uploads the per-node pointers of a chunk of `count` nodes (odd may be empty).
  Status set_ptrs(const std::vector<const u64*>& in, const std::vector<u64*>& even,
                  const std::vector<u64*>& odd) {
    const std::size_t count = in.size();
    OMBRA_CUDA_CHECK(cudaMemcpyAsync(ptrs.get(), in.data(), count * sizeof(u64*),
                                     cudaMemcpyHostToDevice, nullptr));
    if (!even.empty()) {
      OMBRA_CUDA_CHECK(cudaMemcpyAsync(ptrs.get() + cap, even.data(), count * sizeof(u64*),
                                       cudaMemcpyHostToDevice, nullptr));
    }
    if (!odd.empty()) {
      OMBRA_CUDA_CHECK(cudaMemcpyAsync(ptrs.get() + 2 * cap, odd.data(), count * sizeof(u64*),
                                       cudaMemcpyHostToDevice, nullptr));
    }
    return {};
  }

  // Step 1 for `b` nodes (in_ptrs set): the converted towers of the digits of the second
  // components into dbuf. A digit's own towers are not copied: the inner product reads them from
  // the ciphertext. The inverse NTT reads component 1 straight from the ciphertexts.
  Status digits(u32 b) {
    OMBRA_CUDA_CHECK(launch_ntt_inverse(ybuf.get(), rows_inv_q.get(), n_rows_inv_q, ntt(), nullptr,
                                        {b, nq * n}, NttSource{in_ptrs(), nq * n}));
    OMBRA_TRY(marks.mark());
    const ConvLoad up{ybuf.get(), nq * n, mat.get(), mat_shoup.get(), jobs_up.get()};
    OMBRA_CUDA_CHECK(launch_ntt_forward(dbuf.get(), rows_fwd_dig.get(), n_rows_fwd_dig, ntt(),
                                        nullptr, {b, nd * qp * n}, &up));
    OMBRA_TRY(marks.mark());
    return {};
  }

  // Step 2: ext = permuted (digits . key + c0 * P).
  Status inner(u32 b, const KeyDev& key) {
    OMBRA_CUDA_CHECK(launch_ks_inner_product(dbuf.get(), key.words.get(), in_ptrs(), key.map.get(),
                                             p_mod_q.get(), digit_of_tower.get(), ext.get(),
                                             tc.get(), dims(), b, nullptr));
    OMBRA_TRY(marks.mark());
    return {};
  }

  // Step 3 + combine: ModDown(ext) and the tree's combine. even_ptrs (and odd_ptrs if with_odd)
  // are set; mono is the monomial of the level (only read when with_odd).
  Status mod_down_combine(u32 b, const u64* mono, bool with_odd) {
    OMBRA_CUDA_CHECK(launch_ntt_inverse(ext.get(), rows_inv_p.get(), n_rows_inv_p, ntt(), nullptr,
                                        {b, 2 * qp * n}));
    OMBRA_TRY(marks.mark());
    const ConvLoad down{ext.get(), 2 * qp * n, mat.get(), mat_shoup.get(), jobs_down.get()};
    CombineArgs ca{in_ptrs(),
                   ext.get(),
                   2 * qp * n,
                   p_inv_mod_q.get(),
                   p_inv_shoup.get(),
                   mono,
                   even_ptrs(),
                   with_odd ? odd_ptrs() : nullptr,
                   nq,
                   qp};
    OMBRA_CUDA_CHECK(launch_ntt_forward_combine(pq.get(), rows_fwd_pq.get(), n_rows_fwd_pq, ntt(),
                                                ca, nullptr, {b, 2 * nq * n}, &down));
    OMBRA_TRY(marks.mark());
    return {};
  }

  Status copy_in(u64* dst, const std::vector<u64>& host) {
    OMBRA_CUDA_CHECK(
        cudaMemcpy(dst, host.data(), host.size() * sizeof(u64), cudaMemcpyHostToDevice));
    return {};
  }
  Status copy_out(std::vector<u64>& host, const u64* src, std::size_t count) {
    host.assign(count, 0);
    OMBRA_CUDA_CHECK(cudaMemcpy(host.data(), src, count * sizeof(u64), cudaMemcpyDeviceToHost));
    return {};
  }

  const KeyDev& key_for(std::uint32_t galois) const {
    const auto it = keys.find(galois);
    require(it != keys.end(), "no key for this Galois element");
    return it->second;
  }

  // One chunk of a level: nodes [first, first + b) of `in`.
  Status split_chunk(const std::vector<CtHandle>& in, std::size_t first, std::size_t b,
                     const KeyDev& key, const u64* mono, const std::vector<char>& want_odd,
                     std::vector<std::pair<CtHandle, CtHandle>>& out) {
    OMBRA_TRY(ensure_scratch(b));
    std::vector<const u64*> hin(b);
    std::vector<u64*> heven(b), hodd(b, nullptr);
    bool any_odd = false;
    for (std::size_t k = 0; k < b; ++k) {
      CtHandle even, odd;
      OMBRA_TRY(acquire(even));
      if (want_odd[first + k]) {
        OMBRA_TRY(acquire(odd));
        hodd[k] = odd->data();
        any_odd = true;
      }
      hin[k] = in[first + k]->data();
      heven[k] = even->data();
      out[first + k] = {std::move(even), std::move(odd)};
    }
    OMBRA_TRY(set_ptrs(hin, heven, any_odd ? hodd : std::vector<u64*>{}));
    OMBRA_TRY(digits((u32)b));
    OMBRA_TRY(inner((u32)b, key));
    OMBRA_TRY(mod_down_combine((u32)b, mono, any_odd));
    OMBRA_CUDA_CHECK_LAUNCH();
    return {};
  }
};

ExpansionDevice::ExpansionDevice(const ExpansionTables& tables) : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.t = tables;
  s.n = tables.ring_dim;
  s.nq = tables.num_q;
  s.np = tables.num_p;
  s.qp = s.nq + s.np;
  s.nd = tables.digit_first.size();
  require(s.n >= 2 && (s.n & (s.n - 1)) == 0, "ring dimension must be a power of two");
  require(s.nq >= 1 && s.np >= 1 && s.nd >= 1, "empty basis or no digits");
  require(tables.moduli.size() == s.qp && tables.psi_rev.size() == s.qp * s.n &&
              tables.psi_inv_rev.size() == s.qp * s.n && tables.n_inv.size() == s.qp,
          "NTT table sizes");
  require(tables.digit_count.size() == s.nd && tables.part_hat_mod.size() == s.nd &&
              tables.part_hat_inv.size() == s.nq,
          "digit table sizes");
  std::size_t covered = 0;
  for (std::size_t p = 0; p < s.nd; ++p) {
    require(tables.digit_first[p] == covered && tables.digit_count[p] >= 1,
            "digits must tile the Q towers in order");
    covered += tables.digit_count[p];
    require(tables.part_hat_mod[p].size() == tables.digit_count[p] * (s.qp - tables.digit_count[p]),
            "digit conversion matrix size");
  }
  require(covered == s.nq, "digits must cover all Q towers");
  require(tables.p_mod_q.size() == s.nq && tables.p_inv_mod_q.size() == s.nq &&
              tables.p_hat_inv_mod_p.size() == s.np && tables.p_hat_mod_q.size() == s.np * s.nq,
          "ModDown table sizes");
  s.init().throw_if_error();
}

ExpansionDevice::~ExpansionDevice() = default;

void ExpansionDevice::add_key(std::uint32_t galois, const std::vector<std::uint32_t>& auto_map,
                              const std::vector<std::uint64_t>& key_words) {
  Impl& s = *impl_;
  require(auto_map.size() == s.n, "automorphism map size");
  require(key_words.size() == s.nd * 2 * s.qp * s.n, "key size");
  KeyDev k;
  const auto up = [&]() -> Status {
    OMBRA_TRY(to_device(k.map, auto_map));
    OMBRA_TRY(to_device(k.words, key_words));
    return {};
  };
  up().throw_if_error();
  s.keys[galois] = std::move(k);
}

void ExpansionDevice::add_monomial(std::size_t level, const std::vector<std::uint64_t>& words) {
  Impl& s = *impl_;
  require(words.size() == s.nq * s.n, "monomial size");
  DeviceBuffer<u64> b;
  to_device(b, words).throw_if_error();
  s.monos[level] = std::move(b);
}

CtHandle ExpansionDevice::upload(const std::vector<std::uint64_t>& words) {
  Impl& s = *impl_;
  require(words.size() == s.ct_words(), "ciphertext size");
  for (std::size_t i = 0; i < words.size(); ++i) {
    require(words[i] < s.tcs[(i / s.n) % s.nq].q, "unreduced input word");
  }
  CtHandle h;
  const auto go = [&]() -> Status {
    OMBRA_TRY(s.acquire(h));
    return s.copy_in(h->data(), words);
  };
  go().throw_if_error();
  return h;
}

std::vector<std::uint64_t> ExpansionDevice::download(const CtHandle& ct) {
  std::vector<u64> out;
  impl_->copy_out(out, ct->data(), impl_->ct_words()).throw_if_error();
  return out;
}

void ExpansionDevice::synchronize() {
  const auto go = []() -> Status {
    OMBRA_CUDA_CHECK(cudaDeviceSynchronize());
    return {};
  };
  go().throw_if_error();
}

void ExpansionDevice::release_pool() {
  // Buffers of dropped handles sit in the free list; freeing them returns the memory to the driver.
  // Handles still alive are untouched (they return to the now empty list when dropped).
  const auto go = [this]() -> Status {
    OMBRA_CUDA_CHECK(cudaDeviceSynchronize());  // pending kernels may still read freed handles
    impl_->pool->free_list.clear();
    return {};
  };
  go().throw_if_error();
}

std::size_t ExpansionDevice::ct_bytes() const { return impl_->ct_words() * sizeof(u64); }

std::size_t ExpansionDevice::scratch_bytes(std::size_t nodes) const {
  return (impl_->scratch_words_per_node() + 3) * nodes * sizeof(u64);
}

void ExpansionDevice::set_max_batch(std::size_t nodes) {
  require(nodes >= 1 && nodes <= kMaxBatchCeiling, "max batch out of range");
  impl_->max_batch = nodes;
}

std::size_t ExpansionDevice::max_batch() const { return impl_->max_batch; }

std::vector<std::pair<CtHandle, CtHandle>> ExpansionDevice::split_level(
    const std::vector<CtHandle>& nodes, std::uint32_t galois, std::size_t level,
    const std::vector<char>& want_odd) {
  Impl& s = *impl_;
  require(nodes.size() == want_odd.size(), "split_level sizes");
  const KeyDev& key = s.key_for(galois);
  const u64* mono = nullptr;
  if (std::any_of(want_odd.begin(), want_odd.end(), [](char w) { return w != 0; })) {
    const auto it = s.monos.find(level);
    require(it != s.monos.end(), "no monomial for this level");
    mono = it->second.get();
  }
  std::vector<std::pair<CtHandle, CtHandle>> out(nodes.size());
  const auto go = [&]() -> Status {
    for (std::size_t first = 0; first < nodes.size(); first += s.max_batch) {
      const std::size_t b = std::min(s.max_batch, nodes.size() - first);
      OMBRA_TRY(s.split_chunk(nodes, first, b, key, mono, want_odd, out));
    }
    return {};
  };
  go().throw_if_error();
  return out;
}

std::pair<CtHandle, CtHandle> ExpansionDevice::split(const CtHandle& c, std::uint32_t galois,
                                                     std::size_t level, bool want_odd) {
  auto r = split_level({c}, galois, level, {static_cast<char>(want_odd)});
  return std::move(r[0]);
}

SplitProfile ExpansionDevice::profile_level(const CtHandle& c, std::uint32_t galois,
                                            std::size_t level, std::size_t nodes, int reps) {
  Impl& s = *impl_;
  require(reps >= 1 && nodes >= 1 && nodes <= s.max_batch, "profile arguments");
  SplitProfile pr;
  pr.nodes = nodes;
  pr.stage = {"inverse_ntt_q_from_ct", "forward_ntt_digits_with_conv_up", "inner_product",
              "inverse_ntt_p", "forward_ntt_pq_with_conv_down_and_combine"};
  pr.stage_us.assign(pr.stage.size(), 0.0);
  using Clock = std::chrono::steady_clock;
  const auto us = [](Clock::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
  };
  const std::vector<CtHandle> in(nodes, c);
  const std::vector<char> odd(nodes, 1);
  const auto run = [&](auto&& fn) {
    const auto go = [&]() -> Status { return fn(); };
    go().throw_if_error();
  };
  split_level(in, galois, level, odd);  // warm-up (loads kernels, fills the pool and scratch)
  synchronize();
  // (1) stage times: one level with events between the stages, synchronized, `reps` times.
  for (int r = 0; r < reps; ++r) {
    cudaEvent_t start;
    run([&]() -> Status {
      OMBRA_CUDA_CHECK(cudaEventCreate(&start));
      s.marks.on = true;
      OMBRA_CUDA_CHECK(cudaEventRecord(start, nullptr));
      return {};
    });
    split_level(in, galois, level, odd);
    synchronize();
    run([&]() -> Status {
      s.marks.on = false;
      if (s.marks.events.size() != pr.stage.size()) {
        return Status(cudaErrorUnknown, "profile stage count", __FILE__, __LINE__);
      }
      cudaEvent_t prev = start;
      for (std::size_t i = 0; i < s.marks.events.size(); ++i) {
        float ms = 0;
        OMBRA_CUDA_CHECK(cudaEventElapsedTime(&ms, prev, s.marks.events[i]));
        pr.stage_us[i] += 1000.0 * ms / reps;
        prev = s.marks.events[i];
      }
      s.marks.clear();
      OMBRA_CUDA_CHECK(cudaEventDestroy(start));
      return {};
    });
  }
  // (2) host time to enqueue one batch, and (3) `reps` batches back to back (what the tree does).
  auto t0 = Clock::now();
  for (int r = 0; r < reps; ++r) split_level(in, galois, level, odd);
  pr.host_enqueue_us = us(Clock::now() - t0) / reps;
  synchronize();
  pr.pipelined_us = us(Clock::now() - t0) / reps;
  // (4) one synchronized batch from the host's point of view.
  t0 = Clock::now();
  for (int r = 0; r < reps; ++r) {
    split_level(in, galois, level, odd);
    synchronize();
  }
  pr.synced_us = us(Clock::now() - t0) / reps;
  // (5) cost of one empty kernel launch, back to back.
  run([&]() -> Status {
    OMBRA_CUDA_CHECK(launch_noop(nullptr));  // warm-up
    OMBRA_CUDA_CHECK(cudaDeviceSynchronize());
    const auto t1 = Clock::now();
    for (int r = 0; r < 2000; ++r) OMBRA_CUDA_CHECK(launch_noop(nullptr));
    OMBRA_CUDA_CHECK(cudaDeviceSynchronize());
    pr.noop_launch_us = us(Clock::now() - t1) / 2000;
    return {};
  });
  return pr;
}

std::vector<std::uint64_t> ExpansionDevice::test_ntt(bool forward, const std::vector<u64>& rows,
                                                     const std::vector<std::uint32_t>& tower) {
  Impl& s = *impl_;
  require(rows.size() == tower.size() * s.n && !tower.empty(), "test_ntt sizes");
  std::vector<NttRow> list;
  for (std::size_t r = 0; r < tower.size(); ++r) {
    require(tower[r] < s.qp, "test_ntt tower");
    const u64 scale = s.t.n_inv[tower[r]];
    list.push_back({r * s.n, scale, shoup_quotient(scale, s.t.moduli[tower[r]]), tower[r], 0});
  }
  DeviceBuffer<u64> data;
  DeviceBuffer<NttRow> dev_rows;
  std::vector<u64> out;
  const auto go = [&]() -> Status {
    OMBRA_TRY(to_device(data, rows));
    OMBRA_TRY(to_device(dev_rows, list));
    if (forward) {
      OMBRA_CUDA_CHECK(
          launch_ntt_forward(data.get(), dev_rows.get(), (u32)list.size(), s.ntt(), nullptr));
    } else {
      OMBRA_CUDA_CHECK(
          launch_ntt_inverse(data.get(), dev_rows.get(), (u32)list.size(), s.ntt(), nullptr));
    }
    OMBRA_CUDA_CHECK(cudaDeviceSynchronize());
    return s.copy_out(out, data.get(), rows.size());
  };
  go().throw_if_error();
  return out;
}

std::vector<std::uint64_t> ExpansionDevice::test_digits(const std::vector<u64>& ct) {
  Impl& s = *impl_;
  CtHandle h = upload(ct);
  std::vector<u64> out;
  const auto go = [&]() -> Status {
    OMBRA_TRY(s.set_ptrs({h->data()}, {}, {}));
    OMBRA_TRY(s.digits(1));
    // The hot path never copies a digit's own towers; do it here so the whole digit set is checked.
    OMBRA_CUDA_CHECK(launch_ks_copy_own_digits(s.in_ptrs(), s.dbuf.get(), s.digit_of_tower.get(),
                                               s.dims(), 1, nullptr));
    OMBRA_CUDA_CHECK(cudaDeviceSynchronize());
    return s.copy_out(out, s.dbuf.get(), s.nd * s.qp * s.n);
  };
  go().throw_if_error();
  return out;
}

std::vector<std::uint64_t> ExpansionDevice::test_ext(const std::vector<u64>& ct,
                                                     std::uint32_t galois) {
  Impl& s = *impl_;
  const KeyDev& key = s.key_for(galois);
  CtHandle h = upload(ct);
  std::vector<u64> out;
  const auto go = [&]() -> Status {
    OMBRA_TRY(s.set_ptrs({h->data()}, {}, {}));
    OMBRA_TRY(s.digits(1));
    OMBRA_TRY(s.inner(1, key));
    OMBRA_CUDA_CHECK(cudaDeviceSynchronize());
    return s.copy_out(out, s.ext.get(), 2 * s.qp * s.n);
  };
  go().throw_if_error();
  return out;
}

std::vector<std::uint64_t> ExpansionDevice::test_mod_down(const std::vector<u64>& ext_in) {
  Impl& s = *impl_;
  require(ext_in.size() == 2 * s.qp * s.n, "test_mod_down size");
  std::vector<u64> out;
  CtHandle zero, result;
  const auto go = [&]() -> Status {
    OMBRA_TRY(s.copy_in(s.ext.get(), ext_in));
    // ModDown result = (ext_Q - NTT(pq)) * P^-1: the fused combine with a zero ciphertext gives it.
    OMBRA_TRY(s.acquire(zero));
    OMBRA_TRY(s.acquire(result));
    OMBRA_CUDA_CHECK(cudaMemset(zero->data(), 0, s.ct_words() * sizeof(u64)));
    OMBRA_TRY(s.set_ptrs({zero->data()}, {result->data()}, {}));
    OMBRA_TRY(s.mod_down_combine(1, nullptr, false));
    OMBRA_CUDA_CHECK(cudaDeviceSynchronize());
    return s.copy_out(out, result->data(), s.ct_words());
  };
  go().throw_if_error();
  return out;
}

std::vector<std::uint64_t> ExpansionDevice::test_automorphism(const std::vector<u64>& ct,
                                                              std::uint32_t galois) {
  Impl& s = *impl_;
  CtHandle h = upload(ct);
  const auto r = split(h, galois, 0, false);
  synchronize();
  std::vector<u64> ev = download(r.first);
  // even = ct + tau(ct): recover tau(ct) = even - ct on the host, mod each tower.
  for (std::size_t i = 0; i < ev.size(); ++i) {
    ev[i] = sub_mod(ev[i], ct[i], s.tcs[(i / s.n) % s.nq].q);
  }
  return ev;
}

}  // namespace ombra::core
