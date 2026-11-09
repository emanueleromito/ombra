// Host side of the GPU query expansion (kernels: core/kernels/ntt.cu and keyswitch.cu).
// Internal code returns Status; the public functions convert to exceptions at the boundary.
//
// One split of the expansion tree on a ciphertext c (2 components, evaluation form), following
// LeveledSHEBase::EvalAutomorphismCore HYBRID (base-leveledshe.cpp:400):
//   1. digits of c1: inverse NTT of the Q towers, per digit a basis conversion to the other Q towers
//      and the P towers, forward NTT of the converted towers (EvalKeySwitchPrecomputeCore)
//   2. inner product with the key + c0 * P + automorphism permutation       -> ext (QP basis)
//   3. ModDown: inverse NTT of the P towers, P -> Q conversion, forward NTT -> pq
//   4. sub = (ext - pq) * P^-1 = tau(c); even = c + sub; odd = (c - sub) * X^-shift.
// Scratch buffers are allocated once; ciphertext buffers come from a free-list pool.

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

struct Pool {
  std::vector<DeviceBuffer<u64>> free_list;
};

struct KeyDev {
  DeviceBuffer<std::uint32_t> map;
  DeviceBuffer<u64> words;
};

void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("expansion: ") + what);
}

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
  DeviceBuffer<u64> psi, psi_inv, p_mod_q, p_inv_mod_q, mat;
  DeviceBuffer<ConvJob> jobs_up, jobs_down;
  DeviceBuffer<NttRow> rows_inv_q, rows_fwd_dig, rows_inv_p, rows_fwd_pq;
  u32 num_jobs_up = 0, num_jobs_down = 0;
  u32 n_rows_inv_q = 0, n_rows_fwd_dig = 0, n_rows_inv_p = 0, n_rows_fwd_pq = 0;
  DeviceBuffer<u64> ybuf, dbuf, ext, pq;  // scratch: Y [nq][N], digits, ext, pq
  std::map<std::uint32_t, KeyDev> keys;
  std::map<std::size_t, DeviceBuffer<u64>> monos;

  NttTables ntt() const { return {psi.get(), psi_inv.get(), tc.get(), n}; }
  KsDims dims() const { return {n, nq, qp, nd}; }
  u64 ct_words() const { return 2 * nq * n; }

  Status init() {
    tcs = make_tower_consts(t.moduli);
    OMBRA_TRY(to_device(tc, tcs));
    OMBRA_TRY(to_device(psi, t.psi_rev));
    OMBRA_TRY(to_device(psi_inv, t.psi_inv_rev));
    OMBRA_TRY(to_device(p_mod_q, t.p_mod_q));
    OMBRA_TRY(to_device(p_inv_mod_q, t.p_inv_mod_q));

    // Basis-conversion matrices, one contiguous run of source-tower constants per job.
    std::vector<u64> m;
    std::vector<ConvJob> up, down;
    for (u64 p = 0; p < nd; ++p) {
      const u64 first = t.digit_first[p], count = t.digit_count[p], ncomp = qp - count;
      for (u64 jj = 0; jj < ncomp; ++jj) {
        const u64 dst = jj < first ? jj : jj + count;
        up.push_back({first * n, m.size(), (p * qp + dst) * n, (u32)count, (u32)dst});
        for (u64 i = 0; i < count; ++i) m.push_back(t.part_hat_mod[p][i * ncomp + jj]);
      }
    }
    for (u64 r = 0; r < 2; ++r) {
      for (u64 i = 0; i < nq; ++i) {
        down.push_back({(r * qp + nq) * n, m.size(), (r * nq + i) * n, (u32)np, (u32)i});
        for (u64 j = 0; j < np; ++j) m.push_back(t.p_hat_mod_q[j * nq + i]);
      }
    }
    num_jobs_up = (u32)up.size();
    num_jobs_down = (u32)down.size();
    OMBRA_TRY(to_device(mat, m));
    OMBRA_TRY(to_device(jobs_up, up));
    OMBRA_TRY(to_device(jobs_down, down));

    // NTT row lists. Inverse scales fold in N^-1 and the next step's multiplier.
    std::vector<NttRow> inv_q, fwd_dig, inv_p, fwd_pq;
    for (u64 i = 0; i < nq; ++i) {
      inv_q.push_back({i * n, mul_mod(t.n_inv[i], t.part_hat_inv[i], tcs[i]), (u32)i, 0});
    }
    for (u64 p = 0; p < nd; ++p) {
      for (u64 x = 0; x < qp; ++x) {
        if (x >= t.digit_first[p] && x < t.digit_first[p] + t.digit_count[p]) continue;
        fwd_dig.push_back({(p * qp + x) * n, 0, (u32)x, 0});
      }
    }
    for (u64 r = 0; r < 2; ++r) {
      for (u64 j = 0; j < np; ++j) {
        const u64 x = nq + j;
        inv_p.push_back(
            {(r * qp + x) * n, mul_mod(t.n_inv[x], t.p_hat_inv_mod_p[j], tcs[x]), (u32)x, 0});
      }
      for (u64 i = 0; i < nq; ++i) fwd_pq.push_back({(r * nq + i) * n, 0, (u32)i, 0});
    }
    n_rows_inv_q = (u32)inv_q.size();
    n_rows_fwd_dig = (u32)fwd_dig.size();
    n_rows_inv_p = (u32)inv_p.size();
    n_rows_fwd_pq = (u32)fwd_pq.size();
    OMBRA_TRY(to_device(rows_inv_q, inv_q));
    OMBRA_TRY(to_device(rows_fwd_dig, fwd_dig));
    OMBRA_TRY(to_device(rows_inv_p, inv_p));
    OMBRA_TRY(to_device(rows_fwd_pq, fwd_pq));

    OMBRA_TRY(ybuf.allocate(nq * n));
    OMBRA_TRY(dbuf.allocate(nd * qp * n));
    OMBRA_TRY(ext.allocate(2 * qp * n));
    OMBRA_TRY(pq.allocate(2 * nq * n));
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

  // Step 1: digits of the second component of ct into dbuf.
  Status digits(const u64* ct) {
    const u64* c1 = ct + nq * n;
    OMBRA_CUDA_CHECK(
        cudaMemcpyAsync(ybuf.get(), c1, nq * n * sizeof(u64), cudaMemcpyDeviceToDevice, nullptr));
    OMBRA_CUDA_CHECK(launch_ntt_inverse(ybuf.get(), rows_inv_q.get(), n_rows_inv_q, ntt(), nullptr));
    for (u64 p = 0; p < nd; ++p) {
      const u64 first = t.digit_first[p], count = t.digit_count[p];
      OMBRA_CUDA_CHECK(cudaMemcpyAsync(dbuf.get() + (p * qp + first) * n, c1 + first * n,
                                       count * n * sizeof(u64), cudaMemcpyDeviceToDevice, nullptr));
    }
    OMBRA_CUDA_CHECK(launch_basis_conv(ybuf.get(), dbuf.get(), mat.get(), jobs_up.get(),
                                       num_jobs_up, tc.get(), n, nullptr));
    OMBRA_CUDA_CHECK(
        launch_ntt_forward(dbuf.get(), rows_fwd_dig.get(), n_rows_fwd_dig, ntt(), nullptr));
    return {};
  }

  // Step 2: ext = permuted (digits . key + c0 * P).
  Status inner(const u64* ct, const KeyDev& key) {
    OMBRA_CUDA_CHECK(launch_ks_inner_product(dbuf.get(), key.words.get(), ct, key.map.get(),
                                             p_mod_q.get(), ext.get(), tc.get(), dims(), nullptr));
    return {};
  }

  // Step 3: pq = ModDown(ext).
  Status mod_down() {
    OMBRA_CUDA_CHECK(launch_ntt_inverse(ext.get(), rows_inv_p.get(), n_rows_inv_p, ntt(), nullptr));
    OMBRA_CUDA_CHECK(launch_basis_conv(ext.get(), pq.get(), mat.get(), jobs_down.get(),
                                       num_jobs_down, tc.get(), n, nullptr));
    OMBRA_CUDA_CHECK(
        launch_ntt_forward(pq.get(), rows_fwd_pq.get(), n_rows_fwd_pq, ntt(), nullptr));
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

std::pair<CtHandle, CtHandle> ExpansionDevice::split(const CtHandle& c, std::uint32_t galois,
                                                     std::size_t level, bool want_odd) {
  Impl& s = *impl_;
  const KeyDev& key = s.key_for(galois);
  const u64* mono = nullptr;
  if (want_odd) {
    const auto it = s.monos.find(level);
    require(it != s.monos.end(), "no monomial for this level");
    mono = it->second.get();
  }
  CtHandle even, odd;
  const auto go = [&]() -> Status {
    OMBRA_TRY(s.digits(c->data()));
    OMBRA_TRY(s.inner(c->data(), key));
    OMBRA_TRY(s.mod_down());
    OMBRA_TRY(s.acquire(even));
    if (want_odd) OMBRA_TRY(s.acquire(odd));
    OMBRA_CUDA_CHECK(launch_ks_combine(c->data(), s.ext.get(), s.pq.get(), s.p_inv_mod_q.get(),
                                       mono, even->data(), want_odd ? odd->data() : nullptr,
                                       s.tc.get(), s.dims(), nullptr));
    OMBRA_CUDA_CHECK_LAUNCH();
    return {};
  };
  go().throw_if_error();
  return {std::move(even), std::move(odd)};
}

std::vector<std::uint64_t> ExpansionDevice::test_ntt(bool forward, const std::vector<u64>& rows,
                                                     const std::vector<std::uint32_t>& tower) {
  Impl& s = *impl_;
  require(rows.size() == tower.size() * s.n && !tower.empty(), "test_ntt sizes");
  std::vector<NttRow> list;
  for (std::size_t r = 0; r < tower.size(); ++r) {
    require(tower[r] < s.qp, "test_ntt tower");
    list.push_back({r * s.n, s.t.n_inv[tower[r]], tower[r], 0});
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
    OMBRA_TRY(s.digits(h->data()));
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
    OMBRA_TRY(s.digits(h->data()));
    OMBRA_TRY(s.inner(h->data(), key));
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
    OMBRA_TRY(s.mod_down());
    // ModDown result = (ext_Q - pq) * P^-1: the combine kernel with a zero ciphertext gives it.
    OMBRA_TRY(s.acquire(zero));
    OMBRA_TRY(s.acquire(result));
    OMBRA_CUDA_CHECK(cudaMemset(zero->data(), 0, s.ct_words() * sizeof(u64)));
    OMBRA_CUDA_CHECK(launch_ks_combine(zero->data(), s.ext.get(), s.pq.get(), s.p_inv_mod_q.get(),
                                       nullptr, result->data(), nullptr, s.tc.get(), s.dims(),
                                       nullptr));
    OMBRA_CUDA_CHECK(cudaDeviceSynchronize());
    return s.copy_out(out, result->data(), s.ct_words());
  };
  go().throw_if_error();
  return out;
}

std::vector<std::uint64_t> ExpansionDevice::test_automorphism(const std::vector<u64>& ct,
                                                              std::uint32_t galois) {
  Impl& s = *impl_;
  const KeyDev& key = s.key_for(galois);
  CtHandle h = upload(ct);
  CtHandle even;
  std::vector<u64> ev;
  const auto go = [&]() -> Status {
    OMBRA_TRY(s.digits(h->data()));
    OMBRA_TRY(s.inner(h->data(), key));
    OMBRA_TRY(s.mod_down());
    OMBRA_TRY(s.acquire(even));
    OMBRA_CUDA_CHECK(launch_ks_combine(h->data(), s.ext.get(), s.pq.get(), s.p_inv_mod_q.get(),
                                       nullptr, even->data(), nullptr, s.tc.get(), s.dims(),
                                       nullptr));
    OMBRA_CUDA_CHECK(cudaDeviceSynchronize());
    return s.copy_out(ev, even->data(), s.ct_words());
  };
  go().throw_if_error();
  // even = ct + tau(ct): recover tau(ct) = even - ct on the host, mod each tower.
  for (std::size_t i = 0; i < ev.size(); ++i) {
    ev[i] = sub_mod(ev[i], ct[i], s.tcs[(i / s.n) % s.nq].q);
  }
  return ev;
}

}  // namespace ombra::core
