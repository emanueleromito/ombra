// Host side of the GPU CKKS encoder and the ciphertext x plaintext search.
// Kernels: core/kernels/encode.cu, core/kernels/plain_search.cu, core/kernels/ntt.cu.
// Internal code returns Status; the public functions convert to exceptions at the boundary.

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "device_buffer.cuh"
#include "kernel_launch.hpp"
#include "ombra/core/cuda_check.hpp"
#include "ombra/core/plain_search.hpp"

namespace ombra::core {
namespace {

// Columns encoded per pass: bounds the FFT scratch (kChunkCols * slots * 16 bytes).
constexpr std::size_t kChunkCols = 1024;

CodeFormat to_code_format(PlainFormat f) {
  switch (f) {
    case PlainFormat::kDouble:
      return CodeFormat::kDouble;
    case PlainFormat::kFp16:
      return CodeFormat::kFp16;
    case PlainFormat::kInt8:
      return CodeFormat::kInt8;
  }
  throw std::invalid_argument("plain_search: unknown format");
}

template <typename T>
Status to_device(DeviceBuffer<T>& buf, const std::vector<T>& host) {
  OMBRA_TRY(buf.allocate(host.size()));
  OMBRA_CUDA_CHECK(
      cudaMemcpy(buf.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  return {};
}

// The inverse twiddles of OpenFHE's DiscreteFourierTransform::PrecomputedValues(m = 2N, nh = N/2)
// (dftransform.cpp:46-77), same expressions in the same types: inv[lenh - 1 + j] for lenh a power
// of two up to slots/2.
std::vector<double2> make_inv_twiddles(std::size_t ring_dim, std::size_t slots) {
  const std::uint32_t m = static_cast<std::uint32_t>(2 * ring_dim);
  const std::uint32_t nh = static_cast<std::uint32_t>(ring_dim / 2);
  std::vector<std::uint32_t> rot(nh);
  std::uint32_t five = 1;
  for (std::size_t i = 0; i < nh; ++i) {
    rot[i] = five;
    five *= 5;
    five %= m;
  }
  std::vector<double2> ksi(m + 1);
  for (std::size_t j = 0; j < m; ++j) {
    const double angle = 2.0 * M_PI * j / m;
    ksi[j] = make_double2(std::cos(angle), std::sin(angle));
  }
  ksi[m] = ksi[0];
  std::vector<double2> inv(slots > 0 ? slots - 1 : 0);
  for (std::uint32_t lenh = 1; lenh <= slots / 2; lenh <<= 1) {
    const std::uint32_t lenq = lenh << 3;
    const std::uint32_t gap = m / lenq;
    for (std::uint32_t j = 0; j < lenh; ++j) {
      const std::uint32_t rg = rot[j] % lenq;
      inv[lenh - 1 + j] = ksi[(lenq - rg) * gap];
    }
  }
  return inv;
}

void check_reduced(const std::uint64_t* words, std::size_t count, std::size_t ring_dim,
                   const std::vector<std::uint64_t>& moduli) {
  for (std::size_t chunk = 0; chunk * ring_dim < count; ++chunk) {
    const std::uint64_t q = moduli[chunk % moduli.size()];
    const std::uint64_t* p = words + chunk * ring_dim;
    for (std::size_t i = 0; i < ring_dim; ++i) {
      if (p[i] >= q) throw std::invalid_argument("plain_search: unreduced input word");
    }
  }
}

}  // namespace

std::size_t plain_format_bytes(PlainFormat f) {
  switch (f) {
    case PlainFormat::kDouble:
      return 8;
    case PlainFormat::kFp16:
      return 2;
    case PlainFormat::kInt8:
      return 1;
  }
  throw std::invalid_argument("plain_search: unknown format");
}

EncodeTables encode_tables_from(const ExpansionTables& t) {
  EncodeTables e;
  e.ring_dim = t.ring_dim;
  e.num_towers = t.num_q;
  e.moduli.assign(t.moduli.begin(), t.moduli.begin() + t.num_q);
  e.psi_rev.assign(t.psi_rev.begin(), t.psi_rev.begin() + t.num_q * t.ring_dim);
  return e;
}

// ---------------------------------------------------------------------------------------------
// PlainEncoder

struct PlainEncoder::Impl {
  EncodeTables t;
  double scale = 0;
  std::size_t n = 0, slots = 0, nt = 0;
  DeviceBuffer<TowerConst> tc;
  DeviceBuffer<u64> psi, psi_shoup;
  DeviceBuffer<double2> tw, fft;
  DeviceBuffer<int> err;
  DeviceBuffer<NttRow> rows;

  Status init() {
    const std::vector<TowerConst> consts = make_tower_consts(t.moduli);
    OMBRA_TRY(to_device(tc, consts));
    OMBRA_TRY(to_device(psi, t.psi_rev));
    std::vector<u64> shoup(t.psi_rev.size());
    for (std::size_t x = 0; x < nt; ++x) {
      for (std::size_t i = 0; i < n; ++i) {
        shoup[x * n + i] = shoup_quotient(t.psi_rev[x * n + i], t.moduli[x]);
      }
    }
    OMBRA_TRY(to_device(psi_shoup, shoup));
    OMBRA_TRY(to_device(tw, make_inv_twiddles(n, slots)));
    OMBRA_TRY(fft.allocate(kChunkCols * slots));
    OMBRA_TRY(err.allocate(1));
    std::vector<NttRow> r(nt);
    for (std::size_t x = 0; x < nt; ++x) r[x] = {x * n, 0, 0, static_cast<u32>(x), 0};
    OMBRA_TRY(to_device(rows, r));
    return {};
  }

  NttTables ntt() const {
    // The forward transform reads only psi and tc; the inverse tables are never used here.
    return {psi.get(), psi_shoup.get(), psi.get(), psi_shoup.get(), tc.get(), n};
  }

  Status encode(PlainFormat fmt, const void* codes, std::size_t cols, double code_scale, u64* out,
                bool do_ntt) {
    const int zero = 0;
    OMBRA_CUDA_CHECK(cudaMemcpy(err.get(), &zero, sizeof(int), cudaMemcpyHostToDevice));
    const std::size_t value_bytes = plain_format_bytes(fmt);
    const std::size_t poly = nt * n;
    const double inv_len = 1.0 / static_cast<double>(slots);
    for (std::size_t c0 = 0; c0 < cols; c0 += kChunkCols) {
      const std::size_t nc = std::min(kChunkCols, cols - c0);
      const auto* in = static_cast<const unsigned char*>(codes) + c0 * slots * value_bytes;
      u64* o = out + c0 * poly;
      OMBRA_CUDA_CHECK(
          launch_encode_decode(to_code_format(fmt), in, nc, slots, code_scale, fft.get(), nullptr));
      for (std::size_t len = slots; len >= 2; len >>= 1) {
        OMBRA_CUDA_CHECK(launch_encode_fft_stage(fft.get(), tw.get(), nc, slots, len, nullptr));
      }
      OMBRA_CUDA_CHECK(launch_encode_finish(fft.get(), nc, slots, inv_len, scale, tc.get(), nt, n,
                                            o, err.get(), nullptr));
      if (do_ntt) {
        NttBatch batch;
        batch.count = static_cast<u32>(nc);
        batch.stride = poly;
        OMBRA_CUDA_CHECK(
            launch_ntt_forward(o, rows.get(), static_cast<u32>(nt), ntt(), nullptr, batch));
      }
    }
    OMBRA_CUDA_CHECK_LAUNCH();
    int flag = 0;  // blocking copy: waits for all of the above
    OMBRA_CUDA_CHECK(cudaMemcpy(&flag, err.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (flag != 0) {
      return Status(cudaErrorInvalidValue, "scaled value overflows 62 bits", __FILE__, __LINE__);
    }
    return {};
  }
};

PlainEncoder::PlainEncoder(const EncodeTables& tables, double scaling_factor)
    : impl_(std::make_unique<Impl>()) {
  const std::size_t n = tables.ring_dim;
  if (n < 4 || (n & (n - 1)) != 0 || tables.num_towers == 0 ||
      tables.moduli.size() != tables.num_towers || tables.psi_rev.size() != tables.num_towers * n) {
    throw std::invalid_argument("plain encoder: bad tables");
  }
  if (!(scaling_factor >= 1.0)) throw std::invalid_argument("plain encoder: bad scaling factor");
  impl_->t = tables;
  impl_->scale = scaling_factor;
  impl_->n = n;
  impl_->slots = n / 2;
  impl_->nt = tables.num_towers;
  impl_->init().throw_if_error();
}

PlainEncoder::~PlainEncoder() = default;

std::size_t PlainEncoder::slots() const { return impl_->slots; }
std::size_t PlainEncoder::poly_words() const { return impl_->nt * impl_->n; }

void PlainEncoder::encode_device(PlainFormat fmt, const void* d_codes, std::size_t cols,
                                 double code_scale, std::uint64_t* d_out) {
  if (cols == 0) return;
  if (fmt == PlainFormat::kInt8 && !(code_scale > 0)) {
    throw std::invalid_argument("plain encoder: int8 needs a positive code scale");
  }
  impl_->encode(fmt, d_codes, cols, code_scale, d_out, true).throw_if_error();
}

std::vector<std::uint64_t> PlainEncoder::encode_host(PlainFormat fmt, const void* codes,
                                                     std::size_t cols, double code_scale,
                                                     bool coefficients_only) {
  const std::size_t value_bytes = plain_format_bytes(fmt);
  DeviceBuffer<unsigned char> d_codes;
  DeviceBuffer<u64> d_out;
  std::vector<std::uint64_t> out(cols * poly_words());
  const auto run = [&]() -> Status {
    OMBRA_TRY(d_codes.allocate(cols * impl_->slots * value_bytes));
    OMBRA_TRY(d_out.allocate(out.size()));
    OMBRA_CUDA_CHECK(cudaMemcpy(d_codes.get(), codes, cols * impl_->slots * value_bytes,
                                cudaMemcpyHostToDevice));
    OMBRA_TRY(impl_->encode(fmt, d_codes.get(), cols, code_scale, d_out.get(), !coefficients_only));
    OMBRA_CUDA_CHECK(
        cudaMemcpy(out.data(), d_out.get(), out.size() * sizeof(u64), cudaMemcpyDeviceToHost));
    return {};
  };
  run().throw_if_error();
  return out;
}

std::vector<std::uint64_t> PlainEncoder::reduce_ntt_host(const std::int64_t* coeffs,
                                                         std::size_t cols, bool ntt) {
  const std::size_t n = impl_->n, poly = impl_->nt * n;
  DeviceBuffer<long long> d_in;
  DeviceBuffer<u64> d_out;
  std::vector<std::uint64_t> out(cols * poly);
  const auto run = [&]() -> Status {
    OMBRA_TRY(d_in.allocate(cols * n));
    OMBRA_TRY(d_out.allocate(out.size()));
    OMBRA_CUDA_CHECK(cudaMemcpy(d_in.get(), coeffs, cols * n * sizeof(long long),
                                cudaMemcpyHostToDevice));
    OMBRA_CUDA_CHECK(
        launch_reduce_ints(d_in.get(), cols, n, impl_->tc.get(), impl_->nt, d_out.get(), nullptr));
    if (ntt) {
      NttBatch batch;
      batch.count = static_cast<u32>(cols);
      batch.stride = poly;
      OMBRA_CUDA_CHECK(launch_ntt_forward(d_out.get(), impl_->rows.get(),
                                          static_cast<u32>(impl_->nt), impl_->ntt(), nullptr,
                                          batch));
    }
    OMBRA_CUDA_CHECK_LAUNCH();
    OMBRA_CUDA_CHECK(
        cudaMemcpy(out.data(), d_out.get(), out.size() * sizeof(u64), cudaMemcpyDeviceToHost));
    return {};
  };
  run().throw_if_error();
  return out;
}

// ---------------------------------------------------------------------------------------------
// PlainSearchDevice

struct PlainSearchDevice::Impl {
  PlainSearchShape shape;
  std::vector<u64> moduli;
  PlainFormat fmt = PlainFormat::kDouble;
  double code_scale = 1;
  std::size_t chunk = 1;
  std::unique_ptr<PlainEncoder> enc;
  DeviceBuffer<TowerConst> tc;
  DeviceBuffer<unsigned char> codes;  // lazily allocated
  DeviceBuffer<u64> encoded;          // lazily allocated
  DeviceBuffer<u64> scratch, query, out;
  cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;

  ~Impl() {
    // Destructors cannot return a Status; failures are ignored on purpose (nothing to recover).
    for (cudaEvent_t e : {e0, e1, e2}) {
      if (e != nullptr) (void)cudaEventDestroy(e);
    }
  }

  std::size_t block_code_bytes() const {
    return shape.dims * shape.slots() * plain_format_bytes(fmt);
  }

  Status init() {
    for (u64 q : moduli) {
      // the search kernel multiplies with Montgomery reduction (needs odd q; RNS primes are odd)
      if ((q & 1) == 0) throw std::invalid_argument("plain_search: moduli must be odd");
    }
    OMBRA_TRY(to_device(tc, make_tower_consts(moduli)));
    OMBRA_TRY(query.allocate(shape.query_words()));
    OMBRA_TRY(out.allocate(shape.out_words()));
    OMBRA_CUDA_CHECK(cudaEventCreate(&e0));
    OMBRA_CUDA_CHECK(cudaEventCreate(&e1));
    OMBRA_CUDA_CHECK(cudaEventCreate(&e2));
    return {};
  }

  Status ensure_codes() {
    if (codes.get() == nullptr) OMBRA_TRY(codes.allocate(shape.blocks * block_code_bytes()));
    return {};
  }
  Status ensure_encoded() {
    if (encoded.get() == nullptr) {
      OMBRA_TRY(encoded.allocate(shape.blocks * shape.encoded_block_words()));
    }
    return {};
  }

  Status search(const u64* plain, std::size_t blocks, u64* dst) {
    OMBRA_CUDA_CHECK(launch_plain_search(plain, query.get(), dst, tc.get(), shape.dims, blocks,
                                         shape.num_towers, shape.ring_dim, nullptr));
    return {};
  }

  Status time_between(cudaEvent_t a, cudaEvent_t b, double& seconds) {
    float ms = 0;
    OMBRA_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    seconds = ms / 1000.0;
    return {};
  }

  Status run_encoded(double& seconds) {
    OMBRA_TRY(ensure_encoded());
    OMBRA_CUDA_CHECK(cudaEventRecord(e0, nullptr));
    OMBRA_TRY(search(encoded.get(), shape.blocks, out.get()));
    OMBRA_CUDA_CHECK(cudaEventRecord(e1, nullptr));
    OMBRA_CUDA_CHECK(cudaEventSynchronize(e1));
    OMBRA_CUDA_CHECK_LAUNCH();
    return time_between(e0, e1, seconds);
  }

  Status encode_all(double& seconds) {
    OMBRA_TRY(ensure_codes());
    OMBRA_TRY(ensure_encoded());
    OMBRA_CUDA_CHECK(cudaEventRecord(e0, nullptr));
    const auto go = [&]() -> void {
      enc->encode_device(fmt, codes.get(), shape.blocks * shape.dims, code_scale, encoded.get());
    };
    go();  // throws on failure; encode_device waits for completion
    OMBRA_CUDA_CHECK(cudaEventRecord(e1, nullptr));
    OMBRA_CUDA_CHECK(cudaEventSynchronize(e1));
    return time_between(e0, e1, seconds);
  }

  Status run_compact(double& enc_s, double& search_s) {
    OMBRA_TRY(ensure_codes());
    enc_s = search_s = 0;
    for (std::size_t b0 = 0; b0 < shape.blocks; b0 += chunk) {
      const std::size_t nb = std::min(chunk, shape.blocks - b0);
      OMBRA_CUDA_CHECK(cudaEventRecord(e0, nullptr));
      enc->encode_device(fmt, codes.get() + b0 * block_code_bytes(), nb * shape.dims, code_scale,
                         scratch.get());
      OMBRA_CUDA_CHECK(cudaEventRecord(e1, nullptr));
      OMBRA_TRY(search(scratch.get(), nb, out.get() + b0 * 2 * shape.poly_words()));
      OMBRA_CUDA_CHECK(cudaEventRecord(e2, nullptr));
      OMBRA_CUDA_CHECK(cudaEventSynchronize(e2));
      OMBRA_CUDA_CHECK_LAUNCH();
      double a = 0, s = 0;
      OMBRA_TRY(time_between(e0, e1, a));
      OMBRA_TRY(time_between(e1, e2, s));
      enc_s += a;
      search_s += s;
    }
    return {};
  }
};

PlainSearchDevice::PlainSearchDevice(const PlainSearchShape& shape, const EncodeTables& tables,
                                     double scaling_factor, PlainFormat fmt, double code_scale,
                                     std::size_t chunk_blocks)
    : impl_(std::make_unique<Impl>()) {
  if (shape.dims == 0 || shape.blocks == 0 || shape.num_towers != tables.num_towers ||
      shape.ring_dim != tables.ring_dim || chunk_blocks == 0 || (shape.blocks + 3) / 4 > 65535) {
    throw std::invalid_argument("plain search: bad shape");
  }
  impl_->shape = shape;
  impl_->moduli = tables.moduli;
  impl_->fmt = fmt;
  impl_->code_scale = code_scale;
  impl_->chunk = std::min(chunk_blocks, shape.blocks);
  impl_->enc = std::make_unique<PlainEncoder>(tables, scaling_factor);
  impl_->init().throw_if_error();
  impl_->scratch.allocate(impl_->chunk * shape.encoded_block_words()).throw_if_error();
}

PlainSearchDevice::~PlainSearchDevice() = default;

void PlainSearchDevice::upload_codes(std::size_t block, const void* codes) {
  if (block >= impl_->shape.blocks) throw std::invalid_argument("plain search: bad block");
  impl_->ensure_codes().throw_if_error();
  const auto go = [&]() -> Status {
    OMBRA_CUDA_CHECK(cudaMemcpy(impl_->codes.get() + block * impl_->block_code_bytes(), codes,
                                impl_->block_code_bytes(), cudaMemcpyHostToDevice));
    return {};
  };
  go().throw_if_error();
}

void PlainSearchDevice::copy_codes(std::size_t src, std::size_t dst) {
  if (src >= impl_->shape.blocks || dst >= impl_->shape.blocks) {
    throw std::invalid_argument("plain search: bad block");
  }
  impl_->ensure_codes().throw_if_error();
  const auto go = [&]() -> Status {
    const std::size_t bytes = impl_->block_code_bytes();
    OMBRA_CUDA_CHECK(cudaMemcpy(impl_->codes.get() + dst * bytes, impl_->codes.get() + src * bytes,
                                bytes, cudaMemcpyDeviceToDevice));
    return {};
  };
  go().throw_if_error();
}

void PlainSearchDevice::upload_encoded(std::size_t block, const std::uint64_t* words,
                                       std::size_t count) {
  const PlainSearchShape& s = impl_->shape;
  if (block >= s.blocks || count != s.encoded_block_words()) {
    throw std::invalid_argument("plain search: bad block index or size");
  }
  check_reduced(words, count, s.ring_dim, impl_->moduli);
  impl_->ensure_encoded().throw_if_error();
  const auto go = [&]() -> Status {
    OMBRA_CUDA_CHECK(cudaMemcpy(impl_->encoded.get() + block * count, words, count * sizeof(u64),
                                cudaMemcpyHostToDevice));
    return {};
  };
  go().throw_if_error();
}

void PlainSearchDevice::upload_query(const std::uint64_t* words, std::size_t count) {
  if (count != impl_->shape.query_words()) throw std::invalid_argument("plain search: bad query");
  check_reduced(words, count, impl_->shape.ring_dim, impl_->moduli);
  const auto go = [&]() -> Status {
    OMBRA_CUDA_CHECK(
        cudaMemcpy(impl_->query.get(), words, count * sizeof(u64), cudaMemcpyHostToDevice));
    return {};
  };
  go().throw_if_error();
}

double PlainSearchDevice::encode_all() {
  double s = 0;
  impl_->encode_all(s).throw_if_error();
  return s;
}

double PlainSearchDevice::run_encoded() {
  double s = 0;
  impl_->run_encoded(s).throw_if_error();
  return s;
}

double PlainSearchDevice::run_compact(double* encode_seconds, double* search_seconds) {
  double e = 0, s = 0;
  impl_->run_compact(e, s).throw_if_error();
  if (encode_seconds != nullptr) *encode_seconds = e;
  if (search_seconds != nullptr) *search_seconds = s;
  return e + s;
}

void PlainSearchDevice::download_result(std::vector<std::uint64_t>& out) {
  out.assign(impl_->shape.out_words(), 0);
  const auto go = [&]() -> Status {
    OMBRA_CUDA_CHECK(
        cudaMemcpy(out.data(), impl_->out.get(), out.size() * sizeof(u64), cudaMemcpyDeviceToHost));
    return {};
  };
  go().throw_if_error();
}

}  // namespace ombra::core
