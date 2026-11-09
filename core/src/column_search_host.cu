// Host wrapper for the column-packed search kernel (kernel: core/kernels/column_search.cu).
// Internal code returns Status; the public functions convert to exceptions at the boundary.

#include "ombra/core/column_search.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>

#include "device_buffer.cuh"
#include "kernel_launch.hpp"
#include "ombra/core/cuda_check.hpp"

namespace ombra::core {
namespace {

// __launch_bounds__ minimum blocks per SM of the 32-bit search kernel (4, 6 or 8). Default from
// the T1.5 tuning; OMBRA_SEARCH32_MIN_BLOCKS overrides it for benchmarking only.
int min_blocks32() {
    const char* v = std::getenv("OMBRA_SEARCH32_MIN_BLOCKS");
    return v != nullptr ? std::atoi(v) : 4;
}

void validate_shape(const ColumnSearchShape& s, const std::vector<u64>& moduli) {
    if (s.dims == 0 || s.blocks == 0 || s.num_towers == 0 || s.ring_dim == 0) {
        throw std::invalid_argument("column_search: empty shape");
    }
    if (moduli.size() != s.num_towers) {
        throw std::invalid_argument("column_search: one modulus per tower is required");
    }
    for (std::uint64_t q : moduli) {
        // the kernels multiply with Montgomery reduction, which needs an odd modulus
        if ((q & 1) == 0) throw std::invalid_argument("column_search: moduli must be odd");
    }
    if ((s.blocks + 3) / 4 > 65535) {
        throw std::invalid_argument("column_search: too many blocks for one launch");
    }
}

// Words are laid out [poly][tower][coeff]: chunk number i / ring_dim has tower (chunk % towers).
void check_reduced(const u64* words, std::size_t count, const ColumnSearchShape& s,
                   const std::vector<u64>& moduli) {
    for (std::size_t chunk = 0; chunk * s.ring_dim < count; ++chunk) {
        const u64 q = moduli[chunk % s.num_towers];
        const u64* p = words + chunk * s.ring_dim;
        for (std::size_t i = 0; i < s.ring_dim; ++i) {
            if (p[i] >= q) throw std::invalid_argument("column_search: unreduced input word");
        }
    }
}

}  // namespace

struct ColumnSearchDevice::Impl {
    ColumnSearchShape shape;
    std::vector<u64> moduli;
    // T1.5: with `narrow` set (every modulus below 2^31) the stored and query words are held as
    // 32-bit words (stored32, query32) and searched by column_search32.cu; the 64-bit buffers are
    // then unused. The result is 64-bit words in both modes.
    bool narrow = false;
    DeviceBuffer<u64> stored, query, out;
    DeviceBuffer<u32> stored32, query32;
    DeviceBuffer<TowerConst> tc;
    DeviceBuffer<TowerConst32> tc32;
    cudaEvent_t start = nullptr, stop = nullptr;
    double last_upload_s = 0;

    ~Impl() {
        // Destructors cannot return a Status; failures are ignored on purpose (nothing to recover).
        if (start != nullptr) (void)cudaEventDestroy(start);
        if (stop != nullptr) (void)cudaEventDestroy(stop);
    }

    Status init(const std::vector<TowerConst>& consts) {
        OMBRA_TRY(out.allocate(shape.out_words()));
        if (narrow) {
            OMBRA_TRY(stored32.allocate(shape.stored_words()));
            OMBRA_TRY(query32.allocate(shape.query_words()));
            const std::vector<TowerConst32> c32 = make_tower_consts32(moduli);
            OMBRA_TRY(tc32.allocate(c32.size()));
            OMBRA_CUDA_CHECK(cudaMemcpy(tc32.get(), c32.data(),
                                        c32.size() * sizeof(TowerConst32),
                                        cudaMemcpyHostToDevice));
        } else {
            OMBRA_TRY(stored.allocate(shape.stored_words()));
            OMBRA_TRY(query.allocate(shape.query_words()));
            OMBRA_TRY(tc.allocate(consts.size()));
            OMBRA_CUDA_CHECK(cudaMemcpy(tc.get(), consts.data(),
                                        consts.size() * sizeof(TowerConst),
                                        cudaMemcpyHostToDevice));
        }
        OMBRA_CUDA_CHECK(cudaEventCreate(&start));
        OMBRA_CUDA_CHECK(cudaEventCreate(&stop));
        return {};
    }

    Status copy_in(u64* dst, const u64* src, std::size_t count) {
        OMBRA_CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(u64), cudaMemcpyHostToDevice));
        return {};
    }

    // Narrow mode: the words (checked reduced, hence below 2^31) are cut to 32 bits on the host.
    Status copy_in32(u32* dst, const u64* src, std::size_t count) {
        std::vector<u32> tmp(count);
        for (std::size_t i = 0; i < count; ++i) tmp[i] = static_cast<u32>(src[i]);
        OMBRA_CUDA_CHECK(cudaMemcpy(dst, tmp.data(), count * sizeof(u32), cudaMemcpyHostToDevice));
        return {};
    }

    Status copy_block_dev(std::size_t src, std::size_t dst) {
        const std::size_t bw = shape.block_words();
        if (narrow) {
            OMBRA_CUDA_CHECK(cudaMemcpy(stored32.get() + dst * bw, stored32.get() + src * bw,
                                        bw * sizeof(u32), cudaMemcpyDeviceToDevice));
            return {};
        }
        OMBRA_CUDA_CHECK(cudaMemcpy(stored.get() + dst * bw, stored.get() + src * bw,
                                    bw * sizeof(u64), cudaMemcpyDeviceToDevice));
        return {};
    }

    Status run_kernel(double& seconds) {
        const ColumnSearchDims d{shape.dims, shape.blocks, shape.num_towers, shape.ring_dim};
        OMBRA_CUDA_CHECK(cudaEventRecord(start, nullptr));
        if (narrow) {
            OMBRA_CUDA_CHECK(launch_column_search32(stored32.get(), query32.get(), out.get(),
                                                    tc32.get(), d, min_blocks32(), nullptr));
        } else {
            OMBRA_CUDA_CHECK(
                launch_column_search(stored.get(), query.get(), out.get(), tc.get(), d, nullptr));
        }
        OMBRA_CUDA_CHECK(cudaEventRecord(stop, nullptr));
        OMBRA_CUDA_CHECK(cudaEventSynchronize(stop));  // wait for completion before timing is read
        OMBRA_CUDA_CHECK_LAUNCH();
        float ms = 0;
        OMBRA_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        seconds = ms / 1000.0;
        return {};
    }

    Status copy_out(std::vector<u64>& host) {
        host.assign(shape.out_words(), 0);
        OMBRA_CUDA_CHECK(cudaMemcpy(host.data(), out.get(), host.size() * sizeof(u64),
                                    cudaMemcpyDeviceToHost));
        return {};
    }
};

ColumnSearchDevice::ColumnSearchDevice(const ColumnSearchShape& shape,
                                       const std::vector<std::uint64_t>& moduli)
    : impl_(std::make_unique<Impl>()) {
    validate_shape(shape, moduli);
    const std::vector<TowerConst> consts = make_tower_consts(moduli);
    impl_->shape = shape;
    impl_->moduli = moduli;
    // Every prime below 2^31 (OpenFHE composite scaling with a 32-bit register): 32-bit words.
    impl_->narrow = std::all_of(moduli.begin(), moduli.end(),
                                [](std::uint64_t q) { return q < (std::uint64_t(1) << 31); });
    impl_->init(consts).throw_if_error();
}

ColumnSearchDevice::~ColumnSearchDevice() = default;

bool ColumnSearchDevice::uses_32bit_words() const { return impl_->narrow; }

void ColumnSearchDevice::upload_block(std::size_t block, const std::uint64_t* words,
                                      std::size_t count) {
    const ColumnSearchShape& s = impl_->shape;
    if (block >= s.blocks || count != s.block_words()) {
        throw std::invalid_argument("column_search: bad block index or size");
    }
    check_reduced(words, count, s, impl_->moduli);
    const auto t0 = std::chrono::steady_clock::now();
    if (impl_->narrow) {
        impl_->copy_in32(impl_->stored32.get() + block * s.block_words(), words, count)
            .throw_if_error();
    } else {
        impl_->copy_in(impl_->stored.get() + block * s.block_words(), words, count)
            .throw_if_error();
    }
    impl_->last_upload_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

double ColumnSearchDevice::last_upload_seconds() const { return impl_->last_upload_s; }

void ColumnSearchDevice::copy_block(std::size_t src, std::size_t dst) {
    if (src >= impl_->shape.blocks || dst >= impl_->shape.blocks) {
        throw std::invalid_argument("column_search: bad block index");
    }
    impl_->copy_block_dev(src, dst).throw_if_error();
}

void ColumnSearchDevice::upload_query(const std::uint64_t* words, std::size_t count) {
    const ColumnSearchShape& s = impl_->shape;
    if (count != s.query_words()) throw std::invalid_argument("column_search: bad query size");
    check_reduced(words, count, s, impl_->moduli);
    if (impl_->narrow) {
        impl_->copy_in32(impl_->query32.get(), words, count).throw_if_error();
    } else {
        impl_->copy_in(impl_->query.get(), words, count).throw_if_error();
    }
}

void ColumnSearchDevice::set_query_column_from_device(std::size_t j, const std::uint64_t* src) {
    const ColumnSearchShape& s = impl_->shape;
    if (j >= s.dims || src == nullptr) throw std::invalid_argument("column_search: bad column");
    const std::size_t words = 2 * s.poly_words();
    const auto go = [&]() -> Status {
        if (impl_->narrow) {  // the source words are residues below 2^31: cut them to 32 bits
            OMBRA_CUDA_CHECK(launch_narrow_u64_to_u32(src, impl_->query32.get() + j * words, words,
                                                      nullptr));
            return {};
        }
        OMBRA_CUDA_CHECK(cudaMemcpy(impl_->query.get() + j * words, src, words * sizeof(u64),
                                    cudaMemcpyDeviceToDevice));
        return {};
    };
    go().throw_if_error();
}

double ColumnSearchDevice::run() {
    double seconds = 0;
    impl_->run_kernel(seconds).throw_if_error();
    return seconds;
}

void ColumnSearchDevice::download_result(std::vector<std::uint64_t>& out) {
    impl_->copy_out(out).throw_if_error();
}

const std::uint64_t* ColumnSearchDevice::result_device() const { return impl_->out.get(); }

struct BatchedColumnSearchDevice::Impl {
    ColumnSearchShape shape;
    std::vector<u64> moduli;
    std::size_t max_queries = 0;
    DeviceBuffer<u64> stored, queries, out;
    DeviceBuffer<TowerConst> tc;
    cudaEvent_t start = nullptr, stop = nullptr;

    ~Impl() {
        if (start != nullptr) (void)cudaEventDestroy(start);
        if (stop != nullptr) (void)cudaEventDestroy(stop);
    }

    ColumnSearchDims dims() const {
        return {shape.dims, shape.blocks, shape.num_towers, shape.ring_dim};
    }

    Status init(const std::vector<TowerConst>& consts) {
        OMBRA_TRY(stored.allocate(shape.stored_words()));
        OMBRA_TRY(queries.allocate(max_queries * shape.query_words()));
        OMBRA_TRY(out.allocate(max_queries * shape.out_words()));
        OMBRA_TRY(tc.allocate(consts.size()));
        OMBRA_CUDA_CHECK(cudaMemcpy(tc.get(), consts.data(), consts.size() * sizeof(TowerConst),
                                    cudaMemcpyHostToDevice));
        OMBRA_CUDA_CHECK(cudaEventCreate(&start));
        OMBRA_CUDA_CHECK(cudaEventCreate(&stop));
        return {};
    }

    Status copy_in(u64* dst, const u64* src, std::size_t count) {
        OMBRA_CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(u64), cudaMemcpyHostToDevice));
        return {};
    }

    Status copy_dev(u64* dst, const u64* src, std::size_t count) {
        OMBRA_CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(u64), cudaMemcpyDeviceToDevice));
        return {};
    }

    // Default stream, like every other launch in core: the order of the calls is the order of
    // copies and kernels.
    Status run_kernel(std::size_t nq, double& seconds) {
        OMBRA_CUDA_CHECK(cudaEventRecord(start, nullptr));
        OMBRA_CUDA_CHECK(launch_column_search_batched(stored.get(), queries.get(), out.get(),
                                                      tc.get(), dims(), nq, nullptr));
        OMBRA_CUDA_CHECK(cudaEventRecord(stop, nullptr));
        OMBRA_CUDA_CHECK(cudaEventSynchronize(stop));
        OMBRA_CUDA_CHECK_LAUNCH();
        float ms = 0;
        OMBRA_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        seconds = ms / 1000.0;
        return {};
    }

    Status copy_out(std::size_t nq, std::vector<u64>& host) {
        host.assign(nq * shape.out_words(), 0);
        OMBRA_CUDA_CHECK(cudaMemcpy(host.data(), out.get(), host.size() * sizeof(u64),
                                    cudaMemcpyDeviceToHost));
        return {};
    }
};

BatchedColumnSearchDevice::BatchedColumnSearchDevice(const ColumnSearchShape& shape,
                                                     const std::vector<std::uint64_t>& moduli,
                                                     std::size_t max_queries)
    : impl_(std::make_unique<Impl>()) {
    validate_shape(shape, moduli);
    if (max_queries == 0 || (max_queries + 3) / 4 > 65535) {
        throw std::invalid_argument("column_search: bad query capacity");
    }
    impl_->shape = shape;
    impl_->moduli = moduli;
    impl_->max_queries = max_queries;
    impl_->init(make_tower_consts(moduli)).throw_if_error();
}

BatchedColumnSearchDevice::~BatchedColumnSearchDevice() = default;

void BatchedColumnSearchDevice::upload_block(std::size_t block, const std::uint64_t* words,
                                             std::size_t count) {
    const ColumnSearchShape& s = impl_->shape;
    if (block >= s.blocks || count != s.block_words()) {
        throw std::invalid_argument("column_search: bad block index or size");
    }
    check_reduced(words, count, s, impl_->moduli);
    impl_->copy_in(impl_->stored.get() + block * s.block_words(), words, count).throw_if_error();
}

void BatchedColumnSearchDevice::copy_block(std::size_t src, std::size_t dst) {
    const ColumnSearchShape& s = impl_->shape;
    if (src >= s.blocks || dst >= s.blocks) {
        throw std::invalid_argument("column_search: bad block index");
    }
    const std::size_t bw = s.block_words();
    impl_->copy_dev(impl_->stored.get() + dst * bw, impl_->stored.get() + src * bw, bw)
        .throw_if_error();
}

void BatchedColumnSearchDevice::upload_query(std::size_t q, const std::uint64_t* words,
                                             std::size_t count) {
    const ColumnSearchShape& s = impl_->shape;
    if (q >= impl_->max_queries || count != s.query_words()) {
        throw std::invalid_argument("column_search: bad query index or size");
    }
    check_reduced(words, count, s, impl_->moduli);
    impl_->copy_in(impl_->queries.get() + q * s.query_words(), words, count).throw_if_error();
}

void BatchedColumnSearchDevice::copy_query(std::size_t src, std::size_t dst) {
    const ColumnSearchShape& s = impl_->shape;
    if (src >= impl_->max_queries || dst >= impl_->max_queries) {
        throw std::invalid_argument("column_search: bad query index");
    }
    const std::size_t qw = s.query_words();
    impl_->copy_dev(impl_->queries.get() + dst * qw, impl_->queries.get() + src * qw, qw)
        .throw_if_error();
}

double BatchedColumnSearchDevice::run(std::size_t queries) {
    if (queries == 0 || queries > impl_->max_queries) {
        throw std::invalid_argument("column_search: bad query count");
    }
    double seconds = 0;
    impl_->run_kernel(queries, seconds).throw_if_error();
    return seconds;
}

SearchTraffic BatchedColumnSearchDevice::traffic(std::size_t queries) const {
    const BatchSearchTraffic t = column_search_batched_traffic(impl_->dims(), queries);
    return {t.stored_load_bytes, t.query_load_bytes, t.store_bytes, t.compulsory_bytes};
}

void BatchedColumnSearchDevice::download_result(std::size_t queries,
                                                std::vector<std::uint64_t>& out) {
    if (queries == 0 || queries > impl_->max_queries) {
        throw std::invalid_argument("column_search: bad query count");
    }
    impl_->copy_out(queries, out).throw_if_error();
}

std::size_t BatchedColumnSearchDevice::max_queries() const { return impl_->max_queries; }

std::size_t BatchedColumnSearchDevice::device_bytes(const ColumnSearchShape& shape,
                                                    std::size_t max_queries) {
    return (shape.stored_words() + max_queries * (shape.query_words() + shape.out_words())) *
               sizeof(u64) +
           shape.num_towers * sizeof(TowerConst);
}

DeviceMemory cuda_memory_info() {
    DeviceMemory m;
    const auto query = [&]() -> Status {
        OMBRA_CUDA_CHECK(cudaFree(nullptr));  // make sure the context exists
        OMBRA_CUDA_CHECK(cudaMemGetInfo(&m.free_bytes, &m.total_bytes));
        return {};
    };
    query().throw_if_error();
    return m;
}

std::string cuda_device_name() {
    int dev = 0;
    cudaDeviceProp prop{};
    const auto query = [&]() -> Status {
        OMBRA_CUDA_CHECK(cudaGetDevice(&dev));
        OMBRA_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        return {};
    };
    query().throw_if_error();
    return prop.name;
}

void column_search(const std::vector<std::uint64_t>& stored, const std::vector<std::uint64_t>& query,
                   std::vector<std::uint64_t>& out, const ColumnSearchShape& shape,
                   const std::vector<std::uint64_t>& moduli) {
    if (stored.size() != shape.stored_words() || query.size() != shape.query_words()) {
        throw std::invalid_argument("column_search: operand size does not match shape");
    }
    ColumnSearchDevice dev(shape, moduli);
    for (std::size_t b = 0; b < shape.blocks; ++b) {
        dev.upload_block(b, stored.data() + b * shape.block_words(), shape.block_words());
    }
    dev.upload_query(query.data(), query.size());
    dev.run();
    dev.download_result(out);
}

}  // namespace ombra::core
