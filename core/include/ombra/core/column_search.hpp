// Column-packed encrypted search on raw RNS words (no OpenFHE or CUDA dependency).
//
// See core/kernels/column_search.cu for what is computed and the memory layout. All words are
// OpenFHE evaluation-form RNS words; moduli come from the ciphertexts. No relinearization and no
// rescale is done: the result has 3 components per block (T0.4 limitation).
// Public functions throw std::invalid_argument for bad input and CudaError for CUDA failures.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ombra::core {

struct ColumnSearchShape {
    std::size_t dims;        // d ciphertexts per block (and in the query)
    std::size_t blocks;      // stored blocks (each block = dims ciphertexts)
    std::size_t num_towers;  // RNS towers per polynomial
    std::size_t ring_dim;    // N

    std::size_t poly_words() const { return num_towers * ring_dim; }
    std::size_t block_words() const { return dims * 2 * poly_words(); }  // one stored block
    std::size_t stored_words() const { return blocks * block_words(); }
    std::size_t query_words() const { return dims * 2 * poly_words(); }
    std::size_t out_words() const { return blocks * 3 * poly_words(); }  // all blocks
};

// Device-resident search state: stored blocks, query and result stay on the GPU between runs, so
// the kernel can be timed apart from host/device copies.
class ColumnSearchDevice {
  public:
    ColumnSearchDevice(const ColumnSearchShape& shape, const std::vector<std::uint64_t>& moduli);
    ~ColumnSearchDevice();
    ColumnSearchDevice(const ColumnSearchDevice&) = delete;
    ColumnSearchDevice& operator=(const ColumnSearchDevice&) = delete;

    // Copy one stored block (shape.block_words() words) to the device. Words must be reduced.
    void upload_block(std::size_t block, const std::uint64_t* words, std::size_t count);
    // Device-to-device copy of block `src` into block `dst` (for synthetic benchmark data).
    void copy_block(std::size_t src, std::size_t dst);
    // Copy the query (shape.query_words() words) to the device. Words must be reduced.
    void upload_query(const std::uint64_t* words, std::size_t count);
    // Score all blocks. Waits for completion; returns the kernel time in seconds (CUDA events).
    double run();
    // Copy the result (shape.out_words() words) to the host.
    void download_result(std::vector<std::uint64_t>& out);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Name of the current CUDA device as reported by the driver (e.g. for benchmark records).
std::string cuda_device_name();

// One-shot convenience: copies in, runs, copies out.
void column_search(const std::vector<std::uint64_t>& stored, const std::vector<std::uint64_t>& query,
                   std::vector<std::uint64_t>& out, const ColumnSearchShape& shape,
                   const std::vector<std::uint64_t>& moduli);

}  // namespace ombra::core
