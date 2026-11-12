// Column-packed encrypted search on raw RNS words (no OpenFHE or CUDA dependency).
//
// See core/kernels/column_search.cu for what is computed and the memory layout. All words are
// OpenFHE evaluation-form RNS words; moduli come from the ciphertexts. No relinearization and no
// rescale is done: the result has 3 components per block (a limitation of this kernel).
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
  // Seconds the cudaMemcpy of the last upload_block took (excludes the range check).
  double last_upload_seconds() const;
  // Device-to-device copy of block `src` into block `dst` (for synthetic benchmark data).
  void copy_block(std::size_t src, std::size_t dst);
  // Copy the query (shape.query_words() words) to the device. Words must be reduced.
  void upload_query(const std::uint64_t* words, std::size_t count);
  // Device-to-device copy of query dimension j (2 * num_towers * ring_dim words, one
  // ciphertext in the [component][tower][coeff] layout) from `src`, a device pointer, into the
  // query buffer. Ordered after earlier work on the default stream. No host round trip and no
  // range check of the words: the source is the output of the expansion kernels.
  void set_query_column_from_device(std::size_t j, const std::uint64_t* src);
  // Score all blocks. Waits for completion; returns the kernel time in seconds (CUDA events).
  double run();
  // Copy the result (shape.out_words() words) to the host.
  void download_result(std::vector<std::uint64_t>& out);
  // Device pointer to the result [block][component 0..2][tower][coeff] (the GPU-only
  // search feeds it to relinearization without a host round trip). Valid for the object's life.
  const std::uint64_t* result_device() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Scores ONE stored block that lives at an arbitrary device address (e.g. a slot of the
// server's GPU cluster cache) with the same kernel as ColumnSearchDevice, on a caller-given stream.
// launch() is asynchronous and does no checks on the data; the caller keeps the pointers alive and
// ordered (events) until the stream has finished. Layouts: stored shape.block_words() words,
// query shape.query_words() words, out 3 * shape.poly_words() words ([comp 0..2][tower][coeff]).
// shape.blocks is ignored (always one block per launch).
class ColumnSearchLauncher {
 public:
  ColumnSearchLauncher(const ColumnSearchShape& shape, const std::vector<std::uint64_t>& moduli);
  ~ColumnSearchLauncher();
  ColumnSearchLauncher(const ColumnSearchLauncher&) = delete;
  ColumnSearchLauncher& operator=(const ColumnSearchLauncher&) = delete;

  // `stream` is a cudaStream_t (nullptr = default stream), passed as void* to keep CUDA out of
  // this header.
  void launch(const std::uint64_t* stored_dev, const std::uint64_t* query_dev,
              std::uint64_t* out_dev, void* stream) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Batched search: Q queries scored in one pass over the same stored blocks. Layouts:
//   queries [q][dim][comp 0..1][tower][coeff]   (each query as in ColumnSearchShape::query_words)
//   result  [q][block][comp 0..2][tower][coeff]
// Memory is the caller's to plan: stored + max_queries * (query + result) words stay on the device.
struct SearchTraffic {
  std::uint64_t stored_load_bytes = 0;  // stored words requested by the kernel's threads
  std::uint64_t query_load_bytes = 0;   // query words requested
  std::uint64_t store_bytes = 0;        // result words written
  std::uint64_t compulsory_bytes = 0;   // every stored, query, result word touched once
};

class BatchedColumnSearchDevice {
 public:
  // shape.blocks stored blocks of shape.dims ciphertexts; room for max_queries queries.
  BatchedColumnSearchDevice(const ColumnSearchShape& shape,
                            const std::vector<std::uint64_t>& moduli, std::size_t max_queries);
  ~BatchedColumnSearchDevice();
  BatchedColumnSearchDevice(const BatchedColumnSearchDevice&) = delete;
  BatchedColumnSearchDevice& operator=(const BatchedColumnSearchDevice&) = delete;

  void upload_block(std::size_t block, const std::uint64_t* words, std::size_t count);
  void copy_block(std::size_t src, std::size_t dst);  // device to device
  void upload_query(std::size_t q, const std::uint64_t* words, std::size_t count);
  void copy_query(std::size_t src, std::size_t dst);  // device to device
  // Score the first `queries` queries against all blocks. Waits; returns kernel seconds.
  double run(std::size_t queries);
  // Exact bytes of one run(queries) from the kernel's access pattern.
  SearchTraffic traffic(std::size_t queries) const;
  // Copy the results of the first `queries` queries to the host.
  void download_result(std::size_t queries, std::vector<std::uint64_t>& out);

  std::size_t max_queries() const;
  // Device bytes held by an object with this shape and query capacity.
  static std::size_t device_bytes(const ColumnSearchShape& shape, std::size_t max_queries);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Free and total memory of the current CUDA device in bytes (cudaMemGetInfo), for memory planning.
struct DeviceMemory {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};
DeviceMemory cuda_memory_info();

// Name of the current CUDA device as reported by the driver (e.g. for benchmark records).
std::string cuda_device_name();

// One-shot convenience: copies in, runs, copies out.
void column_search(const std::vector<std::uint64_t>& stored,
                   const std::vector<std::uint64_t>& query, std::vector<std::uint64_t>& out,
                   const ColumnSearchShape& shape, const std::vector<std::uint64_t>& moduli);

}  // namespace ombra::core
