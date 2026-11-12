// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Two CUDA calls the search session needs and the cache/core headers do not expose. No CUDA types.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ombra::search {

// Device-to-device copy of `words` 64-bit words on the default stream (ordered after earlier
// default-stream work such as the expansion). Throws ombra::core::CudaError.
void device_copy_words(std::uint64_t* dst, const std::uint64_t* src, std::size_t words);
// Waits for all work on the device (the cache's streams are non-blocking, so they do not order
// themselves against the default stream). Throws ombra::core::CudaError.
void device_sync();

}  // namespace ombra::search
