// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#pragma once

#include <cstdint>

namespace ombra::server {

// cudaMemGetInfo on the current device; false if the call fails.
bool gpu_memory(std::uint64_t* used_bytes, std::uint64_t* total_bytes);

}  // namespace ombra::server
