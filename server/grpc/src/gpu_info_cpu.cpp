// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// CPU-only build (OMBRA_ENABLE_CUDA=OFF): there is no GPU to ask.
#include "gpu_info.hpp"

namespace ombra::server {

bool gpu_memory(std::uint64_t*, std::uint64_t*) { return false; }

}  // namespace ombra::server
