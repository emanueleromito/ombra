# core/: GPU kernels and OpenFHE glue. Built only with CUDA (needs OpenFHE and a CUDA toolchain);
# the CPU-only build (`make test`) skips everything here.
if(NOT OMBRA_ENABLE_CUDA)
  return()
endif()

find_package(OpenFHE REQUIRED)

# The kernels copy 64-bit words, so OpenFHE's native integer must be 64 bits.
if(NOT "${OpenFHE_NATIVE_SIZE}" STREQUAL "64")
  message(FATAL_ERROR "OpenFHE must be built with NATIVE_SIZE=64 (found '${OpenFHE_NATIVE_SIZE}')")
endif()

# OpenFHE's installed flags (backend defines, OpenMP) must match its own build. Drop -Werror so
# warnings from OpenFHE headers do not break our C++20 build.
string(REPLACE "-Werror" "" OMBRA_OPENFHE_CXX_FLAGS "${OpenFHE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMBRA_OPENFHE_CXX_FLAGS}")

# Pure CUDA library: no OpenFHE headers are seen by nvcc.
add_library(ombra_rns_cuda STATIC kernels/rns_basic.cu kernels/column_search.cu src/rns_basic_host.cu
            src/column_search_host.cu)
target_include_directories(ombra_rns_cuda PUBLIC include PRIVATE kernels src)
# CMake 3.22 (the pinned image's apt cmake) does not know the CUDA20 dialect (added in 3.25), and
# this file needs nothing beyond C++17 device code.
set_target_properties(ombra_rns_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_STANDARD 17)

add_library(ombra_core STATIC src/ciphertext_ops.cpp)
target_include_directories(
  ombra_core PUBLIC include ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include
                    ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
target_link_directories(ombra_core PUBLIC ${OpenFHE_LIBDIR})
target_link_libraries(ombra_core PUBLIC ombra_rns_cuda ${OpenFHE_SHARED_LIBRARIES})

add_executable(ombra_roundtrip src/ombra_roundtrip.cpp)
target_link_libraries(ombra_roundtrip PRIVATE ombra_core)

add_executable(ombra_column_search src/ombra_column_search.cpp)
target_link_libraries(ombra_column_search PRIVATE ombra_core)

add_executable(ombra_bench_column_search src/ombra_bench_column_search.cpp)
target_link_libraries(ombra_bench_column_search PRIVATE ombra_core)
