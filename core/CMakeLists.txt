# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# core/: GPU kernels and OpenFHE glue. Built only with CUDA (needs OpenFHE and a CUDA toolchain);
# the CPU-only build (`make test`) skips everything here.
# Pure-number unit test of the security floor (header only, no OpenFHE, builds on CPU).
add_executable(ombra_security_floor_test src/security_floor_test.cpp)
target_include_directories(ombra_security_floor_test PRIVATE include)
add_test(NAME security_floor COMMAND ombra_security_floor_test)

# File envelope unit test (header only, no OpenFHE, builds on CPU). Reads the fixture the
# Python writer produced, to check both languages agree byte for byte.
add_executable(ombra_file_envelope_test src/file_envelope_test.cpp)
target_include_directories(ombra_file_envelope_test PRIVATE include)
add_test(NAME file_envelope COMMAND ombra_file_envelope_test
         ${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures/envelope_v1_ciphertext.bin)

# Lazy column-search arithmetic (core/kernels/column_search_lazy.cuh) on the CPU against a 128-bit
# `%` reference. The header and modarith.cuh compile without CUDA.
add_executable(ombra_column_search_lazy_test src/column_search_lazy_test.cpp)
target_include_directories(ombra_column_search_lazy_test PRIVATE include kernels)
add_test(NAME column_search_lazy COMMAND ombra_column_search_lazy_test)

# Expansion tree shape (expand_tree vs expand_tree_into) with a fake ciphertext, header only.
add_executable(ombra_expansion_tree_test src/expansion_tree_test.cpp)
target_include_directories(ombra_expansion_tree_test PRIVATE include)
add_test(NAME expansion_tree COMMAND ombra_expansion_tree_test)

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
add_library(
  ombra_rns_cuda STATIC
  kernels/rns_basic.cu
  kernels/column_search.cu
  kernels/ntt.cu
  kernels/keyswitch.cu
  kernels/rescale.cu
  kernels/encode.cu
  kernels/plain_search.cu
  src/rns_basic_host.cu
  src/column_search_host.cu
  src/plain_search_host.cu
  src/expansion_device.cu)
target_include_directories(ombra_rns_cuda PUBLIC include PRIVATE kernels src)
# CMake 3.22 (the pinned image's apt cmake) does not know the CUDA20 dialect (added in 3.25), and
# this file needs nothing beyond C++17 device code.
set_target_properties(ombra_rns_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_STANDARD 17)

add_library(ombra_core STATIC src/ciphertext_ops.cpp src/expansion_gpu.cpp src/gpu_search.cpp
                      src/security_floor.cpp)
target_include_directories(
  ombra_core PUBLIC include ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include
                    ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
target_link_directories(ombra_core PUBLIC ${OpenFHE_LIBDIR})
target_link_libraries(ombra_core PUBLIC ombra_rns_cuda ${OpenFHE_SHARED_LIBRARIES})

add_executable(ombra_roundtrip src/ombra_roundtrip.cpp)
target_link_libraries(ombra_roundtrip PRIVATE ombra_core)

add_executable(ombra_column_search src/ombra_column_search.cpp)
target_link_libraries(ombra_column_search PRIVATE ombra_core)

add_executable(ombra_column_search_batch src/ombra_column_search_batch.cpp)
target_link_libraries(ombra_column_search_batch PRIVATE ombra_core)

add_executable(ombra_mem_probe src/ombra_mem_probe.cu)
set_target_properties(ombra_mem_probe PROPERTIES CUDA_STANDARD 17)
target_link_libraries(ombra_mem_probe PRIVATE ombra_rns_cuda)

add_executable(ombra_bench_batch src/ombra_bench_batch.cpp)
target_link_libraries(ombra_bench_batch PRIVATE ombra_core)

add_executable(ombra_bench_column_search src/ombra_bench_column_search.cpp)
target_link_libraries(ombra_bench_column_search PRIVATE ombra_core)

add_executable(ombra_expand src/ombra_expand.cpp)
target_link_libraries(ombra_expand PRIVATE ombra_core)

add_executable(ombra_expand_gpu src/ombra_expand_gpu.cpp)
target_link_libraries(ombra_expand_gpu PRIVATE ombra_core)

add_executable(ombra_expand_gpu_batch src/ombra_expand_gpu_batch.cpp)
target_link_libraries(ombra_expand_gpu_batch PRIVATE ombra_core)

add_executable(ombra_bench_expand src/ombra_bench_expand.cpp)
target_link_libraries(ombra_bench_expand PRIVATE ombra_core)

add_executable(ombra_recall_search src/ombra_recall_search.cpp)
target_link_libraries(ombra_recall_search PRIVATE ombra_core)

add_executable(ombra_encoding_check src/ombra_encoding_check.cpp)
target_link_libraries(ombra_encoding_check PRIVATE ombra_core)

add_executable(ombra_plain_search src/ombra_plain_search.cpp)
target_link_libraries(ombra_plain_search PRIVATE ombra_core)

add_executable(ombra_search_gpu src/ombra_search_gpu.cpp)
target_link_libraries(ombra_search_gpu PRIVATE ombra_core)

# Sends mixed keys, metadata and shapes to every GPU entry point, prints the refusals.
add_executable(ombra_refusal_check src/ombra_refusal_check.cpp)
target_link_libraries(ombra_refusal_check PRIVATE ombra_core)

# Calls the library entry points directly (no driver-side check) to show each one refuses.
add_executable(ombra_security_floor_wiring src/ombra_security_floor_wiring.cpp)
target_link_libraries(ombra_security_floor_wiring PRIVATE ombra_core)

add_executable(ombra_bench_e2e src/ombra_bench_e2e.cpp)
target_link_libraries(ombra_bench_e2e PRIVATE ombra_core)
