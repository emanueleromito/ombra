# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
cmake_minimum_required(VERSION 3.22)

project(ombra VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(OMBRA_ENABLE_CUDA "Build CUDA kernels" ON)
# Test hooks: OMBRA_TEST_LOG2QP_LIMIT (can only lower the security limit, for the wiring tests). ON
# for development and test builds; a release or container build passes -DOMBRA_ENABLE_TEST_HOOKS=OFF.
option(OMBRA_ENABLE_TEST_HOOKS "Compile test-only hooks (never in a release)" ON)
option(OMBRA_ENABLE_SANITIZERS "Build with AddressSanitizer and UBSan (CPU code)" OFF)

if(OMBRA_ENABLE_CUDA)
  # Decision 0024: Turing (T4), Ampere (A100, A10), Ada (L4, L40S), Hopper (H100, H200).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Fuzzing (fuzz/): libFuzzer targets for the parsers of untrusted bytes. Needs clang.
option(OMBRA_FUZZ "Build the libFuzzer targets in fuzz/ (clang only)" OFF)

# `make test-asan` sets OMBRA_ENABLE_SANITIZERS. A UBSan finding stops the program (no recovery), so
# a test that trips it fails instead of printing a line and passing.
if(OMBRA_ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=undefined
                      -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

if(OMBRA_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "OMBRA_FUZZ=ON needs clang (libFuzzer), but the C++ compiler is "
                        "'${CMAKE_CXX_COMPILER_ID}'. Configure with -DCMAKE_C_COMPILER=clang "
                        "-DCMAKE_CXX_COMPILER=clang++ (make fuzz does this).")
  endif()
  if(OMBRA_ENABLE_SANITIZERS)
    message(FATAL_ERROR "OMBRA_FUZZ=ON already adds the sanitizers; do not set "
                        "OMBRA_ENABLE_SANITIZERS as well")
  endif()
  # The libraries are instrumented for coverage (fuzzer-no-link); each target links the engine.
  add_compile_options(-fsanitize=fuzzer-no-link,address,undefined -fno-sanitize-recover=undefined
                      -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

enable_testing()

# Components are added as they gain a CMakeLists.txt.
foreach(component core server)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${component}/CMakeLists.txt")
    add_subdirectory(${component})
  endif()
endforeach()

if(OMBRA_FUZZ)
  add_subdirectory(fuzz)
endif()
