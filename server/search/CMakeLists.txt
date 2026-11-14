# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# Search engine library. The pure logic (layout, tombstone penalty vector) is plain C++ and
# tested on the CPU. The engine itself needs OpenFHE and CUDA and is built with
# OMBRA_ENABLE_CUDA=ON. With OMBRA_ENABLE_CUDA=OFF and OpenFHE installed, a CPU-only engine is built
# instead (same API, same key checks, Search fails with a clear error): for tests and development.
add_library(ombra_search_logic STATIC src/layout.cpp)
target_include_directories(ombra_search_logic PUBLIC include)
target_link_libraries(ombra_search_logic PUBLIC ombra_storage)

add_executable(search_layout_test tests/layout_test.cpp)
target_link_libraries(search_layout_test PRIVATE ombra_search_logic)
add_test(NAME search_layout COMMAND search_layout_test)

if(NOT OMBRA_ENABLE_CUDA)
  find_package(OpenFHE 1.6.1 EXACT QUIET)
  if(NOT OpenFHE_FOUND)
    message(STATUS "CPU-only search engine not built: OpenFHE 1.6.1 not found")
    return()
  endif()
  if(NOT "${OpenFHE_NATIVE_SIZE}" STREQUAL "64")
    message(FATAL_ERROR "OpenFHE must be built with NATIVE_SIZE=64 (found '${OpenFHE_NATIVE_SIZE}')")
  endif()
  # Same flags as core/CMakeLists.txt: OpenFHE's backend defines and OpenMP, without -Werror.
  string(REPLACE "-Werror" "" _ofhe_flags "${OpenFHE_CXX_FLAGS}")
  separate_arguments(_ofhe_flags UNIX_COMMAND "${_ofhe_flags}")
  set(_core ${CMAKE_CURRENT_SOURCE_DIR}/../../core)
  if(OMBRA_ENABLE_TEST_HOOKS)
    set_source_files_properties(${_core}/src/security_floor.cpp PROPERTIES COMPILE_DEFINITIONS
                                                                          OMBRA_ENABLE_TEST_HOOKS)
  endif()
  # The security floor of core/ is plain C++ over OpenFHE: compiled here, no CUDA library needed.
  add_library(ombra_search STATIC src/session_cpu.cpp src/key_prep.cpp
                                  ${_core}/src/security_floor.cpp)
  target_include_directories(
    ombra_search PUBLIC include ${_core}/include ${OpenFHE_INCLUDE}
                        ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core
                        ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe PRIVATE src)
  target_compile_options(ombra_search PRIVATE ${_ofhe_flags})
  target_link_directories(ombra_search PUBLIC ${OpenFHE_LIBDIR})
  target_link_libraries(ombra_search PUBLIC ombra_search_logic ombra_cache_logic ombra_storage
                                            ${OpenFHE_SHARED_LIBRARIES})
  return()
endif()

add_library(ombra_search STATIC src/session.cpp src/key_prep.cpp src/device_util.cu)
target_include_directories(ombra_search PUBLIC include PRIVATE src)
set_target_properties(ombra_search PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_STANDARD 17)
target_link_libraries(ombra_search PUBLIC ombra_search_logic ombra_cache ombra_core ombra_storage)

# GPU test driver: exact comparison with the OpenFHE CPU chain (tests/test_gpu_search_engine.py).
add_executable(ombra_search_engine_check src/ombra_search_engine_check.cpp)
# core/src has the CPU reference expansion operations (openfhe_expand_ops.hpp) used as the oracle.
target_include_directories(ombra_search_engine_check
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src)
target_link_libraries(ombra_search_engine_check PRIVATE ombra_search)
