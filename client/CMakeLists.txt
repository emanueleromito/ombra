# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# The compiled part of the client: ombra._native, the OpenFHE CPU code of the
# local-mode server. Built when OpenFHE 1.6.1 (C++) is found; without it the pure-Python client
# still installs, and local mode says what is missing. OMBRA_REQUIRE_NATIVE=1 makes it an error.
cmake_minimum_required(VERSION 3.22)
project(ombra_client LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Headers shared with the server's core/ (the tree and the CPU operations), in the repository.
set(OMBRA_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../core" CACHE PATH "core/ of the repository")

find_package(OpenFHE 1.6.1 EXACT QUIET)
find_package(pybind11 CONFIG QUIET)
if(NOT OpenFHE_FOUND OR NOT pybind11_FOUND OR NOT EXISTS "${OMBRA_CORE_DIR}/include/ombra/core/expansion_tree.hpp")
  if("$ENV{OMBRA_REQUIRE_NATIVE}" STREQUAL "1")
    message(FATAL_ERROR "ombra._native needs OpenFHE 1.6.1, pybind11 and core/ of the repository")
  endif()
  message(WARNING "ombra._native not built (OpenFHE, pybind11 or core/ missing): local mode unavailable")
  return()
endif()
if(NOT "${OpenFHE_NATIVE_SIZE}" STREQUAL "64")
  message(FATAL_ERROR "OpenFHE must be built with NATIVE_SIZE=64 (found '${OpenFHE_NATIVE_SIZE}')")
endif()
string(REPLACE "-Werror" "" OMBRA_OPENFHE_CXX_FLAGS "${OpenFHE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMBRA_OPENFHE_CXX_FLAGS}")

pybind11_add_module(_native native/ombra_native.cpp "${OMBRA_CORE_DIR}/src/security_floor.cpp")
target_include_directories(_native PRIVATE native)
target_include_directories(
  _native PRIVATE "${OMBRA_CORE_DIR}/include" "${OMBRA_CORE_DIR}/src" ${OpenFHE_INCLUDE}
                  ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core
                  ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
target_link_directories(_native PRIVATE ${OpenFHE_LIBDIR})
target_link_libraries(_native PRIVATE ${OpenFHE_SHARED_LIBRARIES})
# The source calls omp_*; on Linux the -fopenmp of OpenFHE's flags links it, Apple clang needs libomp.
find_package(OpenMP COMPONENTS CXX QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_native PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(_native PROPERTIES INSTALL_RPATH "${OpenFHE_LIBDIR}" BUILD_WITH_INSTALL_RPATH ON)
# pybind11 hides symbols by default. On macOS that gives _native its own copies of the statics and
# type info that OpenFHE's headers define, so a ciphertext read here does not match the context
# held by libOPENFHEpke ("not generated with the same crypto context"). openfhe-python is built
# with default visibility on macOS for the same reason (packaging/ombra-openfhe/CMakeLists.txt).
if(APPLE)
  set_target_properties(_native PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
endif()
install(TARGETS _native DESTINATION ombra)

# Exactness test of the fast CPU operations against the reference (PERF-B); run by
# tests/test_openfhe_native_exact.py. Not installed.
find_package(OpenMP QUIET)
add_executable(ombra_native_exact native/exact_test.cpp "${OMBRA_CORE_DIR}/src/security_floor.cpp")
target_include_directories(
  ombra_native_exact PRIVATE native "${OMBRA_CORE_DIR}/include" "${OMBRA_CORE_DIR}/src" ${OpenFHE_INCLUDE}
                             ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core
                             ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
target_link_directories(ombra_native_exact PRIVATE ${OpenFHE_LIBDIR})
target_link_libraries(ombra_native_exact PRIVATE ${OpenFHE_SHARED_LIBRARIES})
set_target_properties(ombra_native_exact PROPERTIES INSTALL_RPATH "${OpenFHE_LIBDIR}" BUILD_WITH_INSTALL_RPATH ON)

# Exact test of seeded stored ciphertexts (SEED-1, docs/decisions/0023) against OpenFHE; run by
# tests/test_openfhe_seeded.py. Not installed.
add_executable(ombra_seeded_exact native/seeded_test.cpp)
target_include_directories(
  ombra_seeded_exact PRIVATE native "${OMBRA_CORE_DIR}/include" ${OpenFHE_INCLUDE}
                             ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core
                             ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
target_link_directories(ombra_seeded_exact PRIVATE ${OpenFHE_LIBDIR})
target_link_libraries(ombra_seeded_exact PRIVATE ${OpenFHE_SHARED_LIBRARIES})
set_target_properties(ombra_seeded_exact PROPERTIES INSTALL_RPATH "${OpenFHE_LIBDIR}" BUILD_WITH_INSTALL_RPATH ON)
