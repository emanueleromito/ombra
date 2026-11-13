# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# Experiment 0011: seeded ciphertexts, load cost. Stand-alone build against OpenFHE (no CUDA).
cmake_minimum_required(VERSION 3.22)
project(ombra_research_0011 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenFHE REQUIRED)
find_package(OpenMP REQUIRED)
string(REPLACE "-Werror" "" OFHE_FLAGS "${OpenFHE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OFHE_FLAGS}")
add_executable(seeded_load seeded_load.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/src/security_floor.cpp)
target_include_directories(seeded_load PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/include ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include
  ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
target_link_directories(seeded_load PRIVATE ${OpenFHE_LIBDIR})
target_link_libraries(seeded_load PRIVATE ${OpenFHE_SHARED_LIBRARIES} OpenMP::OpenMP_CXX)
