# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# Experiment 0003: CPU fused score. Stand-alone build against OpenFHE (no CUDA).
cmake_minimum_required(VERSION 3.22)
project(ombra_research_0003 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenFHE REQUIRED)
find_package(OpenMP REQUIRED)
string(REPLACE "-Werror" "" OFHE_FLAGS "${OpenFHE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OFHE_FLAGS}")
add_executable(fused_score fused_score.cpp)
target_include_directories(fused_score PRIVATE ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include
  ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
target_link_directories(fused_score PRIVATE ${OpenFHE_LIBDIR})
target_link_libraries(fused_score PRIVATE ${OpenFHE_SHARED_LIBRARIES} OpenMP::OpenMP_CXX)
