# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
cmake_minimum_required(VERSION 3.22)

project(ombra LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(OMBRA_ENABLE_CUDA "Build CUDA kernels" ON)
option(OMBRA_ENABLE_SANITIZERS "Build with AddressSanitizer and UBSan (CPU code)" OFF)

if(OMBRA_ENABLE_CUDA)
  # Ampere only for now: sm_80 (A100), sm_86 (A10).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(OMBRA_ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

enable_testing()

# Components are added as they gain a CMakeLists.txt.
foreach(component core server)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${component}/CMakeLists.txt")
    add_subdirectory(${component})
  endif()
endforeach()
