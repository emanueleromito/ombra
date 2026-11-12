# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# Search engine library. The pure logic (layout, tombstone penalty vector) is plain C++ and
# tested on the CPU. The engine itself needs OpenFHE and CUDA and is built only with
# OMBRA_ENABLE_CUDA=ON.
add_library(ombra_search_logic STATIC src/layout.cpp)
target_include_directories(ombra_search_logic PUBLIC include)
target_link_libraries(ombra_search_logic PUBLIC ombra_storage)

add_executable(search_layout_test tests/layout_test.cpp)
target_link_libraries(search_layout_test PRIVATE ombra_search_logic)
add_test(NAME search_layout COMMAND search_layout_test)

if(NOT OMBRA_ENABLE_CUDA)
  return()
endif()

add_library(ombra_search STATIC src/session.cpp src/device_util.cu)
target_include_directories(ombra_search PUBLIC include PRIVATE src)
set_target_properties(ombra_search PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_STANDARD 17)
target_link_libraries(ombra_search PUBLIC ombra_search_logic ombra_cache ombra_core ombra_storage)

# GPU test driver: exact comparison with the OpenFHE CPU chain (tests/test_gpu_search_engine.py).
add_executable(ombra_search_engine_check src/ombra_search_engine_check.cpp)
# core/src has the CPU reference expansion operations (openfhe_expand_ops.hpp) used as the oracle.
target_include_directories(ombra_search_engine_check
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src)
target_link_libraries(ombra_search_engine_check PRIVATE ombra_search)
