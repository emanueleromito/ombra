# T2.6 GPU cluster cache. The pure logic (offset allocator, LRU with pinning, metrics) is plain C++
# and tested on the CPU. The CUDA part is built only with OMBRA_ENABLE_CUDA=ON.
add_library(ombra_cache_logic STATIC src/range_allocator.cpp src/arena_lru.cpp)
target_include_directories(ombra_cache_logic PUBLIC include)

foreach(t range_allocator lru)
  add_executable(cache_${t}_test tests/cache_${t}_test.cpp)
  target_include_directories(cache_${t}_test PRIVATE tests)
  target_link_libraries(cache_${t}_test PRIVATE ombra_cache_logic)
  add_test(NAME cache_${t} COMMAND cache_${t}_test)
endforeach()

if(NOT OMBRA_ENABLE_CUDA)
  return()
endif()

# CUDA runtime wrapper: pinned pool, device arena, paging with a copy stream and events.
add_library(ombra_cache STATIC src/gpu_cache.cu src/cuda_util.cu)
target_include_directories(ombra_cache PUBLIC include)
set_target_properties(ombra_cache PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_STANDARD 17)
target_link_libraries(ombra_cache PUBLIC ombra_cache_logic ombra_rns_cuda)

# GPU test driver: paging does not change the search results (needs OpenFHE for the ciphertexts).
add_executable(ombra_cache_search src/ombra_cache_search.cpp)
target_link_libraries(ombra_cache_search PRIVATE ombra_cache ombra_core)

add_executable(ombra_bench_cache src/ombra_bench_cache.cpp)
target_link_libraries(ombra_bench_cache PRIVATE ombra_cache ombra_core)
