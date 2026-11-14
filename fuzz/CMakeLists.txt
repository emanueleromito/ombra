# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# libFuzzer targets for the parsers of untrusted bytes that build on the CPU. Added from the root
# CMakeLists.txt only with -DOMBRA_FUZZ=ON (clang). Run with `make fuzz`; see fuzz/README.md.
# Every target links ombra_storage / ombra_server_admin, which the root builds with the same
# sanitizer flags (-fsanitize=fuzzer-no-link,address,undefined).

set(OMBRA_FUZZ_TARGETS envelope seeded segment blob wal recovery api_keys http auth_header)

add_custom_target(ombra_fuzzers)
foreach(t ${OMBRA_FUZZ_TARGETS})
  add_executable(ombra_fuzz_${t} fuzz_${t}.cpp)
  target_include_directories(ombra_fuzz_${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(ombra_fuzz_${t} PRIVATE ombra_storage ombra_server_admin)
  target_link_options(ombra_fuzz_${t} PRIVATE -fsanitize=fuzzer)
  add_dependencies(ombra_fuzzers ombra_fuzz_${t})
endforeach()

# Writes the seed corpora (the committed fuzz/corpus is its output).
add_executable(ombra_fuzz_seedgen seedgen.cpp)
target_include_directories(ombra_fuzz_seedgen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ombra_fuzz_seedgen PRIVATE ombra_storage ombra_server_admin)
add_dependencies(ombra_fuzzers ombra_fuzz_seedgen)
