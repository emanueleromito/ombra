# The two OpenFHE command-line drivers the `openfhe` tests call, built without CUDA.
#
# core/CMakeLists.txt links these drivers against the CUDA library, which a GitHub runner does not
# have. Their sources use OpenFHE and header-only code from core/ only, so this file builds the
# same sources on their own. Used by `make test-openfhe-local`.
cmake_minimum_required(VERSION 3.22)
project(ombra_openfhe_drivers LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The tests look for the drivers in <out>/core/<name>.
set(OMBRA_DRIVER_OUT "${CMAKE_BINARY_DIR}" CACHE PATH "directory that gets core/<driver>")
set(OMBRA_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../core" CACHE PATH "core/ of the repository")

find_package(OpenFHE REQUIRED)
if(NOT "${OpenFHE_NATIVE_SIZE}" STREQUAL "64")
  message(FATAL_ERROR "OpenFHE must be built with NATIVE_SIZE=64 (found '${OpenFHE_NATIVE_SIZE}')")
endif()
# Same flags as core/CMakeLists.txt: OpenFHE's backend defines and OpenMP, without -Werror.
string(REPLACE "-Werror" "" OMBRA_OPENFHE_CXX_FLAGS "${OpenFHE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMBRA_OPENFHE_CXX_FLAGS}")

foreach(driver ombra_expand ombra_encoding_check)
  add_executable(${driver} "${OMBRA_CORE_DIR}/src/${driver}.cpp"
                           "${OMBRA_CORE_DIR}/src/security_floor.cpp")
  target_include_directories(
    ${driver} PRIVATE "${OMBRA_CORE_DIR}/include" "${OMBRA_CORE_DIR}/src" ${OpenFHE_INCLUDE}
                      ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core
                      ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
  target_link_directories(${driver} PRIVATE ${OpenFHE_LIBDIR})
  target_link_libraries(${driver} PRIVATE ${OpenFHE_SHARED_LIBRARIES})
  set_target_properties(
    ${driver} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OMBRA_DRIVER_OUT}/core"
                         INSTALL_RPATH "${OpenFHE_LIBDIR}" BUILD_WITH_INSTALL_RPATH ON)
endforeach()
