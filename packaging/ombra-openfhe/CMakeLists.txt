# ombra-openfhe: the openfhe-python 1.6.1.0 module (import name `openfhe`), built against OpenFHE
# 1.6.1 and installed as the package openfhe/ (same layout as the PyPI wheel of openfhe-python).
# The sources are those of the pinned tag, fetched into openfhe-python-src by
# ci/wheel/fetch_openfhe_python.sh; this file replaces its CMakeLists.txt, which installs to
# site-packages root and sets its own rpath.
cmake_minimum_required(VERSION 3.22)
project(ombra_openfhe LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(APPLE)
  set(CMAKE_CXX_VISIBILITY_PRESET default)
endif()

set(OFPY_SRC "${CMAKE_CURRENT_SOURCE_DIR}/openfhe-python-src" CACHE PATH "openfhe-python v1.6.1.0")
set(OFPY_VERSION "1.6.1.0")
if(NOT EXISTS "${OFPY_SRC}/src/lib/bindings.cpp")
  message(FATAL_ERROR "openfhe-python sources not found in ${OFPY_SRC} (ci/wheel/fetch_openfhe_python.sh)")
endif()

find_package(OpenFHE 1.6.1 REQUIRED)
if(NOT "${OpenFHE_NATIVE_SIZE}" STREQUAL "64")
  message(FATAL_ERROR "OpenFHE must be built with NATIVE_SIZE=64 (found '${OpenFHE_NATIVE_SIZE}')")
endif()
string(REPLACE "-Werror" "" OMBRA_OPENFHE_CXX_FLAGS "${OpenFHE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMBRA_OPENFHE_CXX_FLAGS}")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 2.12 CONFIG REQUIRED)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)  # fails to link (upstream note)

pybind11_add_module(
  openfhe
  "${OFPY_SRC}/src/lib/bindings.cpp"
  "${OFPY_SRC}/src/lib/binfhe_bindings.cpp"
  "${OFPY_SRC}/src/lib/binfhe/binfhecontext_wrapper.cpp"
  "${OFPY_SRC}/src/lib/pke/serialization.cpp"
  "${OFPY_SRC}/src/lib/pke/fbt.cpp"
  "${OFPY_SRC}/src/lib/pke/schemeswitching_data_serializer.cpp")
target_compile_definitions(openfhe PRIVATE OPENFHE_PYTHON_VERSION_STR="${OFPY_VERSION}")
target_include_directories(
  openfhe
  PRIVATE "${OFPY_SRC}/src/include" "${OFPY_SRC}/src/include/pke" "${OFPY_SRC}/src/include/binfhe"
          "${OFPY_SRC}/src/include/docstrings" ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include
          ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
target_link_directories(openfhe PRIVATE ${OpenFHE_LIBDIR})
target_link_libraries(openfhe PRIVATE ${OpenFHE_SHARED_LIBRARIES})
find_package(OpenMP COMPONENTS CXX QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(openfhe PRIVATE OpenMP::OpenMP_CXX)
endif()
# auditwheel and delocate find OpenFHE through this rpath and replace it with the bundled copy.
set_target_properties(openfhe PROPERTIES INSTALL_RPATH "${OpenFHE_LIBDIR}" BUILD_WITH_INSTALL_RPATH ON)

install(TARGETS openfhe LIBRARY DESTINATION openfhe)
install(FILES "${OFPY_SRC}/src/__init__.py" DESTINATION openfhe)
