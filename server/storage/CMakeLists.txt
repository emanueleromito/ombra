# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# Storage engine: segments, write-ahead log, SQLite catalog, recovery. No OpenFHE, no GPU.
include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)  # extract time = now (the option below needs CMake 3.24)
endif()
# SQLite amalgamation 3.46.1, pinned by hash (public domain). Built as a static library.
FetchContent_Declare(
  sqlite_amalgamation
  URL https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip
  URL_HASH SHA256=77823cb110929c2bcb0f5d48e4833b5c59a8a6e40cdea3936b99e199dbbe5784)
FetchContent_MakeAvailable(sqlite_amalgamation)

add_library(ombra_sqlite STATIC ${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c)
target_include_directories(ombra_sqlite SYSTEM PUBLIC ${sqlite_amalgamation_SOURCE_DIR})
target_compile_definitions(ombra_sqlite PRIVATE SQLITE_THREADSAFE=1 SQLITE_OMIT_LOAD_EXTENSION=1)
target_compile_options(ombra_sqlite PRIVATE -w)
find_package(Threads REQUIRED)
target_link_libraries(ombra_sqlite PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

add_library(ombra_storage STATIC src/segment.cpp src/wal.cpp src/catalog.cpp src/store.cpp
                                 src/fsutil.cpp src/blob.cpp src/verify.cpp src/backup.cpp)
target_include_directories(ombra_storage PUBLIC include
                           ${CMAKE_CURRENT_SOURCE_DIR}/../../core/include)
target_link_libraries(ombra_storage PUBLIC ombra_sqlite)

# Offline tools: read-only integrity check and online backup. CPU only, no OpenFHE.
add_executable(ombra_verify tools/ombra_verify.cpp)
target_link_libraries(ombra_verify PRIVATE ombra_storage)
add_executable(ombra_backup tools/ombra_backup.cpp)
target_link_libraries(ombra_backup PRIVATE ombra_storage)

foreach(t segment wal catalog recovery blob seeded verify backup diskfull)
  add_executable(ombra_storage_${t}_test tests/${t}_test.cpp)
  target_link_libraries(ombra_storage_${t}_test PRIVATE ombra_storage)
  add_test(NAME storage_${t} COMMAND ombra_storage_${t}_test)
endforeach()
# The CLI tests run the real binaries.
add_dependencies(ombra_storage_verify_test ombra_verify)
target_compile_definitions(ombra_storage_verify_test PRIVATE
                           OMBRA_VERIFY_BIN="$<TARGET_FILE:ombra_verify>")
add_dependencies(ombra_storage_backup_test ombra_backup)
target_compile_definitions(ombra_storage_backup_test PRIVATE
                           OMBRA_BACKUP_BIN="$<TARGET_FILE:ombra_backup>")
