# Server components. Storage (T2.5) needs neither OpenFHE nor a GPU, so it builds on the CPU.
add_subdirectory(storage)
# T2.6 cluster cache: pure logic builds on the CPU, the CUDA part only with OMBRA_ENABLE_CUDA.
add_subdirectory(cache)
# T2.7 search engine: pure logic on the CPU, the engine with OMBRA_ENABLE_CUDA.
add_subdirectory(search)
