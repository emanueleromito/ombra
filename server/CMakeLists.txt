# Server components. Storage needs neither OpenFHE nor a GPU, so it builds on the CPU.
add_subdirectory(storage)
# Cluster cache: pure logic builds on the CPU, the CUDA part only with OMBRA_ENABLE_CUDA.
add_subdirectory(cache)
# Search engine: pure logic on the CPU, the engine with OMBRA_ENABLE_CUDA.
add_subdirectory(search)
# gRPC server: policy on the CPU, service and executable with CUDA and gRPC for C++.
add_subdirectory(grpc)
