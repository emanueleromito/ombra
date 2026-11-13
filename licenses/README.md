# Third-party licenses

Full license texts of the libraries that are compiled into or shipped inside Ombra's binary
distributions (the `ombra` and `ombra-openfhe` wheels, and the `ombra-server` and `ombra-cpu`
images, which carry this folder as `/licenses`). The list of libraries is in `NOTICE`. Ombra's own
license is in `LICENSE`.

| File | Library | Version of the text | Shipped in |
| --- | --- | --- | --- |
| `OpenFHE-LICENSE.txt` | OpenFHE (BSD-2-Clause) | 1.6.1 | both wheels, both images |
| `openfhe-python-LICENSE.txt` | openfhe-python (BSD-2-Clause) | 1.6.1.0 | `ombra-openfhe`, CPU image |
| `cereal-LICENSE.txt` | cereal, headers in OpenFHE (BSD-3-Clause) | OpenFHE 1.6.1 submodule | both wheels, both images |
| `pybind11-LICENSE.txt` | pybind11 (BSD-3-Clause), compiled into the extensions | 3.1.0 | both wheels |
| `LLVM-OpenMP-libomp-LICENSE.txt` | LLVM OpenMP runtime libomp (Apache-2.0 with LLVM exceptions), macOS wheels | llvm 19.1.7 | both wheels |
| `GCC-libgomp-GPL-3.0.txt`, `GCC-Runtime-Library-Exception-3.1.txt` | GCC OpenMP runtime libgomp (GPL-3.0 with the GCC Runtime Library Exception), Linux wheels and images | GCC 14.2.0 | both wheels, both images |
| `gRPC-LICENSE.txt`, `gRPC-NOTICE.txt` | gRPC C++ (Apache-2.0) | 1.30.2, the version of Ubuntu 22.04 | GPU image |
| `protobuf-LICENSE.txt` | Protocol Buffers (BSD-3-Clause) | 3.12.4, the version of Ubuntu 22.04 | GPU image |
| `OpenSSL-LICENSE.txt` | OpenSSL (Apache-2.0) | 3.0.2, the version of Ubuntu 22.04 | GPU image |
| `numpy-LICENSE.txt` | numpy (BSD-3-Clause) | 2.5.4 | CPU image |

SQLite is public domain and has no license text. The NVIDIA CUDA runtime in the GPU image is
under the NVIDIA CUDA Toolkit End User License Agreement (https://docs.nvidia.com/cuda/eula/).
The Debian and Ubuntu packages in the images keep their own copyright files in
`/usr/share/doc/<package>/copyright`.

The texts were fetched from the upstream repositories at the versions above. When a pinned
version changes, fetch the text of the new version. The copies in `client/licenses/` and
`packaging/ombra-openfhe/licenses/` are the files each wheel carries (`license-files`); keep them
equal to the files here (`packaging/.../LICENSE-openfhe` is `OpenFHE-LICENSE.txt`, and `LICENSE-openfhe-python` is `openfhe-python-LICENSE.txt`).
