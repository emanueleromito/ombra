# ombra-openfhe

[openfhe-python](https://github.com/openfheorg/openfhe-python) 1.6.1.0 with OpenFHE 1.6.1 inside the
wheel, built for CPython 3.11 to 3.13 on Linux (x86_64, aarch64) and macOS (arm64, x86_64). The
import name is `openfhe`, the same as the `openfhe` package on PyPI, so install only one of them.
Ombra depends on this one because the `openfhe` wheel on PyPI is a CPython 3.12 x86_64 Linux build.

The build files are in the Ombra repository (`packaging/ombra-openfhe/`). OpenFHE and openfhe-python
are BSD 2-Clause; both license files are in the wheel.
