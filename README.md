# sparsediff_linalg

Sparse linear algebra for [SparseDiffEngine](https://github.com/SparseDifferentiation/SparseDiffEngine),
the C differentiation engine behind CVXPY's nonlinear (DNLP) extension. The library
provides CSR, CSC and COO sparse matrix types, a polymorphic `matrix` abstraction
(sparse, permuted-dense and stacked permuted-dense blocks) with function-pointer
dispatch, and the BLAS-backed kernels the engine uses to assemble sparse Jacobians and
Hessians. It depends on nothing but a C99 compiler and a CBLAS implementation, and
SparseDiffEngine consumes it as a git submodule. Licensed under Apache 2.0.

## Build and test

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build            # or ./build/sparsediff_linalg_tests
```

Add `-DSP_TRACK_MEMORY=ON` to configure with the peak-memory counters compiled in.

A CBLAS implementation is required: the Accelerate framework on macOS (found
automatically), OpenBLAS or netlib BLAS + CBLAS on Linux (`libopenblas-dev` on
Debian/Ubuntu), and OpenBLAS via vcpkg or conda on Windows.
