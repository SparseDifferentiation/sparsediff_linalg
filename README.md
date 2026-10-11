# sparsediff_linalg

Sparse linear algebra for [SparseDiffEngine](https://github.com/SparseDifferentiation/SparseDiffEngine),
the differentiation engine behind CVXPY's nonlinear (DNLP) extension.

## Build and test

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build            # or ./build/sparsediff_linalg_tests
```

Add `-DSP_TRACK_MEMORY=ON` to configure with the peak-memory counters compiled in.
