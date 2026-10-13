/*
 * Copyright 2026 Daniel Cederberg and William Zhang
 *
 * This file is part of the SparseDiffEngine project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CSR_KERNELS_H
#define CSR_KERNELS_H

#include "sparsediff_linalg/CSR_matrix.h"

/* Kernels on the raw CSR format, for consumers that build sparsity patterns by
   hand: the index-map accumulator. Sums and sparse products go through the
   matrix_sum.h and matmul.h dispatchers. */

/* ---- sums ---- */

/* Accumulates `nnz` values from `vals` into `out` at the positions given by
   `idx_map` (length `nnz`). Caller must zero `out` before calling. */
void accumulator(const double *vals, int nnz, const int *idx_map, double *out);

#endif /* CSR_KERNELS_H */
