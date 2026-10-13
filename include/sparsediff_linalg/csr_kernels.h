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

#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"

/* Kernels on the raw CSR / CSC formats, for consumers that build sparsity
   patterns by hand: sums with index maps, sparse products and Kronecker
   products, and the convolution matrix. Two-phase where applicable. */

/* ---- sums ---- */

/* Compute sparsity pattern of C = A + B (and sets C->nnz) */
void sum_csr_alloc(const CSR_matrix *A, const CSR_matrix *B, CSR_matrix *C);

/* Fills values of C = A + B (assuming C's sparsity pattern is set) */
void sum_csr_fill_values(const CSR_matrix *A, const CSR_matrix *B, CSR_matrix *C);

/* Accumulates `nnz` values from `vals` into `out` at the positions given by
   `idx_map` (length `nnz`). Caller must zero `out` before calling. */
void accumulator(const double *vals, int nnz, const int *idx_map, double *out);

/* ---- products ---- */

/* Fill values of C = A @ B where A is CSR_matrix, B is CSC_matrix.
 * C must have sparsity pattern already computed.
 */
void csr_csc_matmul_fill_values(const CSR_matrix *A, const CSC_matrix *B,
                                CSR_matrix *C);

/* C = A @ B where A is CSR_matrix, B is CSC_matrix. Result C is CSR_matrix.
 * Allocates and precomputes sparsity pattern. No workspace required.
 */
CSR_matrix *csr_csc_matmul_alloc(const CSR_matrix *A, const CSC_matrix *B);

/* ---- convolution matrix ---- */

/* Fill T_csr's row pointers and column indices for the 1D full-convolution
   Toeplitz matrix T(a), sized (m+n-1) x n with m*n nonzeros. Values (x) are
   not written; call conv_matrix_fill_values to populate them. */
void conv_matrix_fill_sparsity(CSR_matrix *T_csr, int m, int n);

/* Overwrite T_csr->x from kernel a, using the sparsity already written by
   conv_matrix_fill_sparsity. T[r, col] = a[r - col]. */
void conv_matrix_fill_values(CSR_matrix *T_csr, const double *a);

#endif /* CSR_KERNELS_H */
