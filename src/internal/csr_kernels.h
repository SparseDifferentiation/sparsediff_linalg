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
#ifndef INTERNAL_CSR_KERNELS_H
#define INTERNAL_CSR_KERNELS_H

/* Private CSR / CSC kernels: used by the sparse kind's vtable, the matrix sums
   and the matmul dispatchers, never by consumers. */

#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"

/* Accumulates `nnz` values from `vals` into `out` at the positions given by
   `idx_map` (length `nnz`). Caller must zero `out` before calling. */
void accumulator(const double *vals, int nnz, const int *idx_map, double *out);

/* Compute sparsity pattern of C = A + B (and sets C->nnz). Internal: consumers
   go through sum_matrices_* (matrix_sum.h). */
void sum_csr_alloc(const CSR_matrix *A, const CSR_matrix *B, CSR_matrix *C);

/* Fills values of C = A + B (assuming C's sparsity pattern is set) */
void sum_csr_fill_values(const CSR_matrix *A, const CSR_matrix *B, CSR_matrix *C);

/* Fills values of C = diag(d1) * A + diag(d2) * B (assuming C's sparsity is set)*/
void sum_scaled_csr_matrices_fill_values(const CSR_matrix *A, const CSR_matrix *B,
                                         CSR_matrix *C, const double *d1,
                                         const double *d2);

/* Compute sparsity pattern of out = A + B + C + D. To fill the values in a single
   pass, it also allocates (sp_malloc, caller frees) one idx_map per input:
   idx_maps[k] has size equal to the nnz of the k-th input, and idx_maps[k][j] is
   the position in out->x of that input's j-th nonzero (consumed by accumulator). */
CSR_matrix *sum_4_csr_alloc(const CSR_matrix *A, const CSR_matrix *B,
                            const CSR_matrix *C, const CSR_matrix *D,
                            int *idx_maps[4]);

/* Compute sparsity pattern of out = ops[0] + ... + ops[n-1] (n >= 1, all of the
   same shape). Also fills the caller-allocated idx_map, of length sum_k
   ops[k]->nnz: one segment per input, back to back, where entry j of input k's
   segment is the position in out->x of that input's j-th nonzero (consumed by
   accumulator). Cost is O(m * n + total nnz + per-row sort of the unique
   columns). */
CSR_matrix *sum_n_csr_alloc(const CSR_matrix *const *ops, int n, int *idx_map);

/* Compute y = kron(I_p, A) @ x where A is m x n and x is(n*p)-length vector.
   The output y is m*p-length vector corresponding to
   y = [A @ x1; A @ x2; ...; A @ xp] where x is divided into p blocks of n
   elements.
*/
void block_left_multiply_vec(const CSR_matrix *A, const double *x, double *y, int p);

#endif /* INTERNAL_CSR_KERNELS_H */
