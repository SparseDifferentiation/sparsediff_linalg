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
   and the permuted_dense operator role, never by consumers. */

#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/matrix.h"

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
   same shape). Like sum_4_csr_alloc, it also allocates (sp_malloc, caller frees)
   one idx_map per input: idx_maps[k] has size ops[k]->nnz, and idx_maps[k][j] is
   the position in out->x of that input's j-th nonzero (consumed by accumulator).
   Cost is O(total nnz + per-row sort of the unique columns), independent of n. */
CSR_matrix *sum_n_csr_alloc(const CSR_matrix *const *ops, int n, int **idx_maps);

/* C = A @ B where A is CSR_matrix, B is CSC_matrix. Result C is CSR_matrix.
   Allocates and precomputes the sparsity pattern; no workspace required.
   Internal: consumers go through BA_matrices_* (matmul.h), which reads a sparse
   operand through its CSC mirror. */
CSR_matrix *csr_csc_matmul_alloc(const CSR_matrix *A, const CSC_matrix *B);

/* Fill values of C = A @ B where A is CSR_matrix, B is CSC_matrix. C must have
   its sparsity pattern already computed (csr_csc_matmul_alloc). */
void csr_csc_matmul_fill_values(const CSR_matrix *A, const CSC_matrix *B,
                                CSR_matrix *C);

/* Compute sparsity pattern and values for the matrix-matrix multiplication
   C = (I_p kron A) @ J where A is m x n, J is (n*p) x k, and C is (m*p) x k,
   without relying on generic sparse matrix-matrix multiplication. Specialized
   logic for this is much faster (50-100x) than generic sparse matmul.

    * J is provided in CSC_matrix format and is split into p blocks of n rows each
    * C is returned in CSC_matrix format
    * Mathematically it corresponds to  C = [A @ J1; A @ J2; ...; A @ Jp],
      where J = [J1; J2; ...; Jp]
*/
CSC_matrix *block_left_multiply_fill_sparsity(const CSR_matrix *A,
                                              const CSC_matrix *J, int p);

void block_left_multiply_fill_values(const CSR_matrix *A, const CSC_matrix *J,
                                     CSC_matrix *C);

/* Compute y = kron(I_p, A) @ x where A is m x n and x is(n*p)-length vector.
   The output y is m*p-length vector corresponding to
   y = [A @ x1; A @ x2; ...; A @ xp] where x is divided into p blocks of n
   elements.
*/
void block_left_multiply_vec(const CSR_matrix *A, const double *x, double *y, int p);

/* C = (I_p kron A) @ J via the polymorphic matrix interface.
 * A is dense m x n, J is (n*p) x k in CSC_matrix, C is (m*p) x k in CSC_matrix.
 * `work` must be sized at least A->n doubles — used as a scratch buffer when
 * a sparse column of J needs to be densified before dgemv. The caller is
 * responsible for sizing it (typically pre-sized in the corresponding
 * sparsity-build step). */
// TODO: maybe we can replace these with I_kron_X functionality?
CSC_matrix *I_kron_A_alloc(const matrix *A, const CSC_matrix *J, int p);
void I_kron_A_fill_values(const matrix *A, const CSC_matrix *J, CSC_matrix *C,
                          double *work);

#endif /* INTERNAL_CSR_KERNELS_H */
