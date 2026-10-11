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
