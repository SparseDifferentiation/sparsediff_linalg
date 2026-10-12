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
#ifndef OLD_CSR_H
#define OLD_CSR_H

/* LEGACY: kept only as a test oracle for the library's own kernels. Compiled
   into sparsediff_linalg_tests, never into the library. Not part of the API. */

#include "sparsediff_linalg/CSR_matrix.h"
#include <stdbool.h>

/* Build (I_p kron A) = blkdiag(A, A, ..., A) of size (p*A->m) x (p*A->n) */
CSR_matrix *block_diag_repeat_csr(const CSR_matrix *A, int p);

/* Build (A kron I_p) of size (A->m * p) x (A->n * p) with nnz = A->nnz * p. */
CSR_matrix *kron_identity_csr(const CSR_matrix *A, int p);

/* Computes values of the row matrix C = z^T A (column indices must have been
   pre-computed) and transposed matrix AT must be provided) */
void Ax_csr_fill_values(const CSR_matrix *AT, const double *z, CSR_matrix *C);

/* Insert value into CSR_matrix matrix A with just one row at col_idx. Assumes that A
has enough space and that A does not have an element at col_idx. It does update
nnz. */
void csr_insert_value(CSR_matrix *A, int col_idx, double value);

/* Compute C = diag(d) * A where d is an array and A, C are CSR_matrix matrices
 * d must have length m
 * C must be pre-allocated with same dimensions as A */
void diag_csr_mult(const double *d, const CSR_matrix *A, CSR_matrix *C);

/* y = Ax, where y is returned as dense (no column offset) */
void Ax_csr_wo_offset(const CSR_matrix *A, const double *x, double *y);

/* Count number of columns with nonzero entries in A and marks them in col_nz */
int count_nonzero_cols(const CSR_matrix *A, bool *col_nz);

/* inserts 'idx' into array 'arr' in sorted order, and moves the other elements */
void insert_idx(int idx, int *arr, int len);

/* computes dense y = Ax */
void Ax_csr(const CSR_matrix *A, const double *x, double *y, int col_offset);

#endif /* OLD_CSR_H */
