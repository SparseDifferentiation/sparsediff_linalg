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
#ifndef MATMUL_H
#define MATMUL_H

#include "sparsediff_linalg/matrix.h"

/* Polymorphic dispatchers for C = BT @ A and C = BT @ diag(d) @ A, where A
   and B are any matrix type (permuted_dense, stacked_pd, or sparse_matrix).

   Output type varies with the operand pair:
     - B is PD                          -> output is permuted_dense.
     - B is stacked_pd                  -> output is stacked_pd.
     - B is sparse_matrix, A is PD      -> output is permuted_dense.
     - B is sparse_matrix, A is spd     -> output is stacked_pd.
     - B is sparse_matrix, A is sparse  -> output is sparse_matrix.

   Sparse operands are read through their CSC mirror, which the fill refreshes
   itself (version-guarded); a permuted_dense B against a sparse A is read
   through B's transpose cache, refreshed iff B's values changed. Every fill
   bumps C's values_version. The fills assert that C has the kind the matching
   alloc produced. */
matrix *BTA_matrices_alloc(matrix *A, matrix *B);
void BTA_matrices_fill_values(matrix *A, matrix *B, matrix *C);
void BTDA_matrices_fill_values(matrix *A, const double *d, matrix *B, matrix *C);

/* Dispatchers for C = AT @ A and C = AT @ diag(d) @ A, where A is any kind.
   Each kind uses its symmetric kernel (sparse: upper triangle on the CSC
   mirror, then mirrored; permuted_dense: one gemm; stacked_pd: one gemm per
   block), and the output has A's kind. A sparse A is read through its CSC
   mirror, which the fill refreshes itself. The fill bumps C's values_version
   and asserts C has A's kind. */
matrix *ATA_matrices_alloc(matrix *A);
void ATDA_matrices_fill_values(const matrix *A, const double *d, matrix *C);

/* Polymorphic dispatchers for C = B @ A, where B is permuted_dense or
   sparse_matrix (a stacked_pd B asserts) and A is any kind.

   Output type varies with the operand pair:
     - B is PD                          -> output is permuted_dense.
     - B is sparse_matrix, A is PD      -> output is permuted_dense.
     - B is sparse_matrix, A is spd     -> output is stacked_pd.
     - B is sparse_matrix, A is sparse  -> output is sparse_matrix.

   A sparse B is read in place (its CSR is B^T in CSC form, so no transpose is
   formed); a sparse A is read through its CSC mirror, which the fill refreshes
   itself. The fill bumps C's values_version and asserts C has the kind the
   alloc produced. */
matrix *BA_matrices_alloc(const matrix *B, matrix *A);
void BA_matrices_fill_values(const matrix *B, const matrix *A, matrix *C);

/* Polymorphic dispatchers for C = kron(I_p, A) @ J, where A is a local dense
   matrix and J is any kind. The output C is always stacked_pd.

   Contract (asserted): A must be a "full" permuted_dense, i.e. of kind
   MATRIX_PERMUTED_DENSE with its dense block covering the whole (m, n) shape
   (as built by new_permuted_dense_full). A sparse J is read through its CSC
   mirror, which the fill refreshes itself. The fill bumps C's values_version. */
matrix *BA_dense_kron_matrices_alloc(const matrix *A, int p, matrix *J);
void BA_dense_kron_matrices_fill_values(const matrix *A, int p, const matrix *J,
                                        matrix *C);

/* The sparse sibling: C = kron(I_p, A) @ J for a sparse_matrix A and J of any
   kind with J->m == p * A->n. C is a sparse_matrix. A sparse J is read through
   its CSC mirror; a permuted_dense / stacked_pd J is converted to CSC (through
   its CSR view, which the fill may refresh) into a buffer bound to C. The fill
   bumps C's values_version and leaves C's CSC mirror fresh. */
matrix *BA_sparse_kron_matrices_alloc(const matrix *A, int p, matrix *J);
void BA_sparse_kron_matrices_fill_values(const matrix *A, int p, matrix *J,
                                         matrix *C);

/* The mirror image: C = kron(A, I_m) @ J, same contract on A, J any kind, C
   always stacked_pd with one block per i < m holding output rows i + m*c. No
   Kronecker product is formed. */
matrix *BA_dense_kron_I_matrices_alloc(const matrix *A, int m, matrix *J);
void BA_dense_kron_I_matrices_fill_values(const matrix *A, int m, const matrix *J,
                                          matrix *C);

#endif /* MATMUL_H */
