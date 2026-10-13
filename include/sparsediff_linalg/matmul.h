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

#endif /* MATMUL_H */
