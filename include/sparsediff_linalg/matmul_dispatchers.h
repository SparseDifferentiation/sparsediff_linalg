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
 */
#ifndef MATMUL_DISPATCHERS_H
#define MATMUL_DISPATCHERS_H

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
   itself (version-guarded). Every fill bumps C's values_version. The fills
   assert that C has the kind the matching alloc produced. */
matrix *BTA_matrices_alloc(matrix *A, matrix *B);
void BTA_matrices_fill_values(matrix *A, matrix *B, matrix *C);
void BTDA_matrices_fill_values(matrix *A, const double *d, matrix *B, matrix *C);

/* Polymorphic dispatchers for C = B @ A, where B must be of kind
   MATRIX_PERMUTED_DENSE (asserted) and A is any kind. The output C is always
   permuted_dense. A sparse A is read through its CSC mirror, which the fill
   refreshes itself. The fill bumps C's values_version. */
matrix *BA_pd_matrices_alloc(const matrix *B, matrix *A);
void BA_pd_matrices_fill_values(const matrix *B, const matrix *A, matrix *C);

/* Polymorphic dispatchers for C = kron(I_p, A) @ J, where A is a local dense
   matrix and J is any kind. The output C is always stacked_pd.

   Contract (asserted): A must be a "full" permuted_dense, i.e. of kind
   MATRIX_PERMUTED_DENSE with its dense block covering the whole (m, n) shape
   (as built by new_permuted_dense_full). A sparse J is read through its CSC
   mirror, which the fill refreshes itself. The fill bumps C's values_version. */
matrix *BA_dense_kron_matrices_alloc(const matrix *A, int p, matrix *J);
void BA_dense_kron_matrices_fill_values(const matrix *A, int p, const matrix *J,
                                        matrix *C);

#endif /* MATMUL_DISPATCHERS_H */
