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
#ifndef MATMUL_KRON_H
#define MATMUL_KRON_H

#include "sparsediff_linalg/matrix.h"

/* y = kron(I_p, A) @ x: x is p blocks of length A->n and y is p blocks of
   length A->m, i.e. y = [A x_1; ...; A x_p]. A is a sparse_matrix or a full
   permuted_dense (dense block covering the whole shape); stacked_pd asserts.

   Possible extension: a transposed variant y = kron(I_p, A^T) @ x would let
   callers drop a stored transpose of A that they keep only for this product.
   For a full permuted_dense it is the other dgemm transpose flag; a sparse A
   needs a scatter-style CSR kernel. Not provided yet. */
void BA_kron_vec_matrices(const matrix *A, const double *x, double *y, int p);

/* Polymorphic dispatchers for C = kron(I_p, A) @ J, where A is a local dense
   matrix and J is any kind. The output C is always stacked_pd.

   Contract (asserted): A must be a "full" permuted_dense, i.e. of kind
   MATRIX_PERMUTED_DENSE with its dense block covering the whole (m, n) shape
   (as built by new_permuted_dense_full). A sparse J is read through its CSC
   mirror, which the fill refreshes itself. The fill bumps C's values_version. */
matrix *BA_dense_kron_matrices_alloc(const matrix *A, int p, matrix *J);
void BA_dense_kron_matrices_fill_values(const matrix *A, int p, const matrix *J,
                                        matrix *C);

/* The mirror image: C = kron(A, I_m) @ J, same contract on A, J any kind, C
   always stacked_pd with one block per i < m holding output rows i + m*c. No
   Kronecker product is formed. */
matrix *BA_dense_kron_I_matrices_alloc(const matrix *A, int m, matrix *J);
void BA_dense_kron_I_matrices_fill_values(const matrix *A, int m, const matrix *J,
                                          matrix *C);

/* The sparse sibling of BA_dense_kron_matrices_*: C = kron(I_p, A) @ J for a
   sparse_matrix A and J of any kind with J->m == p * A->n. C is a
   sparse_matrix. J is converted to CSC through its CSR view (which the fill
   may refresh) into a buffer bound to C. The fill bumps C's values_version. */
matrix *BA_sparse_kron_matrices_alloc(const matrix *A, int p, matrix *J);
void BA_sparse_kron_matrices_fill_values(const matrix *A, int p, matrix *J,
                                         matrix *C);

/* Dispatcher for C = kron(I_p, A) @ J, where A is a full permuted_dense or a
   sparse_matrix (stacked_pd asserts) and J is any kind:
     - p == 1         -> BA_matrices_* (output kind as documented there).
     - A is full PD   -> BA_dense_kron_matrices_* (output is stacked_pd).
     - A is sparse    -> BA_sparse_kron_matrices_* (output is sparse_matrix).
   The fill bumps C's values_version. */
matrix *BA_kron_matrices_alloc(const matrix *A, int p, matrix *J);
void BA_kron_matrices_fill_values(const matrix *A, int p, matrix *J, matrix *C);

#endif /* MATMUL_KRON_H */
