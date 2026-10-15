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
#ifndef INTERNAL_MATMUL_KRON_H
#define INTERNAL_MATMUL_KRON_H

/* Private: the per-kind kron(I_p, A) @ J dispatchers that BA_kron_matrices_*
   (sparsediff_linalg/matmul_kron.h) routes to. Library and tests only. */

#include "sparsediff_linalg/matrix.h"

/* C = kron(I_p, A) @ J, where A is a local dense matrix and J is any kind. The
   output C is always stacked_pd.

   Contract (asserted): A must be a "full" permuted_dense, i.e. of kind
   MATRIX_PERMUTED_DENSE with its dense block covering the whole (m, n) shape
   (as built by new_permuted_dense_full). A sparse J is read through its CSC
   mirror, which the fill refreshes itself. The fill bumps C's values_version. */
matrix *BA_dense_kron_matrices_alloc(const matrix *A, int p, matrix *J);
void BA_dense_kron_matrices_fill_values(const matrix *A, int p, const matrix *J,
                                        matrix *C);

/* The sparse sibling of BA_dense_kron_matrices_*: C = kron(I_p, A) @ J for a
   sparse_matrix A and J of any kind with J->m == p * A->n. C is a
   sparse_matrix; J is read through its CSR view. The fill bumps C's
   values_version. */
matrix *BA_sparse_kron_matrices_alloc(const matrix *A, int p, matrix *J);
void BA_sparse_kron_matrices_fill_values(const matrix *A, int p, matrix *J,
                                         matrix *C);

#endif /* INTERNAL_MATMUL_KRON_H */
