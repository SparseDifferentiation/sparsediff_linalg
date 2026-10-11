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
#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include "sparsediff_linalg/matrix.h"

/* The sparse kind: a matrix backed by a CSR_matrix. Its layout is private; use
   the matrix handle (m, n, nnz, x, kind, values_version, to_csr and the other
   vtable slots). */

/* Constructor. Takes ownership of A; the caller must not free A separately
   (free_matrix on the returned matrix frees A). */
matrix *new_sparse_matrix(CSR_matrix *A);

/* Convenience: allocate a sparse matrix of shape (m, n) with capacity for
   nnz entries. Equivalent to new_sparse_matrix(new_CSR_matrix(m, n, nnz)).
   Sparsity pattern and values are uninitialized. */
matrix *new_sparse_matrix_alloc(int m, int n, int nnz);

#endif /* SPARSE_MATRIX_H */
