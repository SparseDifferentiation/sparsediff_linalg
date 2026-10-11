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
#ifndef PERMUTED_DENSE_H
#define PERMUTED_DENSE_H

#include "sparsediff_linalg/matrix.h"

/* permuted_dense represents a matrix whose only nonzeros lie in a dense block
   of size m0 x n0, after rows and columns are restricted to chosen subsets of
   the global index space. For local indices (ii, jj) with 0 <= ii < m0 and 0 <=
   jj < n0,

       M[row_perm[ii], col_perm[jj]] = X[ii, jj].

   All other entries of M are zero. row_perm and col_perm are stored in strictly
   increasing order; the constructor asserts this. */
/* The kind's layout is private; consumers hold it through the matrix handle.
   The typedef exists so block arrays can be passed to new_stacked_pd. */
typedef struct permuted_dense permuted_dense;

/* Constructor. row_perm and col_perm must be strictly increasing in their
   respective ranges. If X_data is NULL the value buffer is allocated but left
   uninitialized. Otherwise m0 * n0 entries are copied. */
matrix *new_permuted_dense(int m, int n, int m0, int n0, const int *row_perm,
                           const int *col_perm, const double *X_data);

/* Convenience constructor for the trivial-perm case: row_perm = [0..m-1],
   col_perm = [0..n-1], dense block fills the full (m, n) shape. */
matrix *new_permuted_dense_full(int m, int n, const double *data);

#endif /* PERMUTED_DENSE_H */
