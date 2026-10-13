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
#ifndef MATRIX_H
#define MATRIX_H

#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include <stdint.h>

/* Polymorphic matrix base. Concrete types embed `matrix` as their first
   member and implement the vtable slots below. Currently implemented:
       1. sparse_matrix  — generic CSR_matrix-backed matrix.
       2. permuted_dense — matrix whose nonzeros lie in a single dense block
                           located at chosen rows and columns of the global
                           index space.
       3. stacked_pd     — vertical stack of permuted_dense blocks.
   `kind` tags which one a handle is. */

typedef struct matrix matrix;

/* Concrete type behind a matrix handle. MATRIX_SPARSE is 0 so a zeroed struct
   is a sparse_matrix. */
typedef enum
{
    MATRIX_SPARSE = 0,     /* sparse_matrix: CSR-backed */
    MATRIX_PERMUTED_DENSE, /* permuted_dense: dense block at permuted rows/cols */
    MATRIX_STACKED_PD      /* stacked_pd: vertical stack of permuted_dense blocks */
} matrix_kind;

/* Returns a CSR_matrix view of A */
typedef CSR_matrix *(*matrix_to_csr_fn)(matrix *A);

/* Allocate C = A[map, :]. C stores map internally, so the fill takes none. */
typedef matrix *(*matrix_row_gather_alloc_fn)(const matrix *A, const int *map,
                                              int m_out);

/* Fill values of C = A[map, :] */
typedef void (*matrix_row_gather_fill_values_fn)(const matrix *A, matrix *C);

/* diag_vec: A is an (n, A->n) Jacobian for a length-n vector; output is
   (n*n, A->n) where row i lands at output row i*(n+1) (column-major
   diagonal positions). Other output rows are structurally zero. */
typedef matrix *(*matrix_diag_vec_alloc_fn)(matrix *A);
typedef void (*matrix_diag_vec_fill_values_fn)(matrix *A, matrix *out);

/* Allocate C = row-reduce of A: C[j, :] = sum of rows i with group[i] == j,
   group[i] in [0, m_out). C stores the reduction map internally, so the fill
   takes none. */
typedef matrix *(*matrix_row_reduce_alloc_fn)(const matrix *A, const int *group,
                                              int m_out);

/* Fill values of C = row-reduce of A. */
typedef void (*matrix_row_reduce_fill_values_fn)(const matrix *A, matrix *C);

typedef void (*matrix_free_fn)(matrix *self);

struct matrix
{
    int m, n, nnz;
    double *x; /* non-owning pointer to the value buffer */
    matrix_kind kind;

    /* Monotone counter bumped whenever the matrix's values change. Consumers
       that mirror the values into a cache (CSC mirror, CSR view, ...) record
       the version they last saw and refresh iff it differs. Every library
       fill bumps the version of the matrix it writes whenever it changes its
       values, and refreshes any operand cache it reads, so the only rule left for
       callers is: code that writes x directly must call matrix_values_changed on the
       OWNER of the buffer — aliased children (spd blocks, cache views) have no
       version of their own. */
    uint64_t values_version;

    /* Views */
    matrix_to_csr_fn to_csr;

    /* Atom-specific ops */
    matrix_row_gather_alloc_fn row_gather_alloc;
    matrix_row_gather_fill_values_fn row_gather_fill_values;
    matrix_diag_vec_alloc_fn diag_vec_alloc;
    matrix_diag_vec_fill_values_fn diag_vec_fill_values;
    matrix_row_reduce_alloc_fn row_reduce_alloc;
    matrix_row_reduce_fill_values_fn row_reduce_fill_values;

    /* Lifecycle */
    matrix_free_fn free_fn;
};

/* Notify the library after writing A->x directly. */
static inline void matrix_values_changed(matrix *A)
{
    A->values_version++;
}

/* Free helper */
static inline void free_matrix(matrix *m)
{
    if (m)
    {
        m->free_fn(m);
    }
}

/* Allocate a new matrix with the same kind and sparsity as A. Values are
   uninitialized. */
matrix *copy_sparsity_matrices_alloc(const matrix *A);

/* Fill values of C = diag(d) @ A, where len(d) = A->m. C must have been
   allocated by copy_sparsity_matrices_alloc(A). Bumps C's values_version. */
void DA_matrices_fill_values(const double *d, const matrix *A, matrix *C);

/* Allocate AT = transpose(A), of A's kind. Values are uninitialized. */
matrix *transpose_matrices_alloc(const matrix *A);

/* Fill values of AT = transpose(A); AT must come from
   transpose_matrices_alloc(A). Bumps AT's values_version. */
void transpose_matrices_fill_values(const matrix *A, matrix *AT);

/* ---- Constructors of the three kinds. Each returns a matrix handle; the kinds'
   layouts are private. ---- */

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

/* stacked_pd represents a matrix that is the (vertical) union of 'n_blocks'
   permuted_dense blocks. Two different blocks have disjoint row permutations,
   but the column permutations may overlap across blocks.

   `src_block_idx_p` and `src_block_idx` form a CSR-style map (offsets + flat
   data) from each output block to the source block indices that contribute to
   it in the spd-producing operations that built this spd. For operations with
   one source per output every entry of `src_block_idx_p` increments by one. For
   coalesce a single output block can be fed by multiple source blocks. */
/* The kind's layout is private; consumers hold it through the matrix handle. */

/* Constructor for stacked_pd. Takes ownership of every block in 'blocks'. The
   arrays 'src_block_idx_p' and 'src_block_idx' map each block in the new
   stacked_pd (ie. the block in 'blocks') to the source block(s) that
   contributed to it in the spd-producing operations that built this spd. If
   src_block_idx_p and src_block_idx are NULL, then src_block_idx is ignored and
   the constructor produces the identity mapping (each output block has itself
   as its one source). */
matrix *new_stacked_pd(int m, int n, int n_blocks, permuted_dense **blocks,
                       const int *src_block_idx_p, const int *src_block_idx);

#endif /* MATRIX_H */
