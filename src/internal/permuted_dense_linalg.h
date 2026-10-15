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
#ifndef PERMUTED_DENSE_LINALG_H
#define PERMUTED_DENSE_LINALG_H

#include "internal/permuted_dense.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"

/* Fill values of C = diag(d) @ A where len(d) = number of (global) rows of A */
void DA_pd_fill_values(const double *d, const permuted_dense *A, permuted_dense *C);

/* Allocate new permuted dense for C = AT @ A */
matrix *ATA_pd_alloc(const permuted_dense *A);

/* Fill values of C = AT @ diag(d) @ A */
void ATDA_pd_fill_values(const permuted_dense *A, const double *d,
                         permuted_dense *C);

/* Allocate new permuted dense forC = BT @ A where A and B are both permuted_dense.
   (If B and A have no overlapping rows, then C is empty) */
matrix *BTA_pd_pd_alloc(const permuted_dense *B, const permuted_dense *A);

/* Fill values of C = BT @ A where A and B are both permuted dense. */
void BTA_pd_pd_fill_values(const permuted_dense *B, const permuted_dense *A,
                           permuted_dense *C);

/* Fill values of C = BT @ diag(d) @ A where A and B are both permuted dense. */
void BTDA_pd_pd_fill_values(const permuted_dense *B, const double *d,
                            const permuted_dense *A, permuted_dense *C);

/* Allocate C = B @ S where B is PD and S is CSR. Extracts the rows of S
   selected by B->col_perm (as of this call), re-indexes them to C's local
   columns and binds them to C as a compact CSC:
     C->kernel_iwork = [cp (C->n0 + 1) | rows (nnz_K) | pos (nnz_K)]
   with rows the local column r of B and pos the index into S->x, and
   C->kernel_dwork sized for the nnz_K values the fill gathers. B's col_perm is
   read here only, so a caller may reuse one mutable B for several allocs. */
matrix *BA_pd_csr_alloc(const permuted_dense *B, const CSR_matrix *S);

/* Fill values of C = B @ S where B is a raw row-major (C->m0 x n0_B) value
   buffer and S is the CSR BA_pd_csr_alloc saw (same pattern, any values).
   Reads the state bound to C; allocates nothing. The raw-buffer signature lets
   callers pass another block's X with the same n0 (stacked_pd, dense kron). */
void BA_pd_csr_fill_values(const double *B, int n0_B, const CSR_matrix *S,
                           permuted_dense *C);

/* Fill values of C = B^T @ A where B^T is supplied as a raw row-major (n0_B x
   m0_B) buffer and A is CSC. row_inv maps A's row indices to columns of the
   buffer (entries with row_inv[r] == -1 are skipped). Sparse-dense dot form;
   used by the BTA_pd_csc path, whose callers pass (diag(d) B)^T from
   B->kernel_dwork or B's transpose cache. */
void BTA_denseT_csc_fill_values(const double *BT, int m0_B, const int *row_inv,
                                const CSC_matrix *A, permuted_dense *C);

/* Allocate new permuted dense for C = B @ A where B and A are both PD */
matrix *BA_pd_pd_alloc(const permuted_dense *B, const permuted_dense *A);

/* Fill values of C = B @ A where A, B, and C are all permuted dense */
void BA_pd_pd_fill_values(const permuted_dense *B, const permuted_dense *A,
                          permuted_dense *C);

/* Allocate new permuted dense for C = B^T @ A where B is PD and A is CSC */
matrix *BTA_pd_csc_alloc(const permuted_dense *B, const CSC_matrix *A);

/* Fill values of C = B^T @ A where B is PD and A is CSC. */
void BTA_pd_csc_fill_values(const permuted_dense *B, const CSC_matrix *A,
                            permuted_dense *C);

/* Fill values of C = B^T @ diag(d) @ A where B is PD and A is CSC */
void BTDA_pd_csc_fill_values(const permuted_dense *B, const double *d,
                             const CSC_matrix *A, permuted_dense *C);

/* Allocate new permuted_dense for C = B^T @ A where B is Sparse CSC and A is PD. */
matrix *BTA_csc_pd_alloc(const CSC_matrix *B, const permuted_dense *A);

/* Fill values of C = B^T @ A where B is CSC and A is PD. */
void BTA_csc_pd_fill_values(const CSC_matrix *B, const permuted_dense *A,
                            permuted_dense *C);

/* Fill values of C = B^T @ diag(d) @ A where B is CSC and A is PD */
void BTDA_csc_pd_fill_values(const CSC_matrix *B, const double *d,
                             const permuted_dense *A, permuted_dense *C);

/* Allocate a new PD C with the same sparsity as A */
matrix *copy_sparsity_pd_alloc(const permuted_dense *A);

/* Allocate C = A^T (transpose). */
matrix *transpose_pd_alloc(const permuted_dense *A);

/* Fill values of C = A^T. */
void transpose_pd_fill_values(const permuted_dense *A, permuted_dense *C);

/* Lazily allocate B's transpose_cache (a PD with swapped (m,n), (m0,n0),
   (row_perm,col_perm)) and return it. Idempotent: on subsequent calls
   the existing cache is returned. The cache's X buffer is allocated but
   uninitialized — callers must refresh it via transpose_pd_fill_values(B,
   B->transpose_cache) before reading X. Const-casts B internally; mirrors
   sparse_matrix_ensure_csc_cache's convention. */
permuted_dense *permuted_dense_ensure_transpose_cache(const permuted_dense *B);

/* Ensure B's transpose cache and refresh its values iff B's values_version
   moved since the last refresh, then return it. Same contract as the ensure:
   B owns its value buffer (not an spd block, which has no version of its own)
   and its perms are immutable between alloc and fill. */
permuted_dense *permuted_dense_transpose_fresh(const permuted_dense *B);

#endif /* PERMUTED_DENSE_LINALG_H */
