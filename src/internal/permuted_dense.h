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
#ifndef INTERNAL_PERMUTED_DENSE_H
#define INTERNAL_PERMUTED_DENSE_H

/* Private: the permuted_dense kind's layout and its raw per-kind kernels.
   Consumers see only matrix.h and the constructors in
   matrix.h. */

#include "sparsediff_linalg/matrix.h"
#include <stdbool.h>
#include <stddef.h>

struct permuted_dense
{
    matrix base;
    int m0;
    int n0;
    int *row_perm;
    int *col_perm;
    int *col_inv; /* col_inv[col_perm[jj]] = jj, otherwise -1 */
    int *row_inv; /* row_inv[row_perm[ii]] = ii, otherwise -1 */

    /* Row-major block of size m0 x n0. Owned by this PD when owns_X == true,
       and otherwise X is a view into a buffer (eg., stacked_pd's shared values
       buffer) and must not be freed by this PD. */
    double *X;
    bool owns_X;

    /* Cached CSR_matrix view of the permuted dense, built on demand by to_csr.
       The structure (p/i arrays) is allocated on the first call to to_csr and
       reused thereafter; the values array aliases self->X so is always live with
       no separate fill needed. */
    CSR_matrix *csr_cache;

    /* Workspace used for BLAS operations. Allocated lazily by the first kernel
       that needs it, and resizes if another kernel needs more space than
       currently allocated (that's why we store dwork_size) */
    double *kernel_dwork;
    size_t kernel_dwork_size;

    /* Private buffers owned by the kernel that produced this PD. The
       producing kernel may use them for per-call scratch and/or small
       persistent state (e.g. cached config the fill_values needs to
       read). Allocated lazily. */
    int *kernel_iwork;
    size_t kernel_iwork_size;

    /* Int state bound by the alloc that produced this PD and read by the
       matching fill: row_gather_alloc stores, per dense row of this PD, the
       dense row of the source it copies (length m0); row_reduce_alloc stores,
       per dense row of the source, the dense row of this PD it adds into
       (length source m0). NULL otherwise; never touched by any other kernel
       (unlike kernel_iwork). */
    int *bound_iwork;

    /* Cached transpose of this PD as another permuted_dense, allocated lazily
       on first call to permuted_dense_ensure_transpose_cache. On the cache PD
       itself, transpose_seen records the source's base.values_version whose
       values the cache holds; consumers refill iff it is stale. */
    struct permuted_dense *transpose_cache;
    uint64_t transpose_seen;
};

/* Ensure A->kernel_dwork is sized at least 'size' doubles. Grows in
   place; contents are NOT preserved. */
void permuted_dense_ensure_kernel_dwork(const permuted_dense *A, size_t size);

/* CSR layout of a union of PD blocks with disjoint row_perms (a single PD, or
   the blocks of a stacked_pd) in an m-row matrix. Shared by the to_csr views
   and vstack_matrices, which writes into a row slice of a larger CSR. */

/* Write the row pointers Cp[1..m] and the column indices at their absolute
   positions Ci[Cp[row]..]. Cp[0] must hold the base offset on entry (0 for a
   standalone CSR). */
void pd_blocks_csr_pattern(permuted_dense *const *blocks, int n_blocks, int m,
                           int *Cp, int *Ci);

/* Scatter the blocks' rows of X into Cx at the absolute row starts Cp[row]
   written by pd_blocks_csr_pattern. */
void pd_blocks_csr_fill_values(permuted_dense *const *blocks, int n_blocks,
                               const int *Cp, double *Cx);

/* Raw per-kind kernels. They are also used blockwise by stacked_pd, whose
   blocks carry no values_version of their own, so unlike the vtable fills
   these fills do NOT bump C's values_version; the caller owning C does. */

/* y = kron(I_p, A) @ x for a full-block A (m0 == m, n0 == n): one dgemm. */
void BA_kron_vec_pd(const permuted_dense *A, const double *x, double *y, int p);

/* Allocate C = A[map, :], where A and C are permuted dense. C stores map
   internally, so the fill takes none. */
matrix *row_gather_pd_alloc(const permuted_dense *A, const int *map, int m_out);

/* Fill values of C = A[map, :], where A and C are permuted dense. */
void row_gather_pd_fill_values(const permuted_dense *A, permuted_dense *C);

/* Allocate C = row-reduce of A (C[j, :] = sum of rows i with group[i] == j,
   group[i] in [0, m_out)), where A and C are permuted dense. C stores the
   reduction map internally, so the fill takes none. */
matrix *row_reduce_pd_alloc(const permuted_dense *A, const int *group, int m_out);

/* Fill values of C = row-reduce of A, where A and C are permuted dense. */
void row_reduce_pd_fill_values(const permuted_dense *A, permuted_dense *C);

/* Allocate C = diag_vec(A), where A and C are permuted dense. */
matrix *diag_vec_pd_alloc(const permuted_dense *A);

/* Fill values of C = diag_vec(A). */
void diag_vec_pd_fill_values(const permuted_dense *A, permuted_dense *C);

#endif /* INTERNAL_PERMUTED_DENSE_H */
