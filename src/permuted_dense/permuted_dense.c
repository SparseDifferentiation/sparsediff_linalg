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
#include "internal/permuted_dense.h"

#include "internal/permuted_dense_linalg.h"
#include "internal/utils.h"
#include "sparsediff_linalg/cblas_wrapper.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void permuted_dense_free(matrix *self)
{
    permuted_dense *pd = (permuted_dense *) self;
    sp_free(pd->row_perm);
    sp_free(pd->col_perm);
    sp_free(pd->col_inv);
    sp_free(pd->row_inv);
    /* csr_cache->x aliases pd->X (set in permuted_dense_to_csr_alloc); NULL it
       so free_CSR_matrix doesn't double-free the shared buffer. */
    if (pd->csr_cache != NULL)
    {
        pd->csr_cache->x = NULL;
    }
    free_CSR_matrix(pd->csr_cache);
    if (pd->owns_X)
    {
        sp_free(pd->X);
    }
    sp_free(pd->kernel_dwork);
    sp_free(pd->kernel_iwork);
    sp_free(pd->bound_iwork);
    free_matrix((matrix *) pd->transpose_cache);
    sp_free(pd);
}

/* Vtable adapters — each delegates to the existing permuted_dense_* kernel. */
/* Forward decl; definition lower in the file. */
static CSR_matrix *permuted_dense_to_csr_alloc(permuted_dense *A);

/* Lazy CSR_matrix view: allocate structure on first call, then return the cache.
   The cache's x array aliases pd->X (see permuted_dense_to_csr_alloc), so
   values are always live without a per-call refresh. */
static CSR_matrix *permuted_dense_to_csr(matrix *self)
{
    permuted_dense *pd = (permuted_dense *) self;
    if (pd->csr_cache == NULL)
    {
        pd->csr_cache = permuted_dense_to_csr_alloc(pd);
    }
    return pd->csr_cache;
}

matrix *row_gather_pd_alloc(const permuted_dense *A, const int *map, int m_out)
{
    /* Output position i is dense iff map[i] hits a row in A->row_perm. The kept
       positions form C's row_perm (strictly increasing by construction, repeats
       in map included); src[k] is the dense row of A that C's row k copies. */
    int *new_row_perm = (int *) sp_malloc(m_out * sizeof(int));
    int *src = (int *) sp_malloc(m_out * sizeof(int));
    int new_m0 = 0;
    for (int i = 0; i < m_out; i++)
    {
        int ii = A->row_inv[map[i]];
        if (ii >= 0)
        {
            new_row_perm[new_m0] = i;
            src[new_m0] = ii;
            new_m0++;
        }
    }

    matrix *out = new_permuted_dense(m_out, A->base.n, new_m0, A->n0, new_row_perm,
                                     A->col_perm, NULL);
    if (new_m0 > 0)
    {
        permuted_dense *C = (permuted_dense *) out;
        C->bound_iwork = (int *) sp_malloc(new_m0 * sizeof(int));
        memcpy(C->bound_iwork, src, new_m0 * sizeof(int));
    }
    sp_free(new_row_perm);
    sp_free(src);
    return out;
}

void row_gather_pd_fill_values(const permuted_dense *A, permuted_dense *C)
{
    assert(A->base.n == C->base.n && (C->m0 == 0 || C->bound_iwork != NULL));
    int n0 = A->n0;
    for (int k = 0; k < C->m0; k++)
    {
        memcpy(C->X + k * n0, A->X + C->bound_iwork[k] * n0, n0 * sizeof(double));
    }
}

static matrix *permuted_dense_vtable_row_gather_alloc(const matrix *self,
                                                      const int *map, int m_out)
{
    return row_gather_pd_alloc((const permuted_dense *) self, map, m_out);
}

static void permuted_dense_vtable_row_gather_fill_values(const matrix *self,
                                                         matrix *out)
{
    row_gather_pd_fill_values((const permuted_dense *) self, (permuted_dense *) out);
    matrix_values_changed(out);
}

/* ===== PD as the constant left operand of y = kron(I_p, A) @ x. Restricted to
   full-block PDs (m0 == m, n0 == n, identity perms) — the only operator shape
   any caller needs today. */

void BA_kron_vec_pd(const permuted_dense *A, const double *x, double *y, int p)
{
    /* Full-block precondition: X is a single contiguous row-major m x n
       block (perms are identity). For a non-trivial PD, X only stores the
       values at the permuted positions; the layout below assumes a full
       m x n matrix, hence the assert. */
    int m = A->base.m;
    int n = A->base.n;
    assert(A->m0 == m && A->n0 == n);

    /* y = kron(I_p, A) @ x via a single dgemm.
       Input x is p blocks of length n (block-interleaved); output y is p
       blocks of length m. That's identical in memory to row-major matrices
       of shape (p, n) and (p, m) respectively, so we can compute
           y (p x m) = x (p x n) * A^T (n x m)
       in one shot. CblasRowMajor + CblasNoTrans on x + CblasTrans on A
       gives exactly that. */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p, m, n, 1.0, x, n, A->X, n,
                0.0, y, m);
}

matrix *row_reduce_pd_alloc(const permuted_dense *A, const int *group, int m_out)
{
    /* C's dense rows are the groups hit by A's dense rows, in increasing order.
       bound_iwork[ii] is the dense row of C that source dense row ii adds into. */
    int *group_to_out = (int *) sp_malloc(m_out * sizeof(int));
    for (int g = 0; g < m_out; g++)
    {
        group_to_out[g] = -1;
    }
    for (int ii = 0; ii < A->m0; ii++)
    {
        int g = group[A->row_perm[ii]];
        assert(g >= 0 && g < m_out);
        group_to_out[g] = 0; /* hit; compacted index assigned below */
    }
    int *C_row_perm = (int *) sp_malloc(m_out * sizeof(int));
    int new_m0 = 0;
    for (int g = 0; g < m_out; g++)
    {
        if (group_to_out[g] >= 0)
        {
            group_to_out[g] = new_m0;
            C_row_perm[new_m0++] = g;
        }
    }

    matrix *out = new_permuted_dense(m_out, A->base.n, new_m0, A->n0, C_row_perm,
                                     A->col_perm, NULL);
    if (A->m0 > 0)
    {
        permuted_dense *C = (permuted_dense *) out;
        C->bound_iwork = (int *) sp_malloc(A->m0 * sizeof(int));
        for (int ii = 0; ii < A->m0; ii++)
        {
            C->bound_iwork[ii] = group_to_out[group[A->row_perm[ii]]];
        }
    }
    sp_free(group_to_out);
    sp_free(C_row_perm);
    return out;
}

void row_reduce_pd_fill_values(const permuted_dense *A, permuted_dense *C)
{
    if (C->base.nnz == 0) return;
    assert(A->base.n == C->base.n && C->bound_iwork != NULL);
    int n0 = A->n0;
    memset(C->X, 0, (size_t) C->m0 * n0 * sizeof(double));
    for (int ii = 0; ii < A->m0; ii++)
    {
        const double *src = A->X + ii * n0;
        double *dst = C->X + C->bound_iwork[ii] * n0;
        for (int jj = 0; jj < n0; jj++)
        {
            dst[jj] += src[jj];
        }
    }
}

static matrix *permuted_dense_vtable_row_reduce_alloc(const matrix *self,
                                                      const int *group, int m_out)
{
    return row_reduce_pd_alloc((const permuted_dense *) self, group, m_out);
}

static void permuted_dense_vtable_row_reduce_fill_values(const matrix *self,
                                                         matrix *out)
{
    row_reduce_pd_fill_values((const permuted_dense *) self, (permuted_dense *) out);
    matrix_values_changed(out);
}

static void wire_vtable(permuted_dense *pd)
{
    pd->base.kind = MATRIX_PERMUTED_DENSE;
    pd->base.free_fn = permuted_dense_free;
    pd->base.to_csr = permuted_dense_to_csr;
    pd->base.row_gather_alloc = permuted_dense_vtable_row_gather_alloc;
    pd->base.row_gather_fill_values = permuted_dense_vtable_row_gather_fill_values;
    pd->base.row_reduce_alloc = permuted_dense_vtable_row_reduce_alloc;
    pd->base.row_reduce_fill_values = permuted_dense_vtable_row_reduce_fill_values;
}

matrix *new_permuted_dense(int m, int n, int m0, int n0, const int *row_perm,
                           const int *col_perm, const double *X_data)
{
    /* Validate sorted invariants. */
    for (int ii = 1; ii < m0; ii++)
    {
        assert(row_perm[ii] > row_perm[ii - 1]);
    }
    for (int jj = 1; jj < n0; jj++)
    {
        assert(col_perm[jj] > col_perm[jj - 1]);
    }
    if (m0 > 0)
    {
        assert(row_perm[0] >= 0 && row_perm[m0 - 1] < m);
    }
    if (n0 > 0)
    {
        assert(col_perm[0] >= 0 && col_perm[n0 - 1] < n);
    }

    permuted_dense *pd = (permuted_dense *) sp_calloc(1, sizeof(permuted_dense));
    pd->base.m = m;
    pd->base.n = n;
    if (m0 > 0 && n0 > INT_MAX / m0)
    {
        fprintf(stderr, "Error in new_permuted_dense: dense block m0 * n0 "
                        "exceeds INT_MAX.\n");
        exit(1);
    }
    pd->base.nnz = m0 * n0;
    wire_vtable(pd);

    pd->m0 = m0;
    pd->n0 = n0;

    int sz = m0 * n0;
    pd->row_perm = (int *) sp_malloc(m0 * sizeof(int));
    pd->col_perm = (int *) sp_malloc(n0 * sizeof(int));
    pd->X = (double *) sp_malloc(sz * sizeof(double));
    pd->base.x = pd->X;
    pd->owns_X = true;
    pd->col_inv = (int *) sp_malloc(n * sizeof(int));
    pd->row_inv = (int *) sp_malloc(m * sizeof(int));

    if (m0 > 0)
    {
        memcpy(pd->row_perm, row_perm, m0 * sizeof(int));
    }
    if (n0 > 0)
    {
        memcpy(pd->col_perm, col_perm, n0 * sizeof(int));
    }

    for (int j = 0; j < n; j++)
    {
        pd->col_inv[j] = -1;
    }
    for (int jj = 0; jj < n0; jj++)
    {
        pd->col_inv[col_perm[jj]] = jj;
    }

    for (int i = 0; i < m; i++)
    {
        pd->row_inv[i] = -1;
    }
    for (int ii = 0; ii < m0; ii++)
    {
        pd->row_inv[row_perm[ii]] = ii;
    }

    if (X_data != NULL && sz > 0)
    {
        memcpy(pd->X, X_data, sz * sizeof(double));
    }

    return &pd->base;
}

matrix *new_permuted_dense_full(int m, int n, const double *data)
{
    int *row_perm = (int *) sp_malloc(m * sizeof(int));
    int *col_perm = (int *) sp_malloc(n * sizeof(int));
    for (int i = 0; i < m; i++) row_perm[i] = i;
    for (int j = 0; j < n; j++) col_perm[j] = j;
    matrix *out = new_permuted_dense(m, n, m, n, row_perm, col_perm, data);
    sp_free(row_perm);
    sp_free(col_perm);
    return out;
}

static CSR_matrix *permuted_dense_to_csr_alloc(permuted_dense *A)
{
    int m0 = A->m0;
    int n0 = A->n0;
    int m = A->base.m;
    CSR_matrix *C = new_CSR_matrix(m, A->base.n, m0 * n0);

    /* Alias C->x to A->X: the dense block layout already matches what the
       CSR_matrix view's value array would hold, so values are always live with no
       memcpy needed. The PD owns the buffer; permuted_dense_free nulls
       C->x before free_CSR_matrix to avoid double-free. */
    sp_free(C->x);
    C->x = A->X;

    pd_blocks_csr_pattern(&A, 1, m, C->p, C->i);
    return C;
}

void pd_blocks_csr_pattern(permuted_dense *const *blocks, int n_blocks, int m,
                           int *Cp, int *Ci)
{
    /* row pointers via row lengths and then a cumulative sum from Cp[0] */
    memset(Cp + 1, 0, (size_t) m * sizeof(int));
    for (int k = 0; k < n_blocks; k++)
    {
        const permuted_dense *blk = blocks[k];
        for (int ii = 0; ii < blk->m0; ii++)
        {
            Cp[blk->row_perm[ii] + 1] = blk->n0;
        }
    }
    cumsum(Cp, m);

    /* column indices (each dense row contributes a copy of its col_perm) */
    for (int k = 0; k < n_blocks; k++)
    {
        const permuted_dense *blk = blocks[k];
        if (blk->n0 == 0) continue;
        for (int ii = 0; ii < blk->m0; ii++)
        {
            memcpy(Ci + Cp[blk->row_perm[ii]], blk->col_perm,
                   (size_t) blk->n0 * sizeof(int));
        }
    }
}

void pd_blocks_csr_fill_values(permuted_dense *const *blocks, int n_blocks,
                               const int *Cp, double *Cx)
{
    for (int k = 0; k < n_blocks; k++)
    {
        const permuted_dense *blk = blocks[k];
        if (blk->n0 == 0) continue;
        for (int ii = 0; ii < blk->m0; ii++)
        {
            memcpy(Cx + Cp[blk->row_perm[ii]], blk->X + (size_t) ii * blk->n0,
                   (size_t) blk->n0 * sizeof(double));
        }
    }
}

void permuted_dense_ensure_kernel_dwork(const permuted_dense *pd_const, size_t size)
{
    /* TODO: refactor this - maybe pass in kernel and size instead? Then this is
     * a general function, not necessarily tied to permuted_dense*/
    permuted_dense *pd = (permuted_dense *) pd_const;
    if (pd->kernel_dwork_size >= size) return;
    sp_free(pd->kernel_dwork);
    pd->kernel_dwork = (double *) sp_malloc(size * sizeof(double));
    pd->kernel_dwork_size = size;
}
