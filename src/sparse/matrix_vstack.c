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
#include "sparsediff_linalg/matrix_vstack.h"
#include "internal/permuted_dense.h"
#include "internal/sparse_matrix.h"
#include "internal/stacked_pd.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Layout of the version state vstack_matrices_alloc binds to its output:
     [n | C_seen | seen_0 ... seen_{n-1}]
   where C_seen is C's values_version after the last fill (a mismatch means C->x
   was written directly, so every operand is recopied) and seen_k is the
   values_version of operand k when it was last copied. */
#define VSTACK_N 0
#define VSTACK_C_SEEN 1
#define VSTACK_SEEN 2

matrix *vstack_matrices_alloc(matrix *const *ops, int n)
{
    assert(n >= 1);
    int n_cols = ops[0]->n;
    size_t m_total = 0;
    size_t nnz_total = 0;
    for (int k = 0; k < n; k++)
    {
        assert(ops[k]->n == n_cols);
        m_total += (size_t) ops[k]->m;
        nnz_total += (size_t) ops[k]->nnz;
    }
    assert(m_total <= INT_MAX && nnz_total <= INT_MAX);

    matrix *C = new_sparse_matrix_alloc((int) m_total, n_cols, (int) nnz_total);
    CSR_matrix *csr = ((sparse_matrix *) C)->csr;
    int *Cp = csr->p;
    int *Ci = csr->i;
    Cp[0] = 0;

    /* each operand writes the row pointers Cp[row_off .. row_off + m_k] (the
       first already set by the previous operand) and its column indices at
       Ci[nnz_off ..] */
    int row_off = 0;
    int nnz_off = 0;
    for (int k = 0; k < n; k++)
    {
        const matrix *op = ops[k];
        if (op->kind == MATRIX_PERMUTED_DENSE)
        {
            const permuted_dense *pd = (const permuted_dense *) op;
            pd_blocks_csr_pattern(&pd, 1, op->m, nnz_off, Cp + row_off, Ci);
        }
        else if (op->kind == MATRIX_STACKED_PD)
        {
            const stacked_pd *spd = (const stacked_pd *) op;
            pd_blocks_csr_pattern((const permuted_dense *const *) spd->blocks,
                                  spd->n_blocks, op->m, nnz_off, Cp + row_off, Ci);
        }
        else
        {
            /* op is sparse_matrix */
            const CSR_matrix *A = ((const sparse_matrix *) op)->csr;
            for (int i = 1; i <= A->m; i++)
            {
                Cp[row_off + i] = nnz_off + A->p[i];
            }
            if (A->nnz > 0)
            {
                memcpy(Ci + nnz_off, A->i, (size_t) A->nnz * sizeof(int));
            }
        }
        row_off += op->m;
        nnz_off += op->nnz;
    }
    assert(row_off == C->m && Cp[row_off] == C->nnz);

    /* deliberately stale, so the first fill copies every operand */
    uint64_t *v = (uint64_t *) sp_malloc(((size_t) n + 2) * sizeof(uint64_t));
    v[VSTACK_N] = (uint64_t) n;
    v[VSTACK_C_SEEN] = C->values_version - 1;
    for (int k = 0; k < n; k++)
    {
        v[VSTACK_SEEN + k] = ops[k]->values_version - 1;
    }
    ((sparse_matrix *) C)->bound_vwork = v;
    return C;
}

void vstack_matrices_fill_values(matrix *const *ops, int n, matrix *C)
{
    assert(C->kind == MATRIX_SPARSE);
    uint64_t *v = ((sparse_matrix *) C)->bound_vwork;
    assert(v != NULL && v[VSTACK_N] == (uint64_t) n);
    const int *Cp = ((sparse_matrix *) C)->csr->p;

    bool copy_all = v[VSTACK_C_SEEN] != C->values_version;
    bool changed = false;
    int row_off = 0;
    int nnz_off = 0;
    for (int k = 0; k < n; k++)
    {
        const matrix *op = ops[k];
        if ((copy_all || v[VSTACK_SEEN + k] != op->values_version) && op->nnz > 0)
        {
            if (op->kind == MATRIX_STACKED_PD)
            {
                /* the spd's own buffer is block-major: scatter rows */
                const stacked_pd *spd = (const stacked_pd *) op;
                pd_blocks_csr_fill_values(
                    (const permuted_dense *const *) spd->blocks, spd->n_blocks,
                    Cp + row_off, C->x);
            }
            else
            {
                /* sparse and permuted_dense values are already in CSR order */
                memcpy(C->x + nnz_off, op->x, (size_t) op->nnz * sizeof(double));
            }
            changed = true;
        }
        v[VSTACK_SEEN + k] = op->values_version;
        row_off += op->m;
        nnz_off += op->nnz;
    }
    assert(row_off == C->m && nnz_off == C->nnz);

    if (changed)
    {
        matrix_values_changed(C);
    }
    v[VSTACK_C_SEEN] = C->values_version;
}
