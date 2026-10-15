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
#include "internal/stacked_pd_kron_linalg.h"

#include "internal/permuted_dense.h"
#include "internal/permuted_dense_linalg.h"
#include "internal/stacked_pd.h"
#include "internal/stacked_pd_linalg.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <stdlib.h>

// ====================================================================================
// All three BA_dense_kron_{pd,csr,spd}_* pairs share the same blockwise skeleton:
//
//   1. At alloc, build a persistent permuted_dense "state" (the scratch view of
//      kron(I_p, A), or its transpose BT for the spd kernel).
//   2. Loop k = 0..p-1: set_block(state, k, p); call the per-kernel alloc op
//      to produce Ck; collect non-empty Ck's.
//   3. Assemble the survivors into a stacked_pd C and stash `state` on
//      C->kernel_pd_scratch so fill can reuse it.
//   4. At fill, read state from C->kernel_pd_scratch (plus any kernel-specific
//      value refresh) and run the per-output-block loop.
//
// The state for pd/csr is the scratch (kron_scratch_init); for spd it is the
// transposed scratch BT (kron_BT_alloc). One set_block function works for
// both because under the "full A" contract, kron_scratch_set_block's col_inv
// writes stay within bounds even for BT's transposed shape.
// ====================================================================================

/* Build the scratch "block view" of kron(I_p, A): a permuted_dense with X
   aliasing A->X (owns_X = false). Initial perms encode block 0; callers
   mutate via kron_scratch_set_block. Free with free_matrix when done. */
static permuted_dense *kron_scratch_init(const permuted_dense *A, int p)
{
    int m = A->m0;
    int n = A->n0;

    /* Initial block-0 perms. */
    int *row_perm = (int *) sp_malloc(m * sizeof(int));
    int *col_perm = (int *) sp_malloc(n * sizeof(int));
    for (int i = 0; i < m; i++) row_perm[i] = i;
    for (int j = 0; j < n; j++) col_perm[j] = j;

    /* X_data=NULL leaves the buffer uninitialized; we drop it and alias A->X. */
    matrix *pd_m = new_permuted_dense(m * p, n * p, m, n, row_perm, col_perm, NULL);
    sp_free(row_perm);
    sp_free(col_perm);

    permuted_dense *pd = (permuted_dense *) pd_m;
    sp_free(pd->X);
    pd->X = A->X;
    pd->base.x = A->X;
    pd->owns_X = false;
    return pd;
}

/* Allocate the transposed scratch BT for the spd kernel: shape (A.n0*p, A.m0*p),
   inner block (A.n0, A.m0). Owns its X buffer (filled at fill time by
   transpose_pd_fill_values). Initial perms are placeholders — kron_scratch_set_block
   overwrites them before BT is read. */
static permuted_dense *kron_BT_alloc(const permuted_dense *A, int p)
{
    int n = A->n0;
    int m = A->m0;
    int *row_perm = (int *) sp_malloc(n * sizeof(int));
    int *col_perm = (int *) sp_malloc(m * sizeof(int));
    for (int j = 0; j < n; j++) row_perm[j] = j;
    for (int i = 0; i < m; i++) col_perm[i] = i;
    matrix *BT_m = new_permuted_dense(n * p, m * p, n, m, row_perm, col_perm, NULL);
    sp_free(row_perm);
    sp_free(col_perm);
    return (permuted_dense *) BT_m;
}

/* Clear the col_inv entries of the block the state currently encodes (block 0
   of kron(I_p, A) right after _init, whatever the layout). */
static void kron_clear_col_inv(permuted_dense *state)
{
    for (int c = 0; c < state->n0; c++)
    {
        state->col_inv[state->col_perm[c]] = -1;
    }
}

/* Mutate the state to represent kron block k of kron(I_p, A): refresh row_perm,
   col_perm and col_inv. O(m + n) work. row_inv is left stale — no per-block
   kernel reads it. Works on both scratch (pd/csr) and BT (spd) under the "full
   A" contract. */
static void kron_set_block_I_A(permuted_dense *state, int k, int p)
{
    (void) p;
    int m = state->m0;
    int n = state->n0;
    kron_clear_col_inv(state);
    for (int i = 0; i < m; i++)
    {
        state->row_perm[i] = k * m + i;
    }
    for (int j = 0; j < n; j++)
    {
        state->col_perm[j] = k * n + j;
        state->col_inv[k * n + j] = j;
    }
}

/* Same for block i of kron(A, I_p): rows i + p*r and columns i + p*c. */
static void kron_set_block_A_I(permuted_dense *state, int i, int p)
{
    int m = state->m0;
    int n = state->n0;
    kron_clear_col_inv(state);
    for (int r = 0; r < m; r++)
    {
        state->row_perm[r] = i + p * r;
    }
    for (int c = 0; c < n; c++)
    {
        state->col_perm[c] = i + p * c;
        state->col_inv[i + p * c] = c;
    }
}

typedef void (*kron_set_block_fn)(permuted_dense *state, int k, int p);

static kron_set_block_fn kron_set_block_for(kron_layout layout)
{
    return layout == KRON_I_A ? kron_set_block_I_A : kron_set_block_A_I;
}

/* Per-block callback signatures. */
typedef matrix *(*kron_block_alloc_fn)(permuted_dense *state, const void *ctx);
typedef void (*kron_block_fill_fn)(permuted_dense *state, const void *ctx,
                                   permuted_dense *Ck);

/* Generic alloc loop: iterate the p kron-blocks of A on `state`, call `op` per
   block, drop blocks whose Ck has nnz == 0, assemble the survivors into a
   stacked_pd of shape (Cm, Cn). Ownership of `state` transfers to
   C->kernel_pd_scratch on return. */
static matrix *kron_alloc_blockwise(permuted_dense *state, int p, kron_layout layout,
                                    int Cm, int Cn, kron_block_alloc_fn op,
                                    const void *ctx)
{
    if (p == 0)
    {
        free_matrix(&state->base);
        return new_stacked_pd(Cm, Cn, 0, NULL, NULL, NULL);
    }
    kron_set_block_fn set_block = kron_set_block_for(layout);

    permuted_dense **C_blocks =
        (permuted_dense **) sp_malloc(p * sizeof(permuted_dense *));
    int *C_src = (int *) sp_malloc(p * sizeof(int));

    int out_nb = 0;
    for (int k = 0; k < p; k++)
    {
        set_block(state, k, p);

        matrix *Ck = op(state, ctx);
        if (Ck->nnz == 0)
        {
            free_matrix(Ck);
        }
        else
        {
            C_blocks[out_nb] = (permuted_dense *) Ck;
            C_src[out_nb] = k;
            out_nb++;
        }
    }

    int *C_src_p = (int *) sp_malloc((out_nb + 1) * sizeof(int));
    for (int k = 0; k <= out_nb; k++) C_src_p[k] = k;

    matrix *C = new_stacked_pd(Cm, Cn, out_nb, C_blocks, C_src_p, C_src);
    sp_free(C_blocks);
    sp_free(C_src);
    sp_free(C_src_p);

    /* Hand state to C for reuse at fill time. stacked_pd_free releases it. */
    ((stacked_pd *) C)->kernel_pd_scratch = state;
    return C;
}

/* Generic fill loop: iterate C's output blocks, set state to the source kron
   block, call `op` per block. Caller has already done any kernel-specific
   value refresh on `state` (e.g. transpose_pd_fill_values for spd). */
static void kron_fill_blockwise(stacked_pd *C, int p, kron_layout layout,
                                kron_block_fill_fn op, const void *ctx)
{
    if (C->n_blocks == 0) return;
    kron_set_block_fn set_block = kron_set_block_for(layout);

    permuted_dense *state = C->kernel_pd_scratch;
    for (int k = 0; k < C->n_blocks; k++)
    {
        int q = C->src_block_idx[C->src_block_idx_p[k]];
        set_block(state, q, p);
        op(state, ctx, C->blocks[k]);
    }
}

// ----------------------------------------------------------------------------------
// Per-kernel ops + public functions.
// ----------------------------------------------------------------------------------

/* pd: state is the (untransposed) scratch. */
static matrix *kron_pd_alloc_op(permuted_dense *state, const void *ctx)
{
    return BA_pd_pd_alloc(state, (const permuted_dense *) ctx);
}
static void kron_pd_fill_op(permuted_dense *state, const void *ctx,
                            permuted_dense *Ck)
{
    BA_pd_pd_fill_values(state, (const permuted_dense *) ctx, Ck);
}

matrix *BA_dense_kron_pd_alloc(const permuted_dense *A, int p,
                               const permuted_dense *J, kron_layout layout)
{
    permuted_dense *scratch = kron_scratch_init(A, p);
    return kron_alloc_blockwise(scratch, p, layout, A->m0 * p, J->base.n,
                                kron_pd_alloc_op, J);
}

void BA_dense_kron_pd_fill_values(const permuted_dense *A, int p,
                                  const permuted_dense *J, stacked_pd *C,
                                  kron_layout layout)
{
    (void) A;
    kron_fill_blockwise(C, p, layout, kron_pd_fill_op, J);
}

/* csr: state is the (untransposed) scratch. BA_pd_csr_alloc reads the
   scratch's col_perm for the block being allocated and binds the matching
   rows of J to Ck, so the fill needs only the scratch's X (aliasing A->X). */
static matrix *kron_csr_alloc_op(permuted_dense *state, const void *ctx)
{
    return BA_pd_csr_alloc(state, (const CSR_matrix *) ctx);
}
static void kron_csr_fill_op(permuted_dense *state, const void *ctx,
                             permuted_dense *Ck)
{
    BA_pd_csr_fill_values(state->X, state->n0, (const CSR_matrix *) ctx, Ck);
}

matrix *BA_dense_kron_csr_alloc(const permuted_dense *A, int p, const CSR_matrix *J,
                                kron_layout layout)
{
    permuted_dense *scratch = kron_scratch_init(A, p);
    return kron_alloc_blockwise(scratch, p, layout, A->m0 * p, J->n,
                                kron_csr_alloc_op, J);
}

void BA_dense_kron_csr_fill_values(const permuted_dense *A, int p,
                                   const CSR_matrix *J, stacked_pd *C,
                                   kron_layout layout)
{
    (void) A;
    kron_fill_blockwise(C, p, layout, kron_csr_fill_op, J);
}

/* spd: state is the transposed scratch BT. BT.X needs to be refreshed each
   fill call (A's values may have changed); pd/csr don't, since their state's
   X aliases A->X. */
static matrix *kron_spd_alloc_op(permuted_dense *state, const void *ctx)
{
    return BTA_pd_spd_alloc(state, (const stacked_pd *) ctx);
}
static void kron_spd_fill_op(permuted_dense *state, const void *ctx,
                             permuted_dense *Ck)
{
    BTA_pd_spd_fill_values(state, (const stacked_pd *) ctx, Ck);
}

matrix *BA_dense_kron_spd_alloc(const permuted_dense *A, int p, const stacked_pd *J,
                                kron_layout layout)
{
    permuted_dense *BT = kron_BT_alloc(A, p);
    return kron_alloc_blockwise(BT, p, layout, A->m0 * p, J->base.n,
                                kron_spd_alloc_op, J);
}

void BA_dense_kron_spd_fill_values(const permuted_dense *A, int p,
                                   const stacked_pd *J, stacked_pd *C,
                                   kron_layout layout)
{
    if (C->n_blocks == 0) return;

    /* spd-only prelude: BT.X = transpose(A.X). A's values may have changed
       between alloc and fill, or between fills (parameter case). */
    transpose_pd_fill_values(A, C->kernel_pd_scratch);
    kron_fill_blockwise(C, p, layout, kron_spd_fill_op, J);
}
