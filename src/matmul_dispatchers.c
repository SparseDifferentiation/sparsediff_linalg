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
 */
#include "sparsediff_linalg/matmul.h"

#include "internal/CSC_matrix.h"
#include "internal/CSR_matrix.h"
#include "internal/csr_kernels.h"
#include "internal/permuted_dense.h"
#include "internal/permuted_dense_linalg.h"
#include "internal/sparse_matrix.h"
#include "internal/sparse_matrix_cache.h"
#include "internal/stacked_pd.h"
#include "internal/stacked_pd_kron_linalg.h"
#include "internal/stacked_pd_linalg.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <assert.h>

/* Forward declarations of the fixed-B dispatchers used internally by
   BTA_matrices_alloc / BTA_matrices_fill_values / BTDA_matrices_fill_values. */
static matrix *BTA_pd_matrices_alloc(const permuted_dense *B, matrix *A);
static void BTA_pd_matrices_fill_values(const permuted_dense *B, const matrix *A,
                                        permuted_dense *C);
static void BTDA_pd_matrices_fill_values(const permuted_dense *B, const double *d,
                                         const matrix *A, permuted_dense *C);
static matrix *BTA_spd_matrices_alloc(const stacked_pd *B, matrix *A);
static void BTA_spd_matrices_fill_values(const stacked_pd *B, const matrix *A,
                                         stacked_pd *C);
static void BTDA_spd_matrices_fill_values(const stacked_pd *B, const double *d,
                                          const matrix *A, stacked_pd *C);
static matrix *BTA_sparse_matrices_alloc(const sparse_matrix *B, matrix *A);
static void BTA_sparse_matrices_fill_values(const sparse_matrix *B, const matrix *A,
                                            matrix *C);
static void BTDA_sparse_matrices_fill_values(const sparse_matrix *B, const double *d,
                                             const matrix *A, matrix *C);

matrix *BTA_matrices_alloc(matrix *A, matrix *B)
{
    if (B->kind == MATRIX_PERMUTED_DENSE)
    {
        return BTA_pd_matrices_alloc((const permuted_dense *) B, A);
    }
    if (B->kind == MATRIX_STACKED_PD)
    {
        return BTA_spd_matrices_alloc((const stacked_pd *) B, A);
    }
    /* B is sparse_matrix */
    return BTA_sparse_matrices_alloc((const sparse_matrix *) B, A);
}

/* The output kind the BTA and BA allocs produce for (B, A): B PD -> PD,
   B spd -> spd, B sparse -> A's kind. Used by the fills' kind asserts. */
#ifndef NDEBUG
static matrix_kind BTA_output_kind(const matrix *A, const matrix *B)
{
    if (B->kind == MATRIX_PERMUTED_DENSE) return MATRIX_PERMUTED_DENSE;
    if (B->kind == MATRIX_STACKED_PD) return MATRIX_STACKED_PD;
    return A->kind;
}
#endif

void BTDA_matrices_fill_values(matrix *A, const double *d, matrix *B, matrix *C)
{
    assert(C->kind == BTA_output_kind(A, B));
    if (B->kind == MATRIX_PERMUTED_DENSE)
    {
        BTDA_pd_matrices_fill_values((const permuted_dense *) B, d, A,
                                     (permuted_dense *) C);
    }
    else if (B->kind == MATRIX_STACKED_PD)
    {
        BTDA_spd_matrices_fill_values((const stacked_pd *) B, d, A,
                                      (stacked_pd *) C);
    }
    else
    {
        /* B is sparse_matrix */
        BTDA_sparse_matrices_fill_values((const sparse_matrix *) B, d, A, C);
    }
    matrix_values_changed(C);
}

void BTA_matrices_fill_values(matrix *A, matrix *B, matrix *C)
{
    assert(C->kind == BTA_output_kind(A, B));
    if (B->kind == MATRIX_PERMUTED_DENSE)
    {
        BTA_pd_matrices_fill_values((const permuted_dense *) B, A,
                                    (permuted_dense *) C);
    }
    else if (B->kind == MATRIX_STACKED_PD)
    {
        BTA_spd_matrices_fill_values((const stacked_pd *) B, A, (stacked_pd *) C);
    }
    else
    {
        /* B is sparse_matrix */
        BTA_sparse_matrices_fill_values((const sparse_matrix *) B, A, C);
    }
    matrix_values_changed(C);
}

static matrix *BA_pd_matrices_alloc(const permuted_dense *B, matrix *A)
{
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        return BA_pd_pd_alloc(B, (const permuted_dense *) A);
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        return BA_pd_spd_alloc(B, (const stacked_pd *) A);
    }

    /* A is sparse */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm_A);
    return BA_pd_csc_alloc(B, sm_A->csc_cache);
}

static void BA_pd_matrices_fill_values(const permuted_dense *B, const matrix *A,
                                       permuted_dense *C)
{
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        BA_pd_pd_fill_values(B, (const permuted_dense *) A, C);
        return;
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        BA_pd_spd_fill_values(B, (const stacked_pd *) A, C);
        return;
    }

    /* A is sparse */
    BA_pd_csc_fill_values(B->X, B->n0, B->col_inv, sparse_csc_fresh(A), C);
}

/* CSR_matrix and CSC_matrix are the same struct, so the CSC reading of a CSR's
   arrays is its transpose: column j of A^T is row j of A, the stored indices
   are A^T's row indices, and x aliases A's live values. A sparse B feeds the
   CSC-left kernels (C = B_csc^T A) as this view of B^T with no transpose
   formed and nothing to refresh. */
static CSC_matrix csc_transpose_view(const CSR_matrix *A)
{
    CSC_matrix AT = {A->p, A->i, A->x, A->n, A->m, A->nnz};
    return AT;
}

static matrix *BA_sparse_matrices_alloc(const sparse_matrix *B, matrix *A)
{
    CSC_matrix BT = csc_transpose_view(B->csr);
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        return BTA_csc_pd_alloc(&BT, (const permuted_dense *) A);
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        return BTA_csc_spd_alloc(&BT, (const stacked_pd *) A);
    }

    /* A is sparse: B's CSR against A's mirror, straight into a CSR. */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm_A);
    return new_sparse_matrix(csr_csc_matmul_alloc(B->csr, sm_A->csc_cache));
}

static void BA_sparse_matrices_fill_values(const sparse_matrix *B, const matrix *A,
                                           matrix *C)
{
    CSC_matrix BT = csc_transpose_view(B->csr);
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        BTA_csc_pd_fill_values(&BT, (const permuted_dense *) A,
                               (permuted_dense *) C);
        return;
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        BTA_csc_spd_fill_values(&BT, (const stacked_pd *) A, (stacked_pd *) C);
        return;
    }

    /* A is sparse */
    csr_csc_matmul_fill_values(B->csr, sparse_csc_fresh(A),
                               ((sparse_matrix *) C)->csr);
}

/* C = B A for a permuted_dense or sparse B; a stacked_pd B asserts. */
matrix *BA_matrices_alloc(const matrix *B_m, matrix *A)
{
    assert(B_m->kind != MATRIX_STACKED_PD);
    if (B_m->kind == MATRIX_PERMUTED_DENSE)
    {
        return BA_pd_matrices_alloc((const permuted_dense *) B_m, A);
    }
    /* B is sparse_matrix */
    return BA_sparse_matrices_alloc((const sparse_matrix *) B_m, A);
}

void BA_matrices_fill_values(const matrix *B_m, const matrix *A, matrix *C_m)
{
    assert(B_m->kind != MATRIX_STACKED_PD);
    assert(C_m->kind == BTA_output_kind(A, B_m));
    if (B_m->kind == MATRIX_PERMUTED_DENSE)
    {
        BA_pd_matrices_fill_values((const permuted_dense *) B_m, A,
                                   (permuted_dense *) C_m);
    }
    else
    {
        /* B is sparse_matrix */
        BA_sparse_matrices_fill_values((const sparse_matrix *) B_m, A, C_m);
    }
    matrix_values_changed(C_m);
}

static matrix *BTA_pd_matrices_alloc(const permuted_dense *B, matrix *A)
{
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        return BTA_pd_pd_alloc(B, (const permuted_dense *) A);
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        return BTA_pd_spd_alloc(B, (const stacked_pd *) A);
    }

    /* A is sparse. The fill reads B^T from B's transpose cache; build it here
       so the fill never allocates. */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm_A);
    permuted_dense_ensure_transpose_cache(B);
    return BTA_pd_csc_alloc(B, sm_A->csc_cache);
}

static void BTDA_pd_matrices_fill_values(const permuted_dense *B, const double *d,
                                         const matrix *A, permuted_dense *C)
{
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        BTDA_pd_pd_fill_values(B, d, (const permuted_dense *) A, C);
        return;
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        BTDA_pd_spd_fill_values(B, d, (const stacked_pd *) A, C);
        return;
    }

    /* A is sparse */
    BTDA_pd_csc_fill_values(B, d, sparse_csc_fresh(A), C);
}

static void BTA_pd_matrices_fill_values(const permuted_dense *B, const matrix *A,
                                        permuted_dense *C)
{
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        BTA_pd_pd_fill_values(B, (const permuted_dense *) A, C);
        return;
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        BTA_pd_spd_fill_values(B, (const stacked_pd *) A, C);
        return;
    }

    /* A is sparse: read B^T from B's transpose cache, refreshed only when B's
       values moved, instead of re-transposing B's block on every fill. */
    const permuted_dense *BT = permuted_dense_transpose_fresh(B);
    BA_pd_csc_fill_values(BT->X, B->m0, B->row_inv, sparse_csc_fresh(A), C);
}

static matrix *BTA_spd_matrices_alloc(const stacked_pd *B, matrix *A)
{
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        return BTA_spd_pd_alloc(B, (const permuted_dense *) A);
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        return BTA_spd_spd_alloc(B, (const stacked_pd *) A);
    }

    /* A is sparse */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm_A);
    return BTA_spd_csc_alloc(B, sm_A->csc_cache);
}

static void BTDA_spd_matrices_fill_values(const stacked_pd *B, const double *d,
                                          const matrix *A, stacked_pd *C)
{
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        BTDA_spd_pd_fill_values(B, d, (const permuted_dense *) A, C);
        return;
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        BTDA_spd_spd_fill_values(B, d, (const stacked_pd *) A, C);
        return;
    }

    /* A is sparse */
    BTDA_spd_csc_fill_values(B, d, sparse_csc_fresh(A), C);
}

static void BTA_spd_matrices_fill_values(const stacked_pd *B, const matrix *A,
                                         stacked_pd *C)
{
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        BTA_spd_pd_fill_values(B, (const permuted_dense *) A, C);
        return;
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        BTA_spd_spd_fill_values(B, (const stacked_pd *) A, C);
        return;
    }

    /* A is sparse */
    BTA_spd_csc_fill_values(B, sparse_csc_fresh(A), C);
}

static matrix *BTA_sparse_matrices_alloc(const sparse_matrix *B, matrix *A)
{
    /* Ensure B's csc_cache structure exists. */
    sparse_matrix_ensure_csc_cache((sparse_matrix *) B);

    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        return BTA_csc_pd_alloc(B->csc_cache, (const permuted_dense *) A);
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        return BTA_csc_spd_alloc(B->csc_cache, (const stacked_pd *) A);
    }

    /* A is sparse */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm_A);
    /* BTA_alloc takes (A_csc, B_csc) — computes B^T A and returns CSR. */
    CSR_matrix *C_csr = BTA_alloc(sm_A->csc_cache, B->csc_cache);
    return new_sparse_matrix(C_csr);
}

static void BTDA_sparse_matrices_fill_values(const sparse_matrix *B, const double *d,
                                             const matrix *A, matrix *C)
{
    const CSC_matrix *B_csc = sparse_csc_fresh(&B->base);
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        BTDA_csc_pd_fill_values(B_csc, d, (const permuted_dense *) A,
                                (permuted_dense *) C);
        return;
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        BTDA_csc_spd_fill_values(B_csc, d, (const stacked_pd *) A, (stacked_pd *) C);
        return;
    }

    /* A is sparse */
    BTDA_fill_values(sparse_csc_fresh(A), B_csc, d, ((sparse_matrix *) C)->csr);
}

static void BTA_sparse_matrices_fill_values(const sparse_matrix *B, const matrix *A,
                                            matrix *C)
{
    const CSC_matrix *B_csc = sparse_csc_fresh(&B->base);
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        BTA_csc_pd_fill_values(B_csc, (const permuted_dense *) A,
                               (permuted_dense *) C);
        return;
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        BTA_csc_spd_fill_values(B_csc, (const stacked_pd *) A, (stacked_pd *) C);
        return;
    }

    /* A is sparse */
    BTDA_fill_values(sparse_csc_fresh(A), B_csc, NULL, ((sparse_matrix *) C)->csr);
}

/* Debug-only check that A is a permuted_dense of the "full" shape the kron
   helpers require: m0 == base.m, n0 == base.n, identity inner perms. */
#ifndef NDEBUG
static void assert_dense_kron_A_is_full(const matrix *A_m)
{
    assert(A_m->kind == MATRIX_PERMUTED_DENSE);
    const permuted_dense *A = (const permuted_dense *) A_m;
    assert(A->m0 == A->base.m);
    assert(A->n0 == A->base.n);
    for (int i = 0; i < A->m0; i++) assert(A->row_perm[i] == i);
    for (int j = 0; j < A->n0; j++) assert(A->col_perm[j] == j);
}
#endif

static matrix *dense_kron_alloc(const matrix *A_m, int p, matrix *J,
                                kron_layout layout)
{
#ifndef NDEBUG
    assert_dense_kron_A_is_full(A_m);
#endif
    const permuted_dense *A = (const permuted_dense *) A_m;

    if (J->kind == MATRIX_PERMUTED_DENSE)
    {
        return BA_dense_kron_pd_alloc(A, p, (const permuted_dense *) J, layout);
    }
    if (J->kind == MATRIX_STACKED_PD)
    {
        return BA_dense_kron_spd_alloc(A, p, (const stacked_pd *) J, layout);
    }
    /* J is sparse */
    sparse_matrix *sm_J = (sparse_matrix *) J;
    sparse_matrix_ensure_csc_cache(sm_J);
    return BA_dense_kron_csc_alloc(A, p, sm_J->csc_cache, layout);
}

static void dense_kron_fill_values(const matrix *A_m, int p, const matrix *J,
                                   matrix *C_m, kron_layout layout)
{
#ifndef NDEBUG
    assert_dense_kron_A_is_full(A_m);
#endif
    assert(C_m->kind == MATRIX_STACKED_PD);
    const permuted_dense *A = (const permuted_dense *) A_m;
    stacked_pd *C = (stacked_pd *) C_m;
    if (J->kind == MATRIX_PERMUTED_DENSE)
    {
        BA_dense_kron_pd_fill_values(A, p, (const permuted_dense *) J, C, layout);
    }
    else if (J->kind == MATRIX_STACKED_PD)
    {
        BA_dense_kron_spd_fill_values(A, p, (const stacked_pd *) J, C, layout);
    }
    else
    {
        /* J is sparse */
        BA_dense_kron_csc_fill_values(A, p, sparse_csc_fresh(J), C, layout);
    }
    matrix_values_changed(&C->base);
}

matrix *BA_dense_kron_matrices_alloc(const matrix *A, int p, matrix *J)
{
    return dense_kron_alloc(A, p, J, KRON_I_A);
}

void BA_dense_kron_matrices_fill_values(const matrix *A, int p, const matrix *J,
                                        matrix *C)
{
    dense_kron_fill_values(A, p, J, C, KRON_I_A);
}

matrix *BA_dense_kron_I_matrices_alloc(const matrix *A, int m, matrix *J)
{
    return dense_kron_alloc(A, m, J, KRON_A_I);
}

void BA_dense_kron_I_matrices_fill_values(const matrix *A, int m, const matrix *J,
                                          matrix *C)
{
    dense_kron_fill_values(A, m, J, C, KRON_A_I);
}

/* C = kron(I_p, A) @ J through the CSC kernel block_left_multiply_*. Its CSC
   output is C's CSC mirror (csc_cache), converted to C's CSR with the
   C->m-long bound_iwork. csc_iwork (C->n = J->n ints) doubles as the
   workspace for converting a non-sparse J into bound_csc. */
matrix *BA_sparse_kron_matrices_alloc(const matrix *A, int p, matrix *J)
{
    assert(A->kind == MATRIX_SPARSE && J->m == p * A->n);
    int *csc_iwork = (int *) sp_malloc((size_t) J->n * sizeof(int));
    CSC_matrix *J_csc = NULL;
    const CSC_matrix *J_read;
    if (J->kind == MATRIX_SPARSE)
    {
        /* the alloc only needs the structure of J's CSC mirror */
        sparse_matrix_ensure_csc_cache((sparse_matrix *) J);
        J_read = ((sparse_matrix *) J)->csc_cache;
    }
    else
    {
        J_csc = csr_to_csc_alloc(J->to_csr(J), csc_iwork);
        J_read = J_csc;
    }

    CSC_matrix *C_csc = block_left_multiply_fill_sparsity(
        ((const sparse_matrix *) A)->csr, J_read, p);
    int *csr_iwork = (int *) sp_malloc((size_t) C_csc->m * sizeof(int));
    matrix *C = new_sparse_matrix(csc_to_csr_alloc(C_csc, csr_iwork));

    sparse_matrix *sm = (sparse_matrix *) C;
    sm->csc_cache = C_csc;
    sm->csc_iwork = csc_iwork;
    sm->csc_seen = C->values_version - 1;
    sm->bound_iwork = csr_iwork;
    sm->bound_csc = J_csc;
    return C;
}

void BA_sparse_kron_matrices_fill_values(const matrix *A, int p, matrix *J,
                                         matrix *C)
{
    assert(A->kind == MATRIX_SPARSE && C->kind == MATRIX_SPARSE);
    assert(J->m == p * A->n);
    (void) p;
    sparse_matrix *sm = (sparse_matrix *) C;
    assert(sm->csc_cache != NULL && sm->bound_iwork != NULL);
    assert((sm->bound_csc == NULL) == (J->kind == MATRIX_SPARSE));

    const CSC_matrix *J_read;
    if (J->kind == MATRIX_SPARSE)
    {
        J_read = sparse_csc_fresh(J);
    }
    else
    {
        csr_to_csc_fill_values(J->to_csr(J), sm->bound_csc, sm->csc_iwork);
        J_read = sm->bound_csc;
    }

    block_left_multiply_fill_values(((const sparse_matrix *) A)->csr, J_read,
                                    sm->csc_cache);
    csc_to_csr_fill_values(sm->csc_cache, sm->csr, sm->bound_iwork);
    matrix_values_changed(C);
    sm->csc_seen = C->values_version;
}

matrix *copy_sparsity_matrices_alloc(const matrix *A)
{
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        return copy_sparsity_pd_alloc((const permuted_dense *) A);
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        return copy_sparsity_spd_alloc((const stacked_pd *) A);
    }
    /* A is sparse_matrix */
    return new_sparse_matrix(
        new_csr_copy_sparsity(((const sparse_matrix *) A)->csr));
}

void DA_matrices_fill_values(const double *d, const matrix *A, matrix *C)
{
    assert(C->kind == A->kind);
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        DA_pd_fill_values(d, (const permuted_dense *) A, (permuted_dense *) C);
    }
    else if (A->kind == MATRIX_STACKED_PD)
    {
        DA_spd_fill_values(d, (const stacked_pd *) A, (stacked_pd *) C);
    }
    else
    {
        /* A is sparse_matrix */
        DA_csr_fill_values(d, ((const sparse_matrix *) A)->csr,
                           ((sparse_matrix *) C)->csr);
    }
    matrix_values_changed(C);
}

matrix *ATA_matrices_alloc(matrix *A)
{
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        return ATA_pd_alloc((const permuted_dense *) A);
    }
    if (A->kind == MATRIX_STACKED_PD)
    {
        return ATA_spd_alloc((const stacked_pd *) A);
    }
    /* A is sparse_matrix; the alloc only needs the CSC mirror's structure */
    sparse_matrix *sm = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm);
    return new_sparse_matrix(ATA_alloc(sm->csc_cache));
}

void ATDA_matrices_fill_values(const matrix *A, const double *d, matrix *C)
{
    assert(C->kind == A->kind);
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        ATDA_pd_fill_values((const permuted_dense *) A, d, (permuted_dense *) C);
    }
    else if (A->kind == MATRIX_STACKED_PD)
    {
        ATDA_spd_fill_values((const stacked_pd *) A, d, (stacked_pd *) C);
    }
    else
    {
        /* A is sparse_matrix */
        ATDA_fill_values(sparse_csc_fresh(A), d, ((sparse_matrix *) C)->csr);
    }
    matrix_values_changed(C);
}
