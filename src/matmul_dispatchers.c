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

#include "internal/permuted_dense.h"
#include "internal/permuted_dense_linalg.h"
#include "internal/sparse_matrix.h"
#include "internal/sparse_matrix_cache.h"
#include "internal/stacked_pd.h"
#include "internal/stacked_pd_kron_linalg.h"
#include "internal/stacked_pd_linalg.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
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

/* The output kind an alloc produced for (B, A): B PD -> PD, B spd -> spd,
   B sparse -> A's kind. Used by the fills' kind asserts. */
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

matrix *BA_pd_matrices_alloc(const matrix *B_m, matrix *A)
{
    assert(B_m->kind == MATRIX_PERMUTED_DENSE);
    const permuted_dense *B = (const permuted_dense *) B_m;
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

void BA_pd_matrices_fill_values(const matrix *B_m, const matrix *A, matrix *C_m)
{
    assert(B_m->kind == MATRIX_PERMUTED_DENSE);
    assert(C_m->kind == MATRIX_PERMUTED_DENSE);
    const permuted_dense *B = (const permuted_dense *) B_m;
    permuted_dense *C = (permuted_dense *) C_m;
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        BA_pd_pd_fill_values(B, (const permuted_dense *) A, C);
    }
    else if (A->kind == MATRIX_STACKED_PD)
    {
        BA_pd_spd_fill_values(B, (const stacked_pd *) A, C);
    }
    else
    {
        /* A is sparse */
        BA_pd_csc_fill_values(B->X, B->n0, B->col_inv, sparse_csc_fresh(A), C);
    }
    matrix_values_changed(&C->base);
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

    /* A is sparse */
    sparse_matrix *sm_A = (sparse_matrix *) A;
    sparse_matrix_ensure_csc_cache(sm_A);
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

    /* A is sparse */
    BTA_pd_csc_fill_values(B, sparse_csc_fresh(A), C);
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

matrix *BA_dense_kron_matrices_alloc(const matrix *A_m, int p, matrix *J)
{
#ifndef NDEBUG
    assert_dense_kron_A_is_full(A_m);
#endif
    const permuted_dense *A = (const permuted_dense *) A_m;

    if (J->kind == MATRIX_PERMUTED_DENSE)
    {
        return BA_dense_kron_pd_alloc(A, p, (const permuted_dense *) J);
    }
    if (J->kind == MATRIX_STACKED_PD)
    {
        return BA_dense_kron_spd_alloc(A, p, (const stacked_pd *) J);
    }
    /* J is sparse */
    sparse_matrix *sm_J = (sparse_matrix *) J;
    sparse_matrix_ensure_csc_cache(sm_J);
    return BA_dense_kron_csc_alloc(A, p, sm_J->csc_cache);
}

void BA_dense_kron_matrices_fill_values(const matrix *A_m, int p, const matrix *J,
                                        matrix *C_m)
{
#ifndef NDEBUG
    assert_dense_kron_A_is_full(A_m);
#endif
    assert(C_m->kind == MATRIX_STACKED_PD);
    const permuted_dense *A = (const permuted_dense *) A_m;
    stacked_pd *C = (stacked_pd *) C_m;
    if (J->kind == MATRIX_PERMUTED_DENSE)
    {
        BA_dense_kron_pd_fill_values(A, p, (const permuted_dense *) J, C);
    }
    else if (J->kind == MATRIX_STACKED_PD)
    {
        BA_dense_kron_spd_fill_values(A, p, (const stacked_pd *) J, C);
    }
    else
    {
        /* J is sparse */
        BA_dense_kron_csc_fill_values(A, p, sparse_csc_fresh(J), C);
    }
    matrix_values_changed(&C->base);
}
