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
#include "internal/matmul_kron.h"
#include "sparsediff_linalg/matmul_kron.h"

#include "internal/csr_kernels.h"
#include "internal/permuted_dense.h"
#include "internal/sparse_matrix.h"
#include "internal/sparse_matrix_cache.h"
#include "internal/stacked_pd.h"
#include "internal/stacked_pd_kron_linalg.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/matmul.h"
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <assert.h>

void BA_kron_vec_matrices(const matrix *A, const double *x, double *y, int p)
{
    assert(A->kind != MATRIX_STACKED_PD);
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        BA_kron_vec_pd((const permuted_dense *) A, x, y, p);
        return;
    }
    /* A is sparse_matrix */
    block_left_multiply_vec(((const sparse_matrix *) A)->csr, x, y, p);
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

/* C = kron(I_p, A) @ J through the CSC kernel block_left_multiply_*. J is
   converted (through its CSR view) into bound_csc, using csc_iwork (C->n = J->n
   ints) as workspace. The kernel's CSC output is C's CSC mirror (csc_cache),
   converted to C's CSR with the C->m-long bound_iwork. */
matrix *BA_sparse_kron_matrices_alloc(const matrix *A, int p, matrix *J)
{
    assert(A->kind == MATRIX_SPARSE && J->m == p * A->n);
    int *csc_iwork = (int *) sp_malloc((size_t) J->n * sizeof(int));
    CSC_matrix *J_csc = csr_to_csc_alloc(J->to_csr(J), csc_iwork);
    CSC_matrix *C_csc = block_left_multiply_fill_sparsity(
        ((const sparse_matrix *) A)->csr, J_csc, p);
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
    assert(sm->bound_csc != NULL && sm->bound_iwork != NULL);

    csr_to_csc_fill_values(J->to_csr(J), sm->bound_csc, sm->csc_iwork);
    block_left_multiply_fill_values(((const sparse_matrix *) A)->csr, sm->bound_csc,
                                    sm->csc_cache);
    csc_to_csr_fill_values(sm->csc_cache, sm->csr, sm->bound_iwork);
    matrix_values_changed(C);
    sm->csc_seen = C->values_version;
}

matrix *BA_kron_matrices_alloc(const matrix *A, int p, matrix *J)
{
    if (p == 1)
    {
        return BA_matrices_alloc(A, J);
    }
    if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        return BA_dense_kron_matrices_alloc(A, p, J);
    }
    /* A is sparse_matrix */
    return BA_sparse_kron_matrices_alloc(A, p, J);
}

void BA_kron_matrices_fill_values(const matrix *A, int p, matrix *J, matrix *C)
{
    if (p == 1)
    {
        BA_matrices_fill_values(A, J, C);
    }
    else if (A->kind == MATRIX_PERMUTED_DENSE)
    {
        BA_dense_kron_matrices_fill_values(A, p, J, C);
    }
    else
    {
        /* A is sparse_matrix */
        BA_sparse_kron_matrices_fill_values(A, p, J, C);
    }
}
