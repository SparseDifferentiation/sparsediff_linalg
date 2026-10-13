#ifndef TEST_CACHE_COHERENCE_H
#define TEST_CACHE_COHERENCE_H

/* Cache coherence by construction: a fill refreshes every operand cache it reads
   and bumps the values_version of the matrix it writes. These tests never call a
   refresh of their own; they mutate operand values, notify the owner, refill, and
   compare against a dense oracle computed from the operands' CSR views. */

#include "internal/matmul_kron.h"
#include "internal/permuted_dense.h"
#include "internal/sparse_matrix.h"
#include "internal/stacked_pd.h"
#include "minunit.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/matmul.h"
#include "sparsediff_linalg/matmul_kron.h"
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/matrix_sum.h"
#include "test_helpers.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- fixtures: three kinds of a 6 x 4 operand and a 6 x 5 operand ---- */

enum cc_kind
{
    CC_SPARSE,
    CC_PD,
    CC_SPD
};

/* 6 x 4 with nonzeros in rows {1, 2, 4, 5} x cols {0, 2, 3} (a single dense
   block for the pd kind, two row-disjoint blocks for the spd kind, and the
   same entries as a hand-built CSR for the sparse kind). */
static matrix *cc_make_A(enum cc_kind kind)
{
    int rp[4] = {1, 2, 4, 5};
    int cp[3] = {0, 2, 3};
    double X[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    if (kind == CC_PD) return new_permuted_dense(6, 4, 4, 3, rp, cp, X);
    if (kind == CC_SPD)
    {
        permuted_dense *blocks[2];
        blocks[0] = (permuted_dense *) new_permuted_dense(6, 4, 2, 3, rp, cp, X);
        blocks[1] =
            (permuted_dense *) new_permuted_dense(6, 4, 2, 3, rp + 2, cp, X + 6);
        return new_stacked_pd(6, 4, 2, blocks, NULL, NULL);
    }
    CSR_matrix *csr = new_CSR_matrix(6, 4, 12);
    int nnz = 0;
    for (int i = 0; i < 6; i++)
    {
        csr->p[i] = nnz;
        for (int ii = 0; ii < 4; ii++)
        {
            if (rp[ii] != i) continue;
            for (int jj = 0; jj < 3; jj++)
            {
                csr->i[nnz] = cp[jj];
                csr->x[nnz] = X[ii * 3 + jj];
                nnz++;
            }
        }
    }
    csr->p[6] = nnz;
    csr->nnz = nnz;
    return new_sparse_matrix(csr);
}

/* 6 x 5 with nonzeros in rows {0, 2, 3} x cols {1, 4}. */
static matrix *cc_make_B(enum cc_kind kind)
{
    int rp[3] = {0, 2, 3};
    int cp[2] = {1, 4};
    double X[6] = {-1, 2, -3, 4, -5, 6};
    if (kind == CC_PD) return new_permuted_dense(6, 5, 3, 2, rp, cp, X);
    if (kind == CC_SPD)
    {
        permuted_dense *blocks[2];
        blocks[0] = (permuted_dense *) new_permuted_dense(6, 5, 1, 2, rp, cp, X);
        blocks[1] =
            (permuted_dense *) new_permuted_dense(6, 5, 2, 2, rp + 1, cp, X + 2);
        return new_stacked_pd(6, 5, 2, blocks, NULL, NULL);
    }
    CSR_matrix *csr = new_CSR_matrix(6, 5, 6);
    int nnz = 0;
    for (int i = 0; i < 6; i++)
    {
        csr->p[i] = nnz;
        for (int ii = 0; ii < 3; ii++)
        {
            if (rp[ii] != i) continue;
            for (int jj = 0; jj < 2; jj++)
            {
                csr->i[nnz] = cp[jj];
                csr->x[nnz] = X[ii * 2 + jj];
                nnz++;
            }
        }
    }
    csr->p[6] = nnz;
    csr->nnz = nnz;
    return new_sparse_matrix(csr);
}

/* ---- dense oracle ---- */

/* Row-major dense copy of M through its CSR view. */
static void cc_to_dense(matrix *M, double *D)
{
    CSR_matrix *csr = M->to_csr(M);
    memset(D, 0, (size_t) csr->m * csr->n * sizeof(double));
    for (int i = 0; i < csr->m; i++)
        for (int jj = csr->p[i]; jj < csr->p[i + 1]; jj++)
            D[i * csr->n + csr->i[jj]] = csr->x[jj];
}

/* Dense C (nB x nA) = B^T diag(d) A with B (m x nB), A (m x nA); d NULL = ones. */
static void cc_dense_BTDA(const double *B, const double *d, const double *A, int m,
                          int nB, int nA, double *C)
{
    memset(C, 0, (size_t) nB * nA * sizeof(double));
    for (int k = 0; k < m; k++)
    {
        double dk = d ? d[k] : 1.0;
        for (int i = 0; i < nB; i++)
            for (int j = 0; j < nA; j++)
                C[i * nA + j] += B[k * nB + i] * dk * A[k * nA + j];
    }
}

static int cc_dense_eq(const double *X, const double *Y, int len)
{
    for (int k = 0; k < len; k++)
        if (fabs(X[k] - Y[k]) > 1e-12 * (1.0 + fabs(Y[k]))) return 0;
    return 1;
}

/* Scale every value of M by s and notify the owner, as a consumer that writes
   x directly must. */
static void cc_scale(matrix *M, double s)
{
    for (int k = 0; k < M->nnz; k++) M->x[k] *= s;
    matrix_values_changed(M);
}

/* ---- 1. fills refresh the sparse operand caches they read ---- */

/* One (B kind, A kind) pair: C = B^T diag(d) A through the dispatchers; fill
   once, mutate both operands, fill again with no refresh of any kind, and
   compare to the dense oracle both times. d NULL exercises BTA, else BTDA. */
static const char *cc_check_BTDA_pair(enum cc_kind kB, enum cc_kind kA,
                                      const double *d)
{
    matrix *A = cc_make_A(kA);
    matrix *B = cc_make_B(kB);
    double Ad[24], Bd[30], Cd[20], Cref[20];
    matrix *C = BTA_matrices_alloc(A, B);
    for (int pass = 0; pass < 2; pass++)
    {
        if (pass == 1)
        {
            cc_scale(A, 2.0);
            cc_scale(B, -0.5);
        }
        if (d)
            BTDA_matrices_fill_values(A, d, B, C);
        else
            BTA_matrices_fill_values(A, B, C);
        cc_to_dense(A, Ad);
        cc_to_dense(B, Bd);
        cc_dense_BTDA(Bd, d, Ad, 6, 5, 4, Cref);
        cc_to_dense(C, Cd);
        int ok = cc_dense_eq(Cd, Cref, 20);
        if (!ok)
        {
            free_matrix(C);
            free_matrix(B);
            free_matrix(A);
            return pass == 0 ? "first fill" : "refill after operand mutation";
        }
    }
    free_matrix(C);
    free_matrix(B);
    free_matrix(A);
    return 0;
}

/* Every pair with at least one sparse operand (the others have no cache). */
static const char *cc_check_all_sparse_pairs(const double *d)
{
    const enum cc_kind kinds[3] = {CC_SPARSE, CC_PD, CC_SPD};
    for (int b = 0; b < 3; b++)
    {
        for (int a = 0; a < 3; a++)
        {
            if (kinds[b] != CC_SPARSE && kinds[a] != CC_SPARSE) continue;
            const char *err = cc_check_BTDA_pair(kinds[b], kinds[a], d);
            if (err) return err;
        }
    }
    return 0;
}

const char *test_fills_refresh_sparse_operands_BTA(void)
{
    const char *err = cc_check_all_sparse_pairs(NULL);
    mu_assert(err ? err : "", err == 0);
    return 0;
}

const char *test_fills_refresh_sparse_operands_BTDA(void)
{
    double d[6] = {2.0, -1.5, 0.5, 1.25, 3.0, -2.0};
    const char *err = cc_check_all_sparse_pairs(d);
    mu_assert(err ? err : "", err == 0);
    return 0;
}

/* C = B A with B pd (5 x 6) and A sparse (6 x 4). */
const char *test_fill_refreshes_sparse_operand_BA_pd(void)
{
    int rp[3] = {0, 2, 4};
    int cp[4] = {1, 2, 3, 5};
    double X[12] = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12};
    matrix *B = new_permuted_dense(5, 6, 3, 4, rp, cp, X);
    matrix *A = cc_make_A(CC_SPARSE);
    matrix *C = BA_matrices_alloc(B, A);
    double Bd[30], Ad[24], Cd[20], Cref[20];
    for (int pass = 0; pass < 2; pass++)
    {
        if (pass == 1) cc_scale(A, 3.0);
        BA_matrices_fill_values(B, A, C);
        cc_to_dense(B, Bd);
        cc_to_dense(A, Ad);
        /* C = B A: C[i, j] = sum_k B[i, k] A[k, j] */
        memset(Cref, 0, sizeof Cref);
        for (int i = 0; i < 5; i++)
            for (int k = 0; k < 6; k++)
                for (int j = 0; j < 4; j++)
                    Cref[i * 4 + j] += Bd[i * 6 + k] * Ad[k * 4 + j];
        cc_to_dense(C, Cd);
        mu_assert(pass == 0 ? "BA_pd first fill" : "BA_pd refill after mutation",
                  cc_dense_eq(Cd, Cref, 20));
    }
    free_matrix(C);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* C = kron(I_2, A) J with A a full 2 x 3 pd and J sparse (6 x 4). */
const char *test_fill_refreshes_sparse_operand_kron(void)
{
    double Xa[6] = {1, 2, 3, 4, 5, 6};
    matrix *A = new_permuted_dense_full(2, 3, Xa);
    matrix *J = cc_make_A(CC_SPARSE);
    matrix *C = BA_dense_kron_matrices_alloc(A, 2, J);
    double Jd[24], Cd[16], Cref[16];
    for (int pass = 0; pass < 2; pass++)
    {
        if (pass == 1) cc_scale(J, -2.0);
        BA_dense_kron_matrices_fill_values(A, 2, J, C);
        cc_to_dense(J, Jd);
        /* kron(I_2, A) is 4 x 6 block-diagonal with A (2 x 3) twice. */
        memset(Cref, 0, sizeof Cref);
        for (int blk = 0; blk < 2; blk++)
            for (int i = 0; i < 2; i++)
                for (int k = 0; k < 3; k++)
                    for (int j = 0; j < 4; j++)
                        Cref[(blk * 2 + i) * 4 + j] +=
                            Xa[i * 3 + k] * Jd[(blk * 3 + k) * 4 + j];
        cc_to_dense(C, Cd);
        mu_assert(pass == 0 ? "kron first fill" : "kron refill after mutation",
                  cc_dense_eq(Cd, Cref, 16));
    }
    free_matrix(C);
    free_matrix(J);
    free_matrix(A);
    return 0;
}

/* Sparse path: C = A^T diag(d) A reads A's CSC mirror. */
const char *test_fill_refreshes_sparse_operand_ATDA(void)
{
    matrix *A = cc_make_A(CC_SPARSE);
    matrix *C = ATA_matrices_alloc(A);
    double d[6] = {1.0, 2.0, -1.0, 0.5, 4.0, -3.0};
    double Ad[24], Cd[16], Cref[16];
    for (int pass = 0; pass < 2; pass++)
    {
        if (pass == 1) cc_scale(A, 1.5);
        ATDA_matrices_fill_values(A, d, C);
        cc_to_dense(A, Ad);
        cc_dense_BTDA(Ad, d, Ad, 6, 4, 4, Cref);
        cc_to_dense(C, Cd);
        mu_assert(pass == 0 ? "ATDA first fill" : "ATDA refill after mutation",
                  cc_dense_eq(Cd, Cref, 16));
    }
    free_matrix(C);
    free_matrix(A);
    return 0;
}

/* The dedicated ATA dispatchers agree with the general BTA ones on B == A, for
   every kind: same output kind and nnz, same (symmetric) values. */
static const char *cc_check_ATA_matches_BTA(enum cc_kind kind)
{
    matrix *A = cc_make_A(kind);
    double d[6] = {1.0, 2.0, -1.0, 0.5, 4.0, -3.0};
    double C1d[16], C2d[16];
    const char *err = 0;

    matrix *C1 = ATA_matrices_alloc(A);
    matrix *C2 = BTA_matrices_alloc(A, A);
    if (C1->kind != C2->kind || C1->kind != A->kind) err = "ATA output kind";
    if (!err && C1->nnz != C2->nnz) err = "ATA nnz differs from BTA";
    if (!err)
    {
        ATDA_matrices_fill_values(A, d, C1);
        BTDA_matrices_fill_values(A, d, A, C2);
        cc_to_dense(C1, C1d);
        cc_to_dense(C2, C2d);
        if (!cc_dense_eq(C1d, C2d, 16)) err = "ATDA values differ from BTDA";
        double C1t[16];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) C1t[j * 4 + i] = C1d[i * 4 + j];
        if (!err && !cc_dense_eq(C1d, C1t, 16)) err = "ATDA not symmetric";
    }
    free_matrix(C2);
    free_matrix(C1);
    free_matrix(A);
    return err;
}

const char *test_ATA_matrices_matches_BTA(void)
{
    const enum cc_kind kinds[3] = {CC_SPARSE, CC_PD, CC_SPD};
    for (int k = 0; k < 3; k++)
    {
        const char *err = cc_check_ATA_matches_BTA(kinds[k]);
        mu_assert(err ? err : "", err == 0);
    }
    return 0;
}

/* ---- 2. every public fill bumps its output by exactly one ---- */

static int cc_bumped_once(const matrix *C, uint64_t before)
{
    return C->values_version == before + 1;
}

static const char *cc_check_vtable_bumps(enum cc_kind kind)
{
    matrix *A = cc_make_A(kind);
    double d[6] = {1, 2, 3, 4, 5, 6};
    int map[3] = {5, 1, 1};
    int group[6] = {0, 1, 0, 1, 0, 1};
    uint64_t v;
    const char *err = 0;

    matrix *DA = copy_sparsity_matrices_alloc(A);
    v = DA->values_version;
    DA_matrices_fill_values(d, A, DA);
    if (!cc_bumped_once(DA, v)) err = "DA_matrices_fill_values";
    free_matrix(DA);

    matrix *ATA = ATA_matrices_alloc(A);
    v = ATA->values_version;
    ATDA_matrices_fill_values(A, d, ATA);
    if (!err && !cc_bumped_once(ATA, v)) err = "ATDA_matrices_fill_values";
    free_matrix(ATA);

    matrix *AT = transpose_matrices_alloc(A);
    v = AT->values_version;
    transpose_matrices_fill_values(A, AT);
    if (!err && !cc_bumped_once(AT, v)) err = "transpose_matrices_fill_values";
    free_matrix(AT);

    matrix *G = A->row_gather_alloc(A, map, 3);
    v = G->values_version;
    A->row_gather_fill_values(A, G);
    if (!err && !cc_bumped_once(G, v)) err = "row_gather_fill_values";
    free_matrix(G);

    matrix *R = A->row_reduce_alloc(A, group, 2);
    v = R->values_version;
    A->row_reduce_fill_values(A, R);
    if (!err && !cc_bumped_once(R, v)) err = "row_reduce_fill_values";
    free_matrix(R);

    free_matrix(A);
    return err;
}

const char *test_public_fills_bump_output(void)
{
    const char *err;
    err = cc_check_vtable_bumps(CC_SPARSE);
    mu_assert(err ? err : "", err == 0);
    err = cc_check_vtable_bumps(CC_PD);
    mu_assert(err ? err : "", err == 0);
    err = cc_check_vtable_bumps(CC_SPD);
    mu_assert(err ? err : "", err == 0);

    /* Dispatchers. */
    double d[6] = {1, 2, 3, 4, 5, 6};
    matrix *A = cc_make_A(CC_SPARSE);
    matrix *Bp = cc_make_B(CC_PD);
    matrix *C = BTA_matrices_alloc(A, Bp);
    uint64_t v = C->values_version;
    BTA_matrices_fill_values(A, Bp, C);
    mu_assert("BTA_matrices_fill_values bumps", cc_bumped_once(C, v));
    v = C->values_version;
    BTDA_matrices_fill_values(A, d, Bp, C);
    mu_assert("BTDA_matrices_fill_values bumps", cc_bumped_once(C, v));
    free_matrix(C);

    int rp[3] = {0, 2, 4};
    int cp[4] = {1, 2, 3, 5};
    double X[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    matrix *Bl = new_permuted_dense(5, 6, 3, 4, rp, cp, X);
    C = BA_matrices_alloc(Bl, A);
    v = C->values_version;
    BA_matrices_fill_values(Bl, A, C);
    mu_assert("BA_pd_matrices_fill_values bumps", cc_bumped_once(C, v));
    free_matrix(C);
    free_matrix(Bl);

    double Xa[6] = {1, 2, 3, 4, 5, 6};
    matrix *Af = new_permuted_dense_full(2, 3, Xa);
    C = BA_dense_kron_matrices_alloc(Af, 2, A);
    v = C->values_version;
    BA_dense_kron_matrices_fill_values(Af, 2, A, C);
    mu_assert("BA_dense_kron_matrices_fill_values bumps", cc_bumped_once(C, v));
    free_matrix(C);
    free_matrix(Af);

    /* Sums (sparse outputs). */
    matrix *A2 = cc_make_A(CC_SPARSE);
    cc_scale(A2, 2.0);
    matrix *S = new_sparse_matrix_alloc(6, 4, A->nnz + A2->nnz);
    sum_matrices_alloc(A, A2, S);
    v = S->values_version;
    sum_matrices_fill_values(A, A2, S);
    mu_assert("sum_matrices_fill_values bumps", cc_bumped_once(S, v));
    v = S->values_version;
    sum_scaled_matrices_fill_values(A, A2, S, d, d);
    mu_assert("sum_scaled_matrices_fill_values bumps", cc_bumped_once(S, v));
    free_matrix(S);
    free_matrix(A2);

    free_matrix(Bp);
    free_matrix(A);
    return 0;
}

#endif /* TEST_CACHE_COHERENCE_H */
