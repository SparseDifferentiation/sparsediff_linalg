#ifndef TEST_MATMUL_DISPATCHERS_H
#define TEST_MATMUL_DISPATCHERS_H

#include "internal/permuted_dense.h"
#include "internal/permuted_dense_linalg.h"
#include "internal/sparse_matrix.h"
#include "internal/sparse_matrix_cache.h"
#include "internal/stacked_pd.h"
#include "internal/stacked_pd_linalg.h"
#include "internal/utils.h"
#include "minunit.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/matmul.h"
#include "sparsediff_linalg/matrix_sum.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include "test_helpers.h"
#include <stdlib.h>
#include <string.h>

/* Wrapper dispatch sanity: (PD, PD). Compare against direct
   BTDA_pd_pd_fill_values. */
const char *test_BTDA_matrices_pd_pd(void)
{
    int row_perm[2] = {0, 1};
    int col_perm_A[2] = {0, 2};
    int col_perm_B[2] = {1, 3};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    double XB[4] = {5.0, 6.0, 7.0, 8.0};
    double d[2] = {2.0, -1.5};

    matrix *A_m = new_permuted_dense(2, 4, 2, 2, row_perm, col_perm_A, XA);
    matrix *B_m = new_permuted_dense(2, 4, 2, 2, row_perm, col_perm_B, XB);

    /* Wrapper path. */
    matrix *C_m = BTA_matrices_alloc(A_m, B_m);
    BTDA_matrices_fill_values(A_m, d, B_m, C_m);

    /* Direct primitive path on independent operands. */
    matrix *A2 = new_permuted_dense(2, 4, 2, 2, row_perm, col_perm_A, XA);
    matrix *B2 = new_permuted_dense(2, 4, 2, 2, row_perm, col_perm_B, XB);
    matrix *C2 = BTA_pd_pd_alloc((permuted_dense *) B2, (permuted_dense *) A2);
    BTDA_pd_pd_fill_values((permuted_dense *) B2, d, (permuted_dense *) A2,
                           (permuted_dense *) C2);

    mu_assert("values", cmp_double_array(C_m->x, C2->x, C_m->nnz));

    free_matrix(C_m);
    free_matrix(B_m);
    free_matrix(A_m);
    free_matrix(C2);
    free_matrix(B2);
    free_matrix(A2);
    return 0;
}

/* Wrapper dispatch sanity: (PD, CSR_matrix). Compare against direct
   BTDA_csc_pd_fill_values. */
const char *test_BTDA_matrices_pd_csr(void)
{
    /* A: 4x5 PD, row_perm = [1, 3], col_perm = [0, 2]. */
    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *A_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);

    /* B: 4x4 CSR_matrix. */
    CSR_matrix *B = new_CSR_matrix(4, 4, 5);
    B->p[0] = 0;
    B->p[1] = 2;
    B->p[2] = 3;
    B->p[3] = 4;
    B->p[4] = 5;
    int Bi[5] = {0, 2, 1, 0, 3};
    double Bx[5] = {10.0, 20.0, 30.0, 40.0, 50.0};
    memcpy(B->i, Bi, sizeof Bi);
    memcpy(B->x, Bx, sizeof Bx);
    matrix *B_m = new_sparse_matrix(B);

    double d[4] = {1.0, -2.0, 0.5, 3.0};

    /* Wrapper path. */
    matrix *C_m = BTA_matrices_alloc(A_m, B_m);
    BTDA_matrices_fill_values(A_m, d, B_m, C_m);

    /* Direct primitive path: production now dispatches the (PD, Sparse)
       branch through CSC-pd kernels. Build a CSC view of B and call
       BTA_csc_pd_alloc + BTDA_csc_pd_fill_values to match. */
    matrix *A2_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);
    permuted_dense *A2 = (permuted_dense *) A2_m;
    CSR_matrix *B2_csr = new_CSR_matrix(4, 4, 5);
    B2_csr->p[0] = 0;
    B2_csr->p[1] = 2;
    B2_csr->p[2] = 3;
    B2_csr->p[3] = 4;
    B2_csr->p[4] = 5;
    memcpy(B2_csr->i, Bi, sizeof Bi);
    memcpy(B2_csr->x, Bx, sizeof Bx);
    int *iwork = (int *) malloc(MAX(B2_csr->m, B2_csr->n) * sizeof(int));
    CSC_matrix *B2_csc = csr_to_csc_alloc(B2_csr, iwork);
    csr_to_csc_fill_values(B2_csr, B2_csc, iwork);
    matrix *C2 = BTA_csc_pd_alloc(B2_csc, A2);
    BTDA_csc_pd_fill_values(B2_csc, d, A2, (permuted_dense *) C2);

    mu_assert("values", cmp_double_array(C_m->x, C2->x, C_m->nnz));

    free_matrix(C_m);
    free_matrix(B_m);
    free_matrix(A_m);
    free_matrix(C2);
    free_CSC_matrix(B2_csc);
    free_CSR_matrix(B2_csr);
    free(iwork);
    free_matrix(A2_m);
    return 0;
}

/* ---------------------------------------------------------------- */
/* spd to_csr fallback in the BTA/BTDA dispatcher: when an operand is */
/* spd, the dispatcher must materialize it as a temp CSC via to_csr  */
/* and produce results identical to dispatching with the same matrix */
/* expressed directly as a sparse_matrix.                            */
/* ---------------------------------------------------------------- */

/* Deep-copy an spd's csr_cache into a fresh sparse_matrix (taking the
   csr by value so we don't share buffers). Used as the "reference"
   route in the equivalence tests below. */
static matrix *spd_to_sparse_matrix_copy(matrix *spd_m)
{
    CSR_matrix *src = spd_m->to_csr(spd_m);
    CSR_matrix *dst = new_CSR_matrix(src->m, src->n, src->nnz);
    memcpy(dst->p, src->p, (size_t) (src->m + 1) * sizeof(int));
    memcpy(dst->i, src->i, (size_t) src->nnz * sizeof(int));
    memcpy(dst->x, src->x, (size_t) src->nnz * sizeof(double));
    dst->nnz = src->nnz;
    return new_sparse_matrix(dst);
}

/* Primitive BTA_csc_pd kernel (no diagonal): C = B^T @ A, B is CSC, A is PD.
   Route 1 calls BTA_csc_pd_alloc + BTA_csc_pd_fill_values directly. The oracle
   flattens BOTH operands to sparse_matrix and runs the dispatcher with d = ones,
   which routes through the csc/csc sparse_dot path — sharing no code with the
   csc_pd kernel. Both compared via to_csr. */
const char *test_BTA_csc_pd_basic(void)
{
    /* A: 4x5 PD, row_perm = [1,3], col_perm = [0,2] (both permuted, so row_inv
       genuinely permutes and the transpose matters). */
    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *A_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);

    /* B: 4x4 CSR -> CSC. Rows: 0:{0,2} 1:{1} 2:{0} 3:{3}. */
    int Bp[5] = {0, 2, 3, 4, 5};
    int Bi[5] = {0, 2, 1, 0, 3};
    double Bx[5] = {10.0, 20.0, 30.0, 40.0, 50.0};

    CSR_matrix *B_csr = new_CSR_matrix(4, 4, 5);
    memcpy(B_csr->p, Bp, sizeof Bp);
    memcpy(B_csr->i, Bi, sizeof Bi);
    memcpy(B_csr->x, Bx, sizeof Bx);
    int *iwork = (int *) malloc(MAX(B_csr->m, B_csr->n) * sizeof(int));
    CSC_matrix *B_csc = csr_to_csc_alloc(B_csr, iwork);
    csr_to_csc_fill_values(B_csr, B_csc, iwork);

    /* Route 1: our kernel. */
    matrix *C_ours = BTA_csc_pd_alloc(B_csc, (permuted_dense *) A_m);
    BTA_csc_pd_fill_values(B_csc, (permuted_dense *) A_m, (permuted_dense *) C_ours);

    /* Oracle: both operands sparse, d = ones, csc/csc dispatch. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_m);
    CSR_matrix *B_csr2 = new_CSR_matrix(4, 4, 5);
    memcpy(B_csr2->p, Bp, sizeof Bp);
    memcpy(B_csr2->i, Bi, sizeof Bi);
    memcpy(B_csr2->x, Bx, sizeof Bx);
    matrix *B_sparse = new_sparse_matrix(B_csr2);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_CSC_matrix(B_csc);
    free_CSR_matrix(B_csr);
    free(iwork);
    free_matrix(A_m);
    return 0;
}

/* BTA_csc_pd corner case: B columns that (a) hit no A-row -> excluded from C's
   row_perm by BTA_csc_pd_alloc; (b) partially overlap A's rows (some B->i map to
   row_inv == -1) -> exercises sparse_dot_dense's -1 filtering; (c) fully land in
   A's rows. A->row_perm = {1,3}. B columns by source rows: col0:{0,2} (excluded),
   col1:{1} (clean), col2:{0,1,3} (partial), col3:{3} (clean). */
const char *test_BTA_csc_pd_partial_and_excluded_cols(void)
{
    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *A_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);

    /* B: 4x4. Rows: 0:{0,2} 1:{1,2} 2:{0} 3:{2,3}. */
    int Bp[5] = {0, 2, 4, 5, 7};
    int Bi[7] = {0, 2, 1, 2, 0, 2, 3};
    double Bx[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    CSR_matrix *B_csr = new_CSR_matrix(4, 4, 7);
    memcpy(B_csr->p, Bp, sizeof Bp);
    memcpy(B_csr->i, Bi, sizeof Bi);
    memcpy(B_csr->x, Bx, sizeof Bx);
    int *iwork = (int *) malloc(MAX(B_csr->m, B_csr->n) * sizeof(int));
    CSC_matrix *B_csc = csr_to_csc_alloc(B_csr, iwork);
    csr_to_csc_fill_values(B_csr, B_csc, iwork);

    matrix *C_ours = BTA_csc_pd_alloc(B_csc, (permuted_dense *) A_m);
    BTA_csc_pd_fill_values(B_csc, (permuted_dense *) A_m, (permuted_dense *) C_ours);

    matrix *A_sparse = spd_to_sparse_matrix_copy(A_m);
    CSR_matrix *B_csr2 = new_CSR_matrix(4, 4, 7);
    memcpy(B_csr2->p, Bp, sizeof Bp);
    memcpy(B_csr2->i, Bi, sizeof Bi);
    memcpy(B_csr2->x, Bx, sizeof Bx);
    matrix *B_sparse = new_sparse_matrix(B_csr2);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_CSC_matrix(B_csc);
    free_CSR_matrix(B_csr);
    free(iwork);
    free_matrix(A_m);
    return 0;
}

/* BTA_csc_pd corner case: every B column misses A's rows -> C->base.nnz == 0
   early-out. A->row_perm = {1,3}; B has nonzeros only in rows {0,2}. */
const char *test_BTA_csc_pd_empty(void)
{
    int row_perm_A[2] = {1, 3};
    int col_perm_A[2] = {0, 2};
    double XA[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *A_m = new_permuted_dense(4, 5, 2, 2, row_perm_A, col_perm_A, XA);

    /* B: 4x4, nonzeros only in rows 0 and 2. Rows: 0:{0,1} 2:{2,3}. */
    int Bp[5] = {0, 2, 2, 4, 4};
    int Bi[4] = {0, 1, 2, 3};
    double Bx[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_matrix *B_csr = new_CSR_matrix(4, 4, 4);
    memcpy(B_csr->p, Bp, sizeof Bp);
    memcpy(B_csr->i, Bi, sizeof Bi);
    memcpy(B_csr->x, Bx, sizeof Bx);
    int *iwork = (int *) malloc(MAX(B_csr->m, B_csr->n) * sizeof(int));
    CSC_matrix *B_csc = csr_to_csc_alloc(B_csr, iwork);
    csr_to_csc_fill_values(B_csr, B_csc, iwork);

    matrix *C_ours = BTA_csc_pd_alloc(B_csc, (permuted_dense *) A_m);
    BTA_csc_pd_fill_values(B_csc, (permuted_dense *) A_m, (permuted_dense *) C_ours);

    matrix *A_sparse = spd_to_sparse_matrix_copy(A_m);
    CSR_matrix *B_csr2 = new_CSR_matrix(4, 4, 4);
    memcpy(B_csr2->p, Bp, sizeof Bp);
    memcpy(B_csr2->i, Bi, sizeof Bi);
    memcpy(B_csr2->x, Bx, sizeof Bx);
    matrix *B_sparse = new_sparse_matrix(B_csr2);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("nnz zero", csr_ours->nnz == 0);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_CSC_matrix(B_csc);
    free_CSR_matrix(B_csr);
    free(iwork);
    free_matrix(A_m);
    return 0;
}

/* Primitive BTA_pd_csc kernel (no diagonal): C = B^T @ A, B is PD, A is CSC.
   Mirror of the csc_pd tests with roles swapped. Route 1 calls BTA_pd_csc_alloc
   + BTA_pd_csc_fill_values. Oracle flattens BOTH operands to sparse_matrix and
   runs the dispatcher with d = ones (csc/csc sparse_dot path), sharing no code
   with the pd_csc kernel. Both compared via to_csr. */
const char *test_BTA_pd_csc_basic(void)
{
    /* B: 4x5 PD, row_perm = [1,3], col_perm = [0,2]. */
    int row_perm_B[2] = {1, 3};
    int col_perm_B[2] = {0, 2};
    double XB[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *B_m = new_permuted_dense(4, 5, 2, 2, row_perm_B, col_perm_B, XB);

    /* A: 4x4 CSR -> CSC. Rows: 0:{1} 1:{0,2} 2:{3} 3:{1,3}. All A-columns
       overlap B's rows {1,3} (some partially). */
    int Ap[5] = {0, 1, 3, 4, 6};
    int Ai[6] = {1, 0, 2, 3, 1, 3};
    double Ax[6] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};

    CSR_matrix *A_csr = new_CSR_matrix(4, 4, 6);
    memcpy(A_csr->p, Ap, sizeof Ap);
    memcpy(A_csr->i, Ai, sizeof Ai);
    memcpy(A_csr->x, Ax, sizeof Ax);
    int *iwork = (int *) malloc(MAX(A_csr->m, A_csr->n) * sizeof(int));
    CSC_matrix *A_csc = csr_to_csc_alloc(A_csr, iwork);
    csr_to_csc_fill_values(A_csr, A_csc, iwork);

    /* Route 1: our kernel. */
    matrix *C_ours = BTA_pd_csc_alloc((permuted_dense *) B_m, A_csc);
    BTA_pd_csc_fill_values((permuted_dense *) B_m, A_csc, (permuted_dense *) C_ours);

    /* Oracle: both operands sparse, d = ones, csc/csc dispatch. Dispatcher arg
       order is (rightA, leftB) and computes B^T @ A, so A goes first. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_m);
    CSR_matrix *A_csr2 = new_CSR_matrix(4, 4, 6);
    memcpy(A_csr2->p, Ap, sizeof Ap);
    memcpy(A_csr2->i, Ai, sizeof Ai);
    memcpy(A_csr2->x, Ax, sizeof Ax);
    matrix *A_sparse = new_sparse_matrix(A_csr2);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_CSC_matrix(A_csc);
    free_CSR_matrix(A_csr);
    free(iwork);
    free_matrix(B_m);
    return 0;
}

/* BTA_pd_csc corner case: A columns that (a) hit no B-row -> excluded from C's
   col_perm by BTA_pd_csc_alloc; (b) partially overlap B's rows (some A->i map to
   row_inv == -1) -> exercises sparse_dot_dense's -1 filtering; (c) fully land in
   B's rows. B->row_perm = {1,3}. A columns by source rows: col0:{0,2} (excluded),
   col1:{1} (clean), col2:{0,1,3} (partial), col3:{3} (clean). */
const char *test_BTA_pd_csc_partial_and_excluded_cols(void)
{
    int row_perm_B[2] = {1, 3};
    int col_perm_B[2] = {0, 2};
    double XB[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *B_m = new_permuted_dense(4, 5, 2, 2, row_perm_B, col_perm_B, XB);

    /* A: 4x4. Rows: 0:{0,2} 1:{1,2} 2:{0} 3:{2,3}. */
    int Ap[5] = {0, 2, 4, 5, 7};
    int Ai[7] = {0, 2, 1, 2, 0, 2, 3};
    double Ax[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    CSR_matrix *A_csr = new_CSR_matrix(4, 4, 7);
    memcpy(A_csr->p, Ap, sizeof Ap);
    memcpy(A_csr->i, Ai, sizeof Ai);
    memcpy(A_csr->x, Ax, sizeof Ax);
    int *iwork = (int *) malloc(MAX(A_csr->m, A_csr->n) * sizeof(int));
    CSC_matrix *A_csc = csr_to_csc_alloc(A_csr, iwork);
    csr_to_csc_fill_values(A_csr, A_csc, iwork);

    matrix *C_ours = BTA_pd_csc_alloc((permuted_dense *) B_m, A_csc);
    BTA_pd_csc_fill_values((permuted_dense *) B_m, A_csc, (permuted_dense *) C_ours);

    matrix *B_sparse = spd_to_sparse_matrix_copy(B_m);
    CSR_matrix *A_csr2 = new_CSR_matrix(4, 4, 7);
    memcpy(A_csr2->p, Ap, sizeof Ap);
    memcpy(A_csr2->i, Ai, sizeof Ai);
    memcpy(A_csr2->x, Ax, sizeof Ax);
    matrix *A_sparse = new_sparse_matrix(A_csr2);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_CSC_matrix(A_csc);
    free_CSR_matrix(A_csr);
    free(iwork);
    free_matrix(B_m);
    return 0;
}

/* BTA_pd_csc corner case: every A column misses B's rows -> C->base.nnz == 0
   early-out. B->row_perm = {1,3}; A has nonzeros only in rows {0,2}. */
const char *test_BTA_pd_csc_empty(void)
{
    int row_perm_B[2] = {1, 3};
    int col_perm_B[2] = {0, 2};
    double XB[4] = {1.0, 2.0, 3.0, 4.0};
    matrix *B_m = new_permuted_dense(4, 5, 2, 2, row_perm_B, col_perm_B, XB);

    /* A: 4x4, nonzeros only in rows 0 and 2. Rows: 0:{0,1} 2:{2,3}. */
    int Ap[5] = {0, 2, 2, 4, 4};
    int Ai[4] = {0, 1, 2, 3};
    double Ax[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_matrix *A_csr = new_CSR_matrix(4, 4, 4);
    memcpy(A_csr->p, Ap, sizeof Ap);
    memcpy(A_csr->i, Ai, sizeof Ai);
    memcpy(A_csr->x, Ax, sizeof Ax);
    int *iwork = (int *) malloc(MAX(A_csr->m, A_csr->n) * sizeof(int));
    CSC_matrix *A_csc = csr_to_csc_alloc(A_csr, iwork);
    csr_to_csc_fill_values(A_csr, A_csc, iwork);

    matrix *C_ours = BTA_pd_csc_alloc((permuted_dense *) B_m, A_csc);
    BTA_pd_csc_fill_values((permuted_dense *) B_m, A_csc, (permuted_dense *) C_ours);

    matrix *B_sparse = spd_to_sparse_matrix_copy(B_m);
    CSR_matrix *A_csr2 = new_CSR_matrix(4, 4, 4);
    memcpy(A_csr2->p, Ap, sizeof Ap);
    memcpy(A_csr2->i, Ai, sizeof Ai);
    memcpy(A_csr2->x, Ax, sizeof Ax);
    matrix *A_sparse = new_sparse_matrix(A_csr2);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("nnz zero", csr_ours->nnz == 0);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_CSC_matrix(A_csc);
    free_CSR_matrix(A_csr);
    free(iwork);
    free_matrix(B_m);
    return 0;
}

/* Wrapper dispatch: (A=spd, B=PD). The spd-fallback path goes through
   BTA_csc_pd_alloc / BTDA_csc_pd_fill_values; result must match the
   equivalent (A=sparse, B=PD) dispatch on the same matrix. */
const char *test_BTDA_matrices_spd_pd(void)
{
    /* A: 4x3 spd, two blocks.
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x5 PD. */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: dispatcher with spd operand. */
    matrix *C_spd = BTA_matrices_alloc(A_spd, B);
    BTDA_matrices_fill_values(A_spd, d, B, C_spd);

    /* Route 2: dispatcher with the same matrix expressed as sparse. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B);
    BTDA_matrices_fill_values(A_sparse, d, B, C_ref);

    /* Both routes produce a PD (B is PD). Compare X buffers. */
    permuted_dense *C_spd_pd = (permuted_dense *) C_spd;
    permuted_dense *C_ref_pd = (permuted_dense *) C_ref;
    mu_assert("m", C_spd->m == C_ref->m);
    mu_assert("n", C_spd->n == C_ref->n);
    mu_assert("m0", C_spd_pd->m0 == C_ref_pd->m0);
    mu_assert("n0", C_spd_pd->n0 == C_ref_pd->n0);
    mu_assert("row_perm",
              cmp_int_array(C_spd_pd->row_perm, C_ref_pd->row_perm, C_spd_pd->m0));
    mu_assert("col_perm",
              cmp_int_array(C_spd_pd->col_perm, C_ref_pd->col_perm, C_spd_pd->n0));
    mu_assert("X", cmp_double_array(C_spd_pd->X, C_ref_pd->X,
                                    (size_t) C_spd_pd->m0 * C_spd_pd->n0));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_spd);
    free_matrix(B);
    free_matrix(A_spd);
    return 0;
}

/* Wrapper dispatch: (A=PD, B=spd). Goes through BTA_pd_csc_alloc /
   BTDA_pd_csc_fill_values; equivalence with sparse-B. */
const char *test_BTDA_matrices_pd_spd(void)
{
    /* A: 4x5 PD. */
    int A_rp[3] = {0, 1, 2};
    int A_cp[3] = {1, 3, 4};
    double AX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *A = new_permuted_dense(4, 5, 3, 3, A_rp, A_cp, AX);

    /* B: 4x3 spd, same as test_BTDA_matrices_spd_pd. */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    matrix *C_spd = BTA_matrices_alloc(A, B_spd);
    BTDA_matrices_fill_values(A, d, B_spd, C_spd);

    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A, B_sparse);
    BTDA_matrices_fill_values(A, d, B_sparse, C_ref);

    /* Output types now differ: the spd-B route returns stacked_pd (via
       BTA_spd_matrices), the sparse-B route returns permuted_dense (via
       BTA_sparse_matrices' PD branch). Compare via to_csr. */
    CSR_matrix *csr_spd = C_spd->to_csr(C_spd);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_spd->m == csr_ref->m);
    mu_assert("n", csr_spd->n == csr_ref->n);
    mu_assert("nnz", csr_spd->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_spd->p, csr_ref->p, csr_spd->m + 1));
    mu_assert("i", cmp_int_array(csr_spd->i, csr_ref->i, csr_spd->nnz));
    mu_assert("x", cmp_double_array(csr_spd->x, csr_ref->x, csr_spd->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_spd);
    free_matrix(B_spd);
    free_matrix(A);
    return 0;
}

/* Wrapper dispatch: (A=spd, B=spd). The spd-spd route returns stacked_pd
   (via BTA_spd_matrices); the sparse-sparse fallback returns sparse_matrix.
   Compare via to_csr. */
const char *test_BTDA_matrices_spd_spd(void)
{
    /* A: 4x3 spd. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *Ablk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *Ablk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) Ablk0,
                                   (permuted_dense *) Ablk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x4 spd. */
    int B0_rp[2] = {0, 2};
    int B0_cp[2] = {0, 3};
    double B0X[4] = {9, 8, 7, 6};
    matrix *Bblk0 = new_permuted_dense(4, 4, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {1, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 4, 3, 2};
    matrix *Bblk1 = new_permuted_dense(4, 4, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) Bblk0,
                                   (permuted_dense *) Bblk1};
    matrix *B_spd = new_stacked_pd(4, 4, 2, B_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    matrix *C_spd = BTA_matrices_alloc(A_spd, B_spd);
    BTDA_matrices_fill_values(A_spd, d, B_spd, C_spd);

    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    BTDA_matrices_fill_values(A_sparse, d, B_sparse, C_ref);

    /* Output types differ: spd-spd route returns stacked_pd, sparse-sparse
       returns sparse_matrix. Compare via to_csr. */
    CSR_matrix *csr_spd = C_spd->to_csr(C_spd);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_spd->m == csr_ref->m);
    mu_assert("n", csr_spd->n == csr_ref->n);
    mu_assert("nnz", csr_spd->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_spd->p, csr_ref->p, csr_spd->m + 1));
    mu_assert("i", cmp_int_array(csr_spd->i, csr_ref->i, csr_spd->nnz));
    mu_assert("x", cmp_double_array(csr_spd->x, csr_ref->x, csr_spd->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_spd);
    free_matrix(B_spd);
    free_matrix(A_spd);
    return 0;
}

/* Primitive BTA_pd_spd kernel (transpose + BA_pd_spd implementation):
   C = B^T @ A with B a permuted_dense and A a 2-block stacked_pd whose
   col_perms overlap (share col 2), exercising the scatter-accumulate
   path inside BA_pd_spd. Reference is the production dispatcher path
   with A flattened to a sparse_matrix and d = ones (BTDA with all-ones
   d is equivalent to BTA). */
const char *test_BTA_pd_spd_two_blocks_both_kept(void)
{
    /* A: 4x3 spd, two blocks.
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x5 PD with row_perm = {0,1,2}, col_perm = {1,3,4}. */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    /* Route 1: our new BTA_pd_spd_* primitives. */
    matrix *C_ours = BTA_pd_spd_alloc((permuted_dense *) B, (stacked_pd *) A_spd);
    BTA_pd_spd_fill_values((permuted_dense *) B, (stacked_pd *) A_spd,
                           (permuted_dense *) C_ours);

    /* Route 2: dispatcher with A flattened to sparse_matrix, d = ones. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B, C_ref);

    /* Both routes produce a PD with row_perm = B->col_perm and col_perm
       = union of contributing A_k->col_perms. Compare shape, perms, X. */
    permuted_dense *C_ours_pd = (permuted_dense *) C_ours;
    permuted_dense *C_ref_pd = (permuted_dense *) C_ref;
    mu_assert("m", C_ours->m == C_ref->m);
    mu_assert("n", C_ours->n == C_ref->n);
    mu_assert("m0", C_ours_pd->m0 == C_ref_pd->m0);
    mu_assert("n0", C_ours_pd->n0 == C_ref_pd->n0);
    mu_assert("row_perm",
              cmp_int_array(C_ours_pd->row_perm, C_ref_pd->row_perm, C_ours_pd->m0));
    mu_assert("col_perm",
              cmp_int_array(C_ours_pd->col_perm, C_ref_pd->col_perm, C_ours_pd->n0));
    mu_assert("X", cmp_double_array(C_ours_pd->X, C_ref_pd->X,
                                    (size_t) C_ours_pd->m0 * C_ours_pd->n0));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(B);
    free_matrix(A_spd);
    return 0;
}

/* Primitive BTDA_pd_spd kernel (temp-DA composition):
   C = B^T @ diag(d) @ A. Same B and A layout as
   test_BTA_pd_spd_two_blocks_both_kept, but with a non-trivial d so
   BTDA differs from BTA. Reference is the production dispatcher path
   with A flattened to a sparse_matrix. */
const char *test_BTDA_pd_spd_two_blocks_both_kept(void)
{
    /* A: 4x3 spd, two blocks.
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x5 PD with row_perm = {0,1,2}, col_perm = {1,3,4}. */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    /* Non-trivial d so BTDA != BTA. */
    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: our new BTDA_pd_spd_fill_values. C structure comes from
       BTA_pd_spd_alloc (BTDA shares sparsity with BTA). */
    matrix *C_ours = BTA_pd_spd_alloc((permuted_dense *) B, (stacked_pd *) A_spd);
    BTDA_pd_spd_fill_values((permuted_dense *) B, d, (stacked_pd *) A_spd,
                            (permuted_dense *) C_ours);

    /* Route 2: dispatcher with A flattened to sparse_matrix. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B);
    BTDA_matrices_fill_values(A_sparse, d, B, C_ref);

    /* Both routes produce a PD with row_perm = B->col_perm and col_perm
       = union of contributing A_k->col_perms. Compare shape, perms, X. */
    permuted_dense *C_ours_pd = (permuted_dense *) C_ours;
    permuted_dense *C_ref_pd = (permuted_dense *) C_ref;
    mu_assert("m", C_ours->m == C_ref->m);
    mu_assert("n", C_ours->n == C_ref->n);
    mu_assert("m0", C_ours_pd->m0 == C_ref_pd->m0);
    mu_assert("n0", C_ours_pd->n0 == C_ref_pd->n0);
    mu_assert("row_perm",
              cmp_int_array(C_ours_pd->row_perm, C_ref_pd->row_perm, C_ours_pd->m0));
    mu_assert("col_perm",
              cmp_int_array(C_ours_pd->col_perm, C_ref_pd->col_perm, C_ours_pd->n0));
    mu_assert("X", cmp_double_array(C_ours_pd->X, C_ref_pd->X,
                                    (size_t) C_ours_pd->m0 * C_ours_pd->n0));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(B);
    free_matrix(A_spd);
    return 0;
}

/* Primitive BTDA_spd_pd kernel (ATA-style direct: per-block BTDA_pd_pd
   + accumulating coalesce). C = B^T @ diag(d) @ A with B a 2-block
   stacked_pd whose col_perms share column 2 — exercises the
   accumulating coalesce path (multiple source blocks contribute to the
   same output row). Reference is the production dispatcher with B
   flattened to a sparse_matrix; outputs differ in storage (stacked_pd
   vs permuted_dense), so compare via to_csr. */
const char *test_BTDA_spd_pd_overlapping_cp(void)
{
    /* B: 4x3 spd, two blocks with overlapping col_perms (share col 2).
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 PD, full row_perm, full col_perm. */
    int A_rp[4] = {0, 1, 2, 3};
    int A_cp[3] = {0, 1, 2};
    double AX[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    matrix *A = new_permuted_dense(4, 3, 4, 3, A_rp, A_cp, AX);

    /* Non-trivial d so BTDA != BTA. */
    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: our new BTDA_spd_pd_fill_values. */
    matrix *C_ours = BTA_spd_pd_alloc((stacked_pd *) B_spd, (permuted_dense *) A);
    BTDA_spd_pd_fill_values((stacked_pd *) B_spd, d, (permuted_dense *) A,
                            (stacked_pd *) C_ours);

    /* Route 2: dispatcher with B flattened to sparse_matrix. Note
       BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A, B_sparse);
    BTDA_matrices_fill_values(A, d, B_sparse, C_ref);

    /* C_ours is stacked_pd, C_ref is permuted_dense — compare via to_csr.
       Both represent the same global matrix; rows are emitted in sorted
       order, and the spd's per-signature blocks share col_perm = A->col_perm,
       so the CSR layouts match exactly. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A);
    free_matrix(B_spd);
    return 0;
}

/* Primitive BTA_spd_pd kernel (no diagonal). Same matrices as the BTDA
   variant above. Reference is the production dispatcher with B flattened to a
   sparse_matrix and d = ones, which equals plain B^T @ A. */
const char *test_BTA_spd_pd_overlapping_cp(void)
{
    /* B: 4x3 spd, two blocks with overlapping col_perms (share col 2).
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 PD, full row_perm, full col_perm. */
    int A_rp[4] = {0, 1, 2, 3};
    int A_cp[3] = {0, 1, 2};
    double AX[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    matrix *A = new_permuted_dense(4, 3, 4, 3, A_rp, A_cp, AX);

    /* Route 1: our new BTA_spd_pd_fill_values. */
    matrix *C_ours = BTA_spd_pd_alloc((stacked_pd *) B_spd, (permuted_dense *) A);
    BTA_spd_pd_fill_values((stacked_pd *) B_spd, (permuted_dense *) A,
                           (stacked_pd *) C_ours);

    /* Route 2: dispatcher with B flattened to sparse_matrix and d = ones, so
       BTDA == BTA. Note BTA_matrices_alloc(A, B) computes B^T @ A, so A goes
       first. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A, d_ones, B_sparse, C_ref);

    /* C_ours is stacked_pd, C_ref is permuted_dense — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A);
    free_matrix(B_spd);
    return 0;
}

/* Primitive BTDA_spd_csc kernel (ATA-style direct: per-block BTDA_pd_csc
   + accumulating coalesce). C = B^T @ diag(d) @ A with B a 2-block
   stacked_pd whose col_perms share column 2, and A a sparse_matrix
   wrapping a small CSR. Reference is the production dispatcher with B
   flattened to a sparse_matrix; outputs differ in storage (stacked_pd
   vs sparse_matrix), so compare via to_csr. */
const char *test_BTDA_spd_csc_overlapping_cp(void)
{
    /* B: 4x3 spd, two blocks with overlapping col_perms (share col 2).
       Same layout as test_BTDA_spd_pd_overlapping_cp.
       blk0: rows {0,1}, cols {0,2}, X = [[1,2],[3,4]]
       blk1: rows {2,3}, cols {1,2}, X = [[5,6],[7,8]]                     */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 sparse_matrix wrapping a CSR with arbitrary nonzeros.
       Row 0: col 0 = 1, col 2 = 2
       Row 1: col 1 = 3
       Row 2: col 0 = 4, col 1 = 5, col 2 = 6
       Row 3: col 2 = 7
       p = [0, 2, 3, 6, 7], i = [0,2, 1, 0,1,2, 2], x = [1,2, 3, 4,5,6, 7]. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 7);
    int Ap[5] = {0, 2, 3, 6, 7};
    int Ai[7] = {0, 2, 1, 0, 1, 2, 2};
    double Ax[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    /* Non-trivial d so BTDA != BTA. */
    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: our new BTDA_spd_csc_fill_values on A's CSC mirror. */
    const CSC_matrix *A_sm_csc = sparse_csc_fresh(A_sm);
    matrix *C_ours = BTA_spd_csc_alloc((stacked_pd *) B_spd, A_sm_csc);
    BTDA_spd_csc_fill_values((stacked_pd *) B_spd, d, A_sm_csc,
                             (stacked_pd *) C_ours);

    /* Route 2: dispatcher with B flattened to sparse_matrix. Note
       BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sm, B_sparse);
    BTDA_matrices_fill_values(A_sm, d, B_sparse, C_ref);

    /* C_ours is stacked_pd, C_ref is sparse_matrix — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B_spd);
    return 0;
}

/* Primitive BTA_spd_csc kernel (no diagonal): C = B^T @ A, B is a 2-block
   stacked_pd whose col_perms share a column (accumulating coalesce fires), A is
   CSC. Same matrices as the BTDA variant above. Oracle is the dispatcher with B
   flattened to sparse and d = ones (csc/csc sparse_dot path). */
const char *test_BTA_spd_csc_overlapping(void)
{
    /* B: 4x3 spd, two blocks with overlapping col_perms (share col 2). */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 sparse_matrix wrapping a CSR. Rows: 0:{0,2} 1:{1} 2:{0,1,2} 3:{2}. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 7);
    int Ap[5] = {0, 2, 3, 6, 7};
    int Ai[7] = {0, 2, 1, 0, 1, 2, 2};
    double Ax[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    /* Route 1: new kernel on A's CSC mirror. */
    const CSC_matrix *A_sm_csc = sparse_csc_fresh(A_sm);
    matrix *C_ours = BTA_spd_csc_alloc((stacked_pd *) B_spd, A_sm_csc);
    BTA_spd_csc_fill_values((stacked_pd *) B_spd, A_sm_csc, (stacked_pd *) C_ours);

    /* Route 2: dispatcher with B flattened to sparse and d = ones, so BTDA == BTA.
       BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sm, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sm, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B_spd);
    return 0;
}

/* BTA_spd_csc corner case: one B-block whose rows match no nonzero row of A ->
   that block's per-block partial is empty (m0 = 0), while the other block
   contributes. Exercises the empty-source-block path through BTA_pd_csc_alloc
   and the accumulating coalesce. */
const char *test_BTA_spd_csc_block_no_overlap(void)
{
    /* B: 4x3 spd, two blocks. blk0 rows {0,1}; blk1 rows {2,3}. */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 1};
    double B0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3, nonzeros only in rows {0,1} -> overlaps blk0 only; blk1's rows
       {2,3} hit no A nonzero -> empty partial. Rows: 0:{0,2} 1:{1}. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 3);
    int Ap[5] = {0, 2, 3, 3, 3};
    int Ai[3] = {0, 2, 1};
    double Ax[3] = {1, 2, 3};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    const CSC_matrix *A_sm_csc = sparse_csc_fresh(A_sm);
    matrix *C_ours = BTA_spd_csc_alloc((stacked_pd *) B_spd, A_sm_csc);
    BTA_spd_csc_fill_values((stacked_pd *) B_spd, A_sm_csc, (stacked_pd *) C_ours);

    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sm, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sm, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B_spd);
    return 0;
}

/* Primitive BTDA_spd_spd kernel (per-block BTDA_pd_spd + accumulating coalesce).
   C = B^T @ diag(d) @ A with both B and A 2-block stacked_pds. B's c_k's
   share col 2 (outer coalesce-accumulate path), and A's col_perms also
   share at least one column (inner BTA_pd_spd accumulate path). Reference
   is the production dispatcher with both operands flattened to sparse_matrix.
   Output types differ (stacked_pd vs sparse_matrix); compare via to_csr. */
const char *test_BTDA_spd_spd_overlapping(void)
{
    /* B: 4x3 spd, two blocks with overlapping c_k(B) (share col 2). */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B_blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B_blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 spd, two blocks. A_0's rows overlap B_0's, A_1's rows overlap
       B_1's. A_0 and A_1 share col 1 → inner BTA_pd_spd accumulate fires. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {20, 21, 22, 23};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: new kernel. */
    matrix *C_ours = BTA_spd_spd_alloc((stacked_pd *) B_spd, (stacked_pd *) A_spd);
    BTDA_spd_spd_fill_values((stacked_pd *) B_spd, d, (stacked_pd *) A_spd,
                             (stacked_pd *) C_ours);

    /* Route 2: dispatcher with both flattened to sparse_matrix.
       BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    BTDA_matrices_fill_values(A_sparse, d, B_sparse, C_ref);

    /* C_ours is stacked_pd, C_ref is sparse_matrix — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_spd);
    return 0;
}

/* Primitive BTA_spd_spd kernel (no diagonal). Same matrices as the BTDA
   variant above. Reference is the production dispatcher with both operands
   flattened to sparse_matrix and d = ones, which equals plain B^T @ A. */
const char *test_BTA_spd_spd_overlapping(void)
{
    /* B: 4x3 spd, two blocks with overlapping c_k(B) (share col 2). */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B_blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B_blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 spd, two blocks. A_0's rows overlap B_0's, A_1's rows overlap
       B_1's. A_0 and A_1 share col 1. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {20, 21, 22, 23};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* Route 1: new kernel. */
    matrix *C_ours = BTA_spd_spd_alloc((stacked_pd *) B_spd, (stacked_pd *) A_spd);
    BTA_spd_spd_fill_values((stacked_pd *) B_spd, (stacked_pd *) A_spd,
                            (stacked_pd *) C_ours);

    /* Route 2: dispatcher with both flattened to sparse_matrix and d = ones, so
       BTDA == BTA. BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    /* C_ours is stacked_pd, C_ref is sparse_matrix — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_spd);
    return 0;
}

/* BTA_spd_spd corner case: a single B-block whose rows span MULTIPLE A-blocks,
   forcing BTA_pd_spd_fill_values' inner += accumulation over A-blocks into one
   partial, with overlapping output columns (A_0 and A_1 share col 1). Not
   exercised by test_BTA_spd_spd_overlapping (there each B-block overlaps exactly
   one A-block). */
const char *test_BTA_spd_spd_multi_A_per_block(void)
{
    /* B: 4x2, one block spanning all four contraction rows. */
    int B0_rp[4] = {0, 1, 2, 3};
    int B0_cp[2] = {0, 1};
    double B0X[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    matrix *B_blk0 = new_permuted_dense(4, 2, 4, 2, B0_rp, B0_cp, B0X);
    permuted_dense *B_blocks[1] = {(permuted_dense *) B_blk0};
    matrix *B_spd = new_stacked_pd(4, 2, 1, B_blocks, NULL, NULL);

    /* A: 4x3, two blocks with disjoint rows but overlapping cols (share col 1).
       Both overlap B_0's rows, so both contribute to B_0's single partial. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {20, 21, 22, 23};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    matrix *C_ours = BTA_spd_spd_alloc((stacked_pd *) B_spd, (stacked_pd *) A_spd);
    BTA_spd_spd_fill_values((stacked_pd *) B_spd, (stacked_pd *) A_spd,
                            (stacked_pd *) C_ours);

    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_spd);
    return 0;
}

/* BTA_spd_spd corner case: a B-block that overlaps NO A-block, producing an
   empty partial (n0 = 0) that coalesces to a zero-column output block (output
   row 1, all zeros). Exercises the partial-level nnz == 0 early-out in
   BTA_pd_spd_fill_values and the empty-block path through coalesce/to_csr. */
const char *test_BTA_spd_spd_nonoverlapping_block(void)
{
    /* B: 4x2, two blocks. B_0 rows {0,1}; B_1 rows {2,3}. */
    int B0_rp[2] = {0, 1};
    int B0_cp[1] = {0};
    double B0X[2] = {1, 2};
    matrix *B_blk0 = new_permuted_dense(4, 2, 2, 1, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[1] = {1};
    double B1X[2] = {3, 4};
    matrix *B_blk1 = new_permuted_dense(4, 2, 2, 1, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 2, 2, B_blocks, NULL, NULL);

    /* A: 4x2, one block on rows {0,1}. B_0 overlaps it; B_1 does not. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A_blk0 = new_permuted_dense(4, 2, 2, 2, A0_rp, A0_cp, A0X);
    permuted_dense *A_blocks[1] = {(permuted_dense *) A_blk0};
    matrix *A_spd = new_stacked_pd(4, 2, 1, A_blocks, NULL, NULL);

    matrix *C_ours = BTA_spd_spd_alloc((stacked_pd *) B_spd, (stacked_pd *) A_spd);
    BTA_spd_spd_fill_values((stacked_pd *) B_spd, (stacked_pd *) A_spd,
                            (stacked_pd *) C_ours);

    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_spd);
    return 0;
}

/* BTA_spd_matrices dispatcher: A is permuted_dense. Verifies the
   PD-branch routes to BTA_spd_pd / BTDA_spd_pd. */
const char *test_BTA_spd_matrices_pd_A(void)
{
    /* Same inputs as test_BTDA_spd_pd_overlapping_cp. */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B_blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B_blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    int A_rp[4] = {0, 1, 2, 3};
    int A_cp[3] = {0, 1, 2};
    double AX[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    matrix *A = new_permuted_dense(4, 3, 4, 3, A_rp, A_cp, AX);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with B in its native spd type — routes
       internally through the spd-B branch. */
    matrix *C_ours = BTA_matrices_alloc(A, B_spd);
    BTDA_matrices_fill_values(A, d, B_spd, C_ours);

    /* Route 2: same public dispatcher with B flattened to sparse — routes
       through a different internal branch (csc-PD path). */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A, B_sparse);
    BTDA_matrices_fill_values(A, d, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A);
    free_matrix(B_spd);
    return 0;
}

/* Public dispatcher equivalence: B is spd and A is sparse_matrix. The
   spd-B internal branch is exercised in Route 1; Route 2 flattens B to
   sparse and exercises the csc-csc branch. */
const char *test_BTA_spd_matrices_csc_A(void)
{
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B_blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B_blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A as sparse_matrix wrapping a CSR (same shape/values as
       test_BTDA_spd_csc_overlapping_cp). */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 7);
    int Ap[5] = {0, 2, 3, 6, 7};
    int Ai[7] = {0, 2, 1, 0, 1, 2, 2};
    double Ax[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with B native spd — routes through
       the spd-B branch (BTA_spd_csc internally). */
    matrix *C_ours = BTA_matrices_alloc(A_sm, B_spd);
    BTDA_matrices_fill_values(A_sm, d, B_spd, C_ours);

    /* Route 2: same public dispatcher with B flattened to sparse —
       routes through the csc-csc branch. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sm, B_sparse);
    BTDA_matrices_fill_values(A_sm, d, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B_spd);
    return 0;
}

/* BTA_spd_matrices dispatcher: A is stacked_pd. Verifies the
   spd-branch routes to BTA_spd_spd / BTDA_spd_spd. */
const char *test_BTA_spd_matrices_spd_A(void)
{
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B_blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B_blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {10, 11, 12, 13};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {20, 21, 22, 23};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with both operands as native spd —
       routes through the spd-spd branch (BTA_spd_spd internally). */
    matrix *C_ours = BTA_matrices_alloc(A_spd, B_spd);
    BTDA_matrices_fill_values(A_spd, d, B_spd, C_ours);

    /* Route 2: same public dispatcher with both flattened to sparse —
       routes through the csc-csc branch. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *B_sparse = spd_to_sparse_matrix_copy(B_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sparse);
    BTDA_matrices_fill_values(A_sparse, d, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_spd);
    return 0;
}

/* BA_spd_matrices dispatcher tests previously lived here. The dispatcher
   is now static inside matmul_dispatchers.c (no production caller, no public
   API). The underlying primitives BA_spd_csc_*, BA_spd_pd_*, BA_spd_spd_*
   are tested directly in tests/utils/test_stacked_pd.h. */

/* BTA_pd_matrices dispatcher: A is permuted_dense. Verifies the PD-branch
   routes to BTA_pd_pd / BTDA_pd_pd. Both routes produce a PD; compare X
   buffers directly. */
const char *test_BTA_pd_matrices_pd_A(void)
{
    /* B: 2x4 PD with row_perm = {0,1}, col_perm = {1,3}. */
    int B_rp[2] = {0, 1};
    int B_cp[2] = {1, 3};
    double BX[4] = {1, 2, 3, 4};
    matrix *B = new_permuted_dense(2, 4, 2, 2, B_rp, B_cp, BX);

    /* A: 2x4 PD with row_perm = {0,1}, col_perm = {0,2}. */
    int A_rp[2] = {0, 1};
    int A_cp[2] = {0, 2};
    double AX[4] = {5, 6, 7, 8};
    matrix *A = new_permuted_dense(2, 4, 2, 2, A_rp, A_cp, AX);

    double d[2] = {2.0, -1.5};

    /* Route 1: public dispatcher with both operands native PD —
       routes through BTA_pd_matrices -> BTA_pd_pd internally. */
    matrix *C_ours = BTA_matrices_alloc(A, B);
    BTDA_matrices_fill_values(A, d, B, C_ours);

    /* Route 2: same dispatcher with A flattened to sparse — routes
       through BTA_pd_matrices -> BTA_pd_csc, a different kernel. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B);
    BTDA_matrices_fill_values(A_sparse, d, B, C_ref);

    /* Both routes produce PD (B is PD). Compare X/perms directly. */
    permuted_dense *C_ours_pd = (permuted_dense *) C_ours;
    permuted_dense *C_ref_pd = (permuted_dense *) C_ref;
    mu_assert("m", C_ours->m == C_ref->m);
    mu_assert("n", C_ours->n == C_ref->n);
    mu_assert("m0", C_ours_pd->m0 == C_ref_pd->m0);
    mu_assert("n0", C_ours_pd->n0 == C_ref_pd->n0);
    mu_assert("row_perm",
              cmp_int_array(C_ours_pd->row_perm, C_ref_pd->row_perm, C_ours_pd->m0));
    mu_assert("col_perm",
              cmp_int_array(C_ours_pd->col_perm, C_ref_pd->col_perm, C_ours_pd->n0));
    mu_assert("X", cmp_double_array(C_ours_pd->X, C_ref_pd->X,
                                    (size_t) C_ours_pd->m0 * C_ours_pd->n0));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* BTA_pd_matrices dispatcher: A is sparse_matrix. Verifies the sparse-branch
   routes to BTA_pd_csc / BTDA_pd_csc and keeps A's CSC mirror fresh itself. */
const char *test_BTA_pd_matrices_csc_A(void)
{
    /* B: 4x5 PD. */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    /* A: 4x3 sparse_matrix wrapping a CSR. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 7);
    int Ap[5] = {0, 2, 3, 6, 7};
    int Ai[7] = {0, 2, 1, 0, 1, 2, 2};
    double Ax[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with B native PD, A native sparse —
       routes through BTA_pd_matrices -> BTA_pd_csc internally. */
    matrix *C_ours = BTA_matrices_alloc(A_sm, B);
    BTDA_matrices_fill_values(A_sm, d, B, C_ours);

    /* Route 2: same dispatcher with B flattened to sparse — routes
       through BTA_sparse_matrices -> BTA_alloc (csc-csc), a different
       internal kernel. Output type differs (sparse_matrix vs PD), so
       compare via to_csr. */
    matrix *B_sparse = spd_to_sparse_matrix_copy(B);
    matrix *C_ref = BTA_matrices_alloc(A_sm, B_sparse);
    BTDA_matrices_fill_values(A_sm, d, B_sparse, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B_sparse);
    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B);
    return 0;
}

/* BTA_pd_matrices dispatcher: A is stacked_pd. Verifies the spd-branch
   routes to BTA_pd_spd / BTDA_pd_spd. */
const char *test_BTA_pd_matrices_spd_A(void)
{
    /* B: 4x5 PD. */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    /* A: 4x3 spd, two blocks with shared col 2 (exercises BTA_pd_spd
       scatter-accumulate). */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with A native spd, B native PD —
       routes through BTA_pd_matrices -> BTA_pd_spd internally. */
    matrix *C_ours = BTA_matrices_alloc(A_spd, B);
    BTDA_matrices_fill_values(A_spd, d, B, C_ours);

    /* Route 2: same public dispatcher with A flattened to sparse —
       routes through BTA_pd_matrices -> BTA_pd_csc, a different
       internal kernel. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B);
    BTDA_matrices_fill_values(A_sparse, d, B, C_ref);

    /* Both routes produce PD (B is PD). Compare X/perms directly. */
    permuted_dense *C_ours_pd = (permuted_dense *) C_ours;
    permuted_dense *C_ref_pd = (permuted_dense *) C_ref;
    mu_assert("m", C_ours->m == C_ref->m);
    mu_assert("n", C_ours->n == C_ref->n);
    mu_assert("m0", C_ours_pd->m0 == C_ref_pd->m0);
    mu_assert("n0", C_ours_pd->n0 == C_ref_pd->n0);
    mu_assert("row_perm",
              cmp_int_array(C_ours_pd->row_perm, C_ref_pd->row_perm, C_ours_pd->m0));
    mu_assert("col_perm",
              cmp_int_array(C_ours_pd->col_perm, C_ref_pd->col_perm, C_ours_pd->n0));
    mu_assert("X", cmp_double_array(C_ours_pd->X, C_ref_pd->X,
                                    (size_t) C_ours_pd->m0 * C_ours_pd->n0));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B);
    return 0;
}

/* Primitive BTDA_csc_spd kernel: C = B^T @ diag(d) @ A with B a CSC and A a
   2-block stacked_pd whose col_perms share at least one column (so the
   accumulating coalesce path fires). Reference is the production
   dispatcher with A flattened to a sparse_matrix; outputs differ in
   storage (stacked_pd vs sparse_matrix), so compare via to_csr. */
const char *test_BTDA_csc_spd_overlapping(void)
{
    /* A: 4x3 spd, two blocks with overlapping col_perms (share col 2). */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x3 sparse_matrix wrapping a CSR. */
    CSR_matrix *B_csr = new_CSR_matrix(4, 3, 7);
    int Bp[5] = {0, 2, 3, 6, 7};
    int Bi[7] = {0, 2, 1, 0, 1, 2, 2};
    double Bx[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: new kernel on B's CSC mirror. */
    const CSC_matrix *B_sm_csc = sparse_csc_fresh(B_sm);
    matrix *C_ours = BTA_csc_spd_alloc(B_sm_csc, (stacked_pd *) A_spd);
    BTDA_csc_spd_fill_values(B_sm_csc, d, (stacked_pd *) A_spd,
                             (stacked_pd *) C_ours);

    /* Route 2: production dispatcher with A flattened to sparse_matrix.
       BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sm);
    BTDA_matrices_fill_values(A_sparse, d, B_sm, C_ref);

    /* C_ours is stacked_pd, C_ref is sparse_matrix — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(B_sm);
    free_matrix(A_spd);
    return 0;
}

/* Primitive BTA_csc_spd kernel (no diagonal): C = B^T @ A, B is CSC, A is a
   2-block stacked_pd whose col_perms share a column (so the accumulating coalesce
   path fires). Same matrices as the BTDA variant above. Oracle is the dispatcher
   with A flattened to sparse and d = ones (csc/csc sparse_dot path). */
const char *test_BTA_csc_spd_overlapping(void)
{
    /* A: 4x3 spd, two blocks with overlapping col_perms (share col 2). */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x3 sparse_matrix wrapping a CSR. */
    CSR_matrix *B_csr = new_CSR_matrix(4, 3, 7);
    int Bp[5] = {0, 2, 3, 6, 7};
    int Bi[7] = {0, 2, 1, 0, 1, 2, 2};
    double Bx[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    /* Route 1: new kernel on B's CSC mirror. */
    const CSC_matrix *B_sm_csc = sparse_csc_fresh(B_sm);
    matrix *C_ours = BTA_csc_spd_alloc(B_sm_csc, (stacked_pd *) A_spd);
    BTA_csc_spd_fill_values(B_sm_csc, (stacked_pd *) A_spd, (stacked_pd *) C_ours);

    /* Route 2: dispatcher with A flattened to sparse_matrix and d = ones, so
       BTDA == BTA. BTA_matrices_alloc(A, B) computes B^T @ A, so A goes first. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sm);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sm, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(B_sm);
    free_matrix(A_spd);
    return 0;
}

/* BTA_csc_spd corner case: one A-block whose row_perm matches no nonzero row of
   any B column -> that block's per-block partial is empty (m0 = 0), while the
   other block contributes. Exercises the empty-source-block path through
   BTA_csc_pd_alloc and the accumulating coalesce. */
const char *test_BTA_csc_spd_block_no_overlap(void)
{
    /* A: 4x3 spd, two blocks. blk0 rows {0,1}; blk1 rows {2,3}. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 1};
    double A0X[4] = {1, 2, 3, 4};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    /* B: 4x3, nonzeros only in rows {0,1} -> overlaps blk0 only; blk1's rows
       {2,3} hit no B column -> empty partial for blk1. Rows: 0:{0,2} 1:{1}. */
    CSR_matrix *B_csr = new_CSR_matrix(4, 3, 3);
    int Bp[5] = {0, 2, 3, 3, 3};
    int Bi[3] = {0, 2, 1};
    double Bx[3] = {1, 2, 3};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    const CSC_matrix *B_sm_csc = sparse_csc_fresh(B_sm);
    matrix *C_ours = BTA_csc_spd_alloc(B_sm_csc, (stacked_pd *) A_spd);
    BTA_csc_spd_fill_values(B_sm_csc, (stacked_pd *) A_spd, (stacked_pd *) C_ours);

    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sm);
    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};
    BTDA_matrices_fill_values(A_sparse, d_ones, B_sm, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(B_sm);
    free_matrix(A_spd);
    return 0;
}

/* BTA_sparse_matrices dispatcher: A is permuted_dense. Both routes produce
   PD; compare X / perms directly. */
const char *test_BTA_sparse_matrices_pd_A(void)
{
    /* B: 4x3 sparse_matrix. */
    CSR_matrix *B_csr = new_CSR_matrix(4, 3, 7);
    int Bp[5] = {0, 2, 3, 6, 7};
    int Bi[7] = {0, 2, 1, 0, 1, 2, 2};
    double Bx[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    /* A: 4x3 PD. */
    int A_rp[4] = {0, 1, 2, 3};
    int A_cp[3] = {0, 1, 2};
    double AX[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    matrix *A = new_permuted_dense(4, 3, 4, 3, A_rp, A_cp, AX);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with B native sparse, A native PD —
       routes through BTA_sparse_matrices -> BTA_csc_pd internally. */
    matrix *C_ours = BTA_matrices_alloc(A, B_sm);
    BTDA_matrices_fill_values(A, d, B_sm, C_ours);

    /* Route 2: same dispatcher with A flattened to sparse — routes
       through BTA_sparse_matrices -> BTA_alloc (csc-csc), a different
       kernel. Output type differs (sparse_matrix vs PD), so compare
       via to_csr. */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sm);
    BTDA_matrices_fill_values(A_sparse, d, B_sm, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A);
    free_matrix(B_sm);
    return 0;
}

/* Public dispatcher: B and A both sparse_matrix. Both routes go
   through the same csc-csc kernel, so the reference is a hand-computed
   dense matrix product. */
const char *test_BTA_sparse_matrices_csc_A(void)
{
    /* B: 4x3 sparse_matrix. */
    CSR_matrix *B_csr = new_CSR_matrix(4, 3, 7);
    int Bp[5] = {0, 2, 3, 6, 7};
    int Bi[7] = {0, 2, 1, 0, 1, 2, 2};
    double Bx[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    /* A: 4x3 sparse_matrix. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 6);
    int Ap[5] = {0, 1, 3, 5, 6};
    int Ai[6] = {1, 0, 2, 1, 2, 0};
    double Ax[6] = {10, 20, 30, 40, 50, 60};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1 (only route): public dispatcher with both operands native
       sparse — routes through BTA_sparse_matrices -> BTA_alloc (csc-csc). */
    matrix *C_ours = BTA_matrices_alloc(A_sm, B_sm);
    BTDA_matrices_fill_values(A_sm, d, B_sm, C_ours);

    /* Reference: hand-compute C = B^T @ diag(d) @ A as a dense 3x3 buffer.
       Both operands are 4x3 sparse — too small to be worth a second
       dispatcher route (nothing to flatten). */
    double B_dense[4 * 3] = {
        1, 0, 2, /* row 0 */
        0, 3, 0, /* row 1 */
        4, 5, 6, /* row 2 */
        0, 0, 7  /* row 3 */
    };
    double A_dense[4 * 3] = {
        0,  10, 0,  /* row 0 */
        20, 0,  30, /* row 1 */
        0,  40, 50, /* row 2 */
        60, 0,  0   /* row 3 */
    };
    double C_dense_ref[3 * 3] = {0};
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            double s = 0.0;
            for (int l = 0; l < 4; l++)
            {
                s += B_dense[l * 3 + i] * d[l] * A_dense[l * 3 + j];
            }
            C_dense_ref[i * 3 + j] = s;
        }
    }

    /* Scatter dispatcher's CSR output into a dense buffer for comparison. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    mu_assert("m", csr_ours->m == 3);
    mu_assert("n", csr_ours->n == 3);
    double C_dense_ours[3 * 3] = {0};
    for (int i = 0; i < csr_ours->m; i++)
    {
        for (int jj = csr_ours->p[i]; jj < csr_ours->p[i + 1]; jj++)
        {
            C_dense_ours[i * 3 + csr_ours->i[jj]] = csr_ours->x[jj];
        }
    }
    mu_assert("C dense", cmp_double_array(C_dense_ours, C_dense_ref, 9));

    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B_sm);
    return 0;
}

/* BTA_sparse_matrices dispatcher: A is stacked_pd. Our route returns
   stacked_pd (via BTA_csc_spd_alloc); the production route returns
   sparse_matrix (it materializes A->csc inside resolve_operand, then
   takes the csc-csc branch). Compare via to_csr. */
const char *test_BTA_sparse_matrices_spd_A(void)
{
    /* B: 4x3 sparse_matrix. */
    CSR_matrix *B_csr = new_CSR_matrix(4, 3, 7);
    int Bp[5] = {0, 2, 3, 6, 7};
    int Bi[7] = {0, 2, 1, 0, 1, 2, 2};
    double Bx[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    /* A: 4x3 spd, two blocks with overlapping col_perm (share col 2). */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    double d[4] = {2.0, -1.5, 0.5, 1.25};

    /* Route 1: public dispatcher with B native sparse, A native spd —
       routes through BTA_sparse_matrices -> BTA_csc_spd internally. */
    matrix *C_ours = BTA_matrices_alloc(A_spd, B_sm);
    BTDA_matrices_fill_values(A_spd, d, B_sm, C_ours);

    /* Route 2: same dispatcher with A flattened to sparse — routes
       through BTA_sparse_matrices -> BTA_alloc (csc-csc). */
    matrix *A_sparse = spd_to_sparse_matrix_copy(A_spd);
    matrix *C_ref = BTA_matrices_alloc(A_sparse, B_sm);
    BTDA_matrices_fill_values(A_sparse, d, B_sm, C_ref);

    /* Routes produce different output types — compare via to_csr. */
    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(A_sparse);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B_sm);
    return 0;
}

/* Top-level BTA_matrices_fill_values dispatcher. Each test runs Route 1 =
   BTA_matrices_alloc + BTA_matrices_fill_values and Route 2 = the same alloc +
   BTDA_matrices_fill_values with d = ones (B^T diag(1) A = B^T A). The two
   share structure, so we compare values via to_csr. Together the four cover all
   three B-branches and the special csc/csc (d == NULL) path. Recall
   BTA_matrices_alloc(A, B) computes B^T @ A, so the transposed operand goes
   second. */
const char *test_BTA_matrices_fill_pd_spd(void)
{
    /* B: 4x5 PD (transposed operand). */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    /* A: 4x3 spd, two blocks (share col 2). */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *A_blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 2};
    double A1X[4] = {5, 6, 7, 8};
    matrix *A_blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) A_blk0,
                                   (permuted_dense *) A_blk1};
    matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);

    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};

    matrix *C_ours = BTA_matrices_alloc(A_spd, B);
    BTA_matrices_fill_values(A_spd, B, C_ours);

    matrix *C_ref = BTA_matrices_alloc(A_spd, B);
    BTDA_matrices_fill_values(A_spd, d_ones, B, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(C_ours);
    free_matrix(A_spd);
    free_matrix(B);
    return 0;
}

const char *test_BTA_matrices_fill_spd_csc(void)
{
    /* B: 4x3 spd, two blocks (transposed operand). */
    int B0_rp[2] = {0, 1};
    int B0_cp[2] = {0, 2};
    double B0X[4] = {1, 2, 3, 4};
    matrix *B_blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
    int B1_rp[2] = {2, 3};
    int B1_cp[2] = {1, 2};
    double B1X[4] = {5, 6, 7, 8};
    matrix *B_blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
    permuted_dense *B_blocks[2] = {(permuted_dense *) B_blk0,
                                   (permuted_dense *) B_blk1};
    matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);

    /* A: 4x3 sparse_matrix. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 7);
    int Ap[5] = {0, 2, 3, 6, 7};
    int Ai[7] = {0, 2, 1, 0, 1, 2, 2};
    double Ax[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};

    matrix *C_ours = BTA_matrices_alloc(A_sm, B_spd);
    BTA_matrices_fill_values(A_sm, B_spd, C_ours);

    matrix *C_ref = BTA_matrices_alloc(A_sm, B_spd);
    BTDA_matrices_fill_values(A_sm, d_ones, B_spd, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B_spd);
    return 0;
}

const char *test_BTA_matrices_fill_csc_pd(void)
{
    /* B: 4x4 sparse_matrix (transposed operand). */
    CSR_matrix *B_csr = new_CSR_matrix(4, 4, 5);
    int Bp[5] = {0, 2, 3, 4, 5};
    int Bi[5] = {0, 2, 1, 0, 3};
    double Bx[5] = {10, 20, 30, 40, 50};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    /* A: 4x3 PD. */
    int A_rp[4] = {0, 1, 2, 3};
    int A_cp[3] = {0, 1, 2};
    double AX[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    matrix *A = new_permuted_dense(4, 3, 4, 3, A_rp, A_cp, AX);

    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};

    matrix *C_ours = BTA_matrices_alloc(A, B_sm);
    BTA_matrices_fill_values(A, B_sm, C_ours);

    matrix *C_ref = BTA_matrices_alloc(A, B_sm);
    BTDA_matrices_fill_values(A, d_ones, B_sm, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(C_ours);
    free_matrix(A);
    free_matrix(B_sm);
    return 0;
}

const char *test_BTA_matrices_fill_csc_csc(void)
{
    /* B: 4x3 sparse_matrix (transposed operand). */
    CSR_matrix *B_csr = new_CSR_matrix(4, 3, 7);
    int Bp[5] = {0, 2, 3, 6, 7};
    int Bi[7] = {0, 2, 1, 0, 1, 2, 2};
    double Bx[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    matrix *B_sm = new_sparse_matrix(B_csr);

    /* A: 4x4 sparse_matrix. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 4, 5);
    int Ap[5] = {0, 2, 3, 4, 5};
    int Ai[5] = {0, 2, 1, 0, 3};
    double Ax[5] = {10, 20, 30, 40, 50};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    double d_ones[4] = {1.0, 1.0, 1.0, 1.0};

    matrix *C_ours = BTA_matrices_alloc(A_sm, B_sm);
    BTA_matrices_fill_values(A_sm, B_sm, C_ours);

    matrix *C_ref = BTA_matrices_alloc(A_sm, B_sm);
    BTDA_matrices_fill_values(A_sm, d_ones, B_sm, C_ref);

    CSR_matrix *csr_ours = C_ours->to_csr(C_ours);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(C_ours);
    free_matrix(A_sm);
    free_matrix(B_sm);
    return 0;
}

/* Regression test for the kron-scratch / transpose-cache crash. Exercises
   BA_dense_kron_spd_fill_values with p > 1: the underlying kron_for_each_
   output_block reuses a single scratch PD across iterations, mutating
   its row_perm / col_perm in place. A prior implementation of
   BA_pd_spd_fill_values cached B's transpose on B->transpose_cache,
   which (a) was never populated for the fill-path scratch, and (b)
   would have held stale perms even if populated. BA_pd_spd still uses
   that cache, but the kron path stays correct because
   BA_dense_kron_spd bypasses the wrapper and calls BTA_pd_spd_*
   directly on its mutating scratch (see stacked_pd_linalg.c). The
   reference path runs the same dispatch with J flattened to
   sparse_matrix (which routes via BA_dense_kron_csc, no transpose-cache
   involvement); we compare via to_csr. */
const char *test_BA_pd_kron_spd_no_cache_staleness(void)
{
    /* A: 2x2 PD (kron-block size). */
    int A_rp[2] = {0, 1};
    int A_cp[2] = {0, 1};
    double AX[4] = {1, 2, 3, 4};
    matrix *A = new_permuted_dense(2, 2, 2, 2, A_rp, A_cp, AX);

    /* J: 6x3 spd with three blocks, one block per kron-stride so each
       iteration's scratch perms differ. p = 3 -> kron(I_3, A) is 6x6,
       J is 6x3, output is 6x3. */
    int J0_rp[2] = {0, 1};
    int J0_cp[2] = {0, 1};
    double J0X[4] = {1, 2, 3, 4};
    matrix *Jblk0 = new_permuted_dense(6, 3, 2, 2, J0_rp, J0_cp, J0X);
    int J1_rp[2] = {2, 3};
    int J1_cp[2] = {1, 2};
    double J1X[4] = {5, 6, 7, 8};
    matrix *Jblk1 = new_permuted_dense(6, 3, 2, 2, J1_rp, J1_cp, J1X);
    int J2_rp[2] = {4, 5};
    int J2_cp[2] = {0, 2};
    double J2X[4] = {9, 10, 11, 12};
    matrix *Jblk2 = new_permuted_dense(6, 3, 2, 2, J2_rp, J2_cp, J2X);
    permuted_dense *J_blocks[3] = {(permuted_dense *) Jblk0,
                                   (permuted_dense *) Jblk1,
                                   (permuted_dense *) Jblk2};
    matrix *J_spd = new_stacked_pd(6, 3, 3, J_blocks, NULL, NULL);

    int p = 3;

    /* Route 1: spd path through BA_pd_kron_spd. */
    matrix *C_spd = BA_dense_kron_matrices_alloc(A, p, J_spd);
    BA_dense_kron_matrices_fill_values(A, p, J_spd, C_spd);

    /* Route 2: sparse path via spd_to_sparse_matrix_copy + BA_dense_kron_csc. */
    matrix *J_sparse = spd_to_sparse_matrix_copy(J_spd);
    matrix *C_ref = BA_dense_kron_matrices_alloc(A, p, J_sparse);
    BA_dense_kron_matrices_fill_values(A, p, J_sparse, C_ref);

    /* Compare via to_csr — output structures may differ. */
    CSR_matrix *csr_spd = C_spd->to_csr(C_spd);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_spd->m == csr_ref->m);
    mu_assert("n", csr_spd->n == csr_ref->n);
    mu_assert("nnz", csr_spd->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_spd->p, csr_ref->p, csr_spd->m + 1));
    mu_assert("i", cmp_int_array(csr_spd->i, csr_ref->i, csr_spd->nnz));
    mu_assert("x", cmp_double_array(csr_spd->x, csr_ref->x, csr_spd->nnz));

    free_matrix(C_ref);
    free_matrix(J_sparse);
    free_matrix(C_spd);
    free_matrix(J_spd);
    free_matrix(A);
    return 0;
}

/* Only compiled with -DSP_TRACK_MEMORY=ON: the test reads the tracked
   allocator counters, which do not exist in a default build. */
#ifdef SP_TRACK_MEMORY

/* No-alloc-in-fill contract for the BTDA kernels via fill_is_alloc_free
   (tests/test_helpers.h). Covers BTDA_pd_pd (both the matching-row_perm and
   gather paths), BTDA_pd_spd, and the blockwise BTDA_spd_pd. */
typedef struct
{
    const permuted_dense *B;
    const double *d;
    const void *A;
    matrix *C;
} btda_fill_args;

static void run_BTDA_pd_pd(const void *ctx)
{
    const btda_fill_args *a = (const btda_fill_args *) ctx;
    BTDA_pd_pd_fill_values(a->B, a->d, (const permuted_dense *) a->A,
                           (permuted_dense *) a->C);
}

static void run_BTDA_pd_spd(const void *ctx)
{
    const btda_fill_args *a = (const btda_fill_args *) ctx;
    BTDA_pd_spd_fill_values(a->B, a->d, (const stacked_pd *) a->A,
                            (permuted_dense *) a->C);
}

typedef struct
{
    const stacked_pd *B;
    const double *d;
    const permuted_dense *A;
    stacked_pd *C;
} btda_spd_fill_args;

static void run_BTDA_spd_pd(const void *ctx)
{
    const btda_spd_fill_args *a = (const btda_spd_fill_args *) ctx;
    BTDA_spd_pd_fill_values(a->B, a->d, a->A, a->C);
}

const char *test_BTDA_fill_no_transient_alloc(void)
{
    double d[8] = {2.0, -1.5, 0.5, 1.25, 3.0, -0.5, 1.0, 2.5};

    /* --- pd_pd, matching row_perms (fast path) --- */
    {
        int row_perm[2] = {1, 3};
        int cp_A[2] = {0, 2};
        int cp_B[2] = {1, 3};
        double XA[4] = {1.0, 2.0, 3.0, 4.0};
        double XB[4] = {5.0, 6.0, 7.0, 8.0};
        matrix *A = new_permuted_dense(4, 4, 2, 2, row_perm, cp_A, XA);
        matrix *B = new_permuted_dense(4, 4, 2, 2, row_perm, cp_B, XB);
        matrix *C = BTA_pd_pd_alloc((permuted_dense *) B, (permuted_dense *) A);
        btda_fill_args args = {(permuted_dense *) B, d, A, C};
        run_BTDA_pd_pd(&args); /* warm-up */
        mu_assert("pd_pd fast path allocates in fill",
                  fill_is_alloc_free(run_BTDA_pd_pd, &args));
        free_matrix(C);
        free_matrix(B);
        free_matrix(A);
    }

    /* --- pd_pd, partial overlap (gather path) --- */
    {
        int rp_A[3] = {1, 3, 5};
        int rp_B[3] = {3, 5, 7};
        int cp_A[2] = {0, 2};
        int cp_B[2] = {1, 3};
        double XA[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        double XB[6] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
        matrix *A = new_permuted_dense(8, 4, 3, 2, rp_A, cp_A, XA);
        matrix *B = new_permuted_dense(8, 4, 3, 2, rp_B, cp_B, XB);
        matrix *C = BTA_pd_pd_alloc((permuted_dense *) B, (permuted_dense *) A);
        btda_fill_args args = {(permuted_dense *) B, d, A, C};
        run_BTDA_pd_pd(&args);
        mu_assert("pd_pd gather path allocates in fill",
                  fill_is_alloc_free(run_BTDA_pd_pd, &args));
        free_matrix(C);
        free_matrix(B);
        free_matrix(A);
    }

    /* --- pd_spd (two-block A, both kept) --- */
    {
        int A0_rp[2] = {0, 1};
        int A0_cp[2] = {0, 2};
        double A0X[4] = {1, 2, 3, 4};
        matrix *blk0 = new_permuted_dense(4, 3, 2, 2, A0_rp, A0_cp, A0X);
        int A1_rp[2] = {2, 3};
        int A1_cp[2] = {1, 2};
        double A1X[4] = {5, 6, 7, 8};
        matrix *blk1 = new_permuted_dense(4, 3, 2, 2, A1_rp, A1_cp, A1X);
        permuted_dense *A_blocks[2] = {(permuted_dense *) blk0,
                                       (permuted_dense *) blk1};
        matrix *A_spd = new_stacked_pd(4, 3, 2, A_blocks, NULL, NULL);
        int B_rp[3] = {0, 1, 2};
        int B_cp[3] = {1, 3, 4};
        double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        matrix *B = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);
        matrix *C = BTA_pd_spd_alloc((permuted_dense *) B, (stacked_pd *) A_spd);
        btda_fill_args args = {(permuted_dense *) B, d, A_spd, C};
        run_BTDA_pd_spd(&args);
        mu_assert("pd_spd allocates in fill",
                  fill_is_alloc_free(run_BTDA_pd_spd, &args));
        free_matrix(C);
        free_matrix(B);
        free_matrix(A_spd);
    }

    /* --- spd_pd (blockwise, overlapping col_perms) --- */
    {
        int B0_rp[2] = {0, 1};
        int B0_cp[2] = {0, 2};
        double B0X[4] = {1, 2, 3, 4};
        matrix *blk0 = new_permuted_dense(4, 3, 2, 2, B0_rp, B0_cp, B0X);
        int B1_rp[2] = {2, 3};
        int B1_cp[2] = {1, 2};
        double B1X[4] = {5, 6, 7, 8};
        matrix *blk1 = new_permuted_dense(4, 3, 2, 2, B1_rp, B1_cp, B1X);
        permuted_dense *B_blocks[2] = {(permuted_dense *) blk0,
                                       (permuted_dense *) blk1};
        matrix *B_spd = new_stacked_pd(4, 3, 2, B_blocks, NULL, NULL);
        int A_rp[4] = {0, 1, 2, 3};
        int A_cp[3] = {0, 1, 2};
        double AX[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        matrix *A = new_permuted_dense(4, 3, 4, 3, A_rp, A_cp, AX);
        matrix *C = BTA_spd_pd_alloc((stacked_pd *) B_spd, (permuted_dense *) A);
        btda_spd_fill_args args = {(stacked_pd *) B_spd, d, (permuted_dense *) A,
                                   (stacked_pd *) C};
        run_BTDA_spd_pd(&args);
        mu_assert("spd_pd blockwise allocates in fill",
                  fill_is_alloc_free(run_BTDA_spd_pd, &args));
        free_matrix(C);
        free_matrix(B_spd);
        free_matrix(A);
    }

    return 0;
}

#endif /* SP_TRACK_MEMORY */

/* BA_pd_spd transpose cache: the fill refreshes B's cached transpose iff
   B's values_version moved since the last fill. Fill once, mutate B's
   values + bump, refill, and compare against a fresh computation with the
   mutated values. */
const char *test_BA_pd_spd_transpose_cache_refresh(void)
{
    /* B: 3x4 pd with a non-square 2x3 block. */
    int B_rp[2] = {0, 2};
    int B_cp[3] = {0, 1, 3};
    double BX[6] = {1, 2, 3, 4, 5, 6};
    matrix *B_m = new_permuted_dense(3, 4, 2, 3, B_rp, B_cp, BX);
    permuted_dense *B = (permuted_dense *) B_m;

    /* A: 4x5 spd with two disjoint-row blocks. */
    int A0_rp[2] = {0, 1};
    int A0_cp[2] = {0, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *Ablk0 = new_permuted_dense(4, 5, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[2] = {2, 3};
    int A1_cp[2] = {1, 4};
    double A1X[4] = {5, 6, 7, 8};
    matrix *Ablk1 = new_permuted_dense(4, 5, 2, 2, A1_rp, A1_cp, A1X);
    permuted_dense *A_blocks[2] = {(permuted_dense *) Ablk0,
                                   (permuted_dense *) Ablk1};
    matrix *A_spd = new_stacked_pd(4, 5, 2, A_blocks, NULL, NULL);
    stacked_pd *A = (stacked_pd *) A_spd;

    matrix *C = BA_pd_spd_alloc(B, A);
    BA_pd_spd_fill_values(B, A, (permuted_dense *) C);
    mu_assert("seen must match after fill",
              B->transpose_cache->transpose_seen == B->base.values_version);

    /* Mutate B's values and bump; the guarded refill must refresh the
       cached transpose. */
    double BX2[6] = {-1, 7, 0.5, 2, -3, 6};
    memcpy(B->X, BX2, 6 * sizeof(double));
    matrix_values_changed(B_m);
    BA_pd_spd_fill_values(B, A, (permuted_dense *) C);
    mu_assert("seen must catch up after bump + refill",
              B->transpose_cache->transpose_seen == B->base.values_version);

    /* Reference: fresh B with the mutated values, fresh cache. */
    matrix *B2_m = new_permuted_dense(3, 4, 2, 3, B_rp, B_cp, BX2);
    permuted_dense *B2 = (permuted_dense *) B2_m;
    matrix *C_ref = BA_pd_spd_alloc(B2, A);
    BA_pd_spd_fill_values(B2, A, (permuted_dense *) C_ref);

    CSR_matrix *csr_ours = C->to_csr(C);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B2_m);
    free_matrix(C);
    free_matrix(A_spd);
    free_matrix(B_m);
    return 0;
}

/* BTA_pd_matrices with a sparse A: the fill reads B^T from B's transpose
   cache, refreshed iff B's values_version moved. Fill, refill untouched (cache
   reused, result unchanged), mutate B's values + bump, refill, and compare
   against a fresh computation with the mutated values. */
const char *test_BTA_pd_csc_transpose_cache_refresh(void)
{
    /* B: 4x5 PD. */
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B_m = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);
    permuted_dense *B = (permuted_dense *) B_m;

    /* A: 4x3 sparse_matrix wrapping a CSR. */
    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 7);
    int Ap[5] = {0, 2, 3, 6, 7};
    int Ai[7] = {0, 2, 1, 0, 1, 2, 2};
    double Ax[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    matrix *C = BTA_matrices_alloc(A_sm, B_m);
    mu_assert("alloc builds the cache", B->transpose_cache != NULL);
    BTA_matrices_fill_values(A_sm, B_m, C);
    mu_assert("seen must match after fill",
              B->transpose_cache->transpose_seen == B->base.values_version);

    /* Refill with B untouched: the cache is reused and the result unchanged. */
    double *first = (double *) sp_malloc(C->nnz * sizeof(double));
    memcpy(first, C->x, C->nnz * sizeof(double));
    BTA_matrices_fill_values(A_sm, B_m, C);
    mu_assert("refill with unchanged B", cmp_double_array(C->x, first, C->nnz));
    sp_free(first);

    /* Mutate B's values and bump; the guarded refill must refresh the cache. */
    double BX2[9] = {-1, 7, 0.5, 2, -3, 6, 0, 1.5, -4};
    memcpy(B->X, BX2, sizeof(BX2));
    matrix_values_changed(B_m);
    BTA_matrices_fill_values(A_sm, B_m, C);
    mu_assert("seen must catch up after bump + refill",
              B->transpose_cache->transpose_seen == B->base.values_version);

    /* Reference: fresh B with the mutated values, fresh cache. */
    matrix *B2_m = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX2);
    matrix *C_ref = BTA_matrices_alloc(A_sm, B2_m);
    BTA_matrices_fill_values(A_sm, B2_m, C_ref);

    CSR_matrix *csr_ours = C->to_csr(C);
    CSR_matrix *csr_ref = C_ref->to_csr(C_ref);
    mu_assert("m", csr_ours->m == csr_ref->m);
    mu_assert("n", csr_ours->n == csr_ref->n);
    mu_assert("nnz", csr_ours->nnz == csr_ref->nnz);
    mu_assert("p", cmp_int_array(csr_ours->p, csr_ref->p, csr_ours->m + 1));
    mu_assert("i", cmp_int_array(csr_ours->i, csr_ref->i, csr_ours->nnz));
    mu_assert("x", cmp_double_array(csr_ours->x, csr_ref->x, csr_ours->nnz));

    free_matrix(C_ref);
    free_matrix(B2_m);
    free_matrix(C);
    free_matrix(A_sm);
    free_matrix(B_m);
    return 0;
}

#ifdef SP_TRACK_MEMORY
typedef struct
{
    matrix *A;
    matrix *B;
    matrix *C;
} bta_pd_csc_args;

static void run_BTA_pd_csc(const void *ctx)
{
    const bta_pd_csc_args *a = (const bta_pd_csc_args *) ctx;
    BTA_matrices_fill_values(a->A, a->B, a->C);
}

/* The transpose cache is built at alloc time, so the PD x sparse BTA fill
   stays allocation-free. */
const char *test_BTA_pd_csc_fill_no_transient_alloc(void)
{
    int B_rp[3] = {0, 1, 2};
    int B_cp[3] = {1, 3, 4};
    double BX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *B_m = new_permuted_dense(4, 5, 3, 3, B_rp, B_cp, BX);

    CSR_matrix *A_csr = new_CSR_matrix(4, 3, 7);
    int Ap[5] = {0, 2, 3, 6, 7};
    int Ai[7] = {0, 2, 1, 0, 1, 2, 2};
    double Ax[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    matrix *C = BTA_matrices_alloc(A_sm, B_m);
    bta_pd_csc_args args = {A_sm, B_m, C};
    run_BTA_pd_csc(&args); /* warm-up */
    mu_assert("pd x csc BTA fill allocates",
              fill_is_alloc_free(run_BTA_pd_csc, &args));

    free_matrix(C);
    free_matrix(A_sm);
    free_matrix(B_m);
    return 0;
}
#endif /* SP_TRACK_MEMORY */

/* Dense row-major copy of a handle's current values, read through to_csr. */
static double *dense_of(matrix *M)
{
    CSR_matrix *csr = M->to_csr(M);
    double *out = (double *) sp_calloc((size_t) M->m * M->n, sizeof(double));
    for (int i = 0; i < csr->m; i++)
    {
        for (int jj = csr->p[i]; jj < csr->p[i + 1]; jj++)
        {
            out[i * csr->n + csr->i[jj]] += csr->x[jj];
        }
    }
    return out;
}

/* Build the mixed-kind fixture: A sparse (6x5), B permuted_dense, D stacked_pd
   (two blocks), E sparse with an empty pattern (the affine-child case). */
static void s4_fixture(matrix **A, matrix **B, matrix **D, matrix **E)
{
    CSR_matrix *A_csr = new_CSR_matrix(6, 5, 7);
    int Ap[7] = {0, 2, 3, 3, 5, 6, 7};
    int Ai[7] = {0, 4, 2, 1, 3, 0, 4};
    double Ax[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    *A = new_sparse_matrix(A_csr);

    int B_rp[2] = {1, 4};
    int B_cp[3] = {0, 2, 4};
    double BX[6] = {10, 20, 30, 40, 50, 60};
    *B = new_permuted_dense(6, 5, 2, 3, B_rp, B_cp, BX);

    int D0_rp[2] = {0, 2};
    int D0_cp[2] = {1, 4};
    double D0X[4] = {100, 200, 300, 400};
    matrix *blk0 = new_permuted_dense(6, 5, 2, 2, D0_rp, D0_cp, D0X);
    int D1_rp[1] = {5};
    int D1_cp[3] = {0, 3, 4};
    double D1X[3] = {500, 600, 700};
    matrix *blk1 = new_permuted_dense(6, 5, 1, 3, D1_rp, D1_cp, D1X);
    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    *D = new_stacked_pd(6, 5, 2, blocks, NULL, NULL);

    *E = new_sparse_matrix_alloc(6, 5, 0);
    CSR_matrix *E_csr = (*E)->to_csr(*E);
    memset(E_csr->p, 0, 7 * sizeof(int));
}

/* sum_4_matrices: mixed operand kinds (sparse, permuted_dense, stacked_pd read
   block-major through its re-indexed map, and an empty sparse) against a
   dense reference, before and after mutating an operand's values. */
const char *test_sum_4_matrices_mixed_kinds(void)
{
    matrix *A, *B, *D, *E;
    s4_fixture(&A, &B, &D, &E);

    matrix *C = sum_4_matrices_alloc(A, B, D, E);
    mu_assert("C is sparse", C->kind == MATRIX_SPARSE);
    mu_assert("C shape", C->m == 6 && C->n == 5);
    mu_assert("C pattern valid", csr_is_valid(C->to_csr(C)));

    for (int round = 0; round < 2; round++)
    {
        if (round == 1)
        {
            /* mutate the stacked_pd's values (block-major buffer) and B's */
            for (int ii = 0; ii < D->nnz; ii++) D->x[ii] = -D->x[ii];
            matrix_values_changed(D);
            for (int ii = 0; ii < B->nnz; ii++) B->x[ii] *= 0.5;
            matrix_values_changed(B);
        }
        sum_4_matrices_fill_values(A, B, D, E, C);

        double *ref = dense_of(A);
        double *dB = dense_of(B);
        double *dD = dense_of(D);
        double *dE = dense_of(E);
        double *got = dense_of(C);
        for (int k = 0; k < 30; k++) ref[k] += dB[k] + dD[k] + dE[k];
        mu_assert("sum matches dense reference", cmp_double_array(got, ref, 30));
        sp_free(ref);
        sp_free(dB);
        sp_free(dD);
        sp_free(dE);
        sp_free(got);
    }

    free_matrix(C);
    free_matrix(E);
    free_matrix(D);
    free_matrix(B);
    free_matrix(A);
    return 0;
}

/* sum_n_matrices with n = 4 on the sum_4 fixture: same pattern and values as
   sum_4_matrices, and both match a dense reference before and after mutating
   an operand's values. */
const char *test_sum_n_matrices_mixed_kinds(void)
{
    matrix *A, *B, *D, *E;
    s4_fixture(&A, &B, &D, &E);
    matrix *ops[4] = {A, B, D, E};

    matrix *C = sum_n_matrices_alloc(ops, 4);
    matrix *C4 = sum_4_matrices_alloc(A, B, D, E);
    mu_assert("C is sparse", C->kind == MATRIX_SPARSE);
    mu_assert("C shape", C->m == 6 && C->n == 5);
    CSR_matrix *csr = C->to_csr(C);
    CSR_matrix *csr4 = C4->to_csr(C4);
    mu_assert("C pattern valid", csr_is_valid(csr));
    mu_assert("nnz matches sum_4", csr->nnz == csr4->nnz);
    mu_assert("p matches sum_4", cmp_int_array(csr->p, csr4->p, 7));
    mu_assert("i matches sum_4", cmp_int_array(csr->i, csr4->i, csr->nnz));

    for (int round = 0; round < 2; round++)
    {
        if (round == 1)
        {
            for (int ii = 0; ii < D->nnz; ii++) D->x[ii] = -D->x[ii];
            matrix_values_changed(D);
            for (int ii = 0; ii < B->nnz; ii++) B->x[ii] *= 0.5;
            matrix_values_changed(B);
        }
        sum_n_matrices_fill_values(ops, 4, C);
        sum_4_matrices_fill_values(A, B, D, E, C4);
        mu_assert("values match sum_4", cmp_double_array(C->x, C4->x, C->nnz));

        double *ref = dense_of(A);
        double *dB = dense_of(B);
        double *dD = dense_of(D);
        double *dE = dense_of(E);
        double *got = dense_of(C);
        for (int k = 0; k < 30; k++) ref[k] += dB[k] + dD[k] + dE[k];
        mu_assert("sum matches dense reference", cmp_double_array(got, ref, 30));
        sp_free(ref);
        sp_free(dB);
        sp_free(dD);
        sp_free(dE);
        sp_free(got);
    }

    free_matrix(C4);
    free_matrix(C);
    free_matrix(E);
    free_matrix(D);
    free_matrix(B);
    free_matrix(A);
    return 0;
}

static matrix *sn_sparse(int m, int n, int nnz, const int *p, const int *i,
                         const double *x)
{
    CSR_matrix *csr = new_CSR_matrix(m, n, nnz);
    memcpy(csr->p, p, (size_t) (m + 1) * sizeof(int));
    memcpy(csr->i, i, (size_t) nnz * sizeof(int));
    memcpy(csr->x, x, (size_t) nnz * sizeof(double));
    return new_sparse_matrix(csr);
}

/* sum_n_matrices at n = 1 (C copies the operand) and n = 5 (4x6 sparse
   operands whose columns interleave across operands, one passed twice, one
   empty, and a row empty in all of them) against a dense reference. */
const char *test_sum_n_matrices_n1_n5(void)
{
    int p0[5] = {0, 2, 3, 3, 4};
    int i0[4] = {1, 5, 2, 0};
    double x0[4] = {1, 2, 3, 4};
    int p1[5] = {0, 1, 3, 3, 3};
    int i1[3] = {0, 2, 4};
    double x1[3] = {5, 6, 7};
    int p4[5] = {0, 1, 1, 1, 3};
    int i4[3] = {5, 1, 5};
    double x4[3] = {8, 9, 10};
    matrix *M0 = sn_sparse(4, 6, 4, p0, i0, x0);
    matrix *M1 = sn_sparse(4, 6, 3, p1, i1, x1);
    matrix *Me = new_sparse_matrix_alloc(4, 6, 0);
    memset(Me->to_csr(Me)->p, 0, 5 * sizeof(int));
    matrix *M4 = sn_sparse(4, 6, 3, p4, i4, x4);

    /* n = 1 */
    matrix *ops1[1] = {M0};
    matrix *C1 = sum_n_matrices_alloc(ops1, 1);
    sum_n_matrices_fill_values(ops1, 1, C1);
    mu_assert("n=1 pattern", cmp_sparsity(C1, p0, i0, 4, 4));
    mu_assert("n=1 values", cmp_values(C1, x0, 4));

    /* n = 5 */
    matrix *ops5[5] = {M0, M1, M0, Me, M4};
    matrix *C5 = sum_n_matrices_alloc(ops5, 5);
    sum_n_matrices_fill_values(ops5, 5, C5);
    mu_assert("n=5 pattern valid", csr_is_valid(C5->to_csr(C5)));
    int exp_p[5] = {0, 3, 5, 5, 8};
    int exp_i[8] = {0, 1, 5, 2, 4, 0, 1, 5};
    mu_assert("n=5 pattern", cmp_sparsity(C5, exp_p, exp_i, 4, 8));
    double *ref = (double *) sp_calloc(24, sizeof(double));
    for (int k = 0; k < 5; k++)
    {
        double *d = dense_of(ops5[k]);
        for (int c = 0; c < 24; c++) ref[c] += d[c];
        sp_free(d);
    }
    double *got = dense_of(C5);
    mu_assert("n=5 matches dense reference", cmp_double_array(got, ref, 24));
    sp_free(got);
    sp_free(ref);

    free_matrix(C5);
    free_matrix(C1);
    free_matrix(M4);
    free_matrix(Me);
    free_matrix(M1);
    free_matrix(M0);
    return 0;
}

/* sum_n_matrices_fill_zero + fill_add on the sum_4 fixture: adding the operands
   one by one matches fill_values, and an operand overwritten after it was added
   (as a shared subexpression's later evaluation would) does not leak in. */
const char *test_sum_n_matrices_fill_add(void)
{
    matrix *A, *B, *D, *E;
    s4_fixture(&A, &B, &D, &E);
    matrix *ops[4] = {A, B, D, E};
    matrix *C = sum_n_matrices_alloc(ops, 4);
    matrix *C_ref = sum_n_matrices_alloc(ops, 4);
    sum_n_matrices_fill_values(ops, 4, C_ref);

    /* one operand at a time equals the one-shot fill */
    sum_n_matrices_fill_zero(C);
    for (int k = 0; k < 4; k++) sum_n_matrices_fill_add(ops[k], k, C);
    mu_assert("incremental matches fill_values",
              cmp_double_array(C->x, C_ref->x, C->nnz));

    /* reference from A's values before the overwrite */
    double *ref = dense_of(A);
    double *dB = dense_of(B);
    double *dD = dense_of(D);
    double *dE = dense_of(E);
    for (int c = 0; c < 30; c++) ref[c] += dB[c] + dD[c] + dE[c];

    sum_n_matrices_fill_zero(C);
    sum_n_matrices_fill_add(A, 0, C);
    for (int ii = 0; ii < A->nnz; ii++) A->x[ii] = -A->x[ii];
    matrix_values_changed(A);
    for (int k = 1; k < 4; k++) sum_n_matrices_fill_add(ops[k], k, C);
    double *got = dense_of(C);
    mu_assert("overwrite after add does not leak in",
              cmp_double_array(got, ref, 30));

    sp_free(got);
    sp_free(ref);
    sp_free(dB);
    sp_free(dD);
    sp_free(dE);
    free_matrix(C_ref);
    free_matrix(C);
    free_matrix(E);
    free_matrix(D);
    free_matrix(B);
    free_matrix(A);
    return 0;
}

#ifdef SP_TRACK_MEMORY
typedef struct
{
    matrix *A, *B, *D, *E, *C;
} s4_fill_args;

static void run_sum_4_matrices(const void *ctx)
{
    const s4_fill_args *a = (const s4_fill_args *) ctx;
    sum_4_matrices_fill_values(a->A, a->B, a->D, a->E, a->C);
}

/* The maps are bound at alloc time, so the fill stays allocation-free. */
const char *test_sum_4_matrices_fill_no_transient_alloc(void)
{
    matrix *A, *B, *D, *E;
    s4_fixture(&A, &B, &D, &E);
    matrix *C = sum_4_matrices_alloc(A, B, D, E);
    s4_fill_args args = {A, B, D, E, C};
    run_sum_4_matrices(&args); /* warm-up */
    mu_assert("sum_4_matrices fill allocates",
              fill_is_alloc_free(run_sum_4_matrices, &args));
    free_matrix(C);
    free_matrix(E);
    free_matrix(D);
    free_matrix(B);
    free_matrix(A);
    return 0;
}
#endif /* SP_TRACK_MEMORY */

/* BA_matrices with a sparse B (3x4): row 0 hits k = {0, 2}, row 1 hits
   k = {1, 3}, row 2 hits only k = 0. The PD and spd fixtures below leave
   row 0 of A structurally empty, so B's row 2 is dropped from a PD / spd
   output. */
static matrix *ba_sparse_B(void)
{
    CSR_matrix *B_csr = new_CSR_matrix(3, 4, 5);
    int Bp[4] = {0, 2, 4, 5};
    int Bi[5] = {0, 2, 1, 3, 0};
    double Bx[5] = {1, 2, 3, -1, 0.5};
    memcpy(B_csr->p, Bp, sizeof(Bp));
    memcpy(B_csr->i, Bi, sizeof(Bi));
    memcpy(B_csr->x, Bx, sizeof(Bx));
    return new_sparse_matrix(B_csr);
}

/* Compare C against a dense triple-loop reference of B A read through
   dense_of. Returns 0 or the failing message. */
static const char *ba_matches_dense(matrix *B, matrix *A, matrix *C, const char *msg)
{
    int m = B->m, k = B->n, n = A->n;
    double *Bd = dense_of(B);
    double *Ad = dense_of(A);
    double *Cd = dense_of(C);
    double *ref = (double *) sp_calloc((size_t) m * n, sizeof(double));
    for (int i = 0; i < m; i++)
        for (int l = 0; l < k; l++)
            for (int j = 0; j < n; j++)
                ref[i * n + j] += Bd[i * k + l] * Ad[l * n + j];
    int ok = C->m == m && C->n == n && cmp_double_array(Cd, ref, m * n);
    sp_free(ref);
    sp_free(Cd);
    sp_free(Ad);
    sp_free(Bd);
    return ok ? 0 : msg;
}

/* Rewrite every stored value of M and bump its version. */
static void ba_mutate(matrix *M)
{
    for (int ii = 0; ii < M->nnz; ii++)
    {
        M->x[ii] = -2.0 * M->x[ii] + 1.0;
    }
    matrix_values_changed(M);
}

/* sparse B x PD A -> PD C, refilled after mutating B. */
const char *test_BA_sparse_matrices_pd_A(void)
{
    matrix *B = ba_sparse_B();
    int A_rp[3] = {1, 2, 3};
    int A_cp[3] = {0, 2, 4};
    double AX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *A = new_permuted_dense(4, 5, 3, 3, A_rp, A_cp, AX);

    matrix *C = BA_matrices_alloc(B, A);
    mu_assert("sparse x pd gives pd", C->kind == MATRIX_PERMUTED_DENSE);
    mu_assert("row 2 of B dropped", ((permuted_dense *) C)->m0 == 2);
    BA_matrices_fill_values(B, A, C);
    const char *err = ba_matches_dense(B, A, C, "sparse x pd first fill");
    mu_assert(err ? err : "", err == 0);

    ba_mutate(B);
    BA_matrices_fill_values(B, A, C);
    err = ba_matches_dense(B, A, C, "sparse x pd refill after mutating B");
    mu_assert(err ? err : "", err == 0);

    free_matrix(C);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* sparse B x spd A (two blocks sharing column 2) -> spd C, refilled after
   mutating B. */
const char *test_BA_sparse_matrices_spd_A(void)
{
    matrix *B = ba_sparse_B();
    int A0_rp[2] = {1, 2};
    int A0_cp[2] = {1, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 5, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[1] = {3};
    int A1_cp[2] = {2, 4};
    double A1X[2] = {5, 6};
    matrix *blk1 = new_permuted_dense(4, 5, 1, 2, A1_rp, A1_cp, A1X);
    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A = new_stacked_pd(4, 5, 2, blocks, NULL, NULL);

    matrix *C = BA_matrices_alloc(B, A);
    mu_assert("sparse x spd gives spd", C->kind == MATRIX_STACKED_PD);
    BA_matrices_fill_values(B, A, C);
    const char *err = ba_matches_dense(B, A, C, "sparse x spd first fill");
    mu_assert(err ? err : "", err == 0);

    ba_mutate(B);
    BA_matrices_fill_values(B, A, C);
    err = ba_matches_dense(B, A, C, "sparse x spd refill after mutating B");
    mu_assert(err ? err : "", err == 0);

    free_matrix(C);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

/* sparse B x sparse A -> sparse C, refilled after mutating both operands (A's
   CSC mirror must refresh). */
const char *test_BA_sparse_matrices_csc_A(void)
{
    matrix *B = ba_sparse_B();
    CSR_matrix *A_csr = new_CSR_matrix(4, 5, 6);
    int Ap[5] = {0, 2, 3, 5, 6};
    int Ai[6] = {1, 4, 2, 0, 3, 4};
    double Ax[6] = {1, 2, 3, 4, 5, 6};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A = new_sparse_matrix(A_csr);

    matrix *C = BA_matrices_alloc(B, A);
    mu_assert("sparse x sparse gives sparse", C->kind == MATRIX_SPARSE);
    BA_matrices_fill_values(B, A, C);
    const char *err = ba_matches_dense(B, A, C, "sparse x sparse first fill");
    mu_assert(err ? err : "", err == 0);

    ba_mutate(B);
    ba_mutate(A);
    BA_matrices_fill_values(B, A, C);
    err = ba_matches_dense(B, A, C, "sparse x sparse refill after mutating B, A");
    mu_assert(err ? err : "", err == 0);

    free_matrix(C);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

#ifdef SP_TRACK_MEMORY
typedef struct
{
    matrix *B;
    matrix *A;
    matrix *C;
} ba_fill_args;

static void run_BA_matrices(const void *ctx)
{
    const ba_fill_args *a = (const ba_fill_args *) ctx;
    BA_matrices_fill_values(a->B, a->A, a->C);
}

/* The sparse-B fills read B in place and A through caches the alloc built, so
   none of the three cells allocates. */
const char *test_BA_sparse_fill_no_transient_alloc(void)
{
    matrix *B = ba_sparse_B();

    int A_rp[3] = {1, 2, 3};
    int A_cp[3] = {0, 2, 4};
    double AX[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix *A_pd = new_permuted_dense(4, 5, 3, 3, A_rp, A_cp, AX);

    int A0_rp[2] = {1, 2};
    int A0_cp[2] = {1, 2};
    double A0X[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(4, 5, 2, 2, A0_rp, A0_cp, A0X);
    int A1_rp[1] = {3};
    int A1_cp[2] = {2, 4};
    double A1X[2] = {5, 6};
    matrix *blk1 = new_permuted_dense(4, 5, 1, 2, A1_rp, A1_cp, A1X);
    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    matrix *A_spd = new_stacked_pd(4, 5, 2, blocks, NULL, NULL);

    CSR_matrix *A_csr = new_CSR_matrix(4, 5, 6);
    int Ap[5] = {0, 2, 3, 5, 6};
    int Ai[6] = {1, 4, 2, 0, 3, 4};
    double Ax[6] = {1, 2, 3, 4, 5, 6};
    memcpy(A_csr->p, Ap, sizeof(Ap));
    memcpy(A_csr->i, Ai, sizeof(Ai));
    memcpy(A_csr->x, Ax, sizeof(Ax));
    matrix *A_sm = new_sparse_matrix(A_csr);

    matrix *As[3] = {A_pd, A_spd, A_sm};
    const char *msgs[3] = {"sparse x pd BA fill allocates",
                           "sparse x spd BA fill allocates",
                           "sparse x sparse BA fill allocates"};
    for (int t = 0; t < 3; t++)
    {
        matrix *C = BA_matrices_alloc(B, As[t]);
        ba_fill_args args = {B, As[t], C};
        run_BA_matrices(&args); /* warm-up */
        int ok = fill_is_alloc_free(run_BA_matrices, &args);
        free_matrix(C);
        mu_assert(msgs[t], ok);
    }

    free_matrix(A_sm);
    free_matrix(A_spd);
    free_matrix(A_pd);
    free_matrix(B);
    return 0;
}
#endif /* SP_TRACK_MEMORY */

/* BA_dense_kron_I_matrices: C = kron(A, I_m) @ J against a dense reference.
   A is a 2x3 full PD (p = 2, q = 3) and m = 2, so J is 6 x 5 and C is 4 x 5.
   The three J fixtures: a sparse J; a PD J whose rows 0, 2, 4 are empty, so
   block i = 0 of C is dropped; an spd J with two blocks sharing column 2. */
static matrix *kron_I_A(void)
{
    double AX[6] = {1, 2, 3, 4, 5, 6};
    return new_permuted_dense_full(2, 3, AX);
}

static matrix *kron_I_J_sparse(void)
{
    CSR_matrix *J = new_CSR_matrix(6, 5, 7);
    int Jp[7] = {0, 1, 3, 3, 5, 6, 7};
    int Ji[7] = {1, 0, 3, 2, 4, 1, 0};
    double Jx[7] = {1, 2, 3, 4, 5, 6, 7};
    memcpy(J->p, Jp, sizeof(Jp));
    memcpy(J->i, Ji, sizeof(Ji));
    memcpy(J->x, Jx, sizeof(Jx));
    return new_sparse_matrix(J);
}

static matrix *kron_I_J_pd(void)
{
    int rp[3] = {1, 3, 5};
    int cp[3] = {0, 2, 4};
    double X[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    return new_permuted_dense(6, 5, 3, 3, rp, cp, X);
}

static matrix *kron_I_J_spd(void)
{
    int rp0[2] = {0, 1};
    int cp0[2] = {1, 2};
    double X0[4] = {1, 2, 3, 4};
    matrix *blk0 = new_permuted_dense(6, 5, 2, 2, rp0, cp0, X0);
    int rp1[2] = {3, 5};
    int cp1[2] = {2, 4};
    double X1[4] = {5, 6, 7, 8};
    matrix *blk1 = new_permuted_dense(6, 5, 2, 2, rp1, cp1, X1);
    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    return new_stacked_pd(6, 5, 2, blocks, NULL, NULL);
}

/* Returns 0 or the failing message. */
static const char *kron_I_matches_dense(matrix *A, int m, matrix *J, matrix *C,
                                        const char *msg)
{
    int p = A->m, q = A->n, n = J->n;
    int Km = p * m, Kn = q * m;
    double *Kd = (double *) sp_calloc((size_t) Km * Kn, sizeof(double));
    for (int c = 0; c < p; c++)
        for (int j = 0; j < q; j++)
            for (int i = 0; i < m; i++)
                Kd[(c * m + i) * Kn + (j * m + i)] = A->x[c * q + j];
    double *Jd = dense_of(J);
    double *Cd = dense_of(C);
    double *ref = (double *) sp_calloc((size_t) Km * n, sizeof(double));
    for (int r = 0; r < Km; r++)
        for (int l = 0; l < Kn; l++)
            for (int col = 0; col < n; col++)
                ref[r * n + col] += Kd[r * Kn + l] * Jd[l * n + col];
    int ok = C->kind == MATRIX_STACKED_PD && C->m == Km && C->n == n &&
             cmp_double_array(Cd, ref, Km * n);
    sp_free(ref);
    sp_free(Cd);
    sp_free(Jd);
    sp_free(Kd);
    return ok ? 0 : msg;
}

/* Refilled after mutating A and J (J's mirror must refresh). */
const char *test_BA_dense_kron_I_sparse_J(void)
{
    matrix *A = kron_I_A();
    matrix *J = kron_I_J_sparse();
    matrix *C = BA_dense_kron_I_matrices_alloc(A, 2, J);
    /* Structure, which a dense reference cannot see: block 0 reads J rows
       {0, 2, 4} (columns {1}), block 1 rows {1, 3, 5} (columns {0, 2, 3, 4}). */
    mu_assert("kron_I sparse pattern is tight", C->nnz == 2 * 1 + 2 * 4);
    BA_dense_kron_I_matrices_fill_values(A, 2, J, C);
    const char *err = kron_I_matches_dense(A, 2, J, C, "kron_I sparse first fill");
    mu_assert(err ? err : "", err == 0);

    ba_mutate(A);
    ba_mutate(J);
    BA_dense_kron_I_matrices_fill_values(A, 2, J, C);
    err =
        kron_I_matches_dense(A, 2, J, C, "kron_I sparse refill after mutating A, J");
    mu_assert(err ? err : "", err == 0);

    free_matrix(C);
    free_matrix(J);
    free_matrix(A);
    return 0;
}

const char *test_BA_dense_kron_I_pd_J(void)
{
    matrix *A = kron_I_A();
    matrix *J = kron_I_J_pd();
    matrix *C = BA_dense_kron_I_matrices_alloc(A, 2, J);
    mu_assert("block 0 dropped", ((stacked_pd *) C)->n_blocks == 1);
    BA_dense_kron_I_matrices_fill_values(A, 2, J, C);
    const char *err = kron_I_matches_dense(A, 2, J, C, "kron_I pd first fill");
    mu_assert(err ? err : "", err == 0);

    ba_mutate(A);
    BA_dense_kron_I_matrices_fill_values(A, 2, J, C);
    err = kron_I_matches_dense(A, 2, J, C, "kron_I pd refill after mutating A");
    mu_assert(err ? err : "", err == 0);

    free_matrix(C);
    free_matrix(J);
    free_matrix(A);
    return 0;
}

/* The spd path re-transposes A into its scratch per fill. */
const char *test_BA_dense_kron_I_spd_J(void)
{
    matrix *A = kron_I_A();
    matrix *J = kron_I_J_spd();
    matrix *C = BA_dense_kron_I_matrices_alloc(A, 2, J);
    BA_dense_kron_I_matrices_fill_values(A, 2, J, C);
    const char *err = kron_I_matches_dense(A, 2, J, C, "kron_I spd first fill");
    mu_assert(err ? err : "", err == 0);

    ba_mutate(A);
    BA_dense_kron_I_matrices_fill_values(A, 2, J, C);
    err = kron_I_matches_dense(A, 2, J, C, "kron_I spd refill after mutating A");
    mu_assert(err ? err : "", err == 0);

    free_matrix(C);
    free_matrix(J);
    free_matrix(A);
    return 0;
}

#ifdef SP_TRACK_MEMORY
typedef struct
{
    matrix *A;
    matrix *J;
    matrix *C;
} kron_I_fill_args;

static void run_BA_dense_kron_I(const void *ctx)
{
    const kron_I_fill_args *a = (const kron_I_fill_args *) ctx;
    BA_dense_kron_I_matrices_fill_values(a->A, 2, a->J, a->C);
}

/* None of the three kron_I cells allocates in its fill. */
const char *test_BA_dense_kron_I_fill_no_transient_alloc(void)
{
    matrix *A = kron_I_A();
    matrix *Js[3] = {kron_I_J_sparse(), kron_I_J_pd(), kron_I_J_spd()};
    const char *msgs[3] = {"kron_I sparse fill allocates",
                           "kron_I pd fill allocates", "kron_I spd fill allocates"};
    for (int t = 0; t < 3; t++)
    {
        matrix *C = BA_dense_kron_I_matrices_alloc(A, 2, Js[t]);
        kron_I_fill_args args = {A, Js[t], C};
        run_BA_dense_kron_I(&args); /* warm-up */
        int ok = fill_is_alloc_free(run_BA_dense_kron_I, &args);
        free_matrix(C);
        mu_assert(msgs[t], ok);
    }
    for (int t = 0; t < 3; t++)
    {
        free_matrix(Js[t]);
    }
    free_matrix(A);
    return 0;
}
#endif /* SP_TRACK_MEMORY */

/* ---- BA_sparse_kron_matrices: C = kron(I_p, A) @ J for a sparse A ---- */

/* 3x4 sparse A with an empty row: row 0: (0: 1) (2: 2), row 1 empty,
   row 2: (1: 3) (3: -1). Used with p = 2, so J has 8 rows. */
static matrix *sk_A(void)
{
    int p[4] = {0, 2, 2, 4};
    int i[4] = {0, 2, 1, 3};
    double x[4] = {1, 2, 3, -1};
    return sn_sparse(3, 4, 4, p, i, x);
}

/* 8x5 sparse J with an empty column 2 and entries in both row blocks */
static matrix *sk_J_sparse(void)
{
    int p[9] = {0, 1, 3, 4, 5, 6, 7, 8, 9};
    int i[9] = {0, 1, 4, 3, 0, 1, 4, 3, 0};
    double x[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    return sn_sparse(8, 5, 9, p, i, x);
}

/* 8x5 permuted_dense J: rows {1, 4, 6}, cols {0, 2, 3} */
static matrix *sk_J_pd(void)
{
    int rp[3] = {1, 4, 6};
    int cp[3] = {0, 2, 3};
    double X[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    return new_permuted_dense(8, 5, 3, 3, rp, cp, X);
}

/* 8x5 stacked_pd J whose blocks interleave rows: block 0 rows {0, 2, 5}
   cols {1, 4}, block 1 rows {1, 3, 7} cols {0, 3} */
static matrix *sk_J_spd(void)
{
    int rp0[3] = {0, 2, 5};
    int cp0[2] = {1, 4};
    double X0[6] = {1, 2, 3, 4, 5, 6};
    int rp1[3] = {1, 3, 7};
    int cp1[2] = {0, 3};
    double X1[6] = {-1, -2, -3, -4, -5, -6};
    matrix *b0 = new_permuted_dense(8, 5, 3, 2, rp0, cp0, X0);
    matrix *b1 = new_permuted_dense(8, 5, 3, 2, rp1, cp1, X1);
    permuted_dense *blocks[2] = {(permuted_dense *) b0, (permuted_dense *) b1};
    return new_stacked_pd(8, 5, 2, blocks, NULL, NULL);
}

/* The explicit kron(I_p, A) = blkdiag(A, ..., A), the dense oracle's left
   factor */
static matrix *sk_kron(matrix *A, int p)
{
    const CSR_matrix *a = A->to_csr(A);
    CSR_matrix *K = new_CSR_matrix(p * a->m, p * a->n, p * a->nnz);
    int nnz = 0;
    for (int b = 0; b < p; b++)
    {
        for (int i = 0; i < a->m; i++)
        {
            for (int jj = a->p[i]; jj < a->p[i + 1]; jj++)
            {
                K->i[nnz] = b * a->n + a->i[jj];
                K->x[nnz++] = a->x[jj];
            }
            K->p[b * a->m + i + 1] = nnz;
        }
    }
    return new_sparse_matrix(K);
}

/* C's CSC mirror is fresh and holds C's current values */
static int sk_mirror_fresh(matrix *C)
{
    sparse_matrix *sm = (sparse_matrix *) C;
    if (sm->csc_seen != C->values_version) return 0;
    int *iw = (int *) sp_malloc((size_t) C->n * sizeof(int));
    CSC_matrix *ref = csr_to_csc_alloc(sm->csr, iw);
    csr_to_csc_fill_values(sm->csr, ref, iw);
    int ok = ref->nnz == sm->csc_cache->nnz &&
             cmp_double_array(sm->csc_cache->x, ref->x, ref->nnz);
    free_CSC_matrix(ref);
    sp_free(iw);
    return ok;
}

/* Alloc + fill against the dense oracle, then change J's values and refill.
   Hands C back through C_out (caller frees) for pattern checks. */
static const char *sk_check(matrix *A, matrix *J, matrix **C_out)
{
    matrix *K = sk_kron(A, 2);
    matrix *C = BA_sparse_kron_matrices_alloc(A, 2, J);
    const char *err = C->kind == MATRIX_SPARSE ? 0 : "C kind";
    if (!err && !csr_is_valid(C->to_csr(C))) err = "C pattern invalid";
    for (int round = 0; !err && round < 2; round++)
    {
        if (round == 1) ba_mutate(J);
        BA_sparse_kron_matrices_fill_values(A, 2, J, C);
        err = ba_matches_dense(K, J, C, "C values vs dense oracle");
        if (!err && !sk_mirror_fresh(C)) err = "C CSC mirror stale";
    }
    free_matrix(K);
    *C_out = C;
    return err;
}

const char *test_BA_sparse_kron_sparse_J(void)
{
    matrix *A = sk_A();
    matrix *J = sk_J_sparse();
    matrix *C;
    const char *err = sk_check(A, J, &C);
    /* C row 0 hits columns {0, 3}, row 2 {0, 1, 4}, row 3 {1, 3}, row 5
       {0, 4}; rows 1 and 4 are empty (row 1 of A is empty) */
    int Cp[7] = {0, 2, 2, 5, 7, 7, 9};
    int Ci[9] = {0, 3, 0, 1, 4, 1, 3, 0, 4};
    if (!err && !cmp_sparsity(C, Cp, Ci, 6, 9)) err = "C pattern";
    free_matrix(C);
    free_matrix(J);
    free_matrix(A);
    mu_assert(err ? err : "", err == 0);
    return 0;
}

const char *test_BA_sparse_kron_dense_J(void)
{
    matrix *A = sk_A();
    matrix *Js[2] = {sk_J_pd(), sk_J_spd()};
    const char *err = 0;
    for (int t = 0; t < 2; t++)
    {
        matrix *C;
        const char *e = sk_check(A, Js[t], &C);
        if (!err) err = e;
        free_matrix(C);
        free_matrix(Js[t]);
    }
    free_matrix(A);
    mu_assert(err ? err : "", err == 0);
    return 0;
}

#ifdef SP_TRACK_MEMORY
/* bump J so that every conversion runs */
static void run_BA_sparse_kron(const void *ctx)
{
    const kron_I_fill_args *a = (const kron_I_fill_args *) ctx;
    matrix_values_changed(a->J);
    BA_sparse_kron_matrices_fill_values(a->A, 2, a->J, a->C);
}

/* None of the three J kinds allocates in the fill. */
const char *test_BA_sparse_kron_fill_no_transient_alloc(void)
{
    matrix *A = sk_A();
    matrix *Js[3] = {sk_J_sparse(), sk_J_pd(), sk_J_spd()};
    int ok = 1;
    for (int t = 0; t < 3; t++)
    {
        matrix *C = BA_sparse_kron_matrices_alloc(A, 2, Js[t]);
        kron_I_fill_args args = {A, Js[t], C};
        run_BA_sparse_kron(&args); /* warm-up */
        ok = ok && fill_is_alloc_free(run_BA_sparse_kron, &args);
        free_matrix(C);
        free_matrix(Js[t]);
    }
    free_matrix(A);
    mu_assert("BA_sparse_kron fill allocates", ok);
    return 0;
}
#endif /* SP_TRACK_MEMORY */

#endif /* TEST_MATMUL_DISPATCHERS_H */
