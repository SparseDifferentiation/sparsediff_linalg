#include "internal/CSC_matrix.h"
#include "internal/sparse_matrix.h"
#include "legacy/old_CSC.h"
#include "sparsediff_linalg/matmul.h"
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minunit.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "test_helpers.h"

/* CSR of a CSC fixture: the kernels read A row-wise. */
static CSR_matrix *csc_test_csr_of(const CSC_matrix *A)
{
    int *iwork = (int *) sp_malloc((A->m > 0 ? A->m : 1) * sizeof(int));
    CSR_matrix *R = csc_to_csr_alloc(A, iwork);
    csc_to_csr_fill_values(A, R, iwork);
    sp_free(iwork);
    return R;
}

/* Test ATA_alloc with a simple 3x3 example
 * A is 4x3 (4 rows, 3 columns):
 * [x  0  x]
 * [0  x  0]
 * [0  0  x]
 * [0  x  0]
 *
 * A^T A is 3x3:
 * [x  0  x]
 * [0  x  0]
 * [x  0  x]
 */
const char *test_ATA_alloc_simple(void)
{
    CSC_matrix *A = new_CSC_matrix(4, 3, 5);
    int Ap[4] = {0, 2, 3, 5};
    int Ai[5] = {0, 2, 1, 2, 3};
    double Ax[5] = {1, 1, 1, 1, 1};
    memcpy(A->p, Ap, 4 * sizeof(int));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->x, Ax, 5 * sizeof(double));
    CSR_matrix *A_csr = csc_test_csr_of(A);

    /* Compute C = A^T A */
    CSR_matrix *C = ATA_alloc(A, A_csr);
    int expected_p[4] = {0, 2, 3, 5};
    int expected_i[5] = {0, 2, 1, 0, 2};

    mu_assert("p incorrect", cmp_int_array(C->p, expected_p, 4));
    mu_assert("i incorrect", cmp_int_array(C->i, expected_i, C->nnz));
    mu_assert("nnz incorrect", C->nnz == 5);

    free_CSR_matrix(C);
    free_CSR_matrix(A_csr);
    free_CSC_matrix(A);

    return 0;
}

/* Test ATA_alloc with a sparse 3x4 matrix with no overlaps on some pairs
 * A is 3x4:
 * [1  0  0  2]
 * [0  1  0  0]
 * [0  0  1  0]
 *
 * A^T A is 4x4:
 * [1  0  0  2]
 * [0  1  0  0]
 * [0  0  1  0]
 * [2  0  0  4]
 *
 */
const char *test_ATA_alloc_diagonal_like(void)
{
    /* Create A in CSC_matrix format (3 rows, 4 cols, 4 nonzeros) */
    CSC_matrix *A = new_CSC_matrix(3, 4, 4);
    int Ap[5] = {0, 1, 2, 3, 4};
    int Ai[4] = {0, 1, 2, 0};
    double Ax[4] = {1, 1, 1, 2};
    memcpy(A->p, Ap, 5 * sizeof(int));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->x, Ax, 4 * sizeof(double));
    CSR_matrix *A_csr = csc_test_csr_of(A);
    CSR_matrix *C = ATA_alloc(A, A_csr);

    int expected_p[5] = {0, 2, 3, 4, 6};
    int expected_i[6] = {0, 3, 1, 2, 0, 3};

    mu_assert("p incorrect", cmp_int_array(C->p, expected_p, 5));
    mu_assert("i incorrect", cmp_int_array(C->i, expected_i, C->nnz));
    mu_assert("nnz incorrect", C->nnz == 6);

    free_CSR_matrix(C);
    free_CSR_matrix(A_csr);
    free_CSC_matrix(A);

    return 0;
}

const char *test_ATA_alloc_random(void)
{
    /* Create A in CSC_matrix format  */
    CSC_matrix *A = new_CSC_matrix(10, 15, 15);
    int Ap[16] = {0, 1, 1, 1, 1, 4, 5, 6, 7, 8, 9, 11, 11, 11, 13, 15};
    int Ai[15] = {5, 0, 6, 9, 0, 5, 1, 3, 6, 0, 6, 3, 6, 6, 8};
    double Ax[15] = {7, 4, 8, 5, 7, 3, 7, 8, 5, 4, 8, 8, 3, 6, 5};
    memcpy(A->p, Ap, 16 * sizeof(int));
    memcpy(A->i, Ai, 15 * sizeof(int));
    memcpy(A->x, Ax, 15 * sizeof(double));
    CSR_matrix *A_csr = csc_test_csr_of(A);
    CSR_matrix *C = ATA_alloc(A, A_csr);

    int expected_p[16] = {0, 2, 2, 2, 2, 8, 11, 13, 14, 16, 21, 27, 27, 27, 33, 38};
    int expected_i[38] = {0,  6, 4,  5, 9,  10, 13, 14, 4, 5,  10, 0,  6,
                          7,  8, 13, 4, 9,  10, 13, 14, 4, 5,  9,  10, 13,
                          14, 4, 8,  9, 10, 13, 14, 4,  9, 10, 13, 14};

    mu_assert("p incorrect", cmp_int_array(C->p, expected_p, 16));
    mu_assert("i incorrect", cmp_int_array(C->i, expected_i, C->nnz));
    mu_assert("nnz incorrect", C->nnz == 38);

    double d[10] = {2, 8, 6, 2, 5, 1, 6, 9, 1, 3};

    double *acc = (double *) sp_malloc(A->n * sizeof(double));
    ATDA_fill_values(A, d, A_csr, acc, C);
    sp_free(acc);

    double Cx_correct[38] = {
        49.,  21.,  491., 56.,  240., 416., 144., 288., 56.,  98.,  56.,  21.,  9.,
        392., 128., 128., 240., 150., 240., 90.,  180., 416., 56.,  240., 416., 144.,
        288., 144., 128., 90.,  144., 182., 108., 288., 180., 288., 108., 241.};
    mu_assert("x incorrect", cmp_double_array(C->x, Cx_correct, C->nnz));

    free_CSR_matrix(C);
    free_CSR_matrix(A_csr);
    free_CSC_matrix(A);

    return 0;
}

const char *test_ATA_alloc_random2(void)
{
    /* Create A in CSC_matrix format  */
    int m = 15;
    int n = 10;
    CSC_matrix *A = new_CSC_matrix(m, n, 15);
    int Ap[11] = {0, 2, 4, 6, 6, 9, 12, 12, 14, 14, 15};
    int Ai[15] = {9, 12, 3, 4, 1, 6, 4, 8, 13, 1, 3, 7, 5, 13, 6};
    double Ax[15] = {0.99, 0.9,  0.51, 0.64, 0.39, 0.29, 0.26, 0.91,
                     0.35, 0.18, 0.33, 0.73, 0.97, 0.86, 1.03};
    memcpy(A->p, Ap, 11 * sizeof(int));
    memcpy(A->i, Ai, 15 * sizeof(int));
    memcpy(A->x, Ax, 15 * sizeof(double));
    CSR_matrix *A_csr = csc_test_csr_of(A);
    CSR_matrix *C = ATA_alloc(A, A_csr);

    int expected_p[11] = {0, 1, 4, 7, 7, 10, 13, 13, 15, 15, 17};
    int expected_i[17] = {0, 1, 4, 5, 2, 5, 9, 1, 4, 7, 1, 2, 5, 4, 7, 2, 9};

    mu_assert("p incorrect", cmp_int_array(C->p, expected_p, 11));
    mu_assert("i incorrect", cmp_int_array(C->i, expected_i, C->nnz));
    mu_assert("nnz incorrect", C->nnz == 17);
    double d[15] = {-0.6,  -0.23, -0.29, -1.36, 0.4,   0.36, 0.11, -0.13,
                    -1.32, -0.32, -0.24, -0.7,  -0.06, 0.5,  1.99};

    double *acc = (double *) sp_malloc(A->n * sizeof(double));
    ATDA_fill_values(A, d, A_csr, acc, C);
    sp_free(acc);

    double Cx_correct[17] = {-0.362232, -0.189896, 0.06656,   -0.228888, -0.025732,
                             -0.016146, 0.032857,  0.06656,   -1.004802, 0.1505,
                             -0.228888, -0.016146, -0.224833, 0.1505,    0.708524,
                             0.032857,  0.116699};
    mu_assert("x incorrect", cmp_double_array(C->x, Cx_correct, C->nnz));

    free_CSR_matrix(C);
    free_CSR_matrix(A_csr);
    free_CSC_matrix(A);

    return 0;
}
const char *test_BTA_alloc_and_BTDA_fill(void)
{
    /* Create A: 4x3 CSC_matrix matrix
     * [1.0  0.0  2.0]
     * [0.0  3.0  0.0]
     * [4.0  0.0  5.0]
     * [0.0  6.0  0.0]
     */
    int m = 4;
    int n = 3;
    CSC_matrix *A = new_CSC_matrix(m, n, 6);
    int Ap_A[4] = {0, 2, 4, 6};
    int Ai_A[6] = {0, 2, 1, 3, 0, 2};
    double Ax_A[6] = {1.0, 4.0, 3.0, 6.0, 2.0, 5.0};
    memcpy(A->p, Ap_A, 4 * sizeof(int));
    memcpy(A->i, Ai_A, 6 * sizeof(int));
    memcpy(A->x, Ax_A, 6 * sizeof(double));

    /* Create B: 4x2 CSC_matrix matrix
     * [1.0  0.0]
     * [0.0  2.0]
     * [3.0  0.0]
     * [0.0  4.0]
     */
    int p = 2;
    CSC_matrix *B = new_CSC_matrix(m, p, 4);
    int Bp[3] = {0, 2, 4};
    int Bi[4] = {0, 2, 1, 3};
    double Bx[4] = {1.0, 3.0, 2.0, 4.0};
    memcpy(B->p, Bp, 3 * sizeof(int));
    memcpy(B->i, Bi, 4 * sizeof(int));
    memcpy(B->x, Bx, 4 * sizeof(double));

    /* Allocate C = B^T A (should be 2x3) */
    CSR_matrix *A_csr = csc_test_csr_of(A);
    CSR_matrix *C = BTA_alloc(A_csr, B);

    /* Sparsity pattern check before filling values */
    int expected_p[3] = {0, 2, 3};
    int expected_i[3] = {0, 2, 1};
    mu_assert("C dimensions incorrect", C->m == 2 && C->n == 3);
    mu_assert("C nnz incorrect", C->nnz == 3);
    mu_assert("C->p incorrect", cmp_int_array(C->p, expected_p, 3));
    mu_assert("C->i incorrect", cmp_int_array(C->i, expected_i, 3));

    /* Fill values with diagonal weights d */
    double d[4] = {1.0, 2.0, 3.0, 4.0};
    double acc[3];
    BTDA_fill_values(A_csr, B, d, acc, C);

    double expected_x[3] = {37.0, 47.0, 108.0};
    mu_assert("C values incorrect", cmp_double_array(C->x, expected_x, 3));

    free_CSR_matrix(C);
    free_CSR_matrix(A_csr);
    free_CSC_matrix(A);
    free_CSC_matrix(B);

    return 0;
}

/* ---- The Gustavson-backed kernels against the retired ones (legacy/old_CSC.h)
   on random operands. ---- */

static CSC_matrix *csc_test_csc_of(const CSR_matrix *A)
{
    int *iwork = (int *) sp_malloc((A->n > 0 ? A->n : 1) * sizeof(int));
    CSC_matrix *C = csr_to_csc_alloc(A, iwork);
    csr_to_csc_fill_values(A, C, iwork);
    sp_free(iwork);
    return C;
}

/* Random m x n CSR with the given density; column dense_col is full, row
   skip_row and column skip_col are empty (-1 disables each). */
static CSR_matrix *csc_test_random_csr(int m, int n, double density, int dense_col,
                                       int skip_row, int skip_col)
{
    CSR_matrix *A = new_CSR_matrix(m, n, m * n);
    int nnz = 0;
    A->p[0] = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == skip_row || j == skip_col) continue;
            if (j == dense_col || rand() < density * RAND_MAX)
            {
                A->i[nnz] = j;
                A->x[nnz] = rand() / (double) RAND_MAX - 0.5;
                nnz++;
            }
        }
        A->p[i + 1] = nnz;
    }
    CSR_trim(A);
    return A;
}

static double *csc_test_random_vec(int n)
{
    double *d = (double *) sp_malloc(n * sizeof(double));
    for (int k = 0; k < n; k++)
    {
        d[k] = rand() / (double) RAND_MAX - 0.5;
    }
    return d;
}

/* Same pattern exactly; values equal up to summation order. */
static int csc_test_csr_close(const CSR_matrix *A, const CSR_matrix *B)
{
    if (A->m != B->m || A->n != B->n || A->nnz != B->nnz) return 0;
    if (!cmp_int_array(A->p, B->p, A->m + 1) || !cmp_int_array(A->i, B->i, A->nnz))
        return 0;
    for (int ii = 0; ii < A->nnz; ii++)
    {
        if (fabs(A->x[ii] - B->x[ii]) > 1e-10) return 0;
    }
    return 1;
}

/* B^T A and B^T D A, new kernels vs legacy. */
static const char *csc_test_bta_vs_legacy(const CSR_matrix *A, const CSR_matrix *B,
                                          int with_d)
{
    CSC_matrix *A_csc = csc_test_csc_of(A);
    CSC_matrix *B_csc = csc_test_csc_of(B);
    double *d = with_d ? csc_test_random_vec(A->m) : NULL;
    double *acc = (double *) sp_malloc(A->n * sizeof(double));

    CSR_matrix *C = BTA_alloc(A, B_csc);
    CSR_matrix *C_old = old_BTA_alloc(A_csc, B_csc);
    mu_assert("BTA_alloc: invalid CSR", csr_is_valid(C));
    mu_assert("BTA_alloc: pattern differs from legacy",
              C->nnz == C_old->nnz && cmp_int_array(C->p, C_old->p, C->m + 1) &&
                  cmp_int_array(C->i, C_old->i, C->nnz));

    BTDA_fill_values(A, B_csc, d, acc, C);
    old_BTDA_fill_values(A_csc, B_csc, d, C_old);
    mu_assert("BTDA_fill_values: values differ from legacy",
              csc_test_csr_close(C, C_old));

    /* refill with the same acc */
    BTDA_fill_values(A, B_csc, d, acc, C);
    mu_assert("BTDA_fill_values: refill differs", csc_test_csr_close(C, C_old));

    free_CSR_matrix(C);
    free_CSR_matrix(C_old);
    free_CSC_matrix(A_csc);
    free_CSC_matrix(B_csc);
    sp_free(d);
    sp_free(acc);
    return 0;
}

static const char *csc_test_ata_vs_legacy(const CSR_matrix *A, int with_d)
{
    CSC_matrix *A_csc = csc_test_csc_of(A);
    double *d = with_d ? csc_test_random_vec(A->m) : NULL;
    double *acc = (double *) sp_malloc(A->n * sizeof(double));

    CSR_matrix *C = ATA_alloc(A_csc, A);
    CSR_matrix *C_old = old_ATA_alloc(A_csc);
    mu_assert("ATA_alloc: invalid CSR", csr_is_valid(C));
    mu_assert("ATA_alloc: pattern differs from legacy",
              C->nnz == C_old->nnz && cmp_int_array(C->p, C_old->p, C->m + 1) &&
                  cmp_int_array(C->i, C_old->i, C->nnz));

    ATDA_fill_values(A_csc, d, A, acc, C);
    old_ATDA_fill_values(A_csc, d, C_old);
    mu_assert("ATDA_fill_values: values differ from legacy",
              csc_test_csr_close(C, C_old));

    free_CSR_matrix(C);
    free_CSR_matrix(C_old);
    free_CSC_matrix(A_csc);
    sp_free(d);
    sp_free(acc);
    return 0;
}

const char *test_gustavson_BTA_matches_legacy(void)
{
    /* tall and wide operands, with and without D */
    CSR_matrix *A = csc_test_random_csr(60, 40, 0.1, -1, -1, -1);
    CSR_matrix *B = csc_test_random_csr(60, 25, 0.1, -1, -1, -1);
    const char *err = csc_test_bta_vs_legacy(A, B, 0);
    if (err == NULL) err = csc_test_bta_vs_legacy(A, B, 1);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    mu_assert(err, err == NULL);

    A = csc_test_random_csr(40, 60, 0.1, -1, -1, -1);
    B = csc_test_random_csr(40, 30, 0.15, -1, -1, -1);
    err = csc_test_bta_vs_legacy(A, B, 1);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    mu_assert(err, err == NULL);
    return 0;
}

const char *test_gustavson_ATA_matches_legacy(void)
{
    CSR_matrix *A = csc_test_random_csr(60, 40, 0.1, -1, -1, -1);
    const char *err = csc_test_ata_vs_legacy(A, 0);
    if (err == NULL) err = csc_test_ata_vs_legacy(A, 1);
    free_CSR_matrix(A);
    mu_assert(err, err == NULL);

    A = csc_test_random_csr(40, 60, 0.1, -1, -1, -1);
    err = csc_test_ata_vs_legacy(A, 1);
    free_CSR_matrix(A);
    mu_assert(err, err == NULL);
    return 0;
}

/* A dense column in A and in B. */
const char *test_gustavson_dense_column(void)
{
    CSR_matrix *A = csc_test_random_csr(50, 30, 0.05, 7, -1, -1);
    CSR_matrix *B = csc_test_random_csr(50, 20, 0.05, 3, -1, -1);
    const char *err = csc_test_bta_vs_legacy(A, B, 1);
    if (err == NULL) err = csc_test_ata_vs_legacy(A, 1);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    mu_assert(err, err == NULL);
    return 0;
}

/* Empty rows and columns, including an empty row of C. */
const char *test_gustavson_empty_rows_and_cols(void)
{
    CSR_matrix *A = csc_test_random_csr(30, 20, 0.1, -1, 5, 3);
    CSR_matrix *B = csc_test_random_csr(30, 12, 0.1, -1, 5, 4);
    const char *err = csc_test_bta_vs_legacy(A, B, 1);
    if (err == NULL) err = csc_test_ata_vs_legacy(A, 1);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    mu_assert(err, err == NULL);

    /* no nonzeros at all */
    A = csc_test_random_csr(8, 6, 0.0, -1, -1, -1);
    B = csc_test_random_csr(8, 5, 0.0, -1, -1, -1);
    err = csc_test_bta_vs_legacy(A, B, 0);
    if (err == NULL) err = csc_test_ata_vs_legacy(A, 0);
    free_CSR_matrix(A);
    free_CSR_matrix(B);
    mu_assert(err, err == NULL);
    return 0;
}

/* The sparse x sparse dispatchers bind the accumulator to C at alloc. */
const char *test_sparse_BTA_ATA_bind_accumulator(void)
{
    matrix *A = new_sparse_matrix(csc_test_random_csr(30, 20, 0.1, -1, -1, -1));
    matrix *B = new_sparse_matrix(csc_test_random_csr(30, 12, 0.1, -1, -1, -1));
    matrix *C = BTA_matrices_alloc(A, B);
    matrix *G = ATA_matrices_alloc(A);
    mu_assert("BTA: no accumulator bound",
              ((sparse_matrix *) C)->bound_dwork != NULL);
    mu_assert("ATA: no accumulator bound",
              ((sparse_matrix *) G)->bound_dwork != NULL);
    BTA_matrices_fill_values(A, B, C);
    ATDA_matrices_fill_values(A, NULL, G);
    free_matrix(C);
    free_matrix(G);
    free_matrix(A);
    free_matrix(B);
    return 0;
}

#ifdef SP_TRACK_MEMORY
typedef struct
{
    const matrix *A;
    const matrix *B;
    const double *d;
    matrix *C;
} csc_test_fill_args;

static void csc_test_run_BTDA(const void *ctx)
{
    const csc_test_fill_args *a = (const csc_test_fill_args *) ctx;
    BTDA_matrices_fill_values(a->A, a->d, a->B, a->C);
}

static void csc_test_run_ATDA(const void *ctx)
{
    const csc_test_fill_args *a = (const csc_test_fill_args *) ctx;
    ATDA_matrices_fill_values(a->A, a->d, a->C);
}

/* No-alloc-in-fill contract for the sparse x sparse BTDA / ATDA fills. The
   first fill builds the CSC mirror values, so warm up once. */
const char *test_sparse_BTDA_ATDA_fill_no_transient_alloc(void)
{
    matrix *A = new_sparse_matrix(csc_test_random_csr(40, 30, 0.1, 2, -1, -1));
    matrix *B = new_sparse_matrix(csc_test_random_csr(40, 15, 0.1, -1, -1, -1));
    double *d = csc_test_random_vec(40);
    matrix *C = BTA_matrices_alloc(A, B);
    matrix *G = ATA_matrices_alloc(A);
    csc_test_fill_args args = {A, B, d, C};
    csc_test_fill_args args_ata = {A, NULL, d, G};

    BTDA_matrices_fill_values(A, d, B, C);
    mu_assert("sparse BTDA fill allocated",
              fill_is_alloc_free(csc_test_run_BTDA, &args));
    ATDA_matrices_fill_values(A, d, G);
    mu_assert("sparse ATDA fill allocated",
              fill_is_alloc_free(csc_test_run_ATDA, &args_ata));

    free_matrix(C);
    free_matrix(G);
    free_matrix(A);
    free_matrix(B);
    sp_free(d);
    return 0;
}
#endif /* SP_TRACK_MEMORY */
