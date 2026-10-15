#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/csr_kernels.h"
#include "internal/gustavson.h"
#include "internal/iVec.h"
#include "internal/utils.h"
#include "legacy/old_CSC.h"
#include "legacy/old_CSR.h"
#include "minunit.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include "test_helpers.h"

/* The Gustavson core on the B A and kron(I_p, A) J shapes: the hand-literal and
   random-reference tests of the retired csr_csc_matmul_* and
   block_left_multiply_* kernels, re-targeted, plus new vs legacy on random
   operands. */

static sparse_rows lsm_rows(const CSR_matrix *A)
{
    sparse_rows L = {A->p, A->i, A->x, A->m};
    return L;
}

/* C = kron(I_p, A) J by the core, returned as CSC (the legacy kernels' layout). */
static CSC_matrix *lsm_kron_csc(const CSR_matrix *A, const CSC_matrix *J, int p)
{
    CSR_matrix *Jr = new_csr_of_csc(J);
    sparse_rows L = lsm_rows(A);
    CSR_matrix *C = gustavson_alloc(&L, p, Jr);
    double *acc = (double *) sp_malloc((Jr->n > 0 ? Jr->n : 1) * sizeof(double));
    gustavson_fill_values(&L, p, NULL, Jr, acc, C);
    CSC_matrix *Cc = new_csc_of_csr(C);
    sp_free(acc);
    free_CSR_matrix(C);
    free_CSR_matrix(Jr);
    return Cc;
}

/* C = A B by the core with B given as CSC (pattern only). */
static CSR_matrix *lsm_ba_alloc(const CSR_matrix *A, const CSC_matrix *B)
{
    CSR_matrix *Br = new_csr_of_csc(B);
    sparse_rows L = lsm_rows(A);
    CSR_matrix *C = gustavson_alloc(&L, 1, Br);
    free_CSR_matrix(Br);
    return C;
}

/* kron, single block: C = A J with J one block */
const char *test_gustavson_kron_single_block(void)
{
    /* A is 2x3 CSR_matrix:
     * [1.0  0.0  0.0]
     * [0.0  1.0  1.0]
     */
    CSR_matrix *A = new_CSR_matrix(2, 3, 3);
    double Ax[3] = {1.0, 1.0, 1.0};
    int Ai[3] = {0, 1, 2};
    int Ap[3] = {0, 1, 3};
    memcpy(A->x, Ax, 3 * sizeof(double));
    memcpy(A->i, Ai, 3 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* J is 3x2 CSC_matrix (single block, so p=1):
     * [1.0  0.0]
     * [1.0  0.0]
     * [0.0  1.0]
     */
    CSC_matrix *J = new_CSC_matrix(3, 2, 3);
    double Jx[3] = {1.0, 1.0, 1.0};
    int Ji[3] = {0, 1, 2};
    int Jp[3] = {0, 2, 3};
    memcpy(J->x, Jx, 3 * sizeof(double));
    memcpy(J->i, Ji, 3 * sizeof(int));
    memcpy(J->p, Jp, 3 * sizeof(int));

    /* C = A @ J is 2x2: [[1, 0], [1, 1]] */
    CSC_matrix *C = lsm_kron_csc(A, J, 1);

    int expected_p[3] = {0, 2, 3};
    int expected_i[3] = {0, 1, 1};
    double expected_x[3] = {1.0, 1.0, 1.0};

    mu_assert("C dims incorrect", C->m == 2 && C->n == 2 && C->nnz == 3);
    mu_assert("C col pointers incorrect", cmp_int_array(C->p, expected_p, 3));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, expected_i, 3));
    mu_assert("C values incorrect", cmp_double_array(C->x, expected_x, 3));

    free_CSC_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(J);
    return NULL;
}

/* kron, two blocks */
const char *test_gustavson_kron_two_blocks(void)
{
    /* A is 2x2 CSR_matrix (identity) */
    CSR_matrix *A = new_CSR_matrix(2, 2, 2);
    double Ax[2] = {1.0, 1.0};
    int Ai[2] = {0, 1};
    int Ap[3] = {0, 1, 2};
    memcpy(A->x, Ax, 2 * sizeof(double));
    memcpy(A->i, Ai, 2 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* J is 4x3 CSC_matrix (two blocks of 2 rows each):
     * [1.0  0.0  0.0]
     * [0.0  0.0  0.0]
     * [0.0  1.0  0.0]
     * [0.0  0.0  1.0]
     */
    CSC_matrix *J = new_CSC_matrix(4, 3, 3);
    double Jx[3] = {1.0, 1.0, 1.0};
    int Ji[3] = {0, 2, 3};
    int Jp[4] = {0, 1, 2, 3};
    memcpy(J->x, Jx, 3 * sizeof(double));
    memcpy(J->i, Ji, 3 * sizeof(int));
    memcpy(J->p, Jp, 4 * sizeof(int));

    /* C = [A @ J1; A @ J2] = J here (A is the identity) */
    CSC_matrix *C = lsm_kron_csc(A, J, 2);

    int expected_p2[4] = {0, 1, 2, 3};
    int expected_i2[3] = {0, 2, 3};
    double expected_x2[3] = {1.0, 1.0, 1.0};

    mu_assert("C dims incorrect", C->m == 4 && C->n == 3 && C->nnz == 3);
    mu_assert("C col pointers incorrect", cmp_int_array(C->p, expected_p2, 4));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, expected_i2, 3));
    mu_assert("C values incorrect", cmp_double_array(C->x, expected_x2, 3));

    free_CSC_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(J);
    return NULL;
}

/* kron, an all-zero column in J */
const char *test_gustavson_kron_zero_column(void)
{
    /* A is 2x2 CSR_matrix (identity) */
    CSR_matrix *A = new_CSR_matrix(2, 2, 2);
    double Ax[2] = {1.0, 1.0};
    int Ai[2] = {0, 1};
    int Ap[3] = {0, 1, 2};
    memcpy(A->x, Ax, 2 * sizeof(double));
    memcpy(A->i, Ai, 2 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* J is 2x2 with an empty column:
     * [1.0  0.0]
     * [0.0  0.0]
     */
    CSC_matrix *J = new_CSC_matrix(2, 2, 1);
    double Jx[1] = {1.0};
    int Ji[1] = {0};
    int Jp[3] = {0, 1, 1};
    memcpy(J->x, Jx, 1 * sizeof(double));
    memcpy(J->i, Ji, 1 * sizeof(int));
    memcpy(J->p, Jp, 3 * sizeof(int));

    CSC_matrix *C = lsm_kron_csc(A, J, 1);

    int expected_p3[3] = {0, 1, 1};
    int expected_i3[1] = {0};

    mu_assert("C dims incorrect", C->m == 2 && C->n == 2 && C->nnz == 1);
    mu_assert("C col pointers incorrect", cmp_int_array(C->p, expected_p3, 3));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, expected_i3, 1));

    free_CSC_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(J);
    return NULL;
}

/* B A: C = A @ B where A is CSR_matrix and B is CSC_matrix */
const char *test_gustavson_ba_basic(void)
{
    /* A is 3x2 CSR_matrix:
     * [1.0  0.0]
     * [0.0  1.0]
     * [1.0  1.0]
     */
    CSR_matrix *A = new_CSR_matrix(3, 2, 4);
    double Ax[4] = {1.0, 1.0, 1.0, 1.0};
    int Ai[4] = {0, 1, 0, 1};
    int Ap[4] = {0, 1, 2, 4};
    memcpy(A->x, Ax, 4 * sizeof(double));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    /* B is 2x3 CSC_matrix:
     * [1.0  0.0  1.0]
     * [0.0  1.0  1.0]
     */
    CSC_matrix *B = new_CSC_matrix(2, 3, 4);
    double Bx[4] = {1.0, 1.0, 1.0, 1.0};
    int Bi[4] = {0, 1, 0, 1};
    int Bp[4] = {0, 1, 2, 4};
    memcpy(B->x, Bx, 4 * sizeof(double));
    memcpy(B->i, Bi, 4 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    /* C = A @ B is 3x3:
     * C = [[1, 0, 1],
     *      [0, 1, 1],
     *      [1, 1, 2]]
     */
    CSR_matrix *C = lsm_ba_alloc(A, B);

    int expected_p4[4] = {0, 2, 4, 7};
    int expected_i4[7] = {0, 2, 1, 2, 0, 1, 2};

    mu_assert("C dims incorrect", C->m == 3 && C->n == 3 && C->nnz == 7);
    mu_assert("C row pointers incorrect", cmp_int_array(C->p, expected_p4, 4));
    mu_assert("C col indices incorrect", cmp_int_array(C->i, expected_i4, 7));

    free_CSR_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(B);
    return NULL;
}

/* B A with a sparse result */
const char *test_gustavson_ba_sparse(void)
{
    /* A is 2x3 CSR_matrix:
     * [1.0  0.0  0.0]
     * [0.0  0.0  1.0]
     */
    CSR_matrix *A = new_CSR_matrix(2, 3, 2);
    double Ax[2] = {1.0, 1.0};
    int Ai[2] = {0, 2};
    int Ap[3] = {0, 1, 2};
    memcpy(A->x, Ax, 2 * sizeof(double));
    memcpy(A->i, Ai, 2 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* B is 3x2 CSC_matrix:
     * [1.0  0.0]
     * [0.0  0.0]
     * [0.0  1.0]
     */
    CSC_matrix *B = new_CSC_matrix(3, 2, 2);
    double Bx[2] = {1.0, 1.0};
    int Bi[2] = {0, 2};
    int Bp[3] = {0, 1, 2};
    memcpy(B->x, Bx, 2 * sizeof(double));
    memcpy(B->i, Bi, 2 * sizeof(int));
    memcpy(B->p, Bp, 3 * sizeof(int));

    /* C = A @ B is the 2x2 identity pattern */
    CSR_matrix *C = lsm_ba_alloc(A, B);

    int expected_p5[3] = {0, 1, 2};
    int expected_i5[2] = {0, 1};

    mu_assert("C dims incorrect", C->m == 2 && C->n == 2 && C->nnz == 2);
    mu_assert("C row pointers incorrect", cmp_int_array(C->p, expected_p5, 3));
    mu_assert("C col indices incorrect", cmp_int_array(C->i, expected_i5, 2));

    free_CSR_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(B);
    return NULL;
}

/* Dedup + ordering: one block whose child entries hit the same A row through
 * several columns (dedup), with CSC column lists that interleave so a missing
 * sort would emit rows out of ascending order. */
const char *test_gustavson_kron_dedup_order(void)
{
    /* A is 4x3 CSR_matrix:
     * [1.0  0.0  2.0]     CSC col 0: rows {0, 2}
     * [0.0  3.0  0.0]     CSC col 1: rows {1, 3}
     * [4.0  0.0  0.0]     CSC col 2: rows {0, 3}
     * [0.0  5.0  6.0]
     */
    CSR_matrix *A = new_CSR_matrix(4, 3, 6);
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int Ai[6] = {0, 2, 1, 0, 1, 2};
    int Ap[5] = {0, 2, 3, 4, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 5 * sizeof(int));

    /* J is a 3x1 dense column (p=1). Expected: all four rows, ascending. */
    CSC_matrix *J = new_CSC_matrix(3, 1, 3);
    double Jx[3] = {1.0, 1.0, 1.0};
    int Ji[3] = {0, 1, 2};
    int Jp[2] = {0, 3};
    memcpy(J->x, Jx, 3 * sizeof(double));
    memcpy(J->i, Ji, 3 * sizeof(int));
    memcpy(J->p, Jp, 2 * sizeof(int));

    CSC_matrix *C = lsm_kron_csc(A, J, 1);

    int expected_p[2] = {0, 4};
    int expected_i[4] = {0, 1, 2, 3};
    double expected_x[4] = {3.0, 3.0, 4.0, 11.0};

    mu_assert("C dims incorrect", C->m == 4 && C->n == 1 && C->nnz == 4);
    mu_assert("C col pointers incorrect", cmp_int_array(C->p, expected_p, 2));
    mu_assert("C row indices incorrect", cmp_int_array(C->i, expected_i, 4));
    mu_assert("C values incorrect", cmp_double_array(C->x, expected_x, 4));

    free_CSC_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(J);
    return NULL;
}

/* Reference implementation of the block sparsity (the original per-row
 * has_overlap scan). */
static CSC_matrix *block_left_multiply_fill_sparsity_ref(const CSR_matrix *A,
                                                         const CSC_matrix *J, int p)
{
    int m = A->m;
    int n = A->n;

    int *Cp = (int *) sp_malloc((J->n + 1) * sizeof(int));
    iVec *Ci = iVec_new(J->nnz > 0 ? J->nnz : 1);
    Cp[0] = 0;

    for (int j = 0; j < J->n; j++)
    {
        if (J->p[j] == J->p[j + 1])
        {
            Cp[j + 1] = Cp[j];
            continue;
        }
        int jj = J->p[j];
        for (int block = 0; block < p; block++)
        {
            int block_start = block * n;
            int block_end = block_start + n;
            while (jj < J->p[j + 1] && J->i[jj] < block_start)
            {
                jj++;
            }
            int block_jj_start = jj;
            while (jj < J->p[j + 1] && J->i[jj] < block_end)
            {
                jj++;
            }
            int nnz_in_block = jj - block_jj_start;
            if (nnz_in_block == 0)
            {
                continue;
            }
            for (int i = 0; i < m; i++)
            {
                int a_len = A->p[i + 1] - A->p[i];
                if (has_overlap(A->i + A->p[i], a_len, J->i + block_jj_start,
                                nnz_in_block, block_start))
                {
                    iVec_append(Ci, block * m + i);
                }
            }
        }
        Cp[j + 1] = Ci->len;
    }

    CSC_matrix *C = new_CSC_matrix(m * p, J->n, Ci->len);
    memcpy(C->p, Cp, (J->n + 1) * sizeof(int));
    memcpy(C->i, Ci->data, Ci->len * sizeof(int));
    sp_free(Cp);
    iVec_free(Ci);
    return C;
}

/* Random cross-check against the has_overlap scan. */
const char *test_gustavson_kron_matches_reference_random(void)
{
    srand(42);
    int n = 15, p = 3, k = 8;

    for (int trial = 0; trial < 5; trial++)
    {
        CSR_matrix *A = new_csr_random(20, n, 0.2);
        CSR_matrix *G = new_csr_random(n * p, k, 0.15);
        CSC_matrix *J = new_csc_of_csr(G);

        CSC_matrix *C = lsm_kron_csc(A, J, p);
        CSC_matrix *C_ref = block_left_multiply_fill_sparsity_ref(A, J, p);

        mu_assert("random block sparsity nnz mismatch", C->nnz == C_ref->nnz);
        mu_assert("random block sparsity col pointers mismatch",
                  cmp_int_array(C->p, C_ref->p, J->n + 1));
        mu_assert("random block sparsity row indices mismatch",
                  cmp_int_array(C->i, C_ref->i, C_ref->nnz));

        free_CSC_matrix(C);
        free_CSC_matrix(C_ref);
        free_CSC_matrix(J);
        free_CSR_matrix(G);
        free_CSR_matrix(A);
    }
    return NULL;
}

/* Dedup + ordering for B A: A's row hits the same B column through several B
 * rows (dedup), with B's rows interleaved so a missing sort would emit columns
 * out of ascending order. */
const char *test_gustavson_ba_dedup_order(void)
{
    /* A is 1x2 CSR_matrix: [1.0  1.0] */
    CSR_matrix *A = new_CSR_matrix(1, 2, 2);
    double Ax[2] = {1.0, 1.0};
    int Ai[2] = {0, 1};
    int Ap[2] = {0, 2};
    memcpy(A->x, Ax, 2 * sizeof(double));
    memcpy(A->i, Ai, 2 * sizeof(int));
    memcpy(A->p, Ap, 2 * sizeof(int));

    /* B is 2x3 CSC_matrix:
     * [1.0  0.0  1.0]     CSR row 0: cols {0, 2}
     * [0.0  1.0  1.0]     CSR row 1: cols {1, 2}
     */
    CSC_matrix *B = new_CSC_matrix(2, 3, 4);
    double Bx[4] = {1.0, 1.0, 1.0, 1.0};
    int Bi[4] = {0, 1, 0, 1};
    int Bp[4] = {0, 1, 2, 4};
    memcpy(B->x, Bx, 4 * sizeof(double));
    memcpy(B->i, Bi, 4 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    CSR_matrix *C = lsm_ba_alloc(A, B);

    int expected_p[2] = {0, 3};
    int expected_i[3] = {0, 1, 2};

    mu_assert("C dims incorrect", C->m == 1 && C->n == 3 && C->nnz == 3);
    mu_assert("C row pointers incorrect", cmp_int_array(C->p, expected_p, 2));
    mu_assert("C col indices incorrect", cmp_int_array(C->i, expected_i, 3));

    free_CSR_matrix(C);
    free_CSR_matrix(A);
    free_CSC_matrix(B);
    return NULL;
}

/* Reference implementation of the B A pattern (the original full
 * row-times-column has_overlap loop). */
static CSR_matrix *csr_csc_matmul_alloc_ref(const CSR_matrix *A, const CSC_matrix *B)
{
    int m = A->m;
    int p = B->n;

    int *Cp = (int *) sp_malloc((m + 1) * sizeof(int));
    iVec *Ci = iVec_new(m);
    Cp[0] = 0;

    int nnz = 0;
    for (int i = 0; i < m; i++)
    {
        int len_a = A->p[i + 1] - A->p[i];
        for (int j = 0; j < p; j++)
        {
            int len_b = B->p[j + 1] - B->p[j];
            if (has_overlap(A->i + A->p[i], len_a, B->i + B->p[j], len_b, 0))
            {
                iVec_append(Ci, j);
                nnz++;
            }
        }
        Cp[i + 1] = nnz;
    }

    CSR_matrix *C = new_CSR_matrix(m, p, nnz);
    memcpy(C->p, Cp, (m + 1) * sizeof(int));
    memcpy(C->i, Ci->data, nnz * sizeof(int));
    sp_free(Cp);
    iVec_free(Ci);
    return C;
}

/* Random cross-check against the has_overlap pair loop. */
const char *test_gustavson_ba_matches_reference_random(void)
{
    srand(7);

    for (int trial = 0; trial < 5; trial++)
    {
        CSR_matrix *A = new_csr_random(20, 15, 0.2);
        CSR_matrix *G = new_csr_random(15, 12, 0.2);
        CSC_matrix *B = new_csc_of_csr(G);

        CSR_matrix *C = lsm_ba_alloc(A, B);
        CSR_matrix *C_ref = csr_csc_matmul_alloc_ref(A, B);

        mu_assert("random matmul sparsity nnz mismatch", C->nnz == C_ref->nnz);
        mu_assert("random matmul sparsity row pointers mismatch",
                  cmp_int_array(C->p, C_ref->p, A->m + 1));
        mu_assert("random matmul sparsity col indices mismatch",
                  cmp_int_array(C->i, C_ref->i, C_ref->nnz));

        free_CSR_matrix(C);
        free_CSR_matrix(C_ref);
        free_CSC_matrix(B);
        free_CSR_matrix(G);
        free_CSR_matrix(A);
    }
    return NULL;
}

/* ---- new vs the retired kernels (legacy/old_CSC.h) on random operands ---- */

/* C = B A: core vs old_csr_csc_matmul_*, pattern exact, values within 1e-10,
   then a refill with the same acc. */
static const char *lsm_ba_vs_legacy(const CSR_matrix *B, const CSR_matrix *A)
{
    CSC_matrix *A_csc = new_csc_of_csr(A);
    sparse_rows L = lsm_rows(B);
    double *acc = (double *) sp_malloc((A->n > 0 ? A->n : 1) * sizeof(double));

    CSR_matrix *C = gustavson_alloc(&L, 1, A);
    CSR_matrix *C_old = old_csr_csc_matmul_alloc(B, A_csc);
    mu_assert("BA: invalid CSR", csr_is_valid(C));
    gustavson_fill_values(&L, 1, NULL, A, acc, C);
    old_csr_csc_matmul_fill_values(B, A_csc, C_old);
    mu_assert("BA: differs from legacy", csr_equal_tol(C, C_old, 1e-10));
    gustavson_fill_values(&L, 1, NULL, A, acc, C);
    mu_assert("BA: refill differs", csr_equal_tol(C, C_old, 1e-10));

    free_CSR_matrix(C);
    free_CSR_matrix(C_old);
    free_CSC_matrix(A_csc);
    sp_free(acc);
    return 0;
}

const char *test_gustavson_ba_matches_legacy(void)
{
    const char *err = 0;
    /* tall, wide, dense columns in both, empty rows / columns, all zero */
    int shapes[5][3] = {
        {60, 40, 25}, {30, 60, 45}, {50, 30, 20}, {30, 20, 12}, {8, 6, 5}};
    double dens[5] = {0.1, 0.1, 0.05, 0.1, 0.0};
    int dense_col[5] = {-1, -1, 7, -1, -1};
    int skip[5] = {-1, -1, -1, 3, -1};
    for (int t = 0; t < 5 && !err; t++)
    {
        int m = shapes[t][0], k = shapes[t][1], n = shapes[t][2];
        CSR_matrix *B =
            new_csr_random_ex(m, k, dens[t], dense_col[t], skip[t], skip[t]);
        CSR_matrix *A =
            new_csr_random_ex(k, n, dens[t], dense_col[t], skip[t], skip[t]);
        err = lsm_ba_vs_legacy(B, A);
        free_CSR_matrix(A);
        free_CSR_matrix(B);
    }
    mu_assert(err ? err : "", err == 0);
    return 0;
}

/* C = kron(I_p, A) J: core vs old_block_left_multiply_* (CSC output converted),
   and vs the core on the explicit blkdiag(A, ..., A) from legacy/old_CSR.h. */
static const char *lsm_kron_vs_legacy(const CSR_matrix *A, const CSR_matrix *J,
                                      int p)
{
    CSC_matrix *J_csc = new_csc_of_csr(J);
    sparse_rows L = lsm_rows(A);
    double *acc = (double *) sp_malloc((J->n > 0 ? J->n : 1) * sizeof(double));

    CSR_matrix *C = gustavson_alloc(&L, p, J);
    mu_assert("kron: invalid CSR", csr_is_valid(C));
    mu_assert("kron: wrong shape", C->m == p * A->m && C->n == J->n);
    gustavson_fill_values(&L, p, NULL, J, acc, C);

    CSC_matrix *C_old_csc = old_block_left_multiply_fill_sparsity(A, J_csc, p);
    old_block_left_multiply_fill_values(A, J_csc, C_old_csc);
    CSR_matrix *C_old = new_csr_of_csc(C_old_csc);
    mu_assert("kron: differs from legacy", csr_equal_tol(C, C_old, 1e-10));

    CSR_matrix *K = block_diag_repeat_csr(A, p);
    sparse_rows LK = lsm_rows(K);
    CSR_matrix *C_K = gustavson_alloc(&LK, 1, J);
    gustavson_fill_values(&LK, 1, NULL, J, acc, C_K);
    mu_assert("kron: differs from explicit blkdiag product",
              csr_equal_tol(C, C_K, 1e-12));

    free_CSR_matrix(C_K);
    free_CSR_matrix(K);
    free_CSR_matrix(C_old);
    free_CSC_matrix(C_old_csc);
    free_CSR_matrix(C);
    free_CSC_matrix(J_csc);
    sp_free(acc);
    return 0;
}

/* J with its middle block (rows n..2n) empty: [J1; 0; J1]. */
static CSR_matrix *lsm_csr_with_empty_block(const CSR_matrix *J1)
{
    int n = J1->m;
    CSR_matrix *J = new_CSR_matrix(3 * n, J1->n, 2 * J1->nnz);
    for (int i = 0; i <= n; i++)
    {
        J->p[i] = J1->p[i];
        J->p[n + i] = J1->nnz;
        J->p[2 * n + i] = J1->nnz + J1->p[i];
    }
    memcpy(J->i, J1->i, J1->nnz * sizeof(int));
    memcpy(J->i + J1->nnz, J1->i, J1->nnz * sizeof(int));
    memcpy(J->x, J1->x, J1->nnz * sizeof(double));
    memcpy(J->x + J1->nnz, J1->x, J1->nnz * sizeof(double));
    return J;
}

const char *test_gustavson_kron_matches_legacy(void)
{
    const char *err = 0;
    int ps[4] = {1, 2, 3, 7};
    for (int t = 0; t < 4 && !err; t++)
    {
        int p = ps[t];
        CSR_matrix *A = new_csr_random_ex(9, 7, 0.2, -1, 4, -1);
        CSR_matrix *J = new_csr_random_ex(7 * p, 11, 0.15, 2, -1, 5);
        err = lsm_kron_vs_legacy(A, J, p);
        free_CSR_matrix(J);
        free_CSR_matrix(A);
    }
    mu_assert(err ? err : "", err == 0);

    /* an empty block of J, p = 3 */
    CSR_matrix *A = new_csr_random_ex(6, 5, 0.3, -1, -1, -1);
    CSR_matrix *J1 = new_csr_random_ex(5, 8, 0.3, -1, -1, -1);
    CSR_matrix *J = lsm_csr_with_empty_block(J1);
    err = lsm_kron_vs_legacy(A, J, 3);
    free_CSR_matrix(J);
    free_CSR_matrix(J1);
    free_CSR_matrix(A);
    mu_assert(err ? err : "", err == 0);
    return 0;
}

/* Test block_left_multiply_vec with single block: y = A @ x */
const char *test_block_left_multiply_vec_single_block(void)
{
    /* A is 2x3 CSR_matrix:
     * [1.0  0.0  2.0]
     * [0.0  3.0  0.0]
     */
    CSR_matrix *A = new_CSR_matrix(2, 3, 3);
    double Ax[3] = {1.0, 3.0, 2.0};
    int Ai[3] = {0, 1, 2};
    int Ap[3] = {0, 2, 3};
    memcpy(A->x, Ax, 3 * sizeof(double));
    memcpy(A->i, Ai, 3 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* x is (3*1)-length vector = [1.0, 2.0, 3.0] */
    double x[3] = {1.0, 2.0, 3.0};
    double y[2] = {0.0, 0.0};

    block_left_multiply_vec(A, x, y, 1);

    /* Expected: y = [1.0*1.0 + 0.0*2.0 + 2.0*3.0, 0.0*1.0 + 3.0*2.0 + 0.0*3.0]
     *             = [1.0 + 6.0, 6.0]
     *             = [7.0, 6.0]
     */
    double expected_y[2] = {7.0, 6.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 2));

    free_CSR_matrix(A);
    return NULL;
}

/* Test block_left_multiply_vec with two blocks: y = [A @ x1; A @ x2] */
const char *test_block_left_multiply_vec_two_blocks(void)
{
    /* A is 2x3 CSR_matrix:
     * [1.0  2.0  0.0]
     * [0.0  3.0  4.0]
     */
    CSR_matrix *A = new_CSR_matrix(2, 3, 4);
    double Ax[4] = {1.0, 2.0, 3.0, 4.0};
    int Ai[4] = {0, 1, 1, 2};
    int Ap[3] = {0, 2, 4};
    memcpy(A->x, Ax, 4 * sizeof(double));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* x is (3*2)-length vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
     * x1 = [1.0, 2.0, 3.0], x2 = [4.0, 5.0, 6.0]
     */
    double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double y[4] = {0.0, 0.0, 0.0, 0.0};

    block_left_multiply_vec(A, x, y, 2);

    /* Expected block 1: y[0:2] = A @ x1 = [1.0*1.0 + 2.0*2.0, 3.0*2.0 + 4.0*3.0] =
     * [5.0, 18.0] Expected block 2: y[2:4] = A @ x2 = [1.0*4.0 + 2.0*5.0, 3.0*5.0
     * + 4.0*6.0] = [14.0, 39.0]
     */
    double expected_y[4] = {5.0, 18.0, 14.0, 39.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 4));

    free_CSR_matrix(A);
    return NULL;
}

/* Test block_left_multiply_vec with sparse matrix and multiple blocks */
const char *test_block_left_multiply_vec_sparse(void)
{
    /* A is 3x4 CSR_matrix (very sparse):
     * [2.0  0.0  0.0  0.0]
     * [0.0  0.0  3.0  0.0]
     * [0.0  0.0  0.0  4.0]
     */
    CSR_matrix *A = new_CSR_matrix(3, 4, 3);
    double Ax[3] = {2.0, 3.0, 4.0};
    int Ai[3] = {0, 2, 3};
    int Ap[4] = {0, 1, 2, 3};
    memcpy(A->x, Ax, 3 * sizeof(double));
    memcpy(A->i, Ai, 3 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    /* x is (4*2)-length = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
     * x1 = [1.0, 2.0, 3.0, 4.0], x2 = [5.0, 6.0, 7.0, 8.0]
     */
    double x[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    block_left_multiply_vec(A, x, y, 2);

    /* Expected block 1: y[0:3] = A @ x1 = [2.0*1.0, 3.0*3.0, 4.0*4.0] =
     * [2.0, 9.0, 16.0] Expected block 2: y[3:6] = A @ x2 =
     * [2.0*5.0, 3.0*7.0, 4.0*8.0] = [10.0, 21.0, 32.0]
     */
    double expected_y[6] = {2.0, 9.0, 16.0, 10.0, 21.0, 32.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 6));

    free_CSR_matrix(A);
    return NULL;
}

/* Test block_left_multiply_vec with three blocks */
const char *test_block_left_multiply_vec_three_blocks(void)
{
    /* A is 2x2 CSR_matrix:
     * [1.0  2.0]
     * [3.0  4.0]
     */
    CSR_matrix *A = new_CSR_matrix(2, 2, 4);
    double Ax[4] = {1.0, 2.0, 3.0, 4.0};
    int Ai[4] = {0, 1, 0, 1};
    int Ap[3] = {0, 2, 4};
    memcpy(A->x, Ax, 4 * sizeof(double));
    memcpy(A->i, Ai, 4 * sizeof(int));
    memcpy(A->p, Ap, 3 * sizeof(int));

    /* x is (2*3)-length = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
     * x1 = [1.0, 2.0], x2 = [3.0, 4.0], x3 = [5.0, 6.0]
     */
    double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    block_left_multiply_vec(A, x, y, 3);

    /* Expected block 1: y[0:2] = A @ x1 = [1.0*1.0 + 2.0*2.0, 3.0*1.0 + 4.0*2.0] =
     * [5.0, 11.0] Expected block 2: y[2:4] = A @ x2 = [1.0*3.0 + 2.0*4.0, 3.0*3.0
     * + 4.0*4.0] = [11.0, 25.0] Expected block 3: y[4:6] = A @ x3 = [1.0*5.0
     * + 2.0*6.0, 3.0*5.0 + 4.0*6.0] = [17.0, 39.0]
     */
    double expected_y[6] = {5.0, 11.0, 11.0, 25.0, 17.0, 39.0};
    mu_assert("y values incorrect", cmp_double_array(y, expected_y, 6));

    free_CSR_matrix(A);
    return NULL;
}
