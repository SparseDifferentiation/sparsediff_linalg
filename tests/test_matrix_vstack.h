#ifndef TEST_MATRIX_VSTACK_H
#define TEST_MATRIX_VSTACK_H

#include "internal/permuted_dense.h"
#include "internal/stacked_pd.h"
#include "minunit.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/matrix.h"
#include "sparsediff_linalg/matrix_vstack.h"
#include "test_helpers.h"
#include <stdint.h>
#include <string.h>

/* vstack_matrices_alloc / vstack_matrices_fill_values: C = [ops[0]; ...] over
   all three kinds. The fill copies only operands whose values_version changed
   and bumps C only when it copied something. */

/* Five 5-column operands, 15 rows and 19 nonzeros in total:
     ops[0] sparse 3x5:  row 0: (0: 1) (3: 2), row 1 empty, row 2: (1: 3) (2: 4)
                         (4: 5)
     ops[1] PD 4x5:      rows {1, 3}, cols {0, 2, 4}, X = [[10, 20, 30],
                         [40, 50, 60]]
     ops[2] spd 4x5 whose blocks interleave rows, so its block-major value
                         buffer is not in row order:
                         block 0: rows {0, 2}, cols {1, 3}, X = [[100, 200],
                                  [300, 400]]
                         block 1: rows {1, 3}, cols {0, 4}, X = [[500, 600],
                                  [700, 800]]
     ops[3] empty sparse 2x5 (nnz 0)
     ops[4] PD 2x5 with an empty dense block (m0 = n0 = 0) */
#define VS_N_OPS 5
#define VS_M 15
#define VS_NNZ 19

static void vs_fixture(matrix **ops)
{
    CSR_matrix *A = new_CSR_matrix(3, 5, 5);
    int Ap[4] = {0, 2, 2, 5};
    int Ai[5] = {0, 3, 1, 2, 4};
    double Ax[5] = {1, 2, 3, 4, 5};
    memcpy(A->p, Ap, sizeof(Ap));
    memcpy(A->i, Ai, sizeof(Ai));
    memcpy(A->x, Ax, sizeof(Ax));
    ops[0] = new_sparse_matrix(A);

    int B_rp[2] = {1, 3};
    int B_cp[3] = {0, 2, 4};
    double BX[6] = {10, 20, 30, 40, 50, 60};
    ops[1] = new_permuted_dense(4, 5, 2, 3, B_rp, B_cp, BX);

    int D0_rp[2] = {0, 2};
    int D0_cp[2] = {1, 3};
    double D0X[4] = {100, 200, 300, 400};
    matrix *blk0 = new_permuted_dense(4, 5, 2, 2, D0_rp, D0_cp, D0X);
    int D1_rp[2] = {1, 3};
    int D1_cp[2] = {0, 4};
    double D1X[4] = {500, 600, 700, 800};
    matrix *blk1 = new_permuted_dense(4, 5, 2, 2, D1_rp, D1_cp, D1X);
    permuted_dense *blocks[2] = {(permuted_dense *) blk0, (permuted_dense *) blk1};
    ops[2] = new_stacked_pd(4, 5, 2, blocks, NULL, NULL);

    ops[3] = new_sparse_matrix_alloc(2, 5, 0);
    ops[4] = new_permuted_dense(2, 5, 0, 0, NULL, NULL, NULL);
}

static void vs_free(matrix **ops)
{
    for (int k = 0; k < VS_N_OPS; k++)
    {
        free_matrix(ops[k]);
    }
}

static const int vs_exp_p[VS_M + 1] = {0,  2,  2,  5,  5,  8,  8,  11,
                                       13, 15, 17, 19, 19, 19, 19, 19};
static const int vs_exp_i[VS_NNZ] = {0, 3, 1, 2, 4, 0, 2, 4, 0, 2,
                                     4, 1, 3, 0, 4, 1, 3, 0, 4};
static const double vs_exp_x[VS_NNZ] = {
    1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 100, 200, 500, 600, 300, 400, 700, 800};

/* Expected x of the stack, built the old (engine) way through each operand's
   CSR view. Creates the operands' CSR caches; only used where that is fine. */
static void vs_oracle_x(matrix **ops, double *x)
{
    int off = 0;
    for (int k = 0; k < VS_N_OPS; k++)
    {
        CSR_matrix *csr = ops[k]->to_csr(ops[k]);
        if (csr->nnz > 0)
        {
            memcpy(x + off, csr->x, csr->nnz * sizeof(double));
        }
        off += csr->nnz;
    }
}

const char *test_vstack_mixed_kinds(void)
{
    matrix *ops[VS_N_OPS];
    vs_fixture(ops);

    matrix *C = vstack_matrices_alloc(ops, VS_N_OPS);
    mu_assert("C is sparse", C->kind == MATRIX_SPARSE);
    mu_assert("C shape", C->m == VS_M && C->n == 5 && C->nnz == VS_NNZ);
    mu_assert("C pattern valid", csr_is_valid(C->to_csr(C)));
    mu_assert("C pattern", cmp_sparsity(C, vs_exp_p, vs_exp_i, VS_M, VS_NNZ));

    vstack_matrices_fill_values(ops, VS_N_OPS, C);
    mu_assert("C values", cmp_values(C, vs_exp_x, VS_NNZ));

    /* change every operand, refill, and compare with the CSR-view oracle */
    for (int k = 0; k < VS_N_OPS; k++)
    {
        for (int jj = 0; jj < ops[k]->nnz; jj++)
        {
            ops[k]->x[jj] = -(double) (k + 1) * (jj + 1);
        }
        matrix_values_changed(ops[k]);
    }
    vstack_matrices_fill_values(ops, VS_N_OPS, C);
    double oracle[VS_NNZ];
    vs_oracle_x(ops, oracle);
    mu_assert("C values after change", cmp_values(C, oracle, VS_NNZ));

    free_matrix(C);
    vs_free(ops);
    return 0;
}

const char *test_vstack_skips_unchanged(void)
{
    matrix *ops[VS_N_OPS];
    vs_fixture(ops);
    matrix *C = vstack_matrices_alloc(ops, VS_N_OPS);

    uint64_t v = C->values_version;
    vstack_matrices_fill_values(ops, VS_N_OPS, C);
    mu_assert("first fill bumps once", C->values_version == v + 1);

    /* nothing changed: no copy, no bump */
    v = C->values_version;
    vstack_matrices_fill_values(ops, VS_N_OPS, C);
    mu_assert("idle fill does not bump", C->values_version == v);

    /* write the PD operand without bumping it: the fill must skip it */
    ops[1]->x[0] = 99.0;
    vstack_matrices_fill_values(ops, VS_N_OPS, C);
    mu_assert("unbumped operand skipped", C->x[5] == 10.0);
    mu_assert("skip does not bump", C->values_version == v);

    /* bump it: only its slice changes, and C is bumped exactly once */
    matrix_values_changed(ops[1]);
    vstack_matrices_fill_values(ops, VS_N_OPS, C);
    mu_assert("bumped operand copied", C->x[5] == 99.0);
    mu_assert("bump once", C->values_version == v + 1);
    double exp_x[VS_NNZ];
    memcpy(exp_x, vs_exp_x, sizeof(exp_x));
    exp_x[5] = 99.0;
    mu_assert("other slices unchanged", cmp_values(C, exp_x, VS_NNZ));

    /* the spd operand's scatter path */
    ((stacked_pd *) ops[2])->blocks[1]->X[0] = -5.0;
    matrix_values_changed(ops[2]);
    vstack_matrices_fill_values(ops, VS_N_OPS, C);
    exp_x[13] = -5.0;
    mu_assert("spd slice", cmp_values(C, exp_x, VS_NNZ));

    free_matrix(C);
    vs_free(ops);
    return 0;
}

const char *test_vstack_direct_write_recopies(void)
{
    matrix *ops[VS_N_OPS];
    vs_fixture(ops);
    matrix *C = vstack_matrices_alloc(ops, VS_N_OPS);
    vstack_matrices_fill_values(ops, VS_N_OPS, C);

    for (int jj = 0; jj < VS_NNZ; jj++)
    {
        C->x[jj] = 0.0;
    }
    matrix_values_changed(C);
    uint64_t v = C->values_version;

    /* no operand changed, but C was written: every operand is recopied */
    vstack_matrices_fill_values(ops, VS_N_OPS, C);
    mu_assert("values restored", cmp_values(C, vs_exp_x, VS_NNZ));
    mu_assert("restore bumps", C->values_version == v + 1);

    free_matrix(C);
    vs_free(ops);
    return 0;
}

const char *test_vstack_repeated_operand_and_n1(void)
{
    matrix *ops[VS_N_OPS];
    vs_fixture(ops);

    /* the spd twice: each slot is tracked separately */
    matrix *twice[2] = {ops[2], ops[2]};
    matrix *C2 = vstack_matrices_alloc(twice, 2);
    mu_assert("C2 shape", C2->m == 8 && C2->nnz == 16);
    mu_assert("C2 pattern valid", csr_is_valid(C2->to_csr(C2)));
    vstack_matrices_fill_values(twice, 2, C2);
    double exp2[16];
    memcpy(exp2, vs_exp_x + 11, 8 * sizeof(double));
    memcpy(exp2 + 8, vs_exp_x + 11, 8 * sizeof(double));
    mu_assert("C2 values", cmp_values(C2, exp2, 16));

    ((stacked_pd *) ops[2])->blocks[0]->X[3] = 7.5;
    matrix_values_changed(ops[2]);
    vstack_matrices_fill_values(twice, 2, C2);
    exp2[5] = 7.5;
    exp2[13] = 7.5;
    mu_assert("C2 values after change", cmp_values(C2, exp2, 16));

    /* n = 1: the stack of one sparse operand is a copy of it */
    matrix *one[1] = {ops[0]};
    matrix *C1 = vstack_matrices_alloc(one, 1);
    vstack_matrices_fill_values(one, 1, C1);
    mu_assert("C1 pattern", cmp_sparsity(C1, vs_exp_p, vs_exp_i, 3, 5));
    mu_assert("C1 values", cmp_values(C1, vs_exp_x, 5));

    free_matrix(C1);
    free_matrix(C2);
    vs_free(ops);
    return 0;
}

const char *test_vstack_no_operand_cache(void)
{
    matrix *ops[VS_N_OPS];
    vs_fixture(ops);
    matrix *C = vstack_matrices_alloc(ops, VS_N_OPS);
    vstack_matrices_fill_values(ops, VS_N_OPS, C);

    mu_assert("PD csr_cache untouched",
              ((permuted_dense *) ops[1])->csr_cache == NULL);
    mu_assert("spd csr_cache untouched", ((stacked_pd *) ops[2])->csr_cache == NULL);

    free_matrix(C);
    vs_free(ops);
    return 0;
}

#ifdef SP_TRACK_MEMORY
typedef struct
{
    matrix **ops;
    matrix *C;
} vs_fill_args;

/* bump every operand so that every copy path runs */
static void run_vstack_fill(const void *ctx)
{
    const vs_fill_args *a = (const vs_fill_args *) ctx;
    for (int k = 0; k < VS_N_OPS; k++)
    {
        matrix_values_changed(a->ops[k]);
    }
    vstack_matrices_fill_values(a->ops, VS_N_OPS, a->C);
}

const char *test_vstack_fill_no_transient_alloc(void)
{
    matrix *ops[VS_N_OPS];
    vs_fixture(ops);
    matrix *C = vstack_matrices_alloc(ops, VS_N_OPS);
    vs_fill_args args = {ops, C};
    run_vstack_fill(&args); /* warm-up */
    mu_assert("vstack fill allocates", fill_is_alloc_free(run_vstack_fill, &args));
    free_matrix(C);
    vs_free(ops);
    return 0;
}
#endif /* SP_TRACK_MEMORY */

#endif /* TEST_MATRIX_VSTACK_H */
