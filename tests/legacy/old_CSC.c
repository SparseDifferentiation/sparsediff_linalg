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
#include "legacy/old_CSC.h"
#include "internal/CSR_matrix.h"
#include "internal/iVec.h"
#include "internal/utils.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <string.h>

/* Dot-product helpers of the retired kernels. */
static inline double sparse_dot(const double *a_x, const int *a_i, int a_nnz,
                                const double *b_x, const int *b_i, int b_nnz)
{
    int ii = 0;
    int jj = 0;
    double sum = 0.0;

    while (ii < a_nnz && jj < b_nnz)
    {
        if (a_i[ii] == b_i[jj])
        {
            sum += a_x[ii] * b_x[jj];
            ii++;
            jj++;
        }
        else if (a_i[ii] < b_i[jj])
        {
            ii++;
        }
        else
        {
            jj++;
        }
    }

    return sum;
}

/* NOTE: an alternative marker-based approach (scatter A_{k,j} * Q[k,:]
 * into column j of C using a marker array for position lookup) may be
 * faster when Q is dense, since it touches each Q entry exactly once.
 * The sparse_dot approach below is simpler but redundantly scans
 * column j of A for each nonzero row of C. */
void BA_fill_values(const CSR_matrix *Q, const CSC_matrix *A, CSC_matrix *C)
{
    /* fill values of C = Q * A, given the sparsity pattern of C. */
    int i, j, ii;

    /* for each column j of C */
    for (j = 0; j < C->n; j++)
    {
        for (ii = C->p[j]; ii < C->p[j + 1]; ii++)
        {
            i = C->i[ii];
            int nnz_q = Q->p[i + 1] - Q->p[i];
            int nnz_a = A->p[j + 1] - A->p[j];

            /* inner product between row i of Q and column j of A */
            C->x[ii] = sparse_dot(Q->x + Q->p[i], Q->i + Q->p[i], nnz_q,
                                  A->x + A->p[j], A->i + A->p[j], nnz_a);
        }
    }
}

CSC_matrix *symBA_alloc(const CSR_matrix *B, const CSC_matrix *A)
{
    /* Allocate C = B * A (sparsity only). B must be symmetric.
     * B is CSR_matrix (m x m), A is CSC_matrix (m x n), C is CSC_matrix (m x n).
     *
     * Column j of C is B * a_j = sum_k A_{k,j} B[:, k], so the nonzero
     * rows of column j of C are the union of the nonzero rows of B[:, k].
     *
     * Since B is symmetric, we can find the nonzero rows of B[:, k] by
     * finding the nonzero columns of B in row k.
     *
     * We use a marker array to avoid duplicates: marker[l] stores the
     * last column j that registered l as nonzero, so checking
     * marker[l] != j avoids duplicates. */

    int m = B->m;
    int n = A->n;
    int i, j, k, jj, ii, ell;

    /* marker[row] = last column j that registered row as nonzero */
    int *marker = (int *) sp_malloc(m * sizeof(int));
    for (i = 0; i < m; i++)
    {
        marker[i] = -1;
    }

    int *Cp = (int *) sp_malloc((n + 1) * sizeof(int));
    iVec *Ci = iVec_new(A->nnz);
    Cp[0] = 0;

    /* for each column j of C */
    for (j = 0; j < n; j++)
    {
        int col_nnz = 0;

        /* iterate over nonzero rows k in column j of A */
        for (ii = A->p[j]; ii < A->p[j + 1]; ii++)
        {
            k = A->i[ii];

            /* find nonzero rows ell of column k of B */
            for (jj = B->p[k]; jj < B->p[k + 1]; jj++)
            {
                ell = B->i[jj];
                if (marker[ell] != j)
                {
                    marker[ell] = j;
                    iVec_append(Ci, ell);
                    col_nnz++;
                }
            }
        }

        Cp[j + 1] = Cp[j] + col_nnz;
    }

    /* allocate C and copy the computed structure */
    int total_nnz = Cp[n];
    CSC_matrix *C = new_CSC_matrix(m, n, total_nnz);
    memcpy(C->p, Cp, (n + 1) * sizeof(int));
    memcpy(C->i, Ci->data, total_nnz * sizeof(int));

    sp_free(marker);
    sp_free(Cp);
    iVec_free(Ci);

    return C;
}

int count_nonzero_cols_csc(const CSC_matrix *A)
{
    int count = 0;
    for (int j = 0; j < A->n; j++)
    {
        if (A->p[j + 1] > A->p[j])
        {
            count++;
        }
    }
    return count;
}

void yTA_fill_values(const CSC_matrix *A, const double *y, CSR_matrix *C)
{
    for (int col = 0; col < A->n; col++)
    {
        double val = 0;
        for (int j = A->p[col]; j < A->p[col + 1]; j++)
        {
            val += y[A->i[j]] * A->x[j];
        }

        if (A->p[col + 1] - A->p[col] == 0) continue;

        /* find position in C and fill value */
        for (int k = 0; k < C->nnz; k++)
        {
            if (C->i[k] == col)
            {
                C->x[k] = val;
                break;
            }
        }
    }
}

CSR_matrix *old_ATA_alloc(const CSC_matrix *A)
{
    /* A is m x n, A^T A is n x n */
    int n = A->n;
    int m = A->m;
    int nnz = 0;
    int i, j, ii, jj;

    /* row ptr and column idxs for upper triangular part of C = A^T A */
    int *Cp = (int *) sp_malloc((n + 1) * sizeof(int));
    iVec *Ci = iVec_new(m);
    Cp[0] = 0;

    /* compute sparsity pattern, only storing upper triangular part */
    for (i = 0; i < n; i++)
    {
        /* check if Cij != 0 */
        for (j = i; j < n; j++)
        {
            ii = A->p[i];
            jj = A->p[j];

            while (ii < A->p[i + 1] && jj < A->p[j + 1])
            {
                if (A->i[ii] == A->i[jj])
                {
                    nnz += (j != i) ? 2 : 1;
                    iVec_append(Ci, j);
                    break;
                }
                else if (A->i[ii] < A->i[jj])
                {
                    ii++;
                }
                else
                {
                    jj++;
                }
            }
        }
        Cp[i + 1] = Ci->len;
    }

    /* Allocate C and symmetrize it */
    CSR_matrix *C = new_CSR_matrix(n, n, nnz);
    symmetrize_csr(Cp, Ci->data, n, C);

    /* free workspace */
    sp_free(Cp);
    iVec_free(Ci);

    return C;
}

static inline double sparse_wdot(const double *a_x, const int *a_i, int a_nnz,
                                 const double *b_x, const int *b_i, int b_nnz,
                                 const double *d)
{
    int ii = 0;
    int jj = 0;
    double sum = 0.0;

    while (ii < a_nnz && jj < b_nnz)
    {
        if (a_i[ii] == b_i[jj])
        {
            sum += a_x[ii] * b_x[jj] * d[a_i[ii]];
            ii++;
            jj++;
        }
        else if (a_i[ii] < b_i[jj])
        {
            ii++;
        }
        else
        {
            jj++;
        }
    }

    return sum;
}

void old_ATDA_fill_values(const CSC_matrix *A, const double *d, CSR_matrix *C)
{
    int j, ii, jj;
    for (ii = 0; ii < C->m; ii++)
    {
        for (jj = C->p[ii]; jj < C->p[ii + 1]; jj++)
        {
            j = C->i[jj];

            if (j < ii)
            {
                C->x[jj] = csr_get_value(C, j, ii);
            }
            else
            {
                int nnz_ai = A->p[ii + 1] - A->p[ii];
                int nnz_aj = A->p[j + 1] - A->p[j];

                if (d != NULL)
                {
                    C->x[jj] =
                        sparse_wdot(A->x + A->p[ii], A->i + A->p[ii], nnz_ai,
                                    A->x + A->p[j], A->i + A->p[j], nnz_aj, d);
                }
                else
                {
                    C->x[jj] = sparse_dot(A->x + A->p[ii], A->i + A->p[ii], nnz_ai,
                                          A->x + A->p[j], A->i + A->p[j], nnz_aj);
                }
            }
        }
    }
}

CSR_matrix *old_BTA_alloc(const CSC_matrix *A, const CSC_matrix *B)
{
    /* A is m x n, B is m x p, C = B^T A is p x n */
    int n = A->n;
    int p = B->n;
    int nnz = 0;
    int i, j, ii, jj;

    /* row ptr and column idxs for C = B^T A */
    int *Cp = (int *) sp_malloc((p + 1) * sizeof(int));
    iVec *Ci = iVec_new(n);
    Cp[0] = 0;

    /* compute sparsity pattern */
    for (i = 0; i < p; i++)
    {
        /* check if Cij != 0 for each column j of A */
        for (j = 0; j < n; j++)
        {
            ii = B->p[i];
            jj = A->p[j];

            /* check if row i of B^T (column i of B) has common row with column j of
             * A */
            while (ii < B->p[i + 1] && jj < A->p[j + 1])
            {
                if (B->i[ii] == A->i[jj])
                {
                    nnz++;
                    iVec_append(Ci, j);
                    break;
                }
                else if (B->i[ii] < A->i[jj])
                {
                    ii++;
                }
                else
                {
                    jj++;
                }
            }
        }
        Cp[i + 1] = Ci->len;
    }

    /* Allocate C */
    CSR_matrix *C = new_CSR_matrix(p, n, nnz);
    memcpy(C->p, Cp, (p + 1) * sizeof(int));
    memcpy(C->i, Ci->data, nnz * sizeof(int));

    /* free workspace */
    sp_free(Cp);
    iVec_free(Ci);

    return C;
}

/* computes C = B^T * D * A in CSR_matrix */
void old_BTDA_fill_values(const CSC_matrix *A, const CSC_matrix *B, const double *d,
                          CSR_matrix *C)
{
    int i, j, jj;
    for (i = 0; i < C->m; i++)
    {
        for (jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            j = C->i[jj];

            int nnz_bi = B->p[i + 1] - B->p[i];
            int nnz_aj = A->p[j + 1] - A->p[j];

            if (d != NULL)
            {
                C->x[jj] = sparse_wdot(B->x + B->p[i], B->i + B->p[i], nnz_bi,
                                       A->x + A->p[j], A->i + A->p[j], nnz_aj, d);
            }
            else
            {
                C->x[jj] = sparse_dot(B->x + B->p[i], B->i + B->p[i], nnz_bi,
                                      A->x + A->p[j], A->i + A->p[j], nnz_aj);
            }
        }
    }
}

static inline double sparse_dot_offset(const double *a_x, const int *a_i, int a_nnz,
                                       const double *b_x, const int *b_i, int b_nnz,
                                       int b_offset)
{
    int ii = 0;
    int jj = 0;
    double sum = 0.0;

    while (ii < a_nnz && jj < b_nnz)
    {
        if (a_i[ii] == b_i[jj] - b_offset)
        {
            sum += a_x[ii] * b_x[jj];
            ii++;
            jj++;
        }
        else if (a_i[ii] < b_i[jj] - b_offset)
        {
            ii++;
        }
        else
        {
            jj++;
        }
    }

    return sum;
}

/* Symbolic phase of Gustavson's sparse matmul: an output row's (or column's)
   sparsity pattern is the union of the operand index lists selected by keys,
   i.e. the lists list_i[list_p[k] .. list_p[k+1]] for each k = keys[t] -
   key_offset. Compute that union into cand, deduplicated with a
   generation-stamped workspace and sorted ascending (the order the downstream
   sparse dot products require). Returns the number of gathered indices. */
static int sorted_pattern_union(const int *keys, int n_keys, int key_offset,
                                const int *list_p, const int *list_i, int *stamp,
                                int *cand, int *gen)
{
    int g = ++(*gen);
    int n_cand = 0;
    for (int t = 0; t < n_keys; t++)
    {
        int k = keys[t] - key_offset;
        for (int q = list_p[k]; q < list_p[k + 1]; q++)
        {
            int i = list_i[q];
            if (stamp[i] != g)
            {
                stamp[i] = g;
                cand[n_cand++] = i;
            }
        }
    }
    sort_int_array(cand, n_cand);
    return n_cand;
}

CSC_matrix *old_block_left_multiply_fill_sparsity(const CSR_matrix *A,
                                                  const CSC_matrix *J, int p)
{
    /* A is m x n, J is (n*p) x k, C is (m*p) x k */
    int m = A->m;
    int n = A->n;
    int j, jj, block, block_start, block_end, block_jj_start, block_jj_end,
        row_offset;

    /* Allocate column pointers and an estimate of row indices. Capacity hint
       based on J->nnz and not on the dense product */
    int *Cp = (int *) sp_malloc((J->n + 1) * sizeof(int));
    iVec *Ci = iVec_new(J->nnz > 0 ? J->nnz : 1);
    Cp[0] = 0;

    /* Output-driven: row i of A intersects a block's child entries iff i
       appears in the CSC column list of one of those entries, so gather those
       columns instead of testing all m rows per (column, block). */
    int *iwork = (int *) sp_malloc((n > 0 ? n : 1) * sizeof(int));
    CSC_matrix *A_csc = csr_to_csc_alloc(A, iwork);
    sp_free(iwork);

    int *stamp = (int *) sp_calloc(m > 0 ? m : 1, sizeof(int)); /* 0 = unseen */
    int *cand = (int *) sp_malloc((m > 0 ? m : 1) * sizeof(int));
    int gen = 0;

    /* for each column of J */
    for (j = 0; j < J->n; j++)
    {
        /* if empty we continue */
        if (J->p[j] == J->p[j + 1])
        {
            Cp[j + 1] = Cp[j];
            continue;
        }

        /* process each of p blocks of rows in this column of J */
        jj = J->p[j];
        for (block = 0; block < p; block++)
        {

            // -----------------------------------------------------------------
            //  find start and end indices of rows of J in this block
            // -----------------------------------------------------------------
            block_start = block * n;
            block_end = block_start + n;
            while (jj < J->p[j + 1] && J->i[jj] < block_start)
            {
                jj++;
            }

            block_jj_start = jj;

            while (jj < J->p[j + 1] && J->i[jj] < block_end)
            {
                jj++;
            }

            block_jj_end = jj;
            if (block_jj_end == block_jj_start)
            {
                continue;
            }

            // -----------------------------------------------------------------
            // gather the rows of A that hit this block's child entries
            // -----------------------------------------------------------------
            int n_cand = sorted_pattern_union(
                J->i + block_jj_start, block_jj_end - block_jj_start, block_start,
                A_csc->p, A_csc->i, stamp, cand, &gen);
            row_offset = block * m;
            for (int t = 0; t < n_cand; t++)
            {
                iVec_append(Ci, row_offset + cand[t]);
            }
        }
        Cp[j + 1] = Ci->len;
    }

    free_CSC_matrix(A_csc);
    sp_free(stamp);
    sp_free(cand);

    CSC_matrix *C = new_CSC_matrix(m * p, J->n, Ci->len);
    memcpy(C->p, Cp, (J->n + 1) * sizeof(int));
    memcpy(C->i, Ci->data, Ci->len * sizeof(int));
    sp_free(Cp);
    iVec_free(Ci);

    return C;
}

void old_block_left_multiply_fill_values(const CSR_matrix *A, const CSC_matrix *J,
                                         CSC_matrix *C)
{
    /* A is m x n, J is (n*p) x k, C is (m*p) x k */
    int m = A->m;
    int n = A->n;
    int k = J->n;

    int i, j, row_a, block, block_start, block_end, start, end;

    /* to get rid of unitialized warnings */
    block = 0;
    block_start = 0;
    block_end = 0;
    start = 0;
    end = 0;

    /* for each column of J (and C) */
    for (j = 0; j < k; j++)
    {
        int previous_block = -1;

        for (i = C->p[j]; i < C->p[j + 1]; i++)
        {
            /* choose row of A and block of column of J */
            row_a = C->i[i] % m;
            block = C->i[i] / m;

            // -------------------------------------------------------------------------
            //          find the part of the column of J in the current block
            // -------------------------------------------------------------------------
            if (block != previous_block)
            {
                previous_block = block;
                block_start = block * n;
                block_end = block_start + n;
                start = J->p[j];
                end = J->p[j + 1];

                while (start < J->p[j + 1] && J->i[start] < block_start)
                {
                    start++;
                }

                while (end > start && J->i[end - 1] >= block_end)
                {
                    end--;
                }
            }

            // ------------------------------------------------------------------------------
            // compute value as sparse dot product of row of A and column of J in
            // this block
            // ------------------------------------------------------------------------------
            int a_len = A->p[row_a + 1] - A->p[row_a];
            C->x[i] = sparse_dot_offset(A->x + A->p[row_a], A->i + A->p[row_a],
                                        a_len, J->x + start, J->i + start,
                                        end - start, block_start);
        }
    }
}

/* Fill values of C = A @ B where A is CSR_matrix, B is CSC_matrix. */
void old_csr_csc_matmul_fill_values(const CSR_matrix *A, const CSC_matrix *B,
                                    CSR_matrix *C)
{
    for (int i = 0; i < A->m; i++)
    {
        for (int jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            int j = C->i[jj];

            int a_nnz = A->p[i + 1] - A->p[i];
            int b_nnz = B->p[j + 1] - B->p[j];

            /* Compute dot product of row i of A and column j of B */
            double sum = sparse_dot_offset(A->x + A->p[i], A->i + A->p[i], a_nnz,
                                           B->x + B->p[j], B->i + B->p[j], b_nnz, 0);

            C->x[jj] = sum;
        }
    }
}

/* C = A @ B where A is CSR_matrix (m x n), B is CSC_matrix (n x p). Result C is
  CSR_matrix (m x p) with precomputed sparsity pattern */
CSR_matrix *old_csr_csc_matmul_alloc(const CSR_matrix *A, const CSC_matrix *B)
{
    int m = A->m;
    int p = B->n;

    int *Cp = (int *) sp_malloc((m + 1) * sizeof(int));
    iVec *Ci = iVec_new(m);

    Cp[0] = 0;

    /* Output-driven: column j of B intersects row i of A iff j appears in the
       CSR row list of one of A's row-i column indices, so gather those rows
       instead of testing all p columns per row of A. */
    int *iwork = (int *) sp_malloc((B->m > 0 ? B->m : 1) * sizeof(int));
    CSR_matrix *B_csr = csc_to_csr_alloc(B, iwork);
    sp_free(iwork);

    int *stamp = (int *) sp_calloc(p > 0 ? p : 1, sizeof(int)); /* 0 = unseen */
    int *cand = (int *) sp_malloc((p > 0 ? p : 1) * sizeof(int));
    int gen = 0;

    // --------------------------------------------------------------
    //            count nnz and fill column indices
    // --------------------------------------------------------------
    int nnz = 0;
    for (int i = 0; i < A->m; i++)
    {
        int n_cand = sorted_pattern_union(A->i + A->p[i], A->p[i + 1] - A->p[i], 0,
                                          B_csr->p, B_csr->i, stamp, cand, &gen);
        for (int t = 0; t < n_cand; t++)
        {
            iVec_append(Ci, cand[t]);
        }
        nnz += n_cand;

        Cp[i + 1] = nnz;
    }

    free_CSR_matrix(B_csr);
    sp_free(stamp);
    sp_free(cand);

    CSR_matrix *C = new_CSR_matrix(m, p, nnz);
    memcpy(C->p, Cp, (m + 1) * sizeof(int));
    memcpy(C->i, Ci->data, nnz * sizeof(int));
    sp_free(Cp);
    iVec_free(Ci);

    return C;
}
