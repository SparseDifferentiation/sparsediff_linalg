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
#include "legacy/old_CSR.h"
#include "internal/iVec.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <assert.h>
#include <string.h>

CSR_matrix *block_diag_repeat_csr(const CSR_matrix *A, int p)
{
    assert(p > 0);

    int m = A->m;
    int n = A->n;
    int nnz = A->nnz;

    CSR_matrix *A_kron = new_CSR_matrix(m * p, n * p, nnz * p);

    int nnz_cursor = 0;
    for (int block = 0; block < p; block++)
    {
        int row_offset = block * m;
        int col_offset = block * n;

        for (int row = 0; row < m; row++)
        {
            int dest_row = row_offset + row;
            A_kron->p[dest_row] = nnz_cursor;

            for (int j = A->p[row]; j < A->p[row + 1]; j++)
            {
                A_kron->i[nnz_cursor] = A->i[j] + col_offset;
                A_kron->x[nnz_cursor] = A->x[j];
                nnz_cursor++;
            }

            A_kron->p[dest_row + 1] = nnz_cursor;
        }
    }

    return A_kron;
}

CSR_matrix *kron_identity_csr(const CSR_matrix *A, int p)
{
    assert(p > 0);

    int m = A->m;
    int n = A->n;
    int nnz = A->nnz;

    CSR_matrix *A_kron = new_CSR_matrix(m * p, n * p, nnz * p);

    int nnz_cursor = 0;
    for (int row_block = 0; row_block < m; row_block++)
    {
        for (int diag_idx = 0; diag_idx < p; diag_idx++)
        {
            int dest_row = row_block * p + diag_idx;
            A_kron->p[dest_row] = nnz_cursor;

            /* Copy entries from row_block of A, adjusting column indices */
            for (int j = A->p[row_block]; j < A->p[row_block + 1]; j++)
            {
                int col_block = A->i[j];
                /* Column in result: col_block * p + diag_idx */
                A_kron->i[nnz_cursor] = col_block * p + diag_idx;
                A_kron->x[nnz_cursor] = A->x[j];
                nnz_cursor++;
            }

            A_kron->p[dest_row + 1] = nnz_cursor;
        }
    }

    return A_kron;
}

void Ax_csr_fill_values(const CSR_matrix *AT, const double *z, CSR_matrix *C)
{
    int A_ncols = AT->m;

    for (int i = 0; i < A_ncols; i++)
    {
        double val = 0;
        for (int j = AT->p[i]; j < AT->p[i + 1]; j++)
        {
            val += z[AT->i[j]] * AT->x[j];
        }

        if (AT->p[i + 1] - AT->p[i] == 0) continue;

        // find position in C
        for (int k = 0; k < C->nnz; k++)
        {
            if (C->i[k] == i)
            {
                C->x[k] = val;
                break;
            }
        }
    }
}

void csr_insert_value(CSR_matrix *A, int col_idx, double value)
{
    assert(A->m == 1);

    for (int j = 0; j < A->nnz; j++)
    {
        assert(col_idx != A->i[j]);

        if (col_idx < A->i[j])
        {
            /* move the rest of the elements */
            memmove(A->i + (j + 1), A->i + j, (A->nnz - j) * sizeof(int));
            memmove(A->x + (j + 1), A->x + j, (A->nnz - j) * sizeof(double));

            /* insert new value */
            A->i[j] = col_idx;
            A->x[j] = value;
            A->nnz++;
            return;
        }
    }

    /* if we get here it should be inserted in the end */
    A->i[A->nnz] = col_idx;
    A->x[A->nnz] = value;
    A->nnz++;
}

void Ax_csr_wo_offset(const CSR_matrix *A, const double *x, double *y)
{
    for (int row = 0; row < A->m; row++)
    {
        double sum = 0.0;
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            sum += A->x[j] * x[A->i[j]];
        }
        y[row] = sum;
    }
}

void diag_csr_mult(const double *d, const CSR_matrix *A, CSR_matrix *C)
{
    copy_CSR_matrix(A, C);

    for (int row = 0; row < C->m; row++)
    {
        for (int j = C->p[row]; j < C->p[row + 1]; j++)
        {
            C->x[j] *= d[row];
        }
    }
}

int count_nonzero_cols(const CSR_matrix *A, bool *col_nz)
{
    for (int row = 0; row < A->m; row++)
    {
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            col_nz[A->i[j]] = true;
        }
    }
    int count = 0;
    for (int col = 0; col < A->n; col++)
    {
        if (col_nz[col]) count++;
    }

    return count;
}

void insert_idx(int idx, int *arr, int len)
{
    int j = 0;

    while (j < len && arr[j] < idx)
    {
        j++;
    }

    // move elements to the right
    memmove(arr + (j + 1), arr + j, (len - j) * sizeof(int));
    arr[j] = idx;
}

void Ax_csr(const CSR_matrix *A, const double *x, double *y, int col_offset)
{
    for (int row = 0; row < A->m; row++)
    {
        double sum = 0.0;
        for (int j = A->p[row]; j < A->p[row + 1]; j++)
        {
            sum += A->x[j] * x[A->i[j] - col_offset];
        }
        y[row] = sum;
    }
}

/* ---------------------------------------------------------------
 * C = (Y^T kron I_m) @ J
 * Y is k x n (col-major), J is (m*k) x p CSC_matrix, C is (m*n) x p CSR_matrix
 * --------------------------------------------------------------- */
CSR_matrix *YT_kron_I_alloc(int m, int k, int n, const CSC_matrix *J)
{
    (void) k;
    /* C has n blocks of m rows.  All rows at the same position within
       their block (same blk_row) share the same column sparsity.

       Step 1: for each blk_row, find which columns of J contribute.
               Column j contributes iff J[:,j] has any nonzero r
               with r % m == blk_row.
       Step 2: replicate each blk_row's pattern across all n blocks. */

    int i, j, ii, blk_row, total_nnz;

    // ---------------------------------------------------------------
    //           build sparsity pattern per blk_row
    // ---------------------------------------------------------------
    iVec **pattern = (iVec **) sp_malloc(m * sizeof(iVec *));
    total_nnz = 0;
    for (blk_row = 0; blk_row < m; blk_row++)
    {
        pattern[blk_row] = iVec_new(J->n);

        /* check each column of J */
        for (j = 0; j < J->n; j++)
        {
            for (ii = J->p[j]; ii < J->p[j + 1]; ii++)
            {
                if (J->i[ii] % m == blk_row)
                {
                    iVec_append(pattern[blk_row], j);
                    break;
                }
            }
        }
        total_nnz += pattern[blk_row]->len * n;
    }

    // ---------------------------------------------------------------
    //           replicate sparsity pattern across blocks
    // ---------------------------------------------------------------
    CSR_matrix *C = new_CSR_matrix(m * n, J->n, total_nnz);
    int idx = 0;
    for (i = 0; i < m * n; i++)
    {
        blk_row = i % m;
        C->p[i] = idx;
        int len = pattern[blk_row]->len;
        memcpy(C->i + idx, pattern[blk_row]->data, len * sizeof(int));
        idx += len;
    }
    C->p[m * n] = idx;
    assert(idx == total_nnz);

    for (blk_row = 0; blk_row < m; blk_row++)
    {
        iVec_free(pattern[blk_row]);
    }
    sp_free(pattern);
    return C;
}

void YT_kron_I_fill_values(int m, int k, int n, const double *Y, const CSC_matrix *J,
                           CSR_matrix *C)
{
    (void) n;
    assert(C->m == m * n);
    /* C[i, j] = sum_l Y[l, blk] * J[blk_row + l*m, j]
     * where blk_row = i % m, blk = i / m */

    int i, j, ii, jj, blk, blk_row;
    double sum;

    /* for each row i of C */
    for (i = 0; i < C->m; i++)
    {
        blk = i / m;                       /* which block of C */
        blk_row = i % m;                   /* which row within block */
        const double *Y_col = Y + blk * k; /* column blk of Y */

        /* for each column j in row i of C */
        for (jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            j = C->i[jj];
            sum = 0.0;

            /* matching J nonzeros in this column */
            for (ii = J->p[j]; ii < J->p[j + 1]; ii++)
            {
                if (J->i[ii] % m == blk_row)
                {
                    sum += Y_col[J->i[ii] / m] * J->x[ii];
                }
            }
            C->x[jj] = sum;
        }
    }
}

CSR_matrix *I_kron_X_alloc(int m, int k, int n, const CSC_matrix *J)
{
    /* Step 1: for each block, find which columns of J have any
     *         nonzero in row range [blk*k, blk*k + k). */
    int i, j, ii, blk;

    iVec **pattern = (iVec **) sp_malloc(n * sizeof(iVec *));
    int total_nnz = 0;
    for (blk = 0; blk < n; blk++)
    {
        int blk_start = blk * k;
        int blk_end = blk_start + k;
        pattern[blk] = iVec_new(J->n);

        /* check each column of J */
        for (j = 0; j < J->n; j++)
        {
            for (ii = J->p[j]; ii < J->p[j + 1]; ii++)
            {
                if (J->i[ii] >= blk_start && J->i[ii] < blk_end)
                {
                    iVec_append(pattern[blk], j);
                    break;
                }
            }
        }
        total_nnz += (int) pattern[blk]->len * m;
    }

    /* Step 2: replicate each block's pattern for all m rows
     *         within that block. */
    CSR_matrix *C = new_CSR_matrix(m * n, J->n, total_nnz);
    int idx = 0;
    for (i = 0; i < m * n; i++)
    {
        blk = i / m;
        C->p[i] = idx;
        int len = (int) pattern[blk]->len;
        memcpy(C->i + idx, pattern[blk]->data, len * sizeof(int));
        idx += len;
    }
    C->p[m * n] = idx;
    assert(idx == total_nnz);

    for (blk = 0; blk < n; blk++)
    {
        iVec_free(pattern[blk]);
    }
    sp_free(pattern);
    return C;
}

void I_kron_X_fill_values(int m, int k, int n, const double *X, const CSC_matrix *J,
                          CSR_matrix *C)
{
    (void) n;
    assert(C->m == m * n);
    /* C[i, j] = sum_l X[blk_row + l*m] * J[blk*k + l, j]
     * where blk = i / m, blk_row = i % m */

    int i, j, ii, jj, blk, blk_row;
    double sum;

    /* for each row i of C */
    for (i = 0; i < C->m; i++)
    {
        blk = i / m;     /* which block of C */
        blk_row = i % m; /* which row within block */
        int blk_start = blk * k;
        int blk_end = blk_start + k;

        /* for each column j in row i of C */
        for (jj = C->p[i]; jj < C->p[i + 1]; jj++)
        {
            j = C->i[jj];
            sum = 0.0;

            /* J nonzeros in column j within this block's row range */
            for (ii = J->p[j]; ii < J->p[j + 1]; ii++)
            {
                int r = J->i[ii];
                if (r >= blk_start && r < blk_end)
                {
                    int l = r - blk_start;
                    sum += X[blk_row + l * m] * J->x[ii];
                }
            }
            C->x[jj] = sum;
        }
    }
}
