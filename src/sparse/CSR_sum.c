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
#include "internal/csr_kernels.h"
#include "internal/utils.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

void sum_csr_alloc(const CSR_matrix *A, const CSR_matrix *B, CSR_matrix *C)
{
    /* A and B must be different from C */
    assert(A != C && B != C);

    int cursor = 0;

    for (int row = 0; row < A->m; row++)
    {
        int a_ptr = A->p[row];
        int a_end = A->p[row + 1];
        int b_ptr = B->p[row];
        int b_end = B->p[row + 1];
        C->p[row] = cursor;

        /* Merge while both have elements (only column indices) */
        while (a_ptr < a_end && b_ptr < b_end)
        {
            if (A->i[a_ptr] < B->i[b_ptr])
            {
                C->i[cursor] = A->i[a_ptr];
                a_ptr++;
            }
            else if (B->i[b_ptr] < A->i[a_ptr])
            {
                C->i[cursor] = B->i[b_ptr];
                b_ptr++;
            }
            else
            {
                C->i[cursor] = A->i[a_ptr];
                a_ptr++;
                b_ptr++;
            }
            cursor++;
        }

        /* Copy remaining elements from A */
        if (a_ptr < a_end)
        {
            int a_remaining = a_end - a_ptr;
            memcpy(C->i + cursor, A->i + a_ptr, a_remaining * sizeof(int));
            cursor += a_remaining;
        }

        /* Copy remaining elements from B */
        if (b_ptr < b_end)
        {
            int b_remaining = b_end - b_ptr;
            memcpy(C->i + cursor, B->i + b_ptr, b_remaining * sizeof(int));
            cursor += b_remaining;
        }
    }

    C->p[A->m] = cursor;
    C->nnz = cursor;
}

void sum_csr_fill_values(const CSR_matrix *A, const CSR_matrix *B, CSR_matrix *C)
{
    /* Assumes C->p and C->i already contain the sparsity pattern of A+B.
       Fills only C->x accordingly. */
    for (int row = 0; row < A->m; row++)
    {
        int a_ptr = A->p[row];
        int a_end = A->p[row + 1];
        int b_ptr = B->p[row];
        int b_end = B->p[row + 1];

        for (int c_ptr = C->p[row]; c_ptr < C->p[row + 1]; c_ptr++)
        {
            int col = C->i[c_ptr];
            double val = 0.0;

            if (a_ptr < a_end && A->i[a_ptr] == col)
            {
                val += A->x[a_ptr];
                a_ptr++;
            }

            if (b_ptr < b_end && B->i[b_ptr] == col)
            {
                val += B->x[b_ptr];
                b_ptr++;
            }
            C->x[c_ptr] = val;
        }
    }
}

void sum_scaled_csr_matrices_fill_values(const CSR_matrix *A, const CSR_matrix *B,
                                         CSR_matrix *C, const double *d1,
                                         const double *d2)
{
    /* Assumes C->p and C->i already contain the sparsity pattern of A+B.
       Fills only C->x accordingly with scaling. */
    for (int row = 0; row < A->m; row++)
    {
        int a_ptr = A->p[row];
        int a_end = A->p[row + 1];
        int b_ptr = B->p[row];
        int b_end = B->p[row + 1];

        for (int c_ptr = C->p[row]; c_ptr < C->p[row + 1]; c_ptr++)
        {
            int col = C->i[c_ptr];
            double val = 0.0;

            if (a_ptr < a_end && A->i[a_ptr] == col)
            {
                val += d1[row] * A->x[a_ptr];
                a_ptr++;
            }

            if (b_ptr < b_end && B->i[b_ptr] == col)
            {
                val += d2[row] * B->x[b_ptr];
                b_ptr++;
            }
            C->x[c_ptr] = val;
        }
    }
}

void accumulator(const double *vals, int nnz, const int *idx_map, double *out)
{
    /* don't forget to initialize accumulator to 0 before calling this */
    for (int j = 0; j < nnz; j++)
    {
        out[idx_map[j]] += vals[j];
    }
}

CSR_matrix *sum_4_csr_alloc(const CSR_matrix *A, const CSR_matrix *B,
                            const CSR_matrix *C, const CSR_matrix *D,
                            int *idx_maps[4])
{
    const CSR_matrix *inputs[4] = {A, B, C, D};
    int m = A->m;
    int n = A->n;
    int nnz_ub = MIN(A->nnz + B->nnz + C->nnz + D->nnz, sat_mul_int(m, n));

    /* allocate output and index maps */
    CSR_matrix *out = new_CSR_matrix(m, n, nnz_ub);
    for (int k = 0; k < 4; k++)
    {
        idx_maps[k] = (int *) sp_malloc(inputs[k]->nnz * sizeof(int));
    }

    /* 4-way sorted merge per row */
    int ptrs[4], ends[4];
    int nnz = 0;

    for (int row = 0; row < m; row++)
    {
        out->p[row] = nnz;
        for (int k = 0; k < 4; k++)
        {
            ptrs[k] = inputs[k]->p[row];
            ends[k] = inputs[k]->p[row + 1];
        }

        for (;;)
        {

            /* find minimum column in the current row among the 4 inputs */
            int min_col = -1;
            for (int k = 0; k < 4; k++)
            {
                if (ptrs[k] < ends[k])
                {
                    int col = inputs[k]->i[ptrs[k]];
                    if (min_col == -1 || col < min_col)
                    {
                        min_col = col;
                    }
                }
            }

            /* if no elements in the current row, the output row will be empty */
            if (min_col == -1)
            {
                break;
            }

            /* insert min_col into output and update idx_maps */
            out->i[nnz] = min_col;
            for (int k = 0; k < 4; k++)
            {
                if (ptrs[k] < ends[k] && inputs[k]->i[ptrs[k]] == min_col)
                {
                    idx_maps[k][ptrs[k]++] = nnz;
                }
            }
            nnz++;
        }
    }

    out->p[m] = nnz;
    out->nnz = nnz;
    CSR_trim(out);
    return out;
}

CSR_matrix *sum_n_csr_alloc(const CSR_matrix *const *ops, int n, int *idx_map)
{
    assert(n >= 1);
    int m = ops[0]->m;
    int n_cols = ops[0]->n;
    long long total_nnz = 0; /* n operands can overflow an int sum */
    for (int k = 0; k < n; k++)
    {
        assert(ops[k]->m == m && ops[k]->n == n_cols);
        total_nnz += ops[k]->nnz;
    }
    int nnz_ub = (int) MIN(total_nnz, (long long) sat_mul_int(m, n_cols));

    CSR_matrix *out = new_CSR_matrix(m, n_cols, nnz_ub);

    /* pos[j] is column j's position in out. A value below the current row's
       start was set in an earlier row, so it marks column j unseen in this
       row. */
    int *pos = (int *) sp_malloc(n_cols * sizeof(int));
    for (int j = 0; j < n_cols; j++)
    {
        pos[j] = -1;
    }

    int nnz = 0;
    for (int i = 0; i < m; i++)
    {
        int start = nnz;
        out->p[i] = start;

        /* gather the row's unique columns, then sort them */
        for (int k = 0; k < n; k++)
        {
            for (int jj = ops[k]->p[i]; jj < ops[k]->p[i + 1]; jj++)
            {
                int j = ops[k]->i[jj];
                if (pos[j] < start)
                {
                    pos[j] = start;
                    out->i[nnz++] = j;
                }
            }
        }
        sort_int_array(out->i + start, nnz - start);

        /* record positions and fill the index maps */
        for (int jj = start; jj < nnz; jj++)
        {
            pos[out->i[jj]] = jj;
        }
        int *map_k = idx_map;
        for (int k = 0; k < n; k++)
        {
            for (int jj = ops[k]->p[i]; jj < ops[k]->p[i + 1]; jj++)
            {
                map_k[jj] = pos[ops[k]->i[jj]];
            }
            map_k += ops[k]->nnz;
        }
    }

    sp_free(pos);
    out->p[m] = nnz;
    out->nnz = nnz;
    CSR_trim(out);
    return out;
}
