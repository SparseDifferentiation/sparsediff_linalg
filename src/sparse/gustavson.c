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
#include "internal/gustavson.h"
#include "internal/utils.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <assert.h>
#include <stdlib.h>

CSR_matrix *gustavson_alloc(const sparse_rows *L, int blocks, const CSR_matrix *A)
{
    int n = A->n;
    int m_out = L->m * blocks;
    int n_L = A->m / blocks;
    int i, j, k, r, b, ii, jj;
    assert(blocks >= 1 && A->m == n_L * blocks);

    /* mark[j]: last row of C that saw column j. Pass 2 stamps m_out + i, so no
       reset is needed between the passes. */
    int *mark = (int *) sp_malloc(n * sizeof(int));
    int *row_nnz = (int *) sp_malloc(m_out * sizeof(int));
    for (j = 0; j < n; j++)
    {
        mark[j] = -1;
    }

    /* Pass 1: count the distinct columns of each row of C. */
    int nnz = 0;
    for (b = 0; b < blocks; b++)
    {
        for (r = 0; r < L->m; r++)
        {
            i = b * L->m + r;
            int count = 0;
            for (ii = L->p[r]; ii < L->p[r + 1]; ii++)
            {
                k = L->i[ii] + b * n_L;
                assert(0 <= k && k < A->m);
                for (jj = A->p[k]; jj < A->p[k + 1]; jj++)
                {
                    j = A->i[jj];
                    if (mark[j] != i)
                    {
                        mark[j] = i;
                        count++;
                    }
                }
            }
            row_nnz[i] = count;
            nnz += count;
        }
    }

    CSR_matrix *C = new_CSR_matrix(m_out, n, nnz);
    C->p[0] = 0;
    for (i = 0; i < m_out; i++)
    {
        C->p[i + 1] = C->p[i] + row_nnz[i];
    }
    sp_free(row_nnz);

    /* Pass 2: write each row's columns, then sort the row. */
    for (b = 0; b < blocks; b++)
    {
        for (r = 0; r < L->m; r++)
        {
            i = b * L->m + r;
            int pos = C->p[i];
            for (ii = L->p[r]; ii < L->p[r + 1]; ii++)
            {
                k = L->i[ii] + b * n_L;
                for (jj = A->p[k]; jj < A->p[k + 1]; jj++)
                {
                    j = A->i[jj];
                    if (mark[j] != m_out + i)
                    {
                        mark[j] = m_out + i;
                        C->i[pos++] = j;
                    }
                }
            }
            assert(pos == C->p[i + 1]);
            sort_int_array(C->i + C->p[i], C->p[i + 1] - C->p[i]);
        }
    }
    sp_free(mark);

    return C;
}

void gustavson_fill_values(const sparse_rows *L, int blocks, const double *d,
                           const CSR_matrix *A, double *acc, CSR_matrix *C)
{
    int n_L = A->m / blocks;
    int i, k, r, b, ii, jj;
    assert(C->n == A->n && C->m == L->m * blocks && A->m == n_L * blocks);

    for (b = 0; b < blocks; b++)
    {
        for (r = 0; r < L->m; r++)
        {
            i = b * L->m + r;

            /* C's row is exactly the union scattered below, so this zeroes
               every acc entry the scatter touches. */
            for (jj = C->p[i]; jj < C->p[i + 1]; jj++)
            {
                acc[C->i[jj]] = 0.0;
            }

            for (ii = L->p[r]; ii < L->p[r + 1]; ii++)
            {
                k = L->i[ii] + b * n_L;
                double v = (d != NULL) ? L->x[ii] * d[k] : L->x[ii];
                for (jj = A->p[k]; jj < A->p[k + 1]; jj++)
                {
                    acc[A->i[jj]] += v * A->x[jj];
                }
            }

            for (jj = C->p[i]; jj < C->p[i + 1]; jj++)
            {
                C->x[jj] = acc[C->i[jj]];
            }
        }
    }
}
