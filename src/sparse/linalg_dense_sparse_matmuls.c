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
#include "internal/iVec.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/cblas_wrapper.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------
 * C = (I_p kron A) @ J  via the polymorphic matrix interface.
 * A is dense m x n, J is (n*p) x k CSC_matrix, C is (m*p) x k CSC_matrix.
 * --------------------------------------------------------------- */
CSC_matrix *I_kron_A_alloc(const matrix *A, const CSC_matrix *J, int p)
{
    int m = A->m;
    int n = A->n;
    int i, j, jj, block, block_start, block_end, block_jj_start, row_offset;

    /* capacity hint based on J->nnz and not on the dense product */
    int *Cp = (int *) sp_malloc((J->n + 1) * sizeof(int));
    iVec *Ci = iVec_new(J->nnz > 0 ? J->nnz : 1);
    Cp[0] = 0;

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

            /* if no entries in this block, continue */
            if (jj == block_jj_start)
            {
                continue;
            }

            /* dense A: all m rows contribute */
            row_offset = block * m;
            for (i = 0; i < m; i++)
            {
                iVec_append(Ci, row_offset + i);
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

void I_kron_A_fill_values(const matrix *A, const CSC_matrix *J, CSC_matrix *C,
                          double *work)
{
    int m = A->m;
    int n = A->n;
    int k = J->n;

    int i, j, s, block, block_start, block_end, start, end;

    double *j_dense = work;

    /* for each column of J (and C) */
    for (j = 0; j < k; j++)
    {
        for (i = C->p[j]; i < C->p[j + 1]; i += m)
        {
            block = C->i[i] / m;
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

            int count = end - start;

            if (count == 1)
            {
                /* Fast path: C column segment = val * A[:, row_in_block] */
                int row_in_block = J->i[start] - block_start;
                double val = J->x[start];
                cblas_dcopy(m, A->x + row_in_block, n, C->x + i, 1);
                if (val != 1.0)
                {
                    cblas_dscal(m, val, C->x + i, 1);
                }
            }
            else
            {
                /* scatter sparse J col into dense vector and then
                 * compute A @ j_dense */
                memset(j_dense, 0, n * sizeof(double));
                for (s = start; s < end; s++)
                {
                    j_dense[J->i[s] - block_start] = J->x[s];
                }

                cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A->x, n, j_dense,
                            1, 0.0, C->x + i, 1);
            }
        }
    }
}
