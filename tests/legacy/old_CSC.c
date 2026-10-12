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
#include "internal/iVec.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <string.h>

/* Copy of the library's private sparse_dot (src/sparse/CSC_matrix.c). */
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
