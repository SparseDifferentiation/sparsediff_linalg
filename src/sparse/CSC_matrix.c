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
#include "sparsediff_linalg/CSC_matrix.h"
#include "internal/CSC_matrix.h"
#include "internal/gustavson.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

CSC_matrix *new_CSC_matrix(int m, int n, int nnz)
{
    CSC_matrix *matrix = (CSC_matrix *) sp_malloc(sizeof(CSC_matrix));
    if (!matrix) return NULL;

    matrix->p = (int *) sp_malloc((n + 1) * sizeof(int));
    matrix->i = (int *) sp_malloc(nnz * sizeof(int));
    matrix->x = (double *) sp_malloc(nnz * sizeof(double));

    if (!matrix->p || !matrix->i || !matrix->x)
    {
        sp_free(matrix->p);
        sp_free(matrix->i);
        sp_free(matrix->x);
        sp_free(matrix);
        return NULL;
    }

    matrix->m = m;
    matrix->n = n;
    matrix->nnz = nnz;

    return matrix;
}

void free_CSC_matrix(CSC_matrix *matrix)
{
    if (matrix)
    {
        sp_free(matrix->p);
        sp_free(matrix->i);
        sp_free(matrix->x);
        sp_free(matrix);
    }
}

CSC_matrix *csr_to_csc_alloc(const CSR_matrix *A, int *iwork)
{
    CSC_matrix *C = new_CSC_matrix(A->m, A->n, A->nnz);

    int i, j;
    int *count = iwork;
    memset(count, 0, A->n * sizeof(int));

    // -------------------------------------------------------------------
    //              compute nnz in each column of A
    // -------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            count[A->i[j]]++;
        }
    }

    // ------------------------------------------------------------------
    //                      compute column pointers
    // ------------------------------------------------------------------
    C->p[0] = 0;
    for (i = 0; i < A->n; ++i)
    {
        C->p[i + 1] = C->p[i] + count[i];
        count[i] = C->p[i];
    }

    // ------------------------------------------------------------------
    //                         fill row indices
    // ------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            C->i[count[A->i[j]]] = i;
            count[A->i[j]]++;
        }
    }

    return C;
}

void csr_to_csc_fill_values(const CSR_matrix *A, CSC_matrix *C, int *iwork)
{
    int i, j;
    int *count = iwork;
    memcpy(count, C->p, A->n * sizeof(int));

    // ------------------------------------------------------------------
    //                         fill values
    // ------------------------------------------------------------------
    for (i = 0; i < A->m; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            C->x[count[A->i[j]]] = A->x[j];
            count[A->i[j]]++;
        }
    }
}

CSR_matrix *csc_to_csr_alloc(const CSC_matrix *A, int *iwork)
{
    CSR_matrix *C = new_CSR_matrix(A->m, A->n, A->nnz);

    int i, j;
    int *count = iwork;
    memset(count, 0, A->m * sizeof(int));

    // -------------------------------------------------------------------
    //              compute nnz in each row of A, store in count
    // -------------------------------------------------------------------
    for (i = 0; i < A->n; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            assert(A->i[j] < A->m);
            count[A->i[j]]++;
        }
    }

    // ------------------------------------------------------------------
    //                      compute row pointers
    // ------------------------------------------------------------------
    C->p[0] = 0;
    for (i = 0; i < A->m; ++i)
    {
        C->p[i + 1] = C->p[i] + count[i];
        count[i] = C->p[i];
    }

    // ------------------------------------------------------------------
    //                         fill column indices
    // ------------------------------------------------------------------
    for (i = 0; i < A->n; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            assert(A->i[j] < A->m);
            C->i[count[A->i[j]]] = i;
            count[A->i[j]]++;
        }
    }

    return C;
}

void csc_to_csr_fill_values(const CSC_matrix *A, CSR_matrix *C, int *iwork)
{
    int i, j;
    int *count = iwork;
    memcpy(count, C->p, A->m * sizeof(int));

    // ------------------------------------------------------------------
    //                         fill values
    // ------------------------------------------------------------------
    for (i = 0; i < A->n; ++i)
    {
        for (j = A->p[i]; j < A->p[i + 1]; ++j)
        {
            assert(A->i[j] < A->m);
            C->x[count[A->i[j]]] = A->x[j];
            count[A->i[j]]++;
        }
    }
}

CSR_matrix *BTA_alloc(const CSR_matrix *A, const CSC_matrix *B)
{
    assert(A->m == B->m);
    sparse_rows L = {B->p, B->i, B->x, B->n};
    return gustavson_alloc(&L, 1, A);
}

void BTDA_fill_values(const CSR_matrix *A, const CSC_matrix *B, const double *d,
                      double *acc, CSR_matrix *C)
{
    sparse_rows L = {B->p, B->i, B->x, B->n};
    gustavson_fill_values(&L, 1, d, A, acc, C);
}

CSR_matrix *ATA_alloc(const CSC_matrix *A_csc, const CSR_matrix *A)
{
    assert(A_csc->m == A->m && A_csc->n == A->n);
    sparse_rows L = {A_csc->p, A_csc->i, A_csc->x, A_csc->n};
    return gustavson_alloc(&L, 1, A);
}

void ATDA_fill_values(const CSC_matrix *A_csc, const double *d, const CSR_matrix *A,
                      double *acc, CSR_matrix *C)
{
    sparse_rows L = {A_csc->p, A_csc->i, A_csc->x, A_csc->n};
    gustavson_fill_values(&L, 1, d, A, acc, C);
}
