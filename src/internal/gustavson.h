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
#ifndef INTERNAL_GUSTAVSON_H
#define INTERNAL_GUSTAVSON_H

#include "sparsediff_linalg/CSR_matrix.h"

/* Rows of a compressed matrix (a CSR's rows or a CSC's columns), without a
   copy: row i has entries (i[ii], x[ii]) for p[i] <= ii < p[i + 1]. */
typedef struct sparse_rows
{
    const int *p;
    const int *i;
    const double *x;
    int m;
} sparse_rows;

/* Row-wise sparse x sparse core: C = kron(I_blocks, L) D A in CSR, with A in
   CSR and D diagonal (NULL d: identity). blocks == 1 is C = L D A; otherwise A
   has blocks * n_L rows and block b of C is L D_b A_b. */

/* Allocate C ((blocks * L->m) x A->n) with the exact sparsity, rows sorted. */
CSR_matrix *gustavson_alloc(const sparse_rows *L, int blocks, const CSR_matrix *A);

/* Fill C->x. acc is a workspace of length A->n. */
void gustavson_fill_values(const sparse_rows *L, int blocks, const double *d,
                           const CSR_matrix *A, double *acc, CSR_matrix *C);

#endif /* INTERNAL_GUSTAVSON_H */
