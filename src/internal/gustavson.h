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

/* Row-wise sparse x sparse core: C = L D A in CSR, with A in CSR and D
   diagonal (NULL d: identity). The left operand L is given row by row as
   sparse lists (lp, li, lx): row i has entries (li[ii], lx[ii]) for
   lp[i] <= ii < lp[i + 1]. A CSC column (B^T A, A^T A) and a CSR row (B A)
   both have this shape. */

#include "sparsediff_linalg/CSR_matrix.h"

/* Allocate C (m_out x A->n) with the exact sparsity of L A, rows sorted. */
CSR_matrix *gustavson_alloc(const int *lp, const int *li, int m_out,
                            const CSR_matrix *A);

/* Fill C->x. acc is a workspace of length A->n. */
void gustavson_fill_values(const int *lp, const int *li, const double *lx,
                           const double *d, const CSR_matrix *A, double *acc,
                           CSR_matrix *C);

#endif /* INTERNAL_GUSTAVSON_H */
