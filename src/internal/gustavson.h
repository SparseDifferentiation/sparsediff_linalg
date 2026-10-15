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

/* Private row-wise sparse x sparse core (Gustavson): C = L D A in CSR, with A
   in CSR and D diagonal (d == NULL means identity).

   The left operand L (m_out x A->m) is given row by row as sparse lists over a
   compressed triplet (lp, li, lx): row i of L has entries (li[ii], lx[ii]) for
   lp[i] <= ii < lp[i + 1], with li[ii] in [0, A->m). Any compressed storage has
   this shape, so the callers pick what row i means:
     - B^T A:  B's CSC mirror (column i of B is row i of B^T),
     - A^T A:  A's own CSC mirror,
     - B A:    B's CSR (row i of B).
   Row i of C is the union of A's rows indexed by L's row i; its values are
   sum_k lx[k] d[k] A[k, :]. Rows of C are sorted by column. */

#include "sparsediff_linalg/CSR_matrix.h"

/* Symbolic phase: allocate C (m_out x A->n) with the exact sparsity of L A,
   each row sorted. Two passes over a marker array of length A->n (count, then
   fill), so C is allocated at its final size. lx is not needed. */
CSR_matrix *gustavson_alloc(const int *lp, const int *li, int m_out,
                            const CSR_matrix *A);

/* Numeric phase: fill C->x for C allocated by gustavson_alloc from the same
   (lp, li) and the same sparsity of A. acc is a caller-owned workspace of
   length A->n; its contents on entry are irrelevant. Per row: zero acc at C's
   pattern positions, scatter-add lx[ii] * d[k] * A[k, :], gather back along
   C's row. No marker, no sort, no allocation. */
void gustavson_fill_values(const int *lp, const int *li, const double *lx,
                           const double *d, const CSR_matrix *A, double *acc,
                           CSR_matrix *C);

#endif /* INTERNAL_GUSTAVSON_H */
