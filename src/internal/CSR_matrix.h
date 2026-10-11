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
#ifndef INTERNAL_CSR_MATRIX_H
#define INTERNAL_CSR_MATRIX_H

/* Private CSR helpers used inside the library only. */

#include "sparsediff_linalg/CSR_matrix.h"

/* Allocate a CSR with A's sparsity pattern copied and values uninitialized. */
CSR_matrix *new_csr_copy_sparsity(const CSR_matrix *A);

/* fills values of C = diag(d) @ A */
void DA_fill_values(const double *d, const CSR_matrix *A, CSR_matrix *C);

/* get value at position (row, col) in A */
double csr_get_value(const CSR_matrix *A, int row, int col);

/* Expand symmetric CSR_matrix matrix A to full matrix C. A is assumed to store
   only upper triangle. C must be pre-allocated with sufficient nnz */
void symmetrize_csr(const int *Ap, const int *Ai, int m, CSR_matrix *C);

#endif /* INTERNAL_CSR_MATRIX_H */
