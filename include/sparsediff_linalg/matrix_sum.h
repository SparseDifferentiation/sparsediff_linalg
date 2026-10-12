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
#ifndef MATRIX_SUM_H
#define MATRIX_SUM_H

#include "sparsediff_linalg/matrix.h"

/* Sums of matrix handles. Two-phase: *_alloc builds the output (into a
   caller-provided sparse handle, or returned for sum_4), *_fill_values writes
   its values and bumps its values_version. */

/* Polymorphic wrappers for allocating C = A + B. Right now we always
   convert to CSR matrices internally for the sum. */
void sum_matrices_alloc(matrix *A, matrix *B, matrix *C);

/* Fill values of C = A + B. Uses CSR matrices internally. */
void sum_matrices_fill_values(matrix *A, matrix *B, matrix *C);

/* Fill values of C = diag(d1) * A + diag(d2) * B. Uses CSR matrices internally. */
void sum_scaled_matrices_fill_values(matrix *A, matrix *B, matrix *C,
                                     const double *d1, const double *d2);

/* C = A + B + D + E for handles of any kind. The alloc merges the four patterns
   (read through to_csr) into a new sparse handle and binds, inside it, one index
   map per operand from the operand's native value order (op->x: CSR order for
   sparse, the dense block for permuted_dense, block-major for stacked_pd) to
   positions in C->x. The fill zeroes C, accumulates the four operands' op->x
   through those maps and bumps C; it never calls to_csr and never allocates.
   Operand kinds and patterns must not change between alloc and fill. */
matrix *sum_4_matrices_alloc(matrix *A, matrix *B, matrix *D, matrix *E);
void sum_4_matrices_fill_values(const matrix *A, const matrix *B, const matrix *D,
                                const matrix *E, matrix *C);

#endif /* MATRIX_SUM_H */
