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

/* Fill sparsity of C = A + B. Uses CSR matrices internally. */
void sum_matrices_alloc(matrix *A, matrix *B, matrix *C);

/* Fill values of C = A + B. Uses CSR matrices internally. */
void sum_matrices_fill_values(matrix *A, matrix *B, matrix *C);

/* Fill values of C = diag(d1) * A + diag(d2) * B. Uses CSR matrices internally. */
void sum_scaled_matrices_fill_values(matrix *A, matrix *B, matrix *C,
                                     const double *d1, const double *d2);

/* Fill sparsity of C = A + B + D + E. Uses CSR matrices internally. */
matrix *sum_4_matrices_alloc(matrix *A, matrix *B, matrix *D, matrix *E);

/* Fill values of C = A + B + D + E. Uses CSR matrices internally. */
void sum_4_matrices_fill_values(const matrix *A, const matrix *B, const matrix *D,
                                const matrix *E, matrix *C);

/* Fill sparsity of C = ops[0] + ... + ops[n-1] (n >= 1, any kinds, same shape).
   Uses CSR matrices internally; C is a sparse_matrix. */
matrix *sum_n_matrices_alloc(matrix *const *ops, int n);

/* Fill values of C = ops[0] + ... + ops[n-1] through the maps bound at alloc;
   ops must be the same array as passed to sum_n_matrices_alloc. */
void sum_n_matrices_fill_values(matrix *const *ops, int n, matrix *C);

/* Start an incremental fill of C (from sum_n_matrices_alloc): zero its values.
   Then call sum_n_matrices_fill_add once per operand, each right after that
   operand's values are final. Unlike sum_n_matrices_fill_values, this stays
   correct when evaluating a later operand overwrites an earlier one's values
   (operands sharing a subexpression). */
void sum_n_matrices_fill_zero(matrix *C);

/* C += op, where op is operand k of the sum_n_matrices_alloc that built C. Reads
   op->x through the map bound at alloc; bumps C's values_version. */
void sum_n_matrices_fill_add(const matrix *op, int k, matrix *C);

#endif /* MATRIX_SUM_H */
