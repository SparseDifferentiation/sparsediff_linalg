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
#ifndef MATRIX_VSTACK_H
#define MATRIX_VSTACK_H

#include "sparsediff_linalg/matrix.h"

/* Allocate C = [ops[0]; ops[1]; ...; ops[n-1]] (n >= 1, any kinds, all with
   the same number of columns). C is a sparse_matrix whose rows are the
   operands' rows in order. Reads the operands' storage directly: no operand
   cache is read or created. */
matrix *vstack_matrices_alloc(matrix *const *ops, int n);

/* Fill values of C = [ops[0]; ...; ops[n-1]]; ops must be the same array as
   passed to vstack_matrices_alloc (same matrices, same order). Copies only the
   operands whose values_version changed since the previous fill, and bumps C's
   values_version iff it copied something. If C->x was written directly (and
   matrix_values_changed(C) called), the next fill copies every operand. */
void vstack_matrices_fill_values(matrix *const *ops, int n, matrix *C);

#endif /* MATRIX_VSTACK_H */
