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
#ifndef INTERNAL_CSC_MATRIX_H
#define INTERNAL_CSC_MATRIX_H

/* Private CSC kernels behind the sparse kind's ATA / ATDA vtable slots. */

#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"

/* Fill sparsity of C = A^T D A for diagonal D */
CSR_matrix *ATA_alloc(const CSC_matrix *A);

/* Compute values for C = A^T D A (null d corresponds to D as identity) */
void ATDA_fill_values(const CSC_matrix *A, const double *d, CSR_matrix *C);

#endif /* INTERNAL_CSC_MATRIX_H */
