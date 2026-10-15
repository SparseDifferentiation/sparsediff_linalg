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

/* Private A^T A kernels behind the sparse kind's ATA / ATDA dispatch. */

#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"

/* Fill sparsity of C = A^T D A for diagonal D. */
CSR_matrix *ATA_alloc(const CSC_matrix *A_csc, const CSR_matrix *A);

/* Compute values for C = A^T D A (NULL d corresponds to D as identity). acc is
   a workspace of length A->n. */
void ATDA_fill_values(const CSC_matrix *A_csc, const double *d, const CSR_matrix *A,
                      double *acc, CSR_matrix *C);

#endif /* INTERNAL_CSC_MATRIX_H */
