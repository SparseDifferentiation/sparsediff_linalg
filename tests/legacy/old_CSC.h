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
#ifndef OLD_CSC_H
#define OLD_CSC_H

/* LEGACY: kept only as a test oracle for the library's own kernels. Compiled
   into sparsediff_linalg_tests, never into the library. Not part of the API.
   symBA_alloc, BA_fill_values and yTA_fill_values were the engine's quad_form /
   quad_over_lin helpers, retired by the matrix dispatchers (Q J_f as B^T A with
   Q the transposed operand). The old_* kernels are the dot-product sparse x
   sparse kernels retired by the Gustavson core (src/sparse/gustavson.c);
   old_csr_csc_matmul_* (B A) and old_block_left_multiply_* (kron(I_p, A) J, CSC
   output) likewise. */

#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/CSR_matrix.h"

/* Fill sparsity of C = BA, where B is symmetric. */
CSC_matrix *symBA_alloc(const CSR_matrix *B, const CSC_matrix *A);

/* Fill values of C = BA. The matrix B does not have to be symmetric */
void BA_fill_values(const CSR_matrix *B, const CSC_matrix *A, CSC_matrix *C);

/* Count nonzero columns of a CSC_matrix matrix */
int count_nonzero_cols_csc(const CSC_matrix *A);

/* Fill values of C = x^T A. The matrix C must have filled sparsity. */
void yTA_fill_values(const CSC_matrix *A, const double *x, CSR_matrix *C);

/* Fill sparsity of C = A^T D A for diagonal D. */
CSR_matrix *old_ATA_alloc(const CSC_matrix *A);

/* Compute values for C = A^T D A (NULL d corresponds to D as identity). */
void old_ATDA_fill_values(const CSC_matrix *A, const double *d, CSR_matrix *C);

/* Fill sparsity of C = B^T D A for diagonal D. */
CSR_matrix *old_BTA_alloc(const CSC_matrix *A, const CSC_matrix *B);

/* Compute values for C = B^T D A (NULL d corresponds to D as identity). */
void old_BTDA_fill_values(const CSC_matrix *A, const CSC_matrix *B, const double *d,
                          CSR_matrix *C);

/* C = A @ B, A CSR and B CSC, C CSR. */
CSR_matrix *old_csr_csc_matmul_alloc(const CSR_matrix *A, const CSC_matrix *B);
void old_csr_csc_matmul_fill_values(const CSR_matrix *A, const CSC_matrix *B,
                                    CSR_matrix *C);

/* C = kron(I_p, A) @ J with A m x n, J (n*p) x k in CSC; C is (m*p) x k in CSC. */
CSC_matrix *old_block_left_multiply_fill_sparsity(const CSR_matrix *A,
                                                  const CSC_matrix *J, int p);
void old_block_left_multiply_fill_values(const CSR_matrix *A, const CSC_matrix *J,
                                         CSC_matrix *C);

#endif /* OLD_CSC_H */
