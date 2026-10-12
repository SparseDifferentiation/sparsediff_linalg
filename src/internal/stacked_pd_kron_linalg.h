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
#ifndef STACKED_PD_KRON_LINALG_H
#define STACKED_PD_KRON_LINALG_H

#include "internal/permuted_dense.h"
#include "internal/stacked_pd.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/matrix.h"

/* All three families compute C = kron(I_p, A) @ J or C = kron(A, I_p) @ J as a
   stacked_pd, without materializing the Kronecker product. The layout picks
   which: under KRON_I_A block k holds output rows k*m0 .. and reads J's rows
   k*n0 ..; under KRON_A_I block i holds output rows i + p*r and reads J's rows
   i + p*c, so every row of a block shares one pattern.

   Contract on A: A must be a "full" permuted_dense — m0 == base.m,
   n0 == base.n, and row_perm / col_perm are the identity — i.e., a
   local dense matrix wrapped in the permuted_dense struct. */
typedef enum
{
    KRON_I_A, /* kron(I_p, A) */
    KRON_A_I  /* kron(A, I_p) */
} kron_layout;

/* J is CSC. */
matrix *BA_dense_kron_csc_alloc(const permuted_dense *A, int p, const CSC_matrix *J,
                                kron_layout layout);
void BA_dense_kron_csc_fill_values(const permuted_dense *A, int p,
                                   const CSC_matrix *J, stacked_pd *C,
                                   kron_layout layout);

/* J is permuted_dense. */
matrix *BA_dense_kron_pd_alloc(const permuted_dense *A, int p,
                               const permuted_dense *J, kron_layout layout);
void BA_dense_kron_pd_fill_values(const permuted_dense *A, int p,
                                  const permuted_dense *J, stacked_pd *C,
                                  kron_layout layout);

/* J is stacked_pd. */
matrix *BA_dense_kron_spd_alloc(const permuted_dense *A, int p, const stacked_pd *J,
                                kron_layout layout);
void BA_dense_kron_spd_fill_values(const permuted_dense *A, int p,
                                   const stacked_pd *J, stacked_pd *C,
                                   kron_layout layout);

#endif /* STACKED_PD_KRON_LINALG_H */
