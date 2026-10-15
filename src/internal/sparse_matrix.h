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
#ifndef INTERNAL_SPARSE_MATRIX_H
#define INTERNAL_SPARSE_MATRIX_H

/* Private: the sparse kind's layout. Consumers see only matrix.h and the
   constructors, both in matrix.h. */

#include "sparsediff_linalg/matrix.h"

/* Sparse matrix wrapping CSR_matrix. csc_cache is a lazily-built CSC_matrix mirror
   used by the kernels that consume CSC (ATA / ATDA and the matmul dispatchers);
   it's allocated on first need and refilled by the fills that read it, guarded by
   csc_seen against base.values_version. csc_iwork is the workspace for
   csr_to_csc. */
typedef struct sparse_matrix
{
    matrix base;
    CSR_matrix *csr;
    CSC_matrix *csc_cache;
    uint64_t csc_seen; /* base.values_version the csc_cache values reflect */
    int *csc_iwork;
    /* Int state bound by the alloc that produced this matrix and read by the
       matching fill: transpose_matrices_alloc stores csr->n scratch for
       transpose_matrices_fill_values; row_gather_alloc stores the base.m-long row
       map for row_gather_fill_values; row_reduce_alloc stores the source-nnz-long
       position map for row_reduce_fill_values; BA_sparse_kron_matrices_alloc
       stores the base.m-long csc_to_csr workspace. NULL otherwise; never
       touched by any other kernel. */
    int *bound_iwork;
    /* Version state bound by vstack_matrices_alloc and read and updated by
       vstack_matrices_fill_values; NULL otherwise. Layout, see
       matrix_vstack.c: [n | C_seen | seen_0 ... seen_{n-1}]. */
    uint64_t *bound_vwork;
    /* CSC copy of J bound by BA_sparse_kron_matrices_alloc and refilled by
       its fill; NULL otherwise. */
    CSC_matrix *bound_csc;
    /* Double workspace of length base.n bound by the sparse x sparse BTA / ATA
       allocs (matmul_dispatchers.c) and used as the accumulator of
       BTDA_fill_values / ATDA_fill_values; NULL otherwise. */
    double *bound_dwork;
} sparse_matrix;

/* Trim a wrapped capacity-built CSR to its true size (CSR_trim) and re-sync
   the base's cached x pointer and nnz, since realloc may move the buffer. */
void sparse_matrix_trim(matrix *M);

/* Destructor of the sparse kind; free_matrix dispatches to it. */
void sparse_matrix_free(matrix *self);

#endif /* INTERNAL_SPARSE_MATRIX_H */
