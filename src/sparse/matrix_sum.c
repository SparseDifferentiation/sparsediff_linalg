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
#include "sparsediff_linalg/matrix_sum.h"
#include "internal/csr_kernels.h"
#include "internal/sparse_matrix.h"
#include "sparsediff_linalg/csr_kernels.h"
#include "sparsediff_linalg/tracked_alloc.h"
#include <assert.h>
#include <string.h>

void sum_matrices_alloc(matrix *A, matrix *B, matrix *C)
{
    CSR_matrix *cc = C->to_csr(C);
    sum_csr_alloc(A->to_csr(A), B->to_csr(B), cc);
    sparse_matrix_trim(C);
}

void sum_matrices_fill_values(matrix *A, matrix *B, matrix *C)
{
    sum_csr_fill_values(A->to_csr(A), B->to_csr(B), C->to_csr(C));
    matrix_values_changed(C);
}

void sum_scaled_matrices_fill_values(matrix *A, matrix *B, matrix *C,
                                     const double *d1, const double *d2)
{
    sum_scaled_csr_matrices_fill_values(A->to_csr(A), B->to_csr(B), C->to_csr(C), d1,
                                        d2);
    matrix_values_changed(C);
}

matrix *sum_4_matrices_alloc(matrix *A, matrix *B, matrix *D, matrix *E)
{
    matrix *ops[4] = {A, B, D, E};
    int *maps[4];
    CSR_matrix *csr = sum_4_csr_alloc(A->to_csr(A), B->to_csr(B), D->to_csr(D),
                                      E->to_csr(E), maps);

    /* The maps index the operands in CSR order. A stacked_pd's own value buffer
       is block-major, so re-index its map to read op->x directly at fill time. */
    size_t total = 0;
    for (int k = 0; k < 4; k++)
    {
        if (ops[k]->kind == MATRIX_STACKED_PD)
        {
            compose_csr_idx_map_for_spd(ops[k], ops[k]->to_csr(ops[k]), maps[k]);
        }
        total += (size_t) ops[k]->nnz;
    }

    /* bind the four maps back to back inside the output; the fill splits them
       by operand nnz */
    int *all = (int *) sp_malloc((total > 0 ? total : 1) * sizeof(int));
    size_t off = 0;
    for (int k = 0; k < 4; k++)
    {
        memcpy(all + off, maps[k], (size_t) ops[k]->nnz * sizeof(int));
        off += (size_t) ops[k]->nnz;
        sp_free(maps[k]);
    }

    sparse_matrix *out = (sparse_matrix *) new_sparse_matrix(csr);
    out->bound_iwork = all;
    return &out->base;
}

void sum_4_matrices_fill_values(const matrix *A, const matrix *B, const matrix *D,
                                const matrix *E, matrix *C)
{
    assert(C->kind == MATRIX_SPARSE);
    const int *map = ((sparse_matrix *) C)->bound_iwork;
    assert(map != NULL);

    memset(C->x, 0, C->nnz * sizeof(double));
    accumulator(A->x, A->nnz, map, C->x);
    map += A->nnz;
    accumulator(B->x, B->nnz, map, C->x);
    map += B->nnz;
    accumulator(D->x, D->nnz, map, C->x);
    map += D->nnz;
    accumulator(E->x, E->nnz, map, C->x);
    matrix_values_changed(C);
}
