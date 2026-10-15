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
#include "internal/csr_kernels.h"
#include "sparsediff_linalg/CSR_matrix.h"

/* Compute block-wise matrix-vector products.
 * y = [A @ x1; A @ x2; ...; A @ xp] where A is m x n and x is (n*p)-length vector.
 * x is split into p blocks of n elements each.
 */
void block_left_multiply_vec(const struct CSR_matrix *A, const double *x, double *y,
                             int p)
{
    /* For each block */
    for (int block = 0; block < p; block++)
    {
        int block_start = block * A->n;
        int y_offset = block * A->m;

        /* For each row of A */
        for (int i = 0; i < A->m; i++)
        {
            double row_sum = 0.0;
            int row_nnz = A->p[i + 1] - A->p[i];

            /* Compute sparse dot product of A[i,:] with x_block */
            for (int idx = 0; idx < row_nnz; idx++)
            {
                int col = A->i[A->p[i] + idx];
                row_sum += A->x[A->p[i] + idx] * x[block_start + col];
            }

            y[y_offset + i] = row_sum;
        }
    }
}
