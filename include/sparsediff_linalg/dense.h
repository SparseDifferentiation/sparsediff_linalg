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
#ifndef DENSE_H
#define DENSE_H

/* Small dense helpers on plain double arrays (row-major unless stated). */

/* Example: a = [1, 2], len = 2, repeats = 3, result = [1, 1, 1, 2, 2, 2] */
void repeat(double *result, const double *a, int len, int repeats);

/* Example: a = [1, 2], len = 2, tiles = 3, result = [1, 2, 1, 2, 1, 2] */
void tile_double(double *result, const double *a, int len, int tiles);

/* Example: size = 5, value = 3.0, result = [3.0, 3.0, 3.0, 3.0, 3.0] */
void scaled_ones(double *result, int size, double value);

/* Row-major dense transpose: AT[j*m + i] = A[i*n + j] for an m x n A. */
void A_transpose(double *AT, const double *A, int m, int n);

/* Compute v = (Y kron I_m) @ w where Y is k x n (col-major), len(w) = m * n, and
   len(v) = m * k.  Equivalently, reshape w as the m x n matrix W (col-major) and
   compute v = vec(W @ Y^T). */
void Y_kron_I_vec(int m, int k, int n, const double *Y, const double *w, double *v);

/* Compute v = (I_n kron X^T) @ w where X is m x k (col-major), len(w) = m * n, and
   len(v) = k * n.  Equivalently, reshape w as the m x n matrix W (col-major) and
   compute v = vec(X^T @ W). */
void I_kron_XT_vec(int m, int k, int n, const double *X, const double *w, double *v);

#endif /* DENSE_H */
