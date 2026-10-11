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
#ifndef SPARSE_MATRIX_CACHE_H
#define SPARSE_MATRIX_CACHE_H

#include "internal/sparse_matrix.h"
#include "sparsediff_linalg/CSC_matrix.h"
#include "sparsediff_linalg/matrix.h"

/* The sparse kind keeps a lazily built CSC mirror of its CSR values for the
   kernels that consume CSC. Two accessors, one per phase:

   - ensure: build the mirror's structure if absent. Values are NOT filled, so
     it is safe during the symbolic phase while values are still garbage.
   - fresh: ensure, then refill the values iff they changed since the last
     refill (guarded by base.values_version), and return the mirror. Every
     fill-phase reader goes through this, so a caller never has to refresh
     anything by hand. A is logically const: only the private cache moves. */
void sparse_matrix_ensure_csc_cache(sparse_matrix *sm);
const CSC_matrix *sparse_csc_fresh(const matrix *A);

#endif /* SPARSE_MATRIX_CACHE_H */
