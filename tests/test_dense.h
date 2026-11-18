#ifndef TEST_DENSE_H
#define TEST_DENSE_H

#include "minunit.h"
#include "sparsediff_linalg/dense.h"

/* 70 x 45 is not a multiple of the tile size in either dimension */
static char *test_A_transpose(void)
{
    enum
    {
        M = 70,
        N = 45
    };
    double A[M * N];
    double AT[N * M];
    for (int i = 0; i < M * N; i++)
    {
        A[i] = (double) i;
    }

    A_transpose(AT, A, M, N);

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            mu_assert("test_A_transpose: wrong entry",
                      AT[j * M + i] == A[i * N + j]);
        }
    }
    return 0;
}

#endif /* TEST_DENSE_H */
