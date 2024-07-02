#include <stdio.h>
#include <stdlib.h>
#include "driver.h"
#include "matrix.h"

#ifdef DOUBLE
#define PRECISION_T double
#else
#define PRECISION_T float
#endif

int main(int argc, char* argv[]) {
    if(argc != 4) {
        printf("Usage: %s <M> <N> <K>\n", argv[0]);
        return -1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    PRECISION_T *A = (PRECISION_T*)malloc(M * K * sizeof(PRECISION_T));
    PRECISION_T *B = (PRECISION_T*)malloc(K * N * sizeof(PRECISION_T));
    PRECISION_T *C = (PRECISION_T*)malloc(M * N * sizeof(PRECISION_T));

    initialize_matrix(A, M, K);
    initialize_matrix(B, K, N);

    gemm(M, N, K, A, B, C);

    printf("Result matrix C (partial):\n");
    for(int i = 0; i < M && i < 10; ++i) {
        for(int j = 0; j < N && j < 10; ++j) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
