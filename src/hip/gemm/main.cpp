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
    if(argc != 10) {
        printf("Usage: %s <M> <N> <K> <gridDimX> <gridDimY> <gridDimZ> <blockDimX> <blockDimY> <blockDimZ>\n", argv[0]);
        return -1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int gridDimX = atoi(argv[4]);
    int gridDimY = atoi(argv[5]);
    int gridDimZ = atoi(argv[6]);
    int blockDimX = atoi(argv[7]);
    int blockDimY = atoi(argv[8]);
    int blockDimZ = atoi(argv[9]);

    PRECISION_T *A = (PRECISION_T*)malloc(M * K * sizeof(PRECISION_T));
    PRECISION_T *B = (PRECISION_T*)malloc(K * N * sizeof(PRECISION_T));
    PRECISION_T *C = (PRECISION_T*)malloc(M * N * sizeof(PRECISION_T));

    initialize_matrix(A, M, K);
    initialize_matrix(B, K, N);

    gemm(M, N, K, A, B, C, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);

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
