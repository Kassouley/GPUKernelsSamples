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
        printf("Usage: %s <N> <gridDimX> <blockDimX>\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]);
    int gridDimX = atoi(argv[2]);
    int blockDimX = atoi(argv[3]);

    PRECISION_T alpha = 2.0f;
    PRECISION_T *X = (PRECISION_T*)malloc(N * sizeof(PRECISION_T));
    PRECISION_T *Y = (PRECISION_T*)malloc(N * sizeof(PRECISION_T));

    initialize_matrix(X, N, 1); // Initialize X with random values
    initialize_matrix(Y, N, 1); // Initialize Y with random values

    axpy(alpha, X, Y, N, gridDimX, blockDimX);

    // Print a part of the result vector Y for verification
    printf("Result vector Y (partial):\n");
    for(int i = 0; i < N && i < 10; ++i) {
        printf("%f ", Y[i]);
    }
    printf("\n");

    free(X);
    free(Y);

    return 0;
}
