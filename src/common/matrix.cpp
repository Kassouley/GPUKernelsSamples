#include <cstdlib>
#include "matrix.h"

template void initialize_matrix<float>(float* matrix, int rows, int cols);
template void initialize_matrix<double>(double* matrix, int rows, int cols);

template<typename T>
void initialize_matrix(T* matrix, int rows, int cols) {
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }
    }
}

