#include <hip/hip_runtime.h>
#include "kernel.h"
#include "driver.h"
#include "hip_utils.h"

template<typename T>
void axpy(T alpha, T* X, T* Y, int N, int gridDimX, int blockDimX) {
    T *d_X, *d_Y;
    size_t size = N * sizeof(T);

    SAFECALL(hipMalloc(&d_X, size));
    SAFECALL(hipMalloc(&d_Y, size));

    SAFECALL(hipMemcpy(d_X, X, size, hipMemcpyHostToDevice));
    SAFECALL(hipMemcpy(d_Y, Y, size, hipMemcpyHostToDevice));

    axpy_kernel<T><<<gridDimX, blockDimX>>>(alpha, d_X, d_Y, N);

    SAFECALL(hipMemcpy(Y, d_Y, size, hipMemcpyDeviceToHost));

    SAFECALL(hipFree(d_X));
    SAFECALL(hipFree(d_Y));
}

// Explicit template instantiations
template void axpy<float>(float alpha, float* X, float* Y, int N, int gridDimX, int blockDimX);
template void axpy<double>(double alpha, double* X, double* Y, int N, int gridDimX, int blockDimX);

