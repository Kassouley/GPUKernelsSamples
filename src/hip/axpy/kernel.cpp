#include <hip/hip_runtime.h>
#include "kernel.h"

template<typename T>
__global__ void axpy_kernel(T alpha, T* X, T* Y, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        Y[tid] = alpha * X[tid] + Y[tid];
    }
}

template __global__ void axpy_kernel<float>(float alpha, float* X, float* Y, int N);
template __global__ void axpy_kernel<double>(double alpha, double* X, double* Y, int N);
