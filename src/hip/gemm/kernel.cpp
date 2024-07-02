#include <hip/hip_runtime.h>
#include "kernel.h"

template<typename T>
__global__ void gemm_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N) {
        T value = 0;
        for (int e = 0; e < K; ++e) {
            value += A[row * K + e] * B[e * N + col];
        }
        C[row * N + col] = value;
    }
}

template __global__ void gemm_kernel<float>(const float* A, const float* B, float* C, int M, int N, int K);
template __global__ void gemm_kernel<double>(const double* A, const double* B, double* C, int M, int N, int K);
