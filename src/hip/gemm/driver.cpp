#include <hip/hip_runtime.h>
#include "kernel.h"
#include "driver.h"
#include "hip_utils.h"

template<typename T>
void gemm(int M, int N, int K, T* A, T* B, T* C, 
            int gridDimX, int gridDimY, int gridDimZ, 
            int blockDimX, int blockDimY, int blockDimZ) {
    T *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_C = M * N * sizeof(T);

    SAFECALL(hipMalloc(&d_A, size_A));
    SAFECALL(hipMalloc(&d_B, size_B));
    SAFECALL(hipMalloc(&d_C, size_C));

    SAFECALL(hipMemcpy(d_A, A, size_A, hipMemcpyHostToDevice));
    SAFECALL(hipMemcpy(d_B, B, size_B, hipMemcpyHostToDevice));

    dim3 gridDim(gridDimX, gridDimY, gridDimZ);
    dim3 blockDim(blockDimX, blockDimY, blockDimZ);

    hipLaunchKernelGGL(gemm_kernel<T>, gridDim, blockDim, 0, 0, d_A, d_B, d_C, M, N, K);

    SAFECALL(hipMemcpy(C, d_C, size_C, hipMemcpyDeviceToHost));

    SAFECALL(hipFree(d_A));
    SAFECALL(hipFree(d_B));
    SAFECALL(hipFree(d_C));
}

template void gemm<float>(int, int, int, float*, float*, float*, int, int, int, int, int, int);
template void gemm<double>(int, int, int, double*, double*, double*, int, int, int, int, int, int);
