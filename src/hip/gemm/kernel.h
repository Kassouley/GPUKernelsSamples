#ifndef KERNEL_H
#define KERNEL_H

#include <hip/hip_runtime.h>

template<typename T>
__global__ void gemm_kernel(const T* A, const T* B, T* C, int M, int N, int K);

#endif // KERNEL_H
