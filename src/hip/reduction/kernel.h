#ifndef KERNEL_H
#define KERNEL_H

#include <hip/hip_runtime.h>

template<typename T>
__global__ void atomic_reduction_kernel(T* in, T* out, int N);

template<typename T>
__global__ void atomic_reduction_kernel2(T* in, T* out, int N);

template<typename T>
__global__ void atomic_reduction_kernel3(T* in, T* out, int N);

#endif // KERNEL_H
