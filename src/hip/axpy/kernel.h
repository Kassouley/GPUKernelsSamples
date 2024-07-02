#ifndef KERNEL_H
#define KERNEL_H

#include <hip/hip_runtime.h>

template<typename T>
__global__ void axpy_kernel(T alpha, T* X, T* Y, int N);

#endif // KERNEL_H
