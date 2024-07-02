#ifndef KERNEL_H
#define KERNEL_H

#include <hip/hip_runtime.h>

__global__ void saxpy(int n, float a, float *x, float *y);
__global__ void sgemm(int m, int n, int k, float *A, float *B, float *C);
__global__ void vectorAdd(int n, float *x, float *y, float *z);
__global__ void vectorMul(int n, float *x, float *y, float *z);

#endif
