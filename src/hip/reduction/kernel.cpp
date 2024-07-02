#include <hip/hip_runtime.h>
#include "kernel.h"

template<typename T>
__global__ void atomic_reduction_kernel(T* in, T* out, int N) {
    T sum = 0;
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    for(int i = idx; i < N; i += hipBlockDim_x*hipGridDim_x) {
        sum += in[i];
    }
    atomicAdd(out, sum);
}

template __global__ void atomic_reduction_kernel<float>(float* in, float* out, int N);
template __global__ void atomic_reduction_kernel<double>(double* in, double* out, int N);

template<typename T>
__global__ void atomic_reduction_kernel2(T* in, T* out, int N) {
    T sum = 0;
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    for(int i = idx*16; i < N; i += hipBlockDim_x*hipGridDim_x*16) {
        sum += in[i] + in[i+1] + in[i+2] + in[i+3] 
            + in[i+4] + in[i+5] + in[i+6] + in[i+7] 
            + in[i+8] + in[i+9] + in[i+10] + in[i+11] 
            + in[i+12] + in[i+13] + in[i+14] + in[i+15] ;
    }
    atomicAdd(out, sum);
}

template __global__ void atomic_reduction_kernel2<float>(float* in, float* out, int N);
template __global__ void atomic_reduction_kernel2<double>(double* in, double* out, int N);

template<typename T>
__global__ void atomic_reduction_kernel3(T* in, T* out, int N) {
    T sum = 0;
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    for(int i = idx*4; i < N; i += hipBlockDim_x*hipGridDim_x*4) {
        sum += in[i] + in[i+1] + in[i+2] + in[i+3];
    }
    atomicAdd(out, sum);
}

template __global__ void atomic_reduction_kernel3<float>(float* in, float* out, int N);
template __global__ void atomic_reduction_kernel3<double>(double* in, double* out, int N);
