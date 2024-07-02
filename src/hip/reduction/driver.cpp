#include <hip/hip_runtime.h>
#include "kernel.h"
#include "driver.h"
#include "hip_utils.h"

template<typename T>
void reduction(T* array, int N, int gridDimX, int blockDimX) {
    T *in, *out;
    T sum = 0;
    T checksum = 0;

    size_t size = sizeof(T) * N;

    SAFECALL(hipMalloc(&in, size));
    SAFECALL(hipMalloc(&out, sizeof(T)));


    SAFECALL(hipMemcpy(in, array, size, hipMemcpyHostToDevice));
    SAFECALL(hipDeviceSynchronize());

    // Get device properties
    hipDeviceProp_t props;
    SAFECALL(hipGetDeviceProperties(&props, 0));


    SAFECALL(hipMemsetAsync(out,0,sizeof(int)));
    hipLaunchKernelGGL(atomic_reduction_kernel, dim3(gridDimX), dim3(blockDimX), 0, 0, in, out, N);
    //hipLaunchKernelGGL(atomic_reduction_kernel2, dim3(gridDimX), dim3(blockDimX), 0, 0, in,out,N);
    //hipLaunchKernelGGL(atomic_reduction_kernel3, dim3(gridDimX), dim3(blockDimX), 0, 0, in,out,N);

    for(int i = 0; i < N; i++) {
        checksum+=array[i];
    }
    SAFECALL(hipDeviceSynchronize());
    
    SAFECALL(hipMemcpy(&sum, out, sizeof(T), hipMemcpyDeviceToHost));
    
    if(sum == checksum)
        printf("VERIFICATION: result is CORRECT\n");
    else
        printf("VERIFICATION: result is INCORRECT!! CPU : %f, GPU : %f\n", checksum, sum);

    SAFECALL(hipFree(in));
    SAFECALL(hipFree(out));
}

// Explicit template instantiations
template void reduction<float>(float* array, int N, int gridDimX, int blockDimX);
template void reduction<double>(double* array, int N, int gridDimX, int blockDimX);

