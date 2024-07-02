#include <hip/hip_runtime.h>
#include "driver.h"
#include "hip_utils.h"
#include <rocblas/rocblas.h>

template<typename T>
void gemm(int M, int N, int K, const T* A, const T* B, T* C) {
    T *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_C = M * N * sizeof(T);

    SAFECALL(hipMalloc(&d_A, size_A));
    SAFECALL(hipMalloc(&d_B, size_B));
    SAFECALL(hipMalloc(&d_C, size_C));

    SAFECALL(hipMemcpy(d_A, A, size_A, hipMemcpyHostToDevice));
    SAFECALL(hipMemcpy(d_B, B, size_B, hipMemcpyHostToDevice));
    
    const T alpha = 1.0f; 
    const T beta = 0.0f; 
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    if constexpr (std::is_same_v<T, double>) {
        rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                    M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, N);
    } else if constexpr (std::is_same_v<T, float>) {
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                    M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, N);
    } else {
        static_assert(std::is_same_v<T, T>, "Unsupported type");
    }

    rocblas_destroy_handle(handle);

    SAFECALL(hipMemcpy(C, d_C, size_C, hipMemcpyDeviceToHost));

    SAFECALL(hipFree(d_A));
    SAFECALL(hipFree(d_B));
    SAFECALL(hipFree(d_C));
}

template void gemm<float>(int, int, int, const float*, const float*, float*);
template void gemm<double>(int, int, int, const double*, const double*, double*);
