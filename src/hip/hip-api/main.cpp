#include <stdio.h>
#include <stdlib.h>
#include "driver.h"
#include "matrix.h"
#include "hip_utils.h"

#include <hip/hip_runtime.h>
#include <iostream>

int get_device_info() {
    int deviceCount;
    SAFECALL(hipGetDeviceCount(&deviceCount));

    for (int device = 0; device < deviceCount; ++device) {
        hipDeviceProp_t deviceProp;
        SAFECALL(hipGetDeviceProperties(&deviceProp, device));

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x " << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "  Max grid dimensions: " << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Concurrent kernels: " << (deviceProp.concurrentKernels ? "Yes" : "No") << std::endl;
    }

    return 0;
}


int main(int argc, char* argv[]) {
    if(argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return -1;
    }

    get_device_info();

    int N = atoi(argv[1]);

    float *h_a = (float*)malloc(N * N * sizeof(float));
    float *h_b = (float*)malloc(N * N * sizeof(float));
    float *h_c = (float*)malloc(N * N * sizeof(float));
    float *h_d = (float*)malloc(N * sizeof(float));
    float *h_e = (float*)malloc(N * sizeof(float));
    float *h_f = (float*)malloc(N * sizeof(float));

    initialize_matrix(h_a, N, N);
    initialize_matrix(h_b, N, N);
    initialize_matrix(h_c, N, N);
    initialize_matrix(h_d, N, 1);
    initialize_matrix(h_e, N, 1);
    initialize_matrix(h_f, N, 1);
    
    driver(h_a, h_b, h_c, h_d, h_e, h_f, N);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(h_e);
    free(h_f);

    return 0;
}
