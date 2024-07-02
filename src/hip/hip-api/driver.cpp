#include <hip/hip_runtime.h>
#include "hip_utils.h"
#include "driver.h"
#include "kernel.h"

void driver(float *h_a, float *h_b, float *h_c, float *h_d, float *h_e, float *h_f, int n) {
    float *d_a, *d_b, *d_c, *d_d, *d_e, *d_f;
    hipStream_t streams[4];
    hipEvent_t start[4], stop[4];

    SAFECALL(hipMalloc(&d_a, n * n * sizeof(float)));
    SAFECALL(hipMalloc(&d_b, n * n * sizeof(float)));
    SAFECALL(hipMalloc(&d_c, n * n * sizeof(float)));
    SAFECALL(hipMalloc(&d_d, n * sizeof(float)));
    SAFECALL(hipMalloc(&d_e, n * sizeof(float)));
    SAFECALL(hipMalloc(&d_f, n * sizeof(float)));

    for (int i = 0; i < 4; ++i) {
        SAFECALL(hipStreamCreate(&streams[i]));
        SAFECALL(hipEventCreate(&start[i]));
        SAFECALL(hipEventCreate(&stop[i]));
    }

    SAFECALL(hipMemcpyAsync(d_a, h_a, n * n * sizeof(float), hipMemcpyHostToDevice, streams[0]));
    SAFECALL(hipMemcpyAsync(d_b, h_b, n * n * sizeof(float), hipMemcpyHostToDevice, streams[1]));
    SAFECALL(hipMemcpyAsync(d_d, h_d, n * sizeof(float), hipMemcpyHostToDevice, streams[2]));
    SAFECALL(hipMemcpyAsync(d_e, h_e, n * sizeof(float), hipMemcpyHostToDevice, streams[3]));

    SAFECALL(hipEventRecord(start[0], streams[0]));
    saxpy<<<(n + 255) / 256, 256, 0, streams[0]>>>(n, 2.0f, d_a, d_b);
    SAFECALL(hipEventRecord(stop[0], streams[0]));

    SAFECALL(hipEventRecord(start[1], streams[1]));
    sgemm<<<dim3((n + 15) / 16, (n + 15) / 16), dim3(16, 16), 0, streams[1]>>>(n, n, n, d_a, d_b, d_c);
    SAFECALL(hipEventRecord(stop[1], streams[1]));

    SAFECALL(hipEventRecord(start[2], streams[2]));
    vectorAdd<<<(n + 255) / 256, 256, 0, streams[2]>>>(n, d_d, d_e, d_f);
    SAFECALL(hipEventRecord(stop[2], streams[2]));

    SAFECALL(hipEventRecord(start[3], streams[3]));
    vectorMul<<<(n + 255) / 256, 256, 0, streams[3]>>>(n, d_d, d_e, d_f);
    SAFECALL(hipEventRecord(stop[3], streams[3]));

    SAFECALL(hipMemcpyAsync(h_c, d_c, n * n * sizeof(float), hipMemcpyDeviceToHost, streams[1]));
    SAFECALL(hipMemcpyAsync(h_f, d_f, n * sizeof(float), hipMemcpyDeviceToHost, streams[3]));

    for (int i = 0; i < 4; ++i) {
        SAFECALL(hipStreamSynchronize(streams[i]));
        float milliseconds = 0;
        SAFECALL(hipEventElapsedTime(&milliseconds, start[i], stop[i]));
        printf("Kernel %d execution time: %f ms\n", i, milliseconds);
        SAFECALL(hipEventDestroy(start[i]));
        SAFECALL(hipEventDestroy(stop[i]));
        SAFECALL(hipStreamDestroy(streams[i]));
    }

    SAFECALL(hipFree(d_a));
    SAFECALL(hipFree(d_b));
    SAFECALL(hipFree(d_c));
    SAFECALL(hipFree(d_d));
    SAFECALL(hipFree(d_e));
    SAFECALL(hipFree(d_f));
}
