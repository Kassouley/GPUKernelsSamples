#ifndef DRIVER_H
#define DRIVER_H

template<typename T>
void gemm(int M, int N, int K, T* A, T* B, T* C, int gridDimX, int gridDimY, int gridDimZ, int blockDimX, int blockDimY, int blockDimZ);

#endif // DRIVER_H
