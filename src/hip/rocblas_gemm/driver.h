#ifndef DRIVER_H
#define DRIVER_H

template<typename T>
void gemm(int M, int N, int K, const T* A, const T* B, T* C);

#endif // DRIVER_H
