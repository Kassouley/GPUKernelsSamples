#ifndef DRIVER_H
#define DRIVER_H

template<typename T>
void axpy(T alpha, T* X, T* Y, int N, int gridDimX, int blockDimX);

#endif // DRIVER_H
