/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include "driver.h"
#include "matrix.h"

#ifdef DOUBLE
#define PRECISION_T double
#else
#define PRECISION_T float
#endif


int main(int argc, char** argv)
{
    if(argc != 4) {
        printf("Usage: %s <N> <gridDimX> <blockDimX>\n", argv[0]);
        return -1;
    }
    int N = atoi(argv[1]);
    int gridDimX = atoi(argv[2]);
    int blockDimX = atoi(argv[3]);

    printf("Array size: %f MB\n", (float)(N * sizeof(int))/1024.0/1024.0);
    PRECISION_T* array = (PRECISION_T*)malloc(N*sizeof(PRECISION_T));
    
    initialize_matrix(array, N, 1);


    reduction(array, N, gridDimX, blockDimX);

    free(array);
}