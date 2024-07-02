#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include <hip/hip_runtime.h>
#include <stdio.h>

#define SAFECALL(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

#endif // HIP_UTILS_H
