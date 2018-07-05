#include <stdio.h>

__global__ void vecAdd_kernel(float *a, float *b, float *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}
__global__ void vecSub_kernel(float *a, float *b, float *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}

__global__ void vecScalar_kernel (float *a, float scalar, int n) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n)
    a[id] *= scalar;
}

__global__ void vecCopy_kernel(float *src, float *dest, int n) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n)
    dest[id] = src[id];
}
