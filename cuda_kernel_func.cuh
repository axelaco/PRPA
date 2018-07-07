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

__global__ void triu_kernel(float *d_src,
                                  float *d_dest,
                            const int M, const int N) {
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    const int j = threadIdx.y + blockIdx.y*blockDim.y;
    if ((i * j) < M * N) {
      if (i > j)
        d_dest[j * N + i] = 0;
      else
        d_dest[j * N + i] = d_src[j * N + i];
    }
}

__global__ void update_value_kernel(float *d_A, int idx, float val) {
    d_A[idx] = val;
}
__global__ void get_value_kernel(float *d_A, int idx, float *res) {
  printf("%8.5f\n", d_A[idx]);
  res[0] = d_A[idx];
}
