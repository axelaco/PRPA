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
const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__global__ void transposeNoBankConflicts(float *d_dest,float *d_src)
{
  __shared__ float tile[TILE_DIM * TILE_DIM];

   int x = blockIdx.x * TILE_DIM + threadIdx.x;
   int y = blockIdx.y * TILE_DIM + threadIdx.y;
   int width = gridDim.x * TILE_DIM;

   for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = d_src[(y+j)*width + x];

   __syncthreads();

   for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      d_dest[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];

}
