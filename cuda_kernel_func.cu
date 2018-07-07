#include "cuda_kernel_func.cuh"

extern "C" void vecSub(float *a, float *b, float *res, int blockSize, int n) {
  int gridSize = (n + blockSize - 1)/blockSize;
  vecSub_kernel<<<gridSize, blockSize>>>(a, b, res, n);
}
extern "C" void vecAdd(float *a, float *b, float *res, int blockSize, int n) {
  int gridSize = (n + blockSize - 1)/blockSize;
  vecAdd_kernel<<<gridSize, blockSize>>>(a, b, res, n);
}

extern "C" void vecScalar(float *a, float scalar, int blockSize, int n) {
  int gridSize = (n + blockSize - 1)/blockSize;
  vecScalar_kernel<<<gridSize, blockSize>>>(a, scalar, n);
}

extern "C" void vecCopy(float *src, float* dest, int blockSize, int n) {
  int gridSize = (n + blockSize - 1)/blockSize;
  vecCopy_kernel<<<gridSize, blockSize>>>(src, dest, n);
}

extern "C" void triu(float *d_src,
                                  float *d_dest,
                            const int m, const int n)
{
  dim3 Blocks(1,1);
  dim3 Threads(m, n);
  triu_kernel<<<Blocks, Threads>>>(d_src, d_dest, m, n);
}
extern "C" void update_value(float *d_A, int idx, float val) {
  update_value_kernel<<<1,1>>>(d_A, idx, val);
}

extern "C" float get_value(float *d_A, int idx) {
  float res = 0;
  cudaMemcpy(&res, d_A + idx, sizeof(float), cudaMemcpyDeviceToHost);
  return res;
}

extern "C" void diag(float *d_A, float *d_res, int w, int h) {
  diag_kernel<<<1, w>>>(d_A, d_res, w, h);
}

extern "C" void off_diag(float *d_A, float *d_res, int w, int h) {
  off_diag_kernel<<<1, h>>>(d_A, d_res, w, h);
}
