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
