#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

typedef struct mat_ {
  int m;
  int n;
  float *data;
} Mat;
float *create_gpu_matrix(float *cpu_data, int m, int n);
void vecAdd(float *a, float *b, float *c, int block_size, int n);
void vecSub(float *a, float *b, float *c, int block_size, int n);
void vecScalar(float *a, float scalar, int block_size, int n);
void vecCopy(float *src, float *dest, int block_size, int n);


Mat *matrix_new(int m, int n);
Mat *matrix_mul(cublasHandle_t handle, Mat *A, Mat *B);
void matrix_sub(Mat *A, Mat *B, Mat *res);
void matrix_print(Mat *A);
void matrix_add(Mat *A, Mat *B, Mat *res);
void matrix_scalar(Mat *A, float scalar);
