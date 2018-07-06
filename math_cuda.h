#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <assert.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

typedef struct mat_ {
  int m;
  int n;
  float *data;
} Mat;
float *create_gpu_matrix(float *cpu_data, int m, int n);

// Cuda Kernel Method
void vecAdd(float *a, float *b, float *c, int block_size, int n);
void vecSub(float *a, float *b, float *c, int block_size, int n);
void vecScalar(float *a, float scalar, int block_size, int n);
void vecCopy(float *src, float *dest, int block_size, int n);
void triu(float *d_src, float *d_dest, const int m,
   const int n, const int subM);

void qr(cublasHandle_t handle, cusolverDnHandle_t cusolverH, Mat *A, Mat *R, Mat *Q);
Mat *matrix_new(int m, int n);
Mat *matrix_mul(cublasHandle_t handle, Mat *A, Mat *B);
Mat *matrix_transpose(cublasHandle_t handle, Mat *A);
Mat *matrix_zeros(int m, int n);
Mat *matrix_eye(int m, int n);
void matrix_delete(Mat *A);
void matrix_sub(Mat *A, Mat *B, Mat *res);
void matrix_print(Mat *A);
void matrix_add(Mat *A, Mat *B, Mat *res);
void matrix_scalar(Mat *A, float scalar);
void vect_copy(float *src, float *dest, int m);
void vect_prod_mat(cublasHandle_t handle, Mat *A, float *u, float *res);
float vect_dot(cublasHandle_t handle, float *u, float *v, int n);
void vect_scalar(float *u, float scalar, int n);
void vect_add(float *res, float *u, float *v, int m);
void vect_substract(float *res, float *u, float *v, int m);
void vect_mat_copy(cublasHandle_t handle, Mat *A, float *u, int col);
float *vect_divide_by_scalar(float *u, float scalar, int n);
float vect_norm(cublasHandle_t handle, float *u, int n);
float *get_column(cublasHandle_t handle, Mat *A, int col);
void vect_print(float *u, int n);
