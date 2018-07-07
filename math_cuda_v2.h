#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <assert.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

typedef struct mat_ {
  int m;
  int n;
  float *d_data;
} Mat;
float *create_gpu_matrix(float *cpu_data, int m, int n);

// Cuda Kernel Method
void vecAdd(float *a, float *b, float *c, int block_size, int n);
void vecSub(float *a, float *b, float *c, int block_size, int n);
void vecScalar(float *a, float scalar, int block_size, int n);
void vecCopy(float *src, float *dest, int block_size, int n);
void triu(float *d_src, float *d_dest, const int m,
   const int n, const int subM);
void update_value(float *d_A, int idx, float val);
float get_value(float *d_A, int idx);

float *qr_alg_eigen(Mat *A, Mat *eigVector);
float *matrix_diag(Mat *A);
float *matrix_off_diag(Mat *A);
void qr(cublasHandle_t handle, cusolverDnHandle_t cusolverH, Mat *A, Mat *R, Mat *Q);
Mat *matrix_new(int m, int n);
Mat *matrix_mul(cublasHandle_t handle, Mat *A, Mat *B);
void matrix_mul_bis(cublasHandle_t handle, Mat *res, Mat *A, Mat *B);
Mat *matrix_transpose(cublasHandle_t handle, Mat *A);
Mat *matrix_zeros(int m, int n);
Mat *matrix_eye(int m, int n);
void matrix_delete(Mat *A);
void matrix_sub(Mat *A, Mat *B, Mat *res);
void matrix_print(Mat *A);
void matrix_add(Mat *A, Mat *B, Mat *res);
void matrix_scalar(Mat *A, float scalar);
void vect_copy(float *d_src, float *d_dest, int m);
void vect_prod_mat(cublasHandle_t handle, Mat *A, float *d_u, float *d_res);
float vect_dot(cublasHandle_t handle, float *d_u, float *d_v, int n);
void vect_scalar(float *d_u, float scalar, int n);
void vect_add(float *d_res, float *d_u, float *d_v, int m);
void vect_substract(float *d_res, float *d_u, float *d_v, int m);
void vect_mat_copy(cublasHandle_t handle, Mat *A, float *d_u, int col);
float *vect_divide_by_scalar(float *d_u, float scalar, int n);
float vect_norm(cublasHandle_t handle, float *d_u, int n);
float *get_column(cublasHandle_t handle, Mat *A, int col);
void vect_print(float *d_u, int n);
