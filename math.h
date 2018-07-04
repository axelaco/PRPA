#include <stdlib.h>
#ifdef INTEL_MKL
#include "mkl.h" // Intel MKL
#endif
#include <math.h>
#include <stdio.h>

typedef struct mat_ {
  int m;
  int n;
  float *data;
} Mat;
float *qr_alg_eigen(Mat *A, Mat *eigVector);
float *rritz(Mat *Tm, float *mx, float *fm, int k, float nrmfr);
Mat *matrix_new(int m, int n);
void matrix_copy(Mat *src, Mat *dest);
Mat *matrix_eye(int m, int n);
Mat *matrix_zeros(int m, int n);
Mat *matrix_mul(Mat *A, Mat *B);
float *matrix_get_row(Mat *A, int m);
float *matrix_off_diag(Mat *A);
void matrix_mul_bis(Mat *res, Mat *A, Mat *B);
float *matrix_diag(Mat *A);
void matrix_sub(Mat *A, Mat *B, Mat *res);
void matrix_add(Mat *A, Mat *B, Mat *res);
void matrix_scalar(Mat *A, float scalar);
float *matrix_eye_bis(int m, int n);
void matrix_delete(Mat *m);
void matrix_print(Mat *m);
Mat *matrix_transpose(Mat *m);
Mat *matrix_reduce(Mat *m, int maxCol);
Mat *matrix_reduce_cond(Mat *m, int col);
void matrix_copy_sub(Mat *src, Mat *dest, int col);
void matrix_copy_cond(Mat *A, Mat *B, int col);
float vect_norm(float *u, int n);
float vect_dot(float *u, float *v, int n);
void vect_divide(float *u, float scalar, int n);
void vect_mat_copy(Mat *mat, float *x, int col);
void vect_mat_copy_cond(Mat *mat, float *u, int col, int line);
void vect_prod_mat(Mat *A, float *u, float *res);
float *vect_prod_mat_trans(Mat *mat, float *u);
float *vect_divide_by_scalar(float *u, float scalar, int n);
void vect_substract(float *res, float *u , float *v, int m);
void vect_copy(float *src, float *dest, int m);
void vect_scalar(float *u, float scalar, int n);
void vect_add(float *res, float *a, float *b, int m);
void compute_fm(float *fm, float *u, float *w, int n, int m);
void vect_print(float *u, int n);
float *get_column(Mat *mat, int col);
float *get_column_start(Mat *mat, int col);
float absolute(float nb);