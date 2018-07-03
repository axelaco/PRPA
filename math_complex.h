#include <stdlib.h>
#ifdef INTEL_MKL
#include "mkl.h" // Intel MKL
#endif
#include <math.h>
#include <stdio.h>
#ifndef INTEL_MKL
typedef struct complex_ {
  float real;
  float imag;
} complex;
#else
  typedef MKL_Complex8 complex;
#endif
typedef struct mat_ {
  int m;
  int n;
  complex *data;
} Mat;

Mat *matrix_new(int m, int n);
void matrix_copy(Mat *src, Mat *dest);
Mat *matrix_eye(int m, int n);
Mat *matrix_zeros(int m, int n);
Mat *matrix_mul(Mat *A, Mat *B);
complex *matrix_diag(Mat *A);
void matrix_sub(Mat *A, Mat *B, Mat *res);
void matrix_add(Mat *A, Mat *B, Mat *res);
void matrix_scalar(Mat *A, float scalar);
complex *matrix_eye_bis(int m, int n);
void matrix_delete(Mat *m);
void matrix_print(Mat *m);
Mat *matrix_transpose(Mat *m);
Mat *matrix_reduce(Mat *m, int maxCol);
Mat *matrix_reduce_cond(Mat *m, int col);
void matrix_copy_cond(Mat *src, Mat *dest, int col);
float vect_norm(complex *u, int n);
complex vect_dot(complex *u, complex *v, int n);
void vect_divide(complex *u, float scalar, int n);
void vect_mat_copy(Mat *mat, complex *x, int col);
void vect_mat_copy_cond(Mat *mat, complex *u, int col, int line);
void vect_prod_mat(Mat *A, complex *u, complex *res);
complex *vect_prod_mat_trans(Mat *mat, complex *u);
complex *vect_divide_by_scalar(complex *u, float scalar, int n);
void vect_substract(complex *res, complex *u , complex *v, int m);
void vect_scalar(complex *u, float scalar, int n);
void vect_add(complex *res, complex *a, complex *b, int m);
void compute_fm(complex *fm, complex *u, complex *w, int n, int m);
void vect_print(complex *u, int n);
complex *get_column(Mat *mat, int col);
complex *get_column_start(Mat *mat, int col);
complex absolute(complex nb);