#include <stdlib.h>
#include <math.h>
#include <stdio.h>

typedef struct mat_ {
  int m;
  int n;
  float *data;
} Mat;

Mat *matrix_new(int m, int n);
Mat *matrix_copy(Mat *src);
Mat *matrix_eye(int m, int n);
Mat *matrix_zeros(int m, int n);
Mat *matrix_mul(Mat *A, Mat *B);
float *matrix_eye_bis(int m, int n);
void matrix_delete(Mat *m);
void matrix_print(Mat *m);
Mat *matrix_reduce(Mat *m, int maxCol);
Mat *matrix_reduce_cond(Mat *m, int col);
void matrix_copy_cond(Mat *A, Mat *B, int col);
float vect_norm(float *u, int n);
float vect_dot(float *u, float *v, int n);
void vect_divide(float *u, float scalar, int n);
void vect_mat_copy(Mat *mat, float *x, int col);
void vect_mat_copy_cond(Mat *mat, float *u, int col, int line);
float *vect_prod_mat(Mat *mat, float *u);
float *vect_prod_mat_trans(Mat *mat, float *u);
float *vect_divide_by_scalar(float *u, float scalar, int n);
void vect_substract(float *res, float *u , float *v, int m);
float *compute_fm(float *u, float *uT, float *w, int n);
void vect_print(float *u, int n);
float *get_column(Mat *mat, int col);
float *get_column_start(Mat *mat, int col);
float absolute(float nb);