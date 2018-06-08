#include <stdlib.h>
#include <math.h>
#include <stdio.h>

typedef struct mat_ {
  int m;
  int n;
  float **data;
} Mat;

Mat *matrix_new(int m, int n);
Mat *matrix_copy(Mat *src);
Mat *matrix_eye(int m, int n);
Mat *matrix_zeros(int m, int n);
Mat *matrix_mul(Mat *mat1, Mat *mat2);
void matrix_delete(Mat *m);
void matrix_print(Mat *m);
float vect_norm(float *u, int n);
float vect_dot(float *u, float *v, int n);
void vect_divide(float *u, float scalar, int n);
void vect_mat_copy(Mat *mat, float *x, int col);
float *vect_prod_mat(Mat *mat, float *u);
void vect_print(float *u, int n);
float *get_column(Mat *mat, int col);
float *get_column_start(Mat *mat, int col);
float absolute(float nb);
