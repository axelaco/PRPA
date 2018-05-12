#include <stdlib.h>
#include <math.h>
#include <stdio.h>

typedef struct mat_ {
  int m;
  int n;
  double **data;
} Mat;

Mat *matrix_new(int m, int n);
Mat *matrix_copy(Mat *src);
Mat *matrix_eye(int m, int n);
Mat *matrix_zeros(int m, int n);
Mat *matrix_mul(Mat *mat1, Mat *mat2);
void matrix_delete(Mat *m);
void matrix_print(Mat *m);
double vect_norm(double *u, int n);
double vect_dot(double *u, double *v, int n);
void vect_divide(double *u, double scalar, int n);
void vect_mat_copy(Mat *mat, double *x, int col);
double *vect_prod_mat(Mat *mat, double *u);
void vect_print(double *u, int n);
double *get_column(Mat *mat, int col);
double *get_column_start(Mat *mat, int col);
double absolute(double nb);
