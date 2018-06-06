#include "matrix.h"


Mat *matrix_new(int m, int n) {
  Mat *res = malloc(sizeof(Mat));
  if (!res)
    return NULL;
  res->data = malloc(sizeof(double *) * m);
  if (!res->data)
  {
    free(res);
    return NULL;
  }
  for (int i = 0; i < m; i++)
  {
    res->data[i] = malloc(sizeof(double) * n);
    if (!res->data[i]) {
      for (int k = 0; k < i; k++)
        free(res->data[k]);
      free(res->data);
      free(res);
      return NULL;
    }
  }
  res->n = n;
  res->m = m;
  return res;
}

Mat *matrix_copy(Mat *src) {
  if (!src)
    return NULL;
  Mat *res = matrix_new(src->m, src->n);
  if (!res)
    return NULL;
  for (int i = 0; i < src->n; i++)
  {
    for (int j = 0; j < src->m; j++)
      res->data[i][j] = src->data[i][j];
  }
  return res;
}

Mat *matrix_eye(int m, int n) {
  Mat *res = matrix_new(m, n);
  if (!res)
    return NULL;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++)
    {
      if (i == j)
        res->data[i][j] = 1;
      else
        res->data[i][j] = 0;
    }
  }
  return res;
}


Mat *matrix_zeros(int m, int n) {
  Mat *res = matrix_new(m, n);
  if (!res)
    return NULL;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++)
        res->data[i][j] = 0;
  }
  return res;
}

Mat *matrix_mul(Mat *mat1, Mat *mat2) {
  if (!mat1 || !mat1->data)
    return NULL;
  if (!mat2 || !mat2->data)
    return NULL;
  
  if (mat1->n != mat2->m)
    return NULL;
  
  Mat *res = matrix_new(mat1->m, mat1->n);
    
  for (int i = 0; i < mat1->n; i++) {
    for (int j = 0; j < mat1->m; j++)
    {   
      double sum = 0;
      for (int k = 0; k < mat1->n; k++)
        sum += mat1->data[i][k] * mat2->data[k][j];
      res->data[i][j] = sum;
    }
  }
  return res;

}
void matrix_delete(Mat *mat) {
  if (!mat)
    return;
  if (!mat->data) {
    free(mat);
    return;
  }
  for (int i = 0; i < mat->m; i++)
    free(mat->data[i]);
  free(mat);
}

void matrix_print(Mat *mat) {
  if (!mat)
    return;
  if (!mat->data)
    return;

  for (int i = 0; i < mat->n; i++) {
    for (int j = 0; j < mat->m; j++)
      printf("%8.3f ", mat->data[i][j]);
    printf("\n");
  }
}

double vect_norm(double *u, int n) {
  if (!u)
    return -1;
  double res = 0;
  for (int i = 0; i < n; i++) {
    res += pow(u[i], 2);
  }
  return sqrt(res);
}

double vect_dot(double *u, double *v, int n) {
  if (!u || !v)
    return -1;
  double res = 0;
  for (int i = 0; i < n; i++) {
    res += u[i] * v[i];
  }
  return res;
}

void vect_divide(double *u, double scalar, int n) {
  for (int i = 0; i < n; i++)
    u[i] /= scalar;
}

void vect_mat_copy(Mat *mat, double *u, int col) {
  if (!mat)
  {
    printf("Func vect_copy:\n");
    printf("Error Mat is NULL\n");
    return;
  }
  
  if (!u) {
    printf("Func vect_copy:\n");
    printf("Error u is NULL\n");
    return;
  }

  for (int j = col; j < mat->n; j++)
    mat->data[j][col] = u[j - col];
}

double *vect_prod_mat(Mat *mat, double *u) {
  if (!mat)
  {
    printf("Func vect_prod_mat:\n");
    printf("Error Mat is NULL\n");
    return NULL;
  }
  
  if (!u) {
    printf("Func vect_copy:\n");
    printf("Error u is NULL\n");
    return NULL;
  }
  double *res = malloc(sizeof(double) * mat->m);
  if (!res)
    return NULL;

  for (int i = 0; i < mat->m; i++) {
    double sum = 0;
    for (int j = 0; j < mat->n; j++) {
      sum += u[j] * mat->data[i][j];
    }
    res[i] = sum;
  }
  return res;
}

void vect_print(double *u, int n) {
  if (!u)
    return;

  for (int i = 0; i < n; i++)
    printf("%8.16f\n", u[i]);
}

double *get_column(Mat *mat, int col) {
  if (!mat)
    return NULL;

  double *x = malloc(mat->m * sizeof(double));
  if (!x)
    return NULL;

  for (int i = 0; i < mat->m; i++)
    x[i] = mat->data[i][col];    
  
  return x;
}


double *get_column_start(Mat *mat, int col) {
  if (!mat)
    return NULL;

  double *x = calloc(mat->m, sizeof(double));
  if (!x)
    return NULL;

  for (int i = col; i < mat->m; i++)
    x[i - col] = mat->data[i][col];    
  
  return x;
}

double absolute(double nb) {
  if (nb < 0)
    return -nb;
  return nb;
}


