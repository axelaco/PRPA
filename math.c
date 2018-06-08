#include "math.h"


Mat *matrix_new(int m, int n) {
  Mat *res = malloc(sizeof(Mat));
  if (!res)
    return NULL;
  res->data = calloc(m * n, sizeof(float));
  if (!res->data)
  {
    free(res);
    return NULL;
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
  for (int i = 0; i < src->n * src->m; i++) {
      res->data[i] = src->data[i];
  }
  return res;
}

Mat *matrix_eye(int m, int n) {
  Mat *res = matrix_new(m, n);
  if (!res)
    return NULL;
  for (int i = 0; i < m; i++) {
      res->data[i + (n * i)] = 1;
  }
  return res;
}


Mat *matrix_zeros(int m, int n) {
  Mat *res = matrix_new(m, n);
  return res;
}

Mat *matrix_reduce(Mat *m, int maxCol) {
  if (m->n == (maxCol))
    return m;
  Mat *res = matrix_new(m->m, maxCol);
  if (!res)
    return NULL;
  int j = 0;
  for (int i = 0; i < m->n * m->n; i++) {
    int y = i % m->n;
    res->data[j] = m->data[i];
    if ((y + 1) % maxCol == 0) {
      i += m->n - (y + 1);
    }
    j++;
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

  free(mat->data);
  free(mat);
}

void matrix_print(Mat *mat) {
  if (!mat)
    return;
  if (!mat->data)
    return;
  for (int i = 0; i < mat->n * mat->m; i++) {
    printf("%8.3f ", mat->data[i]);
    if ((i + 1) % mat->n == 0)
        printf("\n");
  }
}

float vect_norm(float *u, int n) {
  if (!u)
    return -1;
  float res = 0;
  for (int i = 0; i < n; i++) {
    res += pow(u[i], 2);
  }
  return sqrt(res);
}

float vect_dot(float *u, float *v, int n) {
  if (!u || !v)
    return -1;
  float res = 0;
  for (int i = 0; i < n; i++) {
    res += u[i] * v[i];
  }
  return res;
}

void vect_divide(float *u, float scalar, int n) {
  for (int i = 0; i < n; i++)
    u[i] /= scalar;
}

void vect_mat_copy(Mat *mat, float *u, int col) {
    if (!mat)
        return;
    if (!u)
        return;
    if (!mat->data)
        return;
    for (int i = 0; i < mat->n; i++) {
        mat->data[col + i * mat->n] = u[i];
    }
}
void vect_mat_copy_cond(Mat *mat, float *u, int col) {
    if (!mat)
        return;
    if (!u)
        return;
    if (!mat->data)
        return;
    for (int i = 0; i < (col + 1); i++) {
        mat->data[col + i * mat->n] = u[i];
    }
}
float *vect_prod_mat(Mat *mat, float *u) {
      float *res = malloc(sizeof(float) * mat->n);
      if (!res)
          return NULL;
      for (int i = 0; i < mat->m; i++) {
          float sum = 0.0;
          for (int j = 0; j < mat->n; j++) {
              sum += mat->data[j + (i * mat->n)] * u[j];
          }
        res[i] = sum;
    }
    return res;
}

float *vect_prod_mat_trans(Mat *mat, float *u) {
  float *res = malloc(sizeof(float) * mat->n);
      if (!res)
          return NULL;
      for (int i = 0; i < mat->n; i++) {
          float sum = 0.0;
          for (int j = 0; j < mat->m; j++) {
              sum += mat->data[i + (j * mat->n)] * u[j];
          }
        res[i] = sum;
    }
    return res;
}

float *compute_fm(float *u, float *uT, float *w, int n) {
   float *res = malloc(sizeof(float) * n);
   for (int i = 0; i < n; i++) {
      float sum = 0;
      for (int j = 0; j < n; j++) {
        sum += u[i] * uT[j] * w[j];
      }
      res[i] = w[i] - sum;
  }
  return res;
}

float *vect_divide_by_scalar(float *u, float scalar, int n) {
  float *res = malloc(sizeof(float) * n);
  if (!res)
    return NULL;
  for (int i = 0; i < n; i++) {
    res[i] = u[i] / scalar;
  }
  return res;
}
void vect_print(float *u, int n) {
  if (!u)
    return;

  for (int i = 0; i < n; i++)
    printf("%8.16f\n", u[i]);
}

float *get_column(Mat *mat, int col) {
  if (!mat)
    return NULL;

  float *x = malloc(mat->m * sizeof(float));
  if (!x)
    return NULL;

  for (int i = 0; i < mat->m; i++)
    x[i] = mat->data[col + i * mat->m];    
  return x;
}

float *get_column_start(Mat *mat, int col) {
  if (!mat)
    return NULL;

  float *x = malloc(mat->m * sizeof(float));
  if (!x)
    return NULL;
 if (col >= mat->m)
    return NULL;
  for (int i = col; i < mat->m; i++)
    x[i] = mat->data[col + i * mat->m];    
  return x;
}
void vect_substract(float *res, float *u , float *v, int m) {
  for (int i = 0; i < m; i++) {
    res[i] = u[i] - v[i];
  }
}
float absolute(float nb) {
  if (nb < 0)
    return -nb;
  return nb;
}


