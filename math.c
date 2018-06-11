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
  for (int i = 0; i < n; i++) {
      res->data[i + (n * i)] = 1;
  }
  return res;
}
float *matrix_eye_bis(int m, int n) {
  float *res = malloc(m * n * sizeof(float));
  if (!res)
    return NULL;
  for (int i = 0; i < m * n; i++) {
    res[i] = 0;
  }
  for (int i = 0; i < n; i++) {
    res[i + (n*i)] = 1;
  }
  return res;
}
Mat *matrix_mul(Mat *A, Mat *B) {
  Mat *res = matrix_new(A->m, B->n);
  if (!res)
    return NULL;
  if (A->n != B->m)
    return NULL;
  for (int i = 0; i < A->n; i++) {
    for (int j = 0; j < B->m; j++) {
      float sum = 0.0;
        for (int k = 0; k < A->n; k++) {
          sum += A->data[k + (i * A->n)] * B->data[j + (k * B->m)];
        res->data[i * A->m + j];
      }
    }
  }
  return res;
}
Mat *matrix_zeros(int m, int n) {
  Mat *res = matrix_new(m, n);
  return res;
}

Mat *matrix_reduce(Mat *m, int maxCol) {



#ifdef NAIVE
  if (m->n == (maxCol))
    return matrix_copy(m);
  Mat *res = matrix_new(m->m, maxCol);
  if (!res)
    return NULL;
  int j = 0;
  for (int i = 0; i < m->m * m->n; i++) {
    int y = i % m->n;
    res->data[j] = m->data[i];
    if ((y + 1) % maxCol == 0) {
      i += m->n - (y + 1);
    }
    j++;
  }
  return res;
#elif INTEL_MKL
  Mat *res = matrix_new(m->m, maxCol);
  if (!res)
    return NULL;
  float *eyeTmp = matrix_eye_bis(m->n, maxCol);  
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m->m, res->n, m->n, 1, m->data, m->n, eyeTmp, res->n, 0, res->data, res->n);
  free(eyeTmp);
  return res;
#endif
}
Mat *matrix_reduce_cond(Mat *m, int col) {
  Mat *res = matrix_new((m->m - col), (m->m - col));
  if (!res)
    return NULL;
  int j = 0;
  for (int i = (m->n * col) + col; i < m->n * m->n; i++) {
    int y = i % m->n;
    res->data[j] = m->data[i];
    if ((y + 1) % m->n == 0) {
      i += col;
    }
    j++;
  }
  return res;
}
// A copy B
void matrix_copy_cond(Mat *A, Mat *B, int col) {
  int j = 0;
  for (int i = (A->n * col) + col; i < A->n * A->n; i++) {
    int y = i % A->n;
    A->data[i] = B->data[j];
    if ((y + 1) % A->n == 0) {
      i += col;
    }
    j++;
  }
}
void matrix_delete(Mat *mat) {
  if (!mat)
    return;
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
#ifdef NAIVE
  if (!u)
    return -1;
  float res = 0;
  for (int i = 0; i < n; i++) {
    res += pow(u[i], 2);
  }
  return sqrt(res);
#elif INTEL_MKL
  return cblas_snrm2(n, u, 1);
#endif
}

float vect_dot(float *u, float *v, int n) {
  if (!u || !v)
    return -1;
#ifdef NAIVE
  float res = 0;
  for (int i = 0; i < n; i++) {
    res += u[i] * v[i];
  }
  return res;
#elif INTEL_MKL
  return cblas_sdot(n, u, 1, v, 1);
#endif
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
#ifdef NAIVE
    for (int i = 0; i < mat->m; i++) {
        mat->data[col + i * mat->n] = u[i];
    }
#elif INTEL_MKL
    cblas_scopy(mat->m, u, 1, (mat->data + col), mat->n);
#endif
}
void vect_mat_copy_cond(Mat *mat, float *u, int col, int line) {
    if (!mat)
        return;
    if (!u)
        return;
    if (!mat->data)
        return;
#ifdef NAIVE    
    if (!line) {
      for (int i = 0; i < (col + 1); i++) {
          mat->data[col + i * mat->n] = u[i];
      }
    }
    else {
      int k = 0;
      for (int i = (mat->m * col + col); k < (mat->n - col); i+= mat->m) {
          mat->data[i] = u[k];
          k++;
      }
    }
#elif INTEL_MKL
    cblas_scopy(col + 1, u, 1, (mat->data + col), mat->m);
#endif
}
// Store result directly in w
void vect_prod_mat(Mat *A, float *u, float *res) {
#ifdef NAIVE
      if (!res)
          return;
      for (int i = 0; i < A->m; i++) {
          float sum = 0.0;
          for (int j = 0; j < A->n; j++) {
              sum += A->data[j + (i * A->n)] * u[j];
          }
        res[i] = sum;
    }
#elif INTEL_MKL
    //    vect_prod_mat(VmReduce, h2, tmp2);

  cblas_sgemv(CblasRowMajor, CblasNoTrans, A->m, A->n, 1, A->data, A->n, u, 1, 0, res, 1);
  //cblas_sgemv(CblasRowMajor, CblasNoTrans, m_vmReduce2, n_vmReduce2, 1, VmReduce2, n_vmReduce2, h, 1, 0, tmp, 1);

#endif
}

float *vect_prod_mat_trans(Mat *mat, float *u) {
  float *res = malloc(sizeof(float) * mat->n);
  if (!res)
          return NULL;
#ifdef NAIVE   
      for (int i = 0; i < mat->n; i++) {
          float sum = 0.0;
          for (int j = 0; j < mat->m; j++) {
              sum += mat->data[i + (j * mat->n)] * u[j];
          }
        res[i] = sum;
    }
    return res;
#elif INTEL_MKL
  cblas_sgemv(CblasRowMajor, CblasTrans, mat->m, mat->n, 1, mat->data, mat->n, u, 1, 0, res, 1);
  return res;
#endif
}

float *compute_fm(float *u,  float *w, int n, int m) {
#ifdef NAIVE   
   float *res = malloc(sizeof(float) * m);

   for (int i = 0; i < m; i++) {
      float sum = 0;
      for (int j = 0; j < n; j++) {
        sum += u[i] * u[j] * w[j];
      }
      res[i] = w[i] - sum;
  }
  return res;
#elif INTEL_MKL
  
  // v0 * v0.T
  float *res = malloc(sizeof(float) * n * m);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, m, 1, 1, u, 1, u, 1, 0, res, m);
        
  float *tmp = malloc(sizeof(float) * m);
  float *fm = malloc(sizeof(float) * m);
  
  // fm = w - v0 * v0.T * w
  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, m, 1, res, m, w, 1, 0, tmp, 1);
  vsSub(m, w, tmp, fm);
  free(tmp);
  free(res);
  return fm;
#endif
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

  float *x = malloc((mat->m - col) * sizeof(float));
  if (!x)
    return NULL;
 if (col >= mat->m)
    return NULL;
  for (int i = col; i < mat->m; i++)
    x[i - col] = mat->data[col + i * mat->m];    
  return x;
}
void vect_substract(float *res, float *u , float *v, int m) {
#ifdef NAIVE
  for (int i = 0; i < m; i++) {
    res[i] = u[i] - v[i];
  }
#elif INTEL_MKL
  vsSub(m, u, v, res);
#endif
}
float absolute(float nb) {
  if (nb < 0)
    return -nb;
  return nb;
}


