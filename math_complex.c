#include "math_complex.h"

// Utils
// complex product
complex complex_prod(complex a1, complex a2) {
  complex res = {0,0};
  res.real = a1.real * a2.real - (a1.imag * a2.imag);
  res.imag = a1.real * a2.imag + a2.real * a1.imag;
  return res;
}
complex complex_add(complex a1, complex a2) {
  complex res = {0,0};
  res.real = a1.real + a2.real;
  res.imag = a1.imag + a2.imag;
  return res;
}
float complex_modulo(complex a1) {
  return sqrt(pow(a1.real, 2) + pow(a1.imag, 2));
}
Mat *matrix_new(int m, int n) {
  Mat *res = malloc(sizeof(Mat));
  if (!res)
    return NULL;
  res->data = malloc(m * n * sizeof(complex));
  if (!res->data)
  {
    free(res);
    return NULL;
  }
  res->n = n;
  res->m = m;
  return res;
}

void matrix_copy(Mat *src, Mat *dest) {
  if (!src || !dest)
    return;
  if (src->m != dest->m && src->n != dest->n)
    return;
  for (int i = 0; i < src->n * src->m; i++) {
      dest->data[i].imag = src->data[i].imag;
      dest->data[i].real = src->data[i].real;      
  }
}

Mat *matrix_eye(int m, int n) {
  Mat *res = matrix_new(m, n);
  if (!res)
    return NULL;
  for (int i = 0; i < n; i++) {
      res->data[i + (n * i)].imag = 0;
      res->data[i + (n * i)].real = 1;
  }
  return res;
}
complex *matrix_eye_bis(int m, int n) {
  complex *res = malloc(m * n * sizeof(complex));
  if (!res)
    return NULL;
  for (int i = 0; i < m * n; i++) {
    res[i].real = 0;
    res[i].imag = 0;
  }
  for (int i = 0; i < n; i++) {
    res[i + (n*i)].real = 1;
    res[i + (n*i)].imag = 0;
  }
  return res;
}
complex *matrix_diag(Mat *A) {
  complex *res = malloc(sizeof(complex) * A->m);
  if (!res)
    return NULL;
#ifdef NAIVE
  res[0] = A->data[0];
  int i = (A->m + 1);
  int j = 1;
  for (;i < A->m * A->n; i += (A->m + 1)) {
    res[j].real = A->data[i].real;
    res[j].imag = A->data[i].imag;
    j++;
  }
  return res;
#elif INTEL_MKL
  cblas_ccopy(A->m, A->data, A->m + 1, res, 1);
  return res;
#endif
}
Mat *matrix_mul(Mat *A, Mat *B) {
  if (A->n != B->m)
  {
    printf("Dim Error\n");
    return NULL;
  }
  Mat *res = matrix_new(A->m, B->n);
  if (!res) {
    printf("Matrix Res error\n");
    return NULL;
  }
#ifdef NAIVE
  for (int i = 0; i < A->n; i++) {
    for (int j = 0; j < B->m; j++) {
      complex sum = {0.0, 0.0};
        for (int k = 0; k < A->n; k++) {
          sum = complex_add(sum, complex_prod(A->data[k + (i * A->n)], B->data[i + (k * B->m)]));
        }
        res->data[i * A->m + j] = sum;
      }
    }
  return res;
#elif INTEL_MKL
  complex alpha = (complex){1.0, 0.0};
  complex beta = (complex){0.0, 0.0};
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->m, B->n, A->n, &alpha, A->data, A->n, B->data, B->n, &beta, res->data, res->n);
  return res;
#endif
}
Mat *matrix_zeros(int m, int n) {
  Mat *res = matrix_new(m, n);
  for (int i = 0; i < m * n; i++) {
    res->data[i] = (complex){0,0};
  }
  return res;
}
void matrix_add(Mat *A, Mat *B, Mat *res) {
    if (!A || !B)
    return;
  if (A->n != B->n && A->m != B->m)
    return;
#ifdef NAIVE
  for (int i = 0; i < A->n * A->m; i++) {
    res->data[i].real = A->data[i].real + B->data[i].real;
    res->data[i].imag = A->data[i].imag + B->data[i].imag;
  }
#elif INTEL_MKL
  vcAdd(A->m * A->n, A->data, B->data, res->data);
#endif
}
void matrix_sub(Mat *A, Mat *B, Mat *res) {
  if (!A || !B)
    return;
  if (A->n != B->n && A->m != B->m)
    return;
  if (!res)
    return;
#ifdef NAIVE
  for (int i = 0; i < A->n * A->m; i++) {
    res->data[i].real = A->data[i].real - B->data[i].real;
    res->data[i].imag = A->data[i].imag - B->data[i].imag;
  }
    
#elif INTEL_MKL
    vcSub(A->m * A->n, A->data, B->data, res->data);
#endif
}
void matrix_scalar(Mat *A, complex scalar) {
  if (!A)
    return;
  if (!A->data)
    return;
  for (int i = 0; i < A->n * A->m; i++) {
    A->data[i] = complex_prod(A->data[i], scalar);
  }
}
Mat *matrix_reduce(Mat *m, int maxCol) {
#ifdef NAIVE
  if (m->n == (maxCol)) {
    Mat *res = matrix_new(m->m, m->n);
    matrix_copy(m, res);
    return res;
  }
  Mat *res = matrix_new(m->m, maxCol);
  if (!res)
    return NULL;
  int j = 0;
  for (int i = 0; i < m->m * m->n; i++) {
    int y = i % m->n;
    res->data[j].real = m->data[i].real;
    res->data[j].imag = m->data[i].imag;    
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
  complex *eyeTmp = matrix_eye_bis(m->n, maxCol); 
  complex alpha = (complex){1.0, 0.0};
  complex beta = (complex){0.0, 0.0};
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m->m, res->n, m->n, &alpha, m->data, m->n, eyeTmp, res->n, &beta, res->data, res->n);
  free(eyeTmp);
  return res;
#endif
}
// A(j:$, j:$)
Mat *matrix_reduce_cond(Mat *m, int col) {
  Mat *res = matrix_new((m->m - col), (m->m - col));
  if (!res)
    return NULL;
  int j = 0;
  for (int i = (m->n * col) + col; i < m->n * m->n; i++) {
    int y = i % m->n;
    res->data[j].real = m->data[i].real;
    res->data[j].imag = m->data[i].imag;
    if ((y + 1) % m->n == 0) {
      i += col;
    }
    j++;
  }
  return res;
}
// A(:, 1:col) = B(:, 1:col)
void matrix_copy_sub(Mat *src, Mat *dest, int col) {
   for (int i = 0; i < dest->m; i++) {
     for (int j = 0; j < col; j++) {
       dest->data[(i * col) + j] = src->data[i * dest->m + j];
     }
   }
}
// A copy B
void matrix_copy_cond(Mat *src, Mat *dest, int col) {
  int j = 0;
  for (int i = (dest->n * col) + col; i < dest->n * dest->n; i++) {
    int y = i % dest->n;
    dest->data[i].real = src->data[j].real;
    dest->data[i].imag = src->data[j].imag;
    if ((y + 1) % dest->n == 0) {
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
    if (mat->data[i].imag == 0)
      printf("%8.6f ", mat->data[i].real);
    else if (mat->data[i].imag < 0)
      printf("%8.6f - %1.3fi ", mat->data[i].real, -mat->data[i].imag);
    else
      printf("%8.6f + %1.3fi ", mat->data[i].real, mat->data[i].imag);
    if ((i + 1) % mat->n == 0)
        printf("\n");
  }
}

Mat *matrix_transpose(Mat *m) {
  Mat *res = matrix_new(m->m, m->n);
  #ifdef NAIVE
    for (int i = 0; i < m->m; i++) {
      for (int j = 0; j < m->n; j++) {
        res->data[j * m->n + i].real = m->data[i * m->m + j].real;
        res->data[j * m->n + i].imag = m->data[i * m->m + j].imag;
      }
    }
    return res;
  #elif INTEL_MKL
    matrix_copy(m, res);
    complex alpha = (complex){1.0, 0.0};
    mkl_cimatcopy('r', 't', m->m, m->n, alpha, res->data, m->n, m->m);
    return res;
  #endif
}

float vect_norm(complex *u, int n) {
#ifdef NAIVE
  if (!u)
    return -1;
  float res = 0;
  for (int i = 0; i < n; i++) {
    res += pow(u[i].real, 2) + pow(u[i].imag, 2);
  }
  return sqrt(res);
#elif INTEL_MKL
  return cblas_scnrm2(n, u, 1);
#endif
}

complex vect_dot(complex *u, complex *v, int n) {
  if (!u || !v)
    return (complex){-1,-1};
  complex res = {0, 0};
  for (int i = 0; i < n; i++) {
    res = complex_add(res, complex_prod(u[i], v[i]));
  }
  return res;
}

void vect_divide(complex *u, complex scalar, int n) {
  for (int i = 0; i < n; i++) {
    float tmp = complex_modulo(scalar) * complex_modulo(scalar);
    u[i].real = ((u[i].real * scalar.real) + (u[i].imag * scalar.imag)) / tmp;
    u[i].imag = ((scalar.real * u[i].imag - u[i].real * scalar.imag))  / tmp;
  }
}

void vect_mat_copy(Mat *mat, complex *u, int col) {
    if (!mat)
        return;
    if (!u)
        return;
    if (!mat->data)
        return;
#ifdef NAIVE
    for (int i = 0; i < mat->m; i++) {
        mat->data[col + i * mat->n].real = u[i].real;
        mat->data[col + i * mat->n].imag = u[i].imag;
    }
#elif INTEL_MKL
    cblas_ccopy(mat->m, u, 1, (mat->data + col), mat->n);
#endif
}
void vect_mat_copy_cond(Mat *mat, complex *u, int col, int line) {
    if (!mat)
        return;
    if (!u)
        return;
    if (!mat->data)
        return;
#ifdef NAIVE    
    if (!line) {
      for (int i = 0; i < (col + 1); i++) {
          mat->data[col + i * mat->n].real = u[i].real;
          mat->data[col + i * mat->n].imag = u[i].imag;

      }
    }
    else {
      int k = 0;
      for (int i = (mat->m * col + col); k < (mat->n - col); i+= mat->m) {
          mat->data[i].real = u[k].real;
          mat->data[i].imag = u[k].imag;
          k++;
      }
    }
#elif INTEL_MKL
    cblas_ccopy(col + 1, u, 1, (mat->data + col), mat->m);
#endif
}
// Store result directly in w
void vect_prod_mat(Mat *A, complex *u, complex *res) {
#ifdef NAIVE
      if (!res)
          return;
      for (int i = 0; i < A->m; i++) {
          complex sum = {0.0, 0.0};
          for (int j = 0; j < A->n; j++) {
              sum = complex_add(sum, complex_prod(A->data[j + (i * A->n)], u[j]));
          }
        res[i] = sum;
    }
#elif INTEL_MKL
  complex *alpha = malloc(sizeof(complex));
  alpha->real = 1.0;
  alpha->imag = 0.0;
  complex beta = (complex){0, 0};
  cblas_cgemv(CblasRowMajor, CblasNoTrans, A->m, A->n, alpha, A->data, A->n, u, 1, &beta, res, 1);
#endif
}

complex *vect_prod_mat_trans(Mat *mat, complex *u) {
  complex *res = malloc(sizeof(complex) * mat->n);
  if (!res)
          return NULL;
#ifdef NAIVE   
      for (int i = 0; i < mat->n; i++) {
          complex sum = {0.0, 0.0};
          for (int j = 0; j < mat->m; j++) {
              sum = complex_add(sum, complex_prod(mat->data[i + (j * mat->n)], u[j]));
          }
        res[i] = sum;
    }
    return res;
#elif INTEL_MKL
  complex alpha = (complex){1.0, 0.0};
  complex beta = (complex){0, 0};
  cblas_cgemv(CblasRowMajor, CblasTrans, mat->m, mat->n, &alpha, mat->data, mat->n, u, 1, &beta, res, 1);
  return res;
#endif
}

void compute_fm(complex *fm, complex *u,  complex *w, int n, int m) {
#ifdef NAIVE   
   for (int i = 0; i < m; i++) {
      complex sum = {0,0};
      for (int j = 0; j < n; j++) {
        sum = complex_add(sum, complex_prod(complex_prod(u[i], u[j]), w[j]));
      }
      fm[i].real = w[i].real - sum.real;
      fm[i].imag = w[i].imag - sum.imag;
  }
#elif INTEL_MKL
  
  // v0 * v0.T
  complex *res = malloc(sizeof(complex) * n * m);
  complex alpha = (complex){1.0, 0};
  complex beta = (complex){0, 0};
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, m, 1, &alpha, u, 1, u, 1, &beta, res, m);
        
  complex *tmp = malloc(sizeof(complex) * m);
  
  // fm = w - v0 * v0.T * w
  cblas_cgemv(CblasRowMajor, CblasNoTrans, m, m, &alpha, res, m, w, 1, &beta, tmp, 1);
  vcSub(m, w, tmp, fm);
  free(tmp);
  free(res);
#endif
}

complex *vect_divide_by_scalar(complex *u, complex scalar, int n) {
  complex *res = malloc(sizeof(complex) * n);
  if (!res)
    return NULL;
  for (int i = 0; i < n; i++) {
    float tmp = complex_modulo(scalar) * complex_modulo(scalar);
    res[i].real = ((u[i].real * scalar.real) + (u[i].imag * scalar.imag)) / tmp;
    res[i].imag = ((scalar.real * u[i].imag - u[i].real * scalar.imag)) / tmp;
  }
  return res;
}
void vect_print(complex *u, int n) {
  if (!u)
    return;

  for (int i = 0; i < n; i++)
    printf("%8.16f + %.2f\n", u[i].real, u[i].imag);
}

complex *get_column(Mat *mat, int col) {
  if (!mat)
    return NULL;

  complex *x = malloc(mat->m * sizeof(complex));
  if (!x)
    return NULL;

  for (int i = 0; i < mat->m; i++)
    x[i] = mat->data[col + i * mat->m];    
  return x;
}

complex *get_column_start(Mat *mat, int col) {
  if (!mat)
    return NULL;

  complex *x = malloc((mat->m - col) * sizeof(complex));
  if (!x)
    return NULL;
 if (col >= mat->m)
    return NULL;
  for (int i = col; i < mat->m; i++)
    x[i - col] = mat->data[col + i * mat->m];    
  return x;
}
void vect_substract(complex *res, complex *u , complex *v, int m) {
#ifdef NAIVE
  for (int i = 0; i < m; i++) {
    res[i].real = u[i].real - v[i].real;
    res[i].imag = u[i].imag - v[i].imag;
  }
#elif INTEL_MKL
  vcSub(m, u, v, res);
#endif
}
float absolute(complex nb) {
  return complex_modulo(nb);
}
void vect_add(complex *res, complex *a, complex *b, int m) {
  //#ifdef NAIVE
    for (int i = 0; i < m; i++) {
      res[i].real = a[i].real + b[i].real;
      res[i].imag = a[i].imag + b[i].imag;
    }
 /* #elif INTEL_MKL
    vsAdd(m, a, b, res);
  #endif*/
}
void vect_scalar(complex *u, complex scalar, int n) {
#ifdef NAIVE
  for (int i = 0; i < n; i++)
  {
    u[i] = complex_prod(u[i], scalar);
  }
#elif INTEL_MKL
  complex *vect_eye = malloc(sizeof(complex) * n);
  for (int i = 0; i < n; i++) {
    vect_eye[i] = (complex) {1, 0};
  }
  cblas_caxpy(n, &scalar, vect_eye, 1, u, 1);
  free(vect_eye);
#endif
}

