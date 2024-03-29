#include "math.h"
#ifdef NAIVE
#include <lapacke.h>
#endif
// QSort Algorithm
static int cmp_abs (void const *a, void const *b)
{
   /* definir des pointeurs type's et initialise's
      avec les parametres */
   float const *pa = a;
   float const *pb = b;

   /* evaluer et retourner l'etat de l'evaluation (tri croissant) */
   return abs(*pa) - abs(*pb);
}

float *rritz(Mat *Tm, float *mx, float *fm, int k, float nrmfr) {
  Mat *q = matrix_zeros(Tm->m, Tm->n);
  float *w = qr_alg_eigen(Tm, q);
  int m = Tm->m;
  float *qRow = matrix_get_row(q, q->n);
  // sort by Magnitude
  qsort(w, Tm->m, sizeof(*w), cmp_abs);
  for (int i = 0; i < m; i++) {
    qRow[i] = abs(qRow[i]);
  }

  float *ritz = malloc(sizeof(float) * k);
  for (int i = 0; i < k; i++) {
    ritz[i] = vect_norm(fm, Tm->m) * qRow[i];
  }
  *mx = vect_norm(ritz, k);
  return w;
}

// TODO Implement method for NAIVE implem
float *qr_alg_eigen(Mat *A, Mat *eigVector) {
#ifdef NAIVE
float *diag = matrix_diag(A);
  float *off_diag = matrix_off_diag(A);
  float *WORK = malloc(sizeof(float) * 2 * A->n - 2);
  int n = A->n;
  int info;
  LAPACK_ssteqr("I", &n, diag, off_diag, eigVector->data, &n, WORK, &info);
  //LAPACKE_shseqr(LAPACK_ROW_MAJOR, 'E', 'I', A->n, 1, A->n, A->data, A->n, wr, wi, z->data, A->n);
 // free(wi);
  return diag;
#elif INTEL_MKL
 // float *wr = malloc(sizeof(float) * A->n);
//  float *wi = malloc(sizeof(float) * A->n);
//  Mat *z = matrix_new(A->m, A->n);
  float *diag = matrix_diag(A);
  float *off_diag = matrix_off_diag(A);
  LAPACKE_ssteqr(LAPACK_ROW_MAJOR, 'I', A->n, diag, off_diag, eigVector->data, eigVector->n);
  //LAPACKE_shseqr(LAPACK_ROW_MAJOR, 'E', 'I', A->n, 1, A->n, A->data, A->n, wr, wi, z->data, A->n);
 // free(wi);
  return diag;
#endif
}


Mat *matrix_new(int m, int n) {
  Mat *res = malloc(sizeof(Mat));
  if (!res)
    return NULL;
  res->data = malloc(m * n * sizeof(float));
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
      dest->data[i] = src->data[i];
  }
}

Mat *matrix_eye(int m, int n) {
  Mat *res = matrix_new(m, n);
  if (!res)
    return NULL;
  for (int i = 0; i < m * n; i++) {
    res->data[i] = 0;
  }
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
float *matrix_off_diag(Mat *A) {
  float *res = malloc(sizeof(float) * A->n - 1);
  int j = 0;
  for (int i = 1; i < A->m * A->n; i += (A->m + 1)) {
    res[j] = A->data[i];
    j++;
  }
  return res;
}
float *matrix_get_row(Mat *A, int m) {
  float *res = malloc(sizeof(float) * A->n);
  for (int i = (A->m*m); i < A->m * A->n; i++) {
    res[i - (A->m * m)] = A->data[i];
  }
  return res;
}
float *matrix_diag(Mat *A) {
  float *res = malloc(sizeof(float) * A->m);
  if (!res)
    return NULL;
#ifdef NAIVE
  res[0] = A->data[0];
  int i = (A->m + 1);
  int j = 1;
  for (;i < A->m * A->n; i += (A->m + 1)) {
    res[j] = A->data[i];
    j++;
  }
  return res;
#elif INTEL_MKL
  cblas_scopy(A->m, A->data, A->m + 1, res, 1);
  return res;
#endif
}
void matrix_mul_bis(Mat *res, Mat *A, Mat *B) {
    if (A->n != B->m)
  {
    printf("Dim Error\n");
    return;
  }
  if (!res) {
    printf("Matrix Res error\n");
    return;
  }
#ifdef NAIVE
  for (int i = 0; i < A->n; i++) {
    for (int j = 0; j < B->m; j++) {
      float sum = 0.0;
        for (int k = 0; k < A->n; k++) {
          sum += A->data[k + (i * A->n)] * B->data[j + (k * B->m)];
        }
        res->data[i * A->m + j] = sum;
      }
    }
#elif INTEL_MKL
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->m, B->n, A->n, 1, A->data, A->n, B->data, B->n, 0, res->data, res->n);
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
      float sum = 0.0;
        for (int k = 0; k < A->n; k++) {
          sum += A->data[k + (i * A->n)] * B->data[j + (k * B->m)];
        }
        res->data[i * A->m + j] = sum;
      }
    }
  return res;
#elif INTEL_MKL
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->m, B->n, A->n, 1, A->data, A->n, B->data, B->n, 0, res->data, res->n);
  return res;
#endif
}
Mat *matrix_zeros(int m, int n) {
  Mat *res = matrix_new(m, n);
  for (int i = 0; i < m * n; i++)
    res->data[i] = 0;
  return res;
}
void matrix_add(Mat *A, Mat *B, Mat *res) {
    if (!A || !B)
    return;
  if (A->n != B->n && A->m != B->m)
    return;
#ifdef NAIVE
  for (int i = 0; i < A->n * A->m; i++)
    res->data[i] = A->data[i] + B->data[i];
#elif INTEL_MKL
  vsAdd(A->m * A->n, A->data, B->data, res->data);
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
  for (int i = 0; i < A->n * A->m; i++)
    res->data[i] = A->data[i] - B->data[i];
#elif INTEL_MKL
    vsSub(A->m * A->n, A->data, B->data, res->data);
#endif
}
void matrix_scalar(Mat *A, float scalar) {
  if (!A)
    return;
  if (!A->data)
    return;
  for (int i = 0; i < A->n * A->m; i++) {
    A->data[i] *= scalar;
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
    dest->data[i] = src->data[j];
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
    printf("%8.6f ", mat->data[i]);
    if ((i + 1) % mat->n == 0)
        printf("\n");
  }
}

Mat *matrix_transpose(Mat *m) {
  Mat *res = matrix_new(m->m, m->n);
  #ifdef NAIVE
    for (int i = 0; i < m->m; i++) {
      for (int j = 0; j < m->n; j++) {
        res->data[j * m->n + i] = m->data[i * m->m + j];
      }
    }
    return res;
  #elif INTEL_MKL
    matrix_copy(m, res);
    mkl_simatcopy ('r', 't', m->m, m->n, 1, res->data, m->n, m->m);
    return res;
  #endif
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
  cblas_sgemv(CblasRowMajor, CblasNoTrans, A->m, A->n, 1, A->data, A->n, u, 1, 0, res, 1);
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

void compute_fm(float *fm, float *u,  float *w, int n, int m) {
#ifdef NAIVE
   for (int i = 0; i < m; i++) {
      float sum = 0;
      for (int j = 0; j < n; j++) {
        sum += u[i] * u[j] * w[j];
      }
      fm[i] = w[i] - sum;
  }
#elif INTEL_MKL

  // v0 * v0.T
  float *res = malloc(sizeof(float) * n * m);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, m, 1, 1, u, 1, u, 1, 0, res, m);

  float *tmp = malloc(sizeof(float) * m);

  // fm = w - v0 * v0.T * w
  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, m, 1, res, m, w, 1, 0, tmp, 1);
  vsSub(m, w, tmp, fm);
  free(tmp);
  free(res);
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
    printf("%8.5f\n", u[i]);
}

float *get_column(Mat *mat, int col) {
  if (!mat)
    return NULL;

  float *x = malloc(mat->m * sizeof(float));
  if (!x)
    return NULL;

  for (int i = 0; i < mat->m; i++) {
    x[i] = mat->data[col + i * mat->n];
  }
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
void vect_copy(float *src, float *dest, int m) {
  for (int i = 0; i < m; i++) {
    dest[i] = src[i];
  }
}
void vect_add(float *res, float *a, float *b, int m) {
  //#ifdef NAIVE
    for (int i = 0; i < m; i++)
      res[i] = a[i] + b[i];
 /* #elif INTEL_MKL
    vsAdd(m, a, b, res);
  #endif*/
}
void vect_scalar(float *u, float scalar, int n) {
//#ifdef NAIVE
  for (int i = 0; i < n; i++)
    u[i] *= scalar;
    /*
#elif INTEL_MKL
  float *vect_eye = malloc(sizeof(float) * n);
  for (int i = 0; i < n; i++)
    vect_eye[i] = 1;
  cblas_saxpy(n, scalar, vect_eye, 1, u, 1);
  free(vect_eye);
#endif
*/
}
