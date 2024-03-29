#include "math_cuda.h"
#include <time.h>
#define N 10
#define K 5
#define M 20

static int cmp_abs (void const *a, void const *b)
{
   /* definir des pointeurs type's et initialise's
      avec les parametres */
   float const *pa = a;
   float const *pb = b;

   /* evaluer et retourner l'etat de l'evaluation (tri croissant) */
   return abs(*pa) - abs(*pb);
}

// QSort Algorithm
static int my_compare (void const *a, void const *b)
{
   /* definir des pointeurs type's et initialise's
      avec les parametres */
   float const *pa = a;
   float const *pb = b;

   /* evaluer et retourner l'etat de l'evaluation (tri croissant) */
   return abs(*pb) - abs(*pa);
}

void lanczos_facto(cublasHandle_t handle, Mat *A, float *v0, int k, int m, Mat *Vm, Mat *Tm, float *fm) {
  float b1 = 0;
  float *wPrime = malloc(sizeof(float) * A->m);
  float *v = malloc(sizeof(float) * A->m);
  vect_copy(v0, v, A->m);
  for (int j = k - 1; j < m - 1; j++) {
    // wj' = A * vj
    vect_prod_mat(handle, A, v, wPrime);

    // Vm(:,j) = vj
    vect_mat_copy(handle, Vm, v, j);

    //T(j,j) = wj'.T * vj
    float a1 = vect_dot(handle, wPrime, v, A->m);
    Tm->data[(Tm->m * j) + j]  = a1;
    // wj = wj' - T(j,j)*vj - Bj*vj-1
    vect_scalar(v, a1, A->m);
    if (j > 0) {
      float* vjOld = get_column(handle, Vm, j - 1);
      vect_scalar(vjOld, b1, A->m);
      float *tmp = malloc(sizeof(float) * A->m);
      vect_substract(tmp, wPrime, v, A->m);
      vect_substract(fm, tmp, vjOld, A->m);
      free(tmp);
      free(vjOld);
    } else {
      vect_substract(fm, wPrime, v, A->m);
    }
    // Bj = ||wj||
    b1 = vect_norm(handle, fm, A->m);
    Tm->data[(Tm->m * j) + j + 1] = b1;
    Tm->data[Tm->m * (j + 1) + j] = b1;
    free(v);
    v = vect_divide_by_scalar(fm, b1, A->m);
  }
  vect_mat_copy(handle, Vm, v, m - 1);
  vect_prod_mat(handle, A, v, fm);
  Tm->data[m * m - 1] = vect_dot(handle, fm, v, A->m);
  free(v);
  free(wPrime);
}

float *lanczos_ir(cublasHandle_t handle, cusolverDnHandle_t cusolverH,
  Mat *A, float *v0, int k, int m) {
  Mat *Vm = matrix_zeros(A->n, m);
  Mat *Tm = matrix_zeros(m, m);
  Mat *q = NULL;
  float residual = 1;
  float *fm = malloc(sizeof(float) * A->m);
  Mat *Qm = NULL;
  Mat *QT = NULL;
  Mat *QmT = NULL;
  float eps = 0.00001;
  lanczos_facto(handle, A, v0, 1, m, Vm, Tm, fm);
  int nb_iter = 0;
  while(residual > eps && nb_iter < 100) {
    q = matrix_zeros(Tm->m, Tm->n);
    float *eigs = qr_alg_eigen(Tm, q);//rritz(Tm, &residual, fm, k, residual);
    qsort(eigs, Tm->m, sizeof(*eigs), cmp_abs);
    Qm = matrix_eye(m, m);
    for (int j = m - k; j >= 0; j--) {
      Mat *mat_tmp = matrix_new(Tm->m, Tm->n);
      Mat *eye = matrix_eye(Tm->n, Tm->n);
      Mat *R = matrix_zeros(Tm->m, Tm->n);
      Mat *Q = matrix_zeros(Tm->m, Tm->n);
      // s * I(:,:)
      matrix_scalar(eye, eigs[j]);
      // H(:,:) - s * I(:,:)
      matrix_sub(Tm, eye, mat_tmp);

      qr(handle, cusolverH, mat_tmp, R, Q);
      // Hm = Qj*HmQj

      QT = matrix_transpose(handle, Q);
      Mat *tmp_mul = matrix_zeros(QT->m, Tm->n);
      matrix_mul_bis(handle, tmp_mul, QT, Tm);
      matrix_mul_bis(handle, Tm, tmp_mul, Q);
      QmT = matrix_transpose(handle, Qm);
      matrix_mul_bis(handle, Qm, QmT, Q);

      // delete all temporaries variables
      matrix_delete(tmp_mul);
      matrix_delete(mat_tmp);
      matrix_delete(eye);
      matrix_delete(R);
      matrix_delete(Q);
      matrix_delete(QT);
      matrix_delete(QmT);
    }
    free(eigs);
    float *tmp_fm = malloc(sizeof(float) * A->m);
    float *tmp_fm2 = malloc(sizeof(float) * A->m);

    vect_copy(fm, tmp_fm, A->m);
    vect_scalar(tmp_fm, Qm->data[m * k], A->m);
    float *QmK = get_column(handle, Qm, k + 1);
    vect_prod_mat(handle, Vm, QmK, tmp_fm2);
    vect_add(fm, tmp_fm2, tmp_fm, A->m);
    float *Vk = get_column(handle, Vm, k);

    // Recreate value Vm, TM, fm
    matrix_delete(Vm);
    matrix_delete(Tm);
    free(fm);

    Vm = matrix_zeros(A->n, m);
    Tm = matrix_zeros(m, m);
    fm = malloc(sizeof(float) * A->m);
    lanczos_facto(handle, A, Vk, k, m, Vm, Tm, fm);
    // delete all temporary variables
    free(QmK);
    matrix_delete(Qm);
    matrix_delete(q);
    free(Vk);
    free(tmp_fm);
    free(tmp_fm2);

    nb_iter++;
  }

  matrix_delete(Vm);
  q = matrix_zeros(Tm->m, Tm->n);
  float *eigs = qr_alg_eigen(Tm, q);
  matrix_delete(Tm);
  free(fm);

  return eigs;
}
void eigen_values(cublasHandle_t handle, cusolverDnHandle_t cusolverH, Mat *A) {

    int nb_iter = 0;
    float *v = calloc(A->n, sizeof(float));
    for (int i = 0; i < A->n - 1; i++)
      v[i] = 1;
    float vNorm =  vect_norm(handle, v, A->n);
    for (int i = 0; i < A->n; i++) {
        v[i] /= vNorm;
    }

    float *res = lanczos_ir(handle, cusolverH, A, v, K, M);
    printf("##### Eigen Values: #####\n");
    qsort(res, M, sizeof(*res), my_compare);
    for (int i = 0; i < K; i++)
      printf("%8.5f\n", res[i]);
    free(res);
    free(v);
}
void init_random_matrix_sym(Mat *A) {
  float tmp;
  srand(10);
  for (int i = 0; i < A->m; i++) {
    for (int j = 0; j < A->n; j++) {
      tmp = (float) (rand() % 100);
      A->data[A->m * i + j] = tmp;
      A->data[A->n * j + i] = tmp;
    }
  }
}

int main(void) {

  int tmp;
  Mat *A = matrix_new(N, N);
  cublasHandle_t handle = NULL;
  cublasStatus_t cublas_status = cublasCreate(&handle);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);
  cusolverDnHandle_t cusolverH = NULL;
  cublasStatus_t cusolver_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

  init_random_matrix_sym(A);
  matrix_print(A);
  puts("");
  clock_t t = clock();
  eigen_values(handle, cusolverH, A);
  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC;
  
  printf("Compute EigenValues took %.4f seconds to execute \n", time_taken);
  matrix_delete(A);

  if (handle ) cublasDestroy(handle);
  if (cusolverH) cusolverDnDestroy(cusolverH);
  cudaDeviceReset();
  return 0;
}
