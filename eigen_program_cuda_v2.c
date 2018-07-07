#include "math_cuda_v2.h"
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

void lanczos_facto(cublasHandle_t handle, Mat *A, float *d_v0, int k, int m, Mat *Vm, Mat *Tm, float *d_fm) {
  float b1 = 0;
  float *d_wPrime = NULL;
  float *d_v = NULL;
  cudaMalloc((void**)&d_wPrime, sizeof(float) * A->m);
  cudaMalloc((void**)&d_v, sizeof(float) * A->m);

  vect_copy(d_v0, d_v, A->m);
  for (int j = k - 1; j < m - 1; j++) {
    // wj' = A * vj
    vect_prod_mat(handle, A, d_v, d_wPrime);

    // Vm(:,j) = vj
    vect_mat_copy(handle, Vm, d_v, j);

    //T(j,j) = wj'.T * vj
    float a1 = vect_dot(handle, d_wPrime, d_v, A->m);
    update_value(Tm->d_data, (Tm->m * j) + j, a1);
    cudaDeviceSynchronize();
    // wj = wj' - T(j,j)*vj - Bj*vj-1
    vect_scalar(d_v, a1, A->m);
    if (j > 0) {
      float* d_vjOld = get_column(handle, Vm, j - 1);
      vect_scalar(d_vjOld, b1, A->m);
      float *d_tmp = NULL; //malloc(sizeof(float) * A->m);
      cudaMalloc((void**)&d_tmp, sizeof(float) * A->m);
      vect_substract(d_tmp, d_wPrime, d_v, A->m);
      vect_substract(d_fm, d_tmp, d_vjOld, A->m);
      cudaFree(d_tmp);
      cudaFree(d_vjOld);
    } else {
      vect_substract(d_fm, d_wPrime, d_v, A->m);
    }
    // Bj = ||wj||
    b1 = vect_norm(handle, d_fm, A->m);
    update_value(Tm->d_data, (Tm->m * j) + j + 1, b1);
    cudaDeviceSynchronize();
    update_value(Tm->d_data, Tm->m * (j + 1) + j, b1);
    cudaDeviceSynchronize();

    cudaFree(d_v);
    d_v = vect_divide_by_scalar(d_fm, b1, A->m);
  }
  vect_mat_copy(handle, Vm, d_v, m - 1);
  vect_prod_mat(handle, A, d_v, d_fm);
  update_value(Tm->d_data, m * m - 1, vect_dot(handle, d_fm, d_v, A->m));
  cudaDeviceSynchronize();
  cudaFree(d_v);
  cudaFree(d_wPrime);
}

float *lanczos_ir(cublasHandle_t handle, cusolverDnHandle_t cusolverH,
  Mat *A, float *d_v0, int k, int m) {
  Mat *Vm = matrix_zeros(A->n, m);
  Mat *Tm = matrix_zeros(m, m);
  Mat *q = NULL;
  float residual = 1;
  float *d_fm = NULL;
  cudaMalloc((void**)&d_fm, sizeof(float) * A->m);
  Mat *Qm = NULL;
  Mat *QT = NULL;
  Mat *QmT = NULL;
  float eps = 0.00001;
  lanczos_facto(handle, A, d_v0, 1, m, Vm, Tm, d_fm);
  int nb_iter = 0;
  while(residual > eps && nb_iter < 100) {
    q = matrix_zeros(Tm->m, Tm->n);
    float *d_eigs = qr_alg_eigen(Tm, q);//rritz(Tm, &residual, fm, k, residual);
    // qsort(d_eigs, Tm->m, sizeof(*eigs), cmp_abs);
    Qm = matrix_eye(m, m);
    for (int j = m - k; j >= 0; j--) {
      Mat *mat_tmp = matrix_new(Tm->m, Tm->n);
      Mat *eye = matrix_eye(Tm->n, Tm->n);
      Mat *R = matrix_zeros(Tm->m, Tm->n);
      Mat *Q = matrix_zeros(Tm->m, Tm->n);
      // s * I(:,:)
      float eig_value = get_value(d_eigs, j);
      cudaDeviceSynchronize();
      matrix_scalar(eye, eig_value);
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
    cudaFree(d_eigs);
    float *d_tmp_fm = NULL;
    float *d_tmp_fm2 = NULL;

    cudaMalloc((void**)&d_tmp_fm, sizeof(float) * A->m);
    cudaMalloc((void**)&d_tmp_fm2, sizeof(float) * A->m);

    vect_copy(d_fm, d_tmp_fm, A->m);
    float QmData = get_value(Qm->d_data, m * k);
    cudaDeviceSynchronize();
    vect_scalar(d_tmp_fm, QmData, A->m);
    float *d_QmK = get_column(handle, Qm, k + 1);
    vect_prod_mat(handle, Vm, d_QmK, d_tmp_fm2);
    vect_add(d_fm, d_tmp_fm2, d_tmp_fm, A->m);
    float *d_Vk = get_column(handle, Vm, k);

    // Recreate value Vm, TM, fm
    matrix_delete(Vm);
    matrix_delete(Tm);
    cudaFree(d_fm);

    Vm = matrix_zeros(A->n, m);
    Tm = matrix_zeros(m, m);
    d_fm = NULL;
    cudaMalloc((void**)&d_fm, sizeof(float) * A->m);
    lanczos_facto(handle, A, d_Vk, k, m, Vm, Tm, d_fm);
    // delete all temporary variables
    cudaFree(d_QmK);
    matrix_delete(Qm);
    matrix_delete(q);
    cudaFree(d_Vk);
    cudaFree(d_tmp_fm);
    cudaFree(d_tmp_fm2);

    nb_iter++;
  }

  matrix_delete(Vm);
  q = matrix_zeros(Tm->m, Tm->n);
  float *d_eigs = qr_alg_eigen(Tm, q);

  matrix_delete(q);
  matrix_delete(Tm);
  cudaFree(d_fm);

  return d_eigs;
  //return NULL;
}
void eigen_values(cublasHandle_t handle, cusolverDnHandle_t cusolverH, Mat *A) {

    int nb_iter = 0;
    float *d_v = NULL;
    float *h_v = calloc(A->n, sizeof(float));
    cudaMalloc((void**)&d_v, sizeof(float) * A->n);
    for (int i = 0; i < A->n - 1; i++)
      h_v[i] = 1;
    cudaMemcpy(d_v, h_v, sizeof(float)  * (A->n - 1), cudaMemcpyHostToDevice);
    float vNorm =  vect_norm(handle, d_v, A->n);
    for (int i = 0; i < A->n; i++) {
        h_v[i] /= vNorm;
    }

    cudaMemcpy(d_v, h_v, sizeof(float) * A->n, cudaMemcpyHostToDevice);
/*
    Mat *Vm = matrix_zeros(A->n, M);
    Mat *Tm = matrix_zeros(M, M);
    float *d_fm = NULL;
    cudaMalloc((void**) &d_fm, sizeof(float) * A->m);
    vect_print(d_v, A->n);
    lanczos_facto(handle, A, d_v, 1, M, Vm, Tm, d_fm);
    puts("VM:");
    matrix_print(Vm);
    puts("Tm:");
    matrix_print(Tm);
*/
    float *d_res = lanczos_ir(handle, cusolverH, A, d_v, K, M);
    printf("##### Eigen Values: #####\n");
    /*qsort(res, M, sizeof(*res), my_compare);
    for (int i = 0; i < K; i++)
      printf("%8.5f\n", res[i]);*/
    vect_print(d_res, K);
    cudaFree(d_res);
    cudaFree(d_v);
}
void init_random_matrix_sym(Mat *A) {
  float tmp;
  srand(10);
  float *h_data = malloc(sizeof(float) * A->m * A->n);
  if (!h_data)
    return;
  for (int i = 0; i < A->m; i++) {
    for (int j = 0; j < A->n; j++) {
      tmp = (float) (rand() % 100);
      h_data[A->m * i + j] = tmp;
      h_data[A->n * j + i] = tmp;
    }
  }
  cudaMemcpy(A->d_data, h_data, sizeof(float) * A->m * A->n, cudaMemcpyHostToDevice);
  free(h_data);
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
  printf("ComputeImage took %.4f seconds to execute \n", time_taken);
  matrix_delete(A);
  //if (handle ) cublasDestroy(handle);

  if (handle ) cublasDestroy(handle);
  //if (cusolverH) cusolverDnDestroy(cusolverH
  cudaDeviceReset();
  return 0;
}
