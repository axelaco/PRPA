
#include "math_cuda_v2.h"

#define BLOCK_SIZE 256
#define IDX2C(i,j,ld) (((j)*(ld))+(i))



Mat *matrix_new(int m, int n) {
  Mat *res = malloc(sizeof(Mat));
  if (!res)
    return NULL;
  cudaError_t cudaStat;
  cudaStat = cudaMalloc((void**)&res->d_data, m * n * sizeof(float));
  assert(cudaSuccess == cudaStat);
  res->m = m;
  res->n = n;
  return res;
}
/*
float *matrix_diag(Mat *A) {
  float *res = malloc(sizeof(float) * A->m);
  if (!res)
    return NULL;
  res[0] = A->data[0];
  int i = (A->m + 1);
  int j = 1;
  for (;i < A->m * A->n; i += (A->m + 1)) {
    res[j] = A->data[i];
    j++;
  }
  return res;
}
// NAIVE
float *matrix_off_diag(Mat *A) {
  float *res = malloc(sizeof(float) * A->n - 1);
  int j = 0;
  for (int i = 1; i < A->m * A->n; i += (A->m + 1)) {
    res[j] = A->data[i];
    j++;
  }
  return res;
}

float *qr_alg_eigen(Mat *A, Mat *eigVector) {
  float *diag = matrix_diag(A);
  float *off_diag = matrix_off_diag(A);
  float *WORK = malloc(sizeof(float) * 2 * A->n - 2);
  int n = A->n;
  int info;
  LAPACK_ssteqr("I", &n, diag, off_diag, eigVector->data, &n, WORK, &info);
  return diag;
}
*/
void qr(cublasHandle_t handle, cusolverDnHandle_t cusolverH, Mat *A, Mat *R, Mat *Q) {
  if (!A || !R || !Q)
    return;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  cudaError_t cudaStat4 = cudaSuccess;
  const int m = A->m;
  const int n = A->n;
  const int lda = m;
  float *d_tmp_A = NULL;
  float *d_tau = NULL;
  int *devInfo = NULL;
  float *d_work = NULL;
  int lwork_geqrf = 0;
  int lwork_orgqr = 0;
  int lwork = 0;

  int info_gpu = 0;

// step 2: copy A and B to device
  cudaStat1 = cudaMalloc((void**)&d_tmp_A, sizeof(float) * m  * n);
  cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(float)*n);
  cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));

  cudaMemcpy(d_tmp_A, A->d_data, sizeof(float) * m * n, cudaMemcpyDeviceToDevice);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);

// step 3: query working space of geqrf and orgqr
  cusolver_status = cusolverDnSgeqrf_bufferSize(cusolverH, m, n, d_tmp_A,
                lda, &lwork_geqrf);
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
  cusolver_status = cusolverDnSorgqr_bufferSize(cusolverH, m, n, n, d_tmp_A,
                lda, d_tau, &lwork_orgqr);
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
// lwork = max(lwork_geqrf, lwork_orgqr)
  lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
  cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
  assert(cudaSuccess == cudaStat1);

// step 4: compute QR factorization
  cusolver_status = cusolverDnSgeqrf(cusolverH, m, n, d_tmp_A, lda, d_tau,
                d_work, lwork, devInfo);
  cudaStat1 = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  assert(cudaSuccess == cudaStat1);

// check if QR is successful or not
  cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat1);

  assert(0 == info_gpu);

  triu(d_tmp_A, R->d_data, A->m, A->m, A->n);
  cudaStat1 = cudaDeviceSynchronize();
  assert(cudaSuccess == cudaStat1);
// step 5: compute Q
  cusolver_status= cusolverDnSorgqr(cusolverH, m, n, n, d_tmp_A, lda, d_tau,
                d_work, lwork, devInfo);
  cudaStat1 = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  assert(cudaSuccess == cudaStat1);
            // check if QR is good or not
  cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat1);

  assert(0 == info_gpu);

  cudaStat1 = cudaMemcpy(Q->d_data, d_tmp_A, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice);
  assert(cudaSuccess == cudaStat1);
// free resources
  if (d_tmp_A) cudaFree(d_tmp_A);
  if (d_tau  ) cudaFree(d_tau);
  if (devInfo) cudaFree(devInfo);
  if (d_work ) cudaFree(d_work);
}
Mat *matrix_eye(int m, int n) {
  Mat *A = matrix_new(m, n);
  float *h_data = malloc(sizeof(float) * m * n);

  for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
          if (i == j)
            h_data[IDX2C(i,j,m)] = 1;
          else
            h_data[IDX2C(i,j,m)] = 0;
      }
  }
  cudaMemcpy(A->d_data, h_data, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  free(h_data);
  return A;
}
Mat *matrix_zeros(int m, int n) {
  Mat *res = matrix_new(m, n);
  float *h_data = malloc(sizeof(float) * m * n);
  for (int i = 0; i < m * n; i++)
    h_data[i] = 0;
  cudaMemcpy(res->d_data, h_data, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  free(h_data);
  return res;
}
Mat *matrix_mul(cublasHandle_t handle, Mat *A, Mat *B) {
  Mat *C = matrix_new(A->m, B->n);
  cublasStatus_t stat;
  cudaError_t cudaStat;
  float alpha = 1;
  float beta = 0;

  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A->m, B->n, A->n, &alpha,
  A->d_data, A->m, B->d_data, B->m, &beta, C->d_data, A->m);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("matrix_mul: %d CUBLAS multiplication failed\n", stat);
      return NULL;
  }

  return C;
}
void matrix_mul_bis(cublasHandle_t handle, Mat *res, Mat *A, Mat *B) {
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  const int m = A->m;
  const int lda = B->m;
  const int ldb = B->m;
  float alpha = 1;
  float beta = 0;
  cublas_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, A->n, A->n, &alpha,
     A->d_data, lda, B->d_data, ldb, &beta, res->d_data, lda);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

}
void matrix_sub(Mat *A, Mat *B, Mat *res) {

  vecSub(A->d_data, B->d_data, res->d_data, BLOCK_SIZE, A->m * A->n);
  cudaThreadSynchronize();
}
void matrix_add(Mat *A, Mat *B, Mat *res) {
  vecAdd(A->d_data, B->d_data, res->d_data, BLOCK_SIZE, A->m * A->n);
  cudaThreadSynchronize();
}

Mat *matrix_transpose(cublasHandle_t handle, Mat *A) {
  Mat *res = matrix_new(A->n, A->m);

  float alpha = 1;
  float beta = 0;

  cublasStatus_t cudaStat = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, A->m,
  A->n, &alpha, A->d_data, A->n, &beta, A->d_data, A->n, res->d_data, A->m);

  return res;
}
void matrix_scalar(Mat *A, float scalar) {
  vecScalar(A->d_data, scalar, BLOCK_SIZE, A->m * A->n);
  cudaThreadSynchronize();
}
void matrix_delete(Mat *mat) {
  if (!mat)
    return;
  cudaFree(mat->d_data);
  free(mat);
}
void vect_copy(float *d_src, float *d_dest, int m) {
  vecCopy(d_src, d_dest, BLOCK_SIZE, m);
  cudaThreadSynchronize();
}

void vect_prod_mat(cublasHandle_t handle, Mat *A, float *d_u, float *d_res) {
  float alpha = 1;
  float beta = 0;
  cublasStatus_t cudaStat = cublasSgemv(handle, CUBLAS_OP_N, A->m, A->n, &alpha,
    A->d_data, A->m, d_u, 1, &beta, d_res, 1);
  if (cudaStat != cudaSuccess) {
    printf("vect_prod_mat: Error in cblas_sgemv function\n");
  }
}

float vect_dot(cublasHandle_t handle, float *d_u, float *d_v, int n) {
  float res;
  cudaError_t cudaStat = cudaSuccess;
  cudaStat = cublasSdot(handle, n, d_u, 1, d_v, 1, &res);
  if (cudaStat != cudaSuccess) {
    printf ("vect_dot: %d Error in cublasSdot function\n", cudaStat);
    return -1;
  }
  return res;
}

void vect_scalar(float *d_u, float scalar, int n) {
  vecScalar(d_u, scalar, BLOCK_SIZE, n);
  cudaThreadSynchronize();
}

void vect_substract(float *d_res, float *d_u, float *d_v, int m) {
  vecSub(d_u, d_v, d_res, BLOCK_SIZE, m);
  cudaDeviceSynchronize();
}

float vect_norm(cublasHandle_t handle, float *d_u, int n) {
  float res;
  cublasStatus_t cudaStat =  cublasSnrm2(handle, n, d_u, 1, &res);
  if (cudaStat != cudaSuccess) {
    printf("vect_norm: Error in cublasSnrm2 function\n");
  }
  return res;
}

float *vect_divide_by_scalar(float *d_u, float scalar, int n) {
  float *d_res = NULL;
  cudaMalloc((void**)&d_res, sizeof(float) * n);
  vecScalar(d_u, 1 / scalar, BLOCK_SIZE, n);
  cudaDeviceSynchronize();
  return d_res;
}
void vect_add(float *d_res, float *d_u, float *d_v, int m) {
  vecAdd(d_u, d_v, d_res, BLOCK_SIZE, m);
  cudaThreadSynchronize();
}


void vect_mat_copy(cublasHandle_t handle, Mat *A, float *d_u, int col) {
  cublasStatus_t cublaStat = cublasScopy(handle, A->m, d_u, 1,
    A->d_data + (A->m * col), 1);
}
float *get_column(cublasHandle_t handle, Mat *A, int col) {
  float *d_res;

  cudaError_t cudaError = cudaMalloc((void**)&d_res, sizeof(float) * A->m);
  if (cudaError != cudaSuccess) {
      printf ("get_column: %d Error in allocation of gpu_mat_A\n", cudaError);
  }

  cublasStatus_t cublaStat = cublasScopy(handle, A->m, A->d_data + (A->m * col), 1,
    d_res, 1);

  return d_res;
}

void vect_print(float *d_u, int n) {
  float *h_u = malloc(sizeof(float) * n);
  cudaMemcpy(h_u, d_u, sizeof(float) * n, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    printf("%8.5f\n", h_u[i]);
  }
  free(h_u);
}
void matrix_print(Mat *A) {
  float *h_data = malloc(sizeof(float) * A->m * A->n);
  cudaMemcpy(h_data, A->d_data, sizeof(float) * A->m * A->n, cudaMemcpyDeviceToHost);

  for (int j = 0; j < A->n; j++) {
       for (int i = 0; i < A->m; i++) {
           printf ("%8.5f ", h_data[IDX2C(i,j,A->m)]);
       }
       printf ("\n");
   }
   free(h_data);
}
