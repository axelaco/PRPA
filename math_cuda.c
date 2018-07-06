
#include "math_cuda.h"

#define BLOCK_SIZE 256
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static int cmp_abs (void const *a, void const *b)
{
   /* definir des pointeurs type's et initialise's
      avec les parametres */
   float const *pa = a;
   float const *pb = b;

   /* evaluer et retourner l'etat de l'evaluation (tri croissant) */
   return abs(*pa) - abs(*pb);
}


Mat *matrix_new(int m, int n) {
  Mat *res = malloc(sizeof(Mat));
  if (!res)
    return NULL;
  res->data = malloc(sizeof(float) * m * n);
  if (!res->data) {
    free(res);
    return NULL;
  }
  res->m = m;
  res->n = n;
  return res;
}

float *qr_alg_eigen(cusolverDnHandle_t cusolverH, Mat *A) {
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  const int m = A->m;
  const int lda = m;
  float V[lda*m]; // eigenvectors
  float *W = malloc(sizeof(float) * m); // eigenvalues
  float *d_A = NULL;
  float *d_W = NULL;
  int *devInfo = NULL;
  float *d_work = NULL;
  int  lwork = 0;

  int info_gpu = 0;

  // step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

  // step 2: copy A and W to device
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(float) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_W, sizeof(float) * m);
    cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    cudaStat1 = cudaMemcpy(d_A, A->data, sizeof(float) * lda * m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
  // step 3: query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnSsyevd_bufferSize(cusolverH, jobz, uplo, m, d_A,
            lda, d_W, &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    assert(cudaSuccess == cudaStat1);
  // step 4: compute spectrum
    cusolver_status = cusolverDnSsyevd(cusolverH, jobz, uplo, m, d_A, lda,
            d_W, d_work, lwork, devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(W, d_W, sizeof(float)*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(V, d_A, sizeof(float)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    printf("after syevd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

    // free resources
    if (d_A    ) cudaFree(d_A);
    if (d_W    ) cudaFree(d_W);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    qsort(W, A->m, sizeof(*W), cmp_abs);
    return W;

}

float *create_gpu_matrix(float *cpu_data, int m, int n) {
  cudaError_t cudaStat;
  cublasStatus_t stat;
  float *gpu_mat_A;
  // Create Matrix A on GPU
  cudaStat = cudaMalloc((void**)&gpu_mat_A, m * n * sizeof(float));
  if (cudaStat != cudaSuccess) {
      printf ("%d device memory allocation failed for cpu_data", cudaStat);
      return NULL;
  }
  stat = cudaMemcpy(gpu_mat_A, cpu_data, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed for Matrix");
      cudaFree (gpu_mat_A);
      return NULL;
  }
  return gpu_mat_A;
}
float *create_gpu_vector(float *cpu_data, int n) {
  cudaError_t cudaStat;
  cublasStatus_t stat;
  float *gpu_vect_A;
  // Create Matrix A on GPU
  cudaStat = cudaMalloc((void**)&gpu_vect_A, n * sizeof(float));
  if (cudaStat != cudaSuccess) {
      printf ("%d device memory allocation failed for cpu_data", cudaStat);
      return NULL;
  }
  stat = cublasSetVector(n, sizeof(*cpu_data),
  cpu_data, 1, gpu_vect_A, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed for Matrix");
      cudaFree (gpu_vect_A);
      return NULL;
  }
  return gpu_vect_A;
}
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
        float *d_A = NULL;
        float *d_tau = NULL;
        int *devInfo = NULL;
        float *d_work = NULL;

        float *d_R = NULL;

        int lwork_geqrf = 0;
        int lwork_orgqr = 0;
        int lwork = 0;

        int info_gpu = 0;


    // step 2: copy A and B to device
        cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(float)*lda*n);
        cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(float)*n);
        cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
        cudaStat4 = cudaMalloc ((void**)&d_R  , sizeof(float)*n*n);
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
        assert(cudaSuccess == cudaStat3);
        assert(cudaSuccess == cudaStat4);

        cudaStat1 = cudaMemcpy(d_A, A->data, sizeof(float)*lda*n, cudaMemcpyHostToDevice);
        assert(cudaSuccess == cudaStat1);
        // step 3: query working space of geqrf and orgqr
            cusolver_status = cusolverDnSgeqrf_bufferSize(
                cusolverH,
                m,
                n,
                d_A,
                lda,
                &lwork_geqrf);
            assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
            cusolver_status = cusolverDnSorgqr_bufferSize(
                cusolverH,
                m,
                n,
                n,
                d_A,
                lda,
                d_tau,
                &lwork_orgqr);
            assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
        // lwork = max(lwork_geqrf, lwork_orgqr)
            lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;

            cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
            assert(cudaSuccess == cudaStat1);

        // step 4: compute QR factorization
            cusolver_status = cusolverDnSgeqrf(
                cusolverH,
                m,
                n,
                d_A,
                lda,
                d_tau,
                d_work,
                lwork,
                devInfo);
            cudaStat1 = cudaDeviceSynchronize();
            assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
            assert(cudaSuccess == cudaStat1);

            // check if QR is successful or not
            cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
            assert(cudaSuccess == cudaStat1);

            printf("after geqrf: info_gpu = %d\n", info_gpu);
            assert(0 == info_gpu);
            cudaError_t cudaError = cudaMalloc((void**)&d_R, sizeof(float) *  R->m * R->n);
            assert(cudaSuccess == cudaError);
            triu(d_A, d_R, A->m, A->m, A->n);
            cudaStat1 = cudaDeviceSynchronize();
            assert(cudaSuccess == cudaStat1);
            cudaStat1 = cudaMemcpy(R->data, d_R, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
            assert(cudaSuccess == cudaStat1);
        // step 5: compute Q
            cusolver_status= cusolverDnSorgqr(
                cusolverH,
                m,
                n,
                n,
                d_A,
                lda,
                d_tau,
                d_work,
                lwork,
                devInfo);
            cudaStat1 = cudaDeviceSynchronize();
            assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
            assert(cudaSuccess == cudaStat1);
            // check if QR is good or not
            cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
            assert(cudaSuccess == cudaStat1);

            printf("after orgqr: info_gpu = %d\n", info_gpu);
            assert(0 == info_gpu);

            cudaStat1 = cudaMemcpy(Q->data, d_A, sizeof(float)*lda*n, cudaMemcpyDeviceToHost);
            assert(cudaSuccess == cudaStat1);



            // free resources
        if (d_A    ) cudaFree(d_A);
        if (d_tau  ) cudaFree(d_tau);
        if (devInfo) cudaFree(devInfo);
        if (d_work ) cudaFree(d_work);
        if (d_R    ) cudaFree(d_R);

        if (handle ) cublasDestroy(handle);
        if (cusolverH) cusolverDnDestroy(cusolverH);

        cudaDeviceReset();
}
Mat *matrix_eye(int m, int n) {
  Mat *A = matrix_new(m, n);
  for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
          if (i == j)
            A->data[IDX2C(i,j,m)] = 1;
          else
            A->data[IDX2C(i,j,m)] = 0;
      }
  }
  return A;
}
Mat *matrix_zeros(int m, int n) {
  Mat *res = matrix_new(m, n);
  for (int i = 0;i < m * n; i++)
    res->data[i] = 0;
  return res;
}
Mat *matrix_mul(cublasHandle_t handle, Mat *A, Mat *B) {
  float *gpu_mat_A;
  float *gpu_mat_B;
  float *gpu_mat_C;
  Mat *C = matrix_new(A->m, B->n);
  cublasStatus_t stat;
  cudaError_t cudaStat;
  gpu_mat_A = create_gpu_matrix(A->data, A->m, A->n);
  if (!gpu_mat_A) {
    printf("matrix_mul: GPU MAT A CREATION ERROR\n");
    return  NULL;
  }
  gpu_mat_B = create_gpu_matrix(B->data, B->m, B->n);
  if (!gpu_mat_B) {
    printf("matrix_mul: GPU MAT B CREATION ERROR\n");
    cudaFree(gpu_mat_A);
    return NULL;
  }
  float alpha = 1;
  float beta = 0;
  cudaStat = cudaMalloc((void**)&gpu_mat_C, C->m * C->n * sizeof(float));
  if (cudaStat != cudaSuccess) {
      printf ("matrix_mul: %d device memory allocation failed for cpu_data", cudaStat);
      return NULL;
  }
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A->m, B->n, A->n, &alpha,
  gpu_mat_A, A->m, gpu_mat_B, B->m, &beta, gpu_mat_C, A->m);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("matrix_mul: CUBLAS multiplication failed\n");
      return NULL;
  }
  stat = cublasGetMatrix(C->m, C->n, sizeof(*C->data), gpu_mat_C, C->m, C->data, C->m);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("matrix_mul: %d data upload failed", stat);
      if (stat == CUBLAS_STATUS_INVALID_VALUE)
        printf("CUBLAS_STATUS_INVALID_VALUE\n");
      if (stat == CUBLAS_STATUS_MAPPING_ERROR)
        printf("CUBLAS_STATUS_MAPPING_ERROR\n");
      if (stat == CUBLAS_STATUS_NOT_INITIALIZED)
        printf("CUBLAS_STATUS_NOT_INITIALIZED\n");
      cudaFree (gpu_mat_A);
      cudaFree (gpu_mat_B);
      cudaFree (gpu_mat_C);
      return NULL;
  }
  cudaFree (gpu_mat_A);
  cudaFree (gpu_mat_B);
  cudaFree (gpu_mat_C);
  return C;
}
void matrix_mul_bis(cublasHandle_t handle, Mat *res, Mat *A, Mat *B) {
  float *gpu_mat_A;
  float *gpu_mat_B;
  float *gpu_mat_C;
  cublasStatus_t stat;
  cudaError_t cudaStat;
  gpu_mat_A = create_gpu_matrix(A->data, A->m, A->n);
  if (!gpu_mat_A) {
    printf("matrix_mul_bis: GPU MAT A CREATION ERROR\n");
    return;
  }
  gpu_mat_B = create_gpu_matrix(B->data, B->m, B->n);
  if (!gpu_mat_B) {
    printf("matrix_mul_bis: GPU MAT B CREATION ERROR\n");
    return;
  }
  float alpha = 1;
  float beta = 0;
  cudaStat = cudaMalloc((void**)&gpu_mat_C, res->m * res->n * sizeof(float));
  if (cudaStat != cudaSuccess) {
      printf ("matrix_mul_bis: %d device memory allocation failed for cpu_data", cudaStat);
      return;
  }
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A->m, B->n, A->n, &alpha,
  gpu_mat_A, A->m, gpu_mat_B, B->m, &beta, gpu_mat_C, A->m);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("matrix_mul_bis: CUBLAS multiplication failed\n");
      return;
  }
  stat = cublasGetMatrix(res->m, res->n, sizeof(*res->data), gpu_mat_C, res->m, res->data, res->m);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("%d data upload failed", stat);
      if (stat == CUBLAS_STATUS_INVALID_VALUE)
        printf("CUBLAS_STATUS_INVALID_VALUE\n");
      if (stat == CUBLAS_STATUS_MAPPING_ERROR)
        printf("CUBLAS_STATUS_MAPPING_ERROR\n");
      if (stat == CUBLAS_STATUS_NOT_INITIALIZED)
        printf("CUBLAS_STATUS_NOT_INITIALIZED\n");
      cudaFree (gpu_mat_A);
      cudaFree (gpu_mat_B);
      cudaFree (gpu_mat_C);
      return;
  }
  cudaFree (gpu_mat_A);
  cudaFree (gpu_mat_B);
  cudaFree (gpu_mat_C);
}
void matrix_sub(Mat *A, Mat *B, Mat *res) {
  float *gpu_mat_A;
  float *gpu_mat_B;
  float *gpu_mat_res;
  gpu_mat_A = create_gpu_matrix(A->data, A->m, A->n);
  if (!gpu_mat_A) {
    printf("Matrix Sub: Error in gpu_mat_A creation\n");
  }
  gpu_mat_B = create_gpu_matrix(B->data, B->m, B->n);
  if (!gpu_mat_B) {
    printf("Matrix Sub: Error in gpu_mat_B creation\n");
  }
  cublasStatus_t cudaStat = cudaMalloc((void**)&gpu_mat_res, res->m * res->n * sizeof(float));
  if (cudaStat != cudaSuccess) {
      printf ("%d device memory allocation failed for cpu_data", cudaStat);
  }

  vecSub(gpu_mat_A, gpu_mat_B, gpu_mat_res, BLOCK_SIZE, A->m * A->n);
  cudaThreadSynchronize();
  cudaStat = cublasGetMatrix(res->m, res->n, sizeof(*res->data), gpu_mat_res, res->m, res->data, res->m);
  if (cudaStat != cudaSuccess) {
      printf ("%d device memory upload failed for res", cudaStat);
  }

  cudaFree(gpu_mat_A);
  cudaFree(gpu_mat_B);
  cudaFree(gpu_mat_res);

}
void matrix_add(Mat *A, Mat *B, Mat *res) {
  float *gpu_mat_A;
  float *gpu_mat_B;
  float *gpu_mat_res;
  gpu_mat_A = create_gpu_matrix(A->data, A->m, A->n);
  if (!gpu_mat_A) {
    printf("Matrix Sub: Error in gpu_mat_A creation\n");
  }
  gpu_mat_B = create_gpu_matrix(B->data, B->m, B->n);
  if (!gpu_mat_B) {
    printf("Matrix Sub: Error in gpu_mat_B creation\n");
  }
  cudaError_t cudaError = cudaMalloc((void**)&gpu_mat_res, res->m * res->n * sizeof(float));
  if (cudaError != cudaSuccess) {
      printf ("%d device memory allocation failed for cpu_data", cudaError);
  }

  vecAdd(gpu_mat_A, gpu_mat_B, gpu_mat_res, BLOCK_SIZE, A->m * A->n);
  cudaThreadSynchronize();
  cublasStatus_t cudaStat = cublasGetMatrix(res->m, res->n, sizeof(*res->data), gpu_mat_res, res->m, res->data, res->m);
  if (cudaStat != cudaSuccess) {
      printf ("%d device memory upload failed for res", cudaStat);
  }

  cudaFree(gpu_mat_A);
  cudaFree(gpu_mat_B);
  cudaFree(gpu_mat_res);

}
Mat *matrix_transpose(cublasHandle_t handle, Mat *A) {
  float *gpu_mat_A;
  float *gpu_mat_res;
  Mat *res = matrix_new(A->n, A->m);
  gpu_mat_A = create_gpu_matrix(A->data, A->m, A->n);
  if (!gpu_mat_A) {
    printf("matrix_transpose: Error in allocation of gpu_mat_A\n");
  }
  float alpha = 1;
  float beta = 0;
  cudaError_t cudaError = cudaMalloc((void**)&gpu_mat_res, sizeof(float) * A->m * A->n);
  if (cudaError != cudaSuccess) {
    printf("matrix_transpose: Error in allocation of gpu_mat_res\n");
  }
  cublasStatus_t cudaStat = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, A->m,
  A->n, &alpha, gpu_mat_A, A->n, &beta, gpu_mat_A, A->n, gpu_mat_res, A->m);

  cudaStat = cublasGetMatrix(res->m, res->n, sizeof(*A->data), gpu_mat_res, res->m, res->data, res->m);
  if (cudaStat != cudaSuccess) {
      printf ("matrix_transpose: %d device memory upload failed for res", cudaStat);
  }
  cudaFree(gpu_mat_A);
  cudaFree(gpu_mat_res);
  return res;
}
void matrix_scalar(Mat *A, float scalar) {
  float *gpu_mat_A;
  gpu_mat_A = create_gpu_matrix(A->data, A->m, A->n);
  vecScalar(gpu_mat_A, scalar, BLOCK_SIZE, A->m * A->n);
  cudaThreadSynchronize();
  cublasStatus_t cudaStat = cublasGetMatrix(A->m, A->n, sizeof(*A->data), gpu_mat_A, A->m, A->data, A->m);
  if (cudaStat != cudaSuccess) {
      printf ("%d device memory upload failed for res", cudaStat);
  }

  cudaFree(gpu_mat_A);
}
void matrix_delete(Mat *mat) {
  if (!mat)
    return;
  free(mat->data);
  free(mat);
}
void vect_copy(float *src, float *dest, int m) {
  float *gpu_src;
  float *gpu_dest;
  gpu_src = create_gpu_vector(src, m);
  cublasStatus_t cudaStat = cudaMalloc((void **)&gpu_dest, m * sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf("vect_copy: Error in allocation of gpu_dest\n");
  }
  vecCopy(gpu_src, gpu_dest, BLOCK_SIZE, m);
  cudaThreadSynchronize();
  cudaError_t error = cudaMemcpy(dest, gpu_dest, m * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    printf ("vect_copy: %d device memory upload failed for res", error);
  }
  cudaFree(gpu_src);
  cudaFree(gpu_dest);
}

void vect_prod_mat(cublasHandle_t handle, Mat *A, float *u, float *res) {
  float *gpu_mat_A;
  float *gpu_vect_u;
  float *gpu_vect_res;
  gpu_mat_A = create_gpu_matrix(A->data, A->m, A->n);
  if (!gpu_mat_A)
  {
    printf("vect_prod_mat: Error in allocation of gpu_mat_A\n");
  }
  gpu_vect_u = create_gpu_vector(u, A->m);
  if (!gpu_vect_u)
  {
    cudaFree(gpu_mat_A);
    printf("vect_prod_mat: Error in allocation of gpu_vect_u\n");

  }
  cublasStatus_t cudaStat = cudaMalloc((void **)&gpu_vect_res, A->m * sizeof(float));
  if (cudaStat != cudaSuccess) {
    cudaFree(gpu_mat_A);
    cudaFree(gpu_vect_u);
    printf("vect_prod_mat: Error in allocation of gpu_dest\n");
  }
  float alpha = 1;
  float beta = 0;
  cudaStat = cublasSgemv(handle, CUBLAS_OP_N, A->m, A->n, &alpha,
    gpu_mat_A, A->m, gpu_vect_u, 1, &beta, gpu_vect_res, 1);
  if (cudaStat != cudaSuccess) {
    cudaFree(gpu_mat_A);
    cudaFree(gpu_vect_u);
    cudaFree(gpu_vect_res);
    printf("vect_prod_mat: Error in cblas_sgemv function\n");
  }
  cudaError_t cudaError = cudaMemcpy(res, gpu_vect_res, sizeof(float) * A->m, cudaMemcpyDeviceToHost);
  if (cudaError != cudaSuccess) {
    cudaFree(gpu_mat_A);
    cudaFree(gpu_vect_u);
    cudaFree(gpu_vect_res);
    printf ("vect_copy: %d device memory upload failed for res", cudaError);
  }
  cudaFree(gpu_mat_A);
  cudaFree(gpu_vect_u);
  cudaFree(gpu_vect_res);
}

float vect_dot(cublasHandle_t handle, float *u, float *v, int n) {
  float *gpu_vect_u;
  float *gpu_vect_v;
  float res;

  gpu_vect_u = create_gpu_vector(u, n);
  if (!gpu_vect_u) {
    printf("vect_dot: Error in allocation of gpu_vect_u\n");
    return -1;
  }
  gpu_vect_v = create_gpu_vector(v, n);
  if (!gpu_vect_v) {
    printf("vect_dot: Error in allocation of gpu_vect_v\n");
    cudaFree(gpu_vect_u);
    return -1;
  }
  cublasStatus_t cudaStat = cublasSdot(handle, n, gpu_vect_u, 1, gpu_vect_v, 1, &res);
  if (cudaStat != cudaSuccess) {
    printf ("vect_dot: %d Error in cublasSdot function\n", cudaStat);
    cudaFree(gpu_vect_u);
    cudaFree(gpu_vect_v);
    return -1;
  }

  if (cudaStat != cudaSuccess) {
    printf ("vect_dot: %d device memory upload failed for res", cudaStat);
    cudaFree(gpu_vect_u);
    cudaFree(gpu_vect_v);
    return -1;
  }
  cudaFree(gpu_vect_u);
  cudaFree(gpu_vect_v);

  return res;
}

void vect_scalar(float *u, float scalar, int n) {
  float *gpu_vect_u;
  gpu_vect_u = create_gpu_vector(u, n);
  if (!gpu_vect_u) {
    printf("vect_scalar: Error in allocation of gpu_vect_u\n");
    return;
  }
  vecScalar(gpu_vect_u, scalar, BLOCK_SIZE, n);
  cudaThreadSynchronize();

  cudaError_t cudaError = cudaMemcpy(u, gpu_vect_u, n * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaError != cudaSuccess) {
      printf ("vect_scalar: %d device memory upload failed for res\n", cudaError);
  }
  cudaFree(gpu_vect_u);
}

void vect_substract(float *res, float *u, float *v, int m) {
  float *gpu_vect_res;
  float *gpu_vect_u;
  float *gpu_vect_v;
  gpu_vect_u = create_gpu_vector(u, m);
  if (!gpu_vect_u) {
    printf("vect_substract: Error in allocation of gpu_vect_u\n");
    return;
  }
  gpu_vect_v = create_gpu_vector(v, m);
  if (!gpu_vect_v) {
    printf("vect_substract: Error in allocation of gpu_vect_v\n");
    cudaFree(gpu_vect_u);
    return;
  }

  cublasStatus_t cudaStat = cudaMalloc((void**)&gpu_vect_res, m * sizeof(float));
  if (cudaStat != cudaSuccess) {
      printf ("vect_substract: %d device memory allocation failed for cpu_data", cudaStat);
  }

  vecSub(gpu_vect_u, gpu_vect_v, gpu_vect_res, BLOCK_SIZE, m);
  cudaThreadSynchronize();
  cudaError_t error = cudaMemcpy(res, gpu_vect_res, sizeof(float) * m, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
      printf ("vect_substract: %d device memory upload failed for res", cudaStat);
  }
  cudaFree(gpu_vect_u);
  cudaFree(gpu_vect_v);
  cudaFree(gpu_vect_res);
}

float vect_norm(cublasHandle_t handle, float *u, int n) {
  float *gpu_vect_u = create_gpu_vector(u, n);
  float res;
  if (!gpu_vect_u) {
    printf("vect_substract: Error in allocation of gpu_vect_u\n");
  }


  cublasStatus_t cudaStat =  cublasSnrm2(handle, n, gpu_vect_u, 1, &res);
  if (cudaStat != cudaSuccess) {
    printf("vect_norm: Error in cublasSnrm2 function\n");
  }
  cudaFree(gpu_vect_u);
  return res;
}
float *vect_divide_by_scalar(float *u, float scalar, int n) {
  float *res = malloc(sizeof(float) * n);
  if (!res)
    return NULL;
  float *gpu_vect_u;
  gpu_vect_u = create_gpu_vector(u, n);
  if (!gpu_vect_u) {
    printf("vect_divide_by_scalar: Error in allocation of gpu_vect_u\n");
  }
  vecScalar(gpu_vect_u, 1 / scalar, BLOCK_SIZE, n);
  cudaThreadSynchronize();
  cudaError_t cudaError = cudaMemcpy(res, gpu_vect_u, sizeof(float) * n, cudaMemcpyDeviceToHost);
  if (cudaError != cudaSuccess) {
    printf ("vect_divide_by_scalar: %d device memory upload failed for res\n", cudaError);
  }
  cudaFree(gpu_vect_u);
  return res;
}
void vect_add(float *res, float *u, float *v, int m) {
  float *gpu_vect_res;
  float *gpu_vect_u;
  float *gpu_vect_v;
  gpu_vect_u = create_gpu_vector(u, m);
  if (!gpu_vect_u) {
    printf("vect_add: Error in allocation of gpu_vect_u\n");
    return;
  }
  gpu_vect_v = create_gpu_vector(v, m);
  if (!gpu_vect_v) {
    printf("vect_add: Error in allocation of gpu_vect_v\n");
    cudaFree(gpu_vect_u);
    return;
  }

  cublasStatus_t cudaStat = cudaMalloc((void**)&gpu_vect_res, m * sizeof(float));
  if (cudaStat != cudaSuccess) {
      cudaFree(gpu_vect_u);
      cudaFree(gpu_vect_v);
      printf ("vect_add: %d device memory allocation failed for cpu_data\n", cudaStat);
  }

  vecAdd(gpu_vect_u, gpu_vect_v, gpu_vect_res, BLOCK_SIZE, m);
  cudaThreadSynchronize();
  cudaError_t error = cudaMemcpy(res, gpu_vect_res, sizeof(float) * m, cudaMemcpyDeviceToHost);

  if (error != cudaSuccess) {
      printf ("vect_add: %d device memory upload failed for res\n", cudaStat);
  }
  cudaFree(gpu_vect_u);
  cudaFree(gpu_vect_v);
  cudaFree(gpu_vect_res);
}

// Check with Col Row Major

void vect_mat_copy(cublasHandle_t handle, Mat *A, float *u, int col) {
  float *gpu_mat_A;
  float *gpu_vect_u;
  gpu_mat_A = create_gpu_matrix(A->data, A->m, A->n);
  if (!gpu_mat_A) {
    printf("vect_mat_copy: Error in allocation of gpu_mat_A\n");
    return;
  }

  gpu_vect_u = create_gpu_vector(u, A->m);

  if (!gpu_vect_u) {
    printf("vect_mat_copy: Error in allocation of gpu_vect_u\n");
    return;
  }
  cublasStatus_t cublaStat = cublasScopy(handle, A->m, gpu_vect_u, 1,
    gpu_mat_A + (A->m * col), 1);

  cublaStat = cudaMemcpy(A->data, gpu_mat_A, sizeof(float) * A->m * A->n, cudaMemcpyDeviceToHost);
//  cublaStat = cublasGetMatrix(A->m, A->n, sizeof(*A->data), gpu_mat_A, A->m, A->data, A->m);
  if (cublaStat != cudaSuccess) {
      printf ("vect_mat_copy: %d device memory upload failed for res", cublaStat);
  }
  cudaFree(gpu_mat_A);
  cudaFree(gpu_vect_u);
}
float *get_column(cublasHandle_t handle, Mat *A, int col) {
  float *gpu_mat_A;
  float *gpu_vect_res;
  float *res = malloc(sizeof(float) * A->m);
  gpu_mat_A = create_gpu_matrix(A->data, A->m, A->n);
  if (!gpu_mat_A) {
    printf("get_column: Error in allocation of gpu_mat_A\n");
    return NULL;
  }
  cudaError_t cudaError = cudaMalloc((void**)&gpu_vect_res, sizeof(float) * A->m);
  if (cudaError != cudaSuccess) {
      printf ("get_column: %d Error in allocation of gpu_mat_A\n", cudaError);
  }

  cublasStatus_t cublaStat = cublasScopy(handle, A->m, gpu_mat_A + (A->m * col), 1,
    gpu_vect_res, 1);

  cudaError = cudaMemcpy(res, gpu_vect_res, sizeof(float) * A->m, cudaMemcpyDeviceToHost);
  //cublasStatus_t cudaStat = cublasGetVector(A->m, A->m, gpu_vect_res, 1, res, 1);

  if (cudaError != cudaSuccess) {
      printf ("get_column: %d device memory upload failed for res\n", cudaError);
  }
  cudaFree(gpu_mat_A);
  cudaFree(gpu_vect_res);

  return res;
}
void vect_print(float *u, int n) {
  for (int i = 0; i < n; i++) {
    printf("%8.5f\n", u[i]);
  }
}
void matrix_print(Mat *A) {
  for (int j = 0; j < A->n; j++) {
       for (int i = 0; i < A->m; i++) {
           printf ("%8.5f ", A->data[IDX2C(i,j,A->m)]);
       }
       printf ("\n");
   }
}
