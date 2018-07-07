#include "math_cuda_v2.h"
#define M 15
#define N 15
#define BLOCK_SIZE 256

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle = NULL;
    float *gpu_mat_A;
    float *d_u = NULL;
    float *d_v = NULL;
    cudaMalloc((void**) &d_u, sizeof(float) * N);
    cudaMalloc((void**) &d_v, sizeof(float) * N);

    float *h_u = malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
      h_u[i] = i + 1;
    }
    cudaMemcpy(d_u, h_u, sizeof(float) * N, cudaMemcpyHostToDevice);

    Mat *A = matrix_new(M, N);
    Mat *B = matrix_new(M, N);
    Mat *C = matrix_zeros(M, N);
    Mat *D = matrix_new(M, N);

    float *h_A = malloc(sizeof(float) * M * N);
    float *h_B = malloc(sizeof(float) * M * N);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            h_A[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            h_B[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }
    cudaMemcpy(A->d_data, h_A, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(B->d_data, h_B, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    free(h_A);
    free(h_B);
    free(h_u);

    cusolverDnHandle_t cusolverH = NULL;
    puts("qr(A)");
    Mat *R = matrix_eye(M, N);
    Mat *Q = matrix_zeros(M, N);

    // step 1: create cusolverDn/cublas handle
    cublasStatus_t cublas_status = cublasCreate(&handle);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cublasStatus_t cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    qr(handle, cusolverH, A, R, Q);
    puts("Q:");
    matrix_print(Q);
    puts("R:");
    matrix_print(R);

    float *d_A = NULL;
    float *d_R = NULL;
/*
    cudaMalloc((void**) &d_A, sizeof(float) * A->m * A->n);
    cudaMalloc((void**) &d_R, sizeof(float) * A->m * A->n);
    cudaError_t cudaError = cudaMemcpy(d_A, A->data, sizeof(float) * A->m * A->n, cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) {
        printf ("%d device memory allocation failed for cpu_data\n", cudaStat);
    }
    matrix_print(A);
    cudaDeviceSynchronize();
    triu(d_A, d_R, BLOCK_SIZE, A->m, A->n, A->n);
    cudaDeviceSynchronize();
    cudaMemcpy(D->data, d_R, sizeof(float) * A->m * A->n, cudaMemcpyDeviceToHost);
    matrix_print(D);
    cudaFree(d_A);
    cudaFree(d_R);
*/
    puts("A:");
    matrix_print(A);
    puts("B:");
    matrix_print(B);
  //  C = matrix_mul(handle, A, B);
//    if (!C)
    //  return EXIT_FAILURE;
  //  puts("A * B:");
  //  matrix_print(C);
    puts("A - B:");
    matrix_sub(A, B, C);
    matrix_print(C);


    puts("5 * A:");
    matrix_scalar(A, 5);
    matrix_print(A);

    // Vector Test
    puts("u:");
    vect_print(d_u, N);
    vect_copy(d_u, d_v, N);


    puts("v:");
    vect_print(d_v, N);
    vect_scalar(d_v, 2, N);

    puts("2*v:");
    vect_print(d_v, N);

    puts("u' * v:");
    float c = vect_dot(handle, d_u, d_v, N);
    printf("%8.5f\n", c);

    puts("u - v");
    float *d_d = NULL;
    cudaMalloc((void**) &d_d, sizeof(float) * N);
    vect_substract(d_d, d_u, d_v, N);
    vect_print(d_d, N);

    puts("||u||");
    float u_norm = vect_norm(handle, d_u, N);
    printf("u_norm: %8.5f\n", u_norm);

    puts("u + v");
    vect_add(d_d,d_u, d_v, N);
    vect_print(d_d, N);

    puts("v / 2 == u");
    float *e = vect_divide_by_scalar(d_v, 2, N);
    vect_print(e, N);


    matrix_print(A);
    puts("A(:, 2)");

    puts("A(:, 2)");
    float *A_2 = get_column(handle, A, 2);
    vect_print(A_2, N);

    puts("A(:, 4) == u");
    vect_mat_copy(handle, A, d_u, 4);
    matrix_print(A);

    puts("A*u");
    float *d_res = NULL;
    cudaMalloc((void**) &d_res, sizeof(float) * A->m);
    vect_prod_mat(handle, A, d_u, d_res);
    vect_print(d_res, N);
    puts("diag(A):");
    float *d_diag = matrix_diag(A);
    vect_print(d_diag, A->m);
    Mat *eigVector = matrix_zeros(A->m, A->n);
    float *d_eigs = qr_alg_eigen(A, eigVector);
    puts("eigs:");
    vect_print(d_eigs, A->m);
    cudaFree(d_diag);
    cudaFree(d_eigs);
    matrix_delete(A);
    matrix_delete(B);
    matrix_delete(D);
    matrix_delete(Q);
    matrix_delete(R);
    cudaFree(d_u);
    cudaFree(d_v);
    //cudaFree(d_d);
    //cudaFree(d_e);
    cudaFree(d_res);
    return EXIT_SUCCESS;
}
