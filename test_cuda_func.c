#include "math_cuda.h"
#define M 15
#define N 15

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    float *gpu_mat_A;
    float *u = malloc(sizeof(float) * N);
    float *v = malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
      u[i] = i + 1;
    }
    Mat *A = matrix_new(M, N);
    Mat *B = matrix_new(M, N);
    Mat *D = matrix_new(M, N);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            A->data[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            B->data[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    puts("A:");
    matrix_print(A);
    puts("B:");
    matrix_print(B);
    Mat *C = matrix_mul(handle, A, B);
    if (!C)
      return EXIT_FAILURE;
    puts("A * B:");
    matrix_print(C);
    puts("A - B:");
    matrix_sub(A, B, C);
    matrix_print(C);


    puts("5 * A:");
    matrix_scalar(A, 5);
    matrix_print(A);


    // Vector Test
    puts("u:");
    vect_print(u, N);
    vect_copy(u, v, N);
    puts("v:");
    vect_print(v, N);
    vect_scalar(v, 2, N);
    puts("2*v:");
    vect_print(v, N);
    puts("u' * v:");
    float c = vect_dot(handle, u, v, N);
    printf("%8.5f\n", c);
    puts("u - v");
    float *d = malloc(sizeof(float) * N);
    vect_substract(d, u, v, N);
    vect_print(d, N);
    float u_norm = vect_norm(handle, u, N);
    printf("u_norm: %8.5f\n", u_norm);
    puts("u + v");
    vect_add(d, u, u, N);
    vect_print(d, N);
    puts("v / 2 == u");
    float *e = vect_divide_by_scalar(v, 2, N);
    vect_print(e, N);
    matrix_print(A);
    puts("A(:, 2)");
    float *A_2 = get_column(handle, A, 2);
    vect_print(A_2, N);

    puts("A(:, 4) == u");
    vect_mat_copy(handle, A, u, 4);
    matrix_print(A);
    puts("A*u");
    vect_print(u, N);
    float *res = malloc(sizeof(float) * A->m);
    vect_prod_mat(handle, A, u, res);
    vect_print(res, N);
    Mat *AT = matrix_transpose(handle, A);
    puts("A.T:");
    matrix_print(AT);
    free(u);
    free(v);
    free(d);
    free(e);
    free(res);
    return EXIT_SUCCESS;
}
