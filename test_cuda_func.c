#include "math_cuda.h"
#define M 3
#define N 3

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    float *gpu_mat_A;
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
    return EXIT_SUCCESS;
}
