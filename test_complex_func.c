#include "math_complex.h"
#define N 4

int main(void) {
    Mat *A = matrix_new(N, N);
    Mat *Asub = matrix_new(N, N);
    Mat *ACop = matrix_new(N, N);
    Mat *D = matrix_eye(N,N);
    for (int i = 0; i < N*N; i++) {
        A->data[i].real = i;
        A->data[i].imag = 1;
    }
    Mat *B = matrix_eye(N,N);
    for (int i = 0; i < N*N; i++) {
        B->data[i].real = i / 2;
        B->data[i].imag = 2;
    }
    // test eye matrix 
    Mat *eye = matrix_eye(N,N);
    puts("I(4,4):");
    matrix_print(eye);
    puts("A:");
    matrix_print(A);
    puts("B:");
    matrix_print(B);
    // Test Matrix Multiplication
    Mat *C = matrix_mul(A,B);
    puts("A*B:");
    matrix_print(C);

    // Test Matrix Addition
    puts("A + B:");
    matrix_add(A,B,D);
    matrix_print(D);

    // Test Matrix Substraction
    puts("A - B:");
    matrix_sub(A,B,D);
    matrix_print(D);
    
    // Test Matrix * scalar
    matrix_scalar(D, 3);
    puts("3*(A - B)");
    matrix_print(D);

    // Test Copy Matrix
    puts("Acopy");
    matrix_copy(A, ACop);
    matrix_print(ACop);

    // Test Matrix Transpose
    puts("A.T");
    Mat *AT = matrix_transpose(A);
    matrix_print(AT);

    // Test Matrix Reduce
    puts("A(:, 1:2)");
    Mat *AReduce = matrix_reduce(A, 2);
    matrix_print(AReduce);
    
    // Test Matrix Reduce Cond
    puts("A(j:$, j:$)");
    Mat * AReduce2 = matrix_reduce_cond(A, 2);
    matrix_print(AReduce2);

    // Test Matrix Copy from started col
    puts("A(j:$, j:$)");
    matrix_copy_cond(A, Asub, 2);
    matrix_print(Asub);



    matrix_delete(eye);
    matrix_delete(Asub);
    matrix_delete(A);
    matrix_delete(B);
    matrix_delete(D);
    matrix_delete(AT);
    matrix_delete(ACop);
    matrix_delete(AReduce);
    matrix_delete(AReduce2);

}