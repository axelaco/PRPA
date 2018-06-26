#include "math.h"
#include <omp.h> // OpenMP
#define N 6
#define K 3
//#include <lapacke.h>

// https://blogs.mathworks.com/cleve/2016/10/03/householder-reflections-and-the-qr-decomposition/
void H(float *u, Mat *R) {
  // u.T*R
  float *res = vect_prod_mat_trans(R, u);
  // u * (u.T * R)
  float *res2 = malloc(sizeof(float) * R->m * R->m);
  for (int i = 0; i < R->m; i++) {
    for (int j = 0; j < R->n; j++) {
      res2[i * R->m + j] = u[i] * res[j];
    }
  }
  free(res);
  // H - u * (u.T * R)
  for (int i = 0; i < R->m * R->m; i++) {
    R->data[i] -= res2[i];
  }
  free(res2);
}

void house_helper(float *x, int n) {
  float normX = vect_norm(x, n);
    if (normX == 0)
      x[0] = sqrt(2);
    vect_divide(x, normX, n);
    if (x[0] >= 0) {
      x[0] += 1;
      normX = -normX;
    } else
      x[0] = x[0] - 1;

    float absX = absolute(x[0]);

    vect_divide(x, sqrt(absX), n);
}
void house_apply(Mat *U, Mat *Q) {
  if (!Q)
    return;
  if (!U)
    return;
  for (int j = U->n - 1; j >= 0; j--) {
    float *u = get_column(U, j);
    H(u, Q);
  }
}

/*
    Param: Matrix to decompose in QR
    Return: [Q, R]
*/

void qr_householder(Mat *A, Mat *R, Mat *Q1) {
  if (!A)
    return;
  if (!R)
    return;
  if (!Q1)
    return;
  matrix_copy(A, R);
  Mat *U = matrix_zeros(A->m, A->n);  
  if (!U || !R)
    return;

  float *u = NULL;

  for (int k = 0; k < A->n - 1; k++) {
    u = get_column_start(R, k);  
    house_helper(u, A->n - k);  
    // U(j:m, j) = u
    vect_mat_copy_cond(U, u, k, 1);
    // R(j:m,j:n)
    Mat *Rreduce =  matrix_reduce_cond(R, k);
    // H(u, R(j:m, j:n))
    H(u, Rreduce);
    // R(j:m, j:n) = H(u, R(j:m, j:n))
    matrix_copy_cond(R, Rreduce, k);
    // R(j+1:m, j) = 0
    for (int j = (k + 1) * R->m + k; j < R->m * R->n; j+=R->m) {
      R->data[j] = 0;
    }
  }
  Mat *Q = matrix_eye(U->m, U->n);
  matrix_copy(Q, Q1);
  house_apply(U,  Q1);
}

#ifdef INTEL_MKL

void intel_mkl_qr(Mat *Ar, Mat *R, Mat *Q) {
  if (!Ar || !R || !Q)
    return;
  float tau[Ar->m];
  // Compute QR
  LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, Ar->m, Ar->n, Ar->data, Ar->m, tau);

  matrix_copy(Ar, Q);
  // Retrieve R tri upper
  LAPACKE_slacpy(LAPACK_ROW_MAJOR, 'U', Ar->m, Ar->n, Ar->data, Ar->m, R->data, Ar->n);  
  // Retrieve Q 
  LAPACKE_sorgqr(LAPACK_ROW_MAJOR, Ar->m, Ar->n, Ar->m, Q->data, Ar->m, tau);
}
#endif

Mat *arnoldi_iteration(Mat *A, float *v0, int MAX_ITER) {
  vect_divide(v0, vect_norm(v0, A->n), A->n);
  Mat *Hm = matrix_zeros(MAX_ITER, MAX_ITER);
  Mat *Vm = matrix_zeros(A->n, MAX_ITER);
  float *w = malloc(sizeof(float) * A->m);
  float *fm = NULL;
  float *v = NULL;
  float *VmReduce2 = NULL;
  float *h = NULL;
  float *res = NULL;
  for (int j = 0; j < MAX_ITER; j++) {
      //printf("%d Ite\n", j);
      if (j == 0) {      
        // w = A * v0
        vect_prod_mat(A, v0, w);
        // Vm(:, j) = v0
        vect_mat_copy(Vm, v0, j);
        // Hm(1,1) = v0.T * w
        Hm->data[0] = vect_dot(v0, w, A->n);

        // fm = w - v0 * v0.T * w
        fm = compute_fm(v0, w, A->n, A->m);
      } else {
        // v = fm/||fm||
        v = vect_divide_by_scalar(fm, vect_norm(fm, A->n), A->n);
        // w = A * v
        vect_prod_mat(A, v, w);
        // Vm(:, j) = v
        vect_mat_copy(Vm, v, j);

        // Hm(j, j − 1) = ||fm||
        Hm->data[j * Hm->n + (j-1)] = vect_norm(fm, A->n);
        
        Mat *VmReduce = matrix_reduce(Vm, j + 1);
        // h = Vm(:,1 : j).T ∗ w

        float *h = vect_prod_mat_trans(VmReduce, w);;
        // Vm(:, 1:j) * h
        float *tmp = malloc(sizeof(float) * VmReduce->m);
        vect_prod_mat(VmReduce, h, tmp);

        // fm = w − Vm(:,1 : j)∗ h
        vect_substract(fm, w, tmp, A->m);

        // Hm(1 : j, j) = h
        vect_mat_copy_cond(Hm, h, j, 0);

        free(tmp);
        free(v);
        free(h);
        matrix_delete(VmReduce);
      }
    }
    free(fm);
    free(w);
    return Hm;

}

void qr_alg_eigen(Mat *A) {
    Mat *mat_tmp = matrix_new(A->m, A->n);
    int k = 0;
    Mat *R = matrix_zeros(A->m, A->n);
    Mat *Q = matrix_zeros(A->m, A->n);
    while (k < 100) {
      // s = A(n,n)
      float s = A->data[A->n * A->m - 1];
      Mat *eye = matrix_eye(A->n, A->n);
      // s * I(:,:)
      matrix_scalar(eye, s);

      //A(:,:) - s * I(:,:)
      matrix_sub(A, eye, mat_tmp);

      // qr(A(:,:) - s * I(:,:))
#ifdef NAIVE
      qr_householder(mat_tmp, R, Q);
#elif INTEL_MKL
      intel_mkl_qr(mat_tmp, R, Q);
#endif
      // A = R*Q + s*I
      matrix_add(matrix_mul(R, Q), eye, A);
      
      R = matrix_zeros(A->m, A->n);
      matrix_delete(eye);
      k++;
    }
}

void iram(Mat *A, float *v0, int k, int MAX_ITER) {
  
}

void eigen_values(Mat *A) {
  
    int nb_iter = 0;
    float *v = calloc(A->n, sizeof(float));
    for (int i = 0; i < A->n - 1; i++)
      v[i] = 1;
    float vNorm =  vect_norm(v, A->n);
    for (int i = 0; i < A->n; i++) {
        v[i] /= vNorm;
    }
    printf("Compute %d krylov space for Matrice A(%d, %d):\n", K, A->m, A->n);
    float start = omp_get_wtime();
    Mat *Ar = arnoldi_iteration(A, v, K);
    float stop = omp_get_wtime();
    free(v);
    printf("Time : %lf\n", stop-start);
    int k = 0;
    matrix_print(Ar);
    qr_alg_eigen(Ar);
    /*while (k < 100) {
      Mat **Qr = qr_householder(Ar);
      puts("Q");
      matrix_print(Qr[0]);
      puts("R");
      matrix_print(Qr[1]);
      puts("R*Q");
      
      Ar = matrix_mul(Qr[1], Qr[0]);
      matrix_print(Ar);
      k++;
    }
    matrix_print(Ar);
    */
    puts("AR");
    matrix_print(Ar);
    matrix_delete(Ar);

}


float in[][3] = {
    {2,1,2},
    {4,2,3},  
    {3,2,2}
};
float in2[][6] = {
    {35,1, 6, 26,19,24},
    {3, 32, 7,21,23,25},
    {31,9,2,22,27,20},
    {8,28,33,17,10,15},
    {30,5,34,12,14,16},
    {4,36,29,13,18,11},
};  
void init_random_matrix(Mat *A) {
  float tmp;
  srand(10);
  for (int i = 0; i < A->m * A->n; i++) {
    tmp = (float) (rand() % 100);
    A->data[i] = tmp;
  }
}
int main(void) {
  int tmp;
  Mat *A = matrix_new(N, N);  
  for (int i = 0; i < N * N; i++)
    A->data[i] = in2[i / N][i % N];
  matrix_print(A);

  // init_random_matrix(A);
  //matrix_print(A);
  puts("");
  float start = omp_get_wtime();
  eigen_values(A);
  float stop = omp_get_wtime();
 // printf("Time : %lf\n", stop-start);
  matrix_delete(A);
  return 0;
}



