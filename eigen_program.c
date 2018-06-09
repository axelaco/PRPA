#include "math.h"
#include <omp.h> // OpenMP
// #include "mkl.h" // Intel MKL
#include <lapacke.h>

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

float house_helper(float *x, int n) {
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

    return normX;
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

Mat **qr_householder(Mat *A) {
  if (!A)
    return NULL;
  Mat **res = malloc(sizeof(Mat *) * 2);
  if (!res)
    return NULL;
  for (int i = 0; i < 2; i++) {
    res[i] = malloc(sizeof(Mat));
  }
  Mat *R = matrix_copy(A);
  Mat *U = matrix_zeros(A->m, A->n);  
  if (!U || !R)
    return NULL;

  float *u = NULL;

  for (int k = 0; k < A->n; k++) {
    u = get_column_start(R, k);  
    float normX = house_helper(u, A->n - k);  
    vect_mat_copy_cond(U, u, k, 1);  
    Mat *Rreduce =  matrix_reduce_cond(R, k);
    H(u, Rreduce);
    matrix_copy_cond(R, Rreduce, k);
    for (int j = (k + 1) * R->m + k; j < R->m * R->n; j+=R->m) {
      R->data[j] = 0;
    }
  }

  Mat *Q = matrix_eye(U->m, U->n);
  house_apply(U,  Q);
  res[0] = Q;
  res[1] = R;
  return res;
}

Mat *arnoldi_iteration(Mat *A, float *v0, int MAX_ITER) {
  vect_divide(v0, vect_norm(v0, A->n), A->n);
  Mat *Hm = matrix_zeros(MAX_ITER, MAX_ITER);
  Mat *Vm = matrix_zeros(A->n, MAX_ITER);
  float *w = NULL;
  float *fm = NULL;
  float *v = NULL;
  Mat *VmReduce = NULL;
  float *h = NULL;
  for (int j = 0; j < MAX_ITER; j++) {
    if (j == 0) {      
      // w = A * v0
      w = vect_prod_mat(A, v0);
      // Vm(:, j) = v0
      vect_mat_copy(Vm, v0, j);
      // Hm(1,1) = v0.T * w
      Hm->data[0] = vect_dot(v0, w, A->n);
      // fm = w - v0 * v0.T * w
      fm = compute_fm(v0, v0, w, A->n);
    } else {
      // v = fm/||fm||
      v = vect_divide_by_scalar(fm, vect_norm(fm, A->n), A->n);
      // w = A * v
      w = vect_prod_mat(A, v);
      // Vm(:, j) = v
      vect_mat_copy(Vm, v, j);
      // Hm(j, j − 1) = ||fm||
      Hm->data[j * Hm->n + (j-1)] = vect_norm(fm, A->n);
      // Vm(:, 1:j)
      VmReduce = matrix_reduce(Vm, j + 1);
      // h = Vm(:,1 : j).T ∗ w
      h = vect_prod_mat_trans(VmReduce, w);
      // fm = w − Vm(:,1 : j)∗ h
      vect_substract(fm, w, vect_prod_mat(VmReduce, h), A->m);
      // Hm(1 : j, j) = h
      vect_mat_copy_cond(Hm, h, j, 0);

    }
  }
  return Hm;
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
    Mat *Ar = arnoldi_iteration(A, v, 3);
    matrix_print(Ar);
    Mat **Qr = qr_householder(Ar);
    puts("Q");
    matrix_print(Qr[0]);
    puts("R");
    matrix_print(Qr[1]);
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
int main(void) {
  int tmp;
  Mat *A = matrix_new(3, 3);
  for (int i = 0; i < 3 * 3; i++)
    A->data[i] = in[i / 3][i % 3];
  float start = omp_get_wtime();
  eigen_values(A);
  float stop = omp_get_wtime();
  printf("Time : %lf\n", stop-start);
  matrix_delete(A);
  return 0;
}



